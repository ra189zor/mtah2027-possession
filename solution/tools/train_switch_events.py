"""External-only learned switch events with cross-fitted upstream training evidence."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import sys
import time
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'solution/tools'),str(ROOT/'solution/src')]
import run_nested_validation as n
from possession.switch_events import physical_columns,extract_event_features,event_targets,event_training_rows
from possession.event_decoder import decode_events
from possession.decoder import decode
from possession.robustness import nested_training_matches,absorb_short_opposition
from possession.inputs import file_digest
from run_decoder import score
from run_oracles import paired_draw_scores

SOURCE=ROOT/'outputs/nested_validation/run_20261030'
SOURCES=('solution/configs/switch_events.json','solution/src/possession/switch_events.py',
         'solution/src/possession/event_decoder.py','solution/tools/train_switch_events.py')


def setup():
    f=n.setup();config=n.read(ROOT/SOURCES[0]);hashes=f[8]|{name:file_digest(ROOT/name) for name in SOURCES}
    if n.read(SOURCE/'protocol.json')['source_sha256']!=f[8]:raise ValueError('Frozen upstream source changed')
    return f[0],config,hashes,f


def prepare(job):
    match,out=job;out=Path(out);cfg,config,hashes,f=setup();folder=out/'physical'/match;folder.mkdir(parents=True,exist_ok=True)
    manifest=folder/'index.json'
    if manifest.exists():
        index=n.read(manifest)
        if index['source_sha256']!=hashes:raise ValueError('Stale switch physical cache')
        for name,digest in index['files'].items():
            if file_digest(folder/name)!=digest:raise ValueError('Switch physical cache changed')
        return
    original=n.read(SOURCE/'prepared'/match/'index.json');names_path=n.owner.cache_folder(match,f[6])/'clean_17_context.npz'
    schema=n.read(names_path.parent/'clean_17.json')
    if file_digest(names_path)!=schema['files'][names_path.name]:raise ValueError('Owner feature schema changed')
    with np.load(names_path,allow_pickle=False) as z:names=z['names'].tolist()
    report={'source_sha256':hashes,'files':{},'names':None,'reference_arrays_read':False}
    for condition,seed in n.cases(cfg,outer=True):
        case=original['cases'][f'{condition}_{seed}']
        if file_digest(Path(case['matrix']))!=original['files'][case['matrix']]:raise ValueError('Original switch input changed')
        matrix=np.load(case['matrix'],mmap_mode='r',allow_pickle=False);physical,selected=physical_columns(matrix,names)
        path=folder/f'{condition}_{seed}.npy';np.save(path,physical,allow_pickle=False)
        report['files'][path.name]=file_digest(path);report['names']=selected
    n.write(manifest,report);print(f'{match}: ten label-free switch physical caches ready',flush=True)


def physical_evidence(out,match,condition,seed):
    folder=out/'physical'/match;index=n.read(folder/'index.json');name=f'{condition}_{seed}.npy'
    if file_digest(folder/name)!=index['files'][name]:raise ValueError('Switch physical checksum mismatch')
    return np.load(folder/name,mmap_mode='r',allow_pickle=False),index['names']


def upstream_training_case(outer,match,condition,seed,cfg):
    pair=[m for m in cfg['matches'] if m in (outer,match)]
    if len(pair)!=2 or outer==match:raise ValueError('Expected distinct outer/training fixtures')
    folder=SOURCE/'inner'/'_'.join(pair);fold=n.read(folder/'report.json')
    if set(fold['training_matches'])&set(pair) or len(fold['training_matches'])!=5:raise ValueError('Leaked stacking probabilities')
    path=folder/f'{match}_{condition}_{seed}_probabilities.npz'
    if file_digest(path)!=fold['files'][path.name]:raise ValueError('Cross-fitted probability drift')
    with np.load(path,allow_pickle=False) as z:h=z['home'];d=z['dead']
    return h,d,{'training_fixture':match,'excluded':pair,'upstream_training_matches':fold['training_matches'],
                'probability_file':str(path),'sha256':fold['files'][path.name]}


def fit(held,out,cfg,config,hashes):
    xs=[];ys=[];ws=[];lineage=[];sample_audits=[];training=nested_training_matches(cfg['matches'],[held]);feature_names=None
    for match in training:
        labels=n.dead.targets(match);index=n.read(SOURCE/'prepared'/match/'index.json')
        for condition,seed in n.cases(cfg):
            h,d,trace=upstream_training_case(held,match,condition,seed,cfg)
            physical,names=physical_evidence(out,match,condition,seed)
            with np.load(index['cases'][f'{condition}_{seed}']['context'],allow_pickle=False) as z:period=z['period']
            x,feature_names=extract_event_features(physical,names,h,d,period,config)
            rows,y,w,audit=event_training_rows(labels,period,seed,config)
            if condition!='clean':w/=len(cfg['seeds'])
            xs.append(x[rows]);ys.append(y);ws.append(w);lineage.append(trace)
            sample_audits.append({'match_id':match,'condition':condition,'seed':seed,**audit})
    x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws);w*=len(w)/w.sum();del xs,ys,ws
    started=time.perf_counter();model=HistGradientBoostingClassifier(**config['model']).fit(x,y,sample_weight=w)
    if not np.array_equal(model.classes_,[0,1,2]):raise ValueError('Switch class ordering changed')
    return {'model':model,'training_matches':training,'held_out':held,'source_sha256':hashes,
        'feature_names':feature_names,'lineage':lineage,'sampling_audits':sample_audits,
        'fit_info':{'rows':len(y),'features':x.shape[1],'fit_seconds':time.perf_counter()-started}}


def infer(model,x):
    q=np.empty((len(x),3),dtype=np.float32)
    for a in range(0,len(x),16384):q[a:a+16384]=model.predict_proba(x[a:a+16384])
    return q


def direct_switch_stats(truth,pred):
    gt=n.pm.label_codes(truth);p=n.pm.label_codes(pred);runs=n.pm.valid_runs(gt)
    direct=lambda events:[t for t in events if t[1] in (0,1) and t[2] in (0,1)]
    a=direct(n.pm.transitions(gt,runs));b=direct(n.pm.transitions(p,runs))
    # The supplied evaluator codes H/A as 0/1; D is 2. Reuse its one-to-one typed matching.
    result=n.pm.transition_scores(a,b,12)
    return result


def fit_evaluate(job):
    held,out=job;out=Path(out);cfg,config,hashes,f=setup();folder=out/'outer'/held;folder.mkdir(parents=True,exist_ok=True)
    complete=folder/'report.json'
    if complete.exists():
        if n.read(complete)['source_sha256']!=hashes:raise ValueError('Stale switch evaluation')
        return
    with threadpool_limits(limits=config['threads_per_worker']):
        model_path=folder/'switch_model.joblib'
        if model_path.exists():
            bundle=joblib.load(model_path)
            if bundle['source_sha256']!=hashes or bundle['training_matches']!=nested_training_matches(cfg['matches'],[held]):raise ValueError('Stale/leaked switch model')
        else:bundle=fit(held,out,cfg,config,hashes);joblib.dump(bundle,model_path)
        model=bundle['model'];print(f'{held}: cross-fitted switch model trained, {bundle["fit_info"]}',flush=True)
        index=n.read(SOURCE/'prepared'/held/'index.json');old=n.read(SOURCE/'evaluation'/held/'report.json')
        base=n.read(SOURCE/'selection.json')['folds'][held]['selected']['profile']
        labels=n.dead.targets(held)
        with np.load(index['cases']['clean_17']['context'],allow_pickle=False) as z:period=z['period']
        alternate,_=absorb_short_opposition(labels,period,cfg['label_sensitivity_s']);references={'provider':labels,'absorbed_under3s':alternate}
        report={'source_sha256':hashes,'held_out':held,'training_matches':bundle['training_matches'],'fit_info':bundle['fit_info'],
                'lineage':bundle['lineage'],'model_sha256':file_digest(model_path),'results':[],'paired_deltas':[],
                'files':{},'upstream_sha256':{},'event_frame_metrics':[],'short_spell_recovery':[]}
        for condition in cfg['outer_conditions']:
            draws={k:[] for k in references};points={k:[] for k in references};picks={}
            for key,truth in references.items():
                gt=n.pm.label_codes(truth);ng=len(n.pm.possession_spells(gt,n.pm.valid_runs(gt)))
                picks[key]=np.random.RandomState(cfg['bootstrap_seed']).randint(0,ng,(cfg['bootstrap_draws'],ng))
            for c,seed in n.cases(cfg,outer=True):
                if c!=condition:continue
                path=SOURCE/'outer'/held/f'{held}_{condition}_{seed}_probabilities.npz';digest=file_digest(path)
                expected=n.read(SOURCE/'outer'/held/'report.json')['files'][path.name]
                if digest!=expected:raise ValueError('Outer upstream probabilities changed')
                report['upstream_sha256'][str(path)]=digest
                with np.load(path,allow_pickle=False) as z:h=z['home'];d=z['dead']
                physical,names=physical_evidence(out,held,condition,seed)
                x,feature_names=extract_event_features(physical,names,h,d,period,config)
                if feature_names!=bundle['feature_names']:raise ValueError('Event feature schema changed')
                q=infer(model,x);probability_path=folder/f'{condition}_{seed}_events.npy';np.save(probability_path,q,allow_pickle=False)
                report['files'][probability_path.name]=file_digest(probability_path)
                target,_=event_targets(labels,period,config['event_radius_frames']);active=target>=0;argmax=q.argmax(axis=1)
                cm=np.bincount(target[active]*3+argmax[active],minlength=9).reshape(3,3)
                report['event_frame_metrics'].append({'condition':condition,'seed':seed,'confusion':cm.tolist(),
                    'true_event_frames':int((target[active]>0).sum()),'predicted_event_frames_argmax':int((argmax[active]>0).sum()),
                    'event_probability_over_point1':float((q[active,1:].max(axis=1)>.1).mean())})
                predictions={'current':decode(h,d,period,base,f[7]['epsilon'])['labels']}
                for strength in config['reward_strengths']:
                    name=f'event_reward_{strength}'
                    pred=decode_events(h,d,period,q,base,strength,config,f[7]['epsilon'])['labels'];predictions[name]=pred
                    csv_path=folder/f'{condition}_{seed}_{name}.csv'
                    with csv_path.open('w',encoding='utf-8',newline='') as handle:
                        handle.write('frame,label\n');handle.writelines(f'{i},{v}\n' for i,v in enumerate(pred))
                    error=n.pm.validate_submission(n.pm.read_label_csv(str(csv_path)),len(labels))
                    if error:raise ValueError(error)
                    report['files'][csv_path.name]=file_digest(csv_path)
                for key,truth in references.items():
                    seed_draws={};seed_points={}
                    for name,pred in predictions.items():
                        metric=score(truth,pred,period)
                        if name=='current':
                            previous=next(r for r in old['results'] if r['condition']==condition and r['seed']==seed and r['label_set']==key and r['model']=='nested_decoder')
                            if any(metric[k]!=previous[k] for k in ('spell_f1_mean','macro_f1','n_pred_spells')):raise ValueError('Switch baseline failed reproduction')
                        report['results'].append({'match_id':held,'condition':condition,'seed':seed,'label_set':key,'model':name,
                                                 **metric,'direct_switches_12':direct_switch_stats(truth,pred)})
                        seed_draws[name]=paired_draw_scores(n.pm.label_codes(truth),n.pm.label_codes(pred),picks[key]);seed_points[name]=metric['spell_f1_mean']
                        if key=='provider':
                            gt=n.pm.label_codes(truth);runs=n.pm.valid_runs(gt);spells=n.pm.possession_spells(gt,runs)
                            flags=n.pm.matched_gt_flags(n.pm.spell_overlaps(spells,n.pm.possession_spells(n.pm.label_codes(pred),runs)),len(spells),.5)
                            for lo,hi in [(0,1),(1,3),(3,10),(10,1e9)]:
                                ids=[i for i,(a,b,k) in enumerate(spells) if lo<=(b-a)/25<hi]
                                report['short_spell_recovery'].append({'condition':condition,'seed':seed,'model':name,
                                    'duration_low_s':lo,'n_gt':len(ids),'matched_iou50':int(flags[ids].sum())})
                    draws[key].append(seed_draws);points[key].append(seed_points)
            for key in references:
                for strength in config['reward_strengths']:
                    name=f'event_reward_{strength}';delta=np.mean([r[name]-r['current'] for r in draws[key]],axis=0)
                    lo,hi=np.percentile(delta,[2.5,97.5])
                    report['paired_deltas'].append({'condition':condition,'label_set':key,'model':name,
                        'delta':float(np.mean([r[name]-r['current'] for r in points[key]])),
                        'conditional_low':float(lo),'conditional_high':float(hi)})
            print(f'{held} {condition}: fixed learned-switch rewards evaluated',flush=True)
        n.write(complete,report)


def aggregate(out,cfg,config,hashes):
    reports=[n.read(out/'outer'/m/'report.json') for m in cfg['matches']];rows=[r for p in reports for r in p['results']]
    summary=[];paired=[];models=['current']+[f'event_reward_{s}' for s in config['reward_strengths']]
    for key in ('provider','absorbed_under3s'):
        for condition in cfg['outer_conditions']:
            for model in models:
                means=[];errors=[]
                for match in cfg['matches']:
                    group=[r for r in rows if r['match_id']==match and r['label_set']==key and r['condition']==condition and r['model']==model]
                    means.append({k:float(np.mean([r[k] for r in group])) for k in ('spell_f1_mean','macro_f1','n_pred_spells','n_gt_spells','mean_duration_s','dead_precision','dead_recall')})
                    errors.append(float(np.mean([r['n_pred_spells']/r['n_gt_spells']-1 for r in group])))
                summary.append({'condition':condition,'label_set':key,'model':model,'count_gate_passes':sum(abs(e)<=.2 for e in errors),
                    **{k:float(np.mean([r[k] for r in means])) for k in means[0]}})
            for model in models[1:]:
                delta=np.array([next(r['delta'] for r in p['paired_deltas'] if r['condition']==condition and r['label_set']==key and r['model']==model) for p in reports])
                picks=np.random.RandomState(cfg['bootstrap_seed']).randint(0,7,(cfg['bootstrap_draws'],7));lo,hi=np.percentile(delta[picks].mean(axis=1),[2.5,97.5])
                paired.append({'condition':condition,'label_set':key,'model':model,'delta':float(delta.mean()),
                    'descriptive_match_low':float(lo),'descriptive_match_high':float(hi),'improved_matches':int((delta>0).sum())})
    if len(rows)!=420:raise ValueError('Incomplete learned-switch results')
    for p in reports:
        folder=out/'outer'/p['held_out']
        if file_digest(folder/'switch_model.joblib')!=p['model_sha256']:raise ValueError('Event model drift')
        for name,digest in p['files'].items():
            if file_digest(folder/name)!=digest:raise ValueError('Switch output drift')
        for path,digest in p['upstream_sha256'].items():
            if file_digest(Path(path))!=digest:raise ValueError('Upstream probability drift')
    n.write(out/'validation_report.json',{'status':'passed','source_sha256':hashes,'summary':summary,'paired_match_deltas':paired,
        'results':rows,'protocol':config['protocol'],'uncertainty':cfg['uncertainty']})
    print(f'PASS: learned switch-event experiment: {out / "validation_report.json"}',flush=True)


def parallel(worker,jobs,workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(worker,job) for job in jobs]):future.result()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=True);cfg,config,hashes,_=setup();protocol=out/'protocol.json'
    if protocol.exists():
        if n.read(protocol)['source_sha256']!=hashes:raise ValueError('Switch protocol changed; use a new output directory')
    else:n.write(protocol,{'source_sha256':hashes,'config':config,'foundation':str(SOURCE)})
    parallel(prepare,[(m,str(out)) for m in cfg['matches']],config['workers'])
    parallel(fit_evaluate,[(m,str(out)) for m in cfg['matches']],config['workers'])
    aggregate(out,cfg,config,hashes)


if __name__=='__main__':main()
