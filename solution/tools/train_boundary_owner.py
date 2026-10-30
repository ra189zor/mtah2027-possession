"""Fixed boundary-sampling owner ablation on seven external outer holdouts."""
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
from possession.boundary_sampling import owner_training_rows
from possession.decoder import decode
from possession.inputs import file_digest
from possession.robustness import nested_training_matches,absorb_short_opposition
from run_decoder import score
from run_oracles import paired_draw_scores

SOURCE=ROOT/'outputs/nested_validation/run_20261030'
SOURCES=('solution/configs/boundary_sampling.json','solution/src/possession/boundary_sampling.py','solution/tools/train_boundary_owner.py')


def setup():
    foundation=n.setup();config=n.read(ROOT/SOURCES[0]);hashes=foundation[8]|{name:file_digest(ROOT/name) for name in SOURCES}
    if n.read(SOURCE/'protocol.json')['source_sha256']!=foundation[8]:raise ValueError('Frozen foundation changed')
    return foundation[0],config,hashes,foundation


def prepare(job):
    match,out=job;out=Path(out);cfg,config,hashes,_=setup();folder=out/'training'/match;folder.mkdir(parents=True,exist_ok=True)
    manifest=folder/'index.json'
    if manifest.exists():
        info=n.read(manifest)
        if info['source_sha256']!=hashes:raise ValueError('Stale boundary samples')
        for name,digest in info['files'].items():
            if file_digest(folder/name)!=digest:raise ValueError('Boundary sample drift')
        return
    labels=n.dead.targets(match);index=n.read(SOURCE/'prepared'/match/'index.json')
    xs=[];ys=[];ws=[];audits=[]
    for condition,seed in n.cases(cfg):
        case=index['cases'][f'{condition}_{seed}']
        if file_digest(Path(case['matrix']))!=index['files'][case['matrix']]:raise ValueError('Feature drift')
        matrix=np.load(case['matrix'],mmap_mode='r',allow_pickle=False)
        with np.load(case['context'],allow_pickle=False) as z:period=z['period']
        rows,weights,info=owner_training_rows(labels,period,seed,config)
        if condition!='clean':weights/=len(cfg['seeds'])
        xs.append(np.asarray(matrix[rows],dtype=np.float32));ys.append(labels[rows]=='H');ws.append(weights)
        audits.append({'condition':condition,'seed':seed,**info})
    np.save(folder/'x.npy',np.concatenate(xs),allow_pickle=False)
    np.savez_compressed(folder/'targets.npz',y=np.concatenate(ys),weight=np.concatenate(ws))
    n.write(manifest,{'source_sha256':hashes,'sampling_audits':audits,
                     'files':{name:file_digest(folder/name) for name in ('x.npy','targets.npz')}})
    print(f'{match}: boundary-aware training samples prepared',flush=True)


def fit_evaluate(job):
    held,out=job;out=Path(out);cfg,config,hashes,foundation=setup()
    folder=out/'outer'/held;folder.mkdir(parents=True,exist_ok=True);complete=folder/'report.json'
    if complete.exists():
        if n.read(complete)['source_sha256']!=hashes:raise ValueError('Stale boundary evaluation')
        return
    training=nested_training_matches(cfg['matches'],[held]);model_path=folder/'owner.joblib'
    with threadpool_limits(limits=config['threads_per_worker']):
        if model_path.exists():
            bundle=joblib.load(model_path)
            if bundle['source_sha256']!=hashes or bundle['training_matches']!=training:raise ValueError('Stale/leaked boundary model')
            model=bundle['model'];fit_info=bundle['fit_info']
        else:
            xs=[];ys=[];ws=[]
            for match in training:
                native=out/'training'/match
                xs.append(np.load(native/'x.npy',allow_pickle=False))
                with np.load(native/'targets.npz',allow_pickle=False) as z:ys.append(z['y']);ws.append(z['weight'])
            x=np.concatenate(xs);y=np.concatenate(ys);weight=np.concatenate(ws);weight*=len(weight)/weight.sum()
            del xs,ys,ws
            started=time.perf_counter()
            model=HistGradientBoostingClassifier(**foundation[5]['model']).fit(x,y,sample_weight=weight)
            fit_info={'rows':len(y),'fit_seconds':time.perf_counter()-started}
            bundle={'model':model,'training_matches':training,'source_sha256':hashes,'fit_info':fit_info}
            joblib.dump(bundle,model_path)
            probe=x[::101]
            if not np.array_equal(model.predict_proba(probe),joblib.load(model_path)['model'].predict_proba(probe)):
                raise ValueError('Boundary owner serialization changed predictions')
            del x,y,weight
        print(f'{held}: owner trained on six other fixtures, {fit_info}',flush=True)
        index=n.read(SOURCE/'prepared'/held/'index.json');old=n.read(SOURCE/'evaluation'/held/'report.json')
        base=n.read(SOURCE/'selection.json')['folds'][held]['selected']['profile']
        labels=n.dead.targets(held)
        with np.load(index['cases']['clean_17']['context'],allow_pickle=False) as z:period=z['period']
        alternate,_=absorb_short_opposition(labels,period,cfg['label_sensitivity_s'])
        references={'provider':labels,'absorbed_under3s':alternate}
        result={'source_sha256':hashes,'held_out':held,'training_matches':training,'fit_info':fit_info,
                'model_sha256':file_digest(model_path),'results':[],'paired_deltas':[],'files':{},'probability_sha256':{}}
        for condition in cfg['outer_conditions']:
            picks={};draws={key:[] for key in references};points={key:[] for key in references}
            for key,truth in references.items():
                gt=n.pm.label_codes(truth);ng=len(n.pm.possession_spells(gt,n.pm.valid_runs(gt)))
                picks[key]=np.random.RandomState(cfg['bootstrap_seed']).randint(0,ng,(cfg['bootstrap_draws'],ng))
            for c,seed in n.cases(cfg,outer=True):
                if c!=condition:continue
                case=index['cases'][f'{condition}_{seed}'];matrix=np.load(case['matrix'],mmap_mode='r',allow_pickle=False)
                new_home=n.dead.predict_probability(model,matrix,np.arange(298))
                path=SOURCE/'outer'/held/f'{held}_{condition}_{seed}_probabilities.npz'
                result['probability_sha256'][str(path)]=file_digest(path)
                with np.load(path,allow_pickle=False) as z:home=z['home'];dead=z['dead']
                probability_path=folder/f'{condition}_{seed}_new_home.npy';np.save(probability_path,new_home,allow_pickle=False)
                result['files'][probability_path.name]=file_digest(probability_path)
                predictions={}
                for name,strength in [('current',0),('blend_half',.5),('boundary_head',1.)]:
                    h=home if strength==0 else (1-strength)*home+strength*new_home
                    pred=decode(h,dead,period,base,foundation[7]['epsilon'])['labels'];predictions[name]=pred
                    if (pred[period>0]=='X').any():raise ValueError('Boundary decoder left unknowns')
                    if strength:
                        csv_path=folder/f'{condition}_{seed}_{name}.csv'
                        with csv_path.open('w',encoding='utf-8',newline='') as handle:
                            handle.write('frame,label\n');handle.writelines(f'{i},{v}\n' for i,v in enumerate(pred))
                        error=n.pm.validate_submission(n.pm.read_label_csv(str(csv_path)),len(labels))
                        if error:raise ValueError(error)
                        result['files'][csv_path.name]=file_digest(csv_path)
                for key,truth in references.items():
                    seed_draws={};seed_points={}
                    for name,pred in predictions.items():
                        row=score(truth,pred,period)
                        if name=='current':
                            prior=next(r for r in old['results'] if r['condition']==condition and r['seed']==seed and r['label_set']==key and r['model']=='nested_decoder')
                            if any(row[k]!=prior[k] for k in ('spell_f1_mean','macro_f1','n_pred_spells')):raise ValueError('Original boundary comparator drift')
                        result['results'].append({'match_id':held,'condition':condition,'seed':seed,'label_set':key,'model':name,**row})
                        seed_draws[name]=paired_draw_scores(n.pm.label_codes(truth),n.pm.label_codes(pred),picks[key]);seed_points[name]=row['spell_f1_mean']
                    draws[key].append(seed_draws);points[key].append(seed_points)
                if file_digest(path)!=result['probability_sha256'][str(path)]:raise ValueError('Original probabilities changed')
            for key in references:
                for name in ('blend_half','boundary_head'):
                    delta=np.mean([r[name]-r['current'] for r in draws[key]],axis=0);lo,hi=np.percentile(delta,[2.5,97.5])
                    result['paired_deltas'].append({'condition':condition,'label_set':key,'model':name,
                        'delta':float(np.mean([r[name]-r['current'] for r in points[key]])),
                        'conditional_low':float(lo),'conditional_high':float(hi)})
            print(f'{held} {condition}: fixed boundary-owner variants evaluated',flush=True)
        n.write(complete,result)


def aggregate(out,cfg,hashes):
    reports=[n.read(out/'outer'/match/'report.json') for match in cfg['matches']]
    rows=[row for report in reports for row in report['results']];summary=[];paired=[]
    for key in ('provider','absorbed_under3s'):
        for condition in cfg['outer_conditions']:
            for model in ('current','blend_half','boundary_head'):
                means=[];errors=[]
                for match in cfg['matches']:
                    group=[r for r in rows if r['match_id']==match and r['label_set']==key and r['condition']==condition and r['model']==model]
                    means.append({k:float(np.mean([r[k] for r in group])) for k in ('spell_f1_mean','macro_f1','n_pred_spells','n_gt_spells','mean_duration_s','dead_precision','dead_recall')})
                    errors.append(float(np.mean([r['n_pred_spells']/r['n_gt_spells']-1 for r in group])))
                summary.append({'label_set':key,'condition':condition,'model':model,
                    'count_gate_passes':sum(abs(e)<=.2 for e in errors),'mean_absolute_count_error':float(np.mean(np.abs(errors))),
                    **{k:float(np.mean([r[k] for r in means])) for k in means[0]}})
            for model in ('blend_half','boundary_head'):
                values=np.array([next(r['delta'] for r in report['paired_deltas'] if r['condition']==condition and r['label_set']==key and r['model']==model) for report in reports])
                picks=np.random.RandomState(cfg['bootstrap_seed']).randint(0,7,(cfg['bootstrap_draws'],7));lo,hi=np.percentile(values[picks].mean(axis=1),[2.5,97.5])
                paired.append({'condition':condition,'label_set':key,'model':model,'delta':float(values.mean()),
                    'descriptive_match_low':float(lo),'descriptive_match_high':float(hi),'improved_matches':int((values>0).sum())})
    if len(rows)!=420:raise ValueError('Incomplete boundary ablation')
    for report in reports:
        folder=out/'outer'/report['held_out']
        if file_digest(folder/'owner.joblib')!=report['model_sha256']:raise ValueError('New owner model changed')
        for name,digest in report['files'].items():
            if file_digest(folder/name)!=digest:raise ValueError('Boundary output drift')
    n.write(out/'validation_report.json',{'status':'passed','source_sha256':hashes,'summary':summary,'results':rows,'paired_match_deltas':paired,
        'protocol':'Fixed boundary-aware sampling and unchanged owner architecture; seven six-match outer fits. No new inner selection of blends/costs. Adaptive research on opened fixtures, not an independent final test.'})
    print(f'PASS: boundary-sampling ablation: {out / "validation_report.json"}',flush=True)


def parallel(worker,jobs,workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(worker,job) for job in jobs]):future.result()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=True)
    cfg,config,hashes,_=setup();protocol=out/'protocol.json'
    if protocol.exists():
        if n.read(protocol)['source_sha256']!=hashes:raise ValueError('Boundary protocol source changed')
    else:n.write(protocol,{'source_sha256':hashes,'config':config,'foundation':str(SOURCE)})
    parallel(prepare,[(m,str(out)) for m in cfg['matches']],config['workers'])
    parallel(fit_evaluate,[(m,str(out)) for m in cfg['matches']],config['workers'])
    aggregate(out,cfg,hashes)


if __name__=='__main__':main()
