"""Nested follow-up on frozen external heads: recover short observed possessions."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'solution/tools'),str(ROOT/'solution/src')]
import run_nested_validation as n
from possession.short_control import control_log_odds,combine_ownership,profiles
from possession.decoder import decode
from possession.inputs import file_digest
from possession.robustness import excluded_pairs,absorb_short_opposition
from run_decoder import score
from run_oracles import paired_draw_scores

SOURCE=ROOT/'outputs/nested_validation/run_20261030'
SOURCES=('solution/configs/short_possession.json','solution/src/possession/short_control.py',
         'solution/tools/improve_short_possessions.py')


def setup():
    foundation=n.setup();cfg=foundation[0];config=n.read(ROOT/SOURCES[0])
    old=n.read(SOURCE/'protocol.json')
    if old['source_sha256']!=foundation[8]:raise ValueError('Frozen nested foundation changed')
    hashes=foundation[8]|{name:file_digest(ROOT/name) for name in SOURCES}
    return cfg,config,hashes,foundation


def control_path(out,match,condition,seed):return out/'control'/f'{match}_{condition}_{seed}.npy'


def prepare_worker(job):
    match,out=job;out=Path(out);cfg,config,hashes,foundation=setup()
    index=n.read(SOURCE/'prepared'/match/'index.json');report={'source_sha256':hashes,'files':{},'coverage':[]}
    manifest=out/'control'/f'{match}.json'
    if manifest.exists():
        report=n.read(manifest)
        if report['source_sha256']!=hashes:raise ValueError('Stale control cache')
        for path,digest in report['files'].items():
            if file_digest(Path(path))!=digest:raise ValueError('Control cache changed')
        return
    names_path=n.owner.cache_folder(match,foundation[6])/'clean_17_context.npz'
    names_manifest=n.read(names_path.parent/'clean_17.json')
    if file_digest(names_path)!=names_manifest['files'][names_path.name]:raise ValueError('Control feature schema changed')
    with np.load(names_path,allow_pickle=False) as z:names=z['names'].tolist()
    for condition,seed in n.cases(cfg,outer=True):
        case=index['cases'][f'{condition}_{seed}']
        for path in (case['matrix'],case['context']):
            if file_digest(Path(path))!=index['files'][path]:raise ValueError('Frozen features changed')
        x=np.load(case['matrix'],mmap_mode='r',allow_pickle=False)
        with np.load(case['context'],allow_pickle=False) as z:period=z['period']
        odds=control_log_odds(x,names,period,config)
        path=control_path(out,match,condition,seed);np.save(path,odds,allow_pickle=False)
        report['files'][str(path)]=file_digest(path)
        report['coverage'].append({'condition':condition,'seed':seed,'fraction':float((odds[period>0]!=0).mean())})
    n.write(manifest,report);print(f'{match}: ten tracking-only short-control cases prepared',flush=True)


def predict(h,d,period,control,candidate,base,epsilon):
    probability=combine_ownership(h,control,candidate['strength'],period)
    profile=base|{'switch':candidate['switch']}
    return decode(probability,d,period,profile,epsilon)['labels']


def inner_worker(job):
    pair,out=job;out=Path(out);cfg,config,hashes,foundation=setup();epsilon=foundation[7]['epsilon']
    destination=out/'inner'/('_'.join(pair)+'.json')
    if destination.exists():
        result=n.read(destination)
        if result['source_sha256']!=hashes:raise ValueError('Stale short inner scores')
        return
    old=n.read(SOURCE/'inner'/'_'.join(pair)/'report.json')
    selection=n.read(SOURCE/'selection.json');results=[]
    if set(old['training_matches'])&set(pair):raise ValueError('Leaked inner model')
    for held in pair:
        outer=next(m for m in pair if m!=held)
        base=selection['folds'][outer]['selected']['profile']
        labels=n.dead.targets(held);gt=n.pm.label_codes(labels)
        index=n.read(SOURCE/'prepared'/held/'index.json')
        for condition,seed in n.cases(cfg):
            path=SOURCE/'inner'/'_'.join(pair)/f'{held}_{condition}_{seed}_probabilities.npz'
            if file_digest(path)!=old['files'][path.name]:raise ValueError('Inner probabilities changed')
            with np.load(path,allow_pickle=False) as z:h=z['home'];d=z['dead']
            with np.load(index['cases'][f'{condition}_{seed}']['context'],allow_pickle=False) as z:period=z['period']
            control=np.load(control_path(out,held,condition,seed),allow_pickle=False)
            for candidate in profiles(config):
                prediction=predict(h,d,period,control,candidate,base,epsilon)
                metric=n.pm.evaluate(gt,n.pm.label_codes(prediction),bootstrap_n=0)
                row={'outer':outer,'held_out':held,'condition':condition,'seed':seed,'candidate':candidate['id'],
                    **{key:metric[key] for key in ('spell_f1_mean','macro_f1','n_pred_spells','n_gt_spells')}}
                results.append(row)
                if candidate['strength']==0 and candidate['switch']==8:
                    previous=next(r for r in old['scores'] if r['held_out']==held and r['condition']==condition and r['seed']==seed and r['candidate']==base['id'])
                    if any(row[key]!=previous[key] for key in ('spell_f1_mean','macro_f1')):
                        raise ValueError('Original inner decoder does not reproduce exactly')
    n.write(destination,{'source_sha256':hashes,'training_matches':old['training_matches'],'excluded':pair,'results':results})
    print(f'{pair}: {len(results)} short-possession inner candidates scored',flush=True)


def selection(out,cfg,config,hashes):
    all_rows=[r for path in (out/'inner').glob('*.json') for r in n.read(path)['results']]
    selected={}
    for outer in cfg['matches']:
        rows=[r for r in all_rows if r['outer']==outer];ranking=[]
        for candidate in profiles(config):
            groups=[];count_errors=[]
            for held in cfg['matches']:
                if held==outer:continue
                for condition in cfg['inner_conditions']:
                    group=[r for r in rows if r['held_out']==held and r['condition']==condition and r['candidate']==candidate['id']]
                    if len(group)!=(1 if condition=='clean' else len(cfg['seeds'])):raise ValueError('Incomplete inner selection')
                    groups.append([np.mean([r[k] for r in group]) for k in ('spell_f1_mean','macro_f1')])
                    if condition=='moderate':count_errors.append(float(np.mean([r['n_pred_spells']/r['n_gt_spells']-1 for r in group])))
            spell,macro=np.mean(groups,axis=0)
            ranking.append({'profile':candidate,'spell_f1_mean':float(spell),'macro_f1':float(macro),
                'moderate_count_passes':sum(abs(e)<=config['count_tolerance'] for e in count_errors),
                'moderate_mean_absolute_count_error':float(np.mean(np.abs(count_errors)))})
        ranking.sort(key=lambda r:(-r['spell_f1_mean'],-r['macro_f1'],r['profile']['id']))
        reference=next(r for r in ranking if r['profile']['strength']==0 and r['profile']['switch']==8)
        eligible=[r for r in ranking if r['spell_f1_mean']>=reference['spell_f1_mean']-config['maximum_inner_spell_loss_for_balanced']]
        balanced=sorted(eligible,key=lambda r:(-r['moderate_count_passes'],r['moderate_mean_absolute_count_error'],-r['spell_f1_mean'],r['profile']['id']))[0]
        selected[outer]={'score_first':ranking[0],'count_balanced':balanced,'reference':reference,'ranking':ranking}
        print(f'{outer}: frozen score-first {ranking[0]["profile"]["id"]}; balanced {balanced["profile"]["id"]}',flush=True)
    n.write(out/'selection.json',{'source_sha256':hashes,'folds':selected})


def evaluate_worker(job):
    held,out=job;out=Path(out);cfg,config,hashes,foundation=setup();epsilon=foundation[7]['epsilon']
    destination=out/'evaluation'/held;destination.mkdir(parents=True,exist_ok=True)
    complete=destination/'report.json'
    if complete.exists():
        if n.read(complete)['source_sha256']!=hashes:raise ValueError('Stale evaluation')
        return
    chosen=n.read(out/'selection.json')['folds'][held]
    old_selection=n.read(SOURCE/'selection.json')['folds'][held]
    base=old_selection['selected']['profile']
    index=n.read(SOURCE/'prepared'/held/'index.json')
    with np.load(index['cases']['clean_17']['context'],allow_pickle=False) as z:period=z['period']
    labels=n.dead.targets(held);alternate,_=absorb_short_opposition(labels,period,cfg['label_sensitivity_s'])
    references={'provider':labels,'absorbed_under3s':alternate}
    old_report=n.read(SOURCE/'evaluation'/held/'report.json')
    result={'source_sha256':hashes,'match_id':held,'selected':chosen,'results':[],'paired_deltas':[],
            'probability_sha256':{},'csv_sha256':{},'short_spell_recovery':[]}
    for condition in cfg['outer_conditions']:
        draws={name:[] for name in references};points={name:[] for name in references};picks={}
        for label_set,y in references.items():
            gt=n.pm.label_codes(y);ng=len(n.pm.possession_spells(gt,n.pm.valid_runs(gt)))
            picks[label_set]=np.random.RandomState(cfg['bootstrap_seed']).randint(0,ng,(cfg['bootstrap_draws'],ng))
        for c,seed in n.cases(cfg,outer=True):
            if condition!=c:continue
            path=SOURCE/'outer'/held/f'{held}_{condition}_{seed}_probabilities.npz'
            digest=file_digest(path);result['probability_sha256'][str(path)]=digest
            with np.load(path,allow_pickle=False) as z:h=z['home'];d=z['dead']
            control=np.load(control_path(out,held,condition,seed),allow_pickle=False)
            decoded={};predictions={}
            for name,candidate in [('current',{'strength':0,'switch':8}),
                                   ('score_first',chosen['score_first']['profile']),('count_balanced',chosen['count_balanced']['profile'])]:
                key=(candidate['strength'],candidate['switch'])
                if key not in decoded:decoded[key]=predict(h,d,period,control,candidate,base,epsilon)
                predictions[name]=decoded[key]
            for name in ('score_first','count_balanced'):
                pred=predictions[name]
                if (pred[period>0]=='X').any():raise ValueError('Short decoder left unknowns')
                path_csv=destination/f'{held}_{condition}_{seed}_{name}.csv'
                with path_csv.open('w',encoding='utf-8',newline='') as handle:
                    handle.write('frame,label\n');handle.writelines(f'{i},{v}\n' for i,v in enumerate(pred))
                error=n.pm.validate_submission(n.pm.read_label_csv(str(path_csv)),len(labels))
                if error:raise ValueError(error)
                result['csv_sha256'][path_csv.name]=file_digest(path_csv)
            for label_set,y in references.items():
                seed_draws={};seed_points={}
                for name,pred in predictions.items():
                    row=score(y,pred,period)
                    if name=='current':
                        prior=next(r for r in old_report['results'] if r['condition']==condition and r['seed']==seed and r['label_set']==label_set and r['model']=='nested_decoder')
                        if any(row[k]!=prior[k] for k in ('spell_f1_mean','macro_f1','n_pred_spells')):raise ValueError('Outer baseline did not reproduce')
                    result['results'].append({'match_id':held,'condition':condition,'seed':seed,'label_set':label_set,'model':name,**row})
                    seed_draws[name]=paired_draw_scores(n.pm.label_codes(y),n.pm.label_codes(pred),picks[label_set])
                    seed_points[name]=row['spell_f1_mean']
                    if label_set=='provider':
                        gt=n.pm.label_codes(y);runs=n.pm.valid_runs(gt);spells=n.pm.possession_spells(gt,runs)
                        flags=n.pm.matched_gt_flags(n.pm.spell_overlaps(spells,n.pm.possession_spells(n.pm.label_codes(pred),runs)),len(spells),.5)
                        for lo,hi in [(0,1),(1,3),(3,10),(10,1e9)]:
                            ids=[i for i,(a,b,k) in enumerate(spells) if lo<=(b-a)/25<hi]
                            result['short_spell_recovery'].append({'condition':condition,'seed':seed,'model':name,
                                'duration_low_s':lo,'n_gt':len(ids),'matched_iou50':int(flags[ids].sum())})
                draws[label_set].append(seed_draws);points[label_set].append(seed_points)
            if file_digest(path)!=digest:raise ValueError('Frozen head probabilities changed')
        for label_set in references:
            for name in ('score_first','count_balanced'):
                delta=np.mean([r[name]-r['current'] for r in draws[label_set]],axis=0)
                low,high=np.percentile(delta,[2.5,97.5])
                result['paired_deltas'].append({'condition':condition,'label_set':label_set,'model':name,
                    'delta':float(np.mean([r[name]-r['current'] for r in points[label_set]])),
                    'conditional_low':float(low),'conditional_high':float(high)})
        print(f'{held} {condition}: frozen short-possession variants evaluated',flush=True)
    n.write(complete,result)


def aggregate(out,cfg,hashes):
    reports=[n.read(out/'evaluation'/m/'report.json') for m in cfg['matches']]
    rows=[r for report in reports for r in report['results']];summary=[];paired=[]
    for label_set in ('provider','absorbed_under3s'):
        for condition in cfg['outer_conditions']:
            for model in ('current','score_first','count_balanced'):
                means=[];errors=[]
                for match in cfg['matches']:
                    group=[r for r in rows if r['match_id']==match and r['label_set']==label_set and r['condition']==condition and r['model']==model]
                    means.append({k:float(np.mean([r[k] for r in group])) for k in ('spell_f1_mean','macro_f1','n_pred_spells','n_gt_spells','mean_duration_s','dead_precision','dead_recall')})
                    errors.append(float(np.mean([r['n_pred_spells']/r['n_gt_spells']-1 for r in group])))
                summary.append({'label_set':label_set,'condition':condition,'model':model,
                    'count_gate_passes':sum(abs(e)<=.2 for e in errors),'mean_absolute_count_error':float(np.mean(np.abs(errors))),
                    **{k:float(np.mean([x[k] for x in means])) for k in means[0]}})
            for model in ('score_first','count_balanced'):
                delta=np.array([next(r['delta'] for r in report['paired_deltas'] if r['condition']==condition and r['label_set']==label_set and r['model']==model) for report in reports])
                picks=np.random.RandomState(cfg['bootstrap_seed']).randint(0,7,(cfg['bootstrap_draws'],7))
                low,high=np.percentile(delta[picks].mean(axis=1),[2.5,97.5])
                paired.append({'label_set':label_set,'condition':condition,'model':model,'delta':float(delta.mean()),
                    'descriptive_match_low':float(low),'descriptive_match_high':float(high),'improved_matches':int((delta>0).sum())})
    for report in reports:
        for path,digest in report['probability_sha256'].items():
            if file_digest(Path(path))!=digest:raise ValueError('Probability drift')
        for name,digest in report['csv_sha256'].items():
            if file_digest(out/'evaluation'/report['match_id']/name)!=digest:raise ValueError('CSV drift')
    if len(rows)!=420:raise ValueError('Incomplete follow-up scores')
    n.write(out/'validation_report.json',{'status':'passed','source_sha256':hashes,'summary':summary,
        'paired_match_deltas':paired,'results':rows,'selection':n.read(out/'selection.json'),
        'limitations':'Adaptive research on previously opened outer fixtures. Frozen heads and nested postprocessor selection, not a new independent test. Conditional within-match and descriptive match bootstrap only.'})
    print(f'PASS: short-possession follow-up: {out / "validation_report.json"}',flush=True)


def parallel(worker,jobs,workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(worker,job) for job in jobs]):future.result()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    for folder in ('control','inner','evaluation'):(out/folder).mkdir(parents=True,exist_ok=True)
    cfg,config,hashes,_=setup();protocol=out/'protocol.json'
    if protocol.exists():
        if n.read(protocol)['source_sha256']!=hashes:raise ValueError('Frozen follow-up sources changed; use a new directory')
    else:n.write(protocol,{'source_sha256':hashes,'config':config,'foundation':str(SOURCE),'diagnosis':str(ROOT/'outputs/short_possession/diagnosis_20261030.json')})
    parallel(prepare_worker,[(m,str(out)) for m in cfg['matches']],config['workers'])
    parallel(inner_worker,[(pair,str(out)) for pair in excluded_pairs(cfg['matches'])],config['workers'])
    selection(out,cfg,config,hashes)
    parallel(evaluate_worker,[(m,str(out)) for m in cfg['matches']],config['workers'])
    aggregate(out,cfg,hashes)


if __name__=='__main__':main()
