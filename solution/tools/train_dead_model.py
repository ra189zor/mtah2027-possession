"""Prepare label-free evidence, select on development match folds, validate frozen models."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'solution/src'), str(ROOT/'solution/tools'),
               str(ROOT/'more-than-a-hack-2027-main repo/evaluation'),
               str(ROOT/'more-than-a-hack-2027-main repo/starter')]
from possession.dead_features import extract_dead_features, feature_indices
from possession.dead_model import training_rows, sample_weights, binary_metrics, assert_training_split
from possession.features import extract_features
from possession.inputs import read_external, file_digest, array_digest
from possession.state_machine import predict
from possession.validation import corrupt, nearest_team_codes, period_blocks
from benchmark_external import summarize_score
from run_oracles import paired_draw_scores
import baseline
import possession_metrics as pm

DATA_ROOT = ROOT/'data/external/sportec'
SOURCE_NAMES = ('solution/src/possession/dead_features.py','solution/src/possession/dead_model.py',
    'solution/src/possession/features.py','solution/src/possession/validation.py',
    'solution/src/possession/inputs.py','solution/src/possession/state_machine.py',
    'solution/tools/train_dead_model.py','solution/tools/run_oracles.py',
    'solution/tools/benchmark_external.py', 'solution/configs/dead_model.json',
    'solution/configs/features.json','solution/configs/validation.json','solution/configs/fallback.json',
    'solution/requirements-lock.txt',
    'more-than-a-hack-2027-main repo/evaluation/possession_metrics.py',
    'more-than-a-hack-2027-main repo/starter/baseline.py')


def settings():
    config=json.loads((ROOT/'solution/configs/dead_model.json').read_text())
    split=json.loads((ROOT/'solution/configs/validation.json').read_text())
    assert_training_split(config['development_matches'],config['validation_matches'])
    for name in ('development','validation'):
        if not set(config[name+'_matches']) <= set(split[name+'_matches']):
            raise ValueError('Dead model changed frozen match membership')
    base=json.loads((ROOT/'solution/configs/features.json').read_text())
    hashes={name:file_digest(ROOT/name) for name in SOURCE_NAMES}
    # Feature cache signature excludes this runner/model helper, which do not define features.
    feature_sources=('solution/src/possession/dead_features.py','solution/src/possession/features.py',
                     'solution/src/possession/validation.py','solution/src/possession/inputs.py',
                     'solution/src/possession/state_machine.py','solution/configs/dead_model.json',
                     'solution/configs/features.json','solution/configs/validation.json','solution/configs/fallback.json')
    signature=hashlib.sha256(json.dumps({k:hashes[k] for k in feature_sources},sort_keys=True).encode()).hexdigest()[:16]
    return config,split,base,hashes,signature


def scenarios(split):
    return [(regime,seed) for regime in split['regimes']
            for seed in (split['seeds'][:1] if regime=='clean' else split['seeds'])]


def folder_for(match,signature):
    return DATA_ROOT/match/'dead_features'/signature


def prepare(match,config,split,base,signature):
    inputs=read_external(DATA_ROOT,match,split)
    before=array_digest(inputs)
    destination=folder_for(match,signature)
    destination.mkdir(parents=True,exist_ok=True)
    fallback=json.loads((ROOT/'solution/configs/fallback.json').read_text())
    frozen=fallback['common']|fallback['profiles']['responsive']
    for regime,seed in scenarios(split):
        stem=f'{regime}_{seed}'
        manifest=destination/(stem+'.json')
        if manifest.exists():
            info=json.loads(manifest.read_text())
            if info['original_input_fingerprint']!=before:
                raise ValueError('Cached source inputs changed')
            for name,digest in info['files'].items():
                if file_digest(destination/name)!=digest: raise ValueError('Cache integrity failure')
            print(f'{match} {stem}: verified existing cache',flush=True)
            continue
        started=time.perf_counter()
        noisy=corrupt(inputs,split['regimes'][regime],seed)
        noise_hash=array_digest(noisy)
        basic=extract_features(noisy,base)
        evidence=extract_dead_features(noisy,base,config,basic)
        if np.isinf(evidence['values']).any(): raise ValueError('Infinite dead features')
        np.save(destination/(stem+'.npy'),evidence['values'],allow_pickle=False)
        original=predict(basic,noisy['period'],frozen)['labels']
        owner_basic={key:value.copy() for key,value in basic.items()}
        for name in ('ball_speed_mps','ball_goal_line_distance_m','ball_touchline_distance_m'):
            owner_basic['valid'][:,basic['names'].tolist().index(name)]=False
        owner=predict(owner_basic,noisy['period'],frozen)['labels']
        codes=nearest_team_codes(noisy)
        reference=np.full(len(codes),'X',dtype='<U1')
        for a,b in period_blocks(noisy['period']): reference[a:b]=baseline.smooth(codes[a:b],50)
        np.savez_compressed(destination/(stem+'_context.npz'),period=noisy['period'],
            ball_observed=np.isfinite(noisy['ball_xy']).all(axis=1),
            solver=original,owner=owner,baseline=reference,names=evidence['names'])
        files={name:file_digest(destination/name) for name in (stem+'.npy',stem+'_context.npz')}
        manifest.write_text(json.dumps({'match_id':match,'regime':regime,'seed':seed,
            'original_input_fingerprint':before,'noisy_input_fingerprint':noise_hash,
            'files':files,'rows':len(codes),'features':len(evidence['names']),
            'reference_arrays_read':False},indent=2)+'\n',encoding='utf-8')
        if array_digest(noisy)!=noise_hash: raise ValueError('Feature extraction modified inputs')
        print(f'{match} {stem}: {evidence["values"].shape}, label-free cache '
              f'({time.perf_counter()-started:.1f} s)',flush=True)
        del noisy,basic,owner_basic,evidence
    if array_digest(inputs)!=before: raise ValueError('Source inputs changed')


def targets(match):
    folder=DATA_ROOT/match/'prepared'
    audit=json.loads((folder/'audit.json').read_text())
    if file_digest(folder/'targets.npz')!=audit['targets_sha256']: raise ValueError('Target integrity failure')
    with np.load(folder/'targets.npz',allow_pickle=False) as bundle: return bundle['labels'].copy()


def load_case(match,regime,seed,signature):
    folder=folder_for(match,signature)
    stem=f'{regime}_{seed}'
    info=json.loads((folder/(stem+'.json')).read_text())
    for name,digest in info['files'].items():
        if file_digest(folder/name)!=digest: raise ValueError('Feature cache changed')
    matrix=np.load(folder/(stem+'.npy'),mmap_mode='r',allow_pickle=False)
    with np.load(folder/(stem+'_context.npz'),allow_pickle=False) as z:
        context={key:z[key] for key in z.files}
    return matrix,context,info


def training_data(matches,variant,config,split,signature):
    blocks,truth,weights=[],[],[]
    for match in matches:
        labels=targets(match)
        for regime,seed in scenarios(split):
            matrix,context,_=load_case(match,regime,seed,signature)
            indices=feature_indices(context['names'],variant['features'])
            rows=training_rows(labels,context['period'],config['training_stride_frames'],seed)
            block=np.asarray(matrix[rows][:,indices],dtype=np.float32)
            weight=sample_weights(labels,context['period'],rows,variant['weighting'],config['sample_weight_cap'])
            weight/=weight.sum()
            if regime!='clean': weight/=len(split['seeds'])
            blocks.append(block); truth.append((labels[rows]=='D').astype(np.uint8)); weights.append(weight)
    x,y,w=np.concatenate(blocks),np.concatenate(truth),np.concatenate(weights)
    w*=len(w)/w.sum()
    return x,y,w


def fit(matches,variant,config,split,signature):
    x,y,w=training_data(matches,variant,config,split,signature)
    model=HistGradientBoostingClassifier(**config['model'])
    started=time.perf_counter()
    model.fit(x,y,sample_weight=w)
    print(f'Fit {variant["features"]}/{variant["weighting"]}: {len(matches)} matches, '
          f'{x.shape}, {time.perf_counter()-started:.1f} s',flush=True)
    return model,{'training_matches':matches,'training_rows':len(y),'features':x.shape[1],
                  'feature_array_mib':round(x.nbytes/1024**2,2)}


def predict_probability(model,matrix,indices):
    # Keep native-frame inference memory bounded; no reference labels involved.
    out=np.empty(len(matrix),dtype=np.float32)
    for a in range(0,len(matrix),16384):
        b=min(len(matrix),a+16384)
        out[a:b]=model.predict_proba(np.asarray(matrix[a:b][:,indices]))[:,1]
    return out


def develop(config,split,hashes,signature,destination):
    report={'stage':'development','status':'running','config':config,'source_sha256':hashes,
            'cache_signature':signature,'python':platform.python_version(),'sklearn':sklearn.__version__,
            'results':[],'fits':[],'models':{},'ranking':[]}
    for variant_name,variant in config['variants'].items():
        for held_out in config['development_matches']:
            training=[m for m in config['development_matches'] if m!=held_out]
            assert_training_split(training,[held_out]+config['validation_matches'])
            model,fit_info=fit(training,variant,config,split,signature)
            report['fits'].append({'variant':variant_name,'held_out':held_out,**fit_info})
            labels=targets(held_out)
            for regime,seed in scenarios(split):
                matrix,context,_=load_case(held_out,regime,seed,signature)
                indices=feature_indices(context['names'],variant['features'])
                probability=predict_probability(model,matrix,indices)
                for threshold in config['thresholds']:
                    report['results'].append({'match_id':held_out,'regime':regime,'seed':seed,
                        'variant':variant_name,'threshold':threshold,
                        **binary_metrics(labels,probability,threshold)})
            print(f'{variant_name}: held-out {held_out} scored',flush=True)
    for variant_name in config['variants']:
        for threshold in config['thresholds']:
            groups=[]
            for match in config['development_matches']:
                for regime in split['regimes']:
                    rows=[r for r in report['results'] if r['variant']==variant_name and
                        r['threshold']==threshold and r['match_id']==match and r['regime']==regime]
                    groups.append([np.mean([r[k] for r in rows]) for k in ('f1','recall','precision')])
            f1,recall,precision=np.mean(groups,axis=0)
            report['ranking'].append({'variant':variant_name,'threshold':threshold,
                'mean_dead_f1':float(f1),'mean_recall':float(recall),'mean_precision':float(precision)})
    report['ranking'].sort(key=lambda r:(-r['mean_dead_f1'],-r['mean_recall'],r['variant'],r['threshold']))
    report['selected']=report['ranking'][0]
    for name,variant in config['variants'].items():
        chosen=next(r for r in report['ranking'] if r['variant']==name)
        model,info=fit(config['development_matches'],variant,config,split,signature)
        matrix,context,_=load_case(config['development_matches'][0],'clean',split['seeds'][0],signature)
        indices=feature_indices(context['names'],variant['features'])
        bundle={'model':model,'names':context['names'][indices], 'indices':indices,
                'threshold':chosen['threshold'],'source_sha256':hashes,'training_matches':config['development_matches']}
        model_path=destination/(name+'.joblib')
        joblib.dump(bundle,model_path)
        loaded=joblib.load(model_path)
        # Serialization must preserve predictions exactly, without retraining on validation.
        probe=np.asarray(matrix[::101][:,indices])
        if not np.array_equal(model.predict_proba(probe),loaded['model'].predict_proba(probe)):
            raise ValueError('Serialized model changed probabilities')
        report['models'][name]={'file':model_path.name,'sha256':file_digest(model_path),**chosen,**info}
    report['status']='passed'
    path=destination/'development_report.json'
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f'PASS: frozen development selection: {path}',flush=True)


def validate(config,split,hashes,signature,selection_path,destination):
    selection=json.loads(selection_path.read_text(encoding='utf-8'))
    if selection.get('status')!='passed' or selection.get('stage')!='development' or selection['source_sha256']!=hashes:
        raise ValueError('Selection is incomplete or frozen source/settings changed')
    report={'stage':'validation','status':'running','selection':str(selection_path),
            'source_sha256':hashes,'selected':selection['selected'],'results':[], 'paired_deltas':[]}
    bundles={}
    for name,info in selection['models'].items():
        path=selection_path.parent/info['file']
        if file_digest(path)!=info['sha256']: raise ValueError('Model changed')
        bundles[name]=joblib.load(path)  # trusted locally produced, checksum-verified models only
        assert_training_split(bundles[name]['training_matches'],config['validation_matches'])
    for match in config['validation_matches']:
        labels=targets(match); gt=pm.label_codes(labels)
        n_gt=len(pm.possession_spells(gt,pm.valid_runs(gt)))
        picks=np.random.RandomState(20261030).randint(0,n_gt,(1000,n_gt))
        for regime in split['regimes']:
            seed_draws=[]
            for seed in (split['seeds'][:1] if regime=='clean' else split['seeds']):
                matrix,context,info=load_case(match,regime,seed,signature)
                predictions={'solver':context['solver'],'baseline_50':context['baseline']}
                probabilities={}
                for name,bundle in bundles.items():
                    indices=bundle['indices']
                    if not np.array_equal(context['names'][indices],bundle['names']): raise ValueError('Feature schema changed')
                    probability=predict_probability(bundle['model'],matrix,indices)
                    probabilities[name]=probability
                    predictions[name]=np.where(probability>=bundle['threshold'],'D',context['owner'])
                draws={}
                for name,prediction in predictions.items():
                    metric=pm.evaluate(gt,pm.label_codes(prediction),bootstrap_n=1000)
                    probability=probabilities.get(name,(prediction=='D').astype(np.float32))
                    threshold=bundles[name]['threshold'] if name in bundles else .5
                    binary=binary_metrics(labels,probability,threshold)
                    missing=binary_metrics(labels,probability,threshold,~context['ball_observed'])
                    report['results'].append({'match_id':match,'regime':regime,'seed':seed,'model':name,
                        'noisy_input_fingerprint':info['noisy_input_fingerprint'],
                        'binary':binary,'binary_ball_missing':missing,
                        'confusion':pm.scored_confusion(pm.confusion(gt,pm.label_codes(prediction))).tolist(),
                        'abstain_fraction_on_graded':metric['n_abstain_on_graded']/metric['n_graded'],
                        'mean_duration_s':metric['mean_spell_duration_s'],**summarize_score(metric)})
                    if name==selection['selected']['variant'] or name=='baseline_50':
                        draws[name]=paired_draw_scores(gt,pm.label_codes(prediction),picks)
                seed_draws.append(draws)
                np.savez_compressed(destination/f'{match}_{regime}_{seed}_probabilities.npz',**probabilities)
                print(f'{match} {regime} seed {seed}: fixed models and baselines scored',flush=True)
            winner=selection['selected']['variant']
            delta=np.mean([r[winner]-r['baseline_50'] for r in seed_draws],axis=0)
            lo,hi=np.percentile(delta,[2.5,97.5])
            report['paired_deltas'].append({'match_id':match,'regime':regime,'model':winner,
                'conditional_95_low':float(lo),'conditional_95_high':float(hi),
                'method':'Paired official-style conditional matched-possession resampling, fixed prediction counts. Seeds share draws; not between-match uncertainty.'})
    report['status']='passed'
    path=destination/'validation_report.json'
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f'PASS: frozen model validation: {path}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['prepare','development','validation'])
    parser.add_argument('--matches',nargs='+')
    parser.add_argument('--selection',type=Path)
    args=parser.parse_args()
    config,split,base,hashes,signature=settings()
    with threadpool_limits(limits=config['threads']):
        if args.stage=='prepare':
            for match in args.matches or config['development_matches']:
                if match not in config['development_matches']+config['validation_matches']: raise ValueError('Unapproved match')
                prepare(match,config,split,base,signature)
            return
        destination=ROOT/'outputs/dead_model'/(args.stage+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        destination.mkdir(parents=True,exist_ok=False)
        if args.stage=='development': develop(config,split,hashes,signature,destination)
        else:
            if args.selection is None: parser.error('Validation requires --selection')
            validate(config,split,hashes,signature,args.selection,destination)


if __name__=='__main__': main()
