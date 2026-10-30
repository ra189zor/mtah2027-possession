"""External-only, match-fold ownership recovery with the selected dead detector frozen."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'solution/src'),str(ROOT/'solution/tools')]
import train_dead_model as dead
from possession.features import extract_features
from possession.gap_ownership import extract_owner_features,compose_ownership
from possession.inputs import read_external,array_digest,file_digest
from possession.validation import corrupt
from possession.dead_model import training_rows,sample_weights,assert_training_split
from benchmark_external import summarize_score
from run_oracles import paired_draw_scores
import possession_metrics as pm

OWN_SOURCES=('solution/configs/gap_ownership.json','solution/src/possession/gap_ownership.py',
             'solution/tools/train_gap_ownership.py')


def setup(path):
    dc,split,base,dh,ds=dead.settings()
    foundation=json.loads(path.read_text(encoding='utf-8'))
    if foundation.get('status')!='passed' or foundation.get('stage')!='development' or foundation['source_sha256']!=dh:
        raise ValueError('Dead-model selection/source is not the frozen completed development artifact')
    if foundation['selected']['variant']!='full_frame' or foundation['selected']['threshold']!=.5:
        raise ValueError('This experiment requires the selected frozen full/frame dead detector at 0.50')
    config=json.loads((ROOT/OWN_SOURCES[0]).read_text())
    own_hashes={name:file_digest(ROOT/name) for name in OWN_SOURCES}
    signature=hashlib.sha256(json.dumps({'dead_cache':ds,**{name:own_hashes[name] for name in OWN_SOURCES[:2]}},sort_keys=True).encode()).hexdigest()[:16]
    return dc,split,base,dh,ds,foundation,config,own_hashes,signature


def cache_folder(match,signature):
    return dead.DATA_ROOT/match/'owner_features'/signature


def prepare(match,dc,split,base,ds,config,signature):
    if match not in dc['development_matches']+dc['validation_matches']:raise ValueError('Unapproved external fixture')
    inputs=read_external(dead.DATA_ROOT,match,split)
    original=array_digest(inputs)
    folder=cache_folder(match,signature);folder.mkdir(parents=True,exist_ok=True)
    for regime,seed in dead.scenarios(split):
        stem=f'{regime}_{seed}';manifest=folder/(stem+'.json')
        if manifest.exists():
            info=json.loads(manifest.read_text())
            if info['original_input_fingerprint']!=original:raise ValueError('Input fingerprint changed')
            for name,digest in info['files'].items():
                if file_digest(folder/name)!=digest:raise ValueError('Owner feature cache changed')
            print(f'{match} {stem}: verified owner cache',flush=True);continue
        started=time.perf_counter()
        matrix,context,old_info=dead.load_case(match,regime,seed,ds)
        noisy=corrupt(inputs,split['regimes'][regime],seed)
        noise_hash=array_digest(noisy)
        if noise_hash!=old_info['noisy_input_fingerprint']:raise ValueError('Corruption differs from frozen dead cache')
        basic=extract_features(noisy,base)
        result=extract_owner_features(basic,matrix,context['names'],noisy['period'],config)
        if np.isinf(result['values']).any():raise ValueError('Infinite ownership features')
        np.save(folder/(stem+'.npy'),result['values'],allow_pickle=False)
        np.savez_compressed(folder/(stem+'_context.npz'),names=result['names'],
            bridge_probability=result['bridge_probability'],anchor_sides=result['anchor_sides'])
        files={name:file_digest(folder/name) for name in (stem+'.npy',stem+'_context.npz')}
        manifest.write_text(json.dumps({'match_id':match,'regime':regime,'seed':seed,
            'original_input_fingerprint':original,'noisy_input_fingerprint':noise_hash,
            'files':files,'shape':list(result['values'].shape),'reference_arrays_read':False},indent=2)+'\n')
        if array_digest(noisy)!=noise_hash:raise ValueError('Ownership features modified inputs')
        print(f'{match} {stem}: {result["values"].shape}, bidirectional cache '
              f'({time.perf_counter()-started:.1f} s)',flush=True)
    if array_digest(inputs)!=original:raise ValueError('Original input modified')


def load(match,regime,seed,signature):
    folder=cache_folder(match,signature);stem=f'{regime}_{seed}'
    info=json.loads((folder/(stem+'.json')).read_text())
    for name,digest in info['files'].items():
        if file_digest(folder/name)!=digest:raise ValueError('Owner cache integrity failure')
    matrix=np.load(folder/(stem+'.npy'),mmap_mode='r',allow_pickle=False)
    with np.load(folder/(stem+'_context.npz'),allow_pickle=False) as bundle:
        context={key:bundle[key] for key in bundle.files}
    return matrix,context,info


def fit_owner(matches,weighting,dc,split,config,ds,signature):
    xs,ys,ws=[],[],[]
    for match in matches:
        labels=dead.targets(match)
        alive_labels=np.where(np.isin(labels,['H','A']),labels,'X')
        for regime,seed in dead.scenarios(split):
            matrix,_,_=load(match,regime,seed,signature)
            _,context,_=dead.load_case(match,regime,seed,ds)
            rows=training_rows(alive_labels,context['period'],dc['training_stride_frames'],seed)
            weight=sample_weights(labels,context['period'],rows,weighting,dc['sample_weight_cap'])
            weight/=weight.sum()
            if regime!='clean':weight/=len(split['seeds'])
            xs.append(np.asarray(matrix[rows],dtype=np.float32));ys.append((labels[rows]=='H').astype(np.uint8));ws.append(weight)
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws)
    w*=len(w)/w.sum()
    started=time.perf_counter()
    model=HistGradientBoostingClassifier(**config['model']).fit(x,y,sample_weight=w)
    info={'training_matches':matches,'training_rows':len(y),'features':x.shape[1],
          'fit_seconds':time.perf_counter()-started,'weighting':weighting}
    print(f'Owner {weighting}: {len(matches)} matches, {x.shape}, {info["fit_seconds"]:.1f} s',flush=True)
    return model,info


def score(labels,prediction,probability,context,result,bootstrap=0):
    gt=pm.label_codes(labels);pred=pm.label_codes(prediction)
    metric=pm.evaluate(gt,pred,bootstrap_n=bootstrap)
    alive=np.isin(labels,['H','A'])
    missing=alive&~context['ball_observed']
    error=(probability-(labels=='H'))**2
    owner_accuracy=(probability>=.5)==(labels=='H')
    inferred_alive=(context['period']>0)&(prediction!='D')
    return {**summarize_score(metric),'mean_duration_s':metric['mean_spell_duration_s'],
        'abstain_fraction_on_graded':metric['n_abstain_on_graded']/metric['n_graded'],
        'head_accuracy_true_alive':float(owner_accuracy[alive].mean()),
        'head_brier_true_alive':float(error[alive].mean()),
        'head_accuracy_true_alive_ball_missing':float(owner_accuracy[missing].mean()) if missing.any() else None,
        'low_confidence_fraction_predicted_alive':float(result['low_confidence'][inferred_alive].mean()) if inferred_alive.any() else None}


def write_report(report,destination,name):
    path=destination/name
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return path


def develop(dc,split,ds,foundation,config,hashes,dh,signature,destination):
    report={'stage':'development','status':'running','source_sha256':hashes,'dead_source_sha256':dh,
        'config':config,'cache_signature':signature,'results':[],'folds':[],'models':{},
        'development_matches':dc['development_matches'],'validation_matches':dc['validation_matches'],
        'dead_selected':foundation['selected'],'ranking':[]}
    for held in dc['development_matches']:
        training=[m for m in dc['development_matches'] if m!=held]
        assert_training_split(training,[held]+dc['validation_matches'])
        # The final all-four-match dead model cannot be used on its own training fixtures here.
        dead_model,dead_info=dead.fit(training,dc['variants']['full_frame'],dc,split,ds)
        fold={'held_out':held,'training_matches':training,'dead_fit':dead_info,'owner_fits':{}}
        owner_models={}
        for name,weighting in config['variants'].items():
            owner_models[name],info=fit_owner(training,weighting,dc,split,config,ds,signature)
            fold['owner_fits'][name]=info
        fold_path=destination/(held+'_fold_models.joblib')
        joblib.dump({'dead':dead_model,'owners':owner_models,'training_matches':training},fold_path)
        fold['model_file']=fold_path.name;fold['model_sha256']=file_digest(fold_path)
        labels=dead.targets(held)
        for regime,seed in dead.scenarios(split):
            dm,context,info=dead.load_case(held,regime,seed,ds)
            matrix,own,_=load(held,regime,seed,signature)
            dp=dead.predict_probability(dead_model,dm,np.arange(dm.shape[1]))
            probabilities={'bridge':own['bridge_probability']}
            for name,model in owner_models.items():
                probabilities[name]=dead.predict_probability(model,matrix,np.arange(matrix.shape[1]))
            for name,probability in probabilities.items():
                for confirmation in config['confirmation_s']:
                    result=compose_ownership(probability,dp,context['period'],confirmation,config['switch_confidence'])
                    report['results'].append({'match_id':held,'regime':regime,'seed':seed,
                        'variant':name,'confirmation_s':confirmation,'noisy_input_fingerprint':info['noisy_input_fingerprint'],
                        **score(labels,result['labels'],probability,context,result)})
            np.savez_compressed(destination/f'{held}_{regime}_{seed}_oof.npz',dead_probability=dp,**probabilities)
            print(f'OOF {held} {regime} seed {seed}: ownership candidates scored',flush=True)
        report['folds'].append(fold)
        write_report(report,destination,'development_checkpoint.json')
    for name in ['bridge',*config['variants']]:
        for confirmation in config['confirmation_s']:
            groups=[]
            for match in dc['development_matches']:
                for regime in split['regimes']:
                    rows=[r for r in report['results'] if r['variant']==name and
                        r['confirmation_s']==confirmation and r['match_id']==match and r['regime']==regime]
                    groups.append([np.mean([r[k] for r in rows]) for k in ('spell_f1_mean','macro_f1')])
            spell,macro=np.mean(groups,axis=0)
            report['ranking'].append({'variant':name,'confirmation_s':confirmation,
                                     'spell_f1_mean':float(spell),'macro_f1':float(macro)})
    report['ranking'].sort(key=lambda r:(-r['spell_f1_mean'],-r['macro_f1'],r['variant'],r['confirmation_s']))
    report['selected']=report['ranking'][0]
    for name,weighting in config['variants'].items():
        model,fit_info=fit_owner(dc['development_matches'],weighting,dc,split,config,ds,signature)
        matrix,context,_=load(dc['development_matches'][0],'clean',split['seeds'][0],signature)
        chosen=next(r for r in report['ranking'] if r['variant']==name)
        path=destination/(name+'.joblib')
        joblib.dump({'model':model,'names':context['names'],'training_matches':dc['development_matches']},path)
        bundle=joblib.load(path);probe=np.asarray(matrix[::101])
        if not np.array_equal(model.predict_proba(probe),bundle['model'].predict_proba(probe)):
            raise ValueError('Owner model serialization changed probabilities')
        report['models'][name]={'file':path.name,'sha256':file_digest(path),**fit_info,**chosen}
    report['status']='passed'
    path=write_report(report,destination,'development_report.json')
    print(f'PASS: ownership development selection: {path}',flush=True)


def validate(dc,split,ds,foundation,config,hashes,dh,signature,selection_path,dead_selection_path,destination):
    selection=json.loads(selection_path.read_text(encoding='utf-8'))
    if (selection.get('status')!='passed' or selection.get('stage')!='development' or
        selection['source_sha256']!=hashes or selection['dead_source_sha256']!=dh):
        raise ValueError('Owner selection/source changed or is incomplete')
    dead_info=foundation['models']['full_frame'];path=dead_selection_path.parent/dead_info['file']
    if file_digest(path)!=dead_info['sha256']:raise ValueError('Frozen dead model changed')
    dead_bundle=joblib.load(path)
    assert_training_split(dead_bundle['training_matches'],dc['validation_matches'])
    owners={}
    for name,info in selection['models'].items():
        path=selection_path.parent/info['file']
        if file_digest(path)!=info['sha256']:raise ValueError('Owner model changed')
        owners[name]=joblib.load(path)
        assert_training_split(owners[name]['training_matches'],dc['validation_matches'])
    report={'stage':'validation','status':'running','selection':str(selection_path),'source_sha256':hashes,
        'dead_source_sha256':dh,'selected':selection['selected'],'results':[],'paired_deltas':[]}
    chosen={name:next(r for r in selection['ranking'] if r['variant']==name) for name in ['bridge',*config['variants']]}
    for match in dc['validation_matches']:
        labels=dead.targets(match);gt=pm.label_codes(labels)
        count=len(pm.possession_spells(gt,pm.valid_runs(gt)))
        picks=np.random.RandomState(20261030).randint(0,count,(1000,count))
        for regime in split['regimes']:
            all_draws=[];all_points=[]
            for seed in (split['seeds'][:1] if regime=='clean' else split['seeds']):
                dm,context,info=dead.load_case(match,regime,seed,ds)
                matrix,own,_=load(match,regime,seed,signature)
                dp=dead.predict_probability(dead_bundle['model'],dm,dead_bundle['indices'])
                probabilities={'bridge':own['bridge_probability']}
                for name,bundle in owners.items():
                    if not np.array_equal(bundle['names'],own['names']):raise ValueError('Owner feature schema changed')
                    probabilities[name]=dead.predict_probability(bundle['model'],matrix,np.arange(matrix.shape[1]))
                predictions={'baseline_50':context['baseline'],'rule_solver':context['solver'],
                             'dead_overlay':np.where(dp>=.5,'D',context['owner'])}
                diagnostics={}
                for name,probability in probabilities.items():
                    result=compose_ownership(probability,dp,context['period'],chosen[name]['confirmation_s'],config['switch_confidence'])
                    predictions[name]=result['labels'];diagnostics[name]=result
                draws={};points={}
                for name,prediction in predictions.items():
                    p=probabilities.get(name,probabilities[selection['selected']['variant']])
                    result=diagnostics.get(name,{'low_confidence':np.zeros(len(labels),dtype=bool)})
                    row=score(labels,prediction,p,context,result,bootstrap=1000)
                    if name not in probabilities:
                        for key in list(row):
                            if key.startswith('head_') or key.startswith('low_confidence'):del row[key]
                    else:
                        inferred_alive=(context['period']>0)&(prediction!='D')
                        row['no_anchor_fraction_predicted_alive']=float((own['anchor_sides'][inferred_alive]==0).mean()) if inferred_alive.any() else None
                    report['results'].append({'match_id':match,'regime':regime,'seed':seed,'model':name,
                        'noisy_input_fingerprint':info['noisy_input_fingerprint'],**row})
                    points[name]=row['spell_f1_mean']
                    if name in ('baseline_50','rule_solver','dead_overlay',selection['selected']['variant']):
                        draws[name]=paired_draw_scores(gt,pm.label_codes(prediction),picks)
                all_draws.append(draws);all_points.append(points)
                np.savez_compressed(destination/f'{match}_{regime}_{seed}_probabilities.npz',dead_probability=dp,**probabilities)
                print(f'{match} {regime} seed {seed}: frozen ownership systems scored',flush=True)
            winner=selection['selected']['variant']
            for reference in ('baseline_50','rule_solver','dead_overlay'):
                delta=np.mean([r[winner]-r[reference] for r in all_draws],axis=0)
                lo,hi=np.percentile(delta,[2.5,97.5])
                report['paired_deltas'].append({'match_id':match,'regime':regime,'model':winner,
                    'comparator':reference,'spell_f1_delta':float(np.mean([r[winner]-r[reference] for r in all_points])),
                    'conditional_95_low':float(lo),'conditional_95_high':float(hi),
                    'method':'Shared GT-possession resampling, fixed matching/prediction counts; seed scores averaged within draw; no cross-match interval.'})
    report['status']='passed'
    path=write_report(report,destination,'validation_report.json')
    print(f'PASS: ownership validation: {path}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['prepare','development','validation'])
    parser.add_argument('--dead-selection',type=Path,required=True)
    parser.add_argument('--selection',type=Path)
    parser.add_argument('--matches',nargs='+')
    args=parser.parse_args()
    dc,split,base,dh,ds,foundation,config,hashes,signature=setup(args.dead_selection)
    with threadpool_limits(limits=dc['threads']):
        if args.stage=='prepare':
            for match in args.matches or dc['development_matches']:
                prepare(match,dc,split,base,ds,config,signature)
            return
        destination=ROOT/'outputs/gap_ownership'/(args.stage+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        destination.mkdir(parents=True,exist_ok=False)
        if args.stage=='development':develop(dc,split,ds,foundation,config,hashes,dh,signature,destination)
        else:
            if args.selection is None:parser.error('Validation requires --selection')
            validate(dc,split,ds,foundation,config,hashes,dh,signature,args.selection,args.dead_selection,destination)


if __name__=='__main__':main()
