"""Resumable seven-match nested decoder validation; external data only."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
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
import train_gap_ownership as owner
import prepare_sportec
import baseline
import possession_metrics as pm
from possession.dead_features import extract_dead_features
from possession.dead_model import training_rows,sample_weights
from possession.features import extract_features
from possession.gap_ownership import extract_owner_features,compose_ownership
from possession.inputs import read_external,file_digest,array_digest
from possession.validation import nearest_team_codes,period_blocks,summarize_inputs
from possession.robustness import contextual_corrupt,absorb_short_opposition,excluded_pairs,nested_training_matches
from possession.decoder import candidates,decode
from run_decoder import score
from run_oracles import paired_draw_scores

CONFIG=ROOT/'solution/configs/nested_validation.json'
OWN_SELECTION=ROOT/'outputs/gap_ownership/development_20261030T153115114216Z/development_report.json'
DEAD_SELECTION=ROOT/'outputs/dead_model/development_20261030T150036779468Z/development_report.json'
NEW_SOURCES=('solution/configs/nested_validation.json','solution/src/possession/robustness.py',
             'solution/tools/run_nested_validation.py','solution/src/possession/decoder.py',
             'solution/configs/decoder.json','solution/tools/run_decoder.py')


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def setup():
    dc,oldsplit,base,dh,ds,foundation,oc,oh,os=owner.setup(DEAD_SELECTION)
    previous=read(OWN_SELECTION)
    if previous['source_sha256']!=oh or previous['dead_source_sha256']!=dh:
        raise ValueError('Previous ownership foundation changed')
    cfg=read(CONFIG);decoder=read(ROOT/'solution/configs/decoder.json')
    if set(cfg['matches'])!=set(oldsplit['development_matches']+oldsplit['validation_matches']+oldsplit['sealed_test_matches']):
        raise ValueError('New protocol does not cover exactly the seven original external fixtures')
    split=oldsplit|{'development_matches':cfg['matches'],'validation_matches':[],'sealed_test_matches':[]}
    hashes=dh|oh|{name:file_digest(ROOT/name) for name in NEW_SOURCES}
    signature=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    return cfg,dc,split,base,ds,oc,os,decoder,hashes,signature


def cases(cfg,outer=False):
    return [(condition,seed) for condition in cfg['outer_conditions' if outer else 'inner_conditions']
            for seed in (cfg['seeds'][:1] if condition=='clean' else cfg['seeds'])]


def prepare_worker(job):
    match,out=job;out=Path(out)
    cfg,dc,split,base,ds,oc,os,_,hashes,signature=setup()
    destination=out/'prepared'/match;destination.mkdir(parents=True,exist_ok=True)
    complete=destination/'index.json'
    if complete.exists():
        index=read(complete)
        if index['source_sha256']!=hashes:raise ValueError('Stale preparation')
        for name,digest in index['files'].items():
            if file_digest(Path(name))!=digest:raise ValueError('Prepared cache changed')
        print(f'{match}: verified nested preparation',flush=True);return match
    if not (dead.DATA_ROOT/match/'prepared/inputs.npz').exists():
        prepare_sportec.prepare(match,dead.DATA_ROOT,split)
    if not (dead.folder_for(match,ds)/'clean_17.json').exists():
        dead.prepare(match,dc,split,base,ds)
    if not (owner.cache_folder(match,os)/'clean_17.json').exists():
        owner.prepare(match,dc|{'development_matches':cfg['matches'],'validation_matches':[]},split,base,ds,oc,os)
    inputs=read_external(dead.DATA_ROOT,match,split);original=array_digest(inputs)
    index={'match_id':match,'source_sha256':hashes,'input_fingerprint':original,'cases':{},'files':{}}
    train_x=[];train_y=[];train_dweight=[];train_hweight=[]
    labels=dead.targets(match)  # Separate references, only used to build supervised training samples.
    for condition,seed in cases(cfg,outer=True):
        stem=f'{condition}_{seed}'
        if condition!='contextual':
            matrix,_,oinfo=owner.load(match,condition,seed,os)
            _,ctx,dinfo=dead.load_case(match,condition,seed,ds)
            matrix_path=owner.cache_folder(match,os)/(stem+'.npy')
            context_path=dead.folder_for(match,ds)/(stem+'_context.npz')
            audit={'noisy_input_fingerprint':oinfo['noisy_input_fingerprint'],'historical_condition':True}
            if oinfo['noisy_input_fingerprint']!=dinfo['noisy_input_fingerprint']:raise ValueError('Head caches differ')
        else:
            cache=dead.DATA_ROOT/match/'nested_features'/signature;cache.mkdir(parents=True,exist_ok=True)
            matrix_path=cache/(stem+'.npy');context_path=cache/(stem+'_context.npz')
            noisy,masks,noise_audit=contextual_corrupt(inputs,split['regimes'][cfg['contextual']['base_regime']],cfg['contextual'],seed)
            noise_hash=array_digest(noisy)
            basic=extract_features(noisy,base)
            df=extract_dead_features(noisy,base,dc,basic)
            result=extract_owner_features(basic,df['values'],df['names'],noisy['period'],oc)
            matrix=result['values'];np.save(matrix_path,matrix,allow_pickle=False)
            codes=nearest_team_codes(noisy);reference=np.full(len(codes),'X',dtype='<U1')
            for a,b in period_blocks(noisy['period']):reference[a:b]=baseline.smooth(codes[a:b],50)
            ctx={'period':noisy['period'],'baseline':reference,'ball_observed':np.isfinite(noisy['ball_xy']).all(axis=1)}
            np.savez_compressed(context_path,**ctx,**masks)
            if array_digest(noisy)!=noise_hash:raise ValueError('Features modified contextual inputs')
            audit={'noisy_input_fingerprint':noise_hash,'historical_condition':False,**noise_audit}
            del noisy,basic,df,result
        if matrix.shape[1]!=298:raise ValueError('Expected unchanged 275+23 feature layout')
        index['cases'][stem]={'matrix':str(matrix_path),'context':str(context_path),**audit}
        for path in (matrix_path,context_path):index['files'][str(path)]=file_digest(path)
        if condition!='contextual':
            rows=training_rows(labels,ctx['period'],dc['training_stride_frames'],seed)
            dw=sample_weights(labels,ctx['period'],rows,'frame',dc['sample_weight_cap'])
            dw/=dw.sum()
            hw=np.zeros(len(rows));alive=np.isin(labels[rows],['H','A'])
            ow=sample_weights(labels,ctx['period'],rows[alive],cfg['owner_weighting'],dc['sample_weight_cap'])
            ow/=ow.sum();hw[alive]=ow
            if condition!='clean':dw/=len(cfg['seeds']);hw/=len(cfg['seeds'])
            train_x.append(np.asarray(matrix[rows],dtype=np.float32));train_y.append(labels[rows])
            train_dweight.append(dw);train_hweight.append(hw)
        print(f'{match} {stem}: nested inputs ready',flush=True)
        del matrix
    x_path=destination/'training_x.npy';y_path=destination/'training_targets.npz'
    np.save(x_path,np.concatenate(train_x),allow_pickle=False)
    np.savez_compressed(y_path,labels=np.concatenate(train_y),dead_weight=np.concatenate(train_dweight),owner_weight=np.concatenate(train_hweight))
    for path in (x_path,y_path):index['files'][str(path)]=file_digest(path)
    index['training_x']=str(x_path);index['training_targets']=str(y_path)
    if array_digest(inputs)!=original:raise ValueError('Preparation changed source inputs')
    write(complete,index);return match


def fit_heads(training,out,cfg,dc,oc):
    xs=[];ys=[];dw=[];hw=[]
    for match in training:
        index=read(out/'prepared'/match/'index.json')
        xs.append(np.load(index['training_x'],allow_pickle=False))
        with np.load(index['training_targets'],allow_pickle=False) as z:
            ys.append(z['labels']);dw.append(z['dead_weight']);hw.append(z['owner_weight'])
    x=np.concatenate(xs);y=np.concatenate(ys);wd=np.concatenate(dw);wh=np.concatenate(hw)
    wd*=len(wd)/wd.sum();alive=wh>0;wh=wh[alive];wh*=len(wh)/wh.sum()
    del xs,ys,dw,hw
    started=time.perf_counter()
    dm=HistGradientBoostingClassifier(**dc['model']).fit(x[:,:275],y=='D',sample_weight=wd)
    hm=HistGradientBoostingClassifier(**oc['model']).fit(x[alive],y[alive]=='H',sample_weight=wh)
    return {'dead':dm,'owner':hm,'training_matches':training}, {
        'training_matches':training,'dead_rows':len(y),'owner_rows':int(alive.sum()),
        'fit_seconds':time.perf_counter()-started,'dead_weighting':'frame','owner_weighting':cfg['owner_weighting']}


def infer(bundle,index,condition,seed):
    case=index['cases'][f'{condition}_{seed}']
    matrix=np.load(case['matrix'],mmap_mode='r',allow_pickle=False)
    d=dead.predict_probability(bundle['dead'],matrix,np.arange(275))
    h=dead.predict_probability(bundle['owner'],matrix,np.arange(298))
    with np.load(case['context'],allow_pickle=False) as z:ctx={k:z[k] for k in ('period','baseline')}
    return h,d,ctx


def fold_worker(job):
    excluded,out=job;out=Path(out)
    cfg,dc,_,_,_,oc,_,dec,hashes,_=setup()
    training=nested_training_matches(cfg['matches'],excluded)
    folder=out/('inner' if len(excluded)==2 else 'outer')/('_'.join(excluded));folder.mkdir(parents=True,exist_ok=True)
    complete=folder/'report.json'
    if complete.exists():
        report=read(complete)
        if report['source_sha256']!=hashes or report['training_matches']!=training:raise ValueError('Stale fold')
        for name,digest in report['files'].items():
            if file_digest(folder/name)!=digest:raise ValueError('Fold artifact changed')
        print(f'{excluded}: verified completed fold',flush=True);return str(complete)
    with threadpool_limits(limits=cfg['threads_per_worker']):
        model_path=folder/'models.joblib';fit_path=folder/'fit.json'
        if fit_path.exists():
            fit_info=read(fit_path)
            if fit_info['source_sha256']!=hashes or file_digest(model_path)!=fit_info['sha256']:
                raise ValueError('Stale interrupted model fit')
            bundle=joblib.load(model_path)
            if bundle['training_matches']!=training:raise ValueError('Leaked restored model')
        else:
            bundle,fit_info=fit_heads(training,out,cfg,dc,oc)
            joblib.dump(bundle,model_path)
            fit_info|={'source_sha256':hashes,'sha256':file_digest(model_path)};write(fit_path,fit_info)
        print(f'Excluded {excluded}: heads fitted on {len(training)} matches ({fit_info["fit_seconds"]:.1f} s)',flush=True)
        report={'excluded':excluded,'training_matches':training,'source_sha256':hashes,'fit':fit_info,'scores':[],'files':{}}
        for held in excluded:
            index=read(out/'prepared'/held/'index.json')
            # Inner references belong only to the inner held-out match; no outer target argument is used.
            for condition,seed in cases(cfg,outer=len(excluded)==1):
                h,d,ctx=infer(bundle,index,condition,seed)
                path=folder/f'{held}_{condition}_{seed}_probabilities.npz'
                np.savez_compressed(path,home=h,dead=d)
                report['files'][path.name]=file_digest(path)
                if len(excluded)==2:
                    labels=dead.targets(held)
                    for profile in candidates(dec):
                        pred=decode(h,d,ctx['period'],profile,dec['epsilon'])['labels']
                        metric=pm.evaluate(pm.label_codes(labels),pm.label_codes(pred),bootstrap_n=0)
                        report['scores'].append({'held_out':held,'condition':condition,'seed':seed,'candidate':profile['id'],
                            'spell_f1_mean':metric['spell_f1_mean'],'macro_f1':metric['macro_f1']})
                print(f'Excluded {excluded}: {held} {condition} seed {seed} '+('inner scores' if len(excluded)==2 else 'outer probabilities'),flush=True)
        report['files'][model_path.name]=file_digest(model_path)
        report['status']='passed';write(complete,report)
    return str(complete)


def select_inner(out,cfg,decoder,hashes):
    profiles=candidates(decoder);result={}
    reports=[read(out/'inner'/('_'.join(pair))/'report.json') for pair in excluded_pairs(cfg['matches'])]
    for outer in cfg['matches']:
        inner=[r for r in reports if outer in r['excluded']]
        if len(inner)!=6:raise ValueError('Incomplete nested folds')
        rows=[]
        for r in inner:
            if outer in r['training_matches']:raise ValueError('Outer fixture leaked into inner fit')
            held=next(m for m in r['excluded'] if m!=outer)
            rows.extend(x for x in r['scores'] if x['held_out']==held)
        ranking=[]
        for profile in profiles:
            groups=[]
            for held in cfg['matches']:
                if held==outer:continue
                for condition in cfg['inner_conditions']:
                    scores=[r for r in rows if r['candidate']==profile['id'] and r['held_out']==held and r['condition']==condition]
                    expected=1 if condition=='clean' else len(cfg['seeds'])
                    if len(scores)!=expected:raise ValueError('Incomplete inner scores')
                    groups.append([np.mean([r[k] for r in scores]) for k in ('spell_f1_mean','macro_f1')])
            spell,macro=np.mean(groups,axis=0)
            ranking.append({'profile':profile,'spell_f1_mean':float(spell),'macro_f1':float(macro)})
        ranking.sort(key=lambda r:(-r['spell_f1_mean'],-r['macro_f1'],r['profile']['id']))
        result[outer]={'selected':ranking[0],'ranking':ranking,'training_matches':nested_training_matches(cfg['matches'],[outer]),
                       'inner_model_reports':[str(out/'inner'/('_'.join(r['excluded']))/'report.json') for r in inner]}
        print(f'Outer {outer}: frozen inner selection {ranking[0]["profile"]["id"]}',flush=True)
    report={'status':'passed','source_sha256':hashes,'folds':result}
    write(out/'selection.json',report);return report


def evaluate_worker(job):
    held,out=job;out=Path(out)
    cfg,_,_,_,_,oc,_,dec,hashes,_=setup()
    destination=out/'evaluation'/held;destination.mkdir(parents=True,exist_ok=True)
    complete=destination/'report.json'
    if complete.exists():
        result=read(complete)
        if result['source_sha256']!=hashes:raise ValueError('Stale evaluation')
        for name,digest in result['files'].items():
            if file_digest(destination/name)!=digest:raise ValueError('Prediction changed')
        print(f'{held}: verified completed outer evaluation',flush=True);return str(complete)
    selection=read(out/'selection.json')
    profile=selection['folds'][held]['selected']['profile']
    fixed={'id':'prior_fixed_s40_r40_o8','mode':'fixed','stop':40,'restart':40,'switch':8}
    # References are opened only after all inner choices have been frozen.
    labels=dead.targets(held)
    index=read(out/'prepared'/held/'index.json')
    period_path=index['cases']['clean_17']['context']
    with np.load(period_path,allow_pickle=False) as z:period=z['period']
    alternate,label_audit=absorb_short_opposition(labels,period,cfg['label_sensitivity_s'])
    references={'provider':labels,'absorbed_under3s':alternate}
    report={'match_id':held,'source_sha256':hashes,'selected':profile,'label_sensitivity':label_audit,
            'results':[],'paired_deltas':[],'files':{},'noise':[],'probability_sha256':{}}
    with threadpool_limits(limits=cfg['threads_per_worker']):
        for condition in cfg['outer_conditions']:
            draws={label_set:[] for label_set in references};points={label_set:[] for label_set in references}
            picksets={}
            for label_set,y in references.items():
                gt=pm.label_codes(y);ng=len(pm.possession_spells(gt,pm.valid_runs(gt)))
                picksets[label_set]=np.random.RandomState(cfg['bootstrap_seed']).randint(0,ng,(cfg['bootstrap_draws'],ng))
            for c,seed in cases(cfg,outer=True):
                if c!=condition:continue
                path=out/'outer'/held/f'{held}_{condition}_{seed}_probabilities.npz'
                digest=file_digest(path);report['probability_sha256'][str(path)]=digest
                with np.load(path,allow_pickle=False) as z:h=z['home'];d=z['dead']
                case=index['cases'][f'{condition}_{seed}']
                with np.load(case['context'],allow_pickle=False) as z:baseline_labels=z['baseline']
                predictions={'nested_decoder':decode(h,d,period,profile,dec['epsilon'])['labels'],
                    'prior_fixed_costs':decode(h,d,period,fixed,dec['epsilon'])['labels'],
                    'gap_composition':compose_ownership(h,d,period,.8,oc['switch_confidence'])['labels'],
                    'baseline_50':baseline_labels}
                repeated=decode(h,d,period,profile,dec['epsilon'])['labels']
                if not np.array_equal(repeated,predictions['nested_decoder']):raise ValueError('Nonrepeatable decoder')
                if (repeated[period>0]=='X').any():raise ValueError('Decoder left playing-period unknowns')
                csv_path=destination/f'{held}_{condition}_{seed}_decoded.csv'
                with csv_path.open('w',encoding='utf-8',newline='') as handle:
                    handle.write('frame,label\n')
                    handle.writelines(f'{i},{v}\n' for i,v in enumerate(repeated))
                error=pm.validate_submission(pm.read_label_csv(str(csv_path)),len(labels))
                if error:raise ValueError(error)
                report['files'][csv_path.name]=file_digest(csv_path)
                for label_set,y in references.items():
                    seed_draws={};seed_points={}
                    for model,pred in predictions.items():
                        row=score(y,pred,period,bootstrap=0)
                        report['results'].append({'match_id':held,'condition':condition,'seed':seed,
                            'label_set':label_set,'model':model,**row})
                        seed_draws[model]=paired_draw_scores(pm.label_codes(y),pm.label_codes(pred),picksets[label_set])
                        seed_points[model]=row['spell_f1_mean']
                    draws[label_set].append(seed_draws);points[label_set].append(seed_points)
                report['noise'].append({'condition':condition,'seed':seed,**{k:v for k,v in case.items() if k not in ('matrix','context')}})
                if file_digest(path)!=digest:raise ValueError('Evaluation changed probability file')
            for label_set in references:
                for comparator in ('prior_fixed_costs','gap_composition','baseline_50'):
                    delta=np.mean([r['nested_decoder']-r[comparator] for r in draws[label_set]],axis=0)
                    low,high=np.percentile(delta,[2.5,97.5])
                    point=np.mean([r['nested_decoder']-r[comparator] for r in points[label_set]])
                    report['paired_deltas'].append({'condition':condition,'label_set':label_set,'comparator':comparator,
                        'delta':float(point),'conditional_low':float(low),'conditional_high':float(high)})
            print(f'{held} {condition}: both reference conventions evaluated',flush=True)
    report['status']='passed';write(complete,report);return str(complete)


def aggregate(out,cfg,hashes):
    reports=[read(out/'evaluation'/held/'report.json') for held in cfg['matches']]
    rows=[row for report in reports for row in report['results']]
    summary=[];paired=[]
    for label_set in ('provider','absorbed_under3s'):
        for condition in cfg['outer_conditions']:
            for model in ('nested_decoder','prior_fixed_costs','gap_composition','baseline_50'):
                group=[]
                for held in cfg['matches']:
                    subset=[r for r in rows if r['match_id']==held and r['label_set']==label_set and r['condition']==condition and r['model']==model]
                    group.append({k:float(np.mean([r[k] for r in subset])) for k in
                        ('spell_f1_mean','macro_f1','dead_precision','dead_recall','n_pred_spells','n_gt_spells','mean_duration_s','gt_mean_duration_s','abstain_fraction_on_graded')})
                summary.append({'label_set':label_set,'condition':condition,'model':model,
                                **{k:float(np.mean([g[k] for g in group])) for k in group[0]}})
            for comparator in ('prior_fixed_costs','gap_composition','baseline_50'):
                values=np.array([next(r['delta'] for r in report['paired_deltas'] if r['condition']==condition and r['label_set']==label_set and r['comparator']==comparator) for report in reports])
                picks=np.random.RandomState(cfg['bootstrap_seed']).randint(0,len(values),(cfg['bootstrap_draws'],len(values)))
                low,high=np.percentile(values[picks].mean(axis=1),[2.5,97.5])
                paired.append({'label_set':label_set,'condition':condition,'comparator':comparator,'delta':float(values.mean()),
                    'descriptive_match_bootstrap_low':float(low),'descriptive_match_bootstrap_high':float(high),
                    'positive_matches':int((values>0).sum()),'zero_matches':int((values==0).sum())})
    result={'status':'passed','protocol':read(out/'protocol.json'),'selection':read(out/'selection.json'),
        'source_sha256':hashes,'summary':summary,'paired_match_deltas':paired,'results':rows,
        'per_match_reports':[str(out/'evaluation'/held/'report.json') for held in cfg['matches']],
        'uncertainty_limits':cfg['uncertainty'],'historical_adaptation_limit':cfg['protocol']}
    expected=len(cfg['matches'])*len(cases(cfg,outer=True))*2*4
    if len(rows)!=expected:raise ValueError('Incomplete result rows')
    # Recheck every evidence/sample/output artifact before declaring completion.
    checked=0
    for held in cfg['matches']:
        index=read(out/'prepared'/held/'index.json')
        for path,digest in index['files'].items():
            if file_digest(Path(path))!=digest:raise ValueError('Evidence/sample artifact changed')
            checked+=1
        evaluation=read(out/'evaluation'/held/'report.json')
        for path,digest in evaluation['probability_sha256'].items():
            if file_digest(Path(path))!=digest:raise ValueError('Probability artifact changed')
            checked+=1
    result['integrity_checks']=checked
    write(out/'validation_report.json',result)
    print(f'PASS: nested external validation: {out / "validation_report.json"}',flush=True)


def parallel(worker,jobs,workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending=[pool.submit(worker,job) for job in jobs]
        for future in as_completed(pending):future.result()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','inner','outer','evaluate','all'],default='all')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cfg,_,_,_,_,_,_,decoder,hashes,_=setup()
    protocol_path=out/'protocol.json'
    if protocol_path.exists():
        if read(protocol_path)['source_sha256']!=hashes:raise ValueError('Protocol/source changed; use a new output directory')
    else:
        write(protocol_path,{'created_utc':datetime.now(timezone.utc).isoformat(),'config':cfg,
            'decoder_grid':decoder,'source_sha256':hashes,'note':'Frozen before seventh-match preparation and new fold training.'})
    if args.stage in ('prepare','all'):parallel(prepare_worker,[(m,str(out)) for m in cfg['matches']],cfg['workers'])
    if args.stage in ('inner','all'):
        parallel(fold_worker,[(list(pair),str(out)) for pair in excluded_pairs(cfg['matches'])],cfg['workers'])
        select_inner(out,cfg,decoder,hashes)
    if args.stage in ('outer','all'):
        if not (out/'selection.json').exists():raise ValueError('Freeze inner selection before outer inference')
        parallel(fold_worker,[([m],str(out)) for m in cfg['matches']],cfg['workers'])
    if args.stage in ('evaluate','all'):
        parallel(evaluate_worker,[(m,str(out)) for m in cfg['matches']],cfg['workers'])
        aggregate(out,cfg,hashes)


if __name__=='__main__':main()
