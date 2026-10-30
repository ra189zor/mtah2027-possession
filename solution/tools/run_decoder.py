"""Select temporal costs on cached external OOF probabilities; validate a frozen decoder."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'solution/src'),str(ROOT/'solution/tools')]
import train_dead_model as dead
from possession.decoder import candidates,decode
from possession.gap_ownership import compose_ownership
from possession.inputs import file_digest,array_digest
from possession.validation import period_blocks
from benchmark_external import summarize_score
from run_oracles import paired_draw_scores
import possession_metrics as pm

SOURCES=('solution/configs/decoder.json','solution/src/possession/decoder.py','solution/tools/run_decoder.py')


def setup(ownership_path):
    dc,split,_,dh,ds=dead.settings()
    owner=json.loads(ownership_path.read_text(encoding='utf-8'))
    if owner.get('stage')!='development' or owner.get('status')!='passed' or owner['dead_source_sha256']!=dh:
        raise ValueError('Ownership selection is incomplete or upstream sources changed')
    for name,digest in owner['source_sha256'].items():
        if file_digest(ROOT/name)!=digest:raise ValueError('Frozen ownership source changed')
    if owner['development_matches']!=dc['development_matches'] or owner['validation_matches']!=dc['validation_matches']:
        raise ValueError('Frozen fixture roles changed')
    for fold in owner['folds']:
        if fold['held_out'] in fold['training_matches'] or set(fold['training_matches'])&set(dc['validation_matches']):
            raise ValueError('Leaked upstream development fold')
        if file_digest(ownership_path.parent/fold['model_file'])!=fold['model_sha256']:
            raise ValueError('Upstream fold-model artifact changed')
    config=json.loads((ROOT/SOURCES[0]).read_text())
    hashes={name:file_digest(ROOT/name) for name in SOURCES}
    return dc,split,ds,owner,config,hashes


def context(match,regime,seed,ds):
    folder=dead.folder_for(match,ds);stem=f'{regime}_{seed}'
    info=json.loads((folder/(stem+'.json')).read_text())
    name=stem+'_context.npz'
    if file_digest(folder/name)!=info['files'][name]:raise ValueError('Frozen context changed')
    with np.load(folder/name,allow_pickle=False) as z:
        return {key:z[key] for key in ('period','owner','solver','baseline')},info


def load_probabilities(path,variant,period):
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'dead_probability','bridge','learned_frame','learned_spell'}:
            raise ValueError('Unexpected probability fields; labels are not accepted')
        h=z[variant].copy();d=z['dead_probability'].copy()
    if h.shape!=period.shape or d.shape!=period.shape:raise ValueError('Probability/frame mismatch')
    return h,d


def score(labels,prediction,period,bootstrap=0):
    gt=pm.label_codes(labels);pred=pm.label_codes(prediction)
    result=pm.evaluate(gt,pred,bootstrap_n=bootstrap)
    islands=[]
    for a,b in period_blocks(period):
        edges=np.diff(np.r_[0,(prediction[a:b]!='D').astype(np.int8),0])
        islands.extend(zip(np.flatnonzero(edges==1)+a,np.flatnonzero(edges==-1)+a))
    return {**summarize_score(result),'mean_duration_s':result['mean_spell_duration_s'],
        'gt_mean_duration_s':result['gt_mean_spell_duration_s'],
        'dead_precision':result['per_class']['D']['precision'],'dead_recall':result['per_class']['D']['recall'],
        'confusion':pm.scored_confusion(pm.confusion(gt,pred)).tolist(),
        'abstain_fraction_on_graded':result['n_abstain_on_graded']/result['n_graded'],
        'predicted_alive_islands':len(islands),
        'short_alive_islands_1s':int(sum(b-a<=25 for a,b in islands)),
        'alive_islands_wholly_reference_dead':sum(bool((labels[a:b]=='D').all()) for a,b in islands)}


def development_worker(job):
    started=time.perf_counter()
    h,d=load_probabilities(Path(job['probability_file']),job['owner_variant'],job['period'])
    before=array_digest({'home':h,'dead':d})
    rows=[]
    for profile in job['profiles']:
        result=decode(h,d,job['period'],profile,job['epsilon'])
        rows.append({'match_id':job['match_id'],'regime':job['regime'],'seed':job['seed'],
            'candidate':profile['id'],**score(job['labels'],result['labels'],job['period'])})
    if array_digest({'home':h,'dead':d})!=before:raise ValueError('Decoder modified cached probabilities')
    return rows,time.perf_counter()-started


def write(report,destination,name):
    path=destination/name;path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');return path


def development(dc,split,ds,owner,config,hashes,owner_path,destination):
    profiles=candidates(config)
    report={'stage':'development','status':'running','source_sha256':hashes,
        'ownership_selection':str(owner_path),'ownership_source_sha256':owner['source_sha256'],
        'dead_source_sha256':owner['dead_source_sha256'],'owner_selected':owner['selected'],
        'config':config,'profiles':profiles,'probability_sha256':{},'results':[],'ranking':[]}
    jobs=[]
    for match in dc['development_matches']:
        labels=dead.targets(match)
        for regime,seed in dead.scenarios(split):
            ctx,info=context(match,regime,seed,ds)
            path=owner_path.parent/f'{match}_{regime}_{seed}_oof.npz'
            report['probability_sha256'][str(path)]=file_digest(path)
            jobs.append({'match_id':match,'regime':regime,'seed':seed,'labels':labels,'period':ctx['period'],
                'probability_file':str(path),'owner_variant':owner['selected']['variant'],
                'profiles':profiles,'epsilon':config['epsilon']})
    with ProcessPoolExecutor(max_workers=config['workers']) as pool:
        for job,(rows,seconds) in zip(jobs,pool.map(development_worker,jobs)):
            report['results'].extend(rows)
            print(f'{job["match_id"]} {job["regime"]} seed {job["seed"]}: '
                  f'{len(rows)} decoder candidates scored ({seconds:.1f} worker s)',flush=True)
            write(report,destination,'development_checkpoint.json')
    for profile in profiles:
        groups=[]
        for match in dc['development_matches']:
            for regime in split['regimes']:
                rows=[r for r in report['results'] if r['candidate']==profile['id'] and r['match_id']==match and r['regime']==regime]
                groups.append([np.mean([r[k] for r in rows]) for k in ('spell_f1_mean','macro_f1')])
        spell,macro=np.mean(groups,axis=0)
        report['ranking'].append({'profile':profile,'spell_f1_mean':float(spell),'macro_f1':float(macro)})
    report['ranking'].sort(key=lambda r:(-r['spell_f1_mean'],-r['macro_f1'],r['profile']['id']))
    report['selected']=report['ranking'][0]
    report['family_winners']={mode:next(row for row in report['ranking'] if row['profile']['mode']==mode)
                              for mode in config['modes']}
    # Cost sensitivity only: the underlying OOF heads are not refitted in a nested outer loop.
    stability=[]
    for held in dc['development_matches']:
        rankings=[]
        for profile in profiles:
            means=[]
            for match in dc['development_matches']:
                if match==held:continue
                for regime in split['regimes']:
                    rows=[r for r in report['results'] if r['candidate']==profile['id'] and r['match_id']==match and r['regime']==regime]
                    means.append(np.mean([r['spell_f1_mean'] for r in rows]))
            rankings.append((float(np.mean(means)),profile['id']))
        rankings.sort(key=lambda row:(-row[0],row[1]))
        stability.append({'excluded_from_cost_selection':held,'chosen_profile':rankings[0][1]})
    report['cost_selection_sensitivity']=stability
    for name,digest in report['probability_sha256'].items():
        if file_digest(Path(name))!=digest:raise ValueError('Development probability artifact changed')
    report['status']='passed';path=write(report,destination,'development_report.json')
    print(f'PASS: frozen decoder selection: {path}',flush=True)


def validation(dc,split,ds,owner,config,hashes,owner_path,selection_path,probability_report_path,destination):
    selection=json.loads(selection_path.read_text(encoding='utf-8'))
    upstream=json.loads(probability_report_path.read_text(encoding='utf-8'))
    if (selection.get('status')!='passed' or selection.get('stage')!='development' or
        selection['source_sha256']!=hashes or selection['ownership_source_sha256']!=owner['source_sha256'] or
        selection['dead_source_sha256']!=owner['dead_source_sha256']):raise ValueError('Decoder selection is stale/incomplete')
    if (upstream.get('stage')!='validation' or upstream.get('status')!='passed' or
        upstream['source_sha256']!=owner['source_sha256'] or upstream['dead_source_sha256']!=owner['dead_source_sha256'] or
        upstream['selected']!=owner['selected']):raise ValueError('Validation probabilities do not belong to the frozen owner model')
    profiles={'decoder_selected':selection['selected']['profile'],
        'decoder_fixed':selection['family_winners']['fixed']['profile'],
        'decoder_evidence':selection['family_winners']['evidence']['profile'],'frame_map':candidates(config)[0]}
    report={'stage':'validation','status':'running','source_sha256':hashes,
        'selection':str(selection_path),'ownership_selection':str(owner_path),'validation_probabilities':str(probability_report_path),
        'selected':selection['selected'],'profiles':profiles,'results':[],'paired_deltas':[],
        'probability_sha256':{},'prediction_csvs':[]}
    for match in dc['validation_matches']:
        labels=dead.targets(match);gt=pm.label_codes(labels)
        n_gt=len(pm.possession_spells(gt,pm.valid_runs(gt)))
        picks=np.random.RandomState(20261030).randint(0,n_gt,(1000,n_gt))
        for regime in split['regimes']:
            seed_draws=[];seed_points=[]
            for seed in (split['seeds'][:1] if regime=='clean' else split['seeds']):
                ctx,info=context(match,regime,seed,ds);period=ctx['period']
                path=probability_report_path.parent/f'{match}_{regime}_{seed}_probabilities.npz'
                report['probability_sha256'][str(path)]=file_digest(path)
                h,d=load_probabilities(path,owner['selected']['variant'],period)
                previous=compose_ownership(h,d,period,owner['selected']['confirmation_s'],
                    json.loads((ROOT/'solution/configs/gap_ownership.json').read_text())['switch_confidence'])['labels']
                prior=next(r for r in upstream['results'] if r['match_id']==match and r['regime']==regime and r['seed']==seed and r['model']==owner['selected']['variant'])
                reference_score=score(labels,previous,period)
                if any(reference_score[k]!=prior[k] for k in ('spell_f1_mean','macro_f1','n_pred_spells')):
                    raise ValueError('Frozen previous gap system did not reproduce')
                predictions={'previous_gap':previous,'dead_overlay':np.where(d>=.5,'D',ctx['owner']),
                             'rule_solver':ctx['solver'],'baseline_50':ctx['baseline']}
                by_profile={}
                for name,profile in profiles.items():
                    if profile['id'] not in by_profile:by_profile[profile['id']]=decode(h,d,period,profile,config['epsilon'])
                    predictions[name]=by_profile[profile['id']]['labels']
                repeated=decode(h,d,period,profiles['decoder_selected'],config['epsilon'])
                if not np.array_equal(repeated['labels'],predictions['decoder_selected']):raise ValueError('Decoder did not repeat exactly')
                draws={};points={}
                for name,prediction in predictions.items():
                    row=score(labels,prediction,period,bootstrap=1000)
                    report['results'].append({'match_id':match,'regime':regime,'seed':seed,'model':name,
                        'noisy_input_fingerprint':info['noisy_input_fingerprint'],**row})
                    if name not in ('decoder_fixed','decoder_evidence'):
                        draws[name]=paired_draw_scores(gt,pm.label_codes(prediction),picks);points[name]=row['spell_f1_mean']
                seed_draws.append(draws);seed_points.append(points)
                csv_path=destination/f'{match}_{regime}_{seed}_decoded.csv'
                with csv_path.open('w',newline='',encoding='utf-8') as handle:
                    writer=csv.writer(handle,lineterminator='\n');writer.writerow(['frame','label'])
                    writer.writerows(enumerate(predictions['decoder_selected']))
                table=pm.read_label_csv(str(csv_path));error=pm.validate_submission(table,len(labels))
                if error:raise ValueError(error)
                report['prediction_csvs'].append({'match_id':match,'regime':regime,'seed':seed,
                    'file':csv_path.name,'sha256':file_digest(csv_path),'rows':len(labels)})
                print(f'{match} {regime} seed {seed}: frozen decoders scored; external CSV validated',flush=True)
            for reference in ('previous_gap','frame_map','baseline_50','rule_solver','dead_overlay'):
                delta=np.mean([r['decoder_selected']-r[reference] for r in seed_draws],axis=0)
                lo,hi=np.percentile(delta,[2.5,97.5])
                report['paired_deltas'].append({'match_id':match,'regime':regime,'comparator':reference,
                    'spell_f1_delta':float(np.mean([r['decoder_selected']-r[reference] for r in seed_points])),
                    'conditional_95_low':float(lo),'conditional_95_high':float(hi),
                    'method':'Shared GT-possession resampling, fixed matching/prediction counts; seed scores averaged per draw; no cross-match interval.'})
    for name,digest in report['probability_sha256'].items():
        if file_digest(Path(name))!=digest:raise ValueError('Validation probabilities changed')
    report['status']='passed';path=write(report,destination,'validation_report.json')
    print(f'PASS: decoder validation: {path}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['development','validation'])
    parser.add_argument('--ownership-selection',type=Path,required=True)
    parser.add_argument('--selection',type=Path)
    parser.add_argument('--validation-probabilities',type=Path)
    args=parser.parse_args()
    dc,split,ds,owner,config,hashes=setup(args.ownership_selection)
    destination=ROOT/'outputs/decoder'/(args.stage+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    destination.mkdir(parents=True,exist_ok=False)
    if args.stage=='development':development(dc,split,ds,owner,config,hashes,args.ownership_selection,destination)
    else:
        if args.selection is None or args.validation_probabilities is None:
            parser.error('Validation requires --selection and --validation-probabilities')
        validation(dc,split,ds,owner,config,hashes,args.ownership_selection,args.selection,args.validation_probabilities,destination)


if __name__=='__main__':main()
