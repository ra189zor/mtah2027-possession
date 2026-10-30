"""Independently audit completed external nested folds, predictions and realized noise."""
import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'solution/tools'))
import run_nested_validation as nested
from possession.inputs import file_digest
from possession.robustness import excluded_pairs,nested_training_matches
from possession.validation import period_blocks


def audit(output):
    cfg,_,_,_,_,_,_,_,hashes,_=nested.setup()
    protocol=nested.read(output/'protocol.json')
    if protocol['source_sha256']!=hashes:raise ValueError('Protocol source drift')
    report=nested.read(output/'validation_report.json')
    if report['status']!='passed' or report['source_sha256']!=hashes:raise ValueError('Incomplete report')
    checks=0;noise=[];probes=[]
    for pair in excluded_pairs(cfg['matches']):
        path=output/'inner'/'_'.join(pair)/'report.json';fold=nested.read(path)
        if fold['training_matches']!=nested_training_matches(cfg['matches'],pair):raise ValueError('Inner membership mismatch')
        if fold['source_sha256']!=hashes or len(fold['scores'])!=350:raise ValueError('Incomplete inner scores')
        for name,digest in fold['files'].items():
            if file_digest(path.parent/name)!=digest:raise ValueError('Inner artifact hash mismatch')
            checks+=1
        model=joblib.load(path.parent/'models.joblib')
        if model['training_matches']!=fold['training_matches']:raise ValueError('Serialized inner membership mismatch')
        del model
    with threadpool_limits(limits=2):
        for match in cfg['matches']:
            folder=output/'outer'/match;fold=nested.read(folder/'report.json')
            expected=nested_training_matches(cfg['matches'],[match])
            if fold['training_matches']!=expected or fold['source_sha256']!=hashes:raise ValueError('Outer membership/source mismatch')
            for name,digest in fold['files'].items():
                if file_digest(folder/name)!=digest:raise ValueError('Outer artifact hash mismatch')
                checks+=1
            model=joblib.load(folder/'models.joblib')
            if model['training_matches']!=expected:raise ValueError('Serialized outer membership mismatch')
            index=nested.read(output/'prepared'/match/'index.json')
            for name,digest in index['files'].items():
                if file_digest(Path(name))!=digest:raise ValueError('Feature/sample hash mismatch')
                checks+=1
            for condition,seed in nested.cases(cfg,outer=True):
                case=index['cases'][f'{condition}_{seed}']
                with np.load(case['context'],allow_pickle=False) as z:
                    period=z['period'];observed=z['ball_observed'];active=period>0
                lengths=[]
                for a,b in period_blocks(period):
                    edges=np.diff(np.r_[0,(~observed[a:b]).astype(np.int8),0])
                    lengths.extend((np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)).tolist())
                noise.append({'match_id':match,'condition':condition,'seed':seed,
                    'ball_missing_fraction':float((~observed[active]).mean()),
                    'ball_gap_median_s':float(np.median(lengths)/25) if lengths else 0.,
                    'ball_gap_p95_s':float(np.percentile(lengths,95)/25) if lengths else 0.,
                    **{k:v for k,v in case.items() if k.endswith('_fraction') and k!='ball_missing_fraction'}})
            # Reloaded serialized heads must reproduce stored native-frame probabilities exactly.
            case=index['cases']['moderate_17']
            x=np.load(case['matrix'],mmap_mode='r',allow_pickle=False)[::101]
            with np.load(folder/f'{match}_moderate_17_probabilities.npz',allow_pickle=False) as z:
                for key,head,width in [('dead','dead',275),('home','owner',298)]:
                    predicted=model[head].predict_proba(np.asarray(x[:,:width]))[:,1].astype(np.float32)
                    if not np.array_equal(predicted,z[key][::101]):raise ValueError('Reloaded head does not reproduce probabilities')
            probes.append({'match_id':match,'probe_rows':len(x),'serialized_probability_reproduction':'exact'})
            evaluation=nested.read(output/'evaluation'/match/'report.json')
            for name,digest in evaluation['files'].items():
                csv_path=output/'evaluation'/match/name
                if file_digest(csv_path)!=digest:raise ValueError('External prediction CSV changed')
                checks+=1
    rows=[r for r in report['results'] if r['label_set']=='provider' and r['model']=='nested_decoder']
    if len(rows)!=70 or any(r['abstain_fraction_on_graded']!=0 for r in rows):raise ValueError('Missing outer cases/unknowns')
    gates=[]
    for match in cfg['matches']:
        for condition in cfg['outer_conditions']:
            group=[r for r in rows if r['match_id']==match and r['condition']==condition]
            errors=[r['n_pred_spells']/r['n_gt_spells']-1 for r in group]
            gates.append({'match_id':match,'condition':condition,'mean_relative_count_error':float(np.mean(errors)),
                'mean_count_within_20_percent':bool(abs(np.mean(errors))<=.2),
                'seeds_within_20_percent':sum(abs(e)<=.2 for e in errors),'seeds':len(errors)})
    result={'status':'passed','source_sha256':hashes,'audit_source_sha256':file_digest(Path(__file__)),
        'file_hash_checks':checks,'inner_fits':21,'outer_fits':7,'outer_cases':70,'score_rows':len(report['results']),
        'external_csvs':70,'competition_data_used':False,'serialized_probability_probes':probes,
        'realized_noise':noise,'provider_count_gates':gates,
        'source_role_note':'J03WN1 is an outer fixture now; old split/guards are preserved.'}
    nested.write(output/'integrity_and_noise_audit.json',result)
    print(f'PASS: {checks} artifact hashes; 28 disjoint fits; 70 external cases/CSVs; seven exact head reload probes',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    audit(parser.parse_args().output.resolve())
