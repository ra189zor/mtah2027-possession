"""Audit frozen evidence and exact repeated predictions for short-possession recovery."""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'solution/tools'))
import improve_short_possessions as s
n=s.n
from possession.inputs import file_digest
from possession.robustness import excluded_pairs,nested_training_matches


def main(out):
    cfg,config,hashes,foundation=s.setup()
    if n.read(out/'protocol.json')['source_sha256']!=hashes:raise ValueError('Follow-up source drift')
    final=n.read(out/'validation_report.json')
    if final['status']!='passed' or len(final['results'])!=420:raise ValueError('Incomplete follow-up')
    checks=0;repeated=0;originals=0
    for pair in excluded_pairs(cfg['matches']):
        short=n.read(out/'inner'/('_'.join(pair)+'.json'))
        prior=n.read(s.SOURCE/'inner'/'_'.join(pair)/'report.json')
        if short['training_matches']!=nested_training_matches(cfg['matches'],pair):raise ValueError('Leaked inner membership')
        if len(short['results'])!=168 or short['source_sha256']!=hashes:raise ValueError('Incomplete inner follow-up')
        for name,digest in prior['files'].items():
            if file_digest(s.SOURCE/'inner'/'_'.join(pair)/name)!=digest:raise ValueError('Original inner model/probability changed')
            checks+=1
    for match in cfg['matches']:
        controls=n.read(out/'control'/f'{match}.json')
        for path,digest in controls['files'].items():
            if file_digest(Path(path))!=digest:raise ValueError('Control artifact drift')
            checks+=1
        evaluation=n.read(out/'evaluation'/match/'report.json')
        old_eval=n.read(s.SOURCE/'evaluation'/match/'report.json')
        fold=n.read(s.SOURCE/'outer'/match/'report.json')
        if fold['training_matches']!=nested_training_matches(cfg['matches'],[match]):raise ValueError('Outer fit membership')
        if file_digest(s.SOURCE/'outer'/match/'models.joblib')!=fold['files']['models.joblib']:raise ValueError('Original outer model changed')
        checks+=1
        index=n.read(s.SOURCE/'prepared'/match/'index.json')
        selected=n.read(out/'selection.json')['folds'][match]
        base=n.read(s.SOURCE/'selection.json')['folds'][match]['selected']['profile']
        for row in evaluation['results']:
            if row['model']!='current':continue
            old=next(r for r in old_eval['results'] if all(r[k]==row[k] for k in ('condition','seed','label_set')) and r['model']=='nested_decoder')
            if any(row[k]!=old[k] for k in ('spell_f1_mean','macro_f1','n_pred_spells','dead_f1','confusion')):raise ValueError('Original score drift')
            originals+=1
        for condition,seed in n.cases(cfg,outer=True):
            path=s.SOURCE/'outer'/match/f'{match}_{condition}_{seed}_probabilities.npz'
            if file_digest(path)!=evaluation['probability_sha256'][str(path)]:raise ValueError('Original outer probability changed')
            checks+=1
            with np.load(path,allow_pickle=False) as z:h=z['home'];d=z['dead']
            with np.load(index['cases'][f'{condition}_{seed}']['context'],allow_pickle=False) as z:period=z['period']
            control=np.load(s.control_path(out,match,condition,seed),allow_pickle=False)
            cache={}
            for name in ('score_first','count_balanced'):
                candidate=selected[name]['profile'];key=(candidate['strength'],candidate['switch'])
                if key not in cache:cache[key]=s.predict(h,d,period,control,candidate,base,foundation[7]['epsilon'])
                csv_name=f'{match}_{condition}_{seed}_{name}.csv';csv_path=out/'evaluation'/match/csv_name
                if file_digest(csv_path)!=evaluation['csv_sha256'][csv_name]:raise ValueError('New CSV hash mismatch')
                table=n.pm.read_label_csv(str(csv_path));error=n.pm.validate_submission(table,len(period))
                if error:raise ValueError(error)
                if not np.array_equal(cache[key],table['label'].to_numpy()):raise ValueError('Repeated short decoder labels differ')
                if (cache[key][period>0]=='X').any():raise ValueError('Playing-period X')
                repeated+=1;checks+=1
        print(f'{match}: new predictions repeat exactly; original scores/evidence intact',flush=True)
    result={'status':'passed','source_sha256':hashes,'audit_source_sha256':file_digest(Path(__file__)),
        'artifact_hash_checks':checks,'original_score_rows_exact':originals,'new_csvs_validated_and_repeated_exactly':repeated,
        'inner_candidate_rows':3528,'outer_score_rows':420,'models_retrained':0,'competition_data_used':False}
    n.write(out/'integrity_audit.json',result)
    print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output.resolve())
