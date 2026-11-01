"""Audit cross-fitting lineage, serialized event heads and repeated external predictions."""
import argparse
from pathlib import Path
import sys
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'solution/tools'))
import train_switch_events as s
n=s.n
from possession.inputs import file_digest
from possession.robustness import nested_training_matches
from possession.switch_events import extract_event_features
from possession.event_decoder import decode_events
from possession.decoder import decode


def main(out):
    cfg,config,hashes,f=s.setup();protocol=n.read(out/'protocol.json');report=n.read(out/'validation_report.json')
    if protocol['source_sha256']!=hashes or report['source_sha256']!=hashes or report['status']!='passed':
        raise ValueError('Incomplete or changed switch-event protocol')
    if len(report['results'])!=420:raise ValueError('Incomplete event results')
    checks=0;lineage_cases=0;repeated=0;baseline_rows=0;probes=0
    for match in cfg['matches']:
        physical=out/'physical'/match;index=n.read(physical/'index.json')
        if index['source_sha256']!=hashes or index['reference_arrays_read'] is not False:raise ValueError('Invalid physical cache lineage')
        for name,digest in index['files'].items():
            if file_digest(physical/name)!=digest:raise ValueError('Physical cache drift')
            checks+=1
    with threadpool_limits(limits=2):
        for held in cfg['matches']:
            folder=out/'outer'/held;fold=n.read(folder/'report.json');model_path=folder/'switch_model.joblib'
            if file_digest(model_path)!=fold['model_sha256']:raise ValueError('Event model checksum drift')
            checks+=1;bundle=joblib.load(model_path);training=nested_training_matches(cfg['matches'],[held])
            if bundle['source_sha256']!=hashes or bundle['training_matches']!=training or fold['training_matches']!=training:
                raise ValueError('Leaked or changed event training membership')
            if bundle['lineage']!=fold['lineage'] or len(bundle['lineage'])!=42:raise ValueError('Incomplete stacking lineage')
            for trace in bundle['lineage']:
                match=trace['training_fixture'];excluded=trace['excluded']
                if match not in training or set(excluded)!={held,match} or len(excluded)!=2:
                    raise ValueError('Invalid outer/training exclusion')
                expected=nested_training_matches(cfg['matches'],excluded)
                if trace['upstream_training_matches']!=expected or len(expected)!=5:raise ValueError('Stacking label leakage')
                probability_path=Path(trace['probability_file'])
                if file_digest(probability_path)!=trace['sha256']:raise ValueError('Cross-fitted input probability drift')
                checks+=1;lineage_cases+=1
            if not np.array_equal(bundle['model'].classes_,[0,1,2]):raise ValueError('Unexpected event classes')
            prepared=n.read(s.SOURCE/'prepared'/held/'index.json')
            base=n.read(s.SOURCE/'selection.json')['folds'][held]['selected']['profile']
            old=n.read(s.SOURCE/'evaluation'/held/'report.json')
            with np.load(prepared['cases']['clean_17']['context'],allow_pickle=False) as z:period=z['period']
            # Held-out native-frame probability probe after loading the persisted model.
            physical,names=s.physical_evidence(out,held,'moderate',17)
            with np.load(s.SOURCE/'outer'/held/f'{held}_moderate_17_probabilities.npz',allow_pickle=False) as z:h=z['home'];d=z['dead']
            x,feature_names=extract_event_features(physical,names,h,d,period,config)
            if feature_names!=bundle['feature_names']:raise ValueError('Event schema mismatch')
            q=bundle['model'].predict_proba(x[::101]).astype(np.float32)
            stored=np.load(folder/'moderate_17_events.npy',allow_pickle=False)[::101]
            if not np.array_equal(q,stored):raise ValueError('Reloaded held-out event probabilities differ')
            probes+=1;del x
            for row in fold['results']:
                if row['model']!='current':continue
                previous=next(r for r in old['results'] if all(r[k]==row[k] for k in ('condition','seed','label_set')) and r['model']=='nested_decoder')
                if any(row[k]!=previous[k] for k in ('spell_f1_mean','macro_f1','n_pred_spells','confusion','dead_f1')):
                    raise ValueError('Original event comparator did not reproduce')
                baseline_rows+=1
            for condition,seed in n.cases(cfg,outer=True):
                path=s.SOURCE/'outer'/held/f'{held}_{condition}_{seed}_probabilities.npz'
                if file_digest(path)!=fold['upstream_sha256'][str(path)]:raise ValueError('Original outer probability drift')
                checks+=1
                with np.load(path,allow_pickle=False) as z:h=z['home'];d=z['dead']
                q=np.load(folder/f'{condition}_{seed}_events.npy',allow_pickle=False)
                zero=decode_events(h,d,period,q,base,0,config,f[7]['epsilon'])
                original=decode(h,d,period,base,f[7]['epsilon'])
                if not np.array_equal(zero['labels'],original['labels']) or zero['path_cost']!=original['path_cost']:
                    raise ValueError('Native reward-disabled reference differs')
                for strength in config['reward_strengths']:
                    labels=decode_events(h,d,period,q,base,strength,config,f[7]['epsilon'])['labels']
                    filename=f'{condition}_{seed}_event_reward_{strength}.csv'
                    table=n.pm.read_label_csv(str(folder/filename));error=n.pm.validate_submission(table,len(period))
                    if error:raise ValueError(error)
                    if not np.array_equal(labels,table['label'].to_numpy()):raise ValueError('Event labels do not repeat')
                    if (labels[period>0]=='X').any():raise ValueError('Playing-period X')
                    repeated+=1
            for name,digest in fold['files'].items():
                if file_digest(folder/name)!=digest:raise ValueError('Event output drift')
                checks+=1
            print(f'{held}: stacking exclusions, exact event reload, baseline and twenty CSV repetitions pass',flush=True)
    result={'status':'passed','source_sha256':hashes,'audit_source_sha256':file_digest(Path(__file__)),
        'artifact_hash_checks':checks,'cross_fitted_training_cases':lineage_cases,'disjoint_switch_fits':7,
        'exact_held_out_event_probability_probes':probes,'original_score_rows_exact':baseline_rows,
        'csvs_validated_and_repeated_exactly':repeated,'native_zero_reward_equivalence_cases':70,
        'competition_data_used':False}
    n.write(out/'integrity_audit.json',result)
    print(f'PASS: {checks} hashes, {lineage_cases} cross-fitting traces, {probes} exact head probes, {repeated} CSV repetitions',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output.resolve())
