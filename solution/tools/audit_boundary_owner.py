"""Audit held-out membership, sampled-spell coverage and repeated boundary-owner outputs."""
import argparse
from pathlib import Path
import sys
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'solution/tools'))
import train_boundary_owner as b
n=b.n
from possession.inputs import file_digest
from possession.robustness import nested_training_matches
from possession.decoder import decode


def main(out):
    cfg,config,hashes,foundation=b.setup();report=n.read(out/'validation_report.json')
    if report['source_sha256']!=hashes or n.read(out/'protocol.json')['source_sha256']!=hashes:raise ValueError('Boundary source drift')
    checks=0;repeated=0;probes=0;originals=0
    for match in cfg['matches']:
        folder=out/'training'/match;index=n.read(folder/'index.json')
        if index['source_sha256']!=hashes:raise ValueError('Training source drift')
        for name,digest in index['files'].items():
            if file_digest(folder/name)!=digest:raise ValueError('Boundary sample artifact changed')
            checks+=1
        if any(r['provider_spells']!=r['represented_spells'] for r in index['sampling_audits']):raise ValueError('A training spell was omitted')
    with threadpool_limits(limits=2):
        for held in cfg['matches']:
            folder=out/'outer'/held;fold=n.read(folder/'report.json');bundle=joblib.load(folder/'owner.joblib')
            expected=nested_training_matches(cfg['matches'],[held])
            if bundle['training_matches']!=expected or fold['training_matches']!=expected:raise ValueError('Leaked boundary owner fit')
            if bundle['source_sha256']!=hashes or file_digest(folder/'owner.joblib')!=fold['model_sha256']:raise ValueError('Boundary model drift')
            checks+=1
            prepared=n.read(b.SOURCE/'prepared'/held/'index.json')
            base=n.read(b.SOURCE/'selection.json')['folds'][held]['selected']['profile']
            old=n.read(b.SOURCE/'evaluation'/held/'report.json')
            for row in fold['results']:
                if row['model']!='current':continue
                previous=next(r for r in old['results'] if all(r[k]==row[k] for k in ('condition','seed','label_set')) and r['model']=='nested_decoder')
                if any(row[k]!=previous[k] for k in ('spell_f1_mean','n_pred_spells','macro_f1','confusion')):raise ValueError('Original score changed')
                originals+=1
            case=prepared['cases']['moderate_17']
            x=np.load(case['matrix'],mmap_mode='r',allow_pickle=False)[::101]
            predicted=bundle['model'].predict_proba(np.asarray(x))[:,1].astype(np.float32)
            stored=np.load(folder/'moderate_17_new_home.npy',allow_pickle=False)[::101]
            if not np.array_equal(predicted,stored):raise ValueError('Held-out probability reload probe differs')
            probes+=1
            for condition,seed in n.cases(cfg,outer=True):
                probability_path=b.SOURCE/'outer'/held/f'{held}_{condition}_{seed}_probabilities.npz'
                if file_digest(probability_path)!=fold['probability_sha256'][str(probability_path)]:raise ValueError('Original head evidence changed')
                checks+=1
                with np.load(probability_path,allow_pickle=False) as z:h=z['home'];d=z['dead']
                new=np.load(folder/f'{condition}_{seed}_new_home.npy',allow_pickle=False)
                with np.load(prepared['cases'][f'{condition}_{seed}']['context'],allow_pickle=False) as z:period=z['period']
                for name,strength in [('blend_half',.5),('boundary_head',1.)]:
                    probability=(1-strength)*h+strength*new
                    labels=decode(probability,d,period,base,foundation[7]['epsilon'])['labels']
                    filename=f'{condition}_{seed}_{name}.csv';csv_path=folder/filename
                    table=n.pm.read_label_csv(str(csv_path));error=n.pm.validate_submission(table,len(labels))
                    if error:raise ValueError(error)
                    if not np.array_equal(labels,table['label'].to_numpy()):raise ValueError('Repeated boundary labels differ')
                    if (labels[period>0]=='X').any():raise ValueError('Playing-period unknown')
                    repeated+=1
            for name,digest in fold['files'].items():
                if file_digest(folder/name)!=digest:raise ValueError('Boundary output hash changed')
                checks+=1
            print(f'{held}: held-out boundary fit/probabilities and twenty CSV repetitions pass',flush=True)
    result={'status':'passed','source_sha256':hashes,'audit_source_sha256':file_digest(Path(__file__)),
            'artifact_hash_checks':checks,'disjoint_owner_fits':7,'held_out_exact_probability_probes':probes,
            'external_csvs_validated_and_repeated':repeated,'original_score_rows_exact':originals,
            'training_spells_all_represented':True,'competition_data_used':False}
    n.write(out/'integrity_audit.json',result)
    print(f'PASS: {checks} hash checks, {probes} exact held-out head probes, {repeated} CSV repetitions, {originals} original score rows',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output.resolve())
