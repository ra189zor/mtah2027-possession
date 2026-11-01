"""Audit new experiment identities, exclusions, input-only features and exact held-out replay."""
import argparse
import itertools
import json
from pathlib import Path
import sys
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'solution/tools'),str(ROOT/'solution/src')]
import research_reliability as r
from possession.inference import load_bundle,read_tracking_npz
from possession.tracking_reliability import quality_features,compose_candidates
from possession.validation import corrupt
from possession.event_decoder import decode_events
from possession.decoder import decode
from possession.inputs import file_digest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output.resolve();cfg,sc,rc,f=r.setup();protocol=r.n.read(out/'protocol.json')
    checks=0;replays=0;lineages=0;models={}
    for name,digest in protocol['source_sha256'].items():
        if file_digest(ROOT/name)!=digest:raise ValueError('Research source drift')
        checks+=1
    for name,digest in protocol['foundation_source_sha256'].items():
        if file_digest(ROOT/name)!=digest:raise ValueError('Foundation source drift')
        checks+=1
    for excluded in [(m,) for m in cfg['matches']]+list(itertools.combinations(cfg['matches'],2)):
        path=r.quality_path(out,excluded);model=joblib.load(path)
        if set(model['training_matches'])!=set(cfg['matches'])-set(excluded) or model['excluded']!=sorted(excluded):raise ValueError('Quality membership failure')
        models[str(path.relative_to(out))]=file_digest(path)
    for match in cfg['matches']:
        folder=out/'features'/match;index=r.n.read(folder/'index.json')
        for name,digest in index['files'].items():
            if file_digest(folder/name)!=digest:raise ValueError('Feature drift')
            checks+=1
        if index['possession_labels_read']:raise ValueError('Quality features used possession targets')
        report=r.n.read(out/'outer'/match/'report.json');model_path=out/'outer'/match/'specialists.joblib'
        if file_digest(model_path)!=report['model_sha256']:raise ValueError('Specialist drift')
        models[str(model_path.relative_to(out))]=file_digest(model_path);bundle=joblib.load(model_path)
        if set(bundle['training_matches'])!=set(cfg['matches'])-{match}:raise ValueError('Specialist membership failure')
        for trace in report['lineage']:
            exclusions={match,trace['training_fixture']}
            if set(trace['excluded'])!=exclusions or set(trace['quality_excluded'])!=exclusions or set(trace['upstream_training_matches'])!=set(cfg['matches'])-exclusions:
                raise ValueError('Cross-fitting leakage')
            if file_digest(Path(trace['probability_file']))!=trace['sha256']:raise ValueError('Training upstream drift')
            lineages+=1
        for name,digest in report['files'].items():
            if file_digest(out/'outer'/match/name)!=digest:raise ValueError('Prediction drift')
            checks+=1
        raw=read_tracking_npz(ROOT/'data/external/sportec'/match/'prepared/inputs.npz')
        with threadpool_limits(limits=rc['threads']):
            for condition in ('clean','moderate'):
                stem=condition+'_17';noisy=corrupt(raw,f[2]['regimes'][condition],17)
                rebuilt,names=quality_features(noisy);saved=np.load(folder/(stem+'.npy'),allow_pickle=False)
                if names!=index['names'] or not np.array_equal(rebuilt,saved,equal_nan=True):raise ValueError('Raw tracking feature replay differs')
                with np.load(r.s.SOURCE/'outer'/match/f'{match}_{stem}_probabilities.npz',allow_pickle=False) as z:h=z['home'];d=z['dead']
                q=np.load(r.OLD/'outer'/match/(stem+'_events.npy'),allow_pickle=False)
                x,period,missing,bad=r.evidence(out,match,condition,17,h,d,r.quality_model(out,[match]))
                gap=r.probabilities(bundle['gap'],x[:,bundle['gap_columns']])[:,1];boundary=r.probabilities(bundle['boundary'],x[:,bundle['boundary_columns']])
                candidates=compose_candidates(h,q,gap,boundary,missing,bad,rc['blend'])
                base=r.n.read(r.s.SOURCE/'selection.json')['folds'][match]['selected']['profile']
                predictions={name:decode_events(hh,d,period,qq,base,4,sc,f[7]['epsilon'])['labels'] for name,(hh,qq) in candidates.items()}
                predictions['baseline']=decode(h,d,period,base,f[7]['epsilon'])['labels']
                for name,pred in predictions.items():
                    expected=np.load(out/'outer'/match/(stem+'_'+name+'.npy'),allow_pickle=False)
                    if not np.array_equal(pred,expected):raise ValueError('Reloaded prediction differs')
                    replays+=1
        print(match+': exclusions, feature hashes and exact replays passed',flush=True)
    # Existing deployable packages retain their original runtime/weight integrity.
    for name in ('possession-v1','heldout-J03WN1-v1'):load_bundle(ROOT/'models'/name)
    result={'status':'passed','hash_checks':checks,'cross_fit_traces':lineages,'exact_prediction_replays':replays,
            'model_hashes':models,'production_bundle_integrity':True,
            'validation_report_sha256':file_digest(out/'validation_report.json')}
    r.save(out/'integrity_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='model_hashes'}),flush=True)


if __name__=='__main__':main()
