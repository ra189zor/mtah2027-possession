"""Reload the clean-domain quality models and reproduce external predictions."""
import json
import sys
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'solution/tools'),str(ROOT/'solution/src')]
import quality_domain_check as q
from possession.inputs import file_digest
from possession.tracking_reliability import compose_candidates
from possession.event_decoder import decode_events


def main():
    cfg,sc,rc,f=q.r.setup();checks=0;replays=0
    protocol=q.r.n.read(q.OUT/'protocol.json')
    for name,digest in protocol['source_sha256'].items():
        if file_digest(ROOT/name)!=digest:raise ValueError('Follow-up source drift')
    if file_digest(q.SOURCE/'protocol.json')!=protocol['foundation_protocol_sha256']:raise ValueError('Foundation protocol drift')
    with threadpool_limits(limits=2):
        for match in cfg['matches']:
            folder=q.OUT/'outer'/match;report=q.r.n.read(folder/'report.json');bundle=joblib.load(folder/'model.joblib')
            if set(bundle['training_matches'])!=set(cfg['matches'])-{match}:raise ValueError('Leaked quality fit')
            if file_digest(folder/'model.joblib')!=report['model_sha256']:raise ValueError('Model drift')
            for name,digest in report['files'].items():
                if file_digest(folder/name)!=digest:raise ValueError('Prediction drift')
                checks+=1
            for condition in ('clean','contextual'):
                stem=condition+'_17';x=np.load(q.SOURCE/'features'/match/(stem+'.npy'),allow_pickle=False)
                bad=q.r.probabilities(bundle['model'],x)[:,1];bad[x[:,0]!=1]=0
                with np.load(q.r.s.SOURCE/'outer'/match/f'{match}_{stem}_probabilities.npz',allow_pickle=False) as z:h=z['home'];d=z['dead']
                context=q.r.n.read(q.r.s.SOURCE/'prepared'/match/'index.json')['cases'][stem]['context']
                with np.load(context,allow_pickle=False) as z:period=z['period']
                events=np.load(q.r.OLD/'outer'/match/(stem+'_events.npy'),allow_pickle=False)
                base=q.r.n.read(q.r.s.SOURCE/'selection.json')['folds'][match]['selected']['profile']
                hh,qq=compose_candidates(h,events,h,events,x[:,0]!=1,bad)['quality']
                pred=decode_events(hh,d,period,qq,base,4,sc,f[7]['epsilon'])['labels']
                if not np.array_equal(pred,np.load(folder/(stem+'.npy'),allow_pickle=False)):raise ValueError('Quality replay differs')
                replays+=1
    result={'status':'passed','prediction_hash_checks':checks,'exact_replays':replays,'excluded_match_models':len(cfg['matches']),
            'report_sha256':file_digest(q.OUT/'validation_report.json')}
    q.r.save(q.OUT/'integrity_audit.json',result);print(json.dumps(result))


if __name__=='__main__':main()
