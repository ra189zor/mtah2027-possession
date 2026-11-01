"""Fixed external-only isolated-spike ablation; no production changes or tuning."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'solution/src'),str(ROOT/'solution/tools')]
from possession.inference import load_bundle,read_tracking_npz,predict_tracking
from possession.inputs import array_digest,file_digest
from possession.robustness import absorb_short_opposition
from run_decoder import score
from train_dead_model import targets


def clean(inputs):
    result={k:v.copy() for k,v in inputs.items()}
    ball=inputs['ball_xy'];period=inputs['period'];fps=float(inputs['fps'])
    left,centre,right=ball[:-2],ball[1:-1],ball[2:]
    # A single implausible excursion with plausible neighbour-to-neighbour travel.
    valid=np.isfinite(left).all(1)&np.isfinite(centre).all(1)&np.isfinite(right).all(1)
    valid&=(period[:-2]==period[1:-1])&(period[1:-1]==period[2:])&(period[1:-1]>0)
    spike=valid&(np.linalg.norm(centre-left,axis=1)*fps>45)&(np.linalg.norm(right-centre,axis=1)*fps>45)
    spike&=np.linalg.norm(right-left,axis=1)*fps/2<20
    rows=np.flatnonzero(spike)+1
    result['ball_xy'][rows]=(ball[rows-1]+ball[rows+1])/2
    return result,len(rows)


def main():
    out=ROOT/'outputs/ball_cleaning/pilot_20261101';out.mkdir(parents=True,exist_ok=False)
    bundle=load_bundle(ROOT/'models/heldout-J03WN1-v1')
    assert 'J03WN1' not in bundle['manifest']['training_matches']
    protocol={'match':'J03WN1','seed':17,'conditions':['clean','moderate','severe','contextual'],
              'variant':'event4','excursion_speed_m_s':45,'neighbour_speed_m_s':20,
              'scope':'Adaptive single-match screening on previously examined external data; not independent validation. No threshold search. No competition inputs.',
              'script_sha256':file_digest(Path(__file__)), 'model_sha256':bundle['manifest']['model_sha256']}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    truth=targets('J03WN1');results=[]
    for condition in protocol['conditions']:
        inputs=read_tracking_npz(ROOT/'outputs/final_verification/run_20261101/clean_source/inputs'/f'{condition}.npz')
        before=array_digest(inputs);cleaned,count=clean(inputs)
        assert np.array_equal(np.isnan(cleaned['ball_xy']),np.isnan(inputs['ball_xy']))
        alternate,_=absorb_short_opposition(truth,inputs['period'],3)
        row={'condition':condition,'changed_frames':count,'input_sha256':before,'scores':{}}
        for name,data in [('original',inputs),('cleaned',cleaned)]:
            prediction=predict_tracking(bundle,data,'event4')['labels']
            row['scores'][name]={key:score(labels,prediction,inputs['period']) for key,labels in [('provider',truth),('absorbed',alternate)]}
        assert array_digest(inputs)==before
        results.append(row)
        (out/'report.json').write_text(json.dumps({'protocol':protocol,'results':results,'promoted':False},indent=2))
        print(condition+': external comparison complete',flush=True)


if __name__=='__main__':main()
