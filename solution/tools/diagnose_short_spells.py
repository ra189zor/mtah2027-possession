"""External-only diagnostics; oracle labels never form deployable evidence."""
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'solution/tools'))
import run_nested_validation as n
from possession.decoder import decode
from run_decoder import score


def main():
    source=ROOT/'outputs/nested_validation/run_20261030'
    cfg,*_=n.setup();selection=n.read(source/'selection.json')
    report={'purpose':'Failure diagnosis on already opened external fixtures; not candidate selection or deployable oracle evidence.','results':[]}
    for match in cfg['matches']:
        y=n.dead.targets(match);gt=n.pm.label_codes(y);runs=n.pm.valid_runs(gt)
        spells=n.pm.possession_spells(gt,runs)
        index=n.read(source/'prepared'/match/'index.json')
        for condition in ('clean','moderate'):
            seed=17;case=index['cases'][f'{condition}_{seed}']
            with np.load(case['context'],allow_pickle=False) as z:period=z['period']
            with np.load(source/'outer'/match/f'{match}_{condition}_{seed}_probabilities.npz',allow_pickle=False) as z:
                h=z['home'];d=z['dead']
            baseline=n.pm.read_label_csv(str(source/'evaluation'/match/f'{match}_{condition}_{seed}_decoded.csv'))['label'].to_numpy()
            profile=selection['folds'][match]['selected']['profile']
            alive=np.isin(y,['H','A']);oracle_period=np.where(alive,period,0)
            owner=decode(h,np.zeros(len(h),dtype=np.float32),oracle_period,profile)['labels']
            true_dead=np.where(y=='D','D',owner)
            true_home=np.where(y=='H',1.-1e-6,np.where(y=='A',1e-6,h)).astype(np.float32)
            team_oracle=decode(true_home,d,period,profile)['labels']
            flags=n.pm.matched_gt_flags(n.pm.spell_overlaps(spells,n.pm.possession_spells(n.pm.label_codes(baseline),runs)),len(spells),.5)
            bins=[]
            for low,high in [(0,1),(1,3),(3,10),(10,float('inf'))]:
                ids=[i for i,(a,b,k) in enumerate(spells) if low<=(b-a)/25<high]
                if not ids:continue
                support=[];confidence=[]
                for i in ids:
                    a,b,k=spells[i];expected=y[a]=='H'
                    support.append(float(((h[a:b]>=.5)==expected).mean()))
                    confidence.append(float(np.abs(h[a:b]-.5).mean()))
                bins.append({'duration_low_s':low,'duration_high_s':high if np.isfinite(high) else None,
                    'n':len(ids),'matched_iou50':int(flags[ids].sum()),'mean_head_correct_fraction':float(np.mean(support)),
                    'mean_head_distance_from_half':float(np.mean(confidence))})
            predictions={'current':baseline,'true_dead_predicted_owner':true_dead,
                         'true_owner_evidence_predicted_dead':team_oracle,'both_true':y}
            scores={name:score(y,pred,period) for name,pred in predictions.items()}
            report['results'].append({'match_id':match,'condition':condition,'seed':seed,'duration_bins':bins,'scores':scores})
            print(match,condition,{name:(round(r['spell_f1_mean'],4),r['n_pred_spells']) for name,r in scores.items()},flush=True)
    destination=ROOT/'outputs/short_possession';destination.mkdir(exist_ok=True)
    n.write(destination/'diagnosis_20261030.json',report)


if __name__=='__main__':main()
