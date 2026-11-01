"""Follow-up fixed quality ablation: include clean-source training to address domain false alarms."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import sys
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score,roc_auc_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'solution/tools'),str(ROOT/'solution/src')]
import research_reliability as r
from possession.tracking_reliability import inject_training_ghosts,quality_features,compose_candidates
from possession.inference import read_tracking_npz
from possession.inputs import file_digest
from possession.robustness import absorb_short_opposition
from possession.event_decoder import decode_events

SOURCE=ROOT/'outputs/reliability_research/run_20261101_v2'
OUT=ROOT/'outputs/reliability_research/quality_clean_domain_20261101'


def prepare(match):
    cfg,sc,rc,f=r.setup();path=OUT/'training'/f'{match}.npz';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():return
    raw=read_tracking_npz(ROOT/'data/external/sportec'/match/'prepared/inputs.npz')
    noisy,ghost=inject_training_ghosts(raw,rc,829);x,_=quality_features(noisy)
    valid=(raw['period']>0)&np.isfinite(noisy['ball_xy']).all(1)
    rows=np.unique(np.r_[np.flatnonzero(valid)[::25],np.flatnonzero(valid&ghost)[::5]])
    y=ghost[rows].astype(int);full=np.bincount(ghost[valid].astype(int),minlength=2);sample=np.bincount(y,minlength=2)
    w=(full/sample)[y];w/=w.sum()
    original=SOURCE/'features'/match/'quality_training.npz'
    index=r.n.read(original.parent/'index.json')
    if file_digest(original)!=index['files'][original.name]:raise ValueError('Original quality training drift')
    with np.load(original,allow_pickle=False) as z:
        np.savez_compressed(path,x=np.concatenate([z['x'],x[rows]]),y=np.r_[z['y'],y],w=np.r_[z['w'],w])


def fold(held):
    cfg,sc,rc,f=r.setup();folder=OUT/'outer'/held;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'report.json').exists():return
    training=[m for m in cfg['matches'] if m!=held];xs=[];ys=[];ws=[]
    for match in training:
        with np.load(OUT/'training'/f'{match}.npz',allow_pickle=False) as z:xs.append(z['x']);ys.append(z['y']);ws.append(z['w'])
    x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws);w*=len(w)/w.sum()
    with threadpool_limits(limits=2):
        model=HistGradientBoostingClassifier(**rc['quality_model']).fit(x,y,sample_weight=w)
        joblib.dump({'model':model,'training_matches':training,'held_out':held},folder/'model.joblib')
        del x,y,w,xs,ys,ws
        truth=r.n.dead.targets(held);results=[];quality=[];files={}
        base=r.n.read(r.s.SOURCE/'selection.json')['folds'][held]['selected']['profile']
        for condition,seed in r.n.cases(cfg,outer=True):
            stem=f'{condition}_{seed}';feature=SOURCE/'features'/held/(stem+'.npy')
            index=r.n.read(feature.parent/'index.json')
            if file_digest(feature)!=index['files'][feature.name]:raise ValueError('Feature drift')
            xq=np.load(feature,allow_pickle=False);bad=r.probabilities(model,xq)[:,1];bad[xq[:,0]!=1]=0
            with np.load(r.s.SOURCE/'outer'/held/f'{held}_{stem}_probabilities.npz',allow_pickle=False) as z:h=z['home'];d=z['dead']
            context=r.n.read(r.s.SOURCE/'prepared'/held/'index.json')['cases'][stem]['context']
            with np.load(context,allow_pickle=False) as z:period=z['period']
            events=np.load(r.OLD/'outer'/held/(stem+'_events.npy'),allow_pickle=False)
            hh,qq=compose_candidates(h,events,h,events,xq[:,0]!=1,bad)['quality']
            prediction=decode_events(hh,d,period,qq,base,4,sc,f[7]['epsilon'])['labels']
            path=folder/(stem+'.npy');np.save(path,prediction,allow_pickle=False);files[path.name]=file_digest(path)
            references={'provider':truth,**{f'absorbed_{s}s':absorb_short_opposition(truth,period,s)[0] for s in (1,3,5)}}
            for key,labels in references.items():results.append({'match':held,'condition':condition,'seed':seed,'label_set':key,**r.score(labels,prediction,period)})
            visible=(period>0)&(xq[:,0]==1);ghost=np.load(feature.parent/(stem+'_ghost.npy'),allow_pickle=False)[visible];p=bad[visible]
            quality.append({'condition':condition,'seed':seed,'average_precision':float(average_precision_score(ghost,p)) if ghost.any() else None,
                'auroc':float(roc_auc_score(ghost,p)) if ghost.any() else None,'prevalence':float(ghost.mean()),
                'recall_at_half':float((p[ghost]>=.5).mean()) if ghost.any() else None,
                'false_positive_rate_at_half':float((p[~ghost]>=.5).mean()),'brier':float(np.mean((p-ghost)**2))})
        r.save(folder/'report.json',{'held_out':held,'training_matches':training,'results':results,'quality':quality,
            'files':files,'model_sha256':file_digest(folder/'model.joblib')})
        print(held+': clean-domain quality follow-up complete',flush=True)


def main():
    cfg,sc,rc,f=r.setup();OUT.mkdir(parents=True,exist_ok=True)
    protocol={'reason':'Adaptive response to high clean false alarms in v2; no new independent validation claim',
              'additional_training_seed':829,'regime_mass':'equal clean/moderate/severe',
              'source_sha256':{str(Path(__file__).relative_to(ROOT)):file_digest(Path(__file__))},
              'foundation_protocol_sha256':file_digest(SOURCE/'protocol.json')}
    if (OUT/'protocol.json').exists() and r.n.read(OUT/'protocol.json')!=protocol:raise ValueError('Protocol changed')
    r.save(OUT/'protocol.json',protocol)
    for match in cfg['matches']:prepare(match)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(fold,m) for m in cfg['matches']]):future.result()
    reports=[r.n.read(OUT/'outer'/m/'report.json') for m in cfg['matches']]
    rows=[row for report in reports for row in report['results']];reference=r.n.read(SOURCE/'validation_report.json')['results'];summary=[]
    for condition in cfg['outer_conditions']:
        for label in ('provider','absorbed_1s','absorbed_3s','absorbed_5s'):
            scores=[];delta=[]
            for match in cfg['matches']:
                group=[row for row in rows if row['match']==match and row['condition']==condition and row['label_set']==label]
                old=[row for row in reference if row['match']==match and row['condition']==condition and row['label_set']==label and row['model']=='event4']
                mean=np.mean([row['spell_f1_mean'] for row in group]);scores.append(mean);delta.append(mean-np.mean([row['spell_f1_mean'] for row in old]))
            delta=np.array(delta);picks=np.random.RandomState(829).randint(0,7,(1000,7));lo,hi=np.percentile(delta[picks].mean(1),[2.5,97.5])
            summary.append({'condition':condition,'label_set':label,'spell_f1':float(np.mean(scores)),'delta_vs_event4':float(delta.mean()),
                            'low':float(lo),'high':float(hi),'improved_matches':int((delta>0).sum())})
    r.save(OUT/'validation_report.json',{'status':'complete','promoted':False,'summary':summary,'results':rows,
           'quality':[dict(match=p['held_out'],**q) for p in reports for q in p['quality']]})
    print('COMPLETE clean-domain quality follow-up',flush=True)


if __name__=='__main__':main()
