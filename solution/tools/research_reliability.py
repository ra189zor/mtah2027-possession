"""Seven-fold external reliability/gap/boundary experiment; frozen production is never modified."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import itertools
import json
from pathlib import Path
import sys
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score,roc_auc_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'solution/src'),str(ROOT/'solution/tools')]
import train_switch_events as s
from possession.tracking_reliability import quality_features,inject_training_ghosts,combined_features,compose_candidates
from possession.inference import read_tracking_npz
from possession.validation import corrupt
from possession.robustness import contextual_corrupt,absorb_short_opposition
from possession.inputs import array_digest,file_digest
from possession.switch_events import extract_event_features,event_training_rows
from possession.dead_model import sample_weights,training_rows
from possession.event_decoder import decode_events
from possession.decoder import decode
from run_decoder import score
from run_oracles import paired_draw_scores

n=s.n
OLD=ROOT/'outputs/switch_events/run_20261101'
CONFIG=ROOT/'solution/configs/reliability_research.json'
SOURCES=[CONFIG,Path(__file__),ROOT/'solution/src/possession/tracking_reliability.py']


def setup():
    cfg,sc,oldhash,f=s.setup()
    return cfg,sc,n.read(CONFIG),f


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);n.write(path,obj)


def probabilities(model,x):
    result=np.empty((len(x),len(model.classes_)),dtype=np.float32)
    for a in range(0,len(x),16384):result[a:a+16384]=model.predict_proba(x[a:a+16384])
    return result


def prepare(job):
    match,out=job;out=Path(out);cfg,sc,rc,f=setup();folder=out/'features'/match;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'index.json').exists():return
    raw=read_tracking_npz(ROOT/'data/external/sportec'/match/'prepared/inputs.npz')
    original=n.read(s.SOURCE/'prepared'/match/'index.json');files={};trainx=[];trainy=[];trainw=[]
    for condition,seed in n.cases(cfg,outer=True):
        if condition=='contextual':noisy,masks,_=contextual_corrupt(raw,f[2]['regimes']['moderate'],cfg['contextual'],seed);ghost=masks['ghost']
        else:noisy=corrupt(raw,f[2]['regimes'][condition],seed);ghost=np.zeros(len(raw['period']),bool)
        fingerprint=array_digest(noisy)
        if fingerprint!=original['cases'][f'{condition}_{seed}']['noisy_input_fingerprint']:raise ValueError('External input fingerprint differs')
        x,names=quality_features(noisy);stem=f'{condition}_{seed}'
        path=folder/(stem+'.npy');np.save(path,x,allow_pickle=False);files[path.name]=file_digest(path)
        path=folder/(stem+'_ghost.npy');np.save(path,ghost,allow_pickle=False);files[path.name]=file_digest(path)
    for regime,seed in zip(('moderate','severe'),rc['training_ghost_seeds']):
        noisy=corrupt(raw,f[2]['regimes'][regime],seed);noisy,ghost=inject_training_ghosts(noisy,rc,seed)
        x,_=quality_features(noisy);valid=(raw['period']>0)&np.isfinite(noisy['ball_xy']).all(1)
        rows=np.unique(np.r_[np.flatnonzero(valid)[::25],np.flatnonzero(valid&ghost)[::5]])
        y=ghost[rows].astype(int);full=np.bincount(ghost[valid].astype(int),minlength=2);sample=np.bincount(y,minlength=2)
        if (sample==0).any():raise ValueError('Quality classes missing')
        w=(full/sample)[y];w/=w.sum()
        trainx.append(x[rows]);trainy.append(y);trainw.append(w)
    path=folder/'quality_training.npz';np.savez_compressed(path,x=np.concatenate(trainx),y=np.concatenate(trainy),w=np.concatenate(trainw));files[path.name]=file_digest(path)
    save(folder/'index.json',{'match':match,'names':names,'files':files,'synthetic_ghost_seeds':rc['training_ghost_seeds'],'possession_labels_read':False})
    print(match+': quality features and independently seeded ghost examples prepared',flush=True)


def quality_path(out,excluded):return out/'quality_models'/('_'.join(sorted(excluded))+'.joblib')


def fit_quality(job):
    excluded,out=job;out=Path(out);cfg,sc,rc,f=setup();path=quality_path(out,excluded)
    if path.exists():return
    training=[m for m in cfg['matches'] if m not in excluded];xs=[];ys=[];ws=[]
    for match in training:
        with np.load(out/'features'/match/'quality_training.npz',allow_pickle=False) as z:
            xs.append(z['x']);ys.append(z['y']);ws.append(z['w'])
    x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws);w*=len(w)/w.sum()
    with threadpool_limits(limits=rc['threads']):model=HistGradientBoostingClassifier(**rc['quality_model']).fit(x,y,sample_weight=w)
    path.parent.mkdir(parents=True,exist_ok=True);joblib.dump({'model':model,'excluded':sorted(excluded),'training_matches':training},path)
    print('Quality fit excludes '+','.join(excluded),flush=True)


def quality_model(out,excluded):
    bundle=joblib.load(quality_path(out,excluded))
    if set(bundle['training_matches'])&set(excluded) or bundle['excluded']!=sorted(excluded):raise ValueError('Quality leakage')
    return bundle['model']


def evidence(out,match,condition,seed,h,d,quality_model):
    index=n.read(s.SOURCE/'prepared'/match/'index.json')
    with np.load(index['cases'][f'{condition}_{seed}']['context'],allow_pickle=False) as z:period=z['period']
    physical,names=s.physical_evidence(OLD,match,condition,seed)
    event,_=extract_event_features(physical,names,h,d,period,setup()[1])
    xq=np.load(out/'features'/match/f'{condition}_{seed}.npy',allow_pickle=False)
    bad=probabilities(quality_model,xq)[:,1];bad[(xq[:,0]!=1)]=0
    x=combined_features(event,xq,bad)
    return x,period,xq[:,0]!=1,bad


def outer(job):
    held,out=job;out=Path(out);cfg,sc,rc,f=setup();folder=out/'outer'/held;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'report.json').exists():return
    training=[m for m in cfg['matches'] if m!=held]
    with threadpool_limits(limits=rc['threads']):
        model_path=folder/'specialists.joblib'
        if model_path.exists():models=joblib.load(model_path)
        else:
            gx=[];gy=[];gw=[];bx=[];by=[];bw=[];lineage=[]
            for match in training:
                qm=quality_model(out,[held,match]);labels=n.dead.targets(match)
                for condition,seed in n.cases(cfg):
                    h,d,trace=s.upstream_training_case(held,match,condition,seed,cfg)
                    x,period,missing,bad=evidence(out,match,condition,seed,h,d,qm)
                    rows=training_rows(labels,period,25,seed);rows=rows[missing[rows]&np.isin(labels[rows],['H','A'])]
                    if len(rows):
                        w=sample_weights(labels,period,rows,'spell',10);w/=w.sum()
                        if condition!='clean':w/=len(cfg['seeds'])
                        gx.append(x[rows]);gy.append((labels[rows]=='H').astype(int));gw.append(w)
                    rows,y,w,_=event_training_rows(labels,period,seed,sc)
                    if condition!='clean':w/=len(cfg['seeds'])
                    bx.append(x[rows]);by.append(y);bw.append(w)
                    lineage.append({**trace,'quality_excluded':sorted([held,match])})
            models={'held_out':held,'training_matches':training,'lineage':lineage}
            for name,xs,ys,ws,params in [('gap',gx,gy,gw,rc['gap_model']),('boundary',bx,by,bw,rc['boundary_model'])]:
                x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws);w*=len(w)/w.sum()
                columns=np.flatnonzero(np.isfinite(x).any(axis=0))
                if not len(columns):raise ValueError('Specialist has no observed training features')
                models[name+'_columns']=columns
                models[name]=HistGradientBoostingClassifier(**params).fit(x[:,columns],y,sample_weight=w)
                models[name+'_rows']=len(y)
                print(held+': '+name+' specialist fitted',flush=True)
                xs.clear();ys.clear();ws.clear();del x,y,w
            joblib.dump(models,model_path)
        if models['training_matches']!=training:raise ValueError('Specialist leakage')
        labels=n.dead.targets(held);qm=quality_model(out,[held]);results=[];quality=[];files={};paired=[]
        oldreport=n.read(OLD/'outer'/held/'report.json')
        base=n.read(s.SOURCE/'selection.json')['folds'][held]['selected']['profile']
        upstream=n.read(s.SOURCE/'outer'/held/'report.json')
        for condition,seed in n.cases(cfg,outer=True):
            stem=f'{condition}_{seed}';path=s.SOURCE/'outer'/held/f'{held}_{stem}_probabilities.npz'
            if file_digest(path)!=upstream['files'][path.name]:raise ValueError('Upstream drift')
            with np.load(path,allow_pickle=False) as z:h=z['home'];d=z['dead']
            path=OLD/'outer'/held/(stem+'_events.npy')
            if file_digest(path)!=oldreport['files'][path.name]:raise ValueError('Event drift')
            q=np.load(path,allow_pickle=False)
            x,period,missing,bad=evidence(out,held,condition,seed,h,d,qm)
            gap=probabilities(models['gap'],x[:,models['gap_columns']])[:,1];boundary=probabilities(models['boundary'],x[:,models['boundary_columns']])
            candidates=compose_candidates(h,q,gap,boundary,missing,bad,rc['blend'])
            predictions={name:decode_events(hh,d,period,qq,base,4,sc,f[7]['epsilon'])['labels'] for name,(hh,qq) in candidates.items()}
            predictions['baseline']=decode(h,d,period,base,f[7]['epsilon'])['labels']
            original_csv=OLD/'outer'/held/(stem+'_event_reward_4.csv')
            original=n.pm.read_label_csv(str(original_csv))['label'].to_numpy()
            if not np.array_equal(original,predictions['event4']):raise ValueError('Event4 reference failed exact reproduction')
            refs={'provider':labels,**{f'absorbed_{seconds}s':absorb_short_opposition(labels,period,seconds)[0] for seconds in rc['label_sensitivity_seconds']}}
            for name,prediction in predictions.items():
                path=folder/(stem+'_'+name+'.npy');np.save(path,prediction,allow_pickle=False);files[path.name]=file_digest(path)
                if (prediction[period>0]=='X').any():raise ValueError('Unexpected graded abstention')
                for convention,truth in refs.items():
                    metrics=score(truth,prediction,period)
                    if convention=='provider' and name=='event4':
                        previous=next(r for r in oldreport['results'] if r['condition']==condition and r['seed']==seed and r['label_set']=='provider' and r['model']=='event_reward_4')
                        if metrics['spell_f1_mean']!=previous['spell_f1_mean']:raise ValueError('Reference score drift')
                    alive_missing=missing&np.isin(truth,['H','A'])&(period>0)
                    results.append({'match':held,'condition':condition,'seed':seed,'model':name,'label_set':convention,
                        **metrics,'missing_alive_accuracy':float((prediction[alive_missing]==truth[alive_missing]).mean()) if alive_missing.any() else None,
                        'direct_switches_12':s.direct_switch_stats(truth,prediction)})
            visible=(period>0)&~missing
            ghost=np.load(out/'features'/held/(stem+'_ghost.npy'),allow_pickle=False)[visible];p=bad[visible]
            quality.append({'condition':condition,'seed':seed,'visible_ghost_prevalence':float(ghost.mean()),
                'average_precision':float(average_precision_score(ghost,p)) if ghost.any() else None,
                'auroc':float(roc_auc_score(ghost,p)) if ghost.any() and not ghost.all() else None,
                'brier':float(np.mean((p-ghost)**2)),
                'recall_at_half':float((p[ghost]>=.5).mean()) if ghost.any() else None,
                'false_positive_rate_at_half':float((p[~ghost]>=.5).mean())})
            if condition in ('moderate','contextual'):
                for convention in ('provider','absorbed_3s'):
                    gt=n.pm.label_codes(refs[convention]);spells=n.pm.possession_spells(gt,n.pm.valid_runs(gt))
                    picks=np.random.RandomState(811).randint(0,len(spells),(200,len(spells)))
                    delta=paired_draw_scores(gt,n.pm.label_codes(predictions['combined']),picks)-paired_draw_scores(gt,n.pm.label_codes(predictions['event4']),picks)
                    paired.append({'condition':condition,'seed':seed,'label_set':convention,'draws':delta.tolist()})
            print(held+' '+stem+': all ablations and four conventions evaluated',flush=True)
        save(folder/'report.json',{'held_out':held,'training_matches':training,'lineage':models['lineage'],
            'results':results,'quality':quality,'paired':paired,'files':files,'model_sha256':file_digest(model_path)})


def aggregate(out):
    cfg,sc,rc,f=setup();reports=[n.read(out/'outer'/m/'report.json') for m in cfg['matches']]
    rows=[r for report in reports for r in report['results']];summary=[]
    for condition,convention,model in itertools.product(cfg['outer_conditions'],['provider','absorbed_1s','absorbed_3s','absorbed_5s'],['baseline','event4','quality','gap','boundary','combined']):
        means=[];deltas=[];gates=[]
        for match in cfg['matches']:
            group=[r for r in rows if r['match']==match and r['condition']==condition and r['label_set']==convention and r['model']==model]
            reference=[r for r in rows if r['match']==match and r['condition']==condition and r['label_set']==convention and r['model']=='event4']
            mean=float(np.mean([r['spell_f1_mean'] for r in group]));means.append(mean)
            deltas.append(mean-np.mean([r['spell_f1_mean'] for r in reference]))
            gates.append(abs(np.mean([r['n_pred_spells']/r['n_gt_spells']-1 for r in group]))<=.2)
        delta=np.array(deltas);picks=np.random.RandomState(811).randint(0,len(delta),(rc['bootstrap_draws'],len(delta)))
        lo,hi=np.percentile(delta[picks].mean(1),[2.5,97.5])
        summary.append({'condition':condition,'label_set':convention,'model':model,'spell_f1':float(np.mean(means)),
            'delta_vs_event4':float(delta.mean()),'descriptive_match_low':float(lo),'descriptive_match_high':float(hi),
            'improved_matches':int((delta>0).sum()),'count_gate_passes':int(sum(gates))})
    if len(rows)!=1680:raise ValueError('Incomplete validation matrix')
    conditional=[]
    for report in reports:
        for condition,convention in itertools.product(['moderate','contextual'],['provider','absorbed_3s']):
            draws=np.mean([r['draws'] for r in report['paired'] if r['condition']==condition and r['label_set']==convention],axis=0)
            lo,hi=np.percentile(draws,[2.5,97.5]);conditional.append({'match':report['held_out'],'condition':condition,'label_set':convention,'low':float(lo),'high':float(hi)})
    save(out/'validation_report.json',{'status':'complete','promoted':False,'summary':summary,'results':rows,
         'quality':[dict(match=r['held_out'],**q) for r in reports for q in r['quality']],
         'combined_conditional_possession_intervals':conditional,
         'limits':'Adaptive repeated seven-fixture research, overlapping training sets; no new independent dataset or challenge evaluation.'})
    print('COMPLETE: '+str(out/'validation_report.json'),flush=True)


def parallel(fn,jobs,workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(fn,job) for job in jobs]):future.result()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);cfg,sc,rc,f=setup()
    protocol={'config':rc,'matches':cfg['matches'],'source_sha256':{str(p.relative_to(ROOT)):file_digest(p) for p in SOURCES},
              'foundation_source_sha256':s.setup()[2]}
    if (out/'protocol.json').exists():
        if n.read(out/'protocol.json')!=protocol:raise ValueError('Research protocol/source changed; use a new directory')
    else:save(out/'protocol.json',protocol)
    parallel(prepare,[(m,str(out)) for m in cfg['matches']],rc['workers'])
    exclusions=[(m,) for m in cfg['matches']]+list(itertools.combinations(cfg['matches'],2))
    parallel(fit_quality,[(e,str(out)) for e in exclusions],rc['workers'])
    parallel(outer,[(m,str(out)) for m in cfg['matches']],rc['workers'])
    aggregate(out)


if __name__=='__main__':main()
