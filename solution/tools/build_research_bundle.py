"""Fit all-external research specialists; keep the verified AOM deployment untouched."""
from pathlib import Path
import argparse
import sys
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'solution/tools'),str(ROOT/'solution/src')]
import research_reliability as r
from possession.inputs import file_digest
from possession.switch_events import event_training_rows
from possession.dead_model import training_rows,sample_weights

EXPERIMENT=ROOT/'outputs/reliability_research/run_20261101_v2'


def package(folder,base,models,training,kind):
    folder.mkdir(parents=True,exist_ok=False);joblib.dump(models,folder/'models.joblib')
    manifest=r.n.read(base/'manifest.json');sources=list(manifest['runtime_sha256'])+['solution/research_predict.py','solution/src/possession/tracking_reliability.py']
    r.save(folder/'manifest.json',{'kind':kind,'status':'experimental','training_matches':training,
        'base_manifest_sha256':file_digest(base/'manifest.json'),'model_sha256':file_digest(folder/'models.joblib'),
        'runtime_sha256':{name:file_digest(ROOT/name) for name in sources},'blend':.5,
        'research_protocol_sha256':file_digest(EXPERIMENT/'protocol.json'),
        'build_source_sha256':file_digest(Path(__file__)),'notes':'Research candidates; not an automatic replacement of the AOM package. See the linked protocol for quality-training regimes and limitations.'})


def package_heldout(suffix):
    held=joblib.load(EXPERIMENT/'outer/J03WN1/specialists.joblib');held['quality']=r.quality_model(EXPERIMENT,['J03WN1'])
    package(ROOT/('models/research-heldout-J03WN1-'+suffix),ROOT/'models/heldout-J03WN1-v1',held,held['training_matches'],'experimental_heldout')


def main():
    global EXPERIMENT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,default=EXPERIMENT)
    parser.add_argument('--suffix',choices=['v2','clean'],default='v2')
    stage=parser.add_mutually_exclusive_group();stage.add_argument('--final-only',action='store_true');stage.add_argument('--heldout-only',action='store_true')
    args=parser.parse_args();EXPERIMENT=args.experiment.resolve()
    if args.heldout_only:
        package_heldout(args.suffix);return
    cfg,sc,rc,f=r.setup();gx=[];gy=[];gw=[];bx=[];by=[];bw=[];qx=[];qy=[];qw=[];lineage=[]
    with threadpool_limits(limits=2):
        for match in cfg['matches']:
            quality=r.quality_model(EXPERIMENT,[match]);labels=r.n.dead.targets(match)
            with np.load(EXPERIMENT/'features'/match/'quality_training.npz',allow_pickle=False) as z:qx.append(z['x']);qy.append(z['y']);qw.append(z['w'])
            for condition,seed in r.n.cases(cfg):
                path=r.s.SOURCE/'outer'/match/f'{match}_{condition}_{seed}_probabilities.npz'
                upstream=r.n.read(path.parent/'report.json')
                if match in upstream['training_matches'] or file_digest(path)!=upstream['files'][path.name]:raise ValueError('All-external cross-fitting failure')
                with np.load(path,allow_pickle=False) as z:h=z['home'];d=z['dead']
                x,period,missing,bad=r.evidence(EXPERIMENT,match,condition,seed,h,d,quality)
                rows=training_rows(labels,period,25,seed);rows=rows[missing[rows]&np.isin(labels[rows],['H','A'])]
                if len(rows):
                    w=sample_weights(labels,period,rows,'spell',10);w/=w.sum()
                    if condition!='clean':w/=len(cfg['seeds'])
                    gx.append(x[rows]);gy.append((labels[rows]=='H').astype(int));gw.append(w)
                rows,y,w,_=event_training_rows(labels,period,seed,sc)
                if condition!='clean':w/=len(cfg['seeds'])
                bx.append(x[rows]);by.append(y);bw.append(w)
                lineage.append({'fixture':match,'excluded':[match],'upstream_training_matches':upstream['training_matches'],'probability_sha256':file_digest(path)})
        models={'lineage':lineage}
        for name,xs,ys,ws,params in [('quality',qx,qy,qw,rc['quality_model']),('gap',gx,gy,gw,rc['gap_model']),('boundary',bx,by,bw,rc['boundary_model'])]:
            x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws);w*=len(w)/w.sum()
            columns=np.flatnonzero(np.isfinite(x).any(0));models[name+'_columns']=columns
            if name=='quality' and len(columns)!=x.shape[1]:raise ValueError('Quality feature coverage changed')
            models[name]=HistGradientBoostingClassifier(**params).fit(x[:,columns],y,sample_weight=w)
            xs.clear();ys.clear();ws.clear();del x,y,w
            print('Final research '+name+' head fitted',flush=True)
    package(ROOT/('models/research-specialists-'+args.suffix),ROOT/'models/possession-v1',models,cfg['matches'],'experimental_all_external')
    if not args.final_only:package_heldout(args.suffix)
    print('Research bundles ready; original AOM package unchanged',flush=True)


if __name__=='__main__':main()
