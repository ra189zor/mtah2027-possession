"""Build final all-external models or export a held-out bundle for pipeline verification."""
import argparse
from datetime import datetime,timezone
from importlib.metadata import version
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'solution/tools'),str(ROOT/'solution/src')]
import train_switch_events as s
n=s.n
from possession.inference import runtime_hashes,settings_digest,PACKAGES,load_bundle
from possession.inputs import file_digest
from possession.switch_events import extract_event_features,event_training_rows

NESTED=s.SOURCE
SWITCH=ROOT/'outputs/switch_events/run_20261101'
BUILD_SOURCES=('solution/configs/deployment.json','solution/tools/build_inference_bundle.py')


def setup():
    cfg,event_config,upstream,f=s.setup();deployment=n.read(ROOT/BUILD_SOURCES[0])
    final_report=n.read(SWITCH/'validation_report.json');audit=n.read(SWITCH/'integrity_audit.json')
    if final_report['status']!='passed' or audit['status']!='passed' or final_report['source_sha256']!=upstream:
        raise ValueError('Completed audited switch-event foundation required')
    settings={'deployment':deployment,'features':f[3],
        'dead_features':{k:f[1][k] for k in ('windows_s','long_motion_lag_frames','walking_speed_mps','player_boundary_radius_m','cluster_radius_m')},
        'owner_features':{k:f[5][k] for k in ('anchor_confirm_s','control_distance_m','owner_margin_m','geometry_logit_scale_m','anchor_probability')},
        'switch':event_config}
    hashes=upstream|{name:file_digest(ROOT/name) for name in BUILD_SOURCES}
    return cfg,settings,hashes,f


def feature_names(match,f):
    folder=n.owner.cache_folder(match,f[6]);path=folder/'clean_17_context.npz';manifest=n.read(folder/'clean_17.json')
    if file_digest(path)!=manifest['files'][path.name]:raise ValueError('Owner schema artifact changed')
    with np.load(path,allow_pickle=False) as z:owner=z['names'].tolist()
    return owner[:275],owner


def verify_training_caches(matches):
    files={}
    for match in matches:
        index=n.read(NESTED/'prepared'/match/'index.json')
        for key in ('training_x','training_targets'):
            path=Path(index[key]);digest=index['files'][str(path)]
            if file_digest(path)!=digest:raise ValueError('Final head training cache changed')
            files[str(path.relative_to(ROOT))]=digest
    return files


def fit_final_switch(cfg,settings):
    xs=[];ys=[];ws=[];lineage=[];sample_audits=[];names=None
    for match in cfg['matches']:
        labels=n.dead.targets(match);index=n.read(NESTED/'prepared'/match/'index.json');fold=n.read(NESTED/'outer'/match/'report.json')
        expected=[m for m in cfg['matches'] if m!=match]
        if fold['training_matches']!=expected:raise ValueError('Final event cross-fitting membership mismatch')
        for condition,seed in n.cases(cfg):
            path=NESTED/'outer'/match/f'{match}_{condition}_{seed}_probabilities.npz';digest=fold['files'][path.name]
            if file_digest(path)!=digest:raise ValueError('Final event OOF probability changed')
            with np.load(path,allow_pickle=False) as z:h=z['home'];d=z['dead']
            physical,physical_names=s.physical_evidence(SWITCH,match,condition,seed)
            with np.load(index['cases'][f'{condition}_{seed}']['context'],allow_pickle=False) as z:period=z['period']
            x,names=extract_event_features(physical,physical_names,h,d,period,settings['switch'])
            rows,y,w,audit=event_training_rows(labels,period,seed,settings['switch'])
            if condition!='clean':w/=len(cfg['seeds'])
            xs.append(x[rows]);ys.append(y);ws.append(w)
            lineage.append({'training_fixture':match,'excluded':[match],'upstream_training_matches':expected,
                'probability_file':str(path.relative_to(ROOT)),'sha256':digest})
            sample_audits.append({'match_id':match,'condition':condition,'seed':seed,**audit})
    x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws);w*=len(w)/w.sum();del xs,ys,ws
    started=time.perf_counter();model=HistGradientBoostingClassifier(**settings['switch']['model']).fit(x,y,sample_weight=w)
    info={'rows':len(y),'features':x.shape[1],'fit_seconds':time.perf_counter()-started}
    print(f'Final switch head fitted: {info}',flush=True)
    return model,names,lineage,sample_audits,info


def build(output,held_out,resume):
    cfg,settings,source_hashes,f=setup();output=output.resolve();runtime=runtime_hashes()
    kind='held_out_verification' if held_out else 'final_all_external_matches'
    if held_out and held_out not in cfg['matches']:raise ValueError('Unknown held-out fixture')
    if held_out:
        settings['deployment']['decoder']=n.read(NESTED/'selection.json')['folds'][held_out]['selected']['profile']
    digest=settings_digest(settings)
    build_key={'source_sha256':source_hashes,'runtime_sha256':runtime,'settings_sha256':digest,'kind':kind,'held_out':held_out}
    if output.exists():
        if not resume:raise FileExistsError('Bundle directory exists; use --resume for the identical build or choose a new directory')
        if n.read(output/'build.json')['build_key']!=build_key:raise ValueError('Build sources/settings changed; use a new bundle directory')
        if (output/'manifest.json').exists():
            load_bundle(output);print(f'PASS: verified completed bundle {output}',flush=True);return
    else:
        output.mkdir(parents=True,exist_ok=False)
        n.write(output/'build.json',{'build_key':build_key,'created_utc':datetime.now(timezone.utc).isoformat(),'status':'building'})
    dead_names,owner_names=feature_names(cfg['matches'][0],f)
    training=[m for m in cfg['matches'] if m!=held_out] if held_out else cfg['matches']
    cache_hashes={};fit_info={};sample_audits=[]
    with threadpool_limits(limits=settings['deployment']['threads']):
        if held_out:
            base_folder=NESTED/'outer'/held_out;event_folder=SWITCH/'outer'/held_out
            base_report=n.read(base_folder/'report.json');event_report=n.read(event_folder/'report.json')
            if file_digest(base_folder/'models.joblib')!=base_report['files']['models.joblib'] or file_digest(event_folder/'switch_model.joblib')!=event_report['model_sha256']:
                raise ValueError('Held-out fitted model hash mismatch')
            base=joblib.load(base_folder/'models.joblib');event=joblib.load(event_folder/'switch_model.joblib')
            if base['training_matches']!=training or event['training_matches']!=training:raise ValueError('Held-out model membership mismatch')
            heads={'dead':base['dead'],'owner':base['owner'],'switch':event['model']}
            event_names=event['feature_names'];lineage=event['lineage'];fit_info=event['fit_info'];sample_audits=event['sampling_audits']
        else:
            cache_hashes=verify_training_caches(training)
            checkpoint=output/'base_heads.joblib';progress=n.read(output/'build.json')
            if checkpoint.exists():
                if progress.get('base_heads_sha256')!=file_digest(checkpoint):raise ValueError('Interrupted base-head checkpoint hash mismatch')
                base=joblib.load(checkpoint);fit_info=progress['base_fit_info']
                if base['training_matches']!=training:raise ValueError('Final checkpoint membership mismatch')
            else:
                started=time.perf_counter();base,fit_info=n.fit_heads(training,NESTED,cfg,f[1],f[5])
                fit_info['fit_seconds']=time.perf_counter()-started
                joblib.dump(base,checkpoint);progress|={'base_heads_sha256':file_digest(checkpoint),'base_fit_info':fit_info}
                n.write(output/'build.json',progress);print(f'Final dead/owner heads fitted on seven external fixtures: {fit_info}',flush=True)
            event_head,event_names,lineage,sample_audits,event_info=fit_final_switch(cfg,settings)
            fit_info={'base':fit_info,'switch':event_info}
            heads={'dead':base['dead'],'owner':base['owner'],'switch':event_head}
    weights={'heads':heads,'dead_names':dead_names,'owner_names':owner_names,'event_names':event_names,
        'settings_sha256':digest,'training_matches':training,'event_lineage':lineage,'sampling_audits':sample_audits}
    path=output/'models.joblib';joblib.dump(weights,path)
    manifest={'schema_version':1,'status':'complete','kind':kind,'held_out':held_out,
        'created_utc':datetime.now(timezone.utc).isoformat(),'model_file':path.name,'model_sha256':file_digest(path),
        'runtime_sha256':runtime,'training_source_sha256':source_hashes,'settings':settings,'settings_sha256':digest,
        'python_version':platform.python_version(),'packages':{name:version(name) for name in PACKAGES},
        'requirements_lock_sha256':file_digest(ROOT/'solution/requirements-lock.txt'),
        'training_matches':training,'fit_info':fit_info,'training_cache_sha256':cache_hashes,
        'event_lineage':lineage,'input_schema':{'fields':['players_xy','ball_xy','team_of_slot','period','pitch_m','fps'],
            'fps':25,'coordinates':'corner-origin metres; NaN missing; anonymous home=0/away=1 slots','labels_not_accepted':True},
        'provenance':{'dataset':'pysport/idsse-data','revision':'a715a38dfbaf5f58e431727c2b78d174101a703c',
            'licence':'CC-BY-4.0','source_document':'solution/docs/EXTERNAL_DATA_SOURCES.md',
            'switch_experiment':'outputs/switch_events/run_20261101/validation_report.json',
            'competition_training_data_used':False,'contextual_noise_training_used':False},
        'limitations':'Offline future context. Explicit variant choice; event rewards improve provider labels but regress absorbed-short-spell labels. Final all-seven fits have no new held-out quality score. Model bundle is trusted local joblib, not an untrusted interchange format.'}
    n.write(output/'manifest.json',manifest)
    loaded=load_bundle(output)
    if loaded['manifest']['model_sha256']!=manifest['model_sha256']:raise ValueError('Bundle round-trip failure')
    progress=n.read(output/'build.json');progress['status']='complete';n.write(output/'build.json',progress)
    print(f'PASS: inference bundle {output}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--held-out',help='Export existing disjoint evaluation models instead of fitting final models')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();build(args.output,args.held_out,args.resume)
