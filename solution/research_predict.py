"""Run explicitly selected experimental specialists alongside the frozen base bundle."""
import argparse
import json
from pathlib import Path
import sys
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'solution/src'))
from possession.inference import load_bundle,read_tracking_npz,predict_tracking,write_prediction,output_paths,probability
from possession.inputs import file_digest,read_challenge
from possession.features import extract_features
from possession.dead_features import extract_dead_features
from possession.gap_ownership import extract_owner_features
from possession.switch_events import physical_columns,extract_event_features
from possession.tracking_reliability import quality_features,combined_features,compose_candidates
from possession.event_decoder import decode_events


def load_research(path,base):
    path=path.resolve();manifest=json.loads((path/'manifest.json').read_text())
    if manifest['base_manifest_sha256']!=file_digest(base['folder']/'manifest.json'):raise ValueError('Wrong base bundle')
    if manifest['model_sha256']!=file_digest(path/'models.joblib'):raise ValueError('Research model hash differs')
    for name,digest in manifest['runtime_sha256'].items():
        if file_digest(ROOT/name)!=digest:raise ValueError('Research runtime differs: '+name)
    return {'folder':path,'manifest':manifest,'models':joblib.load(path/'models.joblib')}


def predict(base,bundle,inputs,variant):
    result=predict_tracking(base,inputs,'event4');settings=base['manifest']['settings'];models=bundle['models']
    with threadpool_limits(limits=2):
        basic=extract_features(inputs,settings['features'])
        dead=extract_dead_features(inputs,settings['features'],settings['dead_features'],basic)
        owner=extract_owner_features(basic,dead['values'],dead['names'],inputs['period'],settings['owner_features'])
        physical,names=physical_columns(owner['values'],owner['names'])
        h=result['home_probability'];d=result['dead_probability'];q=result['event_probability'];period=inputs['period']
        event,_=extract_event_features(physical,names,h,d,period,settings['switch'])
        xq,_=quality_features(inputs);bad=probability(models['quality'],xq)[:,1];bad[xq[:,0]!=1]=0
        x=combined_features(event,xq,bad)
        gap=probability(models['gap'],x[:,models['gap_columns']])[:,1]
        boundary=probability(models['boundary'],x[:,models['boundary_columns']])
        hh,qq=compose_candidates(h,q,gap,boundary,xq[:,0]!=1,bad,bundle['manifest']['blend'])[variant]
        deployment=settings['deployment'];decoded=decode_events(hh,d,period,qq,deployment['decoder'],4,settings['switch'],deployment['epsilon'])
        result.update(labels=decoded['labels'],home_probability=hh,event_probability=qq,path_cost=decoded['path_cost'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-bundle',type=Path,required=True);parser.add_argument('--research-bundle',type=Path,required=True)
    source=parser.add_mutually_exclusive_group(required=True);source.add_argument('--input',type=Path);source.add_argument('--challenge',action='store_true')
    parser.add_argument('--variant',choices=['gap','boundary','combined'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();csv,sidecar=output_paths(args.output)
    if csv.exists() or sidecar.exists():parser.error('Output exists; choose a new output path')
    base=load_bundle(args.base_bundle);bundle=load_research(args.research_bundle,base)
    if args.input:inputs=read_tracking_npz(args.input);description={'kind':'tracking_npz','path':str(args.input.resolve())}
    else:
        sys.path.insert(0,str(ROOT/'more-than-a-hack-2027-main repo'));import games
        paths=games.asset_paths();inputs,_=read_challenge(paths['atd_meta'],paths['atd_raw'],games.source_frames(),games.GAMES[games.GAME]['pitch'],games.half_video_blocks(games.GAME))
        description={'kind':'local_tactical_tracking','metadata':paths['atd_meta'],'raw_tracking':paths['atd_raw'],'video_read':False,'smart_tags_read':False}
    description.update(experimental=True,base_manifest_sha256=bundle['manifest']['base_manifest_sha256'])
    result=predict(base,bundle,inputs,args.variant)
    write_prediction(csv,result,bundle,'research_'+args.variant,description,inputs['period'])
    print('Experimental prediction: '+str(csv))


if __name__=='__main__':main()
