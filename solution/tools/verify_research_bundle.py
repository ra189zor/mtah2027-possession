"""Input-only CLI replay and reference/research-access guards for experimental bundles."""
import json
import argparse
import os
from pathlib import Path
import runpy
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/reliability_research/runtime_verification'


def worker(spec):
    output=Path(spec['output']).resolve();input_path=Path(spec['input']).resolve();denied=[]
    def guard(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
        path=Path(os.fsdecode(args[0])).resolve();normal=str(path).replace('\\','/').lower()
        if path.name in ('targets.npz','training_targets.npz','possession_by_frame.pkl') or 'possession_gt' in normal or '/solution/tools/' in normal or ('/outputs/' in normal and path!=input_path and not path.is_relative_to(output.parent) and not path.is_relative_to(Path(sys.prefix).resolve())):
            denied.append(path.name);raise PermissionError('Reference/research read blocked')
    sys.addaudithook(guard)
    sys.path=[p for p in sys.path if Path(p or '.').resolve()!=ROOT/'solution/tools']
    sys.argv=[str(ROOT/'solution/research_predict.py'),'--base-bundle',spec['base'],'--research-bundle',spec['bundle'],
              '--input',spec['input'],'--variant',spec['variant'],'--output',spec['output']]
    runpy.run_path(sys.argv[0],run_name='__main__')
    if denied:raise AssertionError('Unexpected forbidden read')
    (output.parent/'guard.json').write_text(json.dumps({'status':'passed','forbidden_access_attempts':denied}))


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suffix',choices=['v2','clean'],default='v2')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();suffix=args.suffix
    experiment='run_20261101_v2' if suffix=='v2' else 'run_20261101_clean'
    if suffix!='v2':OUT=OUT.with_name('runtime_verification_'+suffix)
    if args.output:OUT=args.output.resolve()
    if not OUT.is_relative_to(ROOT/'outputs'):raise ValueError('Verification output must stay inside workspace outputs')
    OUT.mkdir(parents=True,exist_ok=False);jobs=[]
    for condition in ('clean','moderate'):
        for variant in ('gap','boundary','combined'):
            name=condition+'_'+variant;folder=OUT/name;folder.mkdir()
            jobs.append({'name':name,'variant':variant,'base':str(ROOT/'models/heldout-J03WN1-v1'),
                'bundle':str(ROOT/('models/research-heldout-J03WN1-'+suffix)),
                'input':str(ROOT/'outputs/final_verification/run_20261101/clean_source/inputs'/f'{condition}.npz'),
                'output':str(folder/'prediction.csv'),
                'expected':str(ROOT/'outputs/reliability_research'/experiment/'outer/J03WN1'/f'{condition}_17_{variant}.npy')})
    with np.load(jobs[0]['input'],allow_pickle=False) as z:tiny={k:(z[k][:128].copy() if k in ('players_xy','ball_xy','period') else z[k].copy()) for k in z.files}
    tiny['players_xy'][:]=np.nan;tiny['ball_xy'][:]=np.nan;tiny['period'][:]=1
    np.savez_compressed(OUT/'all_missing.npz',**tiny)
    for variant in ('gap','boundary','combined'):
        name='all_missing_'+variant;folder=OUT/name;folder.mkdir()
        jobs.append({'name':name,'variant':variant,'base':str(ROOT/'models/possession-v1'),
            'bundle':str(ROOT/('models/research-specialists-'+suffix)),'input':str(OUT/'all_missing.npz'),
            'output':str(folder/'prediction.csv'),'expected':None})
    def run(spec):
        path=Path(spec['output']).parent/'spec.json';path.write_text(json.dumps(spec))
        done=subprocess.run([sys.executable,str(Path(__file__)),'--worker',str(path)],capture_output=True,text=True)
        (path.parent/'worker.log').write_text(done.stdout+done.stderr,encoding='utf-8');done.check_returncode()
        labels=pd.read_csv(spec['output'])['label'].to_numpy()
        if spec['expected'] and not np.array_equal(labels,np.load(spec['expected'],allow_pickle=False)):raise ValueError('Research CLI differs from held-out experiment')
        record=json.loads(Path(spec['output']).with_suffix('.run.json').read_text())
        if record['reference_arrays_read'] or record['scored'] or not record['official_format_valid']:raise ValueError('Invalid inference provenance')
        return {'name':spec['name'],'status':'passed','exact_heldout_replay':bool(spec['expected']),'reference_arrays_read':False,'scored':False}
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,jobs))
    (OUT/'report.json').write_text(json.dumps({'status':'passed','results':results},indent=2))
    print('PASS: six exact held-out CLI replays and three missing-data deployment checks under read guards')


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(json.loads(Path(sys.argv[2]).read_text()))
    else:main()
