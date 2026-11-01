"""Guarded raw-input CLI checks against frozen external predictions, without references."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]


def guarded_worker(record,arguments):
    import runpy
    attempted_references=[];attempted_research=[]
    def guard(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
        name=args[0].decode(errors='replace') if isinstance(args[0],bytes) else args[0]
        normal=name.replace('\\','/').lower();base=normal.rsplit('/',1)[-1]
        if base in ('targets.npz','training_targets.npz','possession_by_frame.pkl') or 'possession_gt' in base or 'ground_truth' in base or 'groundtruth' in base:
            attempted_references.append(name);raise PermissionError('Inference attempted to open a reference artifact')
        if '/solution/tools/' in normal:
            attempted_research.append(name);raise PermissionError('Inference attempted to import/open a research runner')
    sys.addaudithook(guard);status='failed'
    try:
        sys.argv=[str(ROOT/'solution/predict.py'),*arguments]
        runpy.run_path(str(ROOT/'solution/predict.py'),run_name='__main__');status='passed'
    finally:
        Path(record).write_text(json.dumps({'status':status,'reference_access_attempts':attempted_references,
            'research_runner_access_attempts':attempted_research},indent=2)+'\n',encoding='utf-8')


def main():
    import subprocess
    from concurrent.futures import ThreadPoolExecutor,as_completed
    import numpy as np
    sys.path[:0]=[str(ROOT/'solution/tools'),str(ROOT/'solution/src')]
    import train_switch_events as s
    n=s.n
    from possession.inference import read_tracking_npz,load_bundle
    from possession.validation import corrupt
    from possession.robustness import contextual_corrupt
    from possession.inputs import file_digest,array_digest
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--heldout-bundle',type=Path,required=True)
    parser.add_argument('--final-bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    for child in ('inputs','heldout','final','unrelated_cwd'):(out/child).mkdir()
    held=load_bundle(args.heldout_bundle);final=load_bundle(args.final_bundle)
    match=held['manifest']['held_out']
    if held['manifest']['kind']!='held_out_verification' or match in held['manifest']['training_matches']:
        raise ValueError('Pipeline verification requires a disjoint held-out bundle')
    if final['manifest']['kind']!='final_all_external_matches':raise ValueError('Final deployment bundle required')
    cfg,_,_,f=s.setup();split=f[2]
    original=read_tracking_npz(ROOT/'data/external/sportec'/match/'prepared/inputs.npz')
    index=n.read(s.SOURCE/'prepared'/match/'index.json')
    old_baseline=n.read(s.SOURCE/'evaluation'/match/'report.json')
    event_source=ROOT/'outputs/switch_events/run_20261101';old_events=n.read(event_source/'outer'/match/'report.json')
    fingerprints={};jobs=[]
    for condition in ('clean','moderate','contextual'):
        if condition=='contextual':noisy,_,_=contextual_corrupt(original,split['regimes']['moderate'],cfg['contextual'],17)
        else:noisy=corrupt(original,split['regimes'][condition],17)
        fingerprint=array_digest(noisy)
        if fingerprint!=index['cases'][f'{condition}_17']['noisy_input_fingerprint']:raise ValueError('Raw corruption did not reproduce frozen input')
        input_path=out/'inputs'/f'{condition}.npz';np.savez_compressed(input_path,**noisy);fingerprints[str(input_path)]=file_digest(input_path)
        for variant in ('baseline','event2','event4'):
            if variant=='baseline':
                expected=s.SOURCE/'evaluation'/match/f'{match}_{condition}_17_decoded.csv'
                expected_hash=old_baseline['files'][expected.name]
            else:
                strength=2 if variant=='event2' else 4
                expected=event_source/'outer'/match/f'{condition}_17_event_reward_{strength}.csv'
                expected_hash=old_events['files'][expected.name]
            if file_digest(expected)!=expected_hash:raise ValueError('Frozen expected prediction changed')
            jobs.append({'kind':'held_out','condition':condition,'variant':variant,'input':input_path,
                'bundle':args.heldout_bundle.resolve(),'output':out/'heldout'/f'{condition}_{variant}.csv',
                'expected_sha256':expected_hash,'expected_fingerprint':fingerprint})
    # Final models see this fixture in training: these are run/repeatability checks, never quality scores.
    clean_input=out/'inputs/clean.npz'
    for variant,name in [('baseline','baseline'),('event2','event2'),('event4','event4_a'),('event4','event4_b')]:
        jobs.append({'kind':'final_smoke_in_sample','condition':'clean','variant':variant,'input':clean_input,
            'bundle':args.final_bundle.resolve(),'output':out/'final'/f'{name}.csv',
            'expected_sha256':None,'expected_fingerprint':index['cases']['clean_17']['noisy_input_fingerprint']})
    def run(job):
        csv=job['output'];record=csv.with_suffix('.guard.json')
        command=[sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--guarded-predict',str(record),
            '--bundle',str(job['bundle']),'--input',str(job['input']),'--variant',job['variant'],'--output',str(csv),'--quiet']
        result=subprocess.run(command,cwd=out/'unrelated_cwd',capture_output=True,text=True,encoding='utf-8')
        csv.with_suffix('.stdout.txt').write_text(result.stdout,encoding='utf-8')
        if result.returncode:raise RuntimeError(result.stderr)
        guard_record=n.read(record);run_record=n.read(csv.with_suffix('.run.json'))
        if guard_record['status']!='passed' or guard_record['reference_access_attempts'] or guard_record['research_runner_access_attempts']:
            raise ValueError('Guarded inference attempted forbidden data/tool access')
        if run_record['input_fingerprint']!=job['expected_fingerprint'] or run_record['reference_arrays_read'] or run_record['scored']:
            raise ValueError('Unexpected inference input/reference/scoring behavior')
        digest=file_digest(csv)
        if job['expected_sha256'] is not None and digest!=job['expected_sha256']:raise ValueError('Raw pipeline CSV differs from frozen external prediction')
        if run_record['prediction_sha256']!=digest:raise ValueError('Run sidecar CSV digest differs')
        print(f'{job["kind"]} {job["condition"]} {job["variant"]}: guarded CLI and '+('exact frozen CSV' if job['expected_sha256'] else 'format check')+' passed',flush=True)
        return {'kind':job['kind'],'condition':job['condition'],'variant':job['variant'],'prediction':str(csv),
            'sha256':digest,'timing':run_record['timing'],'reference_access_attempts':0,'research_runner_access_attempts':0,
            'frozen_csv_exact':job['expected_sha256'] is not None}
    results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run,job) for job in jobs]):results.append(future.result())
    a=out/'final/event4_a.csv';b=out/'final/event4_b.csv'
    if a.read_bytes()!=b.read_bytes():raise ValueError('Repeated final CLI did not produce identical bytes')
    if any(file_digest(Path(path))!=digest for path,digest in fingerprints.items()):raise ValueError('CLI modified copied tracking inputs')
    report={'status':'passed','held_out_fixture':match,'held_out_raw_cases':9,'final_in_sample_smoke_runs':4,
        'guarded_subprocesses':13,'frozen_csvs_byte_exact':9,'final_repeat_byte_exact':True,
        'reference_arrays_read':False,'research_runners_loaded_by_inference':False,'quality_scoring_performed':False,
        'heldout_manifest_sha256':file_digest(args.heldout_bundle.resolve()/'manifest.json'),
        'final_manifest_sha256':file_digest(args.final_bundle.resolve()/'manifest.json'),
        'verification_source_sha256':file_digest(Path(__file__)),'results':results}
    n.write(out/'report.json',report);print(f'PASS: packaged raw-input verification {out / "report.json"}',flush=True)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--guarded-predict':guarded_worker(sys.argv[2],sys.argv[3:])
    else:main()
