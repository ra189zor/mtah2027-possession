"""Final clean-source/fresh-environment verification; challenge summaries contain no metrics."""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def peak_memory():
    if os.name!='nt':return None
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in
            ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    counters=Counters();counters.cb=ctypes.sizeof(counters)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=ctypes.c_void_p
    api=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo
    api.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),wintypes.DWORD];api.restype=wintypes.BOOL
    return int(counters.PeakWorkingSetSize) if api(kernel.GetCurrentProcess(),ctypes.byref(counters),ctypes.sizeof(counters)) else None


def worker(spec_path):
    spec=json.loads(Path(spec_path).read_text());root=Path(spec['root']).resolve();record=Path(spec['record'])
    original=ROOT.resolve();permitted=[root,Path(sys.prefix).resolve(),record.parent]
    permitted_files={Path(p).resolve() for p in spec.get('asset_files',[])}
    denied=[]
    def guard(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
        raw=os.fsdecode(args[0]);path=Path(raw).resolve();normal=str(path).replace('\\','/').lower();base=path.name.lower()
        reason=None
        if base in ('targets.npz','training_targets.npz','possession_by_frame.pkl') or 'possession_gt' in base or 'groundtruth' in base or 'ground_truth' in base:
            reason='reference_artifact'
        elif '/solution/tools/' in normal:reason='research_runner'
        elif path.is_relative_to(original) and not any(path.is_relative_to(p) for p in permitted) and path not in permitted_files:
            reason='original_workspace_dependency'
        if reason:
            denied.append({'reason':reason,'name':path.name});raise PermissionError('Final-verification access guard: '+reason)
    sys.addaudithook(guard)
    status='failed';started=time.perf_counter();cpu=time.process_time();error_type=None
    try:
        if spec['kind']=='guard_selftest':
            for name in spec['probe_files']:
                try:
                    with open(name,'rb'):pass
                    raise AssertionError('Access guard did not block the probe')
                except PermissionError:pass
            if [d['reason'] for d in denied]!=['reference_artifact','research_runner']:raise AssertionError('Wrong guard self-test results')
            status='passed';return
        sys.path=[p for p in sys.path if Path(p or '.').resolve() not in (ROOT,ROOT/'solution/tools')]
        sys.argv=[str(root/'solution/predict.py'),*spec['arguments']]
        runpy.run_path(str(root/'solution/predict.py'),run_name='__main__')
        sidecar=Path(spec['output']).with_suffix('.run.json');run_record=json.loads(sidecar.read_text())
        if run_record['reference_arrays_read'] or run_record['scored']:raise AssertionError('Unexpected inference scoring/reference flag')
        # Re-open the actual CSV using the supplied standalone checker. Suppress its descriptive
        # prediction counts, especially on the competition source; no reference scoring is used.
        sys.argv=[str(root/'more-than-a-hack-2027-main repo/evaluation/validate_submission.py'),spec['output']]
        if spec['kind']!='challenge':sys.argv+=['--rows',str(run_record['rows'])]
        with contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(root/'more-than-a-hack-2027-main repo/evaluation/validate_submission.py'),run_name='__main__')
        if denied:raise AssertionError('Forbidden access was attempted')
        status='passed'
    except BaseException as error:
        error_type=type(error).__name__
        raise
    finally:
        result={'status':status,'kind':spec['kind'],'forbidden_access_attempts':denied,'error_type':error_type,
                'standalone_format_passed':status=='passed' and spec['kind']!='guard_selftest',
                'quality_scoring_performed':False}
        if spec['kind']=='external':result['resource']={'wall_seconds':time.perf_counter()-started,'cpu_seconds':time.process_time()-cpu,'peak_working_set_bytes':peak_memory()}
        write(record,result)


def prepare_snapshot(out,bundle,heldout):
    sys.path.insert(0,str(ROOT/'solution/src'))
    from possession.inference import load_bundle,RUNTIME_SOURCES
    final=load_bundle(bundle);held=load_bundle(heldout);snapshot=out/'clean_source';snapshot.mkdir(exist_ok=False)
    paths=list(RUNTIME_SOURCES)+['solution/src/possession/__init__.py','solution/requirements-lock.txt',
        'more-than-a-hack-2027-main repo/evaluation/validate_submission.py',
        'more-than-a-hack-2027-main repo/LICENSE','.gitattributes']
    inventory={}
    for relative in paths:
        source=ROOT/relative;destination=snapshot/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination);inventory[relative]=digest(destination)
    for source,name in [(bundle,'final'),(heldout,'heldout')]:
        target=snapshot/'models'/name;target.mkdir(parents=True)
        for filename in ('manifest.json','models.joblib'):shutil.copyfile(source/filename,target/filename)
    for name in ('inputs','cases','runs','cwd','specs'):(snapshot/name).mkdir()
    assert not (snapshot/'solution/tools').exists() and not (snapshot/'outputs').exists() and not (snapshot/'data').exists()
    write(out/'snapshot.json',{'source':'current candidate working-tree bytes, not a committed Git clone',
        'files':inventory,'research_outputs_copied':False,'reference_arrays_copied':False,
        'final_model_sha256':final['manifest']['model_sha256'],'heldout_model_sha256':held['manifest']['model_sha256']})
    return snapshot,final,held


def main():
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor,as_completed
    sys.path[:0]=[str(ROOT/'solution/tools'),str(ROOT/'solution/src')]
    import train_switch_events as s
    n=s.n
    from possession.inference import read_tracking_npz
    from possession.validation import corrupt
    from possession.robustness import contextual_corrupt
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--heldout-bundle',type=Path,required=True);parser.add_argument('--python',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():raise FileExistsError('Choose a new verification directory for another full run')
    python=args.python.resolve();snapshot,final,held=prepare_snapshot(out,args.bundle.resolve(),args.heldout_bundle.resolve())
    environment=os.environ.copy();environment.pop('PYTHONPATH',None);environment['PYTHONNOUSERSITE']='1';environment['PIP_NO_INDEX']='1';environment['PIP_DISABLE_PIP_VERSION_CHECK']='1'
    identity=subprocess.run([str(python),'-c','import sys,json,numpy; print(json.dumps({"prefix":sys.prefix,"base_prefix":sys.base_prefix,"numpy":numpy.__file__}))'],capture_output=True,text=True,check=True,env=environment)
    identity=json.loads(identity.stdout)
    if Path(identity['prefix']).resolve()==Path(identity['base_prefix']).resolve() or not Path(identity['numpy']).resolve().is_relative_to(Path(identity['prefix']).resolve()):
        raise ValueError('Fresh isolated environment required')
    cfg,_,_,f=s.setup();split=f[2];match=held['manifest']['held_out']
    if match in held['manifest']['training_matches']:raise ValueError('Held-out smoke model contains its test fixture')
    original=read_tracking_npz(ROOT/'data/external/sportec'/match/'prepared/inputs.npz')
    index=n.read(s.SOURCE/'prepared'/match/'index.json');baseline=n.read(s.SOURCE/'evaluation'/match/'report.json')
    event_root=ROOT/'outputs/switch_events/run_20261101';events=n.read(event_root/'outer'/match/'report.json')
    jobs=[];input_hashes={}
    def job(kind,name,variant,arguments,expected_hash=None,expect_failure=None,output=None,**extra):
        output=output or snapshot/'runs'/f'{name}.csv';record=snapshot/'runs'/f'{name}.guard.json'
        spec={'root':str(snapshot),'record':str(record),'kind':kind,'output':str(output),
            'arguments':arguments+['--output',str(output),'--quiet'],**extra}
        spec_path=snapshot/'specs'/f'{name}.json';write(spec_path,spec)
        return {'kind':kind,'name':name,'variant':variant,'spec':spec_path,'record':record,'output':output,
            'expected_hash':expected_hash,'expect_failure':expect_failure}
    for condition in ('clean','moderate','severe','contextual'):
        noisy=contextual_corrupt(original,split['regimes']['moderate'],cfg['contextual'],17)[0] if condition=='contextual' else corrupt(original,split['regimes'][condition],17)
        path=snapshot/'inputs'/f'{condition}.npz';np.savez_compressed(path,**noisy);input_hashes[str(path)]=digest(path)
        from possession.inputs import array_digest
        if array_digest(noisy)!=index['cases'][f'{condition}_17']['noisy_input_fingerprint']:raise ValueError('Frozen external input differs')
        for variant in ('baseline','event2','event4'):
            if variant=='baseline':expected=s.SOURCE/'evaluation'/match/f'{match}_{condition}_17_decoded.csv';expected_hash=baseline['files'][expected.name]
            else:expected=event_root/'outer'/match/f'{condition}_17_event_reward_{2 if variant=="event2" else 4}.csv';expected_hash=events['files'][expected.name]
            if digest(expected)!=expected_hash:raise ValueError('Frozen prediction drift')
            jobs.append(job('external',f'{condition}_{variant}',variant,
                ['--bundle',str(snapshot/'models/heldout'),'--input',str(path),'--variant',variant],expected_hash))
    # Small derived external fixtures exercise extreme missingness, never claim accuracy.
    tiny={k:v[:512].copy() if k in ('players_xy','ball_xy','period') else v.copy() for k,v in original.items()};tiny['period'][:]=1
    synthetic={}
    for name,missing in [('all_missing',('players_xy','ball_xy')),('ball_missing',('ball_xy',)),('players_missing',('players_xy',))]:
        inputs={k:v.copy() for k,v in tiny.items()}
        for field in missing:inputs[field][:]=np.nan
        path=snapshot/'inputs'/f'{name}.npz';np.savez_compressed(path,**inputs);synthetic[name]=path;input_hashes[str(path)]=digest(path)
        for variant in ('baseline','event2','event4'):
            jobs.append(job('synthetic',f'{name}_{variant}',variant,['--bundle',str(snapshot/'models/final'),'--input',str(path),'--variant',variant]))
    jobs.append(job('synthetic','all_missing_event4_repeat','event4',['--bundle',str(snapshot/'models/final'),'--input',str(synthetic['all_missing']),'--variant','event4']))
    bad_inputs={}
    for name in ('extra_field','complex','vector_fps','no_period','infinite'):
        inputs={k:v.copy() for k,v in tiny.items()}
        if name=='extra_field':inputs['labels']=np.full(512,'H')  # Dummy malformed-field test, not reference labels.
        if name=='complex':inputs['ball_xy']=inputs['ball_xy'].astype(np.complex64)
        if name=='vector_fps':inputs['fps']=np.array([25,25])
        if name=='no_period':inputs['period'][:]=0
        if name=='infinite':inputs['ball_xy'][0,0]=np.inf
        path=snapshot/'inputs'/f'bad_{name}.npz';np.savez_compressed(path,**inputs);bad_inputs[name]=path
        jobs.append(job('negative',f'reject_{name}','baseline',['--bundle',str(snapshot/'models/final'),'--input',str(path),'--variant','baseline'],expect_failure=True))
    clean=snapshot/'inputs/clean.npz'
    for name in ('corrupt_model','bad_version','stale_runtime'):
        folder=snapshot/'cases'/name;folder.mkdir()
        for filename in ('manifest.json','models.joblib'):shutil.copyfile(snapshot/'models/final'/filename,folder/filename)
        manifest=json.loads((folder/'manifest.json').read_text())
        if name=='corrupt_model':
            path=folder/'models.joblib';data=bytearray(path.read_bytes());data[0]^=1;path.write_bytes(data)
        if name=='bad_version':manifest['packages']['numpy']='0.0.0';write(folder/'manifest.json',manifest)
        if name=='stale_runtime':manifest['runtime_sha256']['solution/predict.py']='0'*64;write(folder/'manifest.json',manifest)
        jobs.append(job('negative',f'reject_{name}','baseline',['--bundle',str(folder),'--input',str(clean),'--variant','baseline'],expect_failure=True))
    jobs.append(job('negative','reject_missing_variant',None,['--bundle',str(snapshot/'models/final'),'--input',str(clean)],expect_failure=True))
    jobs.append(job('negative','reject_auto_variant','auto',['--bundle',str(snapshot/'models/final'),'--input',str(clean),'--variant','auto'],expect_failure=True))
    jobs.append(job('negative','reject_missing_input','baseline',['--bundle',str(snapshot/'models/final'),'--input',str(snapshot/'inputs/nonexistent.npz'),'--variant','baseline'],expect_failure=True))
    sentinel=snapshot/'runs/protected.csv';sentinel.write_bytes(b'preserve this existing output\n');sentinel_hash=digest(sentinel)
    jobs.append(job('negative','reject_existing_output','baseline',['--bundle',str(snapshot/'models/final'),'--input',str(clean),'--variant','baseline'],expect_failure=True,output=sentinel))
    probe_root=out/'guard_probes';(probe_root/'solution/tools').mkdir(parents=True,exist_ok=True)
    probes=[probe_root/'targets.npz',probe_root/'solution/tools/research_probe.py']
    for path in probes:path.write_bytes(b'dummy access-guard self-test, not reference data')
    jobs.append(job('guard_selftest','guard_selftest',None,[],probe_files=[str(p) for p in probes]))
    def run(item):
        result=subprocess.run([str(python),'-X','utf8',str(Path(__file__).resolve()),'--worker',str(item['spec'])],
            cwd=snapshot/'cwd',capture_output=True,text=True,encoding='utf-8',env=environment,timeout=300)
        private_log=Path(item['record']).with_suffix('.process.txt');private_log.write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        guard=json.loads(item['record'].read_text()) if item['record'].exists() else {'status':'failed','forbidden_access_attempts':[]}
        if item['expect_failure']:
            if result.returncode==0 or guard['forbidden_access_attempts']:raise RuntimeError('Negative input was not rejected cleanly: '+item['name'])
            if item['output']!=sentinel and (item['output'].exists() or item['output'].with_suffix('.run.json').exists()):raise RuntimeError('Rejected case published output')
            summary={'kind':'negative','name':item['name'],'rejected_without_publishing':True}
        else:
            if result.returncode or guard['status']!='passed':raise RuntimeError('Final verification failed: '+item['name']+'; see private process log')
            if item['kind']!='guard_selftest' and guard['forbidden_access_attempts']:raise RuntimeError('Forbidden runtime access')
            if item['expected_hash'] and digest(item['output'])!=item['expected_hash']:raise RuntimeError('Fresh-environment output changed: '+item['name'])
            if item['kind']=='external':
                resource=guard['resource']
                if resource['wall_seconds']>300 or resource['peak_working_set_bytes'] is None or resource['peak_working_set_bytes']>2*1024**3:
                    raise RuntimeError('External runtime budget exceeded or unavailable')
            summary={'kind':item['kind'],'name':item['name'],'variant':item['variant'],'format_valid':guard['standalone_format_passed'],
                'frozen_csv_byte_exact':bool(item['expected_hash']),'reference_arrays_read':False,'quality_scoring_performed':False}
            if item['kind']=='external':summary['resource']=guard['resource']
        print(f'PASS: {item["kind"]} {item["name"]}',flush=True);return summary
    results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run,item) for item in jobs]):results.append(future.result())
    if digest(sentinel)!=sentinel_hash:raise AssertionError('Existing output was replaced')
    if (snapshot/'runs/all_missing_event4.csv').read_bytes()!=(snapshot/'runs/all_missing_event4_repeat.csv').read_bytes():raise AssertionError('Missing-data repeat differs')
    if any(digest(Path(path))!=checksum for path,checksum in input_hashes.items()):raise AssertionError('Tracking inputs mutated')
    # Full local challenge-input acceptance, never score or publish its descriptive numbers.
    sys.path.insert(0,str(ROOT/'more-than-a-hack-2027-main repo'));import games
    assets=games.asset_paths();allowed=[str(Path(assets[k]).resolve()) for k in ('atd_meta','atd_raw')]
    challenge_environment=environment.copy();challenge_environment['MTAH_DATA_DIR']=str(Path(games.DATA_DIR).resolve())
    previous_environment=environment;environment=challenge_environment
    challenge=[]
    for variant in ('baseline','event2','event4'):
        for repeat in ('a','b'):
            challenge.append(job('challenge',f'challenge_{variant}_{repeat}',variant,
                ['--bundle',str(snapshot/'models/final'),'--challenge','--variant',variant],asset_files=allowed))
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run,item) for item in challenge]):results.append(future.result())
    environment=previous_environment
    repeats={}
    for variant in ('baseline','event2','event4'):
        if (snapshot/'runs'/f'challenge_{variant}_a.csv').read_bytes()!=(snapshot/'runs'/f'challenge_{variant}_b.csv').read_bytes():raise AssertionError('Challenge repeatability failure')
        repeats[variant]=True
    pip_check=subprocess.run([str(python),'-m','pip','check'],capture_output=True,text=True,env=environment,check=True)
    write(out/'fresh_environment.json',{'identity':identity,'isolated_system_site_packages':True,'pip_check':pip_check.stdout.strip(),'network_used':False})
    report={'status':'passed','verification_source_sha256':digest(Path(__file__)),'snapshot_manifest_sha256':digest(out/'snapshot.json'),
        'fresh_environment':True,'network_used':False,'heldout_external_exact_cases':12,'synthetic_runs':10,'negative_cases':12,
        'guard_selftest_passed':True,'challenge_contract_passed':True,'challenge_repeat_byte_exact_by_variant':repeats,
        'challenge_quality_scoring_performed':False,'competition_numeric_results_reported':False,
        'source_snapshot_note':'Minimal snapshot of current candidate source bytes, not a Git clone; no training cache/reference/research runner copied.',
        'results':results,'limitations':'No new quality estimate for final all-seven fits; no comparison to competition references; same host/OS/CPU, not cross-platform proof.'}
    write(out/'report.json',report);print(f'PASS: final verification record {out / "report.json"}',flush=True)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(sys.argv[2])
    else:main()
