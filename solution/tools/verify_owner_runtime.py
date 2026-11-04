"""Extract and verify the owner-only package against saved external predictions, without rescoring."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import runpy
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value): Path(path).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def worker(spec):
    snapshot = Path(spec['snapshot']).resolve(); output = Path(spec['output']).resolve()
    input_path = Path(spec['input']).resolve(); original = Path(spec['workspace']).resolve(); denied = []
    harness = Path(__file__).resolve()
    permitted = (snapshot, Path(sys.prefix).resolve(), output.parent)
    def guard(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes)): return
        path = Path(os.fsdecode(args[0])).resolve(); text = str(path).replace('\\', '/').lower(); name = path.name.lower()
        # PyTorch inspects caller source. Permit only this isolated harness copy;
        # original research tools and all benchmark/reference artifacts remain blocked.
        if path == harness: return
        reason = None
        if name in ('references.npz', 'targets.npz', 'training_targets.npz', 'possession_by_frame.pkl') or 'possession_gt' in name or name.endswith('_evidence.npz'):
            reason = 'reference_or_benchmark'
        elif '/solution/tools/' in text: reason = 'research_tool'
        elif path.is_relative_to(original) and path != input_path and not any(path.is_relative_to(p) for p in permitted):
            reason = 'original_workspace_dependency'
        if reason:
            denied.append({'reason': reason, 'name': path.name}); raise PermissionError(reason)
    sys.addaudithook(guard)
    # Remove original repository/tool search paths and run only the extracted code.
    sys.path = [p for p in sys.path if not Path(p or '.').resolve().is_relative_to(original)
                or Path(p or '.').resolve().is_relative_to(Path(sys.prefix).resolve())]
    sys.path.insert(0, str(snapshot/'solution'))
    status = 'failed'; message = None
    try:
        if spec['kind'] == 'guard_selftest':
            for probe in spec['probes']:
                try:
                    with open(probe, 'rb'): pass
                    raise AssertionError('Guard allowed forbidden probe')
                except PermissionError: pass
            if [x['reason'] for x in denied] != ['reference_or_benchmark', 'research_tool']:
                raise AssertionError('Unexpected guard probe reasons')
            status = 'passed'; return
        if spec['kind'] == 'api':
            import numpy as np
            import owner_predict as runtime
            base = runtime.t.load_bundle(snapshot/'models/possession-v1')
            research = runtime.t.current.load_research(snapshot/'models/research-specialists-clean', base)
            tcn = runtime.t.load_tcn(snapshot/'models/tcn-blend-v1', base, research)
            bundle = runtime.load_owner(snapshot/'models/tcn-owner-v1', base, research, tcn)
            runtime.t.torch.set_num_interop_threads(1)
            inputs = runtime.t.read_tracking_npz(input_path)
            result = runtime.predict(base, research, tcn, bundle, inputs)
            np.savez_compressed(output, labels=result['labels'], home=result['home_probability'],
                dead=result['dead_probability'], events=result['event_probability'])
        else:
            sys.argv = [str(snapshot/'solution/owner_predict.py'), '--input', str(input_path), '--output', str(output)]
            runpy.run_path(sys.argv[0], run_name='__main__')
        if spec.get('expected_error'): raise AssertionError('Negative case unexpectedly succeeded')
        if denied: raise AssertionError('Inference attempted a forbidden read')
        status = 'passed'
    except (ValueError, SystemExit) as error:
        expected = spec.get('expected_error'); message = str(error)
        if not expected or type(error).__name__ != expected['type'] or expected['text'] not in message: raise
        if denied: raise AssertionError('Negative case failed for a forbidden read instead')
        status = 'expected_rejection'
    finally:
        save(output.parent/(spec['name']+'.guard.json'), dict(status=status, forbidden_access_attempts=denied,
            exception_message=message, inference_reference_arrays_read=False, accuracy_scoring_performed=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); archive_path = args.archive.resolve(); out = args.output.resolve()
    if not out.is_relative_to(ROOT/'outputs'): raise ValueError('Verification output must be inside workspace outputs')
    if out.exists(): raise FileExistsError('Choose a new verification folder')
    expected_sha = archive_path.with_suffix('.sha256').read_text().split()[0]
    if digest(archive_path) != expected_sha: raise ValueError('Archive checksum differs')
    out.mkdir(parents=True); snapshot = out/'runtime'; snapshot.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read('OWNER_RUNTIME_MANIFEST.json'))
        names = archive.namelist(); expected_names = set(manifest['files']) | {'OWNER_RUNTIME_MANIFEST.json'}
        if len(names) != len(set(names)) or set(names) != expected_names: raise ValueError('Archive inventory differs')
        if manifest['kind'] != 'ownership_only_runtime' or any(manifest[k] for k in ('contains_match_data', 'contains_predictions', 'contains_training_code')):
            raise ValueError('Unexpected archive scope')
        for name in names:
            relative = PurePosixPath(name)
            if '\\' in name or ':' in name or relative.is_absolute() or '..' in relative.parts: raise ValueError('Unsafe archive entry')
            target = (snapshot/name).resolve()
            if not target.is_relative_to(snapshot.resolve()): raise ValueError('Archive path escapes snapshot')
            data = archive.read(name)
            if name in manifest['files'] and hashlib.sha256(data).hexdigest() != manifest['files'][name]: raise ValueError('Archived file hash differs')
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
    if (snapshot/'data').exists() or (snapshot/'outputs').exists() or (snapshot/'solution/tools').exists():
        raise ValueError('Non-runtime directories in archive')
    candidate = read(snapshot/'models/tcn-owner-v1/manifest.json')
    if digest(snapshot/'models/tcn-owner-v1/manifest.json') != manifest['candidate_manifest_sha256']:
        raise ValueError('Archived owner identity differs')
    # Parent verifier can read expected predictions. Guarded runtime workers cannot.
    sys.path.insert(0, str(ROOT/'solution/tools'))
    import validate_owner_game3 as g
    import numpy as np
    import pandas as pd
    source = ROOT/'outputs/owner_transfer_game3/run_20261104'
    report = read(source/'report.json'); protocol = read(source/'protocol.json'); audit = read(source/'audit.json')
    if audit['status'] != 'passed' or any(digest(source/name) != candidate['validation_provenance'][name]
            for name in ('report.json', 'audit.json', 'protocol.json')): raise ValueError('Validated evidence provenance differs')
    for name, value in protocol['source_sha256'].items():
        if digest(ROOT/name) != value: raise ValueError('Validation source differs: '+name)
    for record in read(g.SOURCE/'source_manifest.json')['files']:
        if digest(g.SOURCE/record['path']) != record['sha256']: raise ValueError('External source changed')
    raw, _ = g.read_epts_tracking(g.SOURCE/'Sample_Game_3_metadata.xml', g.SOURCE/'Sample_Game_3_tracking.txt')
    inputs_dir = out/'inputs'; inputs_dir.mkdir(); cases = {}
    for condition in ('native', 'moderate', 'contextual'):
        inputs, _ = g.t.noisy_inputs(raw, condition, 17, protocol); stem = condition+'_17'
        if digest(source/(stem+'.json')) != report['case_report_sha256'][stem]: raise ValueError('Expected case changed')
        case = read(source/(stem+'.json'))
        if digest(source/(stem+'_evidence.npz')) != case['evidence_sha256'] or g.t.runtime.array_digest(inputs) != case['input_fingerprint']:
            raise ValueError('Expected evidence/input differs')
        np.savez_compressed(inputs_dir/(condition+'.npz'), **inputs)
        with np.load(source/(stem+'_evidence.npz'), allow_pickle=False) as z:
            cases[condition] = {k: z[k] for k in ('blend_labels', 'blend_home', 'blend_dead', 'current_events')}
    tiny = {k: (v[:128].copy() if k in ('players_xy', 'ball_xy', 'period') else v.copy()) for k, v in raw.items()}
    tiny['players_xy'][:] = np.nan; tiny['ball_xy'][:] = np.nan; tiny['period'][:] = np.r_[np.ones(50), np.zeros(20), np.full(58, 2)]
    np.savez_compressed(inputs_dir/'all_missing.npz', **tiny)
    np.savez_compressed(inputs_dir/'invalid.npz', **tiny, labels=np.full(128, 'H'))
    specs = out/'specs'; specs.mkdir(); runs = out/'runs'; runs.mkdir(); cwd = out/'cwd'; cwd.mkdir()
    harness = out/'runtime_guard_worker.py'; shutil.copyfile(Path(__file__), harness)
    def job(name, condition, kind='cli', expected_error=None):
        folder = runs/name; folder.mkdir(); output = folder/('emissions.npz' if kind == 'api' else 'prediction.csv')
        return dict(name=name, kind=kind, workspace=str(ROOT), snapshot=str(snapshot), input=str(inputs_dir/(condition+'.npz')),
                    output=str(output), expected_error=expected_error)
    jobs = [job('native', 'native'), job('moderate', 'moderate'), job('contextual', 'contextual'),
            job('native_repeat', 'native'), job('moderate_probabilities', 'moderate', 'api'), job('all_missing', 'all_missing'),
            job('invalid_fields', 'invalid', expected_error={'type': 'ValueError', 'text': 'six model input fields'})]
    probe = job('guard_selftest', 'all_missing', 'guard_selftest')
    probe['probes'] = [str(source/'references.npz'), str(ROOT/'solution/tools/validate_owner_game3.py')]
    jobs.append(probe)
    env = os.environ.copy(); env.pop('PYTHONPATH', None); env['PYTHONNOUSERSITE'] = '1'; env['PIP_NO_INDEX'] = '1'
    def run(spec):
        spec_path = specs/(spec['name']+'.json'); save(spec_path, spec)
        done = subprocess.run([sys.executable, str(harness), '--worker', str(spec_path)],
                              capture_output=True, text=True, cwd=cwd, env=env)
        Path(spec['output']).parent.joinpath('worker.log').write_text(done.stdout+done.stderr, encoding='utf-8')
        done.check_returncode(); guard = read(Path(spec['output']).parent/(spec['name']+'.guard.json'))
        if guard['status'] not in ('passed', 'expected_rejection'): raise ValueError('Guarded runtime failed')
        print(spec['name']+': '+guard['status'], flush=True)
        return {'name': spec['name'], **guard}
    with ThreadPoolExecutor(max_workers=2) as pool: results = list(pool.map(run, jobs))
    for condition in ('native', 'moderate', 'contextual'):
        actual = pd.read_csv(runs/condition/'prediction.csv')
        np.testing.assert_array_equal(actual['frame'], np.arange(len(raw['period'])))
        np.testing.assert_array_equal(actual['label'], cases[condition]['blend_labels'])
        sidecar = read(runs/condition/'prediction.run.json')
        if sidecar['variant'] != 'tcn_owner_blend' or sidecar['scored'] or sidecar['reference_arrays_read'] or not sidecar['official_format_valid']:
            raise ValueError('Inference sidecar differs')
        if sidecar['bundle_manifest_sha256'] != manifest['candidate_manifest_sha256'] or sidecar['input']['components'] != candidate['components']:
            raise ValueError('Inference candidate identity differs')
    if digest(runs/'native/prediction.csv') != digest(runs/'native_repeat/prediction.csv'): raise ValueError('Repeated CSV differs')
    with np.load(runs/'moderate_probabilities/emissions.npz', allow_pickle=False) as z:
        for key, expected in [('labels', 'blend_labels'), ('home', 'blend_home'), ('dead', 'blend_dead'), ('events', 'current_events')]:
            np.testing.assert_array_equal(z[key], cases['moderate'][expected])
    missing = pd.read_csv(runs/'all_missing/prediction.csv')['label'].to_numpy()
    g.t.runtime.validate_prediction(missing, tiny['period'])
    invalid_output = runs/'invalid_fields/prediction.csv'
    if invalid_output.exists() or invalid_output.with_suffix('.run.json').exists(): raise ValueError('Invalid input produced output')
    # An attempted overwrite must leave both prior artifacts unchanged.
    existing = runs/'native/prediction.csv'; old_hashes = [digest(existing), digest(existing.with_suffix('.run.json'))]
    refusal = dict(jobs[0], name='overwrite_refusal', expected_error={'type': 'SystemExit', 'text': '2'})
    results.append(run(refusal))
    if old_hashes != [digest(existing), digest(existing.with_suffix('.run.json'))]: raise ValueError('Overwrite rejection changed outputs')
    for name, value in manifest['files'].items():
        if digest(snapshot/name) != value: raise ValueError('Extracted file changed during verification')
    if digest(archive_path) != expected_sha: raise ValueError('Archive changed')
    summary = dict(status='passed', archive_sha256=expected_sha, archive_file_count=len(manifest['files']),
        candidate_manifest_sha256=manifest['candidate_manifest_sha256'], exact_external_cli_replays=3,
        exact_probability_replays=1, byte_identical_repeat=True, all_missing_halftime_check=True,
        malformed_input_rejected=True, overwrite_refused=True, read_guard_selftest=True,
        runtime_reference_access_attempts=0, active_prediction_unknowns=0,
        dependency_environment='Existing pinned optional TCN environment; no new installation',
        quality_scoring_performed=False, challenge_inputs_read=False, source_sha256=digest(Path(__file__)), results=results)
    save(out/'report.json', summary); print('PASS: portable owner-only runtime verified; no new accuracy scoring', flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker': worker(read(sys.argv[2]))
    else: main()
