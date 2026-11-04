"""Create a private AOM owner-only draft; input-only challenge inference, never score or upload."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import runpy
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT/'solution/docs/OWNER_TECHNICAL_REPORT_TEMPLATE.md'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path, value): Path(path).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def worker(spec):
    original = Path(spec['workspace']).resolve(); runtime = Path(spec['runtime']).resolve()
    output = Path(spec['output']).resolve(); harness = Path(__file__).resolve(); denied = []
    assets = {Path(p).resolve() for p in spec['asset_files']}
    permitted = (runtime, Path(sys.prefix).resolve(), output.parent)
    def guard(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes)): return
        path = Path(os.fsdecode(args[0])).resolve(); normal = str(path).replace('\\', '/').lower(); name = path.name.lower()
        if path == harness: return  # Allow PyTorch's inspection of this isolated verifier only.
        reason = None
        if name in ('references.npz', 'targets.npz', 'training_targets.npz', 'possession_by_frame.pkl') or 'possession_gt' in name or name.endswith('_evidence.npz'):
            reason = 'reference_or_benchmark'
        elif '/solution/tools/' in normal: reason = 'research_tool'
        elif path.is_relative_to(original) and path not in assets and not any(path.is_relative_to(p) for p in permitted):
            reason = 'unapproved_workspace_file'
        if reason:
            denied.append({'reason': reason, 'name': path.name}); raise PermissionError(reason)
    sys.addaudithook(guard)
    sys.path = [p for p in sys.path if not Path(p or '.').resolve().is_relative_to(original)
                or Path(p or '.').resolve().is_relative_to(Path(sys.prefix).resolve())]
    sys.path[:0] = [str(runtime/'solution'), str(runtime/'more-than-a-hack-2027-main repo')]
    status = 'failed'
    try:
        if spec['kind'] == 'guard_selftest':
            for probe in spec['probes']:
                try:
                    with open(probe, 'rb'): pass
                    raise AssertionError('Guard permitted a protected probe')
                except PermissionError: pass
            if [x['reason'] for x in denied] != ['reference_or_benchmark', 'research_tool']:
                raise AssertionError('Unexpected guard self-test')
            status = 'passed'; return
        sys.argv = [str(runtime/'solution/owner_predict.py'), '--challenge', '--output', str(output)]
        runpy.run_path(sys.argv[0], run_name='__main__')
        # Run the supplied standalone validator on the actual CSV, suppressing
        # its possession counts/durations and all other competition statistics.
        checker = runtime/'more-than-a-hack-2027-main repo/evaluation/validate_submission.py'
        sys.argv = [str(checker), str(output)]
        with contextlib.redirect_stdout(io.StringIO()): runpy.run_path(str(checker), run_name='__main__')
        import games
        import numpy as np
        import pandas as pd
        from possession.inference import validate_prediction
        table = pd.read_csv(output); period = np.zeros(games.source_frames(), np.int8)
        for half, first, last in games.half_video_blocks(games.GAME): period[first:last+1] = half
        validate_prediction(table['label'].to_numpy(), period)
        record = read(output.with_suffix('.run.json'))
        if record['scored'] or record['reference_arrays_read'] or record['variant'] != 'tcn_owner_blend':
            raise AssertionError('Unexpected prediction provenance')
        if record['prediction_sha256'] != digest(output): raise AssertionError('Prediction identity differs')
        if denied: raise AssertionError('Inference attempted a forbidden read')
        status = 'passed'
    finally:
        save(output.parent/'guard.json', dict(status=status, kind=spec['kind'], forbidden_access_attempts=denied,
            official_format_passed=status == 'passed' and spec['kind'] == 'inference',
            published_period_coverage_passed=status == 'passed' and spec['kind'] == 'inference',
            reference_arrays_read=False, quality_scoring_performed=False, competition_statistics_reported=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT/'outputs/owner_candidate/AOM_owner_runtime_v1.zip')
    parser.add_argument('--verification', type=Path, default=ROOT/'outputs/owner_candidate/runtime_verification_20261104_v2/report.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve(); archive_path = args.archive.resolve()
    if not out.is_relative_to(ROOT/'outputs'): raise ValueError('Private draft must stay inside ignored workspace outputs')
    if out.exists(): raise FileExistsError('Choose a new draft folder; previous artifacts are preserved')
    verification = read(args.verification)
    if verification['status'] != 'passed' or digest(archive_path) != verification['archive_sha256']:
        raise ValueError('Require the verified owner-only archive')
    if digest(ROOT/'solution/tools/verify_owner_runtime.py') != verification['source_sha256']:
        raise ValueError('Runtime verification source drift')
    out.mkdir(parents=True); snapshot = out/'runtime'; snapshot.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read('OWNER_RUNTIME_MANIFEST.json')); names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(manifest['files']) | {'OWNER_RUNTIME_MANIFEST.json'}:
            raise ValueError('Archive inventory differs')
        for name in names:
            relative = PurePosixPath(name); destination = (snapshot/name).resolve()
            if '\\' in name or ':' in name or relative.is_absolute() or '..' in relative.parts or not destination.is_relative_to(snapshot):
                raise ValueError('Unsafe archive entry')
            data = archive.read(name)
            if name in manifest['files'] and hashlib.sha256(data).hexdigest() != manifest['files'][name]: raise ValueError('Archive file changed')
            destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(data)
    if manifest['candidate_manifest_sha256'] != verification['candidate_manifest_sha256']:
        raise ValueError('Candidate differs from external runtime verification')
    checker_relative = 'more-than-a-hack-2027-main repo/evaluation/validate_submission.py'
    shutil.copyfile(ROOT/checker_relative, snapshot/checker_relative)
    sys.path.insert(0, str(ROOT/'more-than-a-hack-2027-main repo')); import games
    paths = games.asset_paths(); assets = {key: Path(paths[key]).resolve() for key in ('atd_meta', 'atd_raw')}
    asset_hashes = {key: digest(path) for key, path in assets.items()}
    owner = read(snapshot/'models/tcn-owner-v1/manifest.json')
    if digest(snapshot/'models/tcn-owner-v1/manifest.json') != manifest['candidate_manifest_sha256']:
        raise ValueError('Owner manifest hash differs')
    training_frames = 0
    for match in owner['training_matches']:
        audit = read(ROOT/'data/external/sportec'/match/'prepared/audit.json')
        training_frames += sum(audit['label_counts'][label] for label in ('H', 'A', 'D'))
    remote = subprocess.run(['git', 'remote', 'get-url', 'origin'], capture_output=True, text=True, check=True).stdout.strip()
    if remote.endswith('.git'): remote = remote[:-4]
    if not remote.startswith('https://github.com/'): raise ValueError('Review report repository URL before packaging')
    protocol = dict(candidate_manifest_sha256=manifest['candidate_manifest_sha256'],
        archive_sha256=verification['archive_sha256'], external_runtime_verification_sha256=digest(args.verification),
        asset_sha256=asset_hashes, source_sha256=digest(Path(__file__)), report_template_sha256=digest(TEMPLATE),
        standalone_validator_sha256=digest(ROOT/checker_relative), official_local_rules_sha256=digest(ROOT/'more-than-a-hack-2027-main repo/RULES.md'),
        current_online_instructions_reviewed='2026-11-04', team='AOM', variant='tcn_owner_blend',
        policy='Frozen input-only inference and draft preparation; no label scoring, tuning, manual corrections, publication or submission.')
    save(out/'protocol.json', protocol)
    harness = out/'candidate_worker.py'; shutil.copyfile(Path(__file__), harness)
    specs = out/'specs'; specs.mkdir(); runs = out/'runs'; runs.mkdir(); cwd = out/'cwd'; cwd.mkdir()
    env = os.environ.copy(); env.pop('PYTHONPATH', None); env['PYTHONNOUSERSITE'] = '1'
    env['MTAH_DATA_DIR'] = str(Path(games.DATA_DIR).resolve())
    def job(name, kind):
        folder = runs/name; folder.mkdir()
        return dict(name=name, kind=kind, workspace=str(ROOT), runtime=str(snapshot),
            output=str(folder/'AOM.csv'), asset_files=[str(p) for p in assets.values()],
            probes=[str(out/'forbidden_possession_gt.csv'), str(ROOT/'solution/tools/validate_owner_game3.py')])
    def run(spec):
        spec_path = specs/(spec['name']+'.json'); save(spec_path, spec)
        done = subprocess.run([sys.executable, str(harness), '--worker', str(spec_path)], cwd=cwd,
                              env=env, capture_output=True, text=True)
        folder = Path(spec['output']).parent; (folder/'worker.log').write_text(done.stdout+done.stderr, encoding='utf-8')
        if done.returncode or read(folder/'guard.json')['status'] != 'passed':
            raise RuntimeError('Draft verification failed; inspect the private worker log')
        print('PASS: '+spec['name'], flush=True)
    run(job('guard_selftest', 'guard_selftest'))
    with ThreadPoolExecutor(max_workers=2) as pool: list(pool.map(run, [job('run_a', 'inference'), job('run_b', 'inference')]))
    first, second = runs/'run_a/AOM.csv', runs/'run_b/AOM.csv'
    if digest(first) != digest(second): raise ValueError('Full-input predictions are not byte-identical')
    for key, path in assets.items():
        if digest(path) != asset_hashes[key]: raise ValueError('Tracking input changed during execution')
    for name, expected in manifest['files'].items():
        if digest(snapshot/name) != expected: raise ValueError('Frozen runtime artifact changed')
    if digest(archive_path) != protocol['archive_sha256']: raise ValueError('Archive changed during execution')
    candidate = out/'private_candidate'; candidate.mkdir(); csv = candidate/'AOM.csv'
    shutil.copyfile(first, csv); shutil.copyfile(first.with_suffix('.run.json'), csv.with_suffix('.run.json'))
    sys.path.insert(0, str(snapshot/'solution/src'))
    from possession.inference import official_format_check
    official_format_check(csv, games.source_frames())
    if digest(csv) != digest(first): raise ValueError('Draft copy differs from validated CSV')
    sidecars = [read(p.with_suffix('.run.json')) for p in (first, second)]
    if sidecars[0]['input_fingerprint'] != sidecars[1]['input_fingerprint'] or any(
        item['bundle_manifest_sha256'] != manifest['candidate_manifest_sha256'] or item['input']['components'] != owner['components'] for item in sidecars):
        raise ValueError('Repeated run provenance differs')
    text = TEMPLATE.read_text(encoding='utf-8')
    replacements = dict(REPOSITORY_URL=remote, TRAINING_FRAMES=f'{training_frames:,}', TRAINING_HOURS=f'{training_frames/25/3600:.2f}',
        CSV_SHA256=digest(csv), CANDIDATE_SHA256=manifest['candidate_manifest_sha256'], ARCHIVE_SHA256=protocol['archive_sha256'])
    for key, value in replacements.items(): text = text.replace('{{'+key+'}}', value)
    if '{{' in text or '}}' in text: raise ValueError('Unresolved technical-report fields')
    technical = candidate/'AOM.md'; technical.write_text(text, encoding='utf-8')
    record = dict(status='draft', team_name='AOM', candidate='tcn_owner_blend', primary_selection_pending=True,
        csv_filename=csv.name, csv_sha256=digest(csv), technical_report_filename=technical.name, technical_report_sha256=digest(technical),
        sidecar_sha256=digest(csv.with_suffix('.run.json')), candidate_manifest_sha256=manifest['candidate_manifest_sha256'],
        runtime_archive_sha256=protocol['archive_sha256'], protocol_sha256=digest(out/'protocol.json'),
        official_format_passed=True, published_period_coverage_passed=True, byte_identical_repeated_runs=True,
        input_files_unchanged=True, model_and_runtime_unchanged=True, reference_arrays_read=False,
        quality_scoring_performed=False, competition_statistics_reported=False, published=False, submitted=False,
        public_repository_url=remote, public_repository_visibility_verified=False, model_archive_publication_verified=False)
    save(candidate/'candidate_manifest.json', record)
    handoff = f'''# AOM ownership-only draft

Prepared locally for review. Final candidate selection remains pending; nothing is uploaded or submitted.

- Private possession file: `private_candidate/AOM.csv`.
- Matching technical document: `private_candidate/AOM.md`.
- Identity/provenance: `private_candidate/candidate_manifest.json`, the CSV sidecar and `protocol.json`.
- Runtime archive: `{archive_path.relative_to(ROOT).as_posix()}` (already verified separately; contains no match data).
- Configured code remote: {remote}. The current code/model release's public availability is unverified.

Both full-input executions passed the supplied standalone format checker and published-period coverage checks; CSV bytes match exactly. Reference scoring was not performed, and no competition statistics are reported. The only match files opened by guarded inference were the tactical metadata and raw tracking. Earlier candidates remain intact.

For an organizer to reproduce this candidate, the matching code and fitted model archive must both be accessible. Git ignores model weights and local outputs, so pushing code alone will not publish the weights. Keep match data and this private candidate folder out of the public runtime archive. The user handles commits/pushes, release publication, final choice and registered submission.

The [current submission instructions](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md) accept a Markdown technical report and require reproducible code. The [rules](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md) prohibit human labels/corrections from the provided game and reserve its public correctness window for pipeline checks. Reviewed online 4 November 2026. The report discloses all external labelled datasets and amounts.
'''
    (out/'HANDOFF.md').write_text(handoff, encoding='utf-8')
    save(out/'verification.json', dict(status='passed', protocol_sha256=digest(out/'protocol.json'),
        candidate_manifest_sha256=digest(candidate/'candidate_manifest.json'), handoff_sha256=digest(out/'HANDOFF.md'),
        guard_selftest_passed=True, guarded_inference_passed=True, official_format_passed=True,
        published_period_coverage_passed=True, byte_identical_repeated_runs=True,
        input_and_runtime_hashes_unchanged=True, reference_arrays_read=False, quality_scoring_performed=False,
        competition_numeric_results_reported=False, scope='Local full-input reproducibility and draft files; not semantic accuracy or submission.'))
    print('PASS: private AOM CSV and matching technical report prepared; nothing published or scored', flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker': worker(read(sys.argv[2]))
    else: main()
