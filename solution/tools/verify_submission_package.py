"""Audit public archive boundaries and regenerate private candidates from its extracted runtime."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from package_submission import ROOT, VARIANTS, GENERATED, encoded, runtime_paths, safe_name, sha


def audit_archive(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != runtime_paths() | GENERATED:
            raise ValueError('Archive inventory differs from public allowlist')
        files = {}
        for info in archive.infolist():
            safe_name(info.filename)
            if (info.external_attr >> 16) & 0o170000 != 0o100000:
                raise ValueError('Non-regular archive entry')
            files[info.filename] = archive.read(info)
    manifest = json.loads(files['PUBLIC_MANIFEST.json'])
    if set(manifest['files']) != set(files) - {'PUBLIC_MANIFEST.json', 'SHA256SUMS.txt'}:
        raise ValueError('Manifest inventory differs')
    for name, item in manifest['files'].items():
        if sha(files[name]) != item['sha256'] or len(files[name]) != item['bytes']:
            raise ValueError('Public payload hash differs: ' + name)
    sums = {}
    for line in files['SHA256SUMS.txt'].decode().splitlines():
        checksum, name = line.split('  ', 1)
        if name in sums:
            raise ValueError('Duplicate checksum')
        sums[name] = checksum
    if set(sums) != set(files) - {'SHA256SUMS.txt'}:
        raise ValueError('Checksum inventory differs')
    if any(sha(files[name]) != checksum for name, checksum in sums.items()):
        raise ValueError('Checksum differs')
    model = json.loads(files['models/possession-v1/manifest.json'])
    if sha(files['models/possession-v1/models.joblib']) != model['model_sha256']:
        raise ValueError('Model identity differs')
    if any(sha(files[name]) != checksum for name, checksum in model['runtime_sha256'].items()):
        raise ValueError('Runtime identity differs')
    return files


def verify(package, python):
    package = package.resolve()
    if not package.is_relative_to(ROOT/'outputs'):
        raise ValueError('Verification must stay in workspace outputs')
    index = json.loads((package/'PACKAGE_INDEX.json').read_text())
    archive = package/index['public_archive']['name']
    if sha(archive.read_bytes()) != index['public_archive']['sha256']:
        raise ValueError('Archive SHA differs from package index')
    files = audit_archive(archive)
    work = package/'verification'
    work.mkdir(exist_ok=False)
    extracted = work/'runtime'
    extracted.mkdir()
    for name, data in files.items():
        destination = (extracted/name).resolve()
        if not destination.is_relative_to(extracted):
            raise ValueError('Unsafe extraction')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    environment['PYTHONNOUSERSITE'] = '1'
    checked = subprocess.run([str(python), str(extracted/'solution/verify_release.py')],
                             cwd=extracted, capture_output=True, text=True, env=environment)
    (work/'checksum_check.txt').write_text(checked.stdout + checked.stderr, encoding='utf-8')
    checked.check_returncode()
    sys.path.insert(0, str(ROOT/'more-than-a-hack-2027-main repo'))
    import games
    environment['MTAH_DATA_DIR'] = str(Path(games.DATA_DIR).resolve())
    assets = games.asset_paths()
    allowed = [str(Path(assets[key]).resolve()) for key in ('atd_meta', 'atd_raw')]

    def run(variant):
        folder = work/variant
        folder.mkdir()
        output = folder/'prediction.csv'
        record = folder/'guard.json'
        spec = {'root': str(extracted), 'record': str(record), 'kind': 'challenge',
                'output': str(output), 'asset_files': allowed,
                'arguments': ['--bundle', str(extracted/'models/possession-v1'), '--challenge',
                              '--variant', variant, '--output', str(output), '--quiet']}
        spec_path = folder/'spec.json'
        spec_path.write_bytes(encoded(spec))
        completed = subprocess.run([str(python), str(ROOT/'solution/tools/final_verify.py'),
                                    '--worker', str(spec_path)], cwd=extracted,
                                   capture_output=True, text=True, env=environment)
        (folder/'worker.log').write_text(completed.stdout + completed.stderr, encoding='utf-8')
        completed.check_returncode()
        guard = json.loads(record.read_text())
        if guard['status'] != 'passed' or guard['forbidden_access_attempts'] or guard['quality_scoring_performed']:
            raise ValueError('Isolated inference guard failed')
        expected = index['candidates'][variant]
        candidate_name = expected.get('csv_filename', 'mtah2027-possession.csv')
        if Path(candidate_name).name != candidate_name:
            raise ValueError('Unsafe candidate filename')
        candidate = package/'private_candidates'/variant/candidate_name
        if sha(candidate.read_bytes()) != expected['prediction_sha256'] or output.read_bytes() != candidate.read_bytes():
            raise ValueError('Candidate reproduction differs')
        report = candidate.with_suffix('.md').read_bytes()
        if sha(report) != expected['report_sha256'] or b'{{' in report:
            raise ValueError('Candidate report differs')
        return {'variant': variant, 'format_passed': guard['standalone_format_passed'],
                'byte_identical_to_private_candidate': True, 'forbidden_access_attempts': [],
                'quality_scoring_performed': False}

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, VARIANTS))
    result = {'status': 'passed', 'public_archive_sha256': sha(archive.read_bytes()),
              'strict_public_inventory_passed': True, 'extracted_checksums_passed': True,
              'python': str(python), 'candidates': results,
              'team_name': index['team_name'], 'primary_variant': None, 'published': False, 'submitted': False}
    (package/'VERIFICATION.json').write_bytes(encoded(result))
    print('PASS: archive boundaries, checksums and exact isolated reproduction of all three candidates.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    args = parser.parse_args()
    verify(args.package, args.python.resolve())
