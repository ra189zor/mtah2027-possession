"""Build a public runtime archive and separate private draft candidates. Never publish."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ('baseline', 'event2', 'event4')
REPOSITORY = 'https://github.com/ra189zor/mtah2027-possession'
EXTRAS = ('solution/src/possession/__init__.py', 'solution/requirements-lock.txt',
          'solution/verify_release.py', '.gitattributes',
          'more-than-a-hack-2027-main repo/evaluation/validate_submission.py',
          'more-than-a-hack-2027-main repo/LICENSE')
GENERATED = {'README.md', 'TECHNICAL_REPORT.md', 'THIRD_PARTY_NOTICES.md',
             'DATA_PROVENANCE.md', 'DEPENDENCIES.json', 'PUBLIC_MANIFEST.json', 'SHA256SUMS.txt'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def safe_name(name):
    path = PurePosixPath(name)
    if not name or '\\' in name or ':' in name or path.is_absolute() or '..' in path.parts:
        raise ValueError('Unsafe archive path')
    return name


def runtime_paths():
    tree = ast.parse((ROOT/'solution/src/possession/inference.py').read_text(encoding='utf-8'))
    sources = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'RUNTIME_SOURCES' for target in node.targets))
    return set(sources) | set(EXTRAS) | {
        'models/possession-v1/manifest.json', 'models/possession-v1/models.joblib'}


def report(variant):
    description = {'baseline': 'the original decoder without event rewards',
                   'event2': 'directional event rewards with strength 2',
                   'event4': 'directional event rewards with strength 4'}[variant]
    text = (ROOT/'solution/docs/TECHNICAL_REPORT_TEMPLATE.md').read_text(encoding='utf-8')
    values = {'TEAM_NAME': 'Pending registered team name', 'VARIANT': variant,
              'REPOSITORY_URL': REPOSITORY, 'TEAM_STEM': 'mtah2027-possession',
              'DRAFT_NOTICE': '**DRAFT: team name pending; no primary variant selected.**',
              'VARIANT_DESCRIPTION': description}
    for key, value in values.items():
        text = text.replace('{{' + key + '}}', value)
    if '{{' in text:
        raise ValueError('Unresolved report template')
    return text.encode('utf-8')


def zip_bytes(path, files):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(safe_name(name), (2026, 11, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data, compresslevel=9)


def build(output):
    output = output.resolve()
    if not output.is_relative_to(ROOT/'outputs'):
        raise ValueError('Package must stay inside workspace outputs')
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT/'solution/src'))
    sys.path.insert(0, str(ROOT/'more-than-a-hack-2027-main repo'))
    from possession.inference import load_bundle, official_format_check
    import games
    bundle = load_bundle(ROOT/'models/possession-v1')
    files = {name: (ROOT/name).read_bytes() for name in runtime_paths()}
    for name, digest in bundle['manifest']['runtime_sha256'].items():
        if sha(files[name]) != digest:
            raise ValueError('Runtime bytes changed')
    files['TECHNICAL_REPORT.md'] = b'> This public example describes baseline. All three variants are supported; none is selected.\n\n' + report('baseline')
    files['THIRD_PARTY_NOTICES.md'] = (ROOT/'solution/docs/THIRD_PARTY_NOTICES.md').read_bytes()
    files['DATA_PROVENANCE.md'] = (ROOT/'solution/docs/EXTERNAL_DATA_SOURCES.md').read_bytes()
    dependencies = []
    for line in (ROOT/'solution/requirements-lock.txt').read_text().splitlines():
        if '==' not in line or line.startswith('#'):
            continue
        name, version = line.strip().split('==')
        distribution = importlib.metadata.distribution(name)
        if distribution.version != version:
            raise ValueError('Dependency version differs: ' + name)
        metadata = distribution.metadata
        hint = metadata.get('License-Expression') or metadata.get('License', '').split('\n')[0]
        dependencies.append({'name': name, 'version': version, 'license_metadata_hint': hint[:160],
                             'project_urls': metadata.get_all('Project-URL', [])})
    files['DEPENDENCIES.json'] = encoded(dependencies)
    files['README.md'] = ('''# Possession runtime — draft delivery

Team name pending. No primary variant selected. This archive contains fitted models and the exact runtime source; it contains no match data or predictions.

Use Python 3.11 (validated with 3.11.9). From the extracted root:

```text
python -m pip install -r solution/requirements-lock.txt
python solution/verify_release.py
python solution/predict.py --bundle models/possession-v1 --challenge --variant baseline --output outputs/mtah2027-possession.csv --quiet
```

Set MTAH_DATA_DIR to the separately supplied licensed dataset root. Choose baseline, event2 or event4 explicitly; baseline above is an example. Models are included, so no research caches or retraining are required for inference. Cross-platform bit equality is untested. Only load trusted model payloads.

Read TECHNICAL_REPORT.md, DATA_PROVENANCE.md and THIRD_PARTY_NOTICES.md. The public repository is configured as ''' + REPOSITORY + '''; its visibility has not been verified. Code-only clones need this archive's model directory. No new licence is assigned to project-owned code; upstream licences retain their stated scope.
''').encode()
    manifest = {'status': 'draft', 'team_name': None, 'primary_variant': None,
                'source': 'current working-tree bytes; not a committed clone',
                'model_sha256': bundle['manifest']['model_sha256'],
                'files': {name: {'sha256': sha(data), 'bytes': len(data)} for name, data in sorted(files.items())}}
    files['PUBLIC_MANIFEST.json'] = encoded(manifest)
    files['SHA256SUMS.txt'] = ''.join(f'{sha(data)}  {name}\n' for name, data in sorted(files.items())).encode()
    archive = output/'mtah2027-possession-runtime-20261101.zip'
    zip_bytes(archive, files)
    index = {'status': 'draft', 'team_name': None, 'primary_variant': None,
             'published': False, 'submitted': False, 'repository_visibility_verified': False,
             'public_archive': {'name': archive.name, 'sha256': sha(archive.read_bytes())}, 'candidates': {}}
    source = ROOT/'outputs/final_verification/run_20261101/clean_source/runs'
    for variant in VARIANTS:
        original = source/f'challenge_{variant}_a.csv'
        sidecar = json.loads(original.with_suffix('.run.json').read_text())
        data = original.read_bytes()
        if sidecar['reference_arrays_read'] or sidecar['scored'] or sha(data) != sidecar['prediction_sha256']:
            raise ValueError('Candidate provenance failed')
        folder = output/'private_candidates'/variant
        folder.mkdir(parents=True)
        csv = folder/'mtah2027-possession.csv'
        csv.write_bytes(data)
        official_format_check(csv, games.source_frames())
        technical = folder/'mtah2027-possession.md'
        technical.write_bytes(report(variant))
        record = {'variant': variant, 'team_name': None, 'draft': True,
                  'prediction_sha256': sha(data), 'report_sha256': sha(technical.read_bytes()),
                  'format_passed': True, 'reference_arrays_read': False, 'scored': False}
        (folder/'candidate_manifest.json').write_bytes(encoded(record))
        index['candidates'][variant] = record
    (output/'PACKAGE_INDEX.json').write_bytes(encoded(index))
    (output/'HANDOFF.md').write_text('''# Draft delivery handoff

The team name is deliberately pending and all three candidates are preserved, as requested. Filenames use the temporary project slug. No primary is selected, and nothing has been published or submitted.

- Public code/model artifact: `mtah2027-possession-runtime-20261101.zip`. It contains no match inputs or prediction CSVs. Uploading a model release remains a separate user action; model files are ignored by Git.
- Private candidates: `private_candidates/baseline`, `private_candidates/event2`, `private_candidates/event4`. Each contains its algorithm-generated CSV, matching draft Markdown report and checksums. Keep these out of the public code archive.
- Before submitting: recover the registered team name, choose one variant, rename its CSV/report to that name, and update the report team field. Confirm registered members and signed terms; verify public repository access and that its model archive is downloadable. The configured repository is ''' + REPOSITORY + ''' (visibility unverified).
- The final verification tests inference using supplied fitted weights. It does not claim a fresh raw-data-to-model retraining run or challenge accuracy. Baseline performs better on the alternate longer-possession convention; event4 leads the provider-label benchmark. This remains a selection tradeoff.

The official instructions request one CSV, a reproducible public code repository and a short PDF or Markdown report: [submission instructions](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md). Review the [current rules](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md) and signed registration terms before the final user submission. Instructions checked 1 November 2026.
''', encoding='utf-8')
    print('Draft package built; all three candidates preserved; nothing published.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output)
