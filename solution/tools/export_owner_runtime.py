"""Export an explicit code/model allowlist; no datasets, predictions or research caches."""
import argparse
from importlib.metadata import distribution
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'solution'))
import owner_predict as runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); output = args.output.resolve()
    if output.exists(): raise FileExistsError('Archive exists; choose a new output path')
    if output.suffix.lower() != '.zip': raise ValueError('Expected a .zip archive')
    base = runtime.t.load_bundle(ROOT/'models/possession-v1')
    research = runtime.t.current.load_research(ROOT/'models/research-specialists-clean', base)
    tcn = runtime.t.load_tcn(ROOT/'models/tcn-blend-v1', base, research)
    owner = runtime.load_owner(ROOT/'models/tcn-owner-v1', base, research, tcn)
    paths = set(owner['manifest']['runtime_sha256']) | {
        'solution/src/possession/__init__.py', 'solution/requirements-lock.txt', 'solution/requirements-tcn-lock.txt',
        'solution/docs/OWNER_INFERENCE.md', 'solution/docs/OWNER_TRANSFER_GAME3.md',
        'solution/docs/THIRD_PARTY_NOTICES.md', 'solution/docs/EXTERNAL_DATA_SOURCES.md',
        'solution/docs/TCN_DEPENDENCIES.json',
        'more-than-a-hack-2027-main repo/LICENSE'}
    for bundle, payload in [(base, 'models.joblib'), (research, 'models.joblib'), (tcn, 'model.pt'), (owner, None)]:
        paths.add((bundle['folder']/'manifest.json').relative_to(ROOT).as_posix())
        if payload: paths.add((bundle['folder']/payload).relative_to(ROOT).as_posix())
    dependencies = {}
    for lock in ('solution/requirements-lock.txt', 'solution/requirements-tcn-lock.txt'):
        for line in (ROOT/lock).read_text().splitlines():
            if line.startswith('#') or '==' not in line: continue
            name, expected = line.strip().split('=='); package = distribution(name)
            if package.version != expected: raise ValueError('Pinned dependency version differs: '+name)
            metadata = package.metadata
            dependencies[name] = dict(name=name, version=expected,
                license_metadata_hint=(metadata.get('License-Expression') or metadata.get('License', '').split('\n')[0])[:160],
                project_urls=metadata.get_all('Project-URL', []))
    generated = {'DEPENDENCIES.json': (json.dumps(list(dependencies.values()), indent=2)+'\n').encode(),
        'README.md': ('# AOM ownership-only runtime candidate\n\n'
            'One offline tracking-to-CSV pipeline. No match data or installed dependencies included.\n\n'
            'Use Python 3.11 with the pinned CPU PyTorch environment, then run from this extracted root:\n\n'
            '```powershell\npython solution/owner_predict.py --input path/to/tracking.npz --output prediction.csv\n```\n\n'
            'Read `solution/docs/OWNER_INFERENCE.md` for input fields, dependency requirements, model identity and limitations. '
            'This is a research candidate, not an uploaded or automatically selected competition submission.\n').encode()}
    inventory = {name: runtime.t.file_digest(ROOT/name) for name in sorted(paths)} | {
        name: runtime.hashlib.sha256(value).hexdigest() for name, value in generated.items()}
    archive_manifest = dict(schema_version=1, kind='ownership_only_runtime', files=inventory,
        candidate_manifest_sha256=runtime.t.file_digest(owner['folder']/'manifest.json'),
        model_identity=owner['manifest']['model_sha256'], producer_sha256=runtime.t.file_digest(Path(__file__)),
        contains_match_data=False, contains_predictions=False, contains_training_code=False,
        dependencies='Install the pinned environment separately; dependency binaries are not redistributed.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in [*sorted(inventory), 'OWNER_RUNTIME_MANIFEST.json']:
            if name == 'OWNER_RUNTIME_MANIFEST.json': data = (json.dumps(archive_manifest, indent=2)+'\n').encode()
            else: data = generated[name] if name in generated else (ROOT/name).read_bytes()
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 4, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16; archive.writestr(info, data)
    output.with_suffix('.sha256').write_text(runtime.t.file_digest(output)+'  '+output.name+'\n', encoding='utf-8')
    print(f'PASS: {len(inventory)} verified files exported to {output}')


if __name__ == '__main__': main()
