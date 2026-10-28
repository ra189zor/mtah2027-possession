"""Prepare canonical challenge inputs without changing original downloads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / 'more-than-a-hack-2027-main repo'
sys.path.insert(0, str(UPSTREAM))
import games


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(downloads, destination):
    expected = {}
    for source in games.SOURCES:
        for asset, path in games.asset_paths(data_dir=str(destination), source=source).items():
            if asset != 'video':
                expected[f'{source}/{Path(path).name}'] = Path(path)
    found = set()
    manifest = {'data_dir': str(destination), 'assets': []}
    for archive in sorted(downloads.glob('*.zip')):
        with zipfile.ZipFile(archive) as bundle:
            for entry in bundle.infolist():
                parts = Path(entry.filename).parts
                key = '/'.join(parts[-2:])
                if entry.is_dir() or key not in expected:
                    continue
                if key in found:
                    raise ValueError(f'Duplicate archive asset: {key}')
                found.add(key)
                target = expected[key]
                target.parent.mkdir(parents=True, exist_ok=True)
                # Extract to a temporary file first. Reading to EOF also checks ZIP CRC.
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temp:
                    temporary = Path(temp.name)
                    with bundle.open(entry) as incoming:
                        shutil.copyfileobj(incoming, temp)
                try:
                    fingerprint = sha256(temporary)
                    if target.exists():
                        if sha256(target) != fingerprint:
                            raise ValueError(f'Refusing to replace different existing asset: {target}')
                        action = 'verified_existing'
                    else:
                        temporary.replace(target)
                        action = 'extracted'
                    manifest['assets'].append({'path': str(target), 'archive': archive.name,
                        'bytes': target.stat().st_size, 'sha256': fingerprint, 'action': action})
                    print(f'{action}: {key}', flush=True)
                finally:
                    temporary.unlink(missing_ok=True)
    missing = sorted(set(expected) - found)
    if missing:
        raise FileNotFoundError(f'Missing archive assets: {missing}')
    for source in games.SOURCES:
        canonical = f'BAR-ATH_{source}'
        candidates = sorted(p for p in downloads.glob(f'{canonical}*.mp4') if p.is_file())
        if len(candidates) != 1:
            raise ValueError(f'Expected exactly one original {source} video; found {len(candidates)}')
        original = candidates[0]
        target = Path(games.asset_paths(data_dir=str(destination), source=source)['video'])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not os.path.samefile(original, target):
                raise ValueError(f'Existing canonical video is not the original hard link: {target}')
            action = 'verified_hard_link'
        else:
            os.link(original, target)
            action = 'created_hard_link'
        manifest['assets'].append({'path': str(target), 'original': str(original),
            'bytes': target.stat().st_size, 'action': action})
        print(f'{action}: {source}/{target.name}', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--downloads', type=Path,
        default=ROOT / 'More Than A Hack 2027 - Initial Challenge - Dataset')
    parser.add_argument('--destination', type=Path, default=UPSTREAM / 'data')
    args = parser.parse_args()
    manifest = prepare(args.downloads.resolve(), args.destination.resolve())
    out = ROOT / 'research/local_data_manifest.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Ready: {len(manifest["assets"])} assets. Local manifest: {out}')


if __name__ == '__main__':
    main()
