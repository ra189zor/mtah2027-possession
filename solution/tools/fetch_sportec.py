"""Download selected public Sportec inputs with pinned revision and integrity checks."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
REPO = 'pysport/idsse-data'
BASE = 'https://huggingface.co'


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(match_id, root):
    root.mkdir(parents=True, exist_ok=True)
    pin = root / 'dataset_revision.json'
    if not pin.exists():
        info = get_json(f'{BASE}/api/datasets/{REPO}')
        pin.write_text(json.dumps({'repo': REPO, 'revision': info['sha'],
            'license': info.get('cardData', {}).get('license')}, indent=2), encoding='utf-8')
    revision = json.loads(pin.read_text(encoding='utf-8'))['revision']
    inventory_path = root / f'inventory_{revision}.json'
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
    else:
        inventory = get_json(f'{BASE}/api/datasets/{REPO}/tree/{revision}?recursive=true&limit=100')
        inventory_path.write_text(json.dumps(inventory, indent=2), encoding='utf-8')
    files = [x for x in inventory if x['type'] == 'file' and
        (x['path'] == 'README.md' or (match_id in x['path'] and
         ('matchinformation' in x['path'] or 'positions_raw_observed' in x['path'])))]
    if len(files) != 3:
        raise ValueError(f'Expected metadata, tracking and dataset card for {match_id}; found {len(files)}')
    manifest = {'repo': REPO, 'revision': revision, 'match_id': match_id, 'files': []}
    for entry in files:
        target = root / (match_id if entry['path'] != 'README.md' else '') / entry['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f'{BASE}/datasets/{REPO}/resolve/{revision}/{urllib.parse.quote(entry["path"])}?download=true'
        expected_hash = entry.get('lfs', {}).get('oid')
        if not target.exists():
            partial = target.with_suffix(target.suffix + '.partial')
            if partial.exists():
                raise FileExistsError(f'Incomplete prior download exists: {partial}; inspect before retry')
            print(f'Downloading {entry["path"]}: {entry["size"] / 1024**2:.1f} MiB', flush=True)
            downloaded, last = 0, time.monotonic()
            with urllib.request.urlopen(url, timeout=90) as incoming, partial.open('wb') as outgoing:
                while True:
                    chunk = incoming.read(1024 * 1024)
                    if not chunk:
                        break
                    outgoing.write(chunk)
                    downloaded += len(chunk)
                    if time.monotonic() - last >= 20:
                        print(f'{match_id}: {downloaded / 1024**2:.1f} MiB downloaded', flush=True)
                        last = time.monotonic()
            if downloaded != entry['size']:
                raise ValueError(f'Download size mismatch for {target.name}')
            actual = digest(partial)
            if expected_hash and actual != expected_hash:
                raise ValueError(f'SHA-256 mismatch for {target.name}')
            partial.replace(target)
        actual = digest(target)
        if target.stat().st_size != entry['size'] or (expected_hash and actual != expected_hash):
            raise ValueError(f'Existing file failed integrity verification: {target}')
        manifest['files'].append({'path': str(target.relative_to(ROOT)), 'url': url,
            'bytes': target.stat().st_size, 'sha256': actual, 'publisher_sha256': expected_hash})
        print(f'Verified: {target.name}', flush=True)
    (root / match_id / 'download_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'{match_id}: public metadata/tracking ready; revision {revision}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match', required=True)
    parser.add_argument('--root', type=Path, default=ROOT / 'data/external/sportec')
    args = parser.parse_args()
    fetch(args.match, args.root)
