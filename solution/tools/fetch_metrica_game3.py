"""Fetch the previously unused public Metrica Game 3 at a pinned Git revision."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
REVISION = 'e706dd506b360d69d9d123d5b8026e7294b13996'
DEST = ROOT/'data/external/metrica_game3'
FILES = {
    'Sample_Game_3_events.json': (1709486, '992b4cb502f0b0bc1244174f014f3f7ecb6d38ef'),
    'Sample_Game_3_metadata.xml': (81806, '74b55a80f6aeeb8f8df7fd26e4b3a94e2cc1738d'),
    'Sample_Game_3_tracking.txt': (53443115, '7062b7ceaa44d53345a4625b8f447efc74895b1d')}


def main():
    freeze = ROOT/'outputs/owner_transfer_game3/candidate_freeze.json'
    decision = json.loads(freeze.read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(ROOT/'solution/configs/owner_transfer_game3.json') != decision['config_sha256']:
        raise ValueError('Candidate decision changed before download')
    DEST.mkdir(parents=True, exist_ok=True); records = []
    for name, (size, blob) in FILES.items():
        path = DEST/name
        url = f'https://raw.githubusercontent.com/metrica-sports/sample-data/{REVISION}/data/Sample_Game_3/{name}'
        if not path.exists():
            temporary = path.with_suffix(path.suffix+'.part')
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as handle:
                while chunk := response.read(1024*1024): handle.write(chunk)
            temporary.replace(path)
        raw = path.read_bytes()
        if len(raw) != size or hashlib.sha1(f'blob {size}\0'.encode()+raw).hexdigest() != blob:
            raise ValueError('File differs from pinned source: '+name)
        records.append({'path': name, 'bytes': size, 'git_blob_sha1': blob, 'sha256': digest(path), 'url': url})
        print('Verified '+name, flush=True)
    manifest = {'repository': 'metrica-sports/sample-data', 'revision': REVISION,
        'files': records, 'candidate_freeze_sha256': digest(freeze),
        'attribution': 'Metrica Sports public anonymized sample Game 3; source acknowledgement required for public use.',
        'scope': 'New external evaluation match, not challenge assets or model training data.'}
    (DEST/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__': main()
