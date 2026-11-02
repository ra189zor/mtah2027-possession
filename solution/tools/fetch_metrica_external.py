"""Fetch two public, anonymized external Metrica samples at a pinned revision."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT/'data/external/metrica_public'
REPO = 'metrica-sports/sample-data'
REVISION = 'e706dd506b360d69d9d123d5b8026e7294b13996'


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    url = f'https://api.github.com/repos/{REPO}/git/trees/{REVISION}?recursive=1'
    with urllib.request.urlopen(url, timeout=60) as response:
        inventory = json.load(response)
    files = [v for v in inventory['tree'] if v['type'] == 'blob' and
             (v['path'] in ('README.md', 'documentation/events-definitions.pdf') or
              v['path'].startswith(('data/Sample_Game_1/', 'data/Sample_Game_2/')))]
    records = []
    for item in files:
        relative = item['path']
        target = DEST/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source = f'https://raw.githubusercontent.com/{REPO}/{REVISION}/{relative}'
        if not target.exists():
            temporary = target.with_suffix(target.suffix+'.part')
            with urllib.request.urlopen(source, timeout=120) as response, temporary.open('wb') as handle:
                while chunk := response.read(1024*1024):
                    handle.write(chunk)
            temporary.replace(target)
        contents = target.read_bytes()
        blob = hashlib.sha1(f'blob {len(contents)}\0'.encode()+contents).hexdigest()
        if len(contents) != item['size'] or blob != item['sha']:
            raise ValueError('Downloaded file differs from pinned Git blob: '+relative)
        records.append({'path': relative, 'url': source, 'bytes': len(contents),
                        'git_blob_sha1': blob, 'sha256': hashlib.sha256(contents).hexdigest()})
        print('Verified '+relative, flush=True)
    (DEST/'source_manifest.json').write_text(json.dumps({'repository': REPO, 'revision': REVISION,
        'files': records, 'attribution': 'Metrica Sports public anonymized sample data. README requests source acknowledgement.',
        'scope': 'External games, not the hackathon game. Raw files remain locally ignored.'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
