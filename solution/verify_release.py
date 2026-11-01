"""Check the public runtime archive's file inventory and SHA-256 values (standard library only)."""
import hashlib
import json
from pathlib import Path


def verify(root):
    manifest=json.loads((root/'PUBLIC_MANIFEST.json').read_text(encoding='utf-8'))
    expected=set(manifest['files'])|{'PUBLIC_MANIFEST.json','SHA256SUMS.txt'}
    for name,record in manifest['files'].items():
        path=(root/name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(root.resolve()):raise ValueError('Unsafe release path')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256'] or path.stat().st_size!=record['bytes']:
            raise ValueError('Release file identity mismatch: '+name)
    sums={}
    for line in (root/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        checksum,name=line.split('  ',1);sums[name]=checksum
    if set(sums)!=expected-{'SHA256SUMS.txt'}:raise ValueError('Checksum inventory mismatch')
    for name,checksum in sums.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=checksum:raise ValueError('Checksum mismatch: '+name)
    print('PASS: public runtime payload and manifest checksums verified')


if __name__=='__main__':verify(Path(__file__).resolve().parents[1])
