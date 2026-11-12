"""Verify/download the frozen AOM runtime; optionally extract to a new directory."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import urllib.request
import zipfile

NAME = 'AOM_owner_runtime_v1.zip'
SHA256 = '4373a111611d78e2734bae80897ba650e5109f0cba29facfccafd52e5cf6d6ee'
URL = 'https://github.com/ra189zor/mtah2027-possession/releases/download/aom-owner-v1/' + NAME


def verify(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError('Runtime ZIP checksum differs from selected release')
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        manifest = json.loads(z.read('OWNER_RUNTIME_MANIFEST.json'))
        if len(names) != len(set(names)) or set(names) != set(manifest['files']) | {'OWNER_RUNTIME_MANIFEST.json'}:
            raise ValueError('Archive inventory differs')
        for name, digest in manifest['files'].items():
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe archive path')
            if hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise ValueError('Archived file checksum differs: ' + name)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('outputs/releases') / NAME)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--extract', type=Path)
    args = parser.parse_args()
    if args.extract and args.extract.exists():
        parser.error('Extraction destination exists; choose a new directory')
    if args.download and not args.archive.exists():
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != SHA256:
            raise ValueError('Downloaded checksum differs; nothing saved')
        with args.archive.open('xb') as out:
            out.write(payload)
    verify(args.archive)
    if args.extract:
        args.extract.mkdir(parents=True)
        with zipfile.ZipFile(args.archive) as z:
            z.extractall(args.extract)
    print('PASS: selected runtime ZIP and every archived file verified')


if __name__ == '__main__':
    main()
