"""Recover version-pinned wheels from local pip HTTP bodies without network access."""
import argparse
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import zipfile

from pip._vendor.packaging.tags import sys_tags,parse_tag

ROOT=Path(__file__).resolve().parents[2]


def normalize(name):return re.sub(r'[-_.]+','-',name).lower()


def collect(output):
    output.mkdir(parents=True,exist_ok=True)
    required={normalize(line.split('==')[0]):line.split('==')[1] for line in
              (ROOT/'solution/requirements-lock.txt').read_text().splitlines() if line.strip() and not line.startswith('#')}
    roots=[Path.home()/'AppData/Local/pip/Cache',Path(os.environ.get('LOCALAPPDATA',str(Path.home())))/'pip/Cache']
    supported=set(sys_tags());found={}
    for cache in dict.fromkeys(p.resolve() for p in roots if p.exists()):
        for path in cache.rglob('*'):
            if not path.is_file() or not (path.name.endswith('.body') or path.suffix=='.whl'):continue
            try:
                with path.open('rb') as handle:
                    if handle.read(4)!=b'PK\x03\x04':continue
                with zipfile.ZipFile(path) as archive:
                    metadata_names=[n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
                    if len(metadata_names)!=1:continue
                    metadata=BytesParser().parsebytes(archive.read(metadata_names[0]))
                    name=normalize(metadata['Name']);release=metadata['Version']
                    if required.get(name)!=release or name in found:continue
                    wheel_meta=BytesParser().parsebytes(archive.read(metadata_names[0].rsplit('/',1)[0]+'/WHEEL'))
                    tags=set(t for raw in wheel_meta.get_all('Tag',[]) for t in parse_tag(raw))
                    tag=next((t for t in sys_tags() if t in tags and t in supported),None)
                    if tag is None:continue
                    filename=f'{name.replace("-","_")}-{release}-{tag}.whl'
                    destination=output/filename
                    digest=hashlib.sha256(path.read_bytes()).hexdigest()
                    if destination.exists():
                        if hashlib.sha256(destination.read_bytes()).hexdigest()!=digest:raise ValueError('Conflicting cached wheel')
                    else:shutil.copyfile(path,destination)
                    found[name]={'version':release,'filename':filename,'sha256':digest,'source':'local pip cache'}
                    print(f'Cached wheel verified: {name}=={release}',flush=True)
            except (zipfile.BadZipFile,KeyError,FileNotFoundError):continue
    missing=sorted(set(required)-set(found))
    report={'status':'complete' if not missing else 'missing_wheels','wheels':found,'missing':missing,'network_used':False}
    (output/'wheelhouse.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    if missing:raise RuntimeError('Pinned wheels unavailable locally: '+', '.join(missing))
    (output/'requirements-hashed.txt').write_text(''.join(f'{name}=={found[name]["version"]} --hash=sha256:{found[name]["sha256"]}\n' for name in required),encoding='utf-8')
    print(f'PASS: offline wheelhouse contains all {len(found)} pinned dependencies',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    collect(parser.parse_args().output.resolve())
