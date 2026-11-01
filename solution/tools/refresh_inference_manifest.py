"""Record a validator-only runtime update without changing fitted model bytes/settings."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'solution/src'))
from possession.inference import runtime_hashes,load_bundle
from possession.inputs import file_digest


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');temporary.replace(path)


def refresh(folder,archive):
    folder=folder.resolve();archive=archive.resolve();manifest_path=folder/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'));current=runtime_hashes()
    changed=[name for name in current if current[name]!=manifest['runtime_sha256'][name]]
    if changed!=['solution/src/possession/inference.py']:raise ValueError('Only the declared inference validator update may be refreshed')
    if file_digest(folder/'models.joblib')!=manifest['model_sha256']:raise ValueError('Model bytes changed')
    original=subprocess.run(['git','show','HEAD:solution/src/possession/inference.py'],cwd=ROOT,capture_output=True,check=True).stdout
    if hashlib.sha256(original).hexdigest()!=manifest['runtime_sha256'][changed[0]]:raise ValueError('Cannot recover exact prior runtime source from HEAD')
    archive.mkdir(parents=True,exist_ok=False)
    for name in ('manifest.json','models.joblib'):shutil.copyfile(folder/name,archive/name)
    for name in current:
        destination=archive/'runtime_before'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(original if name==changed[0] else (ROOT/name).read_bytes())
    update={'utc':datetime.now(timezone.utc).isoformat(),'reason':'Reject complex-valued tracking/pitch and malformed fps safely; no learned math or settings changed',
        'previous_manifest_sha256':file_digest(archive/'manifest.json'),'changed_files':changed,
        'previous_runtime_sha256':manifest['runtime_sha256'],'model_sha256_unchanged':manifest['model_sha256']}
    manifest['runtime_sha256']=current;manifest['runtime_revision']=manifest.get('runtime_revision',1)+1
    manifest.setdefault('runtime_updates',[]).append(update);write(manifest_path,manifest)
    load_bundle(folder)
    build=folder/'build.json'
    if build.exists():
        progress=json.loads(build.read_text());progress['build_key']['runtime_sha256']=current
        progress.setdefault('runtime_updates',[]).append(update);write(build,progress)
    write(archive/'refresh_record.json',{'status':'passed','update':update,'new_manifest_sha256':file_digest(manifest_path),
        'model_file_sha256':file_digest(folder/'models.joblib'),'settings_sha256_unchanged':manifest['settings_sha256']})
    print(f'PASS: validator-only manifest refresh; fitted weights/settings unchanged: {folder.name}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--archive',type=Path,required=True)
    args=parser.parse_args();refresh(args.bundle,args.archive)
