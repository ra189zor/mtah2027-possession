"""Independent final artifact audit; never report competition match statistics."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'solution/src'))
from possession.inference import load_bundle,official_format_check


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def without_validator(source):
    module=ast.parse(source)
    module.body=[node for node in module.body if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) or node.name!='validate_tracking']
    return ast.dump(module,include_attributes=False)


def audit(out):
    report=read(out/'report.json');snapshot=out/'clean_source';inventory=read(out/'snapshot.json')
    if report['status']!='passed' or report['challenge_quality_scoring_performed'] or report['competition_numeric_results_reported']:
        raise ValueError('Incomplete final verification or unexpected competition reporting')
    if digest(out/'snapshot.json')!=report['snapshot_manifest_sha256']:raise ValueError('Snapshot inventory changed')
    checked=0
    for name,checksum in inventory['files'].items():
        if digest(snapshot/name)!=checksum or digest(ROOT/name)!=checksum:raise ValueError('Source snapshot drift')
        checked+=1
    if any((snapshot/name).exists() for name in ('solution/tools','outputs','data','.git')):
        raise ValueError('Clean source snapshot acquired a research/data/Git dependency')
    for name,old_folder,current_folder in [('final','prior_final_bundle','possession-v1'),('heldout','prior_heldout_bundle','heldout-J03WN1-v1')]:
        previous=read(out/old_folder/'manifest.json');current=load_bundle(ROOT/'models'/current_folder)['manifest']
        copied=read(snapshot/'models'/name/'manifest.json')
        if current!=copied:raise ValueError('Tested bundle differs from current bundle')
        if previous['model_sha256']!=current['model_sha256'] or previous['settings_sha256']!=current['settings_sha256']:
            raise ValueError('Learned model weights or settings changed')
        for path in (ROOT/'models'/current_folder/'models.joblib',snapshot/'models'/name/'models.joblib',out/old_folder/'models.joblib'):
            if digest(path)!=previous['model_sha256']:raise ValueError('Model artifact changed')
            checked+=1
        old_source=(out/old_folder/'runtime_before/solution/src/possession/inference.py').read_text(encoding='utf-8')
        new_source=(ROOT/'solution/src/possession/inference.py').read_text(encoding='utf-8')
        if without_validator(old_source)!=without_validator(new_source):raise ValueError('Runtime logic outside input validation changed')
        changed=[path for path in previous['runtime_sha256'] if previous['runtime_sha256'][path]!=current['runtime_sha256'][path]]
        if changed!=['solution/src/possession/inference.py']:raise ValueError('Unexpected runtime update inventory')
    wheels=read(out/'wheelhouse/wheelhouse.json');installation=read(out/'install_report.json')
    if wheels['status']!='complete' or wheels['network_used']:raise ValueError('Incomplete offline dependencies')
    for item in installation['install']:
        name=item['metadata']['name'].lower().replace('_','-');wheel=wheels['wheels'][name]
        if not item['download_info']['url'].startswith('file:') or item['download_info']['archive_info']['hashes']['sha256']!=wheel['sha256']:
            raise ValueError('Fresh environment did not install the recorded cached wheel')
        if digest(out/'wheelhouse'/wheel['filename'])!=wheel['sha256']:raise ValueError('Cached wheel drift')
        checked+=1
    environment=read(out/'fresh_environment.json')
    if not environment['isolated_system_site_packages'] or environment['network_used'] or 'No broken requirements' not in environment['pip_check']:
        raise ValueError('Fresh environment is not clean/consistent')
    config=(out/'fresh_env/pyvenv.cfg').read_text().lower()
    if 'include-system-site-packages = false' not in config:raise ValueError('Fresh environment shares system packages')
    held=read(snapshot/'models/heldout/manifest.json');match=held['held_out']
    if match in held['training_matches']:raise ValueError('Held-out match is in training')
    for result in report['results']:
        name=result['name'];record=read(snapshot/'runs'/f'{name}.guard.json')
        if result['kind']=='negative':
            if record['status']!='failed' or record['forbidden_access_attempts']:raise ValueError('Negative CLI case was not a clean rejection')
            continue
        if result['kind']=='guard_selftest':
            if record['status']!='passed' or [x['reason'] for x in record['forbidden_access_attempts']]!=['reference_artifact','research_runner']:
                raise ValueError('Access guard self-test failed')
            continue
        if record['status']!='passed' or record['forbidden_access_attempts'] or not record['standalone_format_passed']:
            raise ValueError('Unexpected runtime/reference/format result')
        path=snapshot/'runs'/f'{name}.csv';sidecar=read(path.with_suffix('.run.json'))
        if sidecar['reference_arrays_read'] or sidecar['scored'] or digest(path)!=sidecar['prediction_sha256']:
            raise ValueError('Prediction/sidecar identity or scoring violation')
        checked+=1
        if result['kind']=='external':
            condition=name.rsplit('_',1)[0];variant=result['variant']
            if variant=='baseline':expected=ROOT/'outputs/nested_validation/run_20261030/evaluation'/match/f'{match}_{condition}_17_decoded.csv'
            else:expected=ROOT/'outputs/switch_events/run_20261101/outer'/match/f'{condition}_17_event_reward_{2 if variant=="event2" else 4}.csv'
            if digest(path)!=digest(expected):raise ValueError('Fresh-source prediction differs from frozen held-out output')
    # Recheck the written competition CSV contract independently, without describing/scoring it.
    sys.path.insert(0,str(ROOT/'more-than-a-hack-2027-main repo'));import games
    for variant in ('baseline','event2','event4'):
        a=snapshot/'runs'/f'challenge_{variant}_a.csv';b=snapshot/'runs'/f'challenge_{variant}_b.csv'
        if a.read_bytes()!=b.read_bytes():raise ValueError('Challenge repeatability drift')
        try:
            official_format_check(a,games.source_frames());official_format_check(b,games.source_frames())
        except Exception as error:raise ValueError('Challenge structural acceptance failed; no match statistics reported') from None
    external=[r['resource'] for r in report['results'] if r['kind']=='external']
    result={'status':'passed','audit_source_sha256':digest(Path(__file__)),'verification_report_sha256':digest(out/'report.json'),
        'artifact_identity_checks':checked,'fresh_dependency_wheels':len(installation['install']),
        'learned_weights_and_settings_unchanged':True,'runtime_ast_unchanged_except_input_validator':True,
        'heldout_raw_predictions_byte_exact':12,'challenge_contract_passed':True,'challenge_repeats_byte_exact':True,
        'competition_quality_scoring_performed':False,'competition_numeric_results_reported':False,
        'external_resource_range':{'wall_seconds':[min(x['wall_seconds'] for x in external),max(x['wall_seconds'] for x in external)],
            'peak_working_set_MiB':[min(x['peak_working_set_bytes'] for x in external)/1024**2,max(x['peak_working_set_bytes'] for x in external)/1024**2]}}
    (out/'audit_report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('PASS: independent final audit; clean environment, exact external reproduction, unchanged learned logic and unscored challenge acceptance',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    audit(parser.parse_args().output.resolve())
