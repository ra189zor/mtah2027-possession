"""Tracking-only production inference. No research runner or reference loader imports."""
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import tempfile
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .inputs import array_digest,file_digest
from .validation import check_inputs,period_blocks,INPUT_FIELDS
from .features import extract_features
from .dead_features import extract_dead_features
from .gap_ownership import extract_owner_features
from .decoder import decode
from .switch_events import physical_columns,extract_event_features
from .event_decoder import decode_events

ROOT=Path(__file__).resolve().parents[3]
RUNTIME_SOURCES=('solution/predict.py','solution/src/possession/inference.py',
    'solution/src/possession/inputs.py','solution/src/possession/validation.py',
    'solution/src/possession/features.py','solution/src/possession/tag_features.py',
    'solution/src/possession/dead_features.py','solution/src/possession/dead_model.py',
    'solution/src/possession/gap_ownership.py','solution/src/possession/decoder.py',
    'solution/src/possession/switch_events.py','solution/src/possession/event_decoder.py',
    'more-than-a-hack-2027-main repo/evaluation/possession_metrics.py',
    'more-than-a-hack-2027-main repo/games.py')
PACKAGES=('numpy','pandas','scikit-learn','scipy','joblib','threadpoolctl')


def settings_digest(settings):
    return hashlib.sha256(json.dumps(settings,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def runtime_hashes():return {name:file_digest(ROOT/name) for name in RUNTIME_SOURCES}


def validate_tracking(inputs):
    try:check_inputs(inputs)
    except (TypeError,OverflowError,IndexError) as error:raise ValueError('Invalid tracking field type or shape') from error
    n=len(inputs['period'])
    if n<1 or not (inputs['period']>0).any():raise ValueError('Input needs at least one published playing-period frame')
    real=lambda x:np.issubdtype(x.dtype,np.floating) or np.issubdtype(x.dtype,np.integer)
    fps=np.asarray(inputs['fps'])
    if fps.shape!=() or not real(fps) or not np.isfinite(fps) or fps!=25:raise ValueError('Expected scalar 25 Hz fps')
    pitch=inputs['pitch_m']
    if pitch.shape!=(2,) or not real(pitch) or not np.isfinite(pitch).all() or (pitch<=0).any():raise ValueError('Pitch must contain positive real length/width in metres')
    if set(inputs['team_of_slot'].tolist())!={0,1}:raise ValueError('Anonymous slots must include both teams')
    for key in ('players_xy','ball_xy'):
        if not real(inputs[key]) or np.isinf(inputs[key]).any():raise ValueError('Coordinates must be real numeric metres, with NaN for missing values')
    blocks=period_blocks(inputs['period']);ids=[int(inputs['period'][a]) for a,b in blocks]
    if ids!=sorted(set(ids)):raise ValueError('Each half must form one ordered published block; missing tracking stays NaN inside a half')


def read_tracking_npz(path):
    # This is intentionally not read_external(): no audit, split, target or sibling file is needed.
    with np.load(Path(path),allow_pickle=False) as z:
        if set(z.files)!=INPUT_FIELDS:raise ValueError('Tracking NPZ must contain exactly the six model input fields; labels/targets are rejected')
        inputs={key:z[key] for key in z.files}
    validate_tracking(inputs);return inputs


def load_bundle(path):
    folder=Path(path).resolve();manifest_path=folder/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version')!=1 or manifest.get('status')!='complete' or manifest.get('model_file')!='models.joblib':
        raise ValueError('Unsupported or incomplete model bundle')
    if set(manifest['runtime_sha256'])!=set(RUNTIME_SOURCES):raise ValueError('Unexpected bundle runtime source inventory')
    for name,digest in manifest['runtime_sha256'].items():
        if file_digest(ROOT/name)!=digest:raise ValueError(f'Runtime source changed since packaging: {name}; rebuild the bundle')
    if platform.python_version_tuple()[:2]!=tuple(manifest['python_version'].split('.')[:2]):raise ValueError('Use the bundle Python major/minor version')
    for name in PACKAGES:
        if version(name)!=manifest['packages'][name]:raise ValueError(f'Bundle expects {name}=={manifest["packages"][name]}; use requirements-lock.txt')
    weights=folder/'models.joblib'
    if weights.resolve().parent!=folder or file_digest(weights)!=manifest['model_sha256']:raise ValueError('Model artifact checksum/path mismatch')
    # Only locally built, hash-checked bundles are supported; joblib is a trusted-code format.
    models=joblib.load(weights)
    digest=settings_digest(manifest['settings'])
    if digest!=manifest['settings_sha256'] or models['settings_sha256']!=digest:raise ValueError('Bundle settings changed independently of fitted models')
    if models['training_matches']!=manifest['training_matches']:raise ValueError('Bundle training provenance mismatch')
    for key,width,classes in [('dead',275,[0,1]),('owner',298,[0,1]),('switch',114,[0,1,2])]:
        model=models['heads'][key]
        if model.n_features_in_!=width or not np.array_equal(model.classes_,classes):raise ValueError('Unexpected model feature/class schema')
    return {'folder':folder,'manifest':manifest,'models':models}


def probability(model,matrix,columns=None):
    shape=(len(matrix),len(model.classes_));result=np.empty(shape,dtype=np.float32)
    for a in range(0,len(matrix),16384):
        x=np.asarray(matrix[a:a+16384],dtype=np.float32)
        if columns is not None:x=x[:,columns]
        result[a:a+len(x)]=model.predict_proba(x)
    return result


def predict_tracking(bundle,inputs,variant):
    validate_tracking(inputs);manifest=bundle['manifest'];models=bundle['models'];settings=manifest['settings']
    if variant not in settings['deployment']['variants']:raise ValueError('Choose baseline, event2 or event4 explicitly')
    before=array_digest(inputs);started=time.perf_counter();timing={}
    with threadpool_limits(limits=settings['deployment']['threads']):
        basic=extract_features(inputs,settings['features'])
        dead=extract_dead_features(inputs,settings['features'],settings['dead_features'],basic)
        owner=extract_owner_features(basic,dead['values'],dead['names'],inputs['period'],settings['owner_features'])
        if dead['names'].tolist()!=models['dead_names'] or owner['names'].tolist()!=models['owner_names']:
            raise ValueError('Extracted feature schema differs from the fitted model schema')
        timing['feature_seconds']=time.perf_counter()-started;tick=time.perf_counter()
        d=probability(models['heads']['dead'],dead['values'])[:,1]
        h=probability(models['heads']['owner'],owner['values'])[:,1]
        strength=settings['deployment']['variants'][variant];q=None
        if strength:
            physical,names=physical_columns(owner['values'],owner['names'])
            x,event_names=extract_event_features(physical,names,h,d,inputs['period'],settings['switch'])
            if event_names!=models['event_names']:raise ValueError('Extracted switch feature schema differs')
            q=probability(models['heads']['switch'],x)
        timing['model_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
        deployment=settings['deployment']
        if strength:
            decoded=decode_events(h,d,inputs['period'],q,deployment['decoder'],strength,settings['switch'],deployment['epsilon'])
        else:decoded=decode(h,d,inputs['period'],deployment['decoder'],deployment['epsilon'])
        timing['decode_seconds']=time.perf_counter()-tick
    if array_digest(inputs)!=before:raise ValueError('Inference modified tracking inputs')
    validate_prediction(decoded['labels'],inputs['period'])
    return {'labels':decoded['labels'],'home_probability':h,'dead_probability':d,'event_probability':q,
            'input_fingerprint':before,'timing':timing,'path_cost':decoded['path_cost']}


def validate_prediction(labels,period):
    if labels.shape!=period.shape or not np.isin(labels,['H','A','D','X']).all():raise ValueError('Invalid prediction schema')
    if (labels[period>0]=='X').any() or (labels[period==0]!='X').any():raise ValueError('Prediction violates published periods')


def official_format_check(path,n):
    # Importing this module only defines metric functions. No reference CSV or scoring call is used.
    import sys
    sys.path.insert(0,str(ROOT/'more-than-a-hack-2027-main repo/evaluation'))
    import possession_metrics as pm
    error=pm.validate_submission(pm.read_label_csv(str(path)),n)
    if error:raise ValueError(error)


def output_paths(path):
    path=Path(path).resolve()
    if path.suffix.lower()!='.csv':raise ValueError('Prediction output must be a .csv file')
    return path,path.with_suffix('.run.json')


def write_prediction(path,result,bundle,variant,input_description,period,overwrite=False):
    csv_path,report_path=output_paths(path);validate_prediction(result['labels'],period)
    if not overwrite and (csv_path.exists() or report_path.exists()):raise FileExistsError('Prediction output or sidecar already exists; choose another path or --overwrite')
    csv_path.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'complete','variant':variant,'rows':len(period),'input':input_description,
        'input_fingerprint':result['input_fingerprint'],'bundle_kind':bundle['manifest']['kind'],
        'bundle_manifest_sha256':file_digest(bundle['folder']/'manifest.json'),
        'model_sha256':bundle['manifest']['model_sha256'],'training_matches':bundle['manifest']['training_matches'],
        'runtime_sha256':runtime_hashes(),'timing':result['timing'],
        'reference_arrays_read':False,'scored':False,'official_format_valid':True,
        'notes':'Inference artifact, not an external or competition quality score. Offline future context; explicit variant choice.'}
    temporaries=[]
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',newline='',dir=csv_path.parent,delete=False,suffix='.csv.tmp') as handle:
            temporary=Path(handle.name);temporaries.append(temporary)
            writer=csv.writer(handle,lineterminator='\n');writer.writerow(['frame','label']);writer.writerows(enumerate(result['labels']))
        official_format_check(temporary,len(period));report['prediction_sha256']=file_digest(temporary)
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=csv_path.parent,delete=False,suffix='.json.tmp') as handle:
            temporary_report=Path(handle.name);temporaries.append(temporary_report)
            handle.write(json.dumps(report,indent=2)+'\n')
        if not overwrite and (csv_path.exists() or report_path.exists()):raise FileExistsError('Output appeared during inference; refusing replacement')
        temporary.replace(csv_path);temporary_report.replace(report_path)
    finally:
        for temporary in temporaries:
            if temporary.exists():temporary.unlink()
    return report
