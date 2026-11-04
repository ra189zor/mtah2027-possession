"""Frozen ownership-only TCN blend: one offline tracking input to one possession CSV."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
from threadpoolctl import threadpool_limits
import tcn_predict as t

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('solution/owner_predict.py',)
KIND = 'tcn_owner_blend'
SETTINGS = {'home_weight': .5, 'dead_source': 'corrected_combined', 'event_source': 'corrected_combined',
    'event_reward': 4, 'decoder': {'id': 'fixed_s40_r40_o8', 'mode': 'fixed', 'stop': 40, 'restart': 40, 'switch': 8},
    'epsilon': 1e-6, 'threads': 2}


def component_identities(base, research, tcn):
    return {name: {'manifest_sha256': t.file_digest(bundle['folder']/'manifest.json'),
                   'model_sha256': bundle['manifest']['model_sha256']}
            for name, bundle in [('base', base), ('specialists', research), ('tcn', tcn)]}


def model_identity(components):
    payloads = {name: value['model_sha256'] for name, value in components.items()}
    return hashlib.sha256(json.dumps(payloads, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def runtime_inventory(base, research, tcn):
    hashes = {}
    for bundle in (base, research, tcn):
        for name, digest in bundle['manifest']['runtime_sha256'].items():
            if name in hashes and hashes[name] != digest: raise ValueError('Component runtime identities conflict')
            hashes[name] = digest
    return hashes | {name: t.file_digest(ROOT/name) for name in SOURCES}


def load_owner(path, base, research, tcn):
    """Component loaders must first validate weights, packages and original sources."""
    folder = Path(path).resolve(); manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or manifest.get('kind') != KIND or manifest.get('status') != 'complete':
        raise ValueError('Unsupported/incomplete ownership-only bundle')
    if manifest.get('settings') != SETTINGS: raise ValueError('Frozen ownership-only settings differ')
    deploy = base['manifest']['settings']['deployment']
    if deploy['decoder'] != SETTINGS['decoder'] or deploy['epsilon'] != SETTINGS['epsilon']:
        raise ValueError('Owner candidate requires the validated decoder profile')
    components = component_identities(base, research, tcn)
    if manifest.get('components') != components: raise ValueError('Ownership-only component identities differ')
    if manifest.get('model_sha256') != model_identity(components) or manifest.get('model_identity_kind') != 'sha256_of_component_payload_hashes':
        raise ValueError('Ownership-only model-set identity differs')
    if manifest.get('runtime_sha256') != runtime_inventory(base, research, tcn):
        raise ValueError('Ownership-only runtime inventory differs')
    for name, digest in manifest['runtime_sha256'].items():
        if t.file_digest(ROOT/name) != digest: raise ValueError('Ownership-only runtime source changed: '+name)
    if any(manifest.get('training_matches') != bundle['manifest']['training_matches'] for bundle in (base, research, tcn)):
        raise ValueError('Ownership-only training provenance differs')
    return {'folder': folder, 'manifest': manifest}


def predict(base, research, tcn, bundle, inputs):
    t.validate_tracking(inputs); before = t.array_digest(inputs); started = time.perf_counter()
    config = bundle['manifest']['settings']; settings = base['manifest']['settings']; period = inputs['period']
    torch.set_num_threads(config['threads']); torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.deterministic = True
    with threadpool_limits(limits=config['threads']):
        result = t.current.predict(base, research, inputs, 'combined')
        network = t.tcn_probabilities(tcn, inputs, settings, base['models']['owner_names'])
        # Conditional ownership is the only changed emission; dead/events are retained verbatim.
        h = (1-config['home_weight'])*result['home_probability']+config['home_weight']*network[:, 0]
        decoded = t.current.decode_events(h, result['dead_probability'], period, result['event_probability'],
            config['decoder'], config['event_reward'], settings['switch'], config['epsilon'])
        result.update(labels=decoded['labels'], home_probability=h, path_cost=decoded['path_cost'],
                      timing={'total_pipeline_seconds': time.perf_counter()-started})
    if t.array_digest(inputs) != before: raise ValueError('Ownership-only inference modified inputs')
    t.validate_prediction(result['labels'], period)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT/'models/tcn-owner-v1')
    parser.add_argument('--base-bundle', type=Path, default=ROOT/'models/possession-v1')
    parser.add_argument('--research-bundle', type=Path, default=ROOT/'models/research-specialists-clean')
    parser.add_argument('--tcn-bundle', type=Path, default=ROOT/'models/tcn-blend-v1')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path, help='NPZ containing exactly the six tracking fields')
    source.add_argument('--challenge', action='store_true', help='Read locally installed tactical tracking; no references or video')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); csv, sidecar = t.output_paths(args.output)
    if csv.exists() or sidecar.exists(): parser.error('Output exists; choose a new path')
    base = t.load_bundle(args.base_bundle); research = t.current.load_research(args.research_bundle, base)
    tcn = t.load_tcn(args.tcn_bundle, base, research); bundle = load_owner(args.bundle, base, research, tcn)
    if args.input:
        inputs = t.read_tracking_npz(args.input)
        description = {'kind': 'tracking_npz', 'path': str(args.input.resolve())}
    else:
        sys.path.insert(0, str(ROOT/'more-than-a-hack-2027-main repo')); import games
        paths = games.asset_paths()
        inputs, _ = t.read_challenge(paths['atd_meta'], paths['atd_raw'], games.source_frames(),
            games.GAMES[games.GAME]['pitch'], games.half_video_blocks(games.GAME))
        description = {'kind': 'local_tactical_tracking', 'video_read': False, 'smart_tags_read': False}
    description.update(experimental=True, components=bundle['manifest']['components'],
        fusion=bundle['manifest']['settings'], model_identity_kind=bundle['manifest']['model_identity_kind'])
    torch.set_num_interop_threads(1)
    result = predict(base, research, tcn, bundle, inputs)
    t.write_prediction(csv, result, bundle, KIND, description, inputs['period'])
    print('Ownership-only prediction: '+str(csv))


if __name__ == '__main__': main()
