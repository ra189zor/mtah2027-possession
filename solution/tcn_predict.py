"""One offline tracking-to-CSV pipeline combining the current model and a frozen TCN."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'solution/src'))
import research_predict as current
from possession.inference import (load_bundle, read_tracking_npz, validate_tracking,
    validate_prediction, output_paths, write_prediction)
from possession.inputs import array_digest, file_digest, read_challenge
from possession.temporal_network import (PossessionTCN, sampled_indices, infer_sampled,
    upsample_probabilities, combine_probabilities)

SOURCES = ('solution/tcn_predict.py', 'solution/src/possession/temporal_network.py')


def load_tcn(path, base, research):
    folder = Path(path).resolve(); manifest = json.loads((folder/'manifest.json').read_text())
    if manifest.get('kind') != 'tcn_joint_blend' or manifest.get('status') != 'complete':
        raise ValueError('Unsupported/incomplete TCN bundle')
    if manifest['base_manifest_sha256'] != file_digest(base['folder']/'manifest.json') or manifest['research_manifest_sha256'] != file_digest(research['folder']/'manifest.json'):
        raise ValueError('TCN requires its recorded base and specialist bundles')
    if manifest['torch_version'] != version('torch'): raise ValueError('TCN PyTorch version differs')
    if set(manifest['runtime_sha256']) != set(SOURCES): raise ValueError('TCN runtime inventory differs')
    for name, digest in manifest['runtime_sha256'].items():
        if file_digest(ROOT/name) != digest: raise ValueError('TCN runtime source differs: '+name)
    if file_digest(folder/'model.pt') != manifest['model_sha256']: raise ValueError('TCN model hash differs')
    payload = torch.load(folder/'model.pt', map_location='cpu', weights_only=True)
    if payload['training_matches'] != manifest['training_matches'] or payload['config'] != manifest['config']:
        raise ValueError('TCN training/configuration provenance differs')
    model = PossessionTCN(payload['config']); model.load_state_dict(payload['state_dict']); model.eval()
    return {'folder': folder, 'manifest': manifest, 'model': model, 'normalizer': payload['normalizer']}


def tcn_probabilities(bundle, inputs, settings, feature_names):
    validate_tracking(inputs); config = bundle['manifest']['config']; period = inputs['period']
    basic = current.extract_features(inputs, settings['features'])
    dead = current.extract_dead_features(inputs, settings['features'], settings['dead_features'], basic)
    owner = current.extract_owner_features(basic, dead['values'], dead['names'], period, settings['owner_features'])
    if owner['names'].tolist() != feature_names: raise ValueError('TCN physical feature schema differs')
    indices = sampled_indices(period, config['sample_stride_frames'])
    sampled = infer_sampled(bundle['model'], owner['values'][indices], period[indices], bundle['normalizer'], config)
    return upsample_probabilities(sampled, indices, period)


def predict(base, research, bundle, inputs):
    before = array_digest(inputs); tick = time.perf_counter(); config = bundle['manifest']['config']
    torch.set_num_threads(config['threads']); torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.deterministic = True
    settings = base['manifest']['settings']
    with threadpool_limits(limits=config['threads']):
        result = current.predict(base, research, inputs, 'combined')
        network = tcn_probabilities(bundle, inputs, settings, base['models']['owner_names'])
        h, d = combine_probabilities(result['home_probability'], result['dead_probability'],
            network[:, 0], network[:, 1], inputs['period'], config['blend'])['tcn_blend']
        deployment = settings['deployment']
        decoded = current.decode_events(h, d, inputs['period'], result['event_probability'],
            deployment['decoder'], 4, settings['switch'], deployment['epsilon'])
        result.update(labels=decoded['labels'], home_probability=h, dead_probability=d,
            path_cost=decoded['path_cost'], timing={'total_pipeline_seconds': time.perf_counter()-tick})
    if array_digest(inputs) != before: raise ValueError('TCN inference modified inputs')
    validate_prediction(result['labels'], inputs['period'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-bundle', type=Path, default=ROOT/'models/possession-v1')
    parser.add_argument('--research-bundle', type=Path, default=ROOT/'models/research-specialists-clean')
    parser.add_argument('--tcn-bundle', type=Path, default=ROOT/'models/tcn-blend-v1')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path); source.add_argument('--challenge', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); csv, sidecar = output_paths(args.output)
    if csv.exists() or sidecar.exists(): parser.error('Output exists; choose a new path')
    base = load_bundle(args.base_bundle); research = current.load_research(args.research_bundle, base)
    bundle = load_tcn(args.tcn_bundle, base, research)
    if args.input:
        inputs = read_tracking_npz(args.input)
        description = {'kind': 'tracking_npz', 'path': str(args.input.resolve())}
    else:
        sys.path.insert(0, str(ROOT/'more-than-a-hack-2027-main repo')); import games
        paths = games.asset_paths()
        inputs, _ = read_challenge(paths['atd_meta'], paths['atd_raw'], games.source_frames(),
            games.GAMES[games.GAME]['pitch'], games.half_video_blocks(games.GAME))
        description = {'kind': 'local_tactical_tracking', 'video_read': False, 'smart_tags_read': False}
    description.update(experimental=True, base_manifest_sha256=bundle['manifest']['base_manifest_sha256'],
        research_manifest_sha256=bundle['manifest']['research_manifest_sha256'],
        tcn_runtime_sha256=bundle['manifest']['runtime_sha256'])
    result = predict(base, research, bundle, inputs)
    write_prediction(csv, result, bundle, 'tcn_joint_blend', description, inputs['period'])
    print('TCN blend prediction: '+str(csv))


if __name__ == '__main__': main()
