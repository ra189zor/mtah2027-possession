"""Extract, cache and verify label-free features; emit only aggregate audit output."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / 'more-than-a-hack-2027-main repo'
sys.path.insert(0, str(ROOT / 'solution/src'))
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(UPSTREAM / 'starter'))
from possession.features import extract_features
from possession.inputs import array_digest, file_digest, read_challenge, read_external
from possession.tag_features import build_tag_features
from possession.validation import corrupt
import games
import load_smart_tagging


def extract_and_check(inputs, config, tags):
    before = array_digest(inputs)
    first = extract_features(inputs, config, tags)
    repeated = extract_features(inputs, config, tags)
    if array_digest(first) != array_digest(repeated):
        raise ValueError('Feature extraction is not reproducible')
    if array_digest(inputs) != before:
        raise ValueError('Input arrays were modified')
    permutation = np.arange(inputs['players_xy'].shape[1])
    for team in (0, 1):
        slots = np.flatnonzero(inputs['team_of_slot'] == team)
        permutation[slots] = slots[::-1]
    permuted = dict(inputs, players_xy=inputs['players_xy'][:, permutation],
                    team_of_slot=inputs['team_of_slot'][permutation])
    alternative = extract_features(permuted, config, tags)
    if not np.array_equal(first['valid'], alternative['valid']):
        raise ValueError('Feature validity depends on slot ordering')
    if not np.allclose(first['values'], alternative['values'], rtol=1e-5, atol=2e-4):
        raise ValueError('Feature values depend on slot ordering')
    if first['values'].dtype != np.float32 or not np.isfinite(first['values']).all():
        raise ValueError('Nonfinite or non-float32 feature cache')
    if first['valid'][inputs['period'] == 0].any():
        raise ValueError('Features are valid outside published periods')
    return first


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['challenge', 'external', 'all'], default='all')
    args = parser.parse_args()
    config_path = ROOT / 'solution/configs/features.json'
    config = json.loads(config_path.read_text())
    split = json.loads((ROOT / 'solution/configs/validation.json').read_text())
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = ROOT / 'outputs/features' / stamp
    destination.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'config_sha256': file_digest(config_path),
        'config': config, 'references_read': False, 'datasets': [],
        'code_sha256': {name: file_digest(ROOT / name) for name in (
            'solution/src/possession/features.py', 'solution/src/possession/tag_features.py',
            'solution/src/possession/inputs.py', 'solution/tools/extract_features.py')}}
    jobs = ['BAR-ATH'] if args.source == 'challenge' else []
    if args.source == 'all':
        jobs.append('BAR-ATH')
    if args.source in ('external', 'all'):
        jobs += ['J03WPY', 'J03WQQ']
    for match_id in jobs:
        start = time.perf_counter()
        print(f'{match_id}: reading model inputs, without reference labels', flush=True)
        if match_id == 'BAR-ATH':
            paths = games.asset_paths()
            inputs, team_mapping = read_challenge(paths['atd_meta'], paths['atd_raw'],
                games.source_frames(), games.GAMES[games.GAME]['pitch'], games.half_video_blocks(games.GAME))
            events = load_smart_tagging.load()
            variants = [('automatic', inputs, build_tag_features(events, inputs['period'], team_mapping))]
        else:
            inputs = read_external(ROOT / 'data/external/sportec', match_id, split)
            variants = [('clean', inputs, build_tag_features(None, inputs['period'], {}))]
            if match_id in split['development_matches']:
                for regime in ('moderate', 'severe'):
                    noisy = corrupt(inputs, split['regimes'][regime], 17)
                    variants.append((regime + '_seed17', noisy, build_tag_features(None, noisy['period'], {})))
        for scenario, values, tags in variants:
            print(f'{match_id} {scenario}: extracting and checking features', flush=True)
            features = extract_and_check(values, config, tags)
            cache = destination / f'{match_id}_{scenario}.npz'
            np.savez_compressed(cache, **features)
            report['datasets'].append({'match_id': match_id, 'scenario': scenario,
                'rows': len(features['values']), 'features': len(features['names']),
                'feature_names': features['names'].tolist(),
                'feature_array_mib': round((features['values'].nbytes + features['valid'].nbytes) / 1024**2, 2),
                'valid_fraction_by_feature': dict(zip(features['names'].tolist(),
                    features['valid'].mean(axis=0).tolist())),
                'input_fingerprint': array_digest(values), 'feature_fingerprint': array_digest(features),
                'cache_sha256': file_digest(cache), 'cache': str(cache.relative_to(ROOT)),
                'repeat_exact': True, 'slot_permutation_check': True, 'finite_values': True})
            print(f'{match_id} {scenario}: {features["values"].shape}; checks passed', flush=True)
            del features
        print(f'{match_id}: finished in {time.perf_counter() - start:.1f} seconds', flush=True)
        del variants, inputs
    report['status'] = 'passed'
    (destination / 'verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'PASS: label-free feature audit saved locally: {destination / "verification.json"}', flush=True)


if __name__ == '__main__':
    main()
