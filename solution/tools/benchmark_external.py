"""Score fixed baselines on approved external splits, with labels isolated from inputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'solution/src'))
sys.path.insert(0, str(ROOT / 'more-than-a-hack-2027-main repo/starter'))
sys.path.insert(0, str(ROOT / 'more-than-a-hack-2027-main repo/evaluation'))
from possession.validation import check_inputs, corrupt, nearest_team_codes, period_blocks, summarize_inputs
import baseline
import possession_metrics as pm


def fingerprint(arrays):
    h = hashlib.sha256()
    for key in sorted(arrays):
        h.update(key.encode())
        h.update(str(arrays[key].dtype).encode())
        h.update(str(arrays[key].shape).encode())
        h.update(arrays[key].tobytes())
    return h.hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def summarize_score(result):
    keys = ['spell_f1_mean', 'macro_f1', 'alive_dead_f1', 'n_gt_spells', 'n_pred_spells',
            'fragmentation', 'n_graded', 'spell_bootstrap_low', 'spell_bootstrap_high']
    out = {key: result[key] for key in keys}
    out['dead_f1'] = result['per_class']['D']['f1']
    out['spell_f1_by_iou'] = {str(k): v['f1'] for k, v in result['spell_f1'].items()}
    out['transition_f1_25'] = result['transitions'][25]['f1']
    out['mean_abs_latency_frames_25'] = result['transitions'][25]['mean_abs_latency']
    return out


def benchmark(match_id, split, bootstrap_n, config_path):
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if split not in ('development', 'validation'):
        raise ValueError('Only development and validation evaluation are supported; sealed test is locked')
    if match_id not in config[f'{split}_matches']:
        raise ValueError(f'{match_id} is not in the frozen {split} split')
    folder = ROOT / 'data/external/sportec' / match_id / 'prepared'
    audit = json.loads((folder / 'audit.json').read_text(encoding='utf-8'))
    for name in ('inputs', 'targets'):
        if file_digest(folder / f'{name}.npz') != audit[f'{name}_sha256']:
            raise ValueError(f'Prepared {name} file differs from audited artifact')
    with np.load(folder / 'inputs.npz', allow_pickle=False) as bundle:
        inputs = {key: bundle[key] for key in bundle.files}
    check_inputs(inputs)
    with np.load(folder / 'targets.npz', allow_pickle=False) as bundle:
        labels = bundle['labels'].copy()
    if len(labels) != len(inputs['ball_xy']):
        raise ValueError('Input/target alignment mismatch')
    gt = pm.label_codes(labels)
    original_fingerprint = fingerprint(inputs)
    label_fingerprint = hashlib.sha256(labels.tobytes()).hexdigest()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'outputs/external_validation' / f'{match_id}_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    report = {'match_id': match_id, 'split': split, 'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
        'dataset_revision': audit['revision'], 'python': platform.python_version(), 'numpy': np.__version__,
        'code_sha256': {name: file_digest(ROOT / name) for name in (
            'solution/src/possession/validation.py', 'solution/tools/benchmark_external.py',
            'more-than-a-hack-2027-main repo/evaluation/possession_metrics.py',
            'more-than-a-hack-2027-main repo/starter/baseline.py')},
        'bootstrap_draws': bootstrap_n, 'bootstrap_scope': 'within-match possession resampling; not cross-match uncertainty',
        'original_inputs_fingerprint': original_fingerprint, 'target_fingerprint': label_fingerprint,
        'results': [], 'sealed_test_evaluated': False}
    for regime_name, regime in config['regimes'].items():
        seeds = config['seeds'][:1] if regime_name == 'clean' else config['seeds']
        for seed in seeds:
            start_time = time.perf_counter()
            noisy = corrupt(inputs, regime, seed)
            noisy_hash = fingerprint(noisy)
            repeated = corrupt(inputs, regime, seed)
            if fingerprint(repeated) != noisy_hash:
                raise ValueError('Corruption did not reproduce with the same seed')
            del repeated
            codes = nearest_team_codes(noisy)
            quality = summarize_inputs(noisy)
            for window in config['baseline_windows']:
                prediction = np.full(len(codes), 'X', dtype='<U1')
                for a, b in period_blocks(noisy['period']):
                    prediction[a:b] = baseline.smooth(codes[a:b], window)
                result = pm.evaluate(gt, pm.label_codes(prediction), bootstrap_n=bootstrap_n)
                report['results'].append({'regime': regime_name, 'seed': seed, 'half_window_frames': window,
                    'noisy_inputs_fingerprint': noisy_hash, 'input_quality': quality, **summarize_score(result)})
            del noisy
            print(f'{match_id} {regime_name} seed {seed}: all four baselines scored '
                  f'({time.perf_counter() - start_time:.1f} s); '
                  f'ball missing {quality["ball_missing_fraction"]:.1%}', flush=True)
    if fingerprint(inputs) != original_fingerprint or hashlib.sha256(labels.tobytes()).hexdigest() != label_fingerprint:
        raise ValueError('Original inputs or reference labels were modified')
    report['status'] = 'passed'
    report['limitation'] = 'One-match infrastructure benchmark; no generalization or model-improvement claim.'
    target = output / 'benchmark.json'
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    columns = ['regime', 'seed', 'half_window_frames', 'spell_f1_mean', 'macro_f1', 'dead_f1',
               'n_gt_spells', 'n_pred_spells', 'spell_bootstrap_low', 'spell_bootstrap_high']
    import csv
    with (output / 'scoreboard.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(report['results'])
    print(f'PASS: fixed-baseline external benchmark saved locally: {target}', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match', required=True)
    parser.add_argument('--split', choices=['development', 'validation'], default='development')
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    benchmark(args.match, args.split, args.bootstrap, ROOT / 'solution/configs/validation.json')
