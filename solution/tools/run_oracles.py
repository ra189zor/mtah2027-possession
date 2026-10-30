"""External-only oracle diagnosis of the frozen Step 6 solver, without training."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'solution/src'), str(ROOT / 'solution/tools'),
               str(ROOT / 'more-than-a-hack-2027-main repo/evaluation'),
               str(ROOT / 'more-than-a-hack-2027-main repo/starter')]
from possession.diagnostics import oracle_predictions
from possession.features import extract_features
from possession.inputs import array_digest, file_digest, read_external
from possession.state_machine import predict
from possession.tag_features import build_tag_features
from possession.validation import corrupt, nearest_team_codes, period_blocks
from benchmark_external import summarize_score
import baseline
import possession_metrics as pm


def paired_draw_scores(gt, prediction, picks):
    """Paired analogue of the official conditional matched-possession bootstrap.

    Matching and total prediction count remain fixed, as in the official routine.
    This is not a new block-resampled sequence or cross-match confidence interval.
    """
    runs = pm.valid_runs(gt)
    pred = pm.fill_abstentions(prediction, runs)
    gt_spells, pred_spells = pm.possession_spells(gt, runs), pm.possession_spells(pred, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    flags = np.vstack([pm.matched_gt_flags(overlaps, len(gt_spells), threshold)
                       for threshold in pm.SPELL_IOUS])
    tp = flags[:, picks].sum(axis=2)
    fp = np.maximum(0, len(pred_spells) - tp)
    fn = len(gt_spells) - tp
    denominator = 2 * tp + fp + fn
    scores = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0)
    return scores.mean(axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match', default='J03WQQ')
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error('--bootstrap must be positive')
    split = json.loads((ROOT / 'solution/configs/validation.json').read_text())
    selection = json.loads(args.selection.read_text(encoding='utf-8'))
    if selection.get('stage') != 'development' or selection.get('status') != 'passed':
        raise ValueError('Expected a completed frozen development selection')
    for name, expected in selection['code_sha256'].items():
        if file_digest(ROOT / name) != expected:
            raise ValueError(f'Frozen solver source changed: {name}')
    inputs = read_external(ROOT / 'data/external/sportec', args.match, split)
    input_hash = array_digest(inputs)
    folder = ROOT / 'data/external/sportec' / args.match / 'prepared'
    audit = json.loads((folder / 'audit.json').read_text())
    if file_digest(folder / 'targets.npz') != audit['targets_sha256']:
        raise ValueError('External target integrity failure')
    with np.load(folder / 'targets.npz', allow_pickle=False) as bundle:
        labels = bundle['labels'].copy()
    if len(labels) != len(inputs['period']):
        raise ValueError('External input/target alignment failure')
    label_hash = array_digest({'labels': labels})
    gt = pm.label_codes(labels)
    runs = pm.valid_runs(gt)
    n_gt = len(pm.possession_spells(gt, runs))
    picks = np.random.RandomState(20261030).randint(0, n_gt, (args.bootstrap, n_gt))
    fc = json.loads((ROOT / 'solution/configs/features.json').read_text())
    settings = selection['profiles'][selection['selected_profile']]
    report = {'match_id': args.match, 'selection': str(args.selection), 'status': 'running',
              'frozen_solver_code_sha256': selection['code_sha256'],
              'diagnostic_code_sha256': {name: file_digest(ROOT / name) for name in (
                  'solution/src/possession/diagnostics.py', 'solution/tools/run_oracles.py')},
              'bootstrap_draws': args.bootstrap,
              'bootstrap_method': 'Paired GT-possession resampling with fixed global matching and prediction count, following the official conditional method; seed replicas share the same possession draws and are averaged before percentiles. Not cross-match uncertainty or full block re-evaluation.',
              'input_fingerprint': input_hash, 'reference_fingerprint': label_hash,
              'true_stats': pm.possession_stats(gt, runs), 'results': [], 'paired_deltas': []}
    for regime in ('clean', 'moderate'):
        seed_draws, seed_scores = [], []
        for seed in (split['seeds'][:1] if regime == 'clean' else split['seeds']):
            start = time.perf_counter()
            noisy = corrupt(inputs, split['regimes'][regime], seed)
            noise_hash = array_digest(noisy)
            features = extract_features(noisy, fc, build_tag_features(None, noisy['period'], {}))
            production = predict(features, noisy['period'], settings)['labels']
            # Remove only stoppage/restart evidence for the owner diagnostic.
            # No reference labels enter this branch; all ownership thresholds stay frozen.
            owner_features = {key: value.copy() for key, value in features.items()}
            for name in ('ball_speed_mps', 'ball_goal_line_distance_m', 'ball_touchline_distance_m'):
                owner_features['valid'][:, features['names'].tolist().index(name)] = False
            owner = predict(owner_features, noisy['period'], settings)['labels']
            predictions = {'solver': production, **oracle_predictions(labels, production, owner, noisy['period'])}
            codes = nearest_team_codes(noisy)
            reference = np.full(len(labels), 'X', dtype='<U1')
            for a, b in period_blocks(noisy['period']):
                reference[a:b] = baseline.smooth(codes[a:b], 50)
            predictions['baseline_50'] = reference
            draws, scores = {}, {}
            for name, prediction in predictions.items():
                predicted = pm.label_codes(prediction)
                metric = pm.evaluate(gt, predicted, bootstrap_n=args.bootstrap)
                stats = pm.possession_stats(pm.fill_abstentions(predicted, runs), runs)
                if name == 'both_true' and (metric['spell_f1_mean'] != 1 or metric['macro_f1'] != 1):
                    raise ValueError('Both-true oracle failed the exact scorer sanity check')
                draws[name] = paired_draw_scores(gt, predicted, picks)
                scores[name] = metric['spell_f1_mean']
                report['results'].append({'regime': regime, 'seed': seed, 'model': name,
                    'noisy_inputs_fingerprint': noise_hash, **summarize_score(metric),
                    'mean_duration_s': stats['mean_duration_s'],
                    'abstain_fraction_on_graded': metric['n_abstain_on_graded'] / metric['n_graded']})
            seed_draws.append(draws)
            seed_scores.append(scores)
            if array_digest(noisy) != noise_hash:
                raise ValueError('Oracle diagnostic changed noisy inputs')
            print(f'{args.match} {regime} seed {seed}: all oracle checks passed '
                  f'({time.perf_counter() - start:.1f} s)', flush=True)
        for name in seed_draws[0]:
            if name == 'baseline_50':
                continue
            comparator = 'baseline_50' if name == 'solver' else 'solver'
            delta = np.mean([row[name] - row[comparator] for row in seed_draws], axis=0)
            low, high = np.percentile(delta, [2.5, 97.5])
            report['paired_deltas'].append({'regime': regime, 'model': name, 'comparator': comparator,
                'spell_f1_delta': float(np.mean([row[name] - row[comparator] for row in seed_scores])),
                'conditional_95_low': float(low), 'conditional_95_high': float(high)})
    if array_digest(inputs) != input_hash or array_digest({'labels': labels}) != label_hash:
        raise ValueError('Oracle diagnostic changed source arrays')
    report['status'] = 'passed'
    destination = ROOT / 'outputs/oracles' / (args.match + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    destination.mkdir(parents=True, exist_ok=False)
    target = destination / 'report.json'
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'PASS: external-only oracle report: {target}', flush=True)


if __name__ == '__main__':
    main()
