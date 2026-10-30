"""Develop fixed fallback profiles, validate the winner once, or export a challenge CSV."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / 'more-than-a-hack-2027-main repo'
sys.path[:0] = [str(ROOT / 'solution/src'), str(ROOT / 'solution/tools'),
                str(UPSTREAM), str(UPSTREAM / 'starter'), str(UPSTREAM / 'evaluation')]
from possession.features import extract_features
from possession.inputs import array_digest, file_digest, read_challenge, read_external
from possession.state_machine import predict, REASONS
from possession.tag_features import build_tag_features
from possession.validation import corrupt, nearest_team_codes, period_blocks
from benchmark_external import summarize_score
import baseline
import games
import load_smart_tagging
import possession_metrics as pm

SOURCES = ('solution/src/possession/state_machine.py', 'solution/src/possession/features.py',
           'solution/src/possession/tag_features.py', 'solution/src/possession/inputs.py',
           'solution/src/possession/validation.py', 'solution/tools/run_fallback.py',
           'solution/tools/benchmark_external.py',
           'more-than-a-hack-2027-main repo/evaluation/possession_metrics.py',
           'more-than-a-hack-2027-main repo/starter/baseline.py',
           'solution/configs/fallback.json', 'solution/configs/features.json',
           'solution/configs/validation.json')


def checked_predict(features, period, config):
    before = array_digest(features)
    result = predict(features, period, config)
    repeated = predict(features, period, config)
    if array_digest(result) != array_digest(repeated) or array_digest(features) != before:
        raise ValueError('Prediction repeatability/input immutability check failed')
    if not np.isin(result['labels'], ['H', 'A', 'D', 'X']).all():
        raise ValueError('Invalid output labels')
    if (result['labels'][period == 0] != 'X').any():
        raise ValueError('Prediction crosses period boundaries')
    return result


def choose_profile(rows, profiles):
    # Equal weight per match and per regime; seeds are averaged inside each regime.
    ranking = []
    for name in profiles:
        own = [row for row in rows if row['model'] == name]
        groups = sorted({(row['match_id'], row['regime']) for row in own})
        means = []
        for key in groups:
            group = [row for row in own if (row['match_id'], row['regime']) == key]
            means.append([np.mean([r[k] for r in group]) for k in ('spell_f1_mean', 'macro_f1')])
        score, macro = np.mean(means, axis=0)
        ranking.append({'profile': name, 'objective_spell_f1': float(score), 'macro_f1': float(macro)})
    ranking.sort(key=lambda row: (-row['objective_spell_f1'], -row['macro_f1'], row['profile']))
    return ranking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['development', 'validation', 'challenge'])
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    split = json.loads((ROOT / 'solution/configs/validation.json').read_text())
    config = json.loads((ROOT / 'solution/configs/fallback.json').read_text())
    feature_config = json.loads((ROOT / 'solution/configs/features.json').read_text())
    hashes = {name: file_digest(ROOT / name) for name in SOURCES}
    profiles = {name: config['common'] | values for name, values in config['profiles'].items()}
    selection = None
    if args.stage != 'development':
        if args.selection is None:
            parser.error('--selection is required for validation/challenge')
        selection = json.loads(args.selection.read_text(encoding='utf-8'))
        if (selection.get('stage') != 'development' or selection.get('status') != 'passed' or
                selection['code_sha256'] != hashes):
            raise ValueError('Selection must be a completed development report for these exact sources/settings')
        winner = selection['selected_profile']
        profiles = {winner: profiles[winner]}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'outputs/fallback' / f'{args.stage}_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    report = {'stage': args.stage, 'status': 'running', 'code_sha256': hashes,
              'profiles': profiles, 'bootstrap_draws': args.bootstrap, 'results': [],
              'selection_report': str(args.selection) if args.selection else None,
              'sealed_test_evaluated': False, 'repeat_exact': True}
    if args.stage == 'challenge':
        paths = games.asset_paths()
        inputs, mapping = read_challenge(paths['atd_meta'], paths['atd_raw'], games.source_frames(),
            games.GAMES[games.GAME]['pitch'], games.half_video_blocks(games.GAME))
        tags = build_tag_features(load_smart_tagging.load(), inputs['period'], mapping)
        before = array_digest(inputs)
        features = extract_features(inputs, feature_config, tags)
        result = checked_predict(features, inputs['period'], profiles[winner])
        if array_digest(inputs) != before:
            raise ValueError('Challenge inputs modified')
        path = output / 'submission.csv'
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle, lineterminator='\n')
            writer.writerow(['frame', 'label'])
            writer.writerows(enumerate(result['labels']))
        table = pm.read_label_csv(str(path))
        error = pm.validate_submission(table, games.source_frames())
        if error:
            raise ValueError(error)
        report.update({'selected_profile': winner, 'submission': str(path.relative_to(ROOT)),
            'submission_sha256': file_digest(path), 'rows': len(result['labels']),
            'label_counts': {code: int((result['labels'] == code).sum()) for code in ('H', 'A', 'D', 'X')},
            'reason_counts': {name: int((result['reason'] == code).sum()) for code, name in REASONS.items()},
            'reference_labels_read': False, 'input_fingerprint': before})
    else:
        for match_id in config[f'{args.stage}_matches']:
            if match_id not in split[f'{args.stage}_matches']:
                raise ValueError('Fallback split is inconsistent with the frozen validation policy')
            inputs = read_external(ROOT / 'data/external/sportec', match_id, split)
            original = array_digest(inputs)
            folder = ROOT / 'data/external/sportec' / match_id / 'prepared'
            audit = json.loads((folder / 'audit.json').read_text())
            if file_digest(folder / 'targets.npz') != audit['targets_sha256']:
                raise ValueError('Target checksum mismatch')
            # Labels are held only in this evaluation harness, never passed to features/predict.
            with np.load(folder / 'targets.npz', allow_pickle=False) as bundle:
                labels = bundle['labels'].copy()
            if len(labels) != len(inputs['period']):
                raise ValueError('Target alignment mismatch')
            gt = pm.label_codes(labels)
            for regime_name, regime in split['regimes'].items():
                seeds = split['seeds'][:1] if regime_name == 'clean' else split['seeds']
                for seed in seeds:
                    start = time.perf_counter()
                    noisy = corrupt(inputs, regime, seed)
                    noise_hash = array_digest(noisy)
                    features = extract_features(noisy, feature_config,
                        build_tag_features(None, noisy['period'], {}))
                    predictions = {}
                    codes = nearest_team_codes(noisy)
                    for window in split['baseline_windows']:
                        pred = np.full(len(codes), 'X', dtype='<U1')
                        for a, b in period_blocks(noisy['period']):
                            pred[a:b] = baseline.smooth(codes[a:b], window)
                        predictions[f'baseline_{window}'] = pred
                    for name, settings in profiles.items():
                        predictions[name] = checked_predict(features, noisy['period'], settings)['labels']
                    for name, prediction in predictions.items():
                        result = pm.evaluate(gt, pm.label_codes(prediction), bootstrap_n=args.bootstrap)
                        report['results'].append({'match_id': match_id, 'regime': regime_name, 'seed': seed,
                            'model': name, 'noisy_inputs_fingerprint': noise_hash,
                            'abstain_fraction_on_graded': result['n_abstain_on_graded'] / result['n_graded'],
                            **summarize_score(result)})
                    if array_digest(noisy) != noise_hash:
                        raise ValueError('Noisy inputs modified')
                    print(f'{match_id} {regime_name} seed {seed}: scored {len(predictions)} models '
                          f'in {time.perf_counter() - start:.1f} s', flush=True)
                    del noisy, features, predictions
            if array_digest(inputs) != original:
                raise ValueError('Original inputs modified')
        if args.stage == 'development':
            ranking = choose_profile(report['results'], profiles)
            report.update({'ranking': ranking, 'selected_profile': ranking[0]['profile']})
        else:
            report['selected_profile'] = winner
    report['status'] = 'passed'
    target = output / 'report.json'
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'PASS: {args.stage} report: {target}', flush=True)


if __name__ == '__main__':
    main()
