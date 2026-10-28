"""Rebuild unchanged official baselines twice, validate, compare and score locally."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / 'more-than-a-hack-2027-main repo'
sys.path.insert(0, str(UPSTREAM / 'evaluation'))
import possession_metrics as pm

NAMES = ('baseline_raw', 'baseline_smooth25', 'baseline_smooth50', 'baseline_smooth125')
PUBLIC_SCORES = {
    'baseline_raw': (0.5972, 0.0378),
    'baseline_smooth25': (0.6200, 0.2868),
    'baseline_smooth50': (0.6207, 0.3256),
    'baseline_smooth125': (0.5812, 0.4242),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def run(command, log, environment):
    start = time.perf_counter()
    with log.open('w', encoding='utf-8') as handle:
        process = subprocess.run(command, cwd=ROOT, env=environment,
            stdout=handle, stderr=subprocess.STDOUT)
    code = process.returncode
    result = {'wall_seconds': round(time.perf_counter() - start, 3),
              'exit_code': code, 'log': str(log.relative_to(ROOT))}
    if code:
        raise RuntimeError(f'Command exited {code}; inspect local log {log}')
    return result


def read_valid(path):
    table = pm.read_label_csv(path)
    reason = pm.validate_submission(table, 199319)
    if reason:
        raise ValueError(f'{path.name}: {reason}')
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=UPSTREAM / 'data')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = (args.out or ROOT / 'outputs/baseline_reproduction' / stamp).resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, MTAH_DATA_DIR=str(args.data_dir.resolve()), PYTHONUTF8='1')
    reference_paths = {name: UPSTREAM / 'evaluation/results' / f'{name}.csv' for name in NAMES}
    tracked_inputs = [UPSTREAM / 'starter/baseline.py', UPSTREAM / 'games.py',
        UPSTREAM / 'evaluation/possession_metrics.py', UPSTREAM / 'evaluation/run_evaluation.py',
        UPSTREAM / 'evaluation/validate_submission.py', *reference_paths.values(),
        args.data_dir / 'tactical/BAR-ATH_tactical_atd.xml',
        args.data_dir / 'tactical/BAR-ATH_tactical_atd.txt']
    fingerprints = {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path):
                    sha256(path) for path in tracked_inputs}
    report = {'status': 'running', 'python': platform.python_version(),
        'platform': platform.platform(), 'dependencies': {
            name: importlib.metadata.version(name) for name in ('numpy', 'pandas')},
        'source_and_input_sha256': fingerprints, 'runs': [], 'comparisons': {}}
    for number in (1, 2):
        directory = out / f'run{number}'
        print(f'Building official baselines from tracking: run {number}/2', flush=True)
        memory_report = out / f'build{number}_memory.json'
        measurement = run([sys.executable, str(Path(__file__).with_name('_baseline_worker.py')),
            '--out', str(directory), '--memory-report', str(memory_report)],
            out / f'build{number}.log', env)
        measurement.update(json.loads(memory_report.read_text(encoding='utf-8')))
        report['runs'].append(measurement)
        print(f'Run {number}: {measurement["wall_seconds"]} s; '
              f'peak working set {measurement["peak_working_set_mib"]} MiB', flush=True)
    for name, reference in reference_paths.items():
        first, second = out / 'run1' / f'{name}.csv', out / 'run2' / f'{name}.csv'
        expected, generated, repeated = (read_valid(path) for path in (reference, first, second))
        if not np.array_equal(expected['frame'].to_numpy(), generated['frame'].to_numpy()):
            raise ValueError(f'{name}: frame index differs from reference')
        mismatches = int(np.count_nonzero(pm.label_codes(expected) != pm.label_codes(generated)))
        repeat_mismatches = int(np.count_nonzero(pm.label_codes(generated) != pm.label_codes(repeated)))
        first_hash, second_hash = sha256(first), sha256(second)
        if mismatches or repeat_mismatches or first_hash != second_hash:
            raise ValueError(f'{name}: reference mismatches={mismatches}; '
                             f'repeated-label mismatches={repeat_mismatches}; '
                             f'bytes reproducible={first_hash == second_hash}')
        run([sys.executable, str(UPSTREAM / 'evaluation/validate_submission.py'), str(first)],
            out / f'{name}_validation.log', env)
        report['comparisons'][name] = {'rows': len(generated), 'label_mismatches': mismatches,
            'repeat_label_mismatches': repeat_mismatches, 'sha256': first_hash,
            'repeat_sha256': second_hash, 'reference_sha256': sha256(reference),
            'same_bytes_as_reference': first.read_bytes() == reference.read_bytes(),
            'same_text_after_newline_normalization': first.read_text() == reference.read_text()}
        print(f'{name}: zero label mismatches; repeat SHA-256 identical; validator OK', flush=True)
    print('Scoring regenerated files on public correctness window (1000 bootstrap draws)', flush=True)
    report['evaluation'] = run([sys.executable, str(UPSTREAM / 'evaluation/run_evaluation.py'),
        '--results', str(out / 'run1'), '--out', str(out / 'evaluation'),
        '--no-detail', '--bootstrap', '1000'], out / 'evaluation.log', env)
    board = pd.read_csv(out / 'evaluation/scoreboard.csv').set_index('submission')
    for name, (macro, spell) in PUBLIC_SCORES.items():
        result = board.loc[name]
        if abs(result['macro_f1'] - macro) > 0.00005 or abs(result['spell_f1_mean'] - spell) > 0.00005:
            raise ValueError(f'{name}: public regression score differs')
        report['comparisons'][name].update(macro_f1=float(result['macro_f1']),
                                           spell_f1=float(result['spell_f1_mean']))
    # Confirm neither the supplied implementation, reference CSVs nor raw tracking changed.
    if any(sha256(path) != fingerprint for path, fingerprint in zip(tracked_inputs, fingerprints.values())):
        raise ValueError('Source/reference/input changed during baseline reproduction')
    report.update(status='passed', data_dir=str(args.data_dir.resolve()),
                  reference_files_unchanged=True, full_match_private_evaluation=False)
    report_path = out / 'verification.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'PASS: all four baselines reproduced. Local verification report: {report_path}', flush=True)


if __name__ == '__main__':
    main()
