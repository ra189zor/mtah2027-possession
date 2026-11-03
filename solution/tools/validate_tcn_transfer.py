"""Frozen two-provider transfer comparison; no model training or full-possession scoring."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'solution/src'), str(ROOT/'solution')]
import tcn_predict as runtime
from possession.metrica_external import read_public_tracking
from possession.transfer_validation import partial_reference, score_subset, labelled_intervals, paired_interval_draws
from possession.validation import corrupt, summarize_inputs
from possession.robustness import contextual_corrupt

SOURCE = ROOT/'data/external/metrica_public'
PREVIOUS = ROOT/'outputs/stoppage_research/metrica_transfer_20261102_v2'
CONFIG = ROOT/'solution/configs/tcn_transfer.json'
SOURCE_FILES = [CONFIG, Path(__file__), ROOT/'solution/src/possession/transfer_validation.py',
    ROOT/'solution/src/possession/metrica_external.py', ROOT/'solution/src/possession/validation.py',
    ROOT/'solution/src/possession/robustness.py', ROOT/'solution/configs/validation.json',
    ROOT/'solution/configs/nested_validation.json', ROOT/'solution/tcn_predict.py']


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def cases(config):
    return [(condition, seed) for condition in config['conditions']
            for seed in (config['seeds'][:1] if condition == 'native' else config['seeds'])]


def noisy_inputs(inputs, condition, seed, protocol):
    regimes = protocol['noise_regimes']; extra = protocol['contextual']
    masks = {k: np.zeros(len(inputs['period']), bool) for k in ('ghost', 'wrong_team')}
    if condition == 'contextual':
        noisy, more, _ = contextual_corrupt(inputs, regimes['moderate'], extra, seed)
        masks.update(more)
    elif condition == 'native': noisy = {k: v.copy() for k, v in inputs.items()}
    else:
        regime = regimes['moderate'] if condition == 'moderate' else regimes['clean'] | {
            key: regimes['moderate'][key] for key in ('ball_dropout_fraction', 'ball_gap_median_s', 'ball_gap_lognormal_sigma')}
        noisy = corrupt(inputs, regime, seed)
    visible = np.isfinite(noisy['ball_xy']).all(1)
    native_visible = np.isfinite(inputs['ball_xy']).all(1)
    masks.update(observed_ball_missing=~visible, native_ball_missing=~native_visible,
                 synthetic_ball_missing=native_visible & ~visible)
    return noisy, masks


def infer_pair(base, research, tcn, inputs):
    old = runtime.current.predict(base, research, inputs, 'combined')
    settings = base['manifest']['settings']; period = inputs['period']
    network = runtime.tcn_probabilities(tcn, inputs, settings, base['models']['owner_names'])
    h, d = runtime.combine_probabilities(old['home_probability'], old['dead_probability'],
        network[:, 0], network[:, 1], period, tcn['manifest']['config']['blend'])['tcn_blend']
    deployment = settings['deployment']
    decoded = runtime.current.decode_events(h, d, period, old['event_probability'],
        deployment['decoder'], 4, settings['switch'], deployment['epsilon'])
    runtime.validate_prediction(old['labels'], period); runtime.validate_prediction(decoded['labels'], period)
    return {'current_home': old['home_probability'], 'current_dead': old['dead_probability'],
        'current_events': old['event_probability'], 'current_labels': old['labels'],
        'tcn_home': network[:, 0], 'tcn_dead': network[:, 1],
        'blend_home': h, 'blend_dead': d, 'blend_labels': decoded['labels']}


def worker(job):
    game, output = job; output = Path(output); protocol = read(output/'protocol.json'); config = protocol['config']
    torch.set_num_threads(config['threads'])
    if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); torch.backends.mkldnn.deterministic = True
    base = runtime.load_bundle(ROOT/'models/possession-v1')
    research = runtime.current.load_research(ROOT/'models/research-specialists-clean', base)
    tcn = runtime.load_tcn(ROOT/'models/tcn-blend-v1', base, research)
    folder = output/f'game_{game}'; folder.mkdir(parents=True, exist_ok=True)
    source = SOURCE/'data'/f'Sample_Game_{game}'; prefix = f'Sample_Game_{game}_Raw'
    inputs = read_public_tracking(source/(prefix+'TrackingData_Home_Team.csv'), source/(prefix+'TrackingData_Away_Team.csv'))
    original = runtime.array_digest(inputs)
    with (source/(prefix+'EventsData.csv')).open(newline='') as handle: events = list(csv.DictReader(handle))
    raw, core, reference_audit = partial_reference(events, inputs['period'], config['guard_frames'])
    refs = {'event_interiors': raw, 'guarded_dead': core}
    np.savez_compressed(folder/'references.npz', raw=raw, guarded=core, period=inputs['period'])
    save(folder/'reference_audit.json', reference_audit)
    intervals = labelled_intervals(core, inputs['period'])
    picks = np.random.RandomState(config['bootstrap_seed']).randint(0, len(intervals), (config['bootstrap_draws'], len(intervals)))
    reports = []
    with threadpool_limits(limits=config['threads']):
        for condition, seed in cases(config):
            stem = f'{condition}_{seed}'; case_path = folder/(stem+'.json'); evidence_path = folder/(stem+'_evidence.npz')
            if case_path.exists():
                report = read(case_path)
                if report['protocol_sha256'] != runtime.file_digest(output/'protocol.json') or runtime.file_digest(evidence_path) != report['evidence_sha256']:
                    raise ValueError('Frozen transfer case drift')
                reports.append(report); continue
            tick = time.perf_counter(); noisy, masks = noisy_inputs(inputs, condition, seed, protocol)
            fingerprint = runtime.array_digest(noisy)
            evidence = infer_pair(base, research, tcn, noisy)
            if runtime.array_digest(noisy) != fingerprint or runtime.array_digest(inputs) != original:
                raise ValueError('Transfer prediction/corruption changed inputs')
            if condition == 'native':
                old_report = read(PREVIOUS/f'game_{game}_report.json'); path = PREVIOUS/f'game_{game}_evidence.npz'
                if old_report['input_fingerprint'] != fingerprint or runtime.file_digest(path) != old_report['evidence_sha256']:
                    raise ValueError('Prior native comparator provenance differs')
                with np.load(path, allow_pickle=False) as z:
                    for new_key, old_key in [('current_labels', 'labels'), ('current_home', 'home'), ('current_dead', 'dead'), ('current_events', 'events')]:
                        np.testing.assert_array_equal(evidence[new_key], z[old_key])
            subsets = {'all': np.ones(len(core), bool), **{k: masks[k] for k in (
                'observed_ball_missing', 'native_ball_missing', 'synthetic_ball_missing', 'ghost', 'wrong_team')}}
            scores = {}
            for name, ref in refs.items():
                scores[name] = {model: {key: score_subset(ref, evidence[prediction], evidence[probability], mask)
                    for key, mask in subsets.items()} for model, prediction, probability in
                    [('current', 'current_labels', 'current_dead'), ('tcn_blend', 'blend_labels', 'blend_dead')]}
            paired = {}
            for name, mask in [('all', subsets['all']), ('pass_only', np.isin(core, ['H', 'A'])),
                               ('synthetic_missing_pass', masks['synthetic_ball_missing'] & np.isin(core, ['H', 'A']))]:
                if not (mask & (core != 'X')).any(): paired[name] = None; continue
                draws = paired_interval_draws(core, evidence['current_labels'], evidence['blend_labels'], mask, intervals, picks)
                paired[name] = [float(v) if np.isfinite(v) else None for v in draws]
            np.savez_compressed(evidence_path, **evidence, period=inputs['period'], **masks)
            report = {'game': game, 'condition': condition, 'seed': seed, 'input_fingerprint': fingerprint,
                'native_input_fingerprint': original, 'protocol_sha256': runtime.file_digest(output/'protocol.json'),
                'reference_sha256': runtime.file_digest(folder/'references.npz'), 'evidence_sha256': runtime.file_digest(evidence_path),
                'native_comparator_exact': condition == 'native', 'noise': summarize_inputs(noisy),
                'stress_fractions': {k: float(v[inputs['period'] > 0].mean()) for k, v in masks.items()},
                'scores': scores, 'paired_interval_draws': paired,
                'elapsed_seconds': time.perf_counter()-tick, 'inference_reference_arrays_read': False}
            save(case_path, report); reports.append(report)
            print(f'Game {game} {stem}: frozen model pair evaluated; labelled frames only', flush=True)
    save(folder/'report.json', {'game': game, 'reference_audit': reference_audit, 'cases': reports,
        'reference_sha256': runtime.file_digest(folder/'references.npz'),
        'case_report_sha256': {f'{v["condition"]}_{v["seed"]}': runtime.file_digest(folder/f'{v["condition"]}_{v["seed"]}.json') for v in reports}})


def aggregate(output):
    protocol = read(output/'protocol.json'); config = protocol['config']; summary = []
    reports = [read(output/f'game_{g}'/'report.json') for g in config['games']]
    if sum(len(v['cases']) for v in reports) != 20: raise ValueError('Incomplete transfer comparison')
    for game in reports:
        for condition in config['conditions']:
            group = [v for v in game['cases'] if v['condition'] == condition]
            item = {'game': game['game'], 'condition': condition, 'seeds': [v['seed'] for v in group], 'metrics': {}, 'paired_intervals': {}}
            for ref in ('event_interiors', 'guarded_dead'):
                item['metrics'][ref] = {}
                for model in ('current', 'tcn_blend'):
                    metrics = {}
                    for subset in ('all', 'synthetic_ball_missing', 'observed_ball_missing', 'ghost', 'wrong_team'):
                        values = [v['scores'][ref][model][subset] for v in group if v['scores'][ref][model][subset] is not None]
                        metrics[subset] = {k: float(np.mean([v[k] for v in values if v[k] is not None]))
                            if any(v[k] is not None for v in values) else None for k in
                            ('frames', 'accuracy_on_labelled_subset', 'pass_team_accuracy', 'pass_frames', 'dead_frames', 'dead_recall', 'dead_precision')}
                    item['metrics'][ref][model] = metrics
            for subset in ('all', 'pass_only', 'synthetic_missing_pass'):
                values = [v['paired_interval_draws'][subset] for v in group if v['paired_interval_draws'][subset] is not None]
                if not values: item['paired_intervals'][subset] = None; continue
                draws = np.asarray([[np.nan if x is None else x for x in v] for v in values])
                good = np.isfinite(draws).all(0); averaged = draws[:, good].mean(0)
                item['paired_intervals'][subset] = {'conditional_95_interval': np.percentile(averaged, [2.5, 97.5]).tolist(),
                    'valid_draws': int(good.sum()), 'unit': 'event-derived labelled interval, not a complete possession'}
            summary.append(item)
            a, b = [item['metrics']['guarded_dead'][m]['all'] for m in ('current', 'tcn_blend')]
            print(f"Game {game['game']} {condition}: partial accuracy {a['accuracy_on_labelled_subset']:.4f} -> {b['accuracy_on_labelled_subset']:.4f}; pass-team {a['pass_team_accuracy']:.4f} -> {b['pass_team_accuracy']:.4f}; dead recall {a['dead_recall']:.4f} -> {b['dead_recall']:.4f}", flush=True)
    save(output/'report.json', {'status': 'complete', 'summary': summary,
        'game_report_sha256': {str(g): runtime.file_digest(output/f'game_{g}'/'report.json') for g in config['games']},
        'protocol_sha256': runtime.file_digest(output/'protocol.json'), 'training_or_tuning_performed': False,
        'limits': config['limits'], 'complete_possession_scoring_performed': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    config = read(CONFIG); source = read(SOURCE/'source_manifest.json')
    for item in source['files']:
        if runtime.file_digest(SOURCE/item['path']) != item['sha256']: raise ValueError('Pinned public source drift')
    protocol = {'config': config, 'noise_regimes': read(ROOT/'solution/configs/validation.json')['regimes'],
        'contextual': read(ROOT/'solution/configs/nested_validation.json')['contextual'],
        'source_revision': source['revision'], 'download_manifest_sha256': runtime.file_digest(SOURCE/'source_manifest.json'),
        'source_sha256': {str(p.relative_to(ROOT)): runtime.file_digest(p) for p in SOURCE_FILES},
        'model_manifests': {name: runtime.file_digest(ROOT/'models'/name/'manifest.json') for name in
                            ('possession-v1', 'research-specialists-clean', 'tcn-blend-v1')},
        'packages': {name: version(name) for name in ('torch', 'numpy', 'pandas', 'scikit-learn', 'scipy', 'joblib', 'threadpoolctl')}}
    if (output/'protocol.json').exists() and read(output/'protocol.json') != protocol:
        raise ValueError('Frozen transfer protocol changed; use a new folder')
    save(output/'protocol.json', protocol)
    with ProcessPoolExecutor(max_workers=config['workers']) as pool:
        for future in as_completed([pool.submit(worker, (g, str(output))) for g in config['games']]): future.result()
    aggregate(output)


if __name__ == '__main__': main()
