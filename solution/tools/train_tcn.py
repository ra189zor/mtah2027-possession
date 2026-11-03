"""Resumable seven-fold offline TCN challenger; external tracking only."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path
import json
import platform
import sys
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'solution/src'), str(ROOT/'solution/tools')]
import research_reliability as r
from possession.temporal_network import (PossessionTCN, fit_normalizer, normalize, sampled_indices,
    upsample_probabilities, boundary_targets, losses, infer_sampled, combine_probabilities)
from possession.inputs import file_digest
from possession.validation import period_blocks
from possession.dead_model import sample_weights
from possession.event_decoder import decode_events
from possession.robustness import absorb_short_opposition
from possession.stoppage_diagnostics import binary_boundaries
from run_decoder import score
from run_oracles import paired_draw_scores

CONFIG = ROOT/'solution/configs/tcn.json'
DIAGNOSIS = ROOT/'outputs/stoppage_research/diagnosis_20261102'
BOUNDARY = ROOT/'outputs/stoppage_research/boundaries_20261102'
SOURCES = [CONFIG, Path(__file__), ROOT/'solution/src/possession/temporal_network.py']


def atomic_torch_save(value, path):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary); temporary.replace(path)


def prepare(job):
    match, out = job; out = Path(out)
    folder = out/'prepared'/match; folder.mkdir(parents=True, exist_ok=True)
    if (folder/'index.json').exists():
        index = r.n.read(folder/'index.json')
        for name, digest in index['files'].items():
            if file_digest(folder/name) != digest: raise ValueError('TCN cache drift')
        return
    cfg, _, _, _ = r.setup(); config = r.n.read(CONFIG)
    source = r.n.read(r.s.SOURCE/'prepared'/match/'index.json')
    index = {'match': match, 'files': {}, 'source_files': {}, 'cases': {}}
    indices = period = None
    for condition, seed in r.n.cases(cfg, outer=True):
        stem = f'{condition}_{seed}'; case = source['cases'][stem]
        for kind in ('matrix', 'context'):
            path = Path(case[kind])
            if file_digest(path) != source['files'][str(path)]: raise ValueError('Physical source drift')
            index['source_files'][str(path)] = source['files'][str(path)]
        with np.load(case['context'], allow_pickle=False) as z: p = z['period']
        if period is None:
            period = p; indices = sampled_indices(period, config['sample_stride_frames'])
        elif not np.array_equal(period, p): raise ValueError('Corruption changed half alignment')
        matrix = np.load(case['matrix'], mmap_mode='r', allow_pickle=False)
        if matrix.shape != (len(period), config['input_features']): raise ValueError('Unexpected TCN feature schema')
        path = folder/(stem+'.npy')
        np.save(path, matrix[indices], allow_pickle=False)
        index['files'][path.name] = file_digest(path); index['cases'][stem] = path.name
    path = folder/'timing.npz'
    np.savez_compressed(path, period=period, indices=indices, sampled_period=period[indices])
    index['files'][path.name] = file_digest(path)
    # Separate training references, never passed to normalization or feature extraction.
    labels = r.n.dead.targets(match)
    codes = np.full(len(labels), -1, dtype=np.int64)
    for value, label in enumerate(('H', 'A', 'D')): codes[labels == label] = value
    events = boundary_targets(codes, period, config['boundary_radius_frames'])
    alive = np.isin(labels[indices], ['H', 'A'])
    weight = np.zeros(len(indices), dtype=np.float32)
    weight[alive] = sample_weights(labels, period, indices[alive], 'spell', config['owner_weight_cap'])
    path = folder/'training_targets.npz'
    np.savez_compressed(path, state=codes[indices], boundary=events[indices], owner_weight=weight)
    index['files'][path.name] = file_digest(path)
    index['native_frames'], index['sampled_frames'] = len(period), len(indices)
    r.save(folder/'index.json', index)
    print(f'{match}: ten tracking-feature sequences cached at 5 Hz; references stored separately', flush=True)


def load_training(training, out, cfg):
    cases, indexes = {}, {}
    for match in training:
        folder = out/'prepared'/match; index = r.n.read(folder/'index.json'); indexes[match] = index
        for name, digest in index['files'].items():
            if file_digest(folder/name) != digest: raise ValueError('Training cache drift')
        with np.load(folder/'timing.npz', allow_pickle=False) as z: period = z['sampled_period']
        with np.load(folder/'training_targets.npz', allow_pickle=False) as z: targets = {k: z[k] for k in z.files}
        blocks = period_blocks(period)
        for condition, seed in r.n.cases(cfg):
            matrix = np.load(folder/index['cases'][f'{condition}_{seed}'], mmap_mode='r', allow_pickle=False)
            cases[match, condition, seed] = {'x': matrix, 'blocks': blocks, **targets}
    return cases, indexes


def training_window(case, start, block, core, radius):
    a, b = block
    if not a <= start < b: raise ValueError('Training core must start within its half')
    low, high = max(a, start-radius), min(b, start+core+radius)
    x = np.full((core+2*radius, case['x'].shape[1]), np.nan, dtype=np.float32)
    offset = low-(start-radius); x[offset:offset+high-low] = case['x'][low:high]
    end = min(b, start+core); length = end-start
    state = np.full(core, -1, dtype=np.int64); boundary = state.copy()
    weight = np.zeros(core, dtype=np.float32)
    state[:length] = case['state'][start:end]
    boundary[:length] = case['boundary'][start:end]
    weight[:length] = case['owner_weight'][start:end]
    return x, state, weight, boundary


def batch(cases, training, rng, normalizer, config, cfg, radius):
    xs, ys, ws, bs = [], [], [], []
    for _ in range(config['batch_size']):
        match = training[int(rng.integers(len(training)))]
        condition = cfg['inner_conditions'][int(rng.integers(len(cfg['inner_conditions'])))]
        seed = cfg['seeds'][0] if condition == 'clean' else cfg['seeds'][int(rng.integers(len(cfg['seeds'])))]
        case = cases[match, condition, seed]
        lengths = np.array([b-a for a, b in case['blocks']], dtype=float)
        block = case['blocks'][int(rng.choice(len(lengths), p=lengths/lengths.sum()))]
        start = int(rng.integers(*block))
        x, y, w, b = training_window(case, start, block, config['core_steps'], radius)
        xs.append(x); ys.append(y); ws.append(w); bs.append(b)
    x = normalize(np.stack(xs), normalizer, config['normalization_clip'])
    return (torch.from_numpy(x.transpose(0, 2, 1).copy()), torch.from_numpy(np.stack(ys)),
            torch.from_numpy(np.stack(ws)), torch.from_numpy(np.stack(bs)))


def fit(held, out, config, cfg, protocol_sha):
    folder = out/'outer'/held; folder.mkdir(parents=True, exist_ok=True)
    training = [m for m in cfg['matches'] if m != held]
    if held in training or len(training) != 6: raise ValueError('Outer split violation')
    model_path = folder/'model.pt'
    if model_path.exists():
        report = r.n.read(folder/'fit.json')
        if file_digest(model_path) != report['model_sha256']: raise ValueError('Trained model drift')
        bundle = torch.load(model_path, map_location='cpu', weights_only=True)
        if bundle['training_matches'] != training or bundle['protocol_sha256'] != protocol_sha: raise ValueError('Model provenance drift')
        return bundle
    cases, indexes = load_training(training, out, cfg)
    # At roughly 1 Hz, using only the explicitly provided training-case matrices.
    normalizer = fit_normalizer(case['x'][::5] for case in cases.values())
    normalizer_lists = {k: v.tolist() for k, v in normalizer.items()}
    torch.manual_seed(config['seed'])
    model = PossessionTCN(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['minimum_learning_rate'])
    checkpoint = folder/'checkpoint.pt'; history = []; first = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['protocol_sha256'] != protocol_sha or saved['training_matches'] != training or saved['normalizer'] != normalizer_lists:
            raise ValueError('Checkpoint provenance/normalization mismatch')
        model.load_state_dict(saved['state_dict']); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); torch.set_rng_state(saved['torch_rng'])
        first, history = saved['epoch'], saved['history']
    for epoch in range(first, config['epochs']):
        tick = time.perf_counter(); model.train(); totals = np.zeros(4)
        rng = np.random.default_rng(config['seed']+epoch)
        for step in range(config['steps_per_epoch']):
            x, y, w, b = batch(cases, training, rng, normalizer, config, cfg, model.radius)
            optimizer.zero_grad(set_to_none=True)
            states, boundaries = model(x); lo, hi = model.radius, model.radius+config['core_steps']
            loss, parts = losses(states[:, :, lo:hi], boundaries[:, :, lo:hi], y, w, b, config)
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            totals += [float(loss.detach()), *[float(parts[k].detach()) for k in ('dead', 'home', 'boundary')]]
        scheduler.step()
        item = {'epoch': epoch+1, 'mean_loss': float(totals[0]/config['steps_per_epoch']),
                'dead_loss': float(totals[1]/config['steps_per_epoch']),
                'owner_loss': float(totals[2]/config['steps_per_epoch']),
                'boundary_loss': float(totals[3]/config['steps_per_epoch']),
                'seconds': time.perf_counter()-tick, 'next_learning_rate': scheduler.get_last_lr()[0]}
        history.append(item)
        atomic_torch_save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'torch_rng': torch.get_rng_state(),
            'epoch': epoch+1, 'history': history, 'normalizer': normalizer_lists,
            'training_matches': training, 'protocol_sha256': protocol_sha}, checkpoint)
        r.save(folder/'training_progress.json', {'held_out': held, 'training_matches': training, 'history': history})
        print(f"{held}: epoch {epoch+1}/{config['epochs']} loss={item['mean_loss']:.4f} ({item['seconds']:.1f}s)", flush=True)
    bundle = {'state_dict': model.state_dict(), 'normalizer': normalizer_lists, 'config': config,
              'held_out': held, 'training_matches': training, 'protocol_sha256': protocol_sha,
              'training_index_sha256': {m: file_digest(out/'prepared'/m/'index.json') for m in training},
              'torch_version': str(torch.__version__), 'epochs': config['epochs']}
    atomic_torch_save(bundle, model_path)
    r.save(folder/'fit.json', {'held_out': held, 'training_matches': training,
        'normalization_training_matches': training, 'history': history,
        'parameters': sum(p.numel() for p in model.parameters()), 'model_sha256': file_digest(model_path),
        'protocol_sha256': protocol_sha, 'training_index_sha256': bundle['training_index_sha256']})
    return bundle


def evaluate(held, out, bundle, config, cfg, sc):
    folder = out/'outer'/held
    if (folder/'report.json').exists():
        old = r.n.read(folder/'report.json')
        if old['model_sha256'] != file_digest(folder/'model.pt'): raise ValueError('Evaluation/model drift')
        for name, digest in old['files'].items():
            if file_digest(folder/name) != digest: raise ValueError('Evaluated artifact drift')
        return
    model = PossessionTCN(config); model.load_state_dict(bundle['state_dict']); model.eval()
    prepared = out/'prepared'/held; index = r.n.read(prepared/'index.json')
    for name, digest in index['files'].items():
        if file_digest(prepared/name) != digest: raise ValueError('Held-out cache drift')
    with np.load(prepared/'timing.npz', allow_pickle=False) as z:
        period, indices, small_period = z['period'], z['indices'], z['sampled_period']
    labels = r.n.dead.targets(held)
    references = {'provider': labels, 'absorbed_3s': absorb_short_opposition(labels, period, 3)[0]}
    diagnostic = r.n.read(DIAGNOSIS/held/'report.json')
    previous = r.n.read(BOUNDARY/held/'report.json')
    profile = r.n.read(ROOT/'solution/configs/deployment.json')['decoder']
    rows, files, paired, timings = [], {}, [], []
    for condition, seed in r.n.cases(cfg, outer=True):
        stem = f'{condition}_{seed}'; tick = time.perf_counter()
        matrix = np.load(prepared/index['cases'][stem], mmap_mode='r', allow_pickle=False)
        sampled = infer_sampled(model, matrix, small_period, bundle['normalizer'], config)
        probabilities = upsample_probabilities(sampled, indices, period)
        path = folder/(stem+'_probabilities.npz')
        np.savez_compressed(path, sampled=sampled, home=probabilities[:, 0], dead=probabilities[:, 1])
        files[path.name] = file_digest(path)
        timings.append({'condition': condition, 'seed': seed, 'inference_seconds': time.perf_counter()-tick})
        path = DIAGNOSIS/held/(stem+'_evidence.npz')
        if file_digest(path) != diagnostic['files'][path.name]: raise ValueError('Current-model evidence drift')
        with np.load(path, allow_pickle=False) as z:
            h, d, events, missing = z['home'], z['dead'], z['events'], z['ball_missing']
            if not np.array_equal(period, z['period']): raise ValueError('Original/TCN alignment differs')
        candidates = combine_probabilities(h, d, probabilities[:, 0], probabilities[:, 1], period, config['blend'])
        predictions = {name: decode_events(hh, dd, period, events, profile, 4, sc)['labels']
                       for name, (hh, dd) in candidates.items()}
        path = BOUNDARY/held/(stem+'_current.npy')
        if file_digest(path) != previous['files'][path.name] or not np.array_equal(predictions['current'], np.load(path)):
            raise ValueError('Original corrected-combined comparator did not reproduce exactly')
        for name, prediction in predictions.items():
            path = folder/(stem+'_'+name+'.npy'); np.save(path, prediction, allow_pickle=False)
            files[path.name] = file_digest(path)
            for convention, truth in references.items():
                metrics = score(truth, prediction, period); confusion = np.asarray(metrics['confusion'])
                alive = np.isin(truth, ['H', 'A']) & (period > 0)
                mask = alive & missing
                rows.append({'match': held, 'condition': condition, 'seed': seed, 'model': name,
                    'label_set': convention, **metrics, 'frame_accuracy': float(np.trace(confusion)/confusion.sum()),
                    'team_accuracy_true_alive': float(np.mean(prediction[alive] == truth[alive])),
                    'missing_alive_accuracy': float(np.mean(prediction[mask] == truth[mask])) if mask.any() else None,
                    'dead_boundaries_1s': binary_boundaries(truth, prediction, period),
                    'direct_switches_12': r.s.direct_switch_stats(truth, prediction)})
        if condition in ('moderate', 'contextual'):
            for convention, truth in references.items():
                gt = r.n.pm.label_codes(truth); spells = r.n.pm.possession_spells(gt, r.n.pm.valid_runs(gt))
                picks = np.random.RandomState(config['seed']).randint(0, len(spells), (config['conditional_draws'], len(spells)))
                delta = paired_draw_scores(gt, r.n.pm.label_codes(predictions['tcn_blend']), picks)-paired_draw_scores(gt, r.n.pm.label_codes(predictions['current']), picks)
                paired.append({'condition': condition, 'seed': seed, 'label_set': convention, 'draws': delta.tolist()})
        print(f'{held} {stem}: four candidates scored; original prediction reproduced exactly', flush=True)
    r.save(folder/'report.json', {'held_out': held, 'training_matches': bundle['training_matches'],
        'normalization_training_matches': bundle['training_matches'], 'model_sha256': file_digest(folder/'model.pt'),
        'results': rows, 'files': files, 'paired': paired, 'inference_timings': timings})


def configure_torch_threads(threads):
    torch.set_num_threads(threads)
    # ProcessPoolExecutor reuses workers; PyTorch permits this setting only once.
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def worker(job):
    held, output = job; out = Path(output)
    cfg, sc, _, _ = r.setup(); config = r.n.read(CONFIG)
    configure_torch_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.deterministic = True
    with threadpool_limits(limits=config['threads']):
        bundle = fit(held, out, config, cfg, file_digest(out/'protocol.json'))
        evaluate(held, out, bundle, config, cfg, sc)


def aggregate(out):
    cfg, _, _, _ = r.setup(); config = r.n.read(CONFIG)
    reports = [r.n.read(out/'outer'/m/'report.json') for m in cfg['matches']]
    rows = [v for p in reports for v in p['results']]
    if len(rows) != 560: raise ValueError('Incomplete TCN evaluation matrix')
    summary = []
    for condition in cfg['outer_conditions']:
        for convention in ('provider', 'absorbed_3s'):
            for model in config['candidates']:
                per_match = []
                for match in cfg['matches']:
                    group = [v for v in rows if v['match'] == match and v['condition'] == condition and v['label_set'] == convention]
                    current = np.mean([v['spell_f1_mean'] for v in group if v['model'] == 'current'])
                    chosen = [v for v in group if v['model'] == model]
                    item = {'match': match, 'delta': float(np.mean([v['spell_f1_mean'] for v in chosen])-current),
                            'count_gate': bool(abs(np.mean([v['n_pred_spells']/v['n_gt_spells']-1 for v in chosen])) <= .2)}
                    for key in ('spell_f1_mean', 'frame_accuracy', 'macro_f1', 'dead_precision', 'dead_recall', 'team_accuracy_true_alive'):
                        item[key] = float(np.mean([v[key] for v in chosen]))
                    per_match.append(item)
                delta = np.array([v['delta'] for v in per_match])
                picks = np.random.RandomState(config['seed']).randint(0, len(delta), (config['bootstrap_draws'], len(delta)))
                lo, hi = np.percentile(delta[picks].mean(1), [2.5, 97.5])
                item = {'condition': condition, 'label_set': convention, 'model': model,
                        'delta_vs_current': float(delta.mean()), 'descriptive_match_ci': [float(lo), float(hi)],
                        'improved_matches': int((delta > 0).sum()), 'count_gate_passes': sum(v['count_gate'] for v in per_match),
                        'per_match': per_match}
                for key in ('spell_f1_mean', 'frame_accuracy', 'macro_f1', 'dead_precision', 'dead_recall', 'team_accuracy_true_alive'):
                    item[key] = float(np.mean([v[key] for v in per_match]))
                summary.append(item)
                if convention == 'provider':
                    print(f"{condition} {model}: spell={item['spell_f1_mean']:.5f}, accuracy={item['frame_accuracy']:.4f}, delta={item['delta_vs_current']:+.5f}, CI={[round(lo,5),round(hi,5)]}, counts={item['count_gate_passes']}/7", flush=True)
    r.save(out/'report.json', {'status': 'complete', 'promoted': False, 'primary_challenger': config['primary_challenger'],
        'summary': summary, 'results': rows, 'exact_comparator_replays': 70,
        'input_report_sha256': {m: file_digest(out/'outer'/m/'report.json') for m in cfg['matches']},
        'conditional_paired_draws': [dict(match=p['held_out'], **v) for p in reports for v in p['paired']],
        'limits': 'One fixed architecture and training seed; adaptive reuse of seven fixtures. No new independently labelled full matches or challenge evaluation. Descriptive intervals omit researcher adaptation and shared training dependence. Prespecified blend is primary; no retrospective per-match candidate selection.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['all', 'prepare', 'fit'], default='all')
    args = parser.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    cfg, _, _, _ = r.setup(); config = r.n.read(CONFIG)
    protocol = {'config': config, 'matches': cfg['matches'], 'python': platform.python_version(),
                'source_sha256': {str(p.relative_to(ROOT)): file_digest(p) for p in SOURCES},
                'foundation_source_sha256': r.s.setup()[2],
                'packages': {name: version(name) for name in ('torch', 'numpy', 'scikit-learn', 'scipy', 'joblib', 'threadpoolctl')},
                'diagnosis_sha256': file_digest(DIAGNOSIS/'report.json'),
                'deployment_config_sha256': file_digest(ROOT/'solution/configs/deployment.json')}
    if (out/'protocol.json').exists() and r.n.read(out/'protocol.json') != protocol:
        raise ValueError('TCN protocol/environment changed; use a new output directory')
    r.save(out/'protocol.json', protocol)
    if args.stage in ('all', 'prepare'):
        with ProcessPoolExecutor(max_workers=config['workers']) as pool:
            for future in as_completed([pool.submit(prepare, (m, str(out))) for m in cfg['matches']]): future.result()
    if args.stage in ('all', 'fit'):
        with ProcessPoolExecutor(max_workers=config['workers']) as pool:
            for future in as_completed([pool.submit(worker, (m, str(out))) for m in cfg['matches']]): future.result()
        aggregate(out)


if __name__ == '__main__': main()
