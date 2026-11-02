"""Fixed seven-match stop/restart-head experiment with cross-fitted training evidence."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'solution/src'), str(ROOT/'solution/tools')]
import research_reliability as r
from possession.inputs import file_digest
from possession.stoppage_boundaries import boundary_features, boundary_training_rows, decode_boundaries
from possession.stoppage_diagnostics import binary_boundaries
from possession.robustness import absorb_short_opposition
from run_decoder import score
from run_oracles import paired_draw_scores

CONFIG = ROOT/'solution/configs/stoppage_boundaries.json'
DIAGNOSIS = ROOT/'outputs/stoppage_research/diagnosis_20261102'
SOURCES = [CONFIG, Path(__file__), ROOT/'solution/src/possession/stoppage_boundaries.py']


def features(match, condition, seed, dead, config):
    index = r.n.read(r.s.SOURCE/'prepared'/match/'index.json')
    case = index['cases'][f'{condition}_{seed}']
    path = Path(case['matrix'])
    if file_digest(path) != index['files'][str(path)]:
        raise ValueError('Physical feature drift')
    with np.load(case['context'], allow_pickle=False) as z:
        period = z['period']
    matrix = np.load(path, mmap_mode='r', allow_pickle=False)
    return boundary_features(matrix[:, :275], dead, period, config), period


def fit(held, config, cfg):
    xs, ys, ws, lineage, audits = [], [], [], [], []
    training = [m for m in cfg['matches'] if m != held]
    for match in training:
        labels = r.n.dead.targets(match)
        for condition, seed in r.n.cases(cfg):
            _, d, trace = r.s.upstream_training_case(held, match, condition, seed, cfg)
            x, period = features(match, condition, seed, d, config)
            rows, y, w, audit = boundary_training_rows(labels, period, seed, config)
            if condition != 'clean': w /= len(cfg['seeds'])
            xs.append(x[rows]); ys.append(y); ws.append(w)
            lineage.append(trace)
            audits.append({'match': match, 'condition': condition, 'seed': seed, **audit})
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    del xs, ys, ws
    w *= len(w)/w.sum()
    columns = np.flatnonzero(np.isfinite(x).any(axis=0))
    model = HistGradientBoostingClassifier(**config['model']).fit(x[:, columns], y, sample_weight=w)
    if not np.array_equal(model.classes_, [0, 1, 2]):
        raise ValueError('Boundary class order changed')
    return {'model': model, 'columns': columns, 'training_matches': training, 'held_out': held,
            'lineage': lineage, 'sampling_audits': audits, 'training_rows': len(y)}


def worker(job):
    match, out = job
    folder = Path(out)/match
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'report.json').exists(): return
    cfg, sc, _, _ = r.setup()
    config = r.n.read(CONFIG)
    with threadpool_limits(limits=config['threads']):
        model_path = folder/'model.joblib'
        if model_path.exists(): bundle = joblib.load(model_path)
        else:
            bundle = fit(match, config, cfg)
            joblib.dump(bundle, model_path)
        if match in bundle['training_matches'] or bundle['held_out'] != match:
            raise ValueError('Held-out leakage')
        print(f"{match}: stop/restart head fitted on {bundle['training_rows']} cross-fitted rows", flush=True)
        truth = r.n.dead.targets(match)
        base = r.n.read(ROOT/'solution/configs/deployment.json')['decoder']
        diagnostic = r.n.read(DIAGNOSIS/match/'report.json')
        rows, files, paired = [], {}, []
        for condition, seed in r.n.cases(cfg, outer=True):
            stem = f'{condition}_{seed}'
            path = DIAGNOSIS/match/(stem+'_evidence.npz')
            if file_digest(path) != diagnostic['files'][path.name]:
                raise ValueError('Combined evidence drift')
            with np.load(path, allow_pickle=False) as z:
                h, d, events, period = z['home'], z['dead'], z['events'], z['period']
            x, feature_period = features(match, condition, seed, d, config)
            if not np.array_equal(period, feature_period): raise ValueError('Half alignment drift')
            q = r.probabilities(bundle['model'], x[:, bundle['columns']])
            predictions = {name: decode_boundaries(h, d, period, events, q, base, sc, config, strength)['labels']
                           for name, strength in [('current', 0), ('boundary', config['reward_strength'])]}
            references = {'provider': truth, 'absorbed_3s': absorb_short_opposition(truth, period, 3)[0]}
            for name, pred in predictions.items():
                path = folder/(stem+'_'+name+'.npy')
                np.save(path, pred, allow_pickle=False); files[path.name] = file_digest(path)
                for convention, labels in references.items():
                    metrics = score(labels, pred, period)
                    if name == 'current':
                        original = next(v for v in diagnostic['results'] if v['model'] == 'current_deployment_costs' and
                                        v['condition'] == condition and v['seed'] == seed and v['label_set'] == convention)
                        if metrics['spell_f1_mean'] != original['spell_f1_mean'] or metrics['confusion'] != original['confusion']:
                            raise ValueError('Zero reward comparator did not reproduce')
                    rows.append({'match': match, 'condition': condition, 'seed': seed, 'model': name,
                                 'label_set': convention, **metrics,
                                 'dead_boundaries_1s': binary_boundaries(labels, pred, period)})
            if condition in ('moderate', 'contextual'):
                for convention, labels in references.items():
                    gt = r.n.pm.label_codes(labels)
                    spells = r.n.pm.possession_spells(gt, r.n.pm.valid_runs(gt))
                    picks = np.random.RandomState(config['seed']).randint(0, len(spells), (200, len(spells)))
                    delta = paired_draw_scores(gt, r.n.pm.label_codes(predictions['boundary']), picks)-paired_draw_scores(gt, r.n.pm.label_codes(predictions['current']), picks)
                    paired.append({'condition': condition, 'seed': seed, 'label_set': convention, 'draws': delta.tolist()})
            print(f'{match} {stem}: boundary candidate and unchanged comparator scored', flush=True)
        r.save(folder/'report.json', {'held_out': match, 'training_matches': bundle['training_matches'],
            'lineage': bundle['lineage'], 'results': rows, 'paired': paired,
            'model_sha256': file_digest(model_path), 'files': files})


def aggregate(out):
    cfg, _, _, _ = r.setup()
    config = r.n.read(CONFIG)
    reports = [r.n.read(out/m/'report.json') for m in cfg['matches']]
    rows = [v for p in reports for v in p['results']]
    if len(rows) != 280: raise ValueError('Incomplete boundary matrix')
    summary = []
    for condition in cfg['outer_conditions']:
        for convention in ('provider', 'absorbed_3s'):
            group = [v for v in rows if v['condition'] == condition and v['label_set'] == convention]
            per_match = []
            for match in cfg['matches']:
                means = {model: np.mean([v['spell_f1_mean'] for v in group if v['match'] == match and v['model'] == model])
                         for model in ('current', 'boundary')}
                per_match.append({'match': match, **means, 'delta': means['boundary']-means['current']})
            delta = np.array([v['delta'] for v in per_match])
            picks = np.random.RandomState(config['seed']).randint(0, 7, (config['bootstrap_draws'], 7))
            lo, hi = np.percentile(delta[picks].mean(1), [2.5, 97.5])
            summary.append({'condition': condition, 'label_set': convention,
                'current': float(np.mean([v['current'] for v in per_match])),
                'boundary': float(np.mean([v['boundary'] for v in per_match])),
                'delta': float(delta.mean()), 'descriptive_match_ci': [float(lo), float(hi)],
                'improved_matches': int((delta > 0).sum()), 'per_match': per_match})
            print(condition+' '+convention+': '+str(summary[-1]), flush=True)
    r.save(out/'report.json', {'status': 'complete', 'promoted': False, 'summary': summary, 'results': rows,
        'input_report_sha256': {m: file_digest(out/m/'report.json') for m in cfg['matches']},
        'paired_conditional_draws': [dict(match=p['held_out'], **v) for p in reports for v in p['paired']],
        'limits': 'Adaptive seven-fixture experiment; descriptive uncertainty does not account for shared training or prior architecture adaptation. No independent Metrica evaluation of this candidate yet.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    cfg, _, _, _ = r.setup(); config = r.n.read(CONFIG)
    protocol = {'config': config, 'matches': cfg['matches'],
                'source_sha256': {str(p.relative_to(ROOT)): file_digest(p) for p in SOURCES},
                'foundation_source_sha256': r.s.setup()[2],
                'diagnosis_sha256': file_digest(DIAGNOSIS/'report.json'),
                'deployment_config_sha256': file_digest(ROOT/'solution/configs/deployment.json')}
    if (out/'protocol.json').exists() and r.n.read(out/'protocol.json') != protocol:
        raise ValueError('Boundary protocol changed; use a new output directory')
    r.save(out/'protocol.json', protocol)
    with ProcessPoolExecutor(max_workers=config['workers']) as pool:
        for future in as_completed([pool.submit(worker, (m, str(out))) for m in cfg['matches']]): future.result()
    aggregate(out)


if __name__ == '__main__': main()
