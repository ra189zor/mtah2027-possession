"""Reproduce current external predictions and isolate dead evidence/decoding errors."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'solution/src'), str(ROOT/'solution/tools')]
import research_reliability as r
from possession.inputs import file_digest
from possession.tracking_reliability import compose_candidates
from possession.event_decoder import decode_events
from possession.stoppage_diagnostics import probability_diagnostics, diagnostic_oracles, binary_boundaries
from possession.robustness import absorb_short_opposition
from run_decoder import score

SOURCE = ROOT/'outputs/reliability_research/run_20261101_clean'
CONFIG = ROOT/'solution/configs/stoppage_diagnosis.json'
SOURCES = [CONFIG, Path(__file__), ROOT/'solution/src/possession/stoppage_diagnostics.py']


def read_case(match, condition, seed, models, qm, sc, rc):
    stem = f'{condition}_{seed}'
    upstream = r.n.read(r.s.SOURCE/'outer'/match/'report.json')
    path = r.s.SOURCE/'outer'/match/f'{match}_{stem}_probabilities.npz'
    if file_digest(path) != upstream['files'][path.name]:
        raise ValueError('Upstream probability drift')
    with np.load(path, allow_pickle=False) as z:
        h, d = z['home'], z['dead']
    path = r.OLD/'outer'/match/(stem+'_events.npy')
    old = r.n.read(path.parent/'report.json')
    if file_digest(path) != old['files'][path.name]:
        raise ValueError('Original event drift')
    q = np.load(path, allow_pickle=False)
    x, period, missing, bad = r.evidence(SOURCE, match, condition, seed, h, d, qm)
    gap = r.probabilities(models['gap'], x[:, models['gap_columns']])[:, 1]
    boundary = r.probabilities(models['boundary'], x[:, models['boundary_columns']])
    hh, qq = compose_candidates(h, q, gap, boundary, missing, bad, rc['blend'])['combined']
    return hh, d, qq, period, missing


def worker(job):
    match, output = job
    output = Path(output)/match
    output.mkdir(parents=True, exist_ok=True)
    if (output/'report.json').exists():
        return
    cfg, sc, rc, foundation = r.setup()
    dc = r.n.read(CONFIG)
    frozen = r.n.read(SOURCE/'outer'/match/'report.json')
    model_path = SOURCE/'outer'/match/'specialists.joblib'
    if file_digest(model_path) != frozen['model_sha256']:
        raise ValueError('Specialist artifact drift')
    models = joblib.load(model_path)
    if match in models['training_matches'] or models['held_out'] != match:
        raise ValueError('Held-out specialist leakage')
    qm = r.quality_model(SOURCE, [match])
    truth = r.n.dead.targets(match)
    nested = r.n.read(r.s.SOURCE/'selection.json')['folds'][match]['selected']['profile']
    deploy = r.n.read(ROOT/'solution/configs/deployment.json')['decoder']
    rows, evidence, files = [], [], {}
    with threadpool_limits(limits=dc['threads']):
        for condition, seed in r.n.cases(cfg, outer=True):
            stem = f'{condition}_{seed}'
            h, d, q, period, missing = read_case(match, condition, seed, models, qm, sc, rc)
            def predict(dead, profile):
                return decode_events(h, dead, period, q, profile, 4, sc)['labels']
            original = predict(d, nested)
            path = SOURCE/'outer'/match/(stem+'_combined.npy')
            if file_digest(path) != frozen['files'][path.name] or not np.array_equal(original, np.load(path)):
                raise ValueError('Current combined prediction did not reproduce')
            deployed = predict(d, deploy)
            owner = predict(np.zeros_like(d), deploy)
            raw = owner.copy()
            raw[(d >= dc['dead_threshold']) & (period > 0)] = 'D'
            predictions = {'current_nested': original, 'current_deployment_costs': deployed,
                           'raw_dead_threshold': raw,
                           'zero_stop_restart_costs': predict(d, deploy | {'stop': 0, 'restart': 0}),
                           'true_dead_emissions': predict((truth == 'D').astype(float), deploy),
                           **diagnostic_oracles(truth, deployed, owner, period)}
            refs = {'provider': truth, 'absorbed_3s': absorb_short_opposition(truth, period, 3)[0]}
            for name, prediction in predictions.items():
                for convention, labels in refs.items():
                    metrics = score(labels, prediction, period)
                    if name == 'both_true' and convention == 'provider':
                        if metrics['spell_f1_mean'] != 1 or metrics['macro_f1'] != 1:
                            raise ValueError('Both-true sanity test failed')
                    rows.append({'match': match, 'condition': condition, 'seed': seed, 'model': name,
                                 'label_set': convention, **metrics,
                                 'dead_boundaries_1s': binary_boundaries(labels, prediction, period)})
            evidence.append({'match': match, 'condition': condition, 'seed': seed,
                             **probability_diagnostics(d, truth, period, ~missing)})
            # Cache input-only evidence for follow-up external experiments, separate from targets.
            path = output/(stem+'_evidence.npz')
            np.savez_compressed(path, home=h, dead=d, events=q, period=period, ball_missing=missing)
            files[path.name] = file_digest(path)
            print(f'{match} {stem}: exact replay and eight diagnostic interventions complete', flush=True)
    r.save(output/'report.json', {'held_out': match, 'training_matches': models['training_matches'],
           'source_model_sha256': frozen['model_sha256'], 'results': rows, 'evidence': evidence, 'files': files})


def aggregate(output):
    cfg, _, _, _ = r.setup()
    reports = [r.n.read(output/m/'report.json') for m in cfg['matches']]
    rows = [v for p in reports for v in p['results']]
    for row in rows:
        confusion = np.asarray(row['confusion'])
        row['accuracy'] = float(np.trace(confusion)/confusion.sum())
    expected = 7*10*8*2
    if len(rows) != expected:
        raise ValueError(f'Incomplete diagnostic matrix: {len(rows)} != {expected}')
    summary = []
    models = sorted({v['model'] for v in rows})
    for condition in cfg['outer_conditions']:
        for convention in ('provider', 'absorbed_3s'):
            for model in models:
                group = [v for v in rows if v['condition'] == condition and v['label_set'] == convention and v['model'] == model]
                item = {'condition': condition, 'label_set': convention, 'model': model}
                for key in ('spell_f1_mean', 'macro_f1', 'dead_precision', 'dead_recall', 'accuracy'):
                    item[key] = float(np.mean([np.mean([v[key] for v in group if v['match'] == m]) for m in cfg['matches']]))
                summary.append(item)
    r.save(output/'report.json', {'status': 'complete', 'exact_replays': 70, 'oracle_sanity_cases': 70,
           'aggregation_source_sha256': file_digest(Path(__file__)),
           'input_report_sha256': {m: file_digest(output/m/'report.json') for m in cfg['matches']},
           'summary': summary, 'results': rows, 'evidence': [v for p in reports for v in p['evidence']],
           'limitations': 'Adaptive external-only diagnosis on previously inspected fixtures. Deployment-cost rows use six-match held-out heads, not the all-seven production weights. Oracle interventions are not attainable scores or guaranteed ceilings.'})
    for condition in cfg['outer_conditions']:
        print(condition+': '+', '.join(f"{v['model']}={v['spell_f1_mean']:.4f} Drec={v['dead_recall']:.3f}" for v in summary if v['condition']==condition and v['label_set']=='provider'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--aggregate-only', action='store_true', help='Read completed diagnostic reports without inference or refitting')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        aggregate(output)
        return
    cfg, _, _, _ = r.setup()
    config = r.n.read(CONFIG)
    protocol = {'config': config, 'matches': cfg['matches'],
                'source_sha256': {str(p.relative_to(ROOT)): file_digest(p) for p in SOURCES},
                'foundation_source_sha256': r.s.setup()[2],
                'combined_protocol_sha256': file_digest(SOURCE/'protocol.json'),
                'deployment_config_sha256': file_digest(ROOT/'solution/configs/deployment.json')}
    if (output/'protocol.json').exists() and r.n.read(output/'protocol.json') != protocol:
        raise ValueError('Changed diagnostic protocol; use a new output folder')
    r.save(output/'protocol.json', protocol)
    with ProcessPoolExecutor(max_workers=config['workers']) as pool:
        for future in as_completed([pool.submit(worker, (m, str(output))) for m in cfg['matches']]):
            future.result()
    aggregate(output)


if __name__ == '__main__':
    main()
