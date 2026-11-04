"""Frozen current versus owner-only TCN transfer on previously unscored Metrica Game 3."""
import argparse
from importlib.metadata import version
from pathlib import Path
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import validate_tcn_transfer as t
from possession.metrica_epts import read_epts_tracking, game3_reference

ROOT = t.ROOT
SOURCE = ROOT/'data/external/metrica_game3'
FREEZE = ROOT/'outputs/owner_transfer_game3/candidate_freeze.json'
CONFIG = ROOT/'solution/configs/owner_transfer_game3.json'
MODEL_KEYS = [('current', 'current_labels', 'current_dead'), ('tcn_owner_blend', 'blend_labels', 'blend_dead')]
SUBSETS = ('all', 'observed_ball_missing', 'native_ball_missing', 'synthetic_ball_missing', 'ghost', 'wrong_team')


def infer_pair(base, research, tcn, inputs):
    old = t.runtime.current.predict(base, research, inputs, 'combined')
    settings = base['manifest']['settings']; period = inputs['period']
    network = t.runtime.tcn_probabilities(tcn, inputs, settings, base['models']['owner_names'])
    h, d = t.runtime.combine_probabilities(old['home_probability'], old['dead_probability'],
        network[:, 0], network[:, 1], period, .5)['tcn_owner_blend']
    np.testing.assert_array_equal(d, old['dead_probability'])
    deploy = settings['deployment']
    decoded = t.runtime.current.decode_events(h, d, period, old['event_probability'],
        deploy['decoder'], 4, settings['switch'], deploy['epsilon'])
    t.runtime.validate_prediction(old['labels'], period); t.runtime.validate_prediction(decoded['labels'], period)
    return dict(current_home=old['home_probability'], current_dead=old['dead_probability'],
        current_events=old['event_probability'], current_labels=old['labels'],
        tcn_home=network[:, 0], tcn_dead=network[:, 1], blend_home=h, blend_dead=d, blend_labels=decoded['labels'])


def paired(core, evidence, picks, intervals):
    result = {}
    for name, mask in [('all', np.ones(len(core), bool)), ('pass_only', np.isin(core, ['H', 'A'])),
                       ('synthetic_missing_pass', evidence['synthetic_ball_missing'] & np.isin(core, ['H', 'A']))]:
        values = t.paired_interval_draws(core, evidence['current_labels'], evidence['blend_labels'], mask, intervals, picks)
        result[name] = [float(v) if np.isfinite(v) else None for v in values]
    return result


def aggregate(reports, config):
    summary = []
    for condition in config['conditions']:
        group = [r for r in reports if r['condition'] == condition]
        item = dict(condition=condition, seeds=[r['seed'] for r in group], metrics={}, paired_intervals={})
        for ref in ('event_interiors', 'guarded_dead'):
            item['metrics'][ref] = {}
            for model, _, _ in MODEL_KEYS:
                item['metrics'][ref][model] = {}
                for subset in SUBSETS:
                    values = [r['scores'][ref][model][subset] for r in group if r['scores'][ref][model][subset] is not None]
                    item['metrics'][ref][model][subset] = {key: float(np.mean([v[key] for v in values if v[key] is not None]))
                        if any(v[key] is not None for v in values) else None for key in
                        ('frames', 'accuracy_on_labelled_subset', 'pass_team_accuracy', 'pass_frames', 'dead_frames', 'dead_recall', 'dead_precision')}
        for name in ('all', 'pass_only', 'synthetic_missing_pass'):
            draws = np.asarray([[np.nan if v is None else v for v in r['paired_interval_draws'][name]] for r in group])
            good = np.isfinite(draws).all(0)
            item['paired_intervals'][name] = {'conditional_95_interval': np.percentile(draws[:, good].mean(0), [2.5, 97.5]).tolist(),
                'valid_draws': int(good.sum()), 'unit': 'event-derived labelled interval, not complete possession'} if good.any() else None
        summary.append(item)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve(); out.mkdir(parents=True, exist_ok=True)
    config = t.read(CONFIG); freeze = t.read(FREEZE); source = t.read(SOURCE/'source_manifest.json')
    if freeze['config_sha256'] != t.runtime.file_digest(CONFIG) or freeze['config'] != config:
        raise ValueError('Candidate configuration drift')
    if source['candidate_freeze_sha256'] != t.runtime.file_digest(FREEZE) or freeze['game3_local_data_present_at_freeze']:
        raise ValueError('Candidate freeze chronology differs')
    if source['revision'] != config['source_revision']: raise ValueError('Source revision differs')
    for model, digest in freeze['model_manifest_sha256'].items():
        folder = ROOT/'models'/model; manifest = t.read(folder/'manifest.json')
        payload = folder/('model.pt' if model == 'tcn-blend-v1' else 'models.joblib')
        if t.runtime.file_digest(folder/'manifest.json') != digest or t.runtime.file_digest(payload) != freeze['model_payload_sha256'][model]:
            raise ValueError('Frozen model changed')
    for record in source['files']:
        if t.runtime.file_digest(SOURCE/record['path']) != record['sha256']: raise ValueError('Source drift')
    paths = t.SOURCE_FILES + [CONFIG, Path(__file__), ROOT/'solution/src/possession/metrica_epts.py',
                            ROOT/'solution/tools/validate_tcn_transfer.py']
    protocol = dict(config=config, candidate_freeze_sha256=t.runtime.file_digest(FREEZE),
        download_manifest_sha256=t.runtime.file_digest(SOURCE/'source_manifest.json'),
        model_manifests=freeze['model_manifest_sha256'], model_payloads=freeze['model_payload_sha256'],
        source_sha256={str(p.relative_to(ROOT)): t.runtime.file_digest(p) for p in paths},
        noise_regimes=t.read(ROOT/'solution/configs/validation.json')['regimes'],
        contextual=t.read(ROOT/'solution/configs/nested_validation.json')['contextual'],
        packages={k: version(k) for k in ('torch', 'numpy', 'pandas', 'scikit-learn', 'scipy', 'joblib', 'threadpoolctl')})
    if (out/'protocol.json').exists() and t.read(out/'protocol.json') != protocol: raise ValueError('Run protocol drift')
    t.save(out/'protocol.json', protocol)
    meta = SOURCE/'Sample_Game_3_metadata.xml'
    inputs, alignment = read_epts_tracking(meta, SOURCE/'Sample_Game_3_tracking.txt')
    raw, core, ref_audit = game3_reference(meta, SOURCE/'Sample_Game_3_events.json', inputs['period'], config['guard_frames'])
    # Persist the reference before any inference, and keep it out of prediction APIs.
    refs = {'event_interiors': raw, 'guarded_dead': core}
    reference_path = out/'references.npz'
    if reference_path.exists():
        with np.load(reference_path, allow_pickle=False) as z:
            for key, arr in dict(raw=raw, guarded=core, period=inputs['period']).items(): np.testing.assert_array_equal(z[key], arr)
    else: np.savez_compressed(reference_path, raw=raw, guarded=core, period=inputs['period'])
    t.save(out/'reference_audit.json', ref_audit); t.save(out/'alignment.json', alignment)
    original = t.runtime.array_digest(inputs); ref_hash = t.runtime.file_digest(reference_path)
    intervals = t.labelled_intervals(core, inputs['period'])
    picks = np.random.RandomState(config['bootstrap_seed']).randint(0, len(intervals), (config['bootstrap_draws'], len(intervals)))
    torch.set_num_threads(config['threads']); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); torch.backends.mkldnn.deterministic = True
    base = t.runtime.load_bundle(ROOT/'models/possession-v1')
    research = t.runtime.current.load_research(ROOT/'models/research-specialists-clean', base)
    tcn = t.runtime.load_tcn(ROOT/'models/tcn-blend-v1', base, research)
    reports = []
    print(f'Game 3 aligned: {len(core)} frames; {(core != "X").mean():.2%} partial reference coverage', flush=True)
    with threadpool_limits(limits=config['threads']):
        for condition, seed in t.cases(config):
            stem = f'{condition}_{seed}'; path = out/(stem+'.json'); evidence_path = out/(stem+'_evidence.npz')
            if path.exists():
                report = t.read(path)
                if report['protocol_sha256'] != t.runtime.file_digest(out/'protocol.json') or report['evidence_sha256'] != t.runtime.file_digest(evidence_path):
                    raise ValueError('Existing case drift')
                reports.append(report); continue
            tick = time.perf_counter(); noisy, masks = t.noisy_inputs(inputs, condition, seed, protocol)
            fingerprint = t.runtime.array_digest(noisy); evidence = infer_pair(base, research, tcn, noisy)
            if t.runtime.array_digest(noisy) != fingerprint or t.runtime.array_digest(inputs) != original: raise ValueError('Input mutation')
            evidence.update(period=inputs['period'], **masks)
            subsets = {'all': np.ones(len(core), bool), **masks}
            scores = {name: {model: {subset: t.score_subset(ref, evidence[label], evidence[prob], subsets[subset]) for subset in SUBSETS}
                for model, label, prob in MODEL_KEYS} for name, ref in refs.items()}
            np.savez_compressed(evidence_path, **evidence)
            report = dict(condition=condition, seed=seed, input_fingerprint=fingerprint, native_input_fingerprint=original,
                protocol_sha256=t.runtime.file_digest(out/'protocol.json'), reference_sha256=ref_hash,
                evidence_sha256=t.runtime.file_digest(evidence_path), scores=scores, paired_interval_draws=paired(core, evidence, picks, intervals),
                noise=t.summarize_inputs(noisy), stress_fractions={k: float(v[inputs['period'] > 0].mean()) for k, v in masks.items()},
                elapsed_seconds=time.perf_counter()-tick, inference_reference_arrays_read=False)
            t.save(path, report); reports.append(report)
            print(f'Game 3 {stem}: frozen pair complete ({report["elapsed_seconds"]:.1f}s)', flush=True)
    if len(reports) != len(t.cases(config)): raise ValueError('Incomplete run')
    result = dict(status='complete', summary=aggregate(reports, config), protocol_sha256=t.runtime.file_digest(out/'protocol.json'),
        reference_sha256=ref_hash, reference_audit_sha256=t.runtime.file_digest(out/'reference_audit.json'),
        alignment_sha256=t.runtime.file_digest(out/'alignment.json'),
        case_report_sha256={f'{r["condition"]}_{r["seed"]}': t.runtime.file_digest(out/f'{r["condition"]}_{r["seed"]}.json') for r in reports},
        training_or_tuning_performed=False, complete_possession_scoring_performed=False, limits=config['policy'])
    t.save(out/'report.json', result)
    for item in result['summary']:
        a, b = [item['metrics']['guarded_dead'][model]['all'] for model, _, _ in MODEL_KEYS]
        print(item['condition'], 'partial accuracy', a['accuracy_on_labelled_subset'], '->', b['accuracy_on_labelled_subset'], flush=True)


if __name__ == '__main__': main()
