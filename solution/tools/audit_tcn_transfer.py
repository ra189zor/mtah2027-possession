"""Audit frozen transfer artifacts and replay both raw-input moderate comparisons."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'solution/tools'))
import validate_tcn_transfer as t


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve()
    report, protocol = t.read(out/'report.json'), t.read(out/'protocol.json')
    if report['status'] != 'complete' or len(report['summary']) != 8: raise ValueError('Incomplete transfer summary')
    counts = {'hash_checks': 0, 'confusion_recalculations': 0, 'exact_pipeline_replays': 0, 'active_unknowns': 0}
    def check(path, digest):
        if t.runtime.file_digest(path) != digest: raise ValueError('Artifact drift: '+str(path))
        counts['hash_checks'] += 1
    check(out/'protocol.json', report['protocol_sha256'])
    for name, digest in protocol['source_sha256'].items(): check(ROOT/name, digest)
    for name, digest in protocol['model_manifests'].items(): check(ROOT/'models'/name/'manifest.json', digest)
    check(t.SOURCE/'source_manifest.json', protocol['download_manifest_sha256'])
    for record in t.read(t.SOURCE/'source_manifest.json')['files']:
        check(t.SOURCE/record['path'], record['sha256'])
    base = t.runtime.load_bundle(ROOT/'models/possession-v1')
    research = t.runtime.current.load_research(ROOT/'models/research-specialists-clean', base)
    tcn = t.runtime.load_tcn(ROOT/'models/tcn-blend-v1', base, research)
    if tcn['manifest']['training_matches'] != t.read(ROOT/'solution/configs/nested_validation.json')['matches']:
        raise ValueError('Unexpected TCN training cohort')
    torch.set_num_threads(protocol['config']['threads'])
    if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); torch.backends.mkldnn.deterministic = True
    with threadpool_limits(limits=protocol['config']['threads']):
        for game in protocol['config']['games']:
            folder = out/f'game_{game}'
            check(folder/'report.json', report['game_report_sha256'][str(game)])
            details = t.read(folder/'report.json'); check(folder/'references.npz', details['reference_sha256'])
            with np.load(folder/'references.npz', allow_pickle=False) as z:
                refs = {'event_interiors': z['raw'], 'guarded_dead': z['guarded']}; period = z['period']
            if any(not np.isin(v, ['H', 'A', 'D', 'X']).all() for v in refs.values()): raise ValueError('Invalid partial reference')
            if ((refs['guarded_dead'] == 'D') & (refs['event_interiors'] != 'D')).any():
                raise ValueError('Guard introduced dead reference frames')
            # Guard may remove D but must preserve H/A and X exactly.
            keep = refs['event_interiors'] != 'D'
            np.testing.assert_array_equal(refs['event_interiors'][keep], refs['guarded_dead'][keep])
            intervals = t.labelled_intervals(refs['guarded_dead'], period)
            picks = np.random.RandomState(protocol['config']['bootstrap_seed']).randint(0, len(intervals),
                (protocol['config']['bootstrap_draws'], len(intervals)))
            for condition, seed in t.cases(protocol['config']):
                stem = f'{condition}_{seed}'; check(folder/(stem+'.json'), details['case_report_sha256'][stem])
                case = t.read(folder/(stem+'.json'))
                check(folder/'references.npz', case['reference_sha256'])
                check(folder/(stem+'_evidence.npz'), case['evidence_sha256'])
                if case['protocol_sha256'] != report['protocol_sha256']: raise ValueError('Case protocol differs')
                with np.load(folder/(stem+'_evidence.npz'), allow_pickle=False) as z:
                    evidence = {k: z[k] for k in z.files}
                np.testing.assert_array_equal(period, evidence['period'])
                for prediction in ('current_labels', 'blend_labels'):
                    t.runtime.validate_prediction(evidence[prediction], period)
                    counts['active_unknowns'] += int((evidence[prediction][period > 0] == 'X').sum())
                for ref_name, ref in refs.items():
                    for model, label, prob in [('current', 'current_labels', 'current_dead'), ('tcn_blend', 'blend_labels', 'blend_dead')]:
                        for subset, stored in case['scores'][ref_name][model].items():
                            mask = np.ones(len(ref), bool) if subset == 'all' else evidence[subset]
                            recalculated = t.score_subset(ref, evidence[label], evidence[prob], mask)
                            if recalculated != stored: raise ValueError('Stored partial metrics do not match arrays')
                            counts['confusion_recalculations'] += 1
                core = refs['guarded_dead']
                for name, mask in [('all', np.ones(len(core), bool)), ('pass_only', np.isin(core, ['H', 'A'])),
                    ('synthetic_missing_pass', evidence['synthetic_ball_missing'] & np.isin(core, ['H', 'A']))]:
                    draws = None
                    if (mask & (core != 'X')).any():
                        values = t.paired_interval_draws(core, evidence['current_labels'], evidence['blend_labels'], mask, intervals, picks)
                        draws = [float(v) if np.isfinite(v) else None for v in values]
                    if draws != case['paired_interval_draws'][name]: raise ValueError('Paired interval draws differ')
                if condition == 'moderate' and seed == 17:
                    prefix = f'Sample_Game_{game}_Raw'; source = t.SOURCE/'data'/f'Sample_Game_{game}'
                    raw = t.read_public_tracking(source/(prefix+'TrackingData_Home_Team.csv'), source/(prefix+'TrackingData_Away_Team.csv'))
                    inputs, masks = t.noisy_inputs(raw, condition, seed, protocol)
                    if t.runtime.array_digest(inputs) != case['input_fingerprint']: raise ValueError('Noisy replay differs')
                    replay = t.runtime.predict(base, research, tcn, inputs)
                    for new, old in [('labels', 'blend_labels'), ('home_probability', 'blend_home'), ('dead_probability', 'blend_dead')]:
                        np.testing.assert_array_equal(replay[new], evidence[old])
                    for key, value in masks.items(): np.testing.assert_array_equal(value, evidence[key])
                    counts['exact_pipeline_replays'] += 1
                if condition == 'native':
                    if not case['native_comparator_exact']: raise ValueError('Prior native comparison was not reproduced')
                    previous = t.read(t.PREVIOUS/f'game_{game}_report.json')
                    check(t.PREVIOUS/f'game_{game}_evidence.npz', previous['evidence_sha256'])
                    with np.load(t.PREVIOUS/f'game_{game}_evidence.npz', allow_pickle=False) as z:
                        np.testing.assert_array_equal(evidence['current_labels'], z['labels'])
            print(f'Game {game}: all partial scores recalculated; fresh moderate pipeline replay exact', flush=True)
    result = {'status': 'passed', **counts, 'models_unchanged': True,
        'source_sha256': t.runtime.file_digest(Path(__file__)),
        'scope': 'Integrity and repeatability, not new independent accuracy evidence.'}
    t.save(out/'audit.json', result); print(result, flush=True)


if __name__ == '__main__': main()
