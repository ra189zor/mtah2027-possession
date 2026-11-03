"""Post-result fusion ablations; explicitly exploratory on already inspected samples."""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'solution/tools'))
import validate_tcn_transfer as t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); source, out = args.source.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    original = t.read(source/'report.json'); original_protocol = t.read(source/'protocol.json')
    if original['status'] != 'complete' or t.read(source/'audit.json')['status'] != 'passed':
        raise ValueError('Require audited frozen transfer results')
    config = t.read(ROOT/'solution/configs/tcn.json')
    if config['blend'] != .5 or 'tcn_owner_blend' not in config['candidates']:
        raise ValueError('Require the already prespecified DFL ownership-only ablation')
    protocol = {'kind': 'post-result exploratory decomposition', 'blend': .5,
        'source_report_sha256': t.runtime.file_digest(source/'report.json'),
        'source_protocol_sha256': t.runtime.file_digest(source/'protocol.json'),
        'source_sha256': t.runtime.file_digest(Path(__file__)),
        'original_tcn_config_sha256': t.runtime.file_digest(ROOT/'solution/configs/tcn.json'),
        'policy': 'After observing the frozen native stoppage regression, reuse the original DFL owner-only blend and a dead-only diagnostic. Same 50/50 coefficient, same old event evidence and decoder. No fitting/tuning or runtime changes. These reused fixtures cannot independently validate selection of a new winner.'}
    if (out/'protocol.json').exists() and t.read(out/'protocol.json') != protocol: raise ValueError('Diagnostic protocol changed')
    t.save(out/'protocol.json', protocol)
    base = t.runtime.load_bundle(ROOT/'models/possession-v1'); settings = base['manifest']['settings']
    for name, digest in original_protocol['model_manifests'].items():
        if t.runtime.file_digest(ROOT/'models'/name/'manifest.json') != digest: raise ValueError('Models changed after transfer')
    rows, files = [], {}
    for game in original_protocol['config']['games']:
        folder = source/f'game_{game}'; game_report = t.read(folder/'report.json')
        if t.runtime.file_digest(folder/'report.json') != original['game_report_sha256'][str(game)]: raise ValueError('Source game report drift')
        if t.runtime.file_digest(folder/'references.npz') != game_report['reference_sha256']: raise ValueError('Reference drift')
        with np.load(folder/'references.npz', allow_pickle=False) as z: reference, period = z['guarded'], z['period']
        for condition, seed in t.cases(original_protocol['config']):
            stem = f'{condition}_{seed}'; case = t.read(folder/(stem+'.json')); path = folder/(stem+'_evidence.npz')
            if t.runtime.file_digest(path) != case['evidence_sha256']: raise ValueError('Source evidence drift')
            with np.load(path, allow_pickle=False) as z: e = {k: z[k] for k in z.files}
            h, d = t.runtime.combine_probabilities(e['current_home'], e['current_dead'], e['tcn_home'], e['tcn_dead'], period, .5)['tcn_owner_blend']
            np.testing.assert_array_equal(d, e['current_dead'])
            candidates = {'owner_only': (h, d), 'dead_only': (e['current_home'], e['blend_dead'])}
            for model, (home, dead) in candidates.items():
                prediction = t.runtime.current.decode_events(home, dead, period, e['current_events'],
                    settings['deployment']['decoder'], 4, settings['switch'], settings['deployment']['epsilon'])['labels']
                t.runtime.validate_prediction(prediction, period)
                path = out/f'game_{game}_{stem}_{model}.npy'; np.save(path, prediction, allow_pickle=False)
                files[path.name] = t.runtime.file_digest(path)
                scores = {key: t.score_subset(reference, prediction, dead, mask) for key, mask in
                    [('all', np.ones(len(reference), bool)), ('synthetic_ball_missing', e['synthetic_ball_missing']), ('ghost', e['ghost'])]}
                rows.append({'game': game, 'condition': condition, 'seed': seed, 'model': model, 'scores': scores})
            print(f'Exploratory game {game} {stem}: owner-only/dead-only decomposed', flush=True)
    summary = []
    for game in original_protocol['config']['games']:
        for condition in original_protocol['config']['conditions']:
            for model in ('owner_only', 'dead_only'):
                group = [v for v in rows if v['game'] == game and v['condition'] == condition and v['model'] == model]
                metrics = {}
                for subset in ('all', 'synthetic_ball_missing', 'ghost'):
                    values = [v['scores'][subset] for v in group if v['scores'][subset] is not None]
                    metrics[subset] = {k: float(np.mean([v[k] for v in values if v[k] is not None]))
                        if any(v[k] is not None for v in values) else None for k in
                        ('accuracy_on_labelled_subset', 'pass_team_accuracy', 'dead_recall', 'dead_precision')}
                summary.append({'game': game, 'condition': condition, 'model': model, 'metrics': metrics})
                if model == 'owner_only':
                    print(f"Game {game} {condition} owner-only: partial={metrics['all']['accuracy_on_labelled_subset']:.4f}; pass={metrics['all']['pass_team_accuracy']:.4f}; dead recall={metrics['all']['dead_recall']:.4f}", flush=True)
    t.save(out/'report.json', {'status': 'complete', 'exploratory': True, 'protocol_sha256': t.runtime.file_digest(out/'protocol.json'),
        'results': rows, 'summary': summary, 'files': files, 'training_performed': False, 'deployment_changed': False,
        'limits': protocol['policy']})


if __name__ == '__main__': main()
