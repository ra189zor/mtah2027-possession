"""Frozen-model external transfer check on two previously unused public samples."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'solution/src'), str(ROOT/'solution')]
import research_predict
from possession.inference import load_bundle
from possession.inputs import file_digest, array_digest
from possession.metrica_external import read_public_tracking, event_binary_reference

SOURCE = ROOT/'data/external/metrica_public'
BASE = ROOT/'models/possession-v1'
RESEARCH = ROOT/'models/research-specialists-clean'


def binary_score(reference, predicted, probability, ball_visible):
    out = {}
    for name, subset in [('all', np.ones(len(reference), bool)), ('ball_visible', ball_visible), ('ball_missing', ~ball_visible)]:
        valid = (reference >= 0) & subset
        if not valid.any():
            out[name] = None
            continue
        y, p = reference[valid] == 1, predicted[valid] == 'D'
        tp, fp, fn, tn = int((y&p).sum()), int((~y&p).sum()), int((y&~p).sum()), int((~y&~p).sum())
        out[name] = {'frames': int(valid.sum()), 'true_dead_frames': int(y.sum()),
                     'dead_recall': tp/(tp+fn) if tp+fn else None,
                     'dead_precision': tp/(tp+fp) if tp+fp else None,
                     'alive_recall': tn/(tn+fp) if tn+fp else None, 'accuracy_on_labelled_subset': (tp+tn)/len(y),
                     'confusion_alive_dead': [[tn, fp], [fn, tp]],
                     'brier_on_labelled_subset': float(np.mean((probability[valid]-y)**2))}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-evidence', type=Path, help='Rescore hash-checked frozen predictions without running inference')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = {'version': 1, 'source_revision': 'e706dd506b360d69d9d123d5b8026e7294b13996',
        'games': [1, 2], 'model': 'frozen corrected combined, all-seven-DFL weights',
        'no_training_or_selection': True, 'guard_frames': 25,
        'reference_policy': 'Partial binary event-derived references only. Alive = successful-pass interiors. Dead = BALL OUT or SHOT OUT/GOAL end to first subsequent explicit SET PIECE or GOAL KICK, only when no intervening live action. No foul or missing-ball inference. Conflicts and unlabelled frames excluded. Report untrimmed and 1-second guarded dead interiors; no full-match accuracy or spellF1 claim.',
        'source_sha256': {str(p.relative_to(ROOT)): file_digest(p) for p in
            [Path(__file__), ROOT/'solution/src/possession/metrica_external.py']},
        'base_manifest_sha256': file_digest(BASE/'manifest.json'),
        'research_manifest_sha256': file_digest(RESEARCH/'manifest.json'),
        'download_manifest_sha256': file_digest(SOURCE/'source_manifest.json')}
    if args.reuse_evidence:
        previous_protocol = json.loads((args.reuse_evidence/'protocol.json').read_text())
        for key in ('base_manifest_sha256', 'research_manifest_sha256', 'download_manifest_sha256'):
            if previous_protocol[key] != protocol[key]:
                raise ValueError('Frozen transfer evidence has different models or inputs')
        protocol['reused_evidence_directory'] = str(args.reuse_evidence.resolve())
    protocol_path = output/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Transfer protocol changed; use a new folder')
    protocol_path.write_text(json.dumps(protocol, indent=2)+'\n')
    download = json.loads((SOURCE/'source_manifest.json').read_text())
    for item in download['files']:
        if file_digest(SOURCE/item['path']) != item['sha256']:
            raise ValueError('Public sample source drift')
    base = load_bundle(BASE)
    bundle = research_predict.load_research(RESEARCH, base)
    reports = []
    for game in protocol['games']:
        folder = SOURCE/'data'/f'Sample_Game_{game}'
        prefix = f'Sample_Game_{game}_Raw'
        inputs = read_public_tracking(folder/(prefix+'TrackingData_Home_Team.csv'), folder/(prefix+'TrackingData_Away_Team.csv'))
        with (folder/(prefix+'EventsData.csv')).open(newline='') as handle:
            events = list(csv.DictReader(handle))
        labels, core, audit = event_binary_reference(events, inputs['period'], protocol['guard_frames'])
        prepared = SOURCE/'prepared'/f'game_{game}'
        prepared.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(prepared/'inputs.npz', **inputs)
        np.savez_compressed(prepared/'binary_reference.npz', binary=labels, core=core)
        (prepared/'reference_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
        # Prediction API accepts tracking only; event references never enter features.
        if args.reuse_evidence:
            previous = json.loads((args.reuse_evidence/f'game_{game}_report.json').read_text())
            path = args.reuse_evidence/f'game_{game}_evidence.npz'
            if previous['input_fingerprint'] != array_digest(inputs) or previous['evidence_sha256'] != file_digest(path):
                raise ValueError('Frozen transfer evidence drift')
            with np.load(path, allow_pickle=False) as z:
                result = {'labels': z['labels'], 'dead_probability': z['dead'],
                          'home_probability': z['home'], 'event_probability': z['events']}
        else:
            result = research_predict.predict(base, bundle, inputs, 'combined')
        visible = np.isfinite(inputs['ball_xy']).all(1)
        raw = np.where(result['dead_probability'] >= .5, 'D', 'H')
        report = {'game': game, 'frames': len(labels), 'input_fingerprint': array_digest(inputs),
                  'ball_missing_fraction': float((~visible).mean()), 'reference_audit': audit,
                  'coverage': float((labels >= 0).mean()),
                  'results': {name: {variant: binary_score(ref, prediction, result['dead_probability'], visible)
                     for variant, prediction in [('combined_decoded', result['labels']), ('raw_dead_half', raw)]}
                     for name, ref in [('event_derived', labels), ('guarded_dead_interiors', core)]}}
        evidence_path = output/f'game_{game}_evidence.npz'
        np.savez_compressed(evidence_path, labels=result['labels'], dead=result['dead_probability'],
                            home=result['home_probability'], events=result['event_probability'], period=inputs['period'])
        report['evidence_sha256'] = file_digest(evidence_path)
        reports.append(report)
        (output/f'game_{game}_report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(f"Game {game}: coverage={report['coverage']:.3f}, "+json.dumps(report['results']['guarded_dead_interiors']['combined_decoded']['all']), flush=True)
    (output/'report.json').write_text(json.dumps({'status': 'complete', 'protocol': protocol,
        'reports': reports, 'limitations': 'Two new external fixtures; partial event-derived binary reference, not independent framewise manual truth or complete possession intervals. Same vendor but a different curated product from challenge ATD. Naturally missing ball may reflect provider processing. No model selection or training.'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
