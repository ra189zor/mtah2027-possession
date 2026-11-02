"""Read-only integrity/exclusion audit plus seven exact boundary prediction replays."""
import argparse
from pathlib import Path
import sys
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'solution/tools'), str(ROOT/'solution/src'), str(ROOT/'solution')]
import train_stoppage_boundaries as t
import research_predict
from possession.inference import load_bundle
from possession.inputs import file_digest
from possession.stoppage_boundaries import decode_boundaries
from possession.event_decoder import decode_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve()
    config = t.r.n.read(t.CONFIG)
    cfg, sc, _, _ = t.r.setup()
    protocol = t.r.n.read(out/'protocol.json')
    checks = 0
    for name, digest in protocol['source_sha256'].items():
        if file_digest(ROOT/name) != digest: raise ValueError('Experiment source drift: '+name)
        checks += 1
    summary = t.r.n.read(out/'report.json')
    traces, replays = 0, 0
    with threadpool_limits(limits=2):
        for match in cfg['matches']:
            report = t.r.n.read(out/match/'report.json')
            if file_digest(out/match/'report.json') != summary['input_report_sha256'][match]:
                raise ValueError('Fold report drift')
            model_path = out/match/'model.joblib'
            if file_digest(model_path) != report['model_sha256']: raise ValueError('Model drift')
            bundle = joblib.load(model_path)
            expected = [m for m in cfg['matches'] if m != match]
            if bundle['training_matches'] != expected or report['training_matches'] != expected:
                raise ValueError('Wrong outer training membership')
            for trace in bundle['lineage']:
                training_fixture = trace['training_fixture']
                if set(trace['excluded']) != {match, training_fixture} or set(trace['upstream_training_matches']) != set(cfg['matches'])-{match, training_fixture}:
                    raise ValueError('Cross-fitted training leakage')
                if file_digest(Path(trace['probability_file'])) != trace['sha256']: raise ValueError('Upstream drift')
                traces += 1; checks += 1
            for name, digest in report['files'].items():
                if file_digest(out/match/name) != digest: raise ValueError('Prediction artifact drift')
                prediction = np.load(out/match/name, allow_pickle=False)
                if not np.isin(prediction, ['H', 'A', 'D', 'X']).all(): raise ValueError('Bad prediction label')
                checks += 1
            path = t.DIAGNOSIS/match/'moderate_17_evidence.npz'
            diagnostic = t.r.n.read(t.DIAGNOSIS/match/'report.json')
            if file_digest(path) != diagnostic['files'][path.name]: raise ValueError('Diagnostic evidence drift')
            with np.load(path, allow_pickle=False) as z:
                h, d, events, period = z['home'], z['dead'], z['events'], z['period']
            x, _ = t.features(match, 'moderate', 17, d, config)
            q = t.r.probabilities(bundle['model'], x[:, bundle['columns']])
            profile = t.r.n.read(ROOT/'solution/configs/deployment.json')['decoder']
            prediction = decode_boundaries(h, d, period, events, q, profile, sc, config, config['reward_strength'])['labels']
            np.testing.assert_array_equal(prediction, np.load(out/match/'moderate_17_boundary.npy'))
            original = decode_events(h, d, period, events, profile, 4, sc)['labels']
            np.testing.assert_array_equal(original, np.load(out/match/'moderate_17_current.npy'))
            if (prediction[period > 0] == 'X').any(): raise ValueError('Active abstention')
            replays += 1; checks += 3
            print(match+': exclusion, integrity and exact replay passed', flush=True)
    base = load_bundle(ROOT/'models/possession-v1')
    research_predict.load_research(ROOT/'models/research-specialists-clean', base)
    result = {'status': 'passed', 'hash_checks': checks, 'cross_fitting_traces': traces,
              'exact_boundary_replays': replays, 'exact_original_decoder_replays': replays,
              'original_deployment_bundles_unchanged': True,
              'tests': 'Run separately; this audit does not claim to execute the unit suite.'}
    t.r.save(out/'audit.json', result)
    print(result, flush=True)


if __name__ == '__main__': main()
