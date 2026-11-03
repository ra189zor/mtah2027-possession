"""Verify TCN split/scaler provenance, artifacts and deterministic fold replays."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'solution/tools'), str(ROOT/'solution/src'), str(ROOT/'solution')]
import train_tcn as t
import research_predict
from possession.inference import load_bundle
from possession.inputs import file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve()
    cfg, sc, _, _ = t.r.setup(); config = t.r.n.read(t.CONFIG)
    protocol = t.r.n.read(out/'protocol.json'); summary = t.r.n.read(out/'report.json')
    if summary['status'] != 'complete' or len(summary['results']) != 560:
        raise ValueError('Incomplete experiment')
    if protocol['config'] != config or protocol['foundation_source_sha256'] != t.r.s.setup()[2]:
        raise ValueError('Configuration/foundation drift')
    checks = 0
    def check(path, expected):
        nonlocal checks
        if file_digest(path) != expected: raise ValueError('Artifact drift: '+str(path))
        checks += 1
    for name, digest in protocol['source_sha256'].items(): check(ROOT/name, digest)
    check(t.DIAGNOSIS/'report.json', protocol['diagnosis_sha256'])
    check(ROOT/'solution/configs/deployment.json', protocol['deployment_config_sha256'])
    torch.set_num_threads(config['threads']); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); torch.backends.mkldnn.deterministic = True
    replays = scalers = abstentions = 0
    with threadpool_limits(limits=config['threads']):
        for match in cfg['matches']:
            folder = out/'outer'/match; prepared = out/'prepared'/match
            check(folder/'report.json', summary['input_report_sha256'][match])
            report = t.r.n.read(folder/'report.json'); fit = t.r.n.read(folder/'fit.json')
            check(folder/'model.pt', fit['model_sha256'])
            if report['model_sha256'] != fit['model_sha256']: raise ValueError('Report/model mismatch')
            bundle = torch.load(folder/'model.pt', map_location='cpu', weights_only=True)
            expected = [m for m in cfg['matches'] if m != match]
            for record in (bundle, fit, report):
                if record['training_matches'] != expected: raise ValueError('Outer split violation')
            if bundle['held_out'] != match or bundle['protocol_sha256'] != file_digest(out/'protocol.json'):
                raise ValueError('Model/protocol mismatch')
            if bundle['epochs'] != config['epochs'] or len(fit['history']) != config['epochs']:
                raise ValueError('Incomplete fixed-epoch training')
            for training, digest in bundle['training_index_sha256'].items():
                check(out/'prepared'/training/'index.json', digest)
            cases, _ = t.load_training(expected, out, cfg)
            recomputed = t.fit_normalizer(case['x'][::5] for case in cases.values())
            for key, value in recomputed.items():
                np.testing.assert_array_equal(value, np.asarray(bundle['normalizer'][key], dtype=value.dtype))
            scalers += 1
            del cases
            index = t.r.n.read(prepared/'index.json')
            for name, digest in index['files'].items(): check(prepared/name, digest)
            with np.load(prepared/'timing.npz', allow_pickle=False) as z:
                period, indices, small_period = z['period'], z['indices'], z['sampled_period']
            for name, digest in report['files'].items():
                check(folder/name, digest)
                if name.endswith('.npy'):
                    prediction = np.load(folder/name, allow_pickle=False)
                    if not np.isin(prediction[period > 0], ['H', 'A', 'D']).all():
                        raise ValueError('Invalid/unknown active prediction')
                    abstentions += int((prediction[period > 0] == 'X').sum())
            matrix = np.load(prepared/index['cases']['moderate_17'], mmap_mode='r', allow_pickle=False)
            model = t.PossessionTCN(config); model.load_state_dict(bundle['state_dict'])
            sampled = t.infer_sampled(model, matrix, small_period, bundle['normalizer'], config)
            native = t.upsample_probabilities(sampled, indices, period)
            with np.load(folder/'moderate_17_probabilities.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(sampled, z['sampled'])
                np.testing.assert_array_equal(native[:, 0], z['home'])
                np.testing.assert_array_equal(native[:, 1], z['dead'])
            evidence_path = t.DIAGNOSIS/match/'moderate_17_evidence.npz'
            old = t.r.n.read(t.DIAGNOSIS/match/'report.json')
            check(evidence_path, old['files'][evidence_path.name])
            with np.load(evidence_path, allow_pickle=False) as z:
                h, d, events = z['home'], z['dead'], z['events']
            profile = t.r.n.read(ROOT/'solution/configs/deployment.json')['decoder']
            candidates = t.combine_probabilities(h, d, native[:, 0], native[:, 1], period, config['blend'])
            for name, (hh, dd) in candidates.items():
                prediction = t.decode_events(hh, dd, period, events, profile, 4, sc)['labels']
                np.testing.assert_array_equal(prediction, np.load(folder/('moderate_17_'+name+'.npy')))
                replays += 1
            print(match+': six-match scaler reproduced; probabilities and all four decoders reproduced exactly', flush=True)
        base = load_bundle(ROOT/'models/possession-v1')
        research_predict.load_research(ROOT/'models/research-specialists-clean', base)
    result = {'status': 'passed', 'hash_checks': checks, 'exact_training_scaler_recomputations': scalers,
              'exact_probability_replays': scalers, 'exact_candidate_decoder_replays': replays,
              'active_unknowns_all_280_predictions': abstentions,
              'original_deployment_bundles_valid': True,
              'source_sha256': file_digest(Path(__file__)),
              'limits': 'Reproducibility and exclusion audit; not a new accuracy estimate or independent test set.'}
    t.r.save(out/'audit.json', result); print(result, flush=True)


if __name__ == '__main__': main()
