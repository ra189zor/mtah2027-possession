"""Replay held-out TCN emissions from raw tracking; never score the all-seven refit."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'solution'), str(ROOT/'solution/tools')]
import train_tcn as t
import tcn_predict as runtime
from possession.validation import corrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args(); out = args.experiment.resolve()
    cfg, sc, _, foundation = t.r.setup(); config = t.r.n.read(t.CONFIG)
    base = runtime.load_bundle(ROOT/'models/possession-v1')
    t.configure_torch_threads(config['threads']); torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.deterministic = True
    records = []
    with threadpool_limits(limits=config['threads']):
        for match in cfg['matches']:
            raw = runtime.read_tracking_npz(ROOT/'data/external/sportec'/match/'prepared/inputs.npz')
            inputs = corrupt(raw, foundation[2]['regimes']['moderate'], 17)
            expected = t.r.n.read(t.r.s.SOURCE/'prepared'/match/'index.json')['cases']['moderate_17']['noisy_input_fingerprint']
            if runtime.array_digest(inputs) != expected: raise ValueError('Raw corruption differs from the benchmark')
            folder = out/'outer'/match; fit = t.r.n.read(folder/'fit.json')
            if runtime.file_digest(folder/'model.pt') != fit['model_sha256']: raise ValueError('Fold model differs')
            payload = torch.load(folder/'model.pt', map_location='cpu', weights_only=True)
            if match in payload['training_matches']: raise ValueError('Held-out replay leaked')
            model = t.PossessionTCN(config); model.load_state_dict(payload['state_dict'])
            bundle = {'model': model, 'normalizer': payload['normalizer'], 'manifest': {'config': config}}
            before = runtime.array_digest(inputs)
            native = runtime.tcn_probabilities(bundle, inputs, base['manifest']['settings'], base['models']['owner_names'])
            with np.load(folder/'moderate_17_probabilities.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(native[:, 0], z['home'])
                np.testing.assert_array_equal(native[:, 1], z['dead'])
            with np.load(t.DIAGNOSIS/match/'moderate_17_evidence.npz', allow_pickle=False) as z:
                h, d, q = z['home'], z['dead'], z['events']
            h, d = t.combine_probabilities(h, d, native[:, 0], native[:, 1], inputs['period'], config['blend'])['tcn_blend']
            settings = base['manifest']['settings']
            prediction = t.decode_events(h, d, inputs['period'], q, settings['deployment']['decoder'], 4, sc)['labels']
            np.testing.assert_array_equal(prediction, np.load(folder/'moderate_17_tcn_blend.npy'))
            if runtime.array_digest(inputs) != before: raise ValueError('Raw inputs changed')
            records.append({'held_out': match, 'exact_raw_probability_replay': True,
                'exact_blend_decoder_replay': True, 'input_sha256': before})
            print(match+': raw-tracking runtime reproduces held-out probabilities and blended predictions exactly', flush=True)
    result = {'status': 'passed', 'records': records, 'reference_arrays_read': False,
        'scope': 'New TCN emission extraction from raw inputs plus frozen held-out comparator evidence; no final-fit accuracy scoring.',
        'source_sha256': runtime.file_digest(Path(__file__))}
    t.r.save(out/'raw_runtime_verification.json', result)


if __name__ == '__main__': main()
