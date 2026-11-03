"""The optional TCN runtime rejects mismatched weights and dependencies."""
from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    import tcn_predict as t


@unittest.skipUnless(HAS_TORCH, 'Optional TCN environment required')
class TCNBundleTests(unittest.TestCase):
    def test_bundle_round_trip_and_mismatch_rejection(self):
        cfg = {'input_features': 3, 'hidden_channels': 8, 'kernel_size': 3, 'dilations': [1, 2], 'dropout': 0.}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); base, research = ({'folder': root/name} for name in ('base', 'research'))
            for item in (base, research):
                item['folder'].mkdir(); (item['folder']/'manifest.json').write_text('{}')
            folder = root/'tcn'; folder.mkdir(); model = t.PossessionTCN(cfg)
            torch.save({'state_dict': model.state_dict(), 'normalizer': {'mean': [0]*3, 'scale': [1]*3},
                'config': cfg, 'training_matches': ['external']}, folder/'model.pt')
            manifest = {'kind': 'tcn_joint_blend', 'status': 'complete', 'config': cfg, 'training_matches': ['external'],
                'torch_version': t.version('torch'), 'model_sha256': t.file_digest(folder/'model.pt'),
                'base_manifest_sha256': t.file_digest(base['folder']/'manifest.json'),
                'research_manifest_sha256': t.file_digest(research['folder']/'manifest.json'),
                'runtime_sha256': {n: t.file_digest(t.ROOT/n) for n in t.SOURCES}}
            path = folder/'manifest.json'; path.write_text(json.dumps(manifest))
            loaded = t.load_tcn(folder, base, research)
            self.assertTrue(all(torch.equal(v, loaded['model'].state_dict()[k]) for k, v in model.state_dict().items()))
            for key, value in [('model_sha256', 'bad'), ('base_manifest_sha256', 'bad'),
                               ('research_manifest_sha256', 'bad'), ('torch_version', 'bad'), ('training_matches', ['wrong'])]:
                path.write_text(json.dumps(manifest | {key: value}))
                with self.assertRaises(ValueError): t.load_tcn(folder, base, research)


if __name__ == '__main__': unittest.main()
