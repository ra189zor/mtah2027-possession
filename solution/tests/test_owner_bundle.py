"""Prevent accidental joint-dead fusion and unbound owner-candidate substitutions."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH: import owner_predict as runtime


@unittest.skipUnless(HAS_TORCH, 'Optional TCN environment required')
class OwnerBundleTests(unittest.TestCase):
    def fixture(self, root):
        components = []
        for name in ('base', 'specialists', 'tcn'):
            folder = root/name; folder.mkdir(); manifest = {'model_sha256': name, 'runtime_sha256': {}, 'training_matches': ['external']}
            if name == 'base': manifest['settings'] = {'deployment': {'decoder': runtime.SETTINGS['decoder'], 'epsilon': 1e-6}}
            (folder/'manifest.json').write_text(json.dumps(manifest))
            components.append({'folder': folder, 'manifest': manifest})
        identities = runtime.component_identities(*components)
        folder = root/'owner'; folder.mkdir()
        manifest = dict(schema_version=1, kind=runtime.KIND, status='complete', settings=copy.deepcopy(runtime.SETTINGS),
            components=identities, model_sha256=runtime.model_identity(identities),
            model_identity_kind='sha256_of_component_payload_hashes', training_matches=['external'],
            runtime_sha256=runtime.runtime_inventory(*components))
        (folder/'manifest.json').write_text(json.dumps(manifest))
        return folder, manifest, components

    def test_load_and_settings_component_identity_rejection(self):
        with tempfile.TemporaryDirectory() as name:
            folder, manifest, components = self.fixture(Path(name))
            runtime.load_owner(folder, *components)
            variants = [dict(kind='tcn_joint_blend'), dict(settings=runtime.SETTINGS | {'home_weight': .6}),
                dict(components={}), dict(model_sha256='bad'), dict(runtime_sha256={}), dict(training_matches=['wrong'])]
            for change in variants:
                with self.subTest(change=change):
                    (folder/'manifest.json').write_text(json.dumps(manifest | change))
                    with self.assertRaises(ValueError): runtime.load_owner(folder, *components)

    def test_component_source_drift_and_decoder_drift_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            folder, manifest, components = self.fixture(Path(name))
            (components[0]['folder']/'manifest.json').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'component identities'): runtime.load_owner(folder, *components)
        with tempfile.TemporaryDirectory() as name:
            folder, manifest, components = self.fixture(Path(name))
            components[0]['manifest']['settings']['deployment']['decoder'] = {'stop': 20}
            with self.assertRaisesRegex(ValueError, 'decoder profile'): runtime.load_owner(folder, *components)

    def test_tcn_dead_head_cannot_change_retained_dead_or_event_evidence(self):
        period = np.array([1, 1, 0, 2, 2], np.int8)
        inputs = dict(players_xy=np.zeros((5, 2, 2), np.float32), ball_xy=np.zeros((5, 2), np.float32),
            team_of_slot=np.array([0, 1], np.int8), period=period, fps=np.array(25.), pitch_m=np.array([105., 68.]))
        h = np.array([.1, .2, .5, .7, .8], np.float32); d = np.array([.8, .9, 1., .3, .4], np.float32)
        q = np.ones((5, 3), np.float32); network_h = 1-h; labels = np.array(list('HDXAD'))
        base = {'manifest': {'settings': {'switch': {}}}, 'models': {'owner_names': []}}
        bundle = {'manifest': {'settings': copy.deepcopy(runtime.SETTINGS)}}
        def decode(home, dead, p, events, *args):
            np.testing.assert_array_equal(home, np.full(5, .5, np.float32))
            self.assertIs(dead, d); self.assertIs(events, q)
            return {'labels': labels.copy(), 'path_cost': 0.}
        with patch.object(runtime.t.current, 'predict', side_effect=lambda *a: dict(home_probability=h, dead_probability=d,
                event_probability=q, labels=labels.copy(), input_fingerprint=runtime.t.array_digest(inputs))), \
             patch.object(runtime.t.current, 'decode_events', side_effect=decode), \
             patch.object(runtime.t, 'tcn_probabilities') as network:
            for tcn_dead in (np.zeros(5), np.ones(5)):
                network.return_value = np.column_stack((network_h, tcn_dead)).astype(np.float32)
                result = runtime.predict(base, {}, {}, bundle, inputs)
                self.assertIs(result['dead_probability'], d); self.assertIs(result['event_probability'], q)
                np.testing.assert_array_equal(result['labels'], labels)


if __name__ == '__main__': unittest.main()
