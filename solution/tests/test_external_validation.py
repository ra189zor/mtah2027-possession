"""Guard target isolation, temporal boundaries and deterministic external stress tests."""
import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'solution/src'))
sys.path.insert(0, str(ROOT / 'solution/tools'))
sys.path.insert(0, str(ROOT / 'more-than-a-hack-2027-main repo/evaluation'))
from possession.validation import check_inputs, corrupt, nearest_team_codes, period_blocks
from prepare_sportec import labels_from_codes, prepare
from benchmark_external import benchmark
import possession_metrics as pm


class ExternalValidationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'solution/configs/validation.json').read_text())
        n = 2001
        self.inputs = {'players_xy': np.tile(np.array([[[0., 0.], [10., 0.]]], dtype=np.float32), (n, 1, 1)),
            'ball_xy': np.zeros((n, 2), dtype=np.float32),
            'team_of_slot': np.array([0, 1], dtype=np.int8),
            'period': np.r_[np.ones(1000), 0, np.full(1000, 2)].astype(np.int8),
            'pitch_m': np.array([105., 68.], dtype=np.float32), 'fps': np.array(25)}

    def test_label_mapping_and_dead_precedence(self):
        result = labels_from_codes(np.array([0, 0, 1, 1, 1, -1]), np.array([1, 2, 1, 2, -1, 1]))
        np.testing.assert_array_equal(result, ['D', 'D', 'H', 'A', 'X', 'X'])

    def test_reject_target_field_in_model_inputs(self):
        polluted = dict(self.inputs, ball_status=np.zeros(2001))
        with self.assertRaises(ValueError):
            check_inputs(polluted)

    def test_corruption_repeats_without_mutating_original(self):
        original = copy.deepcopy(self.inputs)
        a = corrupt(self.inputs, self.config['regimes']['moderate'], 17)
        b = corrupt(self.inputs, self.config['regimes']['moderate'], 17)
        for key in self.inputs:
            np.testing.assert_array_equal(self.inputs[key], original[key])
            np.testing.assert_array_equal(a[key], b[key])

    def test_clean_preserves_inputs(self):
        result = corrupt(self.inputs, self.config['regimes']['clean'], 17)
        for key in self.inputs:
            np.testing.assert_array_equal(result[key], self.inputs[key])

    def test_half_separator_is_not_a_playing_frame(self):
        self.assertEqual(period_blocks(self.inputs['period']), [(0, 1000), (1001, 2001)])
        self.assertEqual(nearest_team_codes(self.inputs)[1000], -1)

    def test_same_team_spells_do_not_merge_across_halves(self):
        gt = pm.label_codes(np.array(['H', 'H', 'X', 'H', 'H']))
        result = pm.evaluate(gt, gt)
        self.assertEqual(result['n_gt_spells'], 2)
        self.assertEqual(result['spell_f1_mean'], 1.0)

    def test_sealed_match_preparation_and_scoring_rejected(self):
        match_id = self.config['sealed_test_matches'][0]
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'Sealed'):
                prepare(match_id, Path(folder), self.config)
        with self.assertRaises(ValueError):
            benchmark(match_id, 'development', 0, ROOT / 'solution/configs/validation.json')

    def test_split_membership_disjoint(self):
        groups = [set(self.config[key]) for key in
                  ('development_matches', 'validation_matches', 'sealed_test_matches')]
        self.assertEqual(sum(map(len, groups)), len(set.union(*groups)))
        self.assertNotIn('BAR-ATH', set.union(*groups))

    def test_nearest_geometry_uses_metres_and_team_mapping(self):
        self.inputs['ball_xy'][:] = [9., 0.]
        codes = nearest_team_codes(self.inputs)
        self.assertTrue((codes[self.inputs['period'] > 0] == 1).all())

    def test_sparse_external_frame_ids_are_not_shifted(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / 'J03WPY'
            folder.mkdir()
            meta = folder / 'matchinformation.xml'
            raw = folder / 'positions_raw_observed.xml'
            meta.write_text('''<PutDataRequest><MatchInformation>
              <General MatchId="DFL-MAT-J03WPY" HomeTeamId="home" GuestTeamId="away"/>
              <Environment PitchX="105" PitchY="68"/>
              <Teams><Team TeamId="home"><Players><Player PersonId="p1"/></Players></Team>
              <Team TeamId="away"><Players><Player PersonId="p2"/></Players></Team></Teams>
              </MatchInformation></PutDataRequest>''')
            raw.write_text('''<PutDataRequest><Positions>
              <FrameSet TeamId="ball" GameSection="firstHalf">
                <Frame N="10" T="2020-01-01T00:00:00+00:00" X="0" Y="0" BallStatus="1" BallPossession="1"/>
                <Frame N="12" T="2020-01-01T00:00:00.080+00:00" X="1" Y="0" BallStatus="1" BallPossession="2"/>
              </FrameSet><FrameSet TeamId="ball" GameSection="secondHalf">
                <Frame N="100" T="2020-01-01T00:10:00+00:00" X="0" Y="0" BallStatus="0" BallPossession="2"/>
                <Frame N="101" T="2020-01-01T00:10:00.040+00:00" X="0" Y="0" BallStatus="1" BallPossession="1"/>
              </FrameSet><FrameSet TeamId="home" PersonId="p1" GameSection="firstHalf">
                <Frame N="10" X="0" Y="0"/><Frame N="12" X="1" Y="0"/>
              </FrameSet></Positions></PutDataRequest>''')
            files = [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in (meta, raw)]
            (folder / 'download_manifest.json').write_text(json.dumps({'files': files, 'revision': 'synthetic-test'}))
            with redirect_stdout(io.StringIO()):
                prepare('J03WPY', Path(temporary), self.config)
            with np.load(folder / 'prepared/targets.npz') as targets:
                np.testing.assert_array_equal(targets['labels'], ['H', 'X', 'A', 'X', 'D', 'H'])
            with np.load(folder / 'prepared/inputs.npz') as inputs:
                self.assertTrue(np.isnan(inputs['players_xy'][1]).all())
                self.assertEqual(float(inputs['players_xy'][2, 0, 0]), 53.5)

    def test_slot_permutations_do_not_change_team_geometry(self):
        self.inputs['players_xy'] = np.tile(np.array(
            [[[0., 0.], [3., 0.], [10., 0.], [12., 0.]]], dtype=np.float32), (2001, 1, 1))
        self.inputs['team_of_slot'] = np.array([0, 0, 1, 1], dtype=np.int8)
        regime = dict(self.config['regimes']['clean'], slot_permutation_period_s=1)
        noisy = corrupt(self.inputs, regime, 43)
        self.assertFalse(np.array_equal(noisy['players_xy'], self.inputs['players_xy']))
        np.testing.assert_array_equal(nearest_team_codes(noisy), nearest_team_codes(self.inputs))


if __name__ == '__main__':
    unittest.main()
