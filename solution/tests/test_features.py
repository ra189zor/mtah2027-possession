"""Analytic geometry/motion and missingness tests, without challenge reference labels."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'solution/src'))
from possession.features import extract_features
from possession.inputs import read_challenge
from possession.tag_features import build_tag_features


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'solution/configs/features.json').read_text())
        self.inputs = {'players_xy': np.tile(np.array(
            [[[0., 0.], [10., 10.], [50., 10.], [60., 20.]]], dtype=np.float32), (13, 1, 1)),
            'ball_xy': np.tile(np.array([[2., 0.]], dtype=np.float32), (13, 1)),
            'team_of_slot': np.array([0, 0, 1, 1], dtype=np.int8),
            'period': np.ones(13, dtype=np.int8), 'pitch_m': np.array([105.3, 68.], dtype=np.float32),
            'fps': np.array(25)}

    def column(self, result, name):
        index = result['names'].tolist().index(name)
        return result['values'][:, index], result['valid'][:, index]

    def test_geometry_in_metres(self):
        result = extract_features(self.inputs, self.config)
        home, valid = self.column(result, 'home_ball_distance_m')
        self.assertTrue(valid.all())
        np.testing.assert_allclose(home, 2)
        centroid, _ = self.column(result, 'home_centroid_x_m')
        np.testing.assert_allclose(centroid, 5)
        margin, _ = self.column(result, 'ball_distance_margin_away_minus_home_m')
        np.testing.assert_allclose(margin, np.sqrt(48**2 + 10**2) - 2, rtol=1e-6)

    def test_known_constant_velocity_and_closing_rate(self):
        displacement = np.arange(13, dtype=np.float32) * 0.04
        self.inputs['players_xy'][:, :, 0] += displacement[:, None]
        self.inputs['ball_xy'][:, 0] += displacement
        result = extract_features(self.inputs, self.config)
        speed, valid = self.column(result, 'ball_speed_mps')
        self.assertFalse(valid[:5].any())
        np.testing.assert_allclose(speed[5:], 1, atol=1e-5)
        median, motion_valid = self.column(result, 'home_speed_median_mps')
        self.assertTrue(motion_valid[5:].all())
        np.testing.assert_allclose(median[5:], 1, atol=1e-5)
        closing, _ = self.column(result, 'home_nearest_closing_speed_mps')
        np.testing.assert_allclose(closing[5:], 0, atol=1e-5)

    def test_slot_identity_switches_do_not_create_velocity(self):
        original = extract_features(self.inputs, self.config)
        switched = {key: value.copy() for key, value in self.inputs.items()}
        for index in range(0, 13, 2):
            switched['players_xy'][index] = switched['players_xy'][index, [1, 0, 3, 2]]
        result = extract_features(switched, self.config)
        np.testing.assert_array_equal(result['valid'], original['valid'])
        np.testing.assert_allclose(result['values'], original['values'], atol=1e-5)

    def test_missing_ball_is_unknown_not_zero_distance_or_dead(self):
        self.inputs['ball_xy'][:] = np.nan
        result = extract_features(self.inputs, self.config)
        observed, known = self.column(result, 'ball_observed')
        np.testing.assert_array_equal(observed, 0)
        self.assertTrue(known.all())
        _, distances_valid = self.column(result, 'home_ball_distance_m')
        self.assertFalse(distances_valid.any())
        count, _ = self.column(result, 'home_player_count')
        np.testing.assert_array_equal(count, 2)
        self.assertTrue(np.isfinite(result['values']).all())

    def test_leading_missingness_does_not_backfill_history(self):
        self.inputs['ball_xy'][:3] = np.nan
        result = extract_features(self.inputs, self.config)
        age, known = self.column(result, 'ball_missing_age_s')
        self.assertFalse(known[:3].any())
        future, future_valid = self.column(result, 'ball_next_observation_s_offline')
        self.assertTrue(future_valid[0])
        self.assertAlmostEqual(float(future[0]), 0.12, places=6)
        self.assertEqual(age[3], 0)

    def test_no_motion_or_history_crosses_halftime(self):
        self.inputs['period'] = np.array([1]*6 + [0] + [2]*6, dtype=np.int8)
        result = extract_features(self.inputs, self.config)
        _, speed_valid = self.column(result, 'ball_speed_mps')
        self.assertFalse(speed_valid[7:12].any())
        self.assertTrue(speed_valid[12])
        self.assertFalse(result['valid'][6].any())
        self.inputs['ball_xy'][1:] = np.nan
        result = extract_features(self.inputs, self.config)
        _, age_valid = self.column(result, 'ball_missing_age_s')
        self.assertFalse(age_valid[7:].any())

    def test_short_ball_gap_does_not_create_observed_motion(self):
        self.inputs['ball_xy'][3] = np.nan
        result = extract_features(self.inputs, self.config)
        _, valid = self.column(result, 'ball_speed_mps')
        self.assertFalse(valid[5:9].any())

    def test_implausible_ball_motion_is_masked(self):
        self.inputs['ball_xy'][5:, 0] += 100
        result = extract_features(self.inputs, self.config)
        _, valid = self.column(result, 'ball_speed_mps')
        self.assertFalse(valid[5])
        rejected, _ = self.column(result, 'ball_motion_rejected')
        self.assertEqual(rejected[5], 1)

    def test_duplicate_positions_make_player_motion_ambiguous(self):
        self.inputs['players_xy'][:, 1] = self.inputs['players_xy'][:, 0]
        result = extract_features(self.inputs, self.config)
        count, _ = self.column(result, 'home_motion_observation_count')
        self.assertEqual(count[5], 0)

    def test_reference_fields_are_rejected(self):
        with self.assertRaises(ValueError):
            extract_features(dict(self.inputs, labels=np.full(13, 'H')), self.config)

    def test_defensive_tags_reverse_team_evidence_and_overlap_is_preserved(self):
        events = [
            {'code': 'DEFENDING IN MIDDLE THIRD', 'start_frame': 2, 'end_frame': 5,
             'tags': {'Team': 'Home', 'Half': '1st Half'}},
            {'code': 'BUILD UP', 'start_frame': 3, 'end_frame': 6,
             'tags': {'Team': 'Home', 'Half': '1st Half'}},
            {'code': 'SET PIECES', 'start_frame': 6, 'end_frame': 8,
             'tags': {'Team': 'N/A', 'Half': '1st Half'}}]
        tags = build_tag_features(events, self.inputs['period'], {'Home': 'H', 'Away': 'A'})
        self.assertEqual(tags['tag_away_evidence'][2], 1)
        self.assertEqual(tags['tag_home_evidence'][2], 0)
        self.assertEqual(tags['tag_conflict'][3], 1)
        self.assertEqual(tags['tag_home_evidence_fraction'][3], 0.5)
        self.assertEqual(tags['tag_restart_unknown_team'][7], 1)
        self.assertEqual(tags['tag_restart_window'][8], 0)

    def test_absent_tags_differ_from_available_empty_tags(self):
        absent = build_tag_features(None, self.inputs['period'], {})
        empty = build_tag_features([], self.inputs['period'], {})
        self.assertEqual(list(absent), list(empty))
        self.assertTrue(np.isnan(absent['tag_coverage']).all())
        np.testing.assert_array_equal(empty['tag_coverage'], 0)
        result = extract_features(self.inputs, self.config, absent)
        self.assertEqual(result['values'].shape, (13, 52))

    def test_challenge_loader_uses_one_based_rows_and_metres(self):
        with tempfile.TemporaryDirectory() as folder:
            meta, raw = Path(folder)/'meta.xml', Path(folder)/'raw.txt'
            meta.write_text('<Root><Team id="h"><Name>Home</Name></Team><Team id="a"><Name>Away</Name></Team>'
                            '<Player teamId="h"/><Player teamId="a"/></Root>')
            raw.write_text('1:0,0,0;1,1,0:0.5,0.5,0\n'
                           '2:NaN,NaN,NaN;0.9,0.9,0:NaN,NaN,NaN\n')
            inputs, mapping = read_challenge(meta, raw, 2, [105.3, 68], [(1, 0, 1)])
            np.testing.assert_allclose(inputs['ball_xy'][0], [52.65, 34], atol=1e-5)
            self.assertTrue(np.isnan(inputs['ball_xy'][1]).all())
            self.assertEqual(mapping, {'Home': 'H', 'Away': 'A'})


if __name__ == '__main__':
    unittest.main()
