"""Behavioral tests for owner confirmation, stoppages and uncertainty."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'solution/src'))
from possession.state_machine import predict
sys.path.insert(0, str(ROOT / 'solution/tools'))
from run_fallback import choose_profile


class StateMachineTests(unittest.TestCase):
    def setUp(self):
        cfg = json.loads((ROOT / 'solution/configs/fallback.json').read_text())
        self.config = cfg['common'] | cfg['profiles']['balanced']
        self.period = np.ones(400, dtype=np.int8)
        self.data = {name: np.full(400, value, dtype=np.float32) for name, value in {
            'home_ball_distance_m': 1, 'away_ball_distance_m': 10,
            'ball_speed_mps': 2, 'home_speed_median_mps': 2, 'away_speed_median_mps': 2,
            'home_motion_observation_count': 8, 'away_motion_observation_count': 8,
            'ball_goal_line_distance_m': 10, 'ball_touchline_distance_m': 10,
            'ball_in_pitch': 1}.items()}

    def run_model(self):
        values = np.column_stack(list(self.data.values()))
        features = {'values': np.nan_to_num(values), 'valid': np.isfinite(values),
                    'names': np.array(list(self.data))}
        return predict(features, self.period, self.config)

    def away(self, start, stop):
        self.data['home_ball_distance_m'][start:stop] = 10
        self.data['away_ball_distance_m'][start:stop] = 1

    def missing(self, start, stop):
        for name in ('home_ball_distance_m', 'away_ball_distance_m', 'ball_speed_mps',
                     'ball_goal_line_distance_m', 'ball_touchline_distance_m', 'ball_in_pitch'):
            self.data[name][start:stop] = np.nan

    def test_brief_opponent_touch_does_not_change_owner(self):
        self.away(80, 90)
        np.testing.assert_array_equal(self.run_model()['labels'], 'H')

    def test_sustained_turnover_refines_boundary(self):
        self.away(100, 400)
        result = self.run_model()['labels']
        np.testing.assert_array_equal(result[:100], 'H')
        np.testing.assert_array_equal(result[100:], 'A')

    def test_missing_ball_holds_owner_then_abstains_not_dead(self):
        self.missing(80, 400)
        result = self.run_model()['labels']
        self.assertEqual(result[150], 'H')
        self.assertEqual(result[350], 'X')
        self.assertFalse((result == 'D').any())

    def test_missing_from_start_never_invents_owner(self):
        self.missing(0, 400)
        np.testing.assert_array_equal(self.run_model()['labels'], 'X')

    def test_half_boundary_clears_owner_and_motion_counters(self):
        self.period[100], self.period[101:] = 0, 2
        self.missing(101, 400)
        result = self.run_model()['labels']
        self.assertEqual(result[99], 'H')
        np.testing.assert_array_equal(result[100:], 'X')

    def test_stationary_ball_alone_is_not_dead(self):
        self.data['ball_speed_mps'][:] = 0
        np.testing.assert_array_equal(self.run_model()['labels'], 'H')

    def test_stationary_ball_and_teams_dead_then_restart(self):
        for name in ('ball_speed_mps', 'home_speed_median_mps', 'away_speed_median_mps'):
            self.data[name][80:200] = 0
        result = self.run_model()['labels']
        np.testing.assert_array_equal(result[80:200], 'D')
        self.assertEqual(result[250], 'H')

    def test_outside_pitch_confirmation_and_restart(self):
        self.data['ball_touchline_distance_m'][80:150] = -1
        self.data['ball_in_pitch'][80:150] = 0
        result = self.run_model()['labels']
        np.testing.assert_array_equal(result[80:150], 'D')
        self.assertEqual(result[200], 'H')

    def test_short_outside_glitch_is_not_dead(self):
        self.data['ball_touchline_distance_m'][80:85] = -1
        self.assertFalse((self.run_model()['labels'] == 'D').any())

    def test_tag_conflict_is_not_an_owner(self):
        self.missing(0, 400)
        for name, value in {'tag_available': 1, 'tag_conflict': 1,
                            'tag_home_evidence_fraction': 0.9}.items():
            self.data[name] = np.full(400, value)
        np.testing.assert_array_equal(self.run_model()['labels'], 'X')
        self.data['tag_conflict'][:] = 0
        np.testing.assert_array_equal(self.run_model()['labels'], 'H')

    def test_tags_cannot_overrule_clear_geometry(self):
        for name, value in {'tag_available': 1, 'tag_conflict': 0,
                            'tag_home_evidence_fraction': 0}.items():
            self.data[name] = np.full(400, value)
        np.testing.assert_array_equal(self.run_model()['labels'], 'H')

    def test_masked_zero_is_not_evidence(self):
        self.missing(0, 400)
        self.data['home_speed_median_mps'][:] = np.nan
        self.data['away_speed_median_mps'][:] = np.nan
        np.testing.assert_array_equal(self.run_model()['labels'], 'X')

    def test_dead_state_resets_at_halftime(self):
        self.data['ball_touchline_distance_m'][20:100] = -1
        self.data['ball_in_pitch'][20:100] = 0
        self.period[100], self.period[101:] = 0, 2
        self.missing(101, 400)
        result = self.run_model()['labels']
        self.assertEqual(result[99], 'D')
        np.testing.assert_array_equal(result[100:], 'X')

    def test_short_restart_signal_does_not_end_stoppage(self):
        self.data['ball_touchline_distance_m'][20:100] = -1
        self.data['ball_in_pitch'][20:100] = 0
        self.missing(100, 400)
        self.data['home_speed_median_mps'][100:] = 0
        self.data['away_speed_median_mps'][100:] = 0
        self.data['ball_speed_mps'][120:124] = 2
        self.data['ball_in_pitch'][120:124] = 1
        self.data['home_ball_distance_m'][120:124] = 1
        self.data['away_ball_distance_m'][120:124] = 10
        np.testing.assert_array_equal(self.run_model()['labels'][20:], 'D')

    def test_selection_weights_regimes_not_number_of_seeds(self):
        rows = []
        for name, scores in {'a': [1, .4, .4], 'b': [0, .7, .7]}.items():
            for regime, score, count in zip(('clean', 'moderate', 'severe'), scores, (1, 3, 3)):
                rows.extend([{'model': name, 'match_id': 'synthetic', 'regime': regime,
                              'spell_f1_mean': score, 'macro_f1': .5}] * count)
        ranking = choose_profile(rows, ('a', 'b'))
        self.assertEqual(ranking[0]['profile'], 'a')
        self.assertAlmostEqual(ranking[0]['objective_spell_f1'], .6)


if __name__ == '__main__':
    unittest.main()
