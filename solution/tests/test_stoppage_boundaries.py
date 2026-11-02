"""Global decoder optimality, exact backward compatibility, and target isolation."""
import itertools
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from possession.stoppage_boundaries import boundary_costs, boundary_features, boundary_training_rows, decode_boundaries
from possession.event_decoder import decode_events, directed_costs
from possession.decoder import emission_costs

OWNER = {'reward_threshold': .1, 'minimum_directed_switch_cost': -4, 'minimum_round_trip_cost': 2}
CONFIG = {'reward_threshold': .1, 'minimum_transition_cost': .1, 'lags_frames': [1, 2],
          'event_radius_frames': 1, 'training_stride_frames': 3, 'positive_sample_offsets_frames': [-1, 0, 1]}
PROFILE = {'mode': 'fixed', 'stop': 2, 'restart': 3, 'switch': 1}


class StoppageBoundaryTests(unittest.TestCase):
    def test_zero_reward_exactly_matches_previous_decoder(self):
        rng = np.random.default_rng(907)
        h, d = rng.random(200), rng.random(200)
        period = np.r_[np.ones(99), 0, np.full(100, 2)]
        q = rng.dirichlet([5, 1, 1], 200)
        b = rng.dirichlet([5, 1, 1], 200)
        old = decode_events(h, d, period, q, PROFILE, 4, OWNER)
        new = decode_boundaries(h, d, period, q, b, PROFILE, OWNER, CONFIG, 0)
        np.testing.assert_array_equal(old['labels'], new['labels'])
        self.assertEqual(old['path_cost'], new['path_cost'])

    def test_global_optimum_matches_exhaustive_paths(self):
        h, d = np.array([.8, .6, .2, .4]), np.array([.1, .8, .7, .1])
        q = np.array([[.8, .1, .1]]*4)
        b = np.array([[.8, .1, .1], [.2, .7, .1], [.9, .05, .05], [.1, .1, .8]])
        period = np.ones(4)
        result = decode_boundaries(h, d, period, q, b, PROFILE, OWNER, CONFIG, 1)
        emission = emission_costs(h, d, period)
        ch, ca = directed_costs(q, PROFILE['switch'], 4, OWNER)
        sr = boundary_costs(b, PROFILE, CONFIG, 1)
        values = []
        for states in itertools.product(range(3), repeat=4):
            value = sum(emission[t, s] for t, s in enumerate(states))
            for t in range(1, 4):
                trans = np.array([[0, ca[t], sr[t, 0]], [ch[t], 0, sr[t, 0]], [sr[t, 1], sr[t, 1], 0]])
                value += trans[states[t-1], states[t]]
            values.append(value)
        self.assertAlmostEqual(result['path_cost'], min(values), places=10)

    def test_targets_are_stop_restart_not_team_turnovers(self):
        labels = np.array(list('HHHAAADDDHHHAAADDDHHH'))
        rows, targets, weights, audit = boundary_training_rows(labels, np.ones(len(labels)), 1, CONFIG)
        mapping = dict(zip(rows, targets))
        self.assertEqual(mapping[6], 1)
        self.assertEqual(mapping[9], 2)
        full = np.array(audit['class_frame_counts'])
        np.testing.assert_allclose(np.bincount(targets, weights=weights, minlength=3), full/full.sum())

    def test_features_never_cross_half_boundary(self):
        period = np.array([1, 1, 0, 2, 2])
        x = boundary_features(np.zeros((5, 275)), np.array([.1, .2, np.nan, .8, .9]), period, CONFIG)
        self.assertTrue(np.isnan(x[3, 276]))
        self.assertTrue(np.isnan(x[1, 277]))
        self.assertTrue(np.isnan(x[2]).all())


if __name__ == '__main__': unittest.main()
