"""Learned-evidence feature leakage, boundaries, ablation and weighting checks."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'solution/src'))
from possession.dead_features import extract_dead_features, feature_indices, window_mean
from possession.dead_model import training_rows, sample_weights, binary_metrics, assert_training_split


class DeadLearningTests(unittest.TestCase):
    def setUp(self):
        self.base = json.loads((ROOT / 'solution/configs/features.json').read_text())
        self.config = json.loads((ROOT / 'solution/configs/dead_model.json').read_text())
        self.inputs = {'players_xy': np.tile(np.array([[[5,5],[10,10],[15,15],[70,10],[75,15],[80,20]]], dtype=np.float32), (151,1,1)),
            'ball_xy': np.tile(np.array([[6,5]], dtype=np.float32), (151,1)),
            'team_of_slot': np.array([0,0,0,1,1,1], dtype=np.int8),
            'period': np.r_[np.ones(75),0,np.full(75,2)].astype(np.int8),
            'pitch_m': np.array([105,68], dtype=np.float32), 'fps': np.array(25)}

    def test_player_only_has_no_ball_dependency(self):
        first = extract_dead_features(self.inputs,self.base,self.config)
        self.inputs['ball_xy'][:] = np.nan
        second = extract_dead_features(self.inputs,self.base,self.config)
        indices = feature_indices(first['names'],'player')
        np.testing.assert_allclose(first['values'][:,indices],second['values'][:,indices],equal_nan=True)
        self.assertTrue(all(str(name).startswith('player_') for name in first['names'][indices]))

    def test_team_slot_invariance_and_no_mutation(self):
        before = self.inputs['players_xy'].copy()
        first = extract_dead_features(self.inputs,self.base,self.config)
        np.testing.assert_array_equal(self.inputs['players_xy'], before)
        self.inputs['players_xy'] = before[:,[2,0,1,5,3,4]]
        second = extract_dead_features(self.inputs,self.base,self.config)
        np.testing.assert_allclose(first['values'],second['values'],rtol=1e-5,atol=2e-4,equal_nan=True)

    def test_period_reset_and_label_rejection(self):
        result = extract_dead_features(self.inputs,self.base,self.config)
        self.assertTrue(np.isnan(result['values'][75]).all())
        names = result['names'].tolist()
        self.assertTrue(np.isnan(result['values'][76,names.index('player_home_centroid_drift_2s_mps')]))
        self.inputs['labels'] = np.full(151,'H')
        with self.assertRaises(ValueError):
            extract_dead_features(self.inputs,self.base,self.config)

    def test_windows_do_not_cross_half_and_unknown_is_not_zero(self):
        x = np.array([1,np.nan,3,np.nan,100,100],dtype=np.float32)
        period = np.array([1,1,1,0,2,2])
        mean, coverage = window_mean(x,period,5,True)
        self.assertEqual(mean[2],2)
        self.assertEqual(mean[4],100)
        self.assertTrue(np.isnan(mean[3]))
        mean, coverage = window_mean(np.full(6,np.nan),period,5)
        self.assertTrue(np.isnan(mean).all())
        self.assertEqual(coverage[0],0)

    def test_sampling_is_deterministic_and_excludes_ungraded(self):
        labels = np.full(151,'H'); labels[75]='X'
        first = training_rows(labels,self.inputs['period'],25,17)
        np.testing.assert_array_equal(first,training_rows(labels,self.inputs['period'],25,17))
        self.assertNotIn(75,first)

    def test_spell_weighting_equalizes_sampled_segments(self):
        labels = np.array(list('HHHHHHHHAA'))
        w = sample_weights(labels,np.ones(10),np.arange(10),'spell',100)
        self.assertAlmostEqual(w[:8].sum(),w[8:].sum())
        np.testing.assert_array_equal(sample_weights(labels,np.ones(10),np.arange(10),'frame',10),1)

    def test_binary_metrics_and_unknown_target_exclusion(self):
        result = binary_metrics(np.array(list('DDHHX')),np.array([.9,.2,.8,.1,1]),.5)
        self.assertEqual((result['tp'],result['fn'],result['fp'],result['tn']),(1,1,1,1))
        self.assertEqual(result['n'],4)
        self.assertEqual(result['f1'],.5)

    def test_split_and_variant_guards(self):
        with self.assertRaises(ValueError): assert_training_split(['a'],['a'])
        with self.assertRaises(ValueError): feature_indices(['x'],'invalid')
        cfg=json.loads((ROOT/'solution/configs/validation.json').read_text())
        self.assertFalse(set(self.config['development_matches']) & set(self.config['validation_matches']))
        self.assertFalse(set(self.config['development_matches']+self.config['validation_matches']) & set(cfg['sealed_test_matches']))


if __name__=='__main__': unittest.main()
