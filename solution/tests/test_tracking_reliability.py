import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'solution/src'))
from possession.tracking_reliability import quality_features,inject_training_ghosts,compose_candidates
from possession.inputs import array_digest


class TrackingReliabilityTests(unittest.TestCase):
    def inputs(self):
        return dict(ball_xy=np.column_stack([np.arange(100)/25,np.ones(100)]),
                    players_xy=np.ones((100,4,2)),team_of_slot=np.array([0,0,1,1]),
                    period=np.r_[np.ones(49),0,np.full(50,2)],pitch_m=np.array([105,68]),fps=np.array(25))

    def test_half_barrier_and_missingness(self):
        inputs=self.inputs();inputs['ball_xy'][45:55]=np.nan
        before=array_digest(inputs);x,names=quality_features(inputs)
        self.assertEqual(array_digest(inputs),before)
        self.assertTrue(np.isnan(x[50,names.index('previous_margin')]))
        self.assertTrue(np.isnan(x[48,names.index('following_margin')]))
        self.assertTrue(np.isnan(x[49]).all())

    def test_all_missing_remains_finite_or_nan(self):
        inputs=self.inputs();inputs['ball_xy'][:]=np.nan;inputs['players_xy'][:]=np.nan
        x,names=quality_features(inputs)
        self.assertFalse(np.isinf(x).any())
        self.assertTrue((x[inputs['period']>0,0]==0).all())

    def test_within_team_permutation_invariance(self):
        inputs=self.inputs();inputs['players_xy'][:,1]+=3
        original=quality_features(inputs)[0]
        inputs['players_xy']=inputs['players_xy'][:,[1,0,3,2]]
        np.testing.assert_allclose(original,quality_features(inputs)[0],equal_nan=True)

    def test_corruption_has_explicit_targets_and_preserves_inputs(self):
        inputs=self.inputs();cfg=json.loads((ROOT/'solution/configs/reliability_research.json').read_text())
        cfg['training_ghost_interval_s']=.04;before=array_digest(inputs)
        out,mask=inject_training_ghosts(inputs,cfg,811)
        self.assertEqual(array_digest(inputs),before)
        self.assertTrue(mask.any());self.assertFalse(mask[49])
        np.testing.assert_array_equal(out['players_xy'],inputs['players_xy'])
        np.testing.assert_array_equal(np.isnan(out['ball_xy']),np.isnan(inputs['ball_xy']))

    def test_gap_gate_and_probability_normalization(self):
        h=np.array([.9,.1]);q=np.array([[.8,.1,.1],[.7,.2,.1]])
        result=compose_candidates(h,q,1-h,q[:,[0,2,1]],np.array([False,True]),np.array([1.,0.]))
        self.assertEqual(result['gap'][0][0],h[0])
        self.assertAlmostEqual(result['gap'][0][1],.5)
        for home,events in result.values():
            self.assertTrue(((home>=0)&(home<=1)).all())
            np.testing.assert_allclose(events.sum(1),1)
            self.assertTrue((events>=0).all())
