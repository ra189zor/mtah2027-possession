"""Bidirectional ownership behavior without reference-label input."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'solution/src'))
from possession.features import extract_features
from possession.gap_ownership import anchor_features,compose_ownership,extract_owner_features


class GapOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((ROOT/'solution/configs/gap_ownership.json').read_text())
        self.base=json.loads((ROOT/'solution/configs/features.json').read_text())
        self.inputs={'players_xy':np.tile(np.array([[[0.,0.],[10.,10.],[50.,0.],[60.,10.]]],dtype=np.float32),(240,1,1)),
            'ball_xy':np.tile(np.array([[1.,0.]],dtype=np.float32),(240,1)),
            'team_of_slot':np.array([0,0,1,1],dtype=np.int8),'period':np.ones(240,dtype=np.int8),
            'pitch_m':np.array([105.,68.],dtype=np.float32),'fps':np.array(25)}

    def anchors(self):
        basic=extract_features(self.inputs,self.base)
        return anchor_features(basic,self.inputs['period'],self.config)

    def test_matching_sides_bridge_longer_than_old_timeout(self):
        self.inputs['ball_xy'][40:200]=np.nan
        result=self.anchors()
        np.testing.assert_allclose(result['owner_bridge_home_probability'][40:200],.95)
        np.testing.assert_array_equal(result['owner_anchor_agreement'][40:200],1)
        self.assertGreater(result['owner_anchor_gap_s'][100],4)

    def test_different_sides_have_future_and_motion_time_context(self):
        self.inputs['ball_xy'][50:150]=np.nan
        self.inputs['ball_xy'][150:]=[49.,0.]
        result=self.anchors()
        self.assertEqual(result['owner_previous_home'][100],1)
        self.assertEqual(result['owner_next_home'][100],0)
        self.assertGreater(result['owner_bridge_home_probability'][60],.5)
        self.assertLess(result['owner_bridge_home_probability'][140],.5)

    def test_leading_gap_uses_future_anchor(self):
        self.inputs['ball_xy'][:100]=np.nan
        self.inputs['ball_xy'][100:]=[49.,0.]
        result=self.anchors()
        self.assertTrue(np.isnan(result['owner_previous_home'][0]))
        self.assertAlmostEqual(result['owner_bridge_home_probability'][0],.05)

    def test_half_boundary_prevents_anchor_carry(self):
        self.inputs['period'][120]=0; self.inputs['period'][121:]=2
        self.inputs['ball_xy'][121:]=np.nan
        result=self.anchors()
        self.assertTrue(np.isnan(result['owner_previous_home'][130]))
        self.assertEqual(result['owner_bridge_home_probability'][130],.5)
        self.assertTrue(np.isnan(result['owner_bridge_home_probability'][120]))

    def test_brief_opponent_touch_is_not_an_anchor(self):
        self.inputs['ball_xy'][80:85]=[49.,0.]
        result=self.anchors()
        self.assertEqual(result['owner_previous_home'][82],1)
        self.assertEqual(result['owner_next_home'][82],1)

    def test_composition_fills_unknown_and_preserves_dead_mask(self):
        p=np.r_[np.full(60,.9),np.full(180,.1)]
        dead=np.zeros(240); dead[100:130]=.9
        period=self.inputs['period'].copy(); period[120]=0
        result=compose_ownership(p,dead,period,.4,.55)
        labels=result['labels']
        self.assertFalse((labels[period>0]=='X').any())
        np.testing.assert_array_equal(labels=='D',(dead>=.5)&(period>0))
        self.assertEqual(labels[120],'X')
        np.testing.assert_array_equal(labels[:60],'H')
        np.testing.assert_array_equal(labels[60:100],'A')

    def test_short_posterior_glitch_does_not_switch(self):
        p=np.full(240,.9); p[80:85]=.1
        labels=compose_ownership(p,np.zeros(240),self.inputs['period'],.4,.55)['labels']
        np.testing.assert_array_equal(labels,'H')

    def test_no_anchor_is_explicit_low_confidence_not_hidden_unknown(self):
        self.inputs['ball_xy'][:]=np.nan
        anchors=self.anchors()
        np.testing.assert_array_equal(anchors['owner_anchor_count_sides'],0)
        result=compose_ownership(anchors['owner_bridge_home_probability'],np.zeros(240),self.inputs['period'],.4,.55)
        self.assertTrue(result['low_confidence'].all())
        np.testing.assert_array_equal(result['labels'],'H')

    def test_probability_validation_and_source_immutability(self):
        p=np.full(240,.9); before=p.copy()
        compose_ownership(p,np.zeros(240),self.inputs['period'],.4,.55)
        np.testing.assert_array_equal(p,before)
        p[0]=np.nan
        with self.assertRaises(ValueError):compose_ownership(p,np.zeros(240),self.inputs['period'],.4,.55)

    def test_feature_schema_and_unknown_mask(self):
        basic=extract_features(self.inputs,self.base)
        dead=np.zeros((240,2),dtype=np.float32)
        result=extract_owner_features(basic,dead,np.array(['test_a','test_b']),self.inputs['period'],self.config)
        self.assertEqual(result['values'].shape[1],len(result['names']))
        self.assertEqual(len(set(result['names'])),len(result['names']))
        self.assertTrue(any('next' in x for x in result['names']))


if __name__=='__main__':unittest.main()
