import itertools
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from possession.switch_events import event_targets,event_training_rows,extract_event_features,shifted
from possession.event_decoder import decode_events,directed_costs
from possession.decoder import decode,emission_costs


class SwitchEventTests(unittest.TestCase):
    def setUp(self):
        import json
        self.config=json.loads((Path(__file__).resolve().parents[1]/'configs/switch_events.json').read_text())
        self.profile={'mode':'fixed','stop':2,'restart':3,'switch':8}

    def test_targets_direction_closest_event_and_dead_barrier(self):
        labels=np.array(list('HHHAAHHDDDAAAHH'))
        targets,centers=event_targets(labels,np.ones(len(labels),dtype=np.int8),2)
        self.assertEqual([t for t,a,b in centers],[3,5,13])
        self.assertEqual(targets[3],2);self.assertEqual(targets[4],2);self.assertEqual(targets[5],1)
        np.testing.assert_array_equal(targets[7:10],0)
        self.assertEqual(targets[10],0)

    def test_half_boundaries_are_not_switch_events(self):
        labels=np.array(list('HHHAAA'));period=np.array([1,1,1,2,2,2])
        y,events=event_targets(labels,period,5)
        self.assertFalse(events);np.testing.assert_array_equal(y,0)
        value=shifted(np.arange(6,dtype=np.float32),period,1)
        self.assertTrue(np.isnan(value[0]) and np.isnan(value[3]))
        self.assertEqual(value[4],3)

    def test_sampling_restores_native_class_proportions(self):
        labels=np.array(list('H'*100+'A'*100+'H'*100+'D'*100));period=np.ones(len(labels),dtype=np.int8)
        rows,y,w,info=event_training_rows(labels,period,17,self.config)
        self.assertTrue({100,200}<=set(rows))
        full=np.array(info['class_frame_counts']);np.testing.assert_allclose(np.bincount(y,weights=w,minlength=3),full/full.sum())
        self.assertTrue((rows>=0).all())

    def test_features_only_use_supplied_evidence_and_no_cross_half_context(self):
        names=['owner_home_ball_distance_m','owner_away_ball_distance_m']
        physical=np.tile([1,3],(6,1)).astype(np.float32);h=np.linspace(.1,.9,6).astype(np.float32);d=np.full(6,.1,dtype=np.float32)
        period=np.array([1,1,1,2,2,2]);config=self.config|{'lags_frames':[1]}
        x,names=extract_event_features(physical,names,h,d,period,config)
        self.assertTrue(np.isnan(x[3,names.index('home_past_1')]))
        self.assertTrue(np.isnan(x[2,names.index('home_future_1')]))
        self.assertEqual(x[4,names.index('home_past_1')],h[3])

    def test_event_decoder_global_optimum_with_negative_rewards(self):
        h=np.array([.8,.6,.2,.3,.9]);d=np.full(5,.05);period=np.ones(5,dtype=np.int8)
        q=np.array([[.98,.01,.01],[.01,.01,.98],[.1,.05,.85],[.01,.98,.01],[.1,.85,.05]])
        result=decode_events(h,d,period,q,self.profile,4,self.config)
        to_h,to_a=directed_costs(q,8,4,self.config);self.assertTrue((np.minimum(to_h,to_a)<0).any())
        emission=emission_costs(h,d,period);costs=[]
        for states in itertools.product(range(3),repeat=5):
            value=sum(emission[t,k] for t,k in enumerate(states))
            for t in range(1,5):
                before,after=states[t-1],states[t]
                if before==after:continue
                value+=to_h[t] if (before,after)==(1,0) else to_a[t] if (before,after)==(0,1) else 2 if after==2 else 3
            costs.append(value)
        self.assertAlmostEqual(result['path_cost'],min(costs),places=10)

    def test_disabled_reward_exactly_matches_original_and_halves_reset(self):
        h=np.array([.9,.8,.2,.1]);d=np.full(4,.05);period=np.array([1,1,2,2]);q=np.tile([.8,.1,.1],(4,1))
        actual=decode_events(h,d,period,q,self.profile,0,self.config);expected=decode(h,d,period,self.profile)
        np.testing.assert_array_equal(actual['labels'],expected['labels']);self.assertEqual(actual['path_cost'],expected['path_cost'])
        np.testing.assert_array_equal(actual['labels'],list('HHAA'))

    def test_ambiguous_direction_cannot_receive_negative_round_trip_cost(self):
        q=np.array([[0,.5,.5],[.01,.98,.01],[.01,.01,.98]])
        h,a=directed_costs(q,8,4,self.config)
        self.assertTrue((h+a>=self.config['minimum_round_trip_cost']-1e-9).all())
        with self.assertRaises(ValueError):directed_costs(np.array([[.8,.8,.8]]),8,4,self.config)


if __name__=='__main__':unittest.main()
