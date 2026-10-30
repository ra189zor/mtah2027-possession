"""Global optimality, short events, temporal boundaries and probability guards."""
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from possession.decoder import decode,emission_costs,transition_costs


class DecoderTests(unittest.TestCase):
    def test_global_optimum_matches_exhaustive_search(self):
        h=np.array([.8,.6,.1,.2,.7,.9]);d=np.array([.1,.2,.8,.7,.3,.1]);period=np.ones(6)
        for mode in ('fixed','evidence'):
            profile={'mode':mode,'stop':.7,'restart':1.3,'switch':.9}
            result=decode(h,d,period,profile)
            emission=emission_costs(h,d,period)
            scores=[]
            for states in itertools.product(range(3),repeat=6):
                value=sum(emission[t,state] for t,state in enumerate(states))
                value+=sum(transition_costs(h[t],d[t],profile)[states[t-1],states[t]] for t in range(1,6))
                scores.append(value)
            self.assertAlmostEqual(result['path_cost'],min(scores),places=10)

    def test_zero_cost_matches_joint_frame_map(self):
        h=np.array([.9,.1,.5]);d=np.array([.1,.1,.8]);period=np.ones(3)
        result=decode(h,d,period,{'mode':'fixed','stop':0,'restart':0,'switch':0})
        np.testing.assert_array_equal(result['labels'],['H','A','D'])

    def test_weak_short_alive_hole_is_suppressed(self):
        h=np.full(100,.9);d=np.full(100,.95);d[40:43]=.4
        result=decode(h,d,np.ones(100),{'mode':'fixed','stop':8,'restart':8,'switch':8})
        np.testing.assert_array_equal(result['labels'],'D')

    def test_brief_strong_event_can_survive_no_hard_minimum(self):
        h=np.full(25,.999999);d=np.full(25,.000001);d[12]=.999999
        result=decode(h,d,np.ones(25),{'mode':'fixed','stop':2,'restart':2,'switch':2})
        self.assertEqual(result['labels'][12],'D')

    def test_halves_do_not_share_initial_state_or_cost(self):
        h=np.array([.99,.99,np.nan,.01,.01]);d=np.array([.01,.01,np.nan,.01,.01]);period=np.array([1,1,0,2,2])
        profile={'mode':'fixed','stop':100,'restart':100,'switch':100}
        result=decode(h,d,period,profile)
        np.testing.assert_array_equal(result['labels'],list('HHXAA'))
        separate=decode(h[:2],d[:2],np.ones(2),profile)['path_cost']+decode(h[3:],d[3:],np.ones(2),profile)['path_cost']
        self.assertAlmostEqual(result['path_cost'],separate)

    def test_direct_team_switch_does_not_require_dead(self):
        h=np.r_[np.full(30,.99),np.full(30,.01)];d=np.full(60,.001)
        result=decode(h,d,np.ones(60),{'mode':'fixed','stop':100,'restart':100,'switch':1})
        np.testing.assert_array_equal(result['labels'],list('H'*30+'A'*30))

    def test_probability_guards_and_finite_endpoints(self):
        profile={'mode':'fixed','stop':1,'restart':2,'switch':3}
        self.assertTrue(np.isfinite(decode(np.array([1.,0.]),np.array([0.,1.]),np.ones(2),profile)['path_cost']))
        with self.assertRaises(ValueError):decode(np.array([np.nan]),np.array([.5]),np.ones(1),profile)
        with self.assertRaises(ValueError):decode(np.array([1.1]),np.array([.5]),np.ones(1),profile)
        with self.assertRaises(ValueError):decode(np.array([.5]),np.array([.5]),np.ones(1),profile|{'stop':-1})

    def test_repeats_and_does_not_mutate_inputs(self):
        h=np.linspace(.1,.9,100);d=np.linspace(.9,.1,100);period=np.ones(100)
        before=h.copy(),d.copy(),period.copy();profile={'mode':'evidence','stop':5,'restart':8,'switch':10}
        a=decode(h,d,period,profile);b=decode(h,d,period,profile)
        np.testing.assert_array_equal(a['labels'],b['labels'])
        self.assertEqual(a['path_cost'],b['path_cost'])
        for old,new in zip(before,(h,d,period)):np.testing.assert_array_equal(old,new)


if __name__=='__main__':unittest.main()
