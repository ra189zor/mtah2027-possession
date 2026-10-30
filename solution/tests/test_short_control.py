from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from possession.short_control import control_log_odds,combine_ownership


class ShortControlTests(unittest.TestCase):
    def setUp(self):
        import json
        self.config=json.loads((Path(__file__).resolve().parents[1]/'configs/short_possession.json').read_text())
        self.names=['owner_home_ball_distance_m','owner_away_ball_distance_m','ball_speed_mps',
                    'owner_home_ball_player_speed_difference_mps','owner_away_ball_player_speed_difference_mps']

    def test_short_control_confirmation_and_gaps(self):
        x=np.tile([1,4,5,1,1],(10,1)).astype(float);x[2,:]=np.nan;x[6:8,:2]=[4,1]
        period=np.ones(10,dtype=np.int8)
        odds=control_log_odds(x,self.names,period,self.config)
        np.testing.assert_array_equal(odds[:3],0)
        self.assertTrue((odds[3:6]>0).all())
        np.testing.assert_array_equal(odds[6:],0)

    def test_high_speed_and_halves_do_not_create_confirmed_control(self):
        x=np.tile([1,4,5,1,1],(5,1)).astype(float);period=np.array([1,1,0,2,2])
        np.testing.assert_array_equal(control_log_odds(x,self.names,period,self.config),0)
        x[:,2]=30;period[:]=1
        np.testing.assert_array_equal(control_log_odds(x,self.names,period,self.config),0)

    def test_residual_direction_neutrality_and_nonmutation(self):
        p=np.array([.2,.8,.4,np.nan],dtype=np.float32);before=p.copy();period=np.array([1,1,1,0])
        evidence=np.array([3,-3,0,0],dtype=np.float32)
        result=combine_ownership(p,evidence,1,period)
        self.assertGreater(result[0],.5);self.assertLess(result[1],.5)
        self.assertEqual(result[2],p[2]);self.assertTrue(np.isnan(result[3]))
        np.testing.assert_array_equal(p,before)
        np.testing.assert_array_equal(combine_ownership(p,evidence,0,period),p)
        with self.assertRaises(ValueError):combine_ownership(p,evidence,-1,period)


if __name__=='__main__':unittest.main()
