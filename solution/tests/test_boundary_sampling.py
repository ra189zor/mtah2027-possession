from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from possession.boundary_sampling import owner_training_rows


class BoundarySamplingTests(unittest.TestCase):
    def setUp(self):
        self.config={'sampling_stride_frames':25,'inside_boundary_offsets_frames':[0,3],'include_spell_midpoint':True}

    def test_single_frame_and_short_spells_are_sampled_with_equal_total_weight(self):
        labels=np.array(list('H'+'A'*2+'H'*100+'D'*20+'A'*3+'X'))
        period=np.ones(len(labels),dtype=np.int8)
        rows,weights,info=owner_training_rows(labels,period,17,self.config)
        self.assertEqual(info['provider_spells'],4);self.assertEqual(info['represented_spells'],4)
        for a,b in [(0,1),(1,3),(3,103),(123,126)]:
            self.assertAlmostEqual(weights[(rows>=a)&(rows<b)].sum(),.25)
            self.assertIn(a,rows);self.assertIn(b-1,rows)
        self.assertFalse(np.isin(labels[rows],['D','X']).any())

    def test_halves_reset_sampling_and_no_inputs_mutated(self):
        labels=np.array(list('H'*10+'X'+'H'*10));period=np.array([1]*10+[0]+[2]*10)
        before=labels.copy();rows,weights,info=owner_training_rows(labels,period,43,self.config)
        self.assertEqual(info['provider_spells'],2)
        self.assertAlmostEqual(weights[rows<10].sum(),.5)
        np.testing.assert_array_equal(labels,before)
        again=owner_training_rows(labels,period,43,self.config)
        np.testing.assert_array_equal(rows,again[0]);np.testing.assert_array_equal(weights,again[1])


if __name__=='__main__':unittest.main()
