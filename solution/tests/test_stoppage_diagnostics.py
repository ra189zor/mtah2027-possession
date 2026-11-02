"""Reference interventions must preserve errors in the component not replaced."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from possession.stoppage_diagnostics import diagnostic_oracles, probability_diagnostics, binary_boundaries


class StoppageDiagnosticsTests(unittest.TestCase):
    def test_component_separation_and_half_reset(self):
        truth = np.array(list('DHADDXDAAH'))
        pred = np.array(list('HHDHHXHHAD'))
        owner = np.array(list('AAHHAXAAAH'))
        period = np.array([1]*5+[0]+[2]*4)
        result = diagnostic_oracles(truth, pred, owner, period)
        np.testing.assert_array_equal(result['both_true'], truth)
        # False positive D is retained by the true-team intervention.
        self.assertEqual(result['true_team_mask'][2], 'D')
        # Missed true D stays alive; extension resets at the second half.
        self.assertEqual(result['true_team_mask'][6], 'A')
        alive = np.isin(truth, ['H', 'A'])
        np.testing.assert_array_equal(result['true_dead_mask'][alive], owner[alive])
        np.testing.assert_array_equal(pred, list('HHDHHXHHAD'))

    def test_invalid_owner_branch_is_rejected(self):
        with self.assertRaises(ValueError):
            diagnostic_oracles(np.array(list('DH')), np.array(list('DH')), np.array(list('DH')), np.ones(2))

    def test_probability_metrics_exclude_ungraded_and_split_missingness(self):
        result = probability_diagnostics(np.array([.9, .2, .8, np.nan]), np.array(list('DDHX')),
                                         np.array([1, 1, 1, 0]), np.array([True, False, True, False]))
        self.assertEqual(result['all']['n'], 3)
        self.assertEqual(result['all']['recall_at_half'], .5)
        self.assertEqual(result['all']['precision_at_half'], .5)
        self.assertEqual(result['ball_missing']['recall_at_half'], 0)

    def test_boundaries_do_not_match_across_halves(self):
        truth = np.array(list('HDDXHHH'))
        pred = np.array(list('HHHXHDD'))
        result = binary_boundaries(truth, pred, np.array([1, 1, 1, 0, 2, 2, 2]), 25)
        self.assertEqual(result['stop']['matched'], 0)
        self.assertEqual(result['stop']['missed'], 1)
        self.assertEqual(result['stop']['spurious'], 1)


if __name__ == '__main__':
    unittest.main()
