"""Check oracle definitions and prevent accidental leakage between oracle branches."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from possession.diagnostics import oracle_predictions, extend_reference_team
from run_oracles import paired_draw_scores
import possession_metrics as pm


class OracleTests(unittest.TestCase):
    def setUp(self):
        self.truth = np.array(list('DHADDXDADH'))
        self.period = np.array([1] * 5 + [0] + [2] * 4)
        self.pred = np.array(list('HHXADXXDHA'))
        self.owner = np.array(list('HHXAXXXAHA'))

    def test_team_extension_resets_each_half(self):
        np.testing.assert_array_equal(extend_reference_team(self.truth, self.period), list('HHAAAXAAAH'))

    def test_both_true_reconstructs_truth(self):
        result = oracle_predictions(self.truth, self.pred, self.owner, self.period)
        np.testing.assert_array_equal(result['both_true'], self.truth)

    def test_true_dead_branch_does_not_receive_true_team(self):
        result = oracle_predictions(self.truth, self.pred, self.owner, self.period)
        alive = np.isin(self.truth, ['H', 'A'])
        np.testing.assert_array_equal(result['true_dead_predicted_team'][alive], self.owner[alive])
        np.testing.assert_array_equal(result['true_dead_predicted_team'][self.truth == 'D'], 'D')

    def test_true_team_branch_preserves_wrong_dead_decisions(self):
        result = oracle_predictions(self.truth, self.pred, self.owner, self.period)
        self.assertEqual(result['true_team_predicted_dead'][7], 'D')  # true A, predicted D
        self.assertEqual(result['true_team_predicted_dead'][0], 'H')  # true D, predicted alive
        self.assertEqual(result['true_team_keep_abstentions'][2], 'X')

    def test_no_source_mutation(self):
        before = [x.copy() for x in (self.truth, self.pred, self.owner, self.period)]
        oracle_predictions(self.truth, self.pred, self.owner, self.period)
        for a, b in zip(before, (self.truth, self.pred, self.owner, self.period)):
            np.testing.assert_array_equal(a, b)

    def test_rejects_dead_in_owner_branch_and_undefined_reference_team(self):
        with self.assertRaises(ValueError):
            oracle_predictions(self.truth, self.pred, self.pred, self.period)
        with self.assertRaises(ValueError):
            extend_reference_team(np.array(list('DD')), np.ones(2))

    def test_paired_bootstrap_perfect_prediction_is_one(self):
        gt = pm.label_codes(self.truth)
        count = len(pm.possession_spells(gt, pm.valid_runs(gt)))
        picks = np.random.RandomState(17).randint(0, count, (30, count))
        np.testing.assert_array_equal(paired_draw_scores(gt, gt, picks), 1)

    def test_unresampled_draw_equals_official_spell_score(self):
        gt, pred = pm.label_codes(self.truth), pm.label_codes(self.pred)
        runs = pm.valid_runs(gt)
        count = len(pm.possession_spells(gt, runs))
        picks = np.tile(np.arange(count), (3, 1))
        expected = pm.spell_scores(gt, pm.fill_abstentions(pred, runs), runs)['mean_f1']
        np.testing.assert_allclose(paired_draw_scores(gt, pred, picks), expected)


if __name__ == '__main__':
    unittest.main()
