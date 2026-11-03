"""Partial references must not invent ownership between events or leak labels into noise."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from possession.transfer_validation import partial_reference, score_subset, labelled_intervals, paired_interval_draws


def event(kind, start, end, team='Home', subtype='', period=1):
    return {'Type': kind, 'Start Frame': start, 'End Frame': end, 'Team': team,
            'Subtype': subtype, 'Period': period, 'From': 'p1', 'To': 'p2'}


class TransferValidationTests(unittest.TestCase):
    def test_pass_interiors_only_and_no_gap_or_half_fill(self):
        events = [event('PASS', 1, 5), event('PASS', 10, 14, 'Away', period=2)]
        raw, core, audit = partial_reference(events, np.r_[np.ones(8), np.full(8, 2)], 1)
        np.testing.assert_array_equal(raw[1:4], list('HHH'))
        np.testing.assert_array_equal(raw[9:13], list('XAAA'))
        self.assertTrue((raw[4:9] == 'X').all())
        self.assertEqual(len(audit['accepted_passes']), 2)

    def test_offside_and_opposing_action_reject_pass(self):
        events = [event('PASS', 1, 8, subtype='CROSS-OFFSIDE'), event('PASS', 10, 18), event('RECOVERY', 15, 15, 'Away')]
        raw, _, audit = partial_reference(events, np.ones(25), 1)
        self.assertTrue((raw == 'X').all()); self.assertEqual(len(audit['rejected_passes']), 2)

    def test_simultaneous_opposing_passes_are_unknown(self):
        events = [event('PASS', 1, 6), event('PASS', 1, 6, 'Away')]
        raw, _, audit = partial_reference(events, np.ones(10), 1)
        self.assertTrue((raw == 'X').all()); self.assertEqual(audit['opposing_pass_conflict_frames'], 4)

    def test_dead_reference_guard_and_no_foul_guess(self):
        events = [event('BALL OUT', 1, 3), event('SET PIECE', 12, 12), event('FAULT RECEIVED', 17, 17)]
        raw, core, _ = partial_reference(events, np.ones(22), 2)
        np.testing.assert_array_equal(raw[3:11], list('D'*8))
        np.testing.assert_array_equal(core[5:9], list('DDDD'))
        self.assertTrue((core[12:] == 'X').all())

    def test_masked_score_counts_dead_predictions_as_owner_errors(self):
        ref, pred = np.array(list('HADX')), np.array(list('HDDD'))
        out = score_subset(ref, pred, np.array([.1, .9, .9, .9]), np.ones(4, bool))
        self.assertEqual(out['frames'], 3); self.assertEqual(out['pass_team_accuracy'], .5)
        self.assertEqual(out['dead_precision'], .5)
        self.assertIsNone(score_subset(ref, pred, np.ones(4), np.zeros(4, bool)))
        only = score_subset(ref, pred, np.ones(4), np.array([1, 1, 0, 0], bool))
        self.assertIsNone(only['dead_recall'])

    def test_paired_interval_bootstrap_keeps_uncertain_gaps_separate(self):
        ref = np.array(list('HHXAAADD')); period = np.ones(8)
        intervals = labelled_intervals(ref, period)
        self.assertEqual(intervals, [(0, 2), (3, 6), (6, 8)])
        picks = np.array([[0, 1, 2], [0, 0, 0], [2, 2, 2]])
        old = np.array(list('AAXAAADD'))
        out = paired_interval_draws(ref, old, ref, np.ones(8, bool), intervals, picks)
        np.testing.assert_allclose(out, [2/7, 1, 0])


if __name__ == '__main__': unittest.main()
