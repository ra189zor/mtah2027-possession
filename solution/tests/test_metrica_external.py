"""Prevent sparse events from being silently converted into complete ground truth."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from possession.metrica_external import event_binary_reference
from validate_metrica_transfer import binary_score


def event(kind, start, end, subtype='', period=1):
    return {'Type': kind, 'Start Frame': start, 'End Frame': end, 'Subtype': subtype, 'Period': period}


class MetricaReferenceTests(unittest.TestCase):
    def test_empty_class_metric_is_undefined_not_zero_recall(self):
        report = binary_score(np.array([0, 0, -1]), np.array(list('HAD')),
                              np.array([.1, .2, .9]), np.ones(3, bool))
        self.assertIsNone(report['all']['dead_recall'])
        self.assertIsNone(report['all']['dead_precision'])
        self.assertEqual(report['all']['alive_recall'], 1)
        self.assertIsNone(report['ball_missing'])

    def test_stop_needs_explicit_restart_and_gap_is_not_automatically_dead(self):
        events = [event('PASS', 1, 5), event('BALL OUT', 8, 10), event('SET PIECE', 20, 20, 'THROW IN'),
                  event('PASS', 20, 24)]
        y, core, audit = event_binary_reference(events, np.ones(30), 2)
        np.testing.assert_array_equal(y[1:4], 0)
        np.testing.assert_array_equal(y[4:10], -1)
        np.testing.assert_array_equal(y[10:19], 1)
        np.testing.assert_array_equal(core[12:17], 1)
        self.assertEqual(len(audit['accepted_stoppages']), 1)

    def test_intervening_live_event_invalidates_stoppage(self):
        events = [event('BALL OUT', 1, 4), event('RECOVERY', 7, 7), event('SET PIECE', 20, 20)]
        y, _, audit = event_binary_reference(events, np.ones(30), 2)
        self.assertFalse((y == 1).any())
        self.assertEqual(len(audit['rejected_stoppages']), 1)

    def test_no_foul_assumption_or_cross_half_extension(self):
        events = [event('FAULT RECEIVED', 1, 1), event('BALL OUT', 8, 10),
                  event('SET PIECE', 20, 20, period=2)]
        y, _, _ = event_binary_reference(events, np.r_[np.ones(15), np.full(15, 2)], 2)
        self.assertTrue((y == -1).all())


if __name__ == '__main__':
    unittest.main()
