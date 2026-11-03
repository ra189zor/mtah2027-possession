"""Evaluation-only partial references and paired interval diagnostics."""
import numpy as np
from .metrica_external import event_binary_reference
from .validation import period_blocks


def partial_reference(events, period, guard_frames=25):
    legacy, legacy_core, old_audit = event_binary_reference(events, period, guard_frames)
    bits = np.zeros(len(period), np.uint8)
    accepted, rejected = [], []
    records = sorted(events, key=lambda e: (int(e['Period']), int(e['Start Frame'])))
    possession_actions = {'PASS', 'RECOVERY', 'BALL LOST', 'BALL OUT', 'SHOT', 'SET PIECE'}
    for event in records:
        if event['Type'] != 'PASS': continue
        side = event.get('Team'); p = int(event['Period'])
        a, b = int(event['Start Frame'])-1, int(event['End Frame'])-1
        reason = None
        if side not in ('Home', 'Away'): reason = 'unknown team'
        elif any(tag in event.get('Subtype', '') for tag in ('OFFSIDE', 'REFEREE HIT')): reason = 'ambiguous interruption'
        elif not event.get('From') or not event.get('To') or event['From'] == event['To']: reason = 'invalid pass endpoints'
        elif not 0 <= a < b < len(period) or not np.all(period[a:b+1] == p): reason = 'invalid frame/half interval'
        elif any(int(v['Period']) == p and a < int(v['Start Frame'])-1 < b and
                 v.get('Team') in ('Home', 'Away') and v['Team'] != side and v['Type'] in possession_actions
                 for v in records): reason = 'opposing possession action inside pass'
        if reason:
            rejected.append({'start': a, 'end': b, 'reason': reason}); continue
        ids = np.arange(a+1, b); ids = ids[legacy[ids] == 0]
        if not len(ids):
            rejected.append({'start': a, 'end': b, 'reason': 'no unconflicted interior'}); continue
        bits[ids] |= 1 if side == 'Home' else 2
        accepted.append({'start': a+1, 'end_exclusive': b, 'team': side, 'period': p})
    conflict = bits == 3
    raw = np.full(len(period), 'X', dtype='<U1')
    raw[bits == 1] = 'H'; raw[bits == 2] = 'A'; raw[legacy == 1] = 'D'
    core = raw.copy(); core[(raw == 'D') & (legacy_core != 1)] = 'X'
    raw[period == 0] = 'X'; core[period == 0] = 'X'
    audit = {'legacy_binary_audit': old_audit, 'accepted_passes': accepted, 'rejected_passes': rejected,
             'opposing_pass_conflict_frames': int(conflict.sum()),
             'coverage': {name: {label: int((value == label).sum()) for label in ('H', 'A', 'D', 'X')}
                          for name, value in [('event_interiors', raw), ('guarded_dead', core)]},
             'reference_uses_tracking_or_predictions': False}
    return raw, core, audit


def score_subset(reference, prediction, dead_probability, subset):
    if reference.shape != prediction.shape or subset.shape != reference.shape or dead_probability.shape != reference.shape:
        raise ValueError('Partial-reference scoring alignment differs')
    valid = np.isin(reference, ['H', 'A', 'D']) & subset
    if not valid.any(): return None
    truth, pred, p = reference[valid], prediction[valid], dead_probability[valid]
    if not np.isin(pred, ['H', 'A', 'D', 'X']).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid prediction or probability')
    confusion = np.array([[int(((truth == a) & (pred == b)).sum()) for b in ('H', 'A', 'D', 'X')]
                          for a in ('H', 'A', 'D')])
    alive = truth != 'D'; committed = alive & np.isin(pred, ['H', 'A'])
    true_dead, pred_dead = truth == 'D', pred == 'D'
    tp = int((true_dead & pred_dead).sum())
    return {'frames': int(valid.sum()), 'accuracy_on_labelled_subset': float(np.mean(truth == pred)),
        'confusion_rows_HAD_columns_HADX': confusion.tolist(),
        'pass_team_accuracy': float(np.mean(truth[alive] == pred[alive])) if alive.any() else None,
        'pass_team_accuracy_when_committed': float(np.mean(truth[committed] == pred[committed])) if committed.any() else None,
        'pass_frames': int(alive.sum()), 'dead_frames': int(true_dead.sum()),
        'dead_recall': tp/int(true_dead.sum()) if true_dead.any() else None,
        'dead_precision': tp/int(pred_dead.sum()) if pred_dead.any() else None,
        'alive_recall': float(np.mean(np.isin(pred[alive], ['H', 'A']))) if alive.any() else None,
        'dead_brier_on_labelled_subset': float(np.mean((p-true_dead)**2))}


def labelled_intervals(reference, period):
    intervals = []
    for a, b in period_blocks(period):
        edges = np.r_[a, np.flatnonzero(reference[a+1:b] != reference[a:b-1])+a+1, b]
        intervals.extend((int(lo), int(hi)) for lo, hi in zip(edges[:-1], edges[1:]) if reference[lo] in ('H', 'A', 'D'))
    return intervals


def paired_interval_draws(reference, old, new, subset, intervals, picks):
    counts, old_correct, new_correct = [], [], []
    for a, b in intervals:
        keep = subset[a:b] & np.isin(reference[a:b], ['H', 'A', 'D'])
        counts.append(int(keep.sum()))
        old_correct.append(int(((old[a:b] == reference[a:b]) & keep).sum()))
        new_correct.append(int(((new[a:b] == reference[a:b]) & keep).sum()))
    counts = np.asarray(counts); delta = np.asarray(new_correct)-np.asarray(old_correct)
    if picks.ndim != 2 or picks.shape[1] != len(intervals): raise ValueError('Invalid paired interval draws')
    denominator = counts[picks].sum(1)
    return np.divide(delta[picks].sum(1), denominator, out=np.full(len(picks), np.nan), where=denominator > 0)
