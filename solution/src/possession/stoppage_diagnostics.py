"""Evaluation-only interventions; this module must not enter inference bundles."""
import numpy as np
from .diagnostics import extend_reference_team
from .validation import period_blocks


def probability_diagnostics(probability, truth, period, observed):
    p = np.asarray(probability)
    if p.shape != truth.shape or p.shape != period.shape or p.shape != observed.shape:
        raise ValueError('Misaligned diagnostic inputs')
    result = {}
    for name, mask in [('all', period > 0), ('ball_missing', (period > 0) & ~observed),
                       ('ball_visible', (period > 0) & observed)]:
        if not mask.any():
            result[name] = None
            continue
        q = p[mask]
        if not np.isfinite(q).all() or ((q < 0) | (q > 1)).any():
            raise ValueError('Invalid dead probabilities')
        y = truth[mask] == 'D'
        pred = q >= .5
        tp = int((pred & y).sum())
        bins = []
        for low in np.arange(10) / 10:
            inside = (q >= low) & ((q < low + .1) if low < .9 else (q <= 1))
            if inside.any():
                bins.append({'low': float(low), 'n': int(inside.sum()),
                             'probability': float(q[inside].mean()), 'dead_rate': float(y[inside].mean())})
        result[name] = {'n': int(mask.sum()), 'dead_share': float(y.mean()),
                        'brier': float(np.mean((q - y) ** 2)),
                        'recall_at_half': tp / max(1, int(y.sum())),
                        'precision_at_half': tp / max(1, int(pred.sum())),
                        'mean_probability_true_dead': float(q[y].mean()) if y.any() else None,
                        'mean_probability_true_alive': float(q[~y].mean()) if (~y).any() else None,
                        'bins': bins}
    return result


def diagnostic_oracles(truth, prediction, owner_only, period):
    """Hard-mask interventions, with explicit team extension inside true stoppages.

    owner_only is independently decoded with stoppage evidence disabled. Its
    ownership continuity can differ from production; this is not a tight ceiling.
    """
    for x in (truth, prediction, owner_only):
        if x.shape != period.shape:
            raise ValueError('Misaligned oracle inputs')
    active = period > 0
    if not np.isin(truth[active], ['H', 'A', 'D']).all():
        raise ValueError('Unknown graded reference')
    if not np.isin(prediction[active], ['H', 'A', 'D']).all():
        raise ValueError('Unknown graded prediction')
    if not np.isin(owner_only[active], ['H', 'A']).all():
        raise ValueError('Owner-only branch must define every active team')
    team = extend_reference_team(truth, period)
    dead = truth == 'D'
    true_dead = owner_only.copy()
    true_dead[dead & active] = 'D'
    true_team = team.copy()
    true_team[(prediction == 'D') & active] = 'D'
    both = team.copy()
    both[dead & active] = 'D'
    for x in (true_dead, true_team, both):
        x[~active] = 'X'
    if not np.array_equal(both, truth):
        raise ValueError('Oracle reconstruction failed')
    return {'true_dead_mask': true_dead, 'true_team_mask': true_team, 'both_true': both}


def binary_boundaries(truth, prediction, period, tolerance=25):
    """Greedy nearest-first one-to-one stop/restart matching, separated by half."""
    result = {}
    for name, before, after in [('stop', False, True), ('restart', True, False)]:
        matched = missed = spurious = 0
        offsets = []
        for a, b in period_blocks(period):
            y, p = truth[a:b] == 'D', prediction[a:b] == 'D'
            gt = np.flatnonzero((y[:-1] == before) & (y[1:] == after)) + a + 1
            pr = np.flatnonzero((p[:-1] == before) & (p[1:] == after)) + a + 1
            pairs = sorted((abs(int(j-i)), int(i), int(j)) for i in gt for j in pr if abs(j-i) <= tolerance)
            used_gt, used_pr = set(), set()
            for _, i, j in pairs:
                if i not in used_gt and j not in used_pr:
                    used_gt.add(i); used_pr.add(j); offsets.append((j-i)/25)
            matched += len(used_gt); missed += len(gt)-len(used_gt); spurious += len(pr)-len(used_pr)
        result[name] = {'matched': matched, 'missed': missed, 'spurious': spurious,
                        'recall': matched/max(1, matched+missed),
                        'precision': matched/max(1, matched+spurious),
                        'mean_abs_error_s': float(np.mean(np.abs(offsets))) if offsets else None}
    return result
