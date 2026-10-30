"""Training/evaluation helpers; reference labels never enter dead-feature extraction."""
import numpy as np

from .validation import period_blocks


def training_rows(labels, period, stride, seed):
    if stride < 1 or labels.shape != period.shape:
        raise ValueError('Invalid sampling specification')
    rng = np.random.default_rng(seed)
    rows = [np.arange(a + min(int(rng.integers(stride)), b-a-1), b, stride)
            for a, b in period_blocks(period)]
    rows = np.concatenate(rows) if rows else np.array([], dtype=np.int64)
    return rows[labels[rows] != 'X']


def sample_weights(labels, period, rows, kind, cap):
    if kind == 'frame':
        return np.ones(len(rows), dtype=np.float64)
    if kind != 'spell' or cap < 1:
        raise ValueError('Invalid weighting')
    changes = np.r_[True, (labels[1:] != labels[:-1]) | (period[1:] != period[:-1])]
    segment = np.cumsum(changes) - 1
    sampled_segment = segment[rows]
    counts = np.bincount(sampled_segment, minlength=int(segment.max()) + 1)
    weights = 1 / counts[sampled_segment]
    weights /= weights.mean()
    return np.minimum(weights, cap)


def binary_metrics(labels, probability, threshold, mask=None):
    mask = (labels != 'X') if mask is None else (mask & (labels != 'X'))
    y = labels[mask] == 'D'
    p = probability[mask]
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid dead probabilities')
    pred = p >= threshold
    tp = int((y & pred).sum())
    fp = int((~y & pred).sum())
    fn = int((y & ~pred).sum())
    tn = int((~y & ~pred).sum())
    divide = lambda a, b: a / b if b else 0.0
    calibration = []
    ece = 0.
    for i in range(10):
        in_bin = (p >= i/10) & (p < (i+1)/10 if i < 9 else p <= 1)
        count = int(in_bin.sum())
        if count:
            predicted, observed = float(p[in_bin].mean()), float(y[in_bin].mean())
            ece += count * abs(predicted-observed) / len(p)
            calibration.append({'bin': i, 'count': count, 'mean_probability': predicted, 'dead_fraction': observed})
    return {'n': len(p), 'true_dead': int(y.sum()), 'predicted_dead': int(pred.sum()),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': divide(tp, tp+fp), 'recall': divide(tp, tp+fn),
        'specificity': divide(tn, tn+fp), 'f1': divide(2*tp, 2*tp+fp+fn),
        'brier': float(np.mean((p-y)**2)) if len(p) else None,
        'ece_10': ece if len(p) else None, 'calibration_bins': calibration}


def assert_training_split(training_ids, held_out_ids):
    if not training_ids or set(training_ids) & set(held_out_ids):
        raise ValueError('Training and held-out match IDs must be nonempty and disjoint')
