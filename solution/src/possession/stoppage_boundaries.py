"""Offline stop/restart evidence and exact three-state boundary-aware decoding."""
import numpy as np
from .switch_events import shifted, event_training_rows
from .decoder import emission_costs
from .event_decoder import directed_costs
from .validation import period_blocks


def boundary_features(physical, dead, period, config):
    if physical.shape != (len(period), 275) or dead.shape != period.shape:
        raise ValueError('Invalid boundary feature alignment')
    active = period > 0
    if not np.isfinite(dead[active]).all() or ((dead[active] < 0) | (dead[active] > 1)).any():
        raise ValueError('Invalid dead probabilities')
    d = np.where(active, dead, np.nan).astype(np.float32)
    columns = [d]
    for lag in config['lags_frames']:
        past, future = shifted(d, period, lag), shifted(d, period, -lag)
        columns.extend([past, future, future-past])
    x = np.column_stack([physical, *columns]).astype(np.float32)
    x[~active] = np.nan
    if np.isinf(x).any():
        raise ValueError('Infinite boundary feature')
    return x


def boundary_training_rows(labels, period, seed, config):
    # Reuse audited sampling/nearest-event tie semantics. H maps to dead, so
    # class 1 (enter H) is STOP and class 2 (enter A) is RESTART.
    binary = np.full(labels.shape, 'X', dtype='<U1')
    binary[labels == 'D'] = 'H'
    binary[np.isin(labels, ['H', 'A'])] = 'A'
    return event_training_rows(binary, period, seed, config)


def boundary_costs(probabilities, profile, config, strength):
    q = np.asarray(probabilities, dtype=float)
    if q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all() or (q < 0).any() or (q > 1).any() or not np.allclose(q.sum(1), 1, atol=1e-5):
        raise ValueError('Expected normalized none/stop/restart probabilities')
    threshold, floor = config['reward_threshold'], config['minimum_transition_cost']
    base = np.array([profile['stop'], profile['restart']], dtype=float)
    if not 0 < threshold < 1 or not np.isfinite([floor, strength, *base]).all() or min(floor, strength, *base) < 0:
        raise ValueError('Invalid stop/restart reward settings')
    p = np.clip(q[:, 1:], 1e-6, 1-1e-6)
    reward = np.maximum(0, np.log(p)-np.log1p(-p)-np.log(threshold/(1-threshold)))
    # The floor can never increase a zero/small base cost.
    return np.maximum(np.minimum(floor, base), base-strength*reward)


def decode_boundaries(home, dead, period, owner_events, boundaries, profile, owner_config, config, strength):
    if profile['mode'] != 'fixed':
        raise ValueError('Boundary experiment uses fixed base costs')
    emission = emission_costs(home, dead, period)
    if owner_events.shape != (len(period), 3) or boundaries.shape != (len(period), 3):
        raise ValueError('Event alignment mismatch')
    to_home, to_away = directed_costs(owner_events, profile['switch'], 4, owner_config)
    stop_restart = boundary_costs(boundaries, profile, config, strength)
    eh, ea, ed = (emission[:, j].tolist() for j in range(3))
    ch, ca = to_home.tolist(), to_away.tolist()
    cs, cr = stop_restart[:, 0].tolist(), stop_restart[:, 1].tolist()
    labels = np.full(len(period), 'X', dtype='<U1')
    state_labels = np.array(['H', 'A', 'D'])
    total = 0.
    for a, b in period_blocks(period):
        back = np.zeros((b-a, 3), dtype=np.uint8)
        sh, sa, sd = eh[a], ea[a], ed[a]
        for t in range(a+1, b):
            bh, ph = sh, 0
            if sa+ch[t] < bh: bh, ph = sa+ch[t], 1
            if sd+cr[t] < bh: bh, ph = sd+cr[t], 2
            ba, pa = sa, 1
            if sh+ca[t] < ba: ba, pa = sh+ca[t], 0
            if sd+cr[t] < ba: ba, pa = sd+cr[t], 2
            bd, pd = sd, 2
            if sh+cs[t] < bd: bd, pd = sh+cs[t], 0
            if sa+cs[t] < bd: bd, pd = sa+cs[t], 1
            back[t-a] = ph, pa, pd
            sh, sa, sd = bh+eh[t], ba+ea[t], bd+ed[t]
        state = int(np.argmin([sh, sa, sd])); total += min(sh, sa, sd)
        for t in range(b-1, a-1, -1):
            labels[t] = state_labels[state]
            if t > a: state = int(back[t-a, state])
    if (labels[period > 0] == 'X').any():
        raise ValueError('Boundary decoder left active unknowns')
    return {'labels': labels, 'path_cost': float(total)}
