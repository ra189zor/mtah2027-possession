"""Input-only corruption and geometry baselines for external tracking benchmarks."""
import numpy as np

INPUT_FIELDS = frozenset({'players_xy', 'ball_xy', 'team_of_slot', 'period', 'pitch_m', 'fps'})


def check_inputs(inputs):
    if set(inputs) != INPUT_FIELDS:
        raise ValueError(f'Unexpected model input fields: {set(inputs) ^ INPUT_FIELDS}')
    n = len(inputs['ball_xy'])
    if inputs['players_xy'].ndim != 3 or inputs['players_xy'].shape[0] != n or inputs['players_xy'].shape[2] != 2:
        raise ValueError('Player coordinates must have shape (frames, slots, 2)')
    if inputs['ball_xy'].shape != (n, 2) or inputs['period'].shape != (n,):
        raise ValueError('Ball/period shape mismatch')
    if inputs['team_of_slot'].shape != (inputs['players_xy'].shape[1],):
        raise ValueError('Team-slot shape mismatch')
    if int(inputs['fps']) != 25:
        raise ValueError('Official temporal scorer requires native 25 Hz in this benchmark')
    if not np.isin(inputs['team_of_slot'], [0, 1]).all():
        raise ValueError('Expected home/away anonymous slots')
    if not np.isin(inputs['period'], [0, 1, 2]).all():
        raise ValueError('Unexpected period ID')


def period_blocks(period):
    changes = np.r_[0, np.flatnonzero(period[1:] != period[:-1]) + 1, len(period)]
    return [(int(a), int(b)) for a, b in zip(changes[:-1], changes[1:]) if period[a] > 0]


def burst_mask(n, fraction, median_frames, sigma, rng):
    if not 0 <= fraction <= 1:
        raise ValueError('Dropout fraction must be in [0, 1]')
    if fraction == 0:
        return np.zeros(n, dtype=bool)
    if fraction == 1:
        return np.ones(n, dtype=bool)
    if median_frames <= 0 or sigma < 0:
        raise ValueError('Invalid burst duration distribution')
    expected_gap = median_frames * np.exp(sigma * sigma / 2)
    expected_visible = expected_gap * (1 - fraction) / fraction
    mask, cursor, missing = np.zeros(n, dtype=bool), 0, bool(rng.random() < fraction)
    while cursor < n:
        length = max(1, int(round(rng.lognormal(np.log(median_frames), sigma)
                                 if missing else rng.exponential(expected_visible))))
        end = min(n, cursor + length)
        if missing:
            mask[cursor:end] = True
        cursor, missing = end, not missing
    return mask


def corrupt(inputs, regime, seed):
    """This API cannot receive possession/dead-ball labels; they never guide corruption."""
    check_inputs(inputs)
    out = {key: value.copy() for key, value in inputs.items()}
    rng = np.random.default_rng(seed)
    players, ball, period = out['players_xy'], out['ball_xy'], out['period']
    fps = int(out['fps'])
    teams = out['team_of_slot']
    for start, stop in period_blocks(period):
        n = stop - start
        ball[start:stop][burst_mask(n, regime['ball_dropout_fraction'],
            regime['ball_gap_median_s'] * fps, regime['ball_gap_lognormal_sigma'], rng)] = np.nan
        for slot in range(players.shape[1]):
            mask = burst_mask(n, regime['player_dropout_fraction'], fps, 0.6, rng)
            players[start:stop, slot][mask] = np.nan
        noise = regime['coordinate_noise_m']
        if noise:
            players[start:stop] += rng.normal(0, noise, players[start:stop].shape).astype(np.float32)
            ball[start:stop] += rng.normal(0, noise, ball[start:stop].shape).astype(np.float32)
        permutation_period = int(regime['slot_permutation_period_s'] * fps)
        if permutation_period:
            for low in range(start, stop, permutation_period):
                high = min(stop, low + permutation_period)
                permutation = np.arange(len(teams))
                for team in (0, 1):
                    slots = np.flatnonzero(teams == team)
                    permutation[slots] = rng.permutation(slots)
                players[low:high] = players[low:high][:, permutation]
        probability = regime['duplicate_probability']
        duplicate_frames = np.flatnonzero(rng.random(n) < probability) + start if probability else []
        for frame in duplicate_frames:
            side = int(rng.integers(2))
            present = np.flatnonzero((teams == side) & np.isfinite(players[frame]).all(axis=1))
            empty = np.flatnonzero((teams == side) & ~np.isfinite(players[frame]).any(axis=1))
            if len(present) and len(empty):
                players[frame, rng.choice(empty)] = players[frame, rng.choice(present)] + rng.normal(0, 0.05, 2)
    return out


def nearest_team_codes(inputs, chunk_size=4096):
    check_inputs(inputs)
    n = len(inputs['ball_xy'])
    codes = np.full(n, -1, dtype=np.int8)
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        difference = inputs['players_xy'][start:stop] - inputs['ball_xy'][start:stop, None, :]
        distances = np.sum(difference * difference, axis=2)
        distances[~np.isfinite(distances)] = np.inf
        best = np.argmin(distances, axis=1)
        found = np.isfinite(distances[np.arange(stop - start), best])
        window = codes[start:stop]
        window[found] = inputs['team_of_slot'][best[found]]
    codes[inputs['period'] == 0] = -1
    return codes


def summarize_inputs(inputs):
    inside = inputs['period'] > 0
    present = np.isfinite(inputs['ball_xy']).all(axis=1)
    lengths = []
    for start, stop in period_blocks(inputs['period']):
        mask = ~present[start:stop]
        edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
        lengths.extend((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).tolist())
    return {'ball_missing_fraction': float((~present[inside]).mean()),
        'mean_observed_player_slots': float(np.isfinite(inputs['players_xy'][inside]).all(axis=2).sum(axis=1).mean()),
        'ball_missing_runs': len(lengths),
        'ball_gap_median_s': float(np.median(lengths) / 25) if lengths else 0,
        'ball_gap_p95_s': float(np.percentile(lengths, 95) / 25) if lengths else 0}
