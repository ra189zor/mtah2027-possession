"""Label-free features in metres and seconds, with explicit validity masks."""
import warnings

import numpy as np

from .validation import check_inputs, period_blocks


def masked_mean(values, axis):
    count = np.isfinite(values).sum(axis=axis)
    total = np.nansum(values, axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=np.float32), where=count > 0)


def median(values, axis):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-missing slices remain NaN
        return np.nanmedian(values, axis=axis)


def observation_distances(seen, period, fps):
    age = np.full(len(seen), np.nan, dtype=np.float32)
    future = age.copy()
    for start, stop in period_blocks(period):
        ids = np.arange(stop - start)
        visible = seen[start:stop]
        previous = np.maximum.accumulate(np.where(visible, ids, -1))
        following = np.minimum.accumulate(np.where(visible, ids, len(ids))[::-1])[::-1]
        age[start:stop] = np.where(previous >= 0, (ids - previous) / fps, np.nan)
        future[start:stop] = np.where(following < len(ids), (following - ids) / fps, np.nan)
    return age, future


def rolling_coverage(seen, period, window):
    coverage = np.full(len(seen), np.nan, dtype=np.float32)
    for start, stop in period_blocks(period):
        x = seen[start:stop].astype(np.int64)
        hi = np.arange(1, len(x) + 1)
        low = np.maximum(0, hi - window)
        sums = np.r_[0, np.cumsum(x)]
        coverage[start:stop] = (sums[hi] - sums[low]) / (hi - low)
    return coverage


def ball_velocity(ball, seen, period, lag, max_speed):
    velocity = np.full_like(ball, np.nan)
    rejected = np.zeros(len(ball), dtype=np.float32)
    dt = lag / 25
    for start, stop in period_blocks(period):
        if stop - start <= lag:
            continue
        ids = np.arange(start + lag, stop)
        prefix = np.r_[0, np.cumsum(seen[start:stop])]
        local = ids - start
        complete = (prefix[local + 1] - prefix[local - lag]) == lag + 1
        delta = (ball[ids] - ball[ids - lag]) / dt
        plausible = np.linalg.norm(delta, axis=1) <= max_speed
        valid = complete & plausible
        velocity[ids[valid]] = delta[valid]
        rejected[ids[complete & ~plausible]] = 1
    return velocity, rejected


def team_velocity(positions, period, lag, max_speed, chunk_size):
    """Match spatial observations within a team; reject ties, gaps and implausible moves."""
    velocity = np.full_like(positions, np.nan)
    dt = lag / 25
    for start, stop in period_blocks(period):
        for low in range(start + lag, stop, chunk_size):
            high = min(stop, low + chunk_size)
            now, before = positions[low:high], positions[low - lag:high - lag]
            d = now[:, :, None, :] - before[:, None, :, :]
            d2 = np.sum(d * d, axis=-1)
            d2[~np.isfinite(d2)] = np.inf
            row_min, col_min = d2.min(axis=2), d2.min(axis=1)
            row_unique = ((d2 == row_min[:, :, None]).sum(axis=2) == 1)
            col_unique = ((d2 == col_min[:, None, :]).sum(axis=1) == 1)
            best = d2.argmin(axis=2)
            batch = np.arange(high - low)[:, None]
            current = np.arange(now.shape[1])[None, :]
            reciprocal = (d2[batch, current, best] == col_min[batch, best])
            good = np.isfinite(row_min) & row_unique & col_unique[batch, best] & reciprocal
            good &= row_min <= (max_speed * dt) ** 2
            displacement = (now - before[batch, best]) / dt
            velocity[low:high] = np.where(good[:, :, None], displacement, np.nan)
    return velocity


def extract_features(inputs, config, tags=None):
    check_inputs(inputs)
    lag = config['motion_lag_frames']
    if lag < 1 or config['motion_chunk_frames'] < 1 or config['ball_coverage_window_frames'] < 1:
        raise ValueError('Feature windows/chunk sizes must be positive')
    period, ball, players = inputs['period'], inputs['ball_xy'], inputs['players_xy']
    active, n = period > 0, len(ball)
    pitch = inputs['pitch_m']
    seen = np.isfinite(ball).all(axis=1)
    bv, rejected = ball_velocity(ball, seen, period, lag, config['max_ball_speed_mps'])
    ball_speed = np.linalg.norm(bv, axis=1)
    columns = {}

    def add(name, values):
        values = np.asarray(values, dtype=np.float32).copy()
        if values.shape != (n,):
            raise ValueError(f'Unexpected feature shape: {name}')
        values[~active] = np.nan
        columns[name] = values

    add('ball_observed', seen.astype(np.float32))
    age, future = observation_distances(seen, period, 25)
    add('ball_missing_age_s', age)
    add('ball_next_observation_s_offline', future)
    add('ball_coverage_past_window', rolling_coverage(seen, period, config['ball_coverage_window_frames']))
    add('ball_x_m', np.where(seen, ball[:, 0], np.nan))
    add('ball_y_m', np.where(seen, ball[:, 1], np.nan))
    add('ball_in_pitch', np.where(seen, ((ball >= 0) & (ball <= pitch)).all(axis=1), np.nan))
    add('ball_goal_line_distance_m', np.where(seen, np.minimum(ball[:, 0], pitch[0] - ball[:, 0]), np.nan))
    add('ball_touchline_distance_m', np.where(seen, np.minimum(ball[:, 1], pitch[1] - ball[:, 1]), np.nan))
    add('ball_speed_mps', ball_speed)
    add('ball_motion_rejected', rejected)
    nearest_distances = []
    for team, prefix in ((0, 'home'), (1, 'away')):
        positions = players[:, inputs['team_of_slot'] == team]
        if positions.shape[1] == 0:
            raise ValueError('Both teams need at least one slot')
        present = np.isfinite(positions).all(axis=2)
        positions = np.where(present[:, :, None], positions, np.nan)
        count = present.sum(axis=1)
        centroid = masked_mean(positions, axis=1)
        add(prefix + '_player_count', count)
        add(prefix + '_centroid_x_m', centroid[:, 0])
        add(prefix + '_centroid_y_m', centroid[:, 1])
        spread = np.sqrt(masked_mean(np.sum((positions - centroid[:, None, :]) ** 2, axis=2), axis=1))
        add(prefix + '_rms_spread_m', spread)
        for axis, suffix in ((0, 'x_span_m'), (1, 'y_span_m')):
            minimum = np.where(present, positions[:, :, axis], np.inf).min(axis=1)
            maximum = np.where(present, positions[:, :, axis], -np.inf).max(axis=1)
            add(prefix + '_' + suffix, np.where(count > 0, maximum - minimum, np.nan))
        velocity = team_velocity(positions, period, lag, config['max_player_speed_mps'], config['motion_chunk_frames'])
        speed = np.linalg.norm(velocity, axis=2)
        reliable = np.isfinite(speed)
        reliable_count = reliable.sum(axis=1)
        add(prefix + '_motion_observation_count', reliable_count)
        add(prefix + '_speed_median_mps', median(speed, axis=1))
        moving = (speed >= config['moving_speed_mps']).sum(axis=1)
        add(prefix + '_moving_fraction', np.divide(moving, reliable_count,
            out=np.full(n, np.nan), where=reliable_count > 0))
        distance = np.linalg.norm(positions - ball[:, None, :], axis=2)
        distance[~np.isfinite(distance)] = np.inf
        best = distance.argmin(axis=1)
        closest = distance[np.arange(n), best]
        closest[~np.isfinite(closest)] = np.nan
        nearest_distances.append(closest)
        add(prefix + '_ball_distance_m', closest)
        near = (distance <= config['near_ball_radius_m']).sum(axis=1)
        add(prefix + '_players_near_ball', np.where(seen & (count > 0), near, np.nan))
        # Distance ties provide no unique nearest player's motion evidence.
        unique = (distance == closest[:, None]).sum(axis=1) == 1
        closest_velocity = velocity[np.arange(n), best].copy()
        closest_velocity[~unique] = np.nan
        closest_speed = np.linalg.norm(closest_velocity, axis=1)
        add(prefix + '_nearest_player_speed_mps', closest_speed)
        add(prefix + '_ball_player_speed_difference_mps', np.abs(ball_speed - closest_speed))
        relative_position = ball - positions[np.arange(n), best]
        relative_velocity = bv - closest_velocity
        dot = np.sum(relative_position * relative_velocity, axis=1)
        closing = np.divide(-dot, closest, out=np.full(n, np.nan), where=closest > 0.05)
        add(prefix + '_nearest_closing_speed_mps', closing)
    add('ball_distance_margin_away_minus_home_m', nearest_distances[1] - nearest_distances[0])
    if tags is not None:
        if not all(name.startswith('tag_') for name in tags):
            raise ValueError('Tag feature names must have tag_ prefix')
        for name, values in tags.items():
            add(name, values)
    names = list(columns)
    raw = np.column_stack(list(columns.values())).astype(np.float32)
    valid = np.isfinite(raw)
    matrix = np.where(valid, raw, 0).astype(np.float32)
    return {'values': matrix, 'valid': valid, 'names': np.array(names)}
