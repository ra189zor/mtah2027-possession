"""Multi-second player/ball evidence for dead-ball learning; no label argument."""
import numpy as np

from .features import extract_features, masked_mean, median, observation_distances, team_velocity
from .validation import check_inputs, period_blocks


def window_mean(values, period, width, centered=False):
    out = np.full(len(values), np.nan, dtype=np.float32)
    coverage = np.zeros(len(values), dtype=np.float32)
    for a, b in period_blocks(period):
        x = values[a:b]
        known = np.isfinite(x)
        index = np.arange(len(x))
        low = np.maximum(0, index - (width // 2 if centered else width - 1))
        high = np.minimum(len(x), index + (width - width // 2 if centered else 1))
        sums = np.r_[0., np.cumsum(np.where(known, x, 0), dtype=np.float64)]
        counts = np.r_[0, np.cumsum(known)]
        count = counts[high] - counts[low]
        out[a:b] = np.divide(sums[high] - sums[low], count,
            out=np.full(len(x), np.nan), where=count > 0)
        coverage[a:b] = count / (high - low)
    coverage[period == 0] = np.nan
    return out, coverage


def streak_seconds(signal, known, period):
    out = np.full(len(signal), np.nan, dtype=np.float32)
    for a, b in period_blocks(period):
        run = 0
        for i in range(a, b):
            run = run + 1 if known[i] and signal[i] else 0
            if known[i]:
                out[i] = run / 25
    return out


def extract_dead_features(inputs, base_config, config, base=None):
    check_inputs(inputs)
    if base is None:
        base = extract_features(inputs, base_config)
    period = inputs['period']
    n = len(period)
    names = base['names'].tolist()
    columns = {}

    def col(name):
        j = names.index(name)
        return np.where(base['valid'][:, j], base['values'][:, j], np.nan)

    def add(name, values):
        x = np.array(values, dtype=np.float32, copy=True)
        if x.shape != (n,):
            raise ValueError(f'Invalid dead feature shape: {name}')
        x[period == 0] = np.nan
        columns[name] = x

    summaries = []
    for side, team in (('home', 0), ('away', 1)):
        prefix = 'player_' + side + '_'
        for suffix in ('player_count', 'centroid_x_m', 'centroid_y_m', 'rms_spread_m',
                       'x_span_m', 'y_span_m', 'motion_observation_count', 'speed_median_mps', 'moving_fraction'):
            name = prefix + suffix
            add(name, col(side + '_' + suffix))
            if suffix in ('speed_median_mps', 'moving_fraction', 'motion_observation_count'):
                summaries.append(name)
        positions = inputs['players_xy'][:, inputs['team_of_slot'] == team]
        present = np.isfinite(positions).all(axis=2)
        positions = np.where(present[:, :, None], positions, np.nan)
        count = present.sum(axis=1)
        velocity = team_velocity(positions, period, config['long_motion_lag_frames'],
                                 base_config['max_player_speed_mps'], base_config['motion_chunk_frames'])
        speed = np.linalg.norm(velocity, axis=2)
        known = np.isfinite(speed)
        motion_count = known.sum(axis=1)
        add(prefix + 'speed_1s_mean_mps', masked_mean(speed, axis=1))
        add(prefix + 'speed_1s_median_mps', median(speed, axis=1))
        add(prefix + 'slow_fraction', np.divide(((speed < config['walking_speed_mps']) & known).sum(axis=1),
            motion_count, out=np.full(n, np.nan), where=motion_count > 0))
        add(prefix + 'motion_1s_count', motion_count)
        for suffix in ('speed_1s_mean_mps', 'speed_1s_median_mps', 'slow_fraction', 'motion_1s_count'):
            summaries.append(prefix + suffix)
        pitch = inputs['pitch_m']
        boundary = np.minimum.reduce([positions[:, :, 0], pitch[0] - positions[:, :, 0],
                                      positions[:, :, 1], pitch[1] - positions[:, :, 1]])
        box = ((positions[:, :, 0] <= 16.5) | (positions[:, :, 0] >= pitch[0] - 16.5)) & (np.abs(positions[:, :, 1] - pitch[1]/2) <= 20.16)
        for suffix, mask in (('boundary_fraction', boundary <= config['player_boundary_radius_m']), ('box_fraction', box)):
            add(prefix + suffix, np.divide((mask & present).sum(axis=1), count,
                out=np.full(n, np.nan), where=count > 0))
            summaries.append(prefix + suffix)
        clusters = np.full(n, np.nan, dtype=np.float32)
        for low in range(0, n, 512):
            high = min(n, low + 512)
            p = positions[low:high]
            d2 = np.sum((p[:, :, None] - p[:, None, :]) ** 2, axis=3)
            d2[~np.isfinite(d2)] = np.inf
            diagonal = np.arange(p.shape[1])
            d2[:, diagonal, diagonal] = np.inf
            close = d2.min(axis=2) <= config['cluster_radius_m'] ** 2
            clusters[low:high] = np.divide(close.sum(axis=1), count[low:high],
                out=np.full(high-low, np.nan), where=count[low:high] > 0)
        add(prefix + 'cluster_fraction', clusters)
        summaries.append(prefix + 'cluster_fraction')
        centroid = np.column_stack([col(side + '_centroid_x_m'), col(side + '_centroid_y_m')])
        for seconds in config['windows_s']:
            lag = int(seconds * 25)
            drift = np.full(n, np.nan, dtype=np.float32)
            for a, b in period_blocks(period):
                if b - a > lag:
                    drift[a+lag:b] = np.linalg.norm(centroid[a+lag:b] - centroid[a:b-lag], axis=1) / seconds
            add(prefix + f'centroid_drift_{seconds}s_mps', drift)
            summaries.append(prefix + f'centroid_drift_{seconds}s_mps')
    for name in names:
        if name.startswith('ball_') and name != 'ball_distance_margin_away_minus_home_m':
            add(name, col(name))
    nearest = np.minimum(col('home_ball_distance_m'), col('away_ball_distance_m'))
    add('ball_nearest_player_distance_m', nearest)
    ball = inputs['ball_xy']
    length, width = inputs['pitch_m']
    spots = np.array([(0, 0), (0, width), (length, 0), (length, width),
                      (11, width/2), (length-11, width/2), (length/2, width/2)])
    add('ball_set_piece_spot_distance_m', np.linalg.norm(ball[:, None] - spots[None], axis=2).min(axis=1))
    speed = col('ball_speed_mps')
    add('ball_stationary_streak_s', streak_seconds(speed < .35, np.isfinite(speed), period))
    alive_looking = (speed >= 2) & (nearest <= 4) & (col('ball_in_pitch') == 1)
    age, _ = observation_distances(alive_looking, period, 25)
    add('ball_time_since_input_alive_cue_s', age)
    summaries += ['ball_speed_mps', 'ball_observed', 'ball_stationary_streak_s',
                  'ball_nearest_player_distance_m']
    for name in summaries:
        for seconds in config['windows_s']:
            for centered in (False, True):
                suffix = f'_{seconds}s_' + ('centered_offline' if centered else 'past')
                mean, coverage = window_mean(columns[name], period, int(seconds * 25), centered)
                add(name + suffix, mean)
                add(name + suffix + '_coverage', coverage)
    return {'values': np.column_stack(list(columns.values())).astype(np.float32),
            'names': np.array(list(columns))}


def feature_indices(names, kind):
    if kind not in ('full', 'player'):
        raise ValueError('Unknown feature ablation')
    return np.array([i for i, name in enumerate(names) if kind == 'full' or name.startswith('player_')], dtype=np.int64)
