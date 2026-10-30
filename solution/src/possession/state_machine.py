"""Deterministic offline owner hysteresis and independent stoppage evidence."""
import numpy as np

from .validation import period_blocks

REASONS = {0: 'outside_period', 1: 'geometry_owner', 2: 'owner_persistence',
           3: 'outside_pitch', 4: 'stationary_ball_and_teams', 5: 'dead_persistence',
           6: 'automatic_tag_owner', 7: 'insufficient_evidence'}


def predict(features, period, config):
    names = features['names'].tolist()
    values, valid = features['values'], features['valid']
    n = len(period)
    if values.shape != valid.shape or values.shape != (n, len(names)) or len(set(names)) != len(names):
        raise ValueError('Invalid feature schema/alignment')
    if not np.isin(period, [0, 1, 2]).all():
        raise ValueError('Invalid periods')
    for key, value in config.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'Expected positive finite setting: {key}')

    def col(name):
        if name not in names:
            if name.startswith('tag_'):
                return np.full(n, np.nan)
            raise ValueError(f'Missing feature: {name}')
        j = names.index(name)
        return np.where(valid[:, j], values[:, j], np.nan)

    home, away = col('home_ball_distance_m'), col('away_ball_distance_m')
    margin = away - home
    near = np.minimum(home, away)
    candidate = np.full(n, 'X', dtype='<U1')
    candidate[(near <= config['control_distance_m']) & (margin >= config['owner_margin_m'])] = 'H'
    candidate[(near <= config['control_distance_m']) & (margin <= -config['owner_margin_m'])] = 'A'
    tag_fraction = col('tag_home_evidence_fraction')
    tag_ok = (col('tag_available') == 1) & (col('tag_conflict') == 0)
    tag = np.full(n, 'X', dtype='<U1')
    tag[tag_ok & (tag_fraction >= config['tag_fraction_min'])] = 'H'
    tag[tag_ok & (tag_fraction <= 1 - config['tag_fraction_min'])] = 'A'
    # Tags supplement absent geometry; they cannot overrule clear measured control.
    tag_used = (candidate == 'X') & (tag != 'X')
    candidate[tag_used] = tag[tag_used]

    speed = col('ball_speed_mps')
    hspeed, aspeed = col('home_speed_median_mps'), col('away_speed_median_mps')
    enough = ((col('home_motion_observation_count') >= config['minimum_motion_players']) &
              (col('away_motion_observation_count') >= config['minimum_motion_players']))
    stationary = (enough & (speed < config['stationary_ball_speed_mps']) &
                  (hspeed < config['quiet_team_speed_mps']) & (aspeed < config['quiet_team_speed_mps']))
    outside = ((col('ball_goal_line_distance_m') < -config['outside_margin_m']) |
               (col('ball_touchline_distance_m') < -config['outside_margin_m']))
    restart = (((col('ball_in_pitch') == 1) & (speed >= config['restart_ball_speed_mps']) &
                (near <= config['control_distance_m'] + 1)) |
               (enough & (hspeed >= config['restart_team_speed_mps']) &
                (aspeed >= config['restart_team_speed_mps']) & ~outside))
    frames = lambda key: max(1, int(round(config[key] * 25)))
    labels = np.full(n, 'X', dtype='<U1')
    reason = np.where(period > 0, 7, 0).astype(np.uint8)
    for first, last in period_blocks(period):
        owner, pending, pending_tag = 'X', 'X', False
        owner_run = out_run = quiet_run = restart_run = 0
        last_owner_evidence = first - 1
        dead = False
        for i in range(first, last):
            out_run = out_run + 1 if outside[i] else 0
            quiet_run = quiet_run + 1 if stationary[i] else 0
            dead_start, dead_reason = None, None
            if out_run >= frames('outside_confirm_s'):
                dead_start, dead_reason = i - out_run + 1, 3
            elif quiet_run >= frames('stationary_confirm_s'):
                dead_start, dead_reason = i - quiet_run + 1, 4
            if not dead and dead_start is not None:
                dead = True
                labels[dead_start:i + 1], reason[dead_start:i + 1] = 'D', dead_reason
                owner, pending, owner_run, restart_run = 'X', 'X', 0, 0
            if dead:
                labels[i], reason[i] = 'D', dead_reason or 5
                restart_run = restart_run + 1 if restart[i] and not outside[i] else 0
                if restart_run < frames('restart_confirm_s'):
                    continue
                # Confirmation can refine the boundary back to the first evidence frame.
                boundary = i - restart_run + 1
                labels[boundary:i + 1], reason[boundary:i + 1] = 'X', 7
                dead, quiet_run, out_run, restart_run = False, 0, 0, 0
            proposed = candidate[i]
            if proposed != 'X':
                if proposed == pending and bool(tag_used[i]) == pending_tag:
                    owner_run += 1
                else:
                    pending, pending_tag, owner_run = proposed, bool(tag_used[i]), 1
                required = frames('tag_confirm_s' if pending_tag else 'owner_confirm_s')
                if proposed == owner:
                    last_owner_evidence = i
                elif owner_run >= required:
                    owner = proposed
                    last_owner_evidence = i
                    boundary = max(first, i - owner_run + 1)
                    # Never relabel an already established stoppage.
                    take = labels[boundary:i + 1] != 'D'
                    labels[boundary:i + 1][take] = owner
                    reason[boundary:i + 1][take] = 6 if pending_tag else 1
            else:
                pending, owner_run = 'X', 0
            if owner != 'X' and i - last_owner_evidence > frames('owner_hold_s'):
                owner = 'X'
            labels[i] = owner
            reason[i] = (6 if tag_used[i] else 1) if proposed == owner and owner != 'X' else (2 if owner != 'X' else 7)
    return {'labels': labels, 'reason': reason}
