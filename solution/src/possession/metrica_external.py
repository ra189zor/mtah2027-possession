"""Public external sample conversion and conservative event-derived binary references.

This evaluation adapter is never imported by production inference. Unknown
reference frames remain -1: event sparsity is not evidence that play stopped.
"""
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from .inference import validate_tracking


def read_public_tracking(home_path, away_path):
    teams = []
    for path in (home_path, away_path):
        with Path(path).open(newline='') as handle:
            rows = csv.reader(handle)
            next(rows); next(rows); header = next(rows)
        frame = pd.read_csv(path, skiprows=3, header=None).to_numpy(dtype=float)
        if len(header) != frame.shape[1] or header[-2:] != ['Ball', '']:
            raise ValueError('Unexpected Metrica sample tracking schema')
        if header[:3] != ['Period', 'Frame', 'Time [s]'] or (frame.shape[1]-5) % 2:
            raise ValueError('Unexpected tracking columns')
        teams.append(frame)
    h, a = teams
    if h.shape[0] != a.shape[0] or not np.array_equal(h[:, :3], a[:, :3]):
        raise ValueError('Home/away frame alignment differs')
    if not np.allclose(h[:, -2:], a[:, -2:], equal_nan=True):
        raise ValueError('Home/away ball observations differ')
    ids = h[:, 1].astype(int)
    if not np.array_equal(ids, h[:, 1]) or not np.array_equal(ids, np.arange(1, len(ids)+1)):
        raise ValueError('Expected contiguous one-based frame IDs')
    if not np.allclose(h[:, 2], ids/25, atol=.011):
        raise ValueError('Tracking timestamps do not match 25 Hz frame IDs')
    if not np.isin(h[:, 0], [1, 2]).all():
        raise ValueError('Unexpected period value')
    players = [x[:, 3:-2].reshape(len(x), -1, 2) for x in teams]
    scale = np.array([105., 68.])
    result = {'players_xy': (np.concatenate(players, axis=1)*scale).astype(np.float32),
              'team_of_slot': np.r_[np.zeros(players[0].shape[1], dtype=np.int8),
                                    np.ones(players[1].shape[1], dtype=np.int8)],
              'ball_xy': (h[:, -2:]*scale).astype(np.float32),
              'period': h[:, 0].astype(np.int8), 'pitch_m': scale, 'fps': np.array(25.)}
    validate_tracking(result)
    return result


def event_binary_reference(events, period, guard_frames=25):
    """Use pass interiors as alive and unambiguous out/goal-to-restart gaps as dead.

    Exclude fouls/advantages and uncertain gaps. No interpolation of team ownership,
    no target inference from ball availability, and no hand corrections are made.
    """
    n = len(period)
    alive = np.zeros(n, bool)
    dead = np.zeros(n, bool)
    core_dead = np.zeros(n, bool)
    accepted, rejected = [], []
    action_types = {'PASS', 'SHOT', 'BALL LOST', 'BALL OUT', 'RECOVERY', 'CHALLENGE'}
    records = sorted(events, key=lambda e: (int(e['Period']), int(e['Start Frame'])))
    # Only intervals explicitly occupied by successful passes receive alive truth.
    for e in records:
        p, start, end = int(e['Period']), int(e['Start Frame'])-1, int(e['End Frame'])-1
        if e['Type'] == 'PASS' and 0 <= start < end < n and np.all(period[start:end+1] == p):
            alive[start+1:end] = True
    for e in records:
        tokens = set(e.get('Subtype', '').split('-'))
        if not (e['Type'] == 'BALL OUT' or (e['Type'] == 'SHOT' and tokens.intersection({'OUT', 'GOAL'}))):
            continue
        p, end = int(e['Period']), int(e['End Frame'])-1
        if not 0 <= end < n or period[end] != p:
            continue
        future = [v for v in records if int(v['Period']) == p and int(v['Start Frame'])-1 > end and
                  (v['Type'] in action_types or v['Type'] == 'SET PIECE')]
        if not future:
            rejected.append({'stop_frame': end, 'reason': 'No subsequent action in this half'})
            continue
        start = int(future[0]['Start Frame'])-1
        simultaneous = [v for v in future if int(v['Start Frame'])-1 == start]
        restart = any(v['Type'] == 'SET PIECE' or 'GOAL KICK' in v.get('Subtype', '') for v in simultaneous)
        if not restart or start >= n or not np.all(period[end:start+1] == p):
            rejected.append({'stop_frame': end, 'reason': 'Next action is not an explicit restart'})
            continue
        lo, hi = end+1, start
        if lo >= hi:
            continue
        dead[lo:hi] = True
        if hi-lo > 2*guard_frames:
            core_dead[lo+guard_frames:hi-guard_frames] = True
        accepted.append({'stop_frame': end, 'restart_frame': start, 'period': p,
                         'source_type': e['Type'], 'source_subtype': e.get('Subtype', '')})
    conflict = alive & dead
    labels = np.full(n, -1, dtype=np.int8)
    labels[alive & ~conflict] = 0
    labels[dead & ~conflict] = 1
    core = np.full(n, -1, dtype=np.int8)
    core[alive & ~conflict] = 0
    core[core_dead & ~conflict] = 1
    return labels, core, {'accepted_stoppages': accepted, 'rejected_stoppages': rejected,
                         'conflicting_frames_excluded': int(conflict.sum()),
                         'reference_frames': int((labels >= 0).sum()),
                         'alive_frames': int((labels == 0).sum()), 'dead_frames': int((labels == 1).sum()),
                         'unknown_frames': int((labels < 0).sum()), 'core_guard_frames': guard_frames}
