"""Read model-facing tracking inputs without opening possession reference labels."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .validation import check_inputs


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array_digest(arrays):
    h = hashlib.sha256()
    for key in sorted(arrays):
        h.update(key.encode())
        h.update(str(arrays[key].dtype).encode())
        h.update(str(arrays[key].shape).encode())
        h.update(arrays[key].tobytes())
    return h.hexdigest()


def read_challenge(meta_path, raw_path, n_frames, pitch_m, half_blocks):
    tree = ET.parse(meta_path)
    teams = list(tree.iter('Team'))
    if len(teams) != 2:
        raise ValueError('Expected home and away metadata teams')
    # The provided EPTS export lists the home side first, as the official baseline does.
    home_id, away_id = teams[0].get('id'), teams[1].get('id')
    slots = list(tree.iter('Player'))
    if any(slot.get('teamId') not in (home_id, away_id) for slot in slots):
        raise ValueError('Unknown metadata team assignment')
    membership = np.array([0 if slot.get('teamId') == home_id else 1 for slot in slots], dtype=np.int8)
    players = np.full((n_frames, len(slots), 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    count = 0
    with Path(raw_path).open(encoding='utf-8') as handle:
        for count, line in enumerate(handle, start=1):
            fields = line.strip().split(':')
            if len(fields) != 3 or int(fields[0]) != count or count > n_frames:
                raise ValueError(f'ATD index/schema mismatch at row {count}')
            groups = np.fromstring(fields[1].replace(';', ','), sep=',', dtype=np.float32)
            point = np.fromstring(fields[2], sep=',', dtype=np.float32)
            if groups.size != len(slots) * 3 or point.size != 3:
                raise ValueError('Unexpected coordinate/speed field count')
            players[count - 1] = groups.reshape(-1, 3)[:, :2]
            ball[count - 1] = point[:2]
    if count != n_frames:
        raise ValueError(f'Tracking row count {count} != {n_frames}')
    period = np.zeros(n_frames, dtype=np.int8)
    for half, first, last in half_blocks:
        if not 0 <= first <= last < n_frames or (period[first:last + 1] != 0).any():
            raise ValueError('Invalid/overlapping published period boundaries')
        period[first:last + 1] = half
    pitch = np.array(pitch_m, dtype=np.float32)
    inputs = {'players_xy': players * pitch, 'ball_xy': ball * pitch,
        'team_of_slot': membership, 'period': period, 'pitch_m': pitch, 'fps': np.array(25)}
    check_inputs(inputs)
    return inputs, {teams[0].findtext('Name'): 'H', teams[1].findtext('Name'): 'A'}


def read_external(root, match_id, split_config):
    if match_id in split_config['sealed_test_matches']:
        raise ValueError('Sealed test feature extraction is locked')
    if match_id not in split_config['development_matches'] + split_config['validation_matches']:
        raise ValueError('External match is absent from the frozen split')
    folder = Path(root) / match_id / 'prepared'
    audit = json.loads((folder / 'audit.json').read_text(encoding='utf-8'))
    # Verify only the model-facing artifact. Do not open targets.npz here.
    if file_digest(folder / 'inputs.npz') != audit['inputs_sha256']:
        raise ValueError('Prepared input checksum mismatch')
    with np.load(folder / 'inputs.npz', allow_pickle=False) as bundle:
        inputs = {key: bundle[key] for key in bundle.files}
    check_inputs(inputs)
    return inputs
