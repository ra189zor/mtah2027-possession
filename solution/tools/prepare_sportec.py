"""Stream a public Sportec XML match into separate anonymous inputs and targets."""
import argparse
import collections
from datetime import datetime
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def frames(path, ball):
    """Read attributes before clearing elements; align every observation by its own ID."""
    context = None
    for event, element in ET.iterparse(path, events=('start', 'end')):
        if event == 'start' and element.tag == 'FrameSet':
            context = dict(element.attrib)
        elif event == 'end' and element.tag == 'Frame':
            if context is not None and (context.get('TeamId', '').lower() == 'ball') == ball:
                yield context, dict(element.attrib)
            element.clear()
        elif event == 'end' and element.tag == 'FrameSet':
            context = None
            element.clear()


def code(value):
    try:
        number = float(value)
        return int(number) if np.isfinite(number) and number == int(number) else -1
    except (TypeError, ValueError):
        return -1


def labels_from_codes(status, owner):
    labels = np.full(len(status), 'X', dtype='<U1')
    labels[status == 0] = 'D'
    labels[(status == 1) & (owner == 1)] = 'H'
    labels[(status == 1) & (owner == 2)] = 'A'
    return labels


def prepare(match_id, data_root, config):
    if match_id in config['sealed_test_matches']:
        raise ValueError('Sealed test match is locked; do not inspect or prepare it yet')
    allowed = config['development_matches'] + config['validation_matches']
    if match_id not in allowed:
        raise ValueError('Match absent from frozen development/validation split')
    folder = data_root / match_id
    manifest = json.loads((folder / 'download_manifest.json').read_text(encoding='utf-8'))
    meta_path, = folder.glob('*matchinformation*.xml')
    raw_path, = folder.glob('*positions_raw_observed*.xml')
    for path in (meta_path, raw_path):
        expected, = [x['sha256'] for x in manifest['files'] if Path(x['path']).name == path.name]
        if digest(path) != expected:
            raise ValueError(f'Raw asset hash changed: {path.name}')
    meta = ET.parse(meta_path).getroot().find('MatchInformation')
    general, environment = meta.find('General'), meta.find('Environment')
    if general.get('MatchId') != 'DFL-MAT-' + match_id:
        raise ValueError('Metadata match ID mismatch')
    home = general.get('HomeTeamId')
    away = general.get('AwayTeamId') or general.get('GuestTeamId')
    pitch = np.array([float(environment.get('PitchX')), float(environment.get('PitchY'))], dtype=np.float32)
    roster = {}
    slots_per_team = 23
    for team in meta.find('Teams'):
        if team.get('TeamId') not in (home, away):
            continue
        side = 0 if team.get('TeamId') == home else 1
        players = sorted(p.get('PersonId') for p in team.findall('Players/Player'))
        if len(players) > slots_per_team:
            raise ValueError('External roster exceeds anonymous slot budget')
        for slot, identity in enumerate(players):
            roster[identity] = (side * slots_per_team + slot, team.get('TeamId'))
    ball_records = collections.defaultdict(list)
    for context, frame in frames(raw_path, ball=True):
        if context.get('GameSection') not in ('firstHalf', 'secondHalf'):
            raise ValueError('Unexpected match period')
        ball_records[context['GameSection']].append((int(frame['N']), float(frame['X']),
            float(frame['Y']), code(frame.get('BallStatus')), code(frame.get('BallPossession')),
            datetime.fromisoformat(frame['T']).timestamp()))
    if set(ball_records) != {'firstHalf', 'secondHalf'}:
        raise ValueError('Expected exactly two ball periods')
    spans, cursor, period_reports = {}, 0, []
    for section in ('firstHalf', 'secondHalf'):
        records = ball_records[section]
        ids = np.array([row[0] for row in records], dtype=np.int64)
        if not np.all(np.diff(ids) > 0):
            raise ValueError('Duplicate or unordered ball frame IDs')
        times = np.array([row[5] for row in records])
        if not np.allclose(np.diff(times), np.diff(ids) / 25, atol=0.002, rtol=0):
            raise ValueError('Timestamp/frame-ID increments do not agree with 25 Hz')
        start, stop = int(ids[0]), int(ids[-1]) + 1
        spans[section] = (start, stop, cursor)
        period_reports.append({'section': section, 'external_first_id': start,
            'external_last_id': stop - 1, 'observed_ball_rows': len(ids),
            'missing_ball_frame_ids': stop - start - len(ids), 'output_start': cursor,
            'output_stop': cursor + stop - start, 'fps_verified': 25})
        cursor += stop - start + 1  # ungraded separator prevents artificial cross-half spells
    n = cursor - 1
    players_xy = np.full((n, slots_per_team * 2, 2), np.nan, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)
    status = np.full(n, -1, dtype=np.int8)
    owner = np.full(n, -1, dtype=np.int8)
    period = np.zeros(n, dtype=np.int8)
    for half, section in enumerate(('firstHalf', 'secondHalf'), start=1):
        first, stop, offset = spans[section]
        period[offset:offset + stop - first] = half
        for identity, x, y, state, team, _ in ball_records[section]:
            index = offset + identity - first
            ball_xy[index] = (x, y)
            status[index], owner[index] = state, team
    del ball_records
    excluded = 0
    for context, frame in frames(raw_path, ball=False):
        identity = context.get('PersonId')
        section = context.get('GameSection')
        if identity not in roster or section not in spans:
            excluded += 1
            continue
        slot, expected_team = roster[identity]
        if context.get('TeamId') != expected_team:
            raise ValueError('Player team disagrees with metadata')
        first, stop, offset = spans[section]
        frame_id = int(frame['N'])
        if not first <= frame_id < stop:
            excluded += 1
            continue
        index = offset + frame_id - first
        if np.isfinite(players_xy[index, slot]).any():
            raise ValueError('Duplicate player observation')
        players_xy[index, slot] = float(frame['X']), float(frame['Y'])
    # Sportec uses centred metres. The model receives corner-origin metres, no identities.
    players_xy += pitch / 2
    ball_xy += pitch / 2
    labels = labels_from_codes(status, owner)
    output = folder / 'prepared'
    output.mkdir(exist_ok=True)
    np.savez_compressed(output / 'inputs.npz', players_xy=players_xy, ball_xy=ball_xy,
        team_of_slot=np.repeat(np.array([0, 1], dtype=np.int8), slots_per_team),
        period=period, pitch_m=pitch, fps=np.array(25))
    np.savez_compressed(output / 'targets.npz', labels=labels)
    graded = labels != 'X'
    summary = {'match_id': match_id, 'split': 'development' if match_id in config['development_matches'] else 'validation',
        'revision': manifest['revision'], 'raw_sha256': digest(raw_path),
        'pitch_m': pitch.tolist(), 'fps': 25, 'periods': period_reports,
        'output_rows': n, 'label_counts': dict(collections.Counter(labels.tolist())),
        'ball_status_codes': dict(collections.Counter(map(str, status[period > 0].tolist()))),
        'ownership_codes': dict(collections.Counter(map(str, owner[period > 0].tolist()))),
        'mean_players_on_graded': float(np.isfinite(players_xy[graded, :, 0]).sum(axis=1).mean()),
        'ball_present_on_graded': int(np.isfinite(ball_xy[graded]).all(axis=1).sum()),
        'coordinate_convention': 'corner-origin metres; converted from centred Sportec metres',
        'fraction_player_observations_inside_pitch': float(
            ((players_xy >= 0) & (players_xy <= pitch)).all(axis=2)[np.isfinite(players_xy).all(axis=2)].mean()),
        'excluded_nonroster_or_outside_period_player_rows': excluded,
        'input_fields': ['players_xy', 'ball_xy', 'team_of_slot', 'period', 'pitch_m', 'fps'],
        'targets_separate': True, 'identities_exposed_to_model': False,
        'inputs_sha256': digest(output / 'inputs.npz'), 'targets_sha256': digest(output / 'targets.npz')}
    (output / 'audit.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match', required=True)
    parser.add_argument('--root', type=Path, default=ROOT / 'data/external/sportec')
    args = parser.parse_args()
    config = json.loads((ROOT / 'solution/configs/validation.json').read_text())
    prepare(args.match, args.root, config)
