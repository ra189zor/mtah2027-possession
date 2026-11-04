"""Evaluation-only adapter for Metrica's normalized EPTS/JSON sample schema.

Tracking conversion never reads events. Stable player slots come from metadata,
not row positions, which change at substitutions. References remain partial.
"""
from collections import Counter
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .inference import validate_tracking
from .transfer_validation import partial_reference


def metadata(path):
    root = ET.parse(path).getroot()
    score = root.find('.//Score')
    sides = {score.attrib['idLocalTeam']: 0, score.attrib['idVisitingTeam']: 1}
    if len(sides) != 2: raise ValueError('Expected distinct local and visiting teams')
    players = root.findall('.//Players/Player')
    identities = [p.attrib['id'] for p in players]
    if len(set(identities)) != len(identities): raise ValueError('Duplicate player identity')
    teams = np.array([sides[p.attrib['teamId']] for p in players], np.int8)
    channels = {c.attrib['id']: (identities.index(c.attrib['playerId']), c.attrib['channelId'])
                for c in root.findall('.//PlayerChannels/PlayerChannel')}
    units = {c.attrib['id']: (c.findtext('Name'), c.findtext('Unit'))
             for c in root.findall('.//Devices/Device/Sensors/Sensor/Channels/Channel')}
    if units != {'x': ('position_x', 'normalized'), 'y': ('position_y', 'normalized')}:
        raise ValueError('Expected explicitly normalized x/y channels')
    fps = float(root.findtext('.//GlobalConfig/FrameRate'))
    if fps != 25: raise ValueError('Expected 25 Hz metadata')
    field = root.find('.//FieldSize')
    pitch = np.array([float(field.findtext('Width')), float(field.findtext('Height'))])
    params = {p.findtext('Name'): p.findtext('Value') for p in
              root.findall('.//GlobalConfig/ProviderGlobalParameters/ProviderParameter')}
    halves = [(int(params[k+'_start']), int(params[k+'_end'])) for k in ('first_half', 'second_half')]
    if not 1 <= halves[0][0] <= halves[0][1] < halves[1][0] <= halves[1][1]:
        raise ValueError('Invalid published halves')
    n = halves[-1][1]; period = np.zeros(n, np.int8)
    for p, (a, b) in enumerate(halves, 1): period[a-1:b] = p
    formats = []
    for spec in root.findall('./DataFormatSpecifications/DataFormatSpecification'):
        a, b = int(spec.attrib['startFrame']), int(spec.attrib['endFrame'])
        children = list(spec)
        if spec.attrib['separator'] != ':' or len(children) != 3 or children[0].attrib.get('name') != 'frameCount':
            raise ValueError('Unsupported EPTS row format')
        player_block, ball_block = children[1:]
        if player_block.attrib.get('separator') != ';' or ball_block.attrib.get('separator') != ',':
            raise ValueError('Unsupported coordinate separators')
        mapping = []
        for pair in player_block:
            if pair.attrib.get('separator') != ',' or len(pair) != 2: raise ValueError('Expected player coordinate pair')
            values = [channels[c.attrib['playerChannelId']] for c in pair]
            if values[0][0] != values[1][0] or {v[1] for v in values} != {'x', 'y'}:
                raise ValueError('Mixed player or duplicate coordinate channels')
            mapping.append((values[0][0], [v[1] for v in values]))
        axes = [c.attrib['channelId'] for c in ball_block]
        if len(axes) != 2 or set(axes) != {'x', 'y'}: raise ValueError('Invalid ball channels')
        if len(set(v[0] for v in mapping)) != len(mapping): raise ValueError('Duplicate player within row format')
        formats.append((a, b, mapping, axes))
    formats.sort(key=lambda x: x[0])
    if not formats or formats[0][0] != 1 or formats[-1][1] != n or any(
            a > b or (i and a != formats[i-1][1]+1) for i, (a, b, _, _) in enumerate(formats)):
        raise ValueError('Format ranges must partition the full frame timeline')
    return dict(sides=sides, identities=identities, teams=teams, pitch=pitch, fps=fps,
                halves=halves, period=period, formats=formats)


def read_epts_tracking(meta_path, tracking_path):
    meta = metadata(meta_path); n = len(meta['period'])
    players = np.full((n, len(meta['identities']), 2), np.nan, np.float32)
    ball = np.full((n, 2), np.nan, np.float32); seen = np.zeros(n, bool)
    fmt = 0; previous = 0
    with Path(tracking_path).open(encoding='utf-8') as handle:
        for line in handle:
            fields = line.strip().split(':')
            if len(fields) != 3: raise ValueError('Malformed EPTS row')
            frame = int(fields[0])
            if not previous < frame <= n: raise ValueError('Frames must be ordered, unique and within metadata')
            previous = frame; seen[frame-1] = True
            while frame > meta['formats'][fmt][1]: fmt += 1
            _, _, mapping, axes = meta['formats'][fmt]
            pairs = fields[1].split(';')
            if len(pairs) != len(mapping): raise ValueError('Player columns differ from metadata')
            values = np.array([pair.split(',') for pair in pairs], dtype=np.float32)
            if values.shape != (len(mapping), 2): raise ValueError('Malformed player pair')
            for i, (slot, order) in enumerate(mapping):
                players[frame-1, slot] = values[i, [order.index('x'), order.index('y')]]
            value = np.array(fields[2].split(','), np.float32)
            if value.shape != (2,): raise ValueError('Malformed ball pair')
            ball[frame-1] = value[[axes.index('x'), axes.index('y')]]
    if not seen.any(): raise ValueError('Empty tracking')
    # Never interpolate absent rows or clip out-of-bounds observations.
    result = dict(players_xy=(players*meta['pitch']).astype(np.float32),
                  ball_xy=(ball*meta['pitch']).astype(np.float32), team_of_slot=meta['teams'],
                  pitch_m=meta['pitch'], fps=np.array(meta['fps']), period=meta['period'])
    validate_tracking(result)
    return result, dict(frames=n, tracking_rows=int(seen.sum()), missing_rows=int((~seen).sum()),
        coordinate_scale_m=meta['pitch'].tolist(), fps=meta['fps'], published_halves=meta['halves'],
        format_ranges=[[a, b] for a, b, _, _ in meta['formats']],
        stable_player_slots=len(meta['identities']), team_ids=meta['sides'],
        ball_missing_fraction=float((~np.isfinite(ball).all(1)).mean()),
        input_fields=sorted(result), events_read=False)


def read_json_events(meta_path, event_path):
    meta = metadata(meta_path); records = json.loads(Path(event_path).read_text(encoding='utf-8'))['data']
    events = []; carries = []; clock_errors = []; counts = Counter()
    identities = dict(zip(meta['identities'], meta['teams'].tolist()))
    for event in records:
        kind = event['type']['name']; counts[kind] += 1
        p = int(event['period']); a, b = (event[k]['frame'] for k in ('start', 'end'))
        if not isinstance(a, int) or not isinstance(b, int) or not 1 <= a <= b <= len(meta['period']):
            raise ValueError('Invalid event frame interval')
        if p not in (1, 2) or not np.all(meta['period'][a-1:b] == p): raise ValueError('Event crosses published half')
        for k in ('start', 'end'):
            error = abs(event[k]['time']-event[k]['frame']/meta['fps']); clock_errors.append(error)
            if error > .011: raise ValueError('Event time/frame alignment differs')
        team = meta['sides'][event['team']['id']]
        endpoints = []
        for k in ('from', 'to'):
            identity = event[k]['id'] if event.get(k) else ''
            if identity and identities[identity] != team: raise ValueError('Event endpoint belongs to other team')
            endpoints.append(identity)
        sub = event.get('subtypes') or []
        if isinstance(sub, dict): sub = [sub]
        subtype = '-'.join(x['name'] for x in sub)
        row = {'Type': 'RECOVERY' if kind == 'CARRY' else kind, 'Team': ['Home', 'Away'][team],
            'Period': str(p), 'Start Frame': str(a), 'End Frame': str(b), 'Subtype': subtype,
            'From': endpoints[0], 'To': endpoints[1]}
        events.append(row)
        if kind == 'CARRY':
            carries.append((a-1, b, row['Team']))
            # Both ends block a presumed out-to-restart gap across live carrying.
            events.append(row | {'Start Frame': str(b)})
    audit = dict(source_events=len(records), source_event_types=dict(counts),
        max_event_clock_error_seconds=max(clock_errors, default=0),
        carry_policy='Live exclusion only: RECOVERY blockers at both ends; no H/A labels from CARRY. Remove dead labels and opposite-team pass labels overlapping a carry.',
        normalized_event_records=len(events), frame_indexing='Source frame 1 maps to array index 0; no offset fit')
    return events, carries, audit


def game3_reference(meta_path, event_path, period, guard_frames=25):
    events, carries, event_audit = read_json_events(meta_path, event_path)
    np.testing.assert_array_equal(period, metadata(meta_path)['period'])
    raw, guarded, audit = partial_reference(events, period, guard_frames)
    exclude = np.zeros(len(period), bool)
    for a, b, team in carries:
        opposite = 'A' if team == 'Home' else 'H'
        exclude[a:b] |= np.isin(raw[a:b], ['D', opposite])
    audit['carry_conflict_frames_excluded'] = int(exclude.sum())
    raw[exclude] = 'X'; guarded[exclude] = 'X'
    audit['coverage'] = {name: {label: int((ref == label).sum()) for label in 'HADX'}
                         for name, ref in [('event_interiors', raw), ('guarded_dead', guarded)]}
    audit['event_schema'] = event_audit
    return raw, guarded, audit
