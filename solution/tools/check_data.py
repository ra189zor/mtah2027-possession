"""Verify prepared local assets and report aggregates; never export match records."""
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / 'more-than-a-hack-2027-main repo'
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(UPSTREAM / 'starter'))
import games
import load_smart_tagging


def tracking_check(raw, slots, expected):
    count = 0
    present = 0
    with raw.open(encoding='utf-8') as handle:
        for count, line in enumerate(handle, start=1):
            fields = line.rstrip().split(':')
            if len(fields) != 3 or int(fields[0]) != count:
                raise ValueError(f'Tracking ID/format mismatch at record {count}: {raw.name}')
            player_groups = fields[1].split(';')
            if len(player_groups) != slots:
                raise ValueError(f'Track-slot count mismatch at record {count}')
            players = np.fromstring(fields[1].replace(';', ','), sep=',')
            ball = np.fromstring(fields[2], sep=',')
            if players.size != slots * 3 or ball.size != 3:
                raise ValueError(f'Coordinate/speed field count mismatch at record {count}')
            if np.isinf(players).any() or np.isinf(ball).any():
                raise ValueError(f'Infinite coordinate at record {count}')
            present += int(np.isfinite(ball[:2]).all())
    if count != expected:
        raise ValueError(f'Tracking count {count} != {expected}')
    return {'records': count, 'ids_exact_1_to_n': True,
            'slots': slots, 'ball_present': present, 'schema_valid': True}


def video_check(path, expected, source):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Video decoder could not open {path.name}')
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        probes = [0, n // 2, n - 1]
        if source == 'tactical':
            probes += [11185, 18684, 90651, 110808, 113742, 190197]
        decoded = []
        for frame in sorted(set(probes)):
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame):
                raise ValueError(f'Could not seek to frame {frame}: {path.name}')
            ok, pixels = cap.read()
            if not ok or pixels is None or pixels.shape[:2] != (height, width):
                raise ValueError(f'Failed to decode frame {frame}: {path.name}')
            decoded.append(frame)
        if source == 'tactical' and (n != expected or abs(fps - 25) > 0.001):
            raise ValueError('Tactical video does not match reference count/rate')
        return {'decoder': cap.getBackendName(), 'reported_frames': n,
            'expected_tracking_frames': expected, 'frame_count_matches': n == expected,
            'fps': fps, 'width': width, 'height': height, 'decoded_probe_frames': decoded,
            'full_sequential_decode_performed': False}
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=UPSTREAM / 'data')
    args = parser.parse_args()
    report = {'sources': {}, 'warnings': [], 'opencv_version': cv2.__version__}
    previous = json.loads((ROOT / 'research/dataset_audit.json').read_text(encoding='utf-8'))
    for source, spec in games.SOURCES.items():
        paths = {key: Path(value) for key, value in games.asset_paths(
            data_dir=str(args.data_dir), source=source).items()}
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        meta = ET.parse(paths['atd_meta'])
        teams = list(meta.iter('Team'))
        players = list(meta.iter('Player'))
        if len(teams) != 2 or any(p.get('teamId') not in {t.get('id') for t in teams} for p in players):
            raise ValueError(f'Invalid team assignments: {source}')
        tracking = tracking_check(paths['atd_raw'], len(players), spec['n_video_frames'])
        if tracking['ball_present'] != previous['tracking'][source]['ball_present']:
            raise ValueError(f'Ball coverage differs from archive audit: {source}')
        table = pd.read_csv(paths['homography'])
        matrix_cols = [f'h{r}{c}' for r in range(3) for c in range(3)]
        if len(table) != spec['n_video_frames'] or not np.array_equal(table['frame'], np.arange(len(table))):
            raise ValueError(f'Homography frame mismatch: {source}')
        if not table['valid'].isin([0, 1]).all():
            raise ValueError(f'Invalid homography validity flag: {source}')
        valid = table['valid'].astype(bool).to_numpy()
        matrices = table[matrix_cols].to_numpy().reshape(-1, 3, 3)
        if not np.isfinite(matrices[valid]).all() or not np.isfinite(table.loc[valid, 'reprojection_error']).all():
            raise ValueError(f'Nonfinite valid homography: {source}')
        if (np.abs(np.linalg.det(matrices[valid])) < 1e-12).any():
            raise ValueError(f'Singular valid homography: {source}')
        if not np.isnan(matrices[~valid]).all():
            raise ValueError(f'Invalid homography has unexpected matrix values: {source}')
        video = video_check(paths['video'], spec['n_video_frames'], source)
        report['sources'][source] = {'all_asset_paths_resolve': True, 'tracking': tracking,
            'homography': {'rows': len(table), 'valid_rows': int(valid.sum()),
                           'frames_exact_0_to_n_minus_1': True}, 'video': video}
        if not video['frame_count_matches']:
            report['warnings'].append('Broadcast decoder reports one fewer video frame than tracking rows; '
                'no synchronization or padding applied. Broadcast fusion remains deferred.')
        print(f'{source}: tracking IDs/schema, homography and video probes verified', flush=True)
    events = load_smart_tagging.load(data_dir=str(args.data_dir))
    timing = load_smart_tagging.check_time_base(events)
    if len(events) != 638 or not timing['agrees']:
        raise ValueError('Smart Tagging count/kickoff timing check failed')
    report['smart_tagging'] = {'instances': len(events), 'kickoff_containment_both_halves': timing['agrees']}
    report['status'] = 'passed_with_broadcast_warning' if report['warnings'] else 'passed'
    out = ROOT / 'research/step2_data_check.json'
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
