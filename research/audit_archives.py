"""Read-only local dataset audit; prints aggregates, never raw match records."""
import collections
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'More Than A Hack 2027 - Initial Challenge - Dataset'
REPO = ROOT / 'more-than-a-hack-2027-main repo'
sys.path.insert(0, str(REPO / 'evaluation'))
import possession_metrics as pm

def gaps(mask):
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return {'count': len(lengths), 'median_s': float(np.median(lengths) / 25),
            'p95_s': float(np.percentile(lengths, 95) / 25),
            'max_s': float(lengths.max() / 25)} if len(lengths) else {}

def mp4_metadata(path):
    tracks = []
    with path.open('rb') as f:
        def boxes(start, end):
            pos = start
            while pos + 8 <= end:
                f.seek(pos)
                size, kind = struct.unpack('>I4s', f.read(8))
                header = 8
                if size == 1:
                    size = struct.unpack('>Q', f.read(8))[0]
                    header = 16
                elif size == 0:
                    size = end - pos
                if size < header or pos + size > end:
                    raise ValueError('Invalid MP4 box')
                yield kind, pos + header, pos + size
                pos += size
        def descend(start, end, track=None):
            for kind, a, b in boxes(start, end):
                if kind == b'trak':
                    t = {}
                    tracks.append(t)
                    descend(a, b, t)
                elif kind in (b'moov', b'mdia', b'minf', b'stbl'):
                    descend(a, b, track)
                elif track is not None and kind == b'hdlr':
                    f.seek(a + 8)
                    track.setdefault('handler', f.read(4).decode('ascii'))
                elif track is not None and kind == b'mdhd':
                    f.seek(a)
                    version = f.read(1)[0]
                    f.seek(a + (20 if version else 12))
                    scale = struct.unpack('>I', f.read(4))[0]
                    duration = struct.unpack('>Q' if version else '>I', f.read(8 if version else 4))[0]
                    track.update(timescale=scale, duration_s=duration / scale)
                elif track is not None and kind == b'stsz':
                    f.seek(a + 8)
                    track['samples'] = struct.unpack('>I', f.read(4))[0]
                elif track is not None and kind == b'tkhd':
                    f.seek(b - 8)
                    w, h = struct.unpack('>II', f.read(8))
                    track.update(width=w / 65536, height=h / 65536)
        descend(0, path.stat().st_size)
    return {'bytes': path.stat().st_size, 'tracks': tracks}

report = {'archives': [], 'tracking': {}, 'homography': {}, 'videos': {}}
for archive in sorted(DATA.glob('*.zip')):
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            if entry.is_dir():
                continue
            report['archives'].append({'archive': archive.name, 'member': entry.filename,
                                       'bytes': entry.file_size})
            name = Path(entry.filename).name
            source = 'broadcast' if 'broadcast' in name else 'tactical'
            if name.endswith('_atd.txt'):
                seen, players, ids, graded_seen = [], [], [], 0
                digest = hashlib.sha256()
                with z.open(entry) as raw:
                    for line in raw:
                        digest.update(line)
                        parts = line.decode().strip().split(':')
                        ids.append(int(parts[0]))
                        seen.append(not parts[2].startswith('NaN'))
                        players.append(sum(not g.startswith('NaN') for g in parts[1].split(';')))
                seen, players, ids = np.array(seen), np.array(players), np.array(ids)
                r = {'records': len(ids), 'ids_exact_1_to_n': bool(np.array_equal(ids, np.arange(1, len(ids)+1))),
                     'ball_present': int(seen.sum()), 'ball_present_pct': float(100 * seen.mean()),
                     'sha256': digest.hexdigest()}
                if source == 'tactical':
                    graded = np.zeros(len(seen), dtype=bool)
                    graded[11185:90652] = True
                    graded[110808:190198] = True
                    r.update(graded_frames=int(graded.sum()), ball_present_graded=int(seen[graded].sum()),
                             mean_players_graded=float(players[graded].mean()),
                             missing_ball_runs_graded=gaps(~seen & graded))
                report['tracking'][source] = r
            elif name.endswith('_homography.csv'):
                with z.open(entry) as raw:
                    table = pd.read_csv(raw)
                good = table['valid'].astype(bool)
                report['homography'][source] = {'rows': len(table), 'frames_exact_0_to_n_minus_1':
                    bool(np.array_equal(table['frame'], np.arange(len(table)))),
                    'valid_rows': int(good.sum()), 'valid_pct': float(100 * good.mean()),
                    'error_median_valid': float(table.loc[good, 'reprojection_error'].median())}
            elif name.endswith('_atd.xml'):
                with z.open(entry) as raw:
                    tree = ET.parse(raw)
                report.setdefault('metadata', {})[source] = {'teams': len(list(tree.iter('Team'))),
                    'track_slots': len(list(tree.iter('Player')))}
            elif name.endswith('_smart_tagging.xml'):
                with z.open(entry) as raw:
                    tree = ET.parse(raw)
                events = list(tree.iter('instance'))
                report['tagging'] = {'instances': len(events),
                    'codes': dict(collections.Counter(x.findtext('code') for x in events)),
                    'team_tags': dict(collections.Counter(l.findtext('text') for x in events
                         for l in x.findall('label') if l.findtext('group') == 'Team'))}
            elif name == 'README.md':
                # Read locally; no raw restricted content in the report.
                report['dataset_readme_read'] = bool(z.read(entry).decode('utf-8-sig'))
for path in DATA.glob('*.mp4'):
    report['videos'][path.name] = mp4_metadata(path)
gt = pm.label_codes(pm.read_label_csv(REPO / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
report['public_gt'] = {'rows': len(gt), 'counts': dict(zip(pm.LABELS, np.bincount(gt).tolist())),
                       'graded_runs': pm.valid_runs(gt),
                       'possessions': len(pm.possession_spells(gt, pm.valid_runs(gt)))}
out = ROOT / 'research/dataset_audit.json'
out.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
