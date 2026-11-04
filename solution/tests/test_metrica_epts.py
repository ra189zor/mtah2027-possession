"""Check EPTS substitution mappings, missing rows and conservative JSON references."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from possession.metrica_epts import read_epts_tracking, game3_reference


def fixture(folder):
    formats = []
    for a, b, ids in [(1, 25, ['hx', 'hy', 'ax', 'ay']), (26, 50, ['ay', 'ax', 'hy', 'hx'])]:
        pairs = ''.join('<SplitRegister separator=",">'+''.join(
            f'<PlayerChannelRef playerChannelId="{v}"/>' for v in ids[i:i+2])+'</SplitRegister>' for i in (0, 2))
        formats.append(f'<DataFormatSpecification startFrame="{a}" endFrame="{b}" separator=":">'
            f'<StringRegister name="frameCount"/><SplitRegister separator=";">{pairs}</SplitRegister>'
            '<SplitRegister separator=","><BallChannelRef channelId="y"/><BallChannelRef channelId="x"/>'
            '</SplitRegister></DataFormatSpecification>')
    params = ''.join(f'<ProviderParameter><Name>{k}</Name><Value>{v}</Value></ProviderParameter>' for k, v in
        [('first_half_start', 1), ('first_half_end', 25), ('second_half_start', 26), ('second_half_end', 50)])
    channels = ''.join(f'<PlayerChannel id="{p}{axis}" playerId="{p}" channelId="{axis}"/>'
                       for p in ('h', 'a') for axis in ('x', 'y'))
    meta = folder/'meta.xml'
    meta.write_text(f'<main><Metadata><GlobalConfig><FrameRate>25</FrameRate><ProviderGlobalParameters>{params}'
        '</ProviderGlobalParameters></GlobalConfig><Score idLocalTeam="H" idVisitingTeam="A"/>'
        '<FieldSize><Width>105</Width><Height>68</Height></FieldSize><Players>'
        '<Player id="h" teamId="H"/><Player id="a" teamId="A"/></Players><Devices><Device><Sensors><Sensor><Channels>'
        '<Channel id="x"><Name>position_x</Name><Unit>normalized</Unit></Channel>'
        '<Channel id="y"><Name>position_y</Name><Unit>normalized</Unit></Channel></Channels></Sensor></Sensors></Device>'
        f'</Devices><PlayerChannels>{channels}</PlayerChannels></Metadata><DataFormatSpecifications>'
        + ''.join(formats)+'</DataFormatSpecifications></main>')
    track = folder/'track.txt'
    track.write_text('1:0.1,0.2;0.3,0.4:0.6,0.5\n26:0.4,0.3;0.2,0.1:NaN,NaN\n50:0.4,0.3;0.2,0.1:0.6,1.1\n')
    return meta, track


def event(kind, a, b=None, team='H'):
    b = a if b is None else b
    return dict(type={'name': kind}, period=1, team={'id': team}, subtypes=None,
                start={'frame': a, 'time': a/25}, end={'frame': b, 'time': b/25},
                **{'from': {'id': 'h' if team == 'H' else 'a'}, 'to': None})


class EptsTests(unittest.TestCase):
    def test_reordered_axes_and_players_missing_rows_and_no_clipping(self):
        with tempfile.TemporaryDirectory() as name:
            meta, track = fixture(Path(name)); x, audit = read_epts_tracking(meta, track)
            np.testing.assert_array_equal(x['players_xy'][0], x['players_xy'][25])
            np.testing.assert_allclose(x['players_xy'][0], [[10.5, 13.6], [31.5, 27.2]])
            np.testing.assert_allclose(x['ball_xy'][0], [52.5, 40.8])
            self.assertTrue(np.isnan(x['players_xy'][1:25]).all())
            self.assertTrue(np.isnan(x['ball_xy'][25]).all())
            self.assertGreater(x['ball_xy'][-1, 0], 105)
            np.testing.assert_array_equal(x['period'], [1]*25+[2]*25)
            self.assertEqual(audit['missing_rows'], 47)

    def test_duplicate_frames_and_malformed_columns_fail(self):
        with tempfile.TemporaryDirectory() as name:
            meta, track = fixture(Path(name))
            for content in ['1:0,0;1,1:0,0\n1:0,0;1,1:0,0\n', '1:0,0:0,0\n']:
                track.write_text(content)
                with self.assertRaises(ValueError): read_epts_tracking(meta, track)

    def test_carry_blocks_false_dead_interval_without_ownership_truth(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name); meta, track = fixture(folder); x, _ = read_epts_tracking(meta, track)
            path = folder/'events.json'
            path.write_text(json.dumps({'data': [event('BALL OUT', 2), event('CARRY', 6, 10), event('SET PIECE', 20)]}))
            raw, guarded, audit = game3_reference(meta, path, x['period'], 1)
            self.assertTrue((raw == 'X').all()); self.assertTrue((guarded == 'X').all())
            self.assertEqual(audit['event_schema']['source_event_types']['CARRY'], 1)
            path.write_text(json.dumps({'data': [event('BALL OUT', 2), event('SET PIECE', 20)]}))
            raw, guarded, _ = game3_reference(meta, path, x['period'], 1)
            self.assertEqual(int((raw == 'D').sum()), 17)
            self.assertEqual(int((guarded == 'D').sum()), 15)

    def test_clock_and_half_mismatch_fail(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name); meta, track = fixture(folder); x, _ = read_epts_tracking(meta, track)
            path = folder/'events.json'; bad = event('BALL OUT', 2); bad['start']['time'] = 5
            for value in [bad, event('CARRY', 24, 30)]:
                path.write_text(json.dumps({'data': [value]}))
                with self.assertRaises(ValueError): game3_reference(meta, path, x['period'])


if __name__ == '__main__': unittest.main()
