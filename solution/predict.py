"""One-command tracking -> frozen learned models -> validated prediction CSV."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'solution/src'))
from possession.inference import load_bundle,read_tracking_npz,predict_tracking,write_prediction,output_paths
from possession.inputs import read_challenge


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input',type=Path,help='Tracking-only six-field NPZ in metres, native 25 Hz')
    source.add_argument('--challenge',action='store_true',help='Read local tactical metadata/raw tracking through games.py; no reference labels or tags')
    parser.add_argument('--variant',required=True,choices=['baseline','event2','event4'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--quiet',action='store_true',help='Print only output paths, without row/timing numbers')
    args=parser.parse_args()
    csv_path,sidecar=output_paths(args.output)
    if not args.overwrite and (csv_path.exists() or sidecar.exists()):parser.error('Output exists; use another path or --overwrite')
    bundle=load_bundle(args.bundle)
    protected={bundle['folder']/'manifest.json',bundle['folder']/'models.joblib'}
    if args.input:protected.add(args.input.resolve())
    if csv_path in protected or sidecar in protected:parser.error('Output must not replace input or model artifacts')
    if args.input:
        inputs=read_tracking_npz(args.input);description={'kind':'tracking_npz','path':str(args.input.resolve())}
    else:
        sys.path.insert(0,str(ROOT/'more-than-a-hack-2027-main repo'))
        import games
        paths=games.asset_paths()
        inputs,_=read_challenge(paths['atd_meta'],paths['atd_raw'],games.source_frames(),
            games.GAMES[games.GAME]['pitch'],games.half_video_blocks(games.GAME))
        description={'kind':'local_tactical_tracking','metadata':paths['atd_meta'],'raw_tracking':paths['atd_raw'],
                     'video_read':False,'smart_tags_read':False}
    result=predict_tracking(bundle,inputs,args.variant)
    report=write_prediction(args.output,result,bundle,args.variant,description,inputs['period'],args.overwrite)
    print(f'Prediction: {csv_path}')
    print(f'Run record: {sidecar}')
    if not args.quiet:print(f'PASS: {report["rows"]} rows; official format valid; no reference arrays read; variant {args.variant}')


if __name__=='__main__':main()
