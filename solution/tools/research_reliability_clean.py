"""Refit the full specialist architecture with clean-domain quality supervision."""
import itertools
from pathlib import Path
import shutil
import sys
import joblib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'solution/tools'))
import research_reliability as r
import quality_domain_check as q
from possession.inputs import file_digest

OUT=ROOT/'outputs/reliability_research/run_20261101_clean'


def main():
    cfg,sc,rc,f=r.setup();OUT.mkdir(parents=True,exist_ok=True)
    sources=r.SOURCES+[Path(__file__),Path(q.__file__)]
    protocol={'config':rc,'matches':cfg['matches'],'source_sha256':{str(p.relative_to(ROOT)):file_digest(p) for p in sources},
              'foundation_source_sha256':r.s.setup()[2],'quality_training_regimes':['clean','moderate','severe'],
              'additional_clean_seed':829,'adaptive_reason':'Refit specialists after clean-domain quality false-alarm correction; no independent-test claim.'}
    if (OUT/'protocol.json').exists() and r.n.read(OUT/'protocol.json')!=protocol:raise ValueError('Corrected protocol drift')
    r.save(OUT/'protocol.json',protocol)
    for match in cfg['matches']:
        source=q.SOURCE/'features'/match;folder=OUT/'features'/match;folder.mkdir(parents=True,exist_ok=True)
        index=r.n.read(source/'index.json')
        for name,digest in index['files'].items():
            if name=='quality_training.npz':continue
            if file_digest(source/name)!=digest:raise ValueError('Feature drift')
            if not (folder/name).exists():shutil.copyfile(source/name,folder/name)
        training=q.OUT/'training'/f'{match}.npz'
        if not (folder/'quality_training.npz').exists():shutil.copyfile(training,folder/'quality_training.npz')
        index['files']['quality_training.npz']=file_digest(training);index['synthetic_ghost_seeds']=[811,823,829]
        r.save(folder/'index.json',index)
        # These seven six-match fits are byte-identical models already evaluated in the follow-up.
        source_model=q.OUT/'outer'/match/'model.joblib';report=r.n.read(source_model.parent/'report.json')
        if file_digest(source_model)!=report['model_sha256']:raise ValueError('Follow-up model drift')
        bundle=joblib.load(source_model);bundle['excluded']=[match]
        target=r.quality_path(OUT,[match]);target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():joblib.dump(bundle,target)
    exclusions=list(itertools.combinations(cfg['matches'],2))
    r.parallel(r.fit_quality,[(e,str(OUT)) for e in exclusions],rc['workers'])
    r.parallel(r.outer,[(m,str(OUT)) for m in cfg['matches']],rc['workers'])
    r.aggregate(OUT)


if __name__=='__main__':main()
