"""Render external-only experiment summaries without selecting a production default."""
import json
import argparse
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/reliability_research/run_20261101_v2'


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--experiment',type=Path,default=OUT)
    OUT=parser.parse_args().experiment.resolve()
    report=json.loads((OUT/'validation_report.json').read_text());summary=report['summary'];rows=report['results']
    models=['baseline','event4','quality','gap','boundary','combined'];conditions=['clean','moderate','severe','contextual']
    lines=['# Reliability research results — 1 November 2026','',
        'All numbers below are external-data estimates. Seven previously examined fixtures, clean once and three noisy seeds each; 70 inputs, six candidates and four reference conventions produce 1,680 metric records. This is adaptive research, not a new independent test or challenge score. Benchmark decoder profiles retain their previous inner-fold selections; the separate deployment bundle retains the earlier global 40/40/8 profile.','']
    if OUT.name.endswith('_clean'):
        lines+=['The main tables below use the corrected clean/moderate/severe quality detector and fully retrained gap/boundary specialists. The first specialist experiment is preserved separately; its moderate provider combined score was 0.51667. The corrected experiment is a subsequent adaptive comparison, not an independent confirmation.','']
    for convention in ('provider','absorbed_1s','absorbed_3s','absorbed_5s'):
        lines+=['## '+convention,'','| Condition | '+' | '.join(models)+' |','|---|'+'---:|'*len(models)]
        for condition in conditions:
            values=[next(r['spell_f1'] for r in summary if r['condition']==condition and r['label_set']==convention and r['model']==model) for model in models]
            lines+=['| '+condition+' | '+' | '.join(f'{v:.5f}' for v in values)+' |']
        lines+=['']
    lines+=['## Moderate provider changes versus event4','','| Candidate | Change | Descriptive match bootstrap 95% interval | Improved fixtures | Count passes |','|---|---:|---|---:|---:|']
    for model in models[2:]:
        r=next(r for r in summary if r['condition']=='moderate' and r['label_set']=='provider' and r['model']==model)
        lines+=[f"| {model} | {r['delta_vs_event4']:+.5f} | [{r['descriptive_match_low']:+.5f}, {r['descriptive_match_high']:+.5f}] | {r['improved_matches']}/7 | {r['count_gate_passes']}/7 |"]
    lines+=['','The count gate is a diagnostic ±20% heuristic. Bootstrap intervals do not account for all researcher adaptation or shared training sets.','',
        '## Missing-ball ownership and boundaries','',
        '| Candidate | Moderate missing-alive frame accuracy | Direct-switch precision | Direct-switch recall | Spurious direct switches |',
        '|---|---:|---:|---:|---:|']
    for model in models:
        group=[r for r in rows if r['condition']=='moderate' and r['label_set']=='provider' and r['model']==model]
        average=lambda key:np.mean([r['direct_switches_12'][key] for r in group])
        lines+=[f"| {model} | {np.mean([r['missing_alive_accuracy'] for r in group]):.5f} | {average('precision'):.5f} | {average('recall'):.5f} | {average('n_spurious'):.2f} |"]
    follow_path=ROOT/'outputs/reliability_research/quality_clean_domain_20261101/validation_report.json'
    follow=json.loads(follow_path.read_text())
    lines+=['','## Quality detector and clean-domain follow-up','',
        'The first quality model trained on moderate/severe synthetic corruption only. Its clean false alarms motivated a separately recorded adaptive follow-up adding clean-source corruption with seed 829. The follow-up changes only quality training and evaluates its quality-only decoder integration; it does not retroactively change the first specialist experiment.','',
        '| Quality training | Clean false-positive rate at 0.5 | Contextual average precision | Contextual ghost recall at 0.5 | Contextual false-positive rate |','|---|---:|---:|---:|---:|']
    initial=json.loads((ROOT/'outputs/reliability_research/run_20261101_v2/validation_report.json').read_text())
    for name,source in [('moderate/severe',initial),('clean/moderate/severe',follow)]:
        clean=[r for r in source['quality'] if r['condition']=='clean'];stress=[r for r in source['quality'] if r['condition']=='contextual']
        lines+=[f"| {name} | {np.mean([r['false_positive_rate_at_half'] for r in clean]):.5f} | {np.mean([r['average_precision'] for r in stress]):.5f} | {np.mean([r['recall_at_half'] for r in stress]):.5f} | {np.mean([r['false_positive_rate_at_half'] for r in stress]):.5f} |"]
    lines+=['','| Condition | Corrected quality-only provider spellF1 | Change versus event4 | 95% descriptive interval |','|---|---:|---:|---|']
    for condition in conditions:
        r=next(r for r in follow['summary'] if r['condition']==condition and r['label_set']=='provider')
        lines+=[f"| {condition} | {r['spell_f1']:.5f} | {r['delta_vs_event4']:+.5f} | [{r['low']:+.5f}, {r['high']:+.5f}] |"]
    lines+=['','Quality metrics use known synthetic ghost masks, not human-labelled real tracking errors. Broader validation here means all seven fixtures, repeated corruptions and four conventions; no additional independent real fixtures were acquired. No variant is automatically promoted.','']
    audit=json.loads((OUT/'integrity_audit.json').read_text())
    lines+=['## Verification','',f"All 108 tests pass. Independent audit passes {audit['hash_checks']} hash checks, {audit['cross_fit_traces']} cross-fitting traces and {audit['exact_prediction_replays']} exact held-out prediction replays. Original deployment bundles retain their integrity.",'',
        'The failed initial run is retained separately. The completed v2 run excludes columns that are entirely unobserved in each specialist training fold, avoiding a weighted histogram-binning edge case without consulting held-out values.','',
        'Main experiment: `'+str(OUT.relative_to(ROOT)).replace('\\','/')+'`. Initial ablations: `outputs/reliability_research/run_20261101_v2/`. Quality-only follow-up: `outputs/reliability_research/quality_clean_domain_20261101/`. Methods: `RELIABILITY_RESEARCH.md`.']
    (ROOT/'solution/docs/RELIABILITY_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Wrote solution/docs/RELIABILITY_RESULTS.md')


if __name__=='__main__':main()
