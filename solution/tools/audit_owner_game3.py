"""Audit Game 3 conversion, frozen artifacts, paired scores and fresh runtime replays."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import validate_owner_game3 as g

t = g.t


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve(); result = t.read(out/'report.json'); protocol = t.read(out/'protocol.json')
    counts = dict(hash_checks=0, confusion_recalculations=0, paired_draw_recalculations=0,
                  independent_mapping_rows=0, exact_pipeline_replays=0, active_unknowns=0)
    def check(path, digest):
        if t.runtime.file_digest(path) != digest: raise ValueError('Artifact drift: '+str(path))
        counts['hash_checks'] += 1
    if result['status'] != 'complete' or result['training_or_tuning_performed'] or result['complete_possession_scoring_performed']:
        raise ValueError('Unexpected experiment status')
    check(out/'protocol.json', result['protocol_sha256']); check(g.FREEZE, protocol['candidate_freeze_sha256'])
    freeze = t.read(g.FREEZE); check(g.CONFIG, freeze['config_sha256'])
    if freeze['game3_local_data_present_at_freeze'] or freeze['config'] != protocol['config']: raise ValueError('Freeze differs')
    for path, digest in protocol['source_sha256'].items(): check(g.ROOT/path, digest)
    for model, digest in protocol['model_manifests'].items():
        check(g.ROOT/'models'/model/'manifest.json', digest)
        check(g.ROOT/'models'/model/('model.pt' if model == 'tcn-blend-v1' else 'models.joblib'), protocol['model_payloads'][model])
    check(g.SOURCE/'source_manifest.json', protocol['download_manifest_sha256'])
    source = t.read(g.SOURCE/'source_manifest.json')
    if source['candidate_freeze_sha256'] != protocol['candidate_freeze_sha256']: raise ValueError('Download chronology differs')
    for record in source['files']: check(g.SOURCE/record['path'], record['sha256'])
    for key, filename in [('reference_sha256', 'references.npz'), ('reference_audit_sha256', 'reference_audit.json'),
                          ('alignment_sha256', 'alignment.json')]: check(out/filename, result[key])
    meta = g.SOURCE/'Sample_Game_3_metadata.xml'; track = g.SOURCE/'Sample_Game_3_tracking.txt'
    inputs, alignment = g.read_epts_tracking(meta, track)
    if json.loads(json.dumps(alignment)) != t.read(out/'alignment.json'): raise ValueError('Alignment audit differs')
    # Independent direct row checks at the start, middle and end of every format.
    # These do not use the adapter's compiled channel mapping.
    tree = ET.parse(meta).getroot(); specs = tree.findall('./DataFormatSpecifications/DataFormatSpecification')
    ids = [p.attrib['id'] for p in tree.findall('.//Players/Player')]
    channels = {c.attrib['id']: c.attrib for c in tree.findall('.//PlayerChannels/PlayerChannel')}
    selected = {}
    for spec in specs:
        a, b = int(spec.attrib['startFrame']), int(spec.attrib['endFrame'])
        for frame in (a, (a+b)//2, b): selected[frame] = spec
    with track.open() as handle:
        for line in handle:
            frame_text, players_text, ball_text = line.strip().split(':'); frame = int(frame_text)
            if frame not in selected: continue
            spec = selected[frame]; expected = np.full_like(inputs['players_xy'][frame-1], np.nan)
            values = np.fromstring(players_text.replace(';', ','), sep=',', dtype=np.float32)
            refs = spec.findall('./SplitRegister/SplitRegister/PlayerChannelRef')
            if len(refs) != len(values): raise ValueError('Independent row width differs')
            for value, ref in zip(values, refs):
                channel = channels[ref.attrib['playerChannelId']]; axis = ['x', 'y'].index(channel['channelId'])
                expected[ids.index(channel['playerId']), axis] = float(value)*inputs['pitch_m'][axis]
            np.testing.assert_array_equal(expected, inputs['players_xy'][frame-1])
            value = np.fromstring(ball_text, sep=',', dtype=np.float32)
            np.testing.assert_array_equal((value*inputs['pitch_m']).astype(np.float32), inputs['ball_xy'][frame-1])
            counts['independent_mapping_rows'] += 1
    raw, core, audit = g.game3_reference(meta, g.SOURCE/'Sample_Game_3_events.json', inputs['period'], protocol['config']['guard_frames'])
    if audit != t.read(out/'reference_audit.json'): raise ValueError('Reconstructed reference audit differs')
    with np.load(out/'references.npz', allow_pickle=False) as z:
        for key, value in dict(raw=raw, guarded=core, period=inputs['period']).items(): np.testing.assert_array_equal(z[key], value)
    keep = raw != 'D'; np.testing.assert_array_equal(core[keep], raw[keep])
    if ((core == 'D') & (raw != 'D')).any(): raise ValueError('Guard introduced labels')
    intervals = t.labelled_intervals(core, inputs['period']); config = protocol['config']
    picks = np.random.RandomState(config['bootstrap_seed']).randint(0, len(intervals), (config['bootstrap_draws'], len(intervals)))
    torch.set_num_threads(config['threads']); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); torch.backends.mkldnn.deterministic = True
    base = t.runtime.load_bundle(g.ROOT/'models/possession-v1')
    research = t.runtime.current.load_research(g.ROOT/'models/research-specialists-clean', base)
    tcn = t.runtime.load_tcn(g.ROOT/'models/tcn-blend-v1', base, research)
    if tcn['manifest']['training_matches'] != t.read(g.ROOT/'solution/configs/nested_validation.json')['matches']:
        raise ValueError('Unexpected training cohort')
    reports = []
    with threadpool_limits(limits=config['threads']):
        for condition, seed in t.cases(config):
            stem = f'{condition}_{seed}'; check(out/(stem+'.json'), result['case_report_sha256'][stem]); case = t.read(out/(stem+'.json'))
            if case['protocol_sha256'] != result['protocol_sha256'] or case['reference_sha256'] != result['reference_sha256']:
                raise ValueError('Case provenance differs')
            check(out/(stem+'_evidence.npz'), case['evidence_sha256'])
            with np.load(out/(stem+'_evidence.npz'), allow_pickle=False) as z: evidence = {k: z[k] for k in z.files}
            np.testing.assert_array_equal(evidence['period'], inputs['period'])
            np.testing.assert_array_equal(evidence['blend_dead'], evidence['current_dead'])
            for ref_name, ref in [('event_interiors', raw), ('guarded_dead', core)]:
                for model, label, prob in g.MODEL_KEYS:
                    t.runtime.validate_prediction(evidence[label], inputs['period'])
                    counts['active_unknowns'] += int((evidence[label][inputs['period'] > 0] == 'X').sum())
                    for subset in g.SUBSETS:
                        mask = np.ones(len(core), bool) if subset == 'all' else evidence[subset]
                        score = t.score_subset(ref, evidence[label], evidence[prob], mask)
                        if score != case['scores'][ref_name][model][subset]: raise ValueError('Score mismatch')
                        counts['confusion_recalculations'] += 1
            if g.paired(core, evidence, picks, intervals) != case['paired_interval_draws']: raise ValueError('Paired draws differ')
            counts['paired_draw_recalculations'] += 3
            # Rebuild all corruptions from raw input, checking masks and fingerprints.
            noisy, masks = t.noisy_inputs(inputs, condition, seed, protocol)
            if t.runtime.array_digest(inputs) != case['native_input_fingerprint'] or t.runtime.array_digest(noisy) != case['input_fingerprint']:
                raise ValueError('Input replay differs')
            for key, value in masks.items(): np.testing.assert_array_equal(value, evidence[key])
            if t.summarize_inputs(noisy) != case['noise']: raise ValueError('Noise summary differs')
            if condition in ('native', 'moderate') and seed == 17:
                old = t.runtime.current.predict(base, research, noisy, 'combined'); settings = base['manifest']['settings']
                network = t.runtime.tcn_probabilities(tcn, noisy, settings, base['models']['owner_names'])
                # Assemble the locked ownership-only blend directly, without the runner's fusion helper.
                h = .5*old['home_probability']+.5*network[:, 0]; d = old['dead_probability']; deploy = settings['deployment']
                replay = t.runtime.current.decode_events(h, d, inputs['period'], old['event_probability'],
                    deploy['decoder'], 4, settings['switch'], deploy['epsilon'])
                for key, value in dict(current_labels=old['labels'], current_home=old['home_probability'],
                    current_dead=d, current_events=old['event_probability'], tcn_home=network[:, 0], tcn_dead=network[:, 1],
                    blend_home=h, blend_dead=d, blend_labels=replay['labels']).items():
                    np.testing.assert_array_equal(value, evidence[key])
                counts['exact_pipeline_replays'] += 1
            reports.append(case); print(f'{stem}: artifacts, scores and corruption verified', flush=True)
    if g.aggregate(reports, config) != result['summary']: raise ValueError('Aggregate differs')
    report = dict(status='passed', **counts, model_weights_unchanged=True, challenge_data_used=False,
        audit_source_sha256=t.runtime.file_digest(Path(__file__)), scope='Integrity and repeatability, not independent hidden-match accuracy')
    t.save(out/'audit.json', report); print(report, flush=True)


if __name__ == '__main__': main()
