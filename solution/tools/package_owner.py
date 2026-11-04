"""Bind the validated owner-only fusion to existing weights; no fitting or scoring."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'solution'))
import owner_predict as runtime


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', type=Path, default=ROOT/'outputs/owner_transfer_game3/run_20261104')
    parser.add_argument('--freeze', type=Path, default=ROOT/'outputs/owner_transfer_game3/candidate_freeze.json')
    parser.add_argument('--output', type=Path, default=ROOT/'models/tcn-owner-v1')
    args = parser.parse_args(); source = args.validation.resolve(); freeze = read(args.freeze)
    report, audit, protocol = [read(source/name) for name in ('report.json', 'audit.json', 'protocol.json')]
    check = lambda path, digest: runtime.t.file_digest(path) == digest
    if report['status'] != 'complete' or audit['status'] != 'passed' or not audit['model_weights_unchanged']:
        raise ValueError('Require the complete audited owner-only validation')
    if not check(source/'protocol.json', report['protocol_sha256']) or not check(args.freeze, protocol['candidate_freeze_sha256']):
        raise ValueError('Validation/freeze identity differs')
    if not check(ROOT/'solution/tools/audit_owner_game3.py', audit['audit_source_sha256']): raise ValueError('Audit source differs')
    for name, digest in protocol['source_sha256'].items():
        if not check(ROOT/name, digest): raise ValueError('Frozen validation source differs: '+name)
    for stem, digest in report['case_report_sha256'].items():
        if not check(source/(stem+'.json'), digest): raise ValueError('Validated case report differs')
        case = read(source/(stem+'.json'))
        if not check(source/(stem+'_evidence.npz'), case['evidence_sha256']): raise ValueError('Validated prediction differs')
    for filename, key in [('references.npz', 'reference_sha256'), ('reference_audit.json', 'reference_audit_sha256'), ('alignment.json', 'alignment_sha256')]:
        if not check(source/filename, report[key]): raise ValueError('Validation artifact differs')
    if freeze['config'] != protocol['config'] or freeze['config']['candidate'] != runtime.KIND or freeze['config']['blend'] != .5:
        raise ValueError('Candidate decision differs from evaluated fusion')
    if report['training_or_tuning_performed'] or report['complete_possession_scoring_performed'] or audit['challenge_data_used']:
        raise ValueError('Unexpected validation scope')
    base = runtime.t.load_bundle(ROOT/'models/possession-v1')
    research = runtime.t.current.load_research(ROOT/'models/research-specialists-clean', base)
    tcn = runtime.t.load_tcn(ROOT/'models/tcn-blend-v1', base, research)
    for bundle in (base, research, tcn):
        name = bundle['folder'].name
        if not check(bundle['folder']/'manifest.json', freeze['model_manifest_sha256'][name]) or bundle['manifest']['model_sha256'] != freeze['model_payload_sha256'][name]:
            raise ValueError('Model differs from the pre-download freeze')
    components = runtime.component_identities(base, research, tcn)
    manifest = dict(schema_version=1, kind=runtime.KIND, status='complete', settings=runtime.SETTINGS,
        components=components, model_sha256=runtime.model_identity(components),
        model_identity_kind='sha256_of_component_payload_hashes',
        training_matches=base['manifest']['training_matches'], runtime_sha256=runtime.runtime_inventory(base, research, tcn),
        validation_provenance={name: runtime.t.file_digest(source/name) for name in ('report.json', 'audit.json', 'protocol.json')},
        candidate_freeze_sha256=runtime.t.file_digest(args.freeze), package_source_sha256=runtime.t.file_digest(Path(__file__)),
        fitted_in_this_step=False, candidate_status='Runnable research candidate; not a final submission choice.',
        limits='Partial external references support gap improvement; ghost subsets remain a weakness. Offline future context. No hidden-match accuracy claim.')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True); path = out/'manifest.json'
    if path.exists():
        if read(path) != manifest: raise ValueError('Existing owner bundle differs; choose a new path')
    else:
        with path.open('x', encoding='utf-8') as handle: handle.write(json.dumps(manifest, indent=2)+'\n')
    runtime.load_owner(out, base, research, tcn)
    print('PASS: validated ownership-only bundle: '+str(out))


if __name__ == '__main__': main()
