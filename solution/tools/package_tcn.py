"""Final all-seven refit of the fixed, externally evaluated TCN blend."""
import argparse
from importlib.metadata import version
from pathlib import Path
import sys
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'solution'), str(ROOT/'solution/tools')]
import train_tcn as t
import tcn_predict as runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); source = args.experiment.resolve(); out = args.output.resolve()
    cfg, _, _, _ = t.r.setup(); config = t.r.n.read(t.CONFIG)
    report, audit, protocol = [t.r.n.read(source/name) for name in ('report.json', 'audit.json', 'protocol.json')]
    if report['status'] != 'complete' or audit['status'] != 'passed' or protocol['config'] != config:
        raise ValueError('Require the complete audited fixed experiment')
    for name, digest in protocol['source_sha256'].items():
        if t.file_digest(ROOT/name) != digest: raise ValueError('Frozen training source changed')
    for row in report['summary']:
        if row['model'] == config['primary_challenger'] and row['descriptive_match_ci'][0] <= 0:
            raise ValueError('Primary blend must improve across both conventions/all conditions before final refit')
    base = runtime.load_bundle(ROOT/'models/possession-v1')
    research = runtime.current.load_research(ROOT/'models/research-specialists-clean', base)
    out.mkdir(parents=True, exist_ok=True)
    sources = (*runtime.SOURCES, 'solution/tools/package_tcn.py', 'solution/tools/train_tcn.py', 'solution/configs/tcn.json')
    provenance = {'config': config, 'training_matches': cfg['matches'],
        'source_sha256': {name: t.file_digest(ROOT/name) for name in sources},
        'experiment_report_sha256': t.file_digest(source/'report.json'),
        'experiment_protocol_sha256': t.file_digest(source/'protocol.json'),
        'torch_version': version('torch')}
    if (out/'refit_protocol.json').exists() and t.r.n.read(out/'refit_protocol.json') != provenance:
        raise ValueError('Final refit provenance changed')
    t.r.save(out/'refit_protocol.json', provenance)
    if (out/'manifest.json').exists():
        runtime.load_tcn(out, base, research); print('Existing final bundle verified'); return
    t.configure_torch_threads(config['threads']); torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.deterministic = True
    with threadpool_limits(limits=config['threads']):
        cases, indexes = t.load_training(cfg['matches'], source, cfg)
        normalizer = t.fit_normalizer(case['x'][::5] for case in cases.values())
        normalizer_lists = {k: v.tolist() for k, v in normalizer.items()}
        torch.manual_seed(config['seed']); model = t.PossessionTCN(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['minimum_learning_rate'])
        checkpoint = out/'checkpoint.pt'; first = 0; history = []
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if saved['provenance'] != provenance or saved['normalizer'] != normalizer_lists: raise ValueError('Final checkpoint drift')
            model.load_state_dict(saved['state_dict']); optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler']); torch.set_rng_state(saved['torch_rng'])
            first, history = saved['epoch'], saved['history']
        for epoch in range(first, config['epochs']):
            tick = time.perf_counter(); model.train(); totals = np.zeros(4)
            rng = np.random.default_rng(config['seed']+epoch)
            for _ in range(config['steps_per_epoch']):
                x, y, w, b = t.batch(cases, cfg['matches'], rng, normalizer, config, cfg, model.radius)
                optimizer.zero_grad(set_to_none=True)
                states, boundaries = model(x); lo, hi = model.radius, model.radius+config['core_steps']
                loss, parts = t.losses(states[:, :, lo:hi], boundaries[:, :, lo:hi], y, w, b, config)
                if not torch.isfinite(loss): raise ValueError('Nonfinite refit loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
                optimizer.step(); totals += [float(loss.detach()), *[float(parts[k].detach()) for k in ('dead', 'home', 'boundary')]]
            scheduler.step()
            history.append({'epoch': epoch+1, 'losses': (totals/config['steps_per_epoch']).tolist(), 'seconds': time.perf_counter()-tick})
            t.atomic_torch_save({'provenance': provenance, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'torch_rng': torch.get_rng_state(),
                'epoch': epoch+1, 'normalizer': normalizer_lists, 'history': history}, checkpoint)
            print(f'All-seven refit: epoch {epoch+1}/{config["epochs"]}', flush=True)
        payload = {'state_dict': model.state_dict(), 'normalizer': normalizer_lists,
            'config': config, 'training_matches': cfg['matches'], 'epochs': config['epochs'],
            'training_index_sha256': {m: t.file_digest(source/'prepared'/m/'index.json') for m in cfg['matches']}}
        t.atomic_torch_save(payload, out/'model.pt')
        manifest = {'kind': 'tcn_joint_blend', 'status': 'complete', 'config': config,
            'training_matches': cfg['matches'], 'torch_version': version('torch'),
            'model_sha256': t.file_digest(out/'model.pt'), 'base_manifest_sha256': t.file_digest(base['folder']/'manifest.json'),
            'research_manifest_sha256': t.file_digest(research['folder']/'manifest.json'),
            'runtime_sha256': {name: t.file_digest(ROOT/name) for name in runtime.SOURCES},
            'refit_protocol_sha256': t.file_digest(out/'refit_protocol.json'), 'training_history': history,
            'evaluation': 'Accuracy evidence comes only from separate six-match outer fits in the recorded experiment; this all-seven refit is not scored on its training fixtures.',
            'candidate_status': 'Leading research candidate; final competition submission is not selected or uploaded automatically.'}
        t.r.save(out/'manifest.json', manifest)
    runtime.load_tcn(out, base, research); print('PASS: one-CSV TCN blend bundle: '+str(out), flush=True)


if __name__ == '__main__': main()
