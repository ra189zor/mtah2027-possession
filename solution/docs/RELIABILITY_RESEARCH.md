# Reliability, missing-ball ownership and boundaries

This experiment implements the four requested research directions as explicit ablations. The existing AOM deployment package is preserved. Configuration is frozen in `solution/configs/reliability_research.json` before new model fitting; raw external inputs reproduce the existing benchmark fingerprints.

## Architecture

1. **Tracking reliability:** a gradient-boosted classifier estimates whether a visible ball observation belongs to a synthetic false track. Features describe player proximity, past/future ball speeds, geometric inconsistency, boundary distance and gap context. Training injects separately seeded moving-offset or frozen-offset tracks into degraded external inputs. Synthetic masks are supervision only; they are never input features. The model's probabilities describe these simulated errors, not calibrated certainty about all real-world tracking failures.
2. **Missing-ball ownership:** a second gradient-boosted model learns home/away ownership on alive frames where the ball is missing. It receives the original physical/motion evidence, cross-fitted upstream home/dead probabilities, the reliability estimate, both gap endpoints, time into the gap and team geometry relative to those endpoints. Possession-balanced training weights limit domination by long possessions. Its prediction is blended 50/50 with the original ownership probability only inside missing-ball gaps.
3. **Boundary evidence:** a three-class model learns no direct handover / handover to home / handover to away using the same richer evidence. It retains the original native-class-prior-preserving event sampler and blends 50/50 with the original event probabilities. A reliability-only ablation softens ownership confidence and switch evidence at suspicious observations. The combined candidate uses all three changes. Original dead probabilities and fixed Viterbi transition settings are preserved in this experiment.
4. **Broader validation:** seven held-out fixtures, three noisy seeds, clean/moderate/severe/contextual inputs and provider references plus absorbed-under-1/3/5-second conventions. Six candidates are compared: original baseline, event4, quality only, gap only, boundary only and combined. This expands the single-match cleaning pilot; it does not add previously unseen real matches. The alternate conventions are evaluation-only sensitivity analyses.

Twenty-eight quality models cover seven single-fixture exclusions and 21 pairs. In outer fold O, specialist training data from fixture T receives base probabilities and quality predictions trained without both O and T. The gap/boundary heads train on the remaining six fixtures and are evaluated on O using heads excluding O. No competition match inputs or labels enter this experiment. Repeated architecture work on these same seven fixtures remains adaptive research, even with correct fold exclusions.

## Measurements and promotion

Primary metric is provider-label spellF1. Report alternate-convention spellF1, count errors, direct-switch precision/recall, dead-ball metrics, accuracy on missing-ball alive frames and quality-detector average precision/false alarms. Paired descriptive match bootstrap compares each candidate with event4; conditional possession bootstrap compares the combined model on moderate/contextual data. The intervals do not remove historical adaptation or dependence from overlapping training folds.

A more complex model is not automatically better. Do not promote a candidate based on training fit, possession count alone, one fixture, or whichever reference convention looks most favorable. Any selected research candidate needs an explicit declaration of its tradeoffs and full inference verification before updating the submission package.

Benchmark decoding retains the pre-existing inner-selected profile for each outer fold (stop/restart costs vary across folds). The separate all-seven experimental deployment retains the earlier global 40/40/8 profile. No decoder costs or blend strengths are selected from the new outer outcomes. See `RESEARCH_INFERENCE.md` for this scope distinction.

## Reproduction

```powershell
& '.\.venv\Scripts\python.exe' solution/tools/research_reliability.py --output outputs/reliability_research/run_20261101_v2
& '.\.venv\Scripts\python.exe' solution/tools/quality_domain_check.py
& '.\.venv\Scripts\python.exe' solution/tools/research_reliability_clean.py
& '.\.venv\Scripts\python.exe' solution/tools/audit_reliability_research.py --output outputs/reliability_research/run_20261101_clean
```

The experiment resumes only with identical protocol/source hashes. Artifacts include per-fixture input-only features, synthetic training samples, quality models, gap/boundary specialists, cross-fitting lineage, external prediction arrays, per-fold metrics and aggregate results. All remain under the ignored outputs directory. Original runtime source files and model payloads are unchanged. Results are documented in `RELIABILITY_RESULTS.md`; experimental inference is documented in `RESEARCH_INFERENCE.md`.

## Correcting the quality-training domain

The first detector exhibited excessive clean false alarms despite good contextual ghost detection. A separately frozen follow-up adds clean tracking with injected ghosts using seed 829, with equal total weight per clean/moderate/severe regime. Its quality-only evaluation precedes the fully corrected joint experiment. The joint refit retains identical architectures, blends, raw evaluation cases and references, but retrains all pair-excluded quality models and all seven gap/boundary specialists using the corrected quality predictions. The six-match quality models already fitted in the follow-up are reused with checked hashes and memberships. This avoids swapping a differently calibrated quality head into specialists trained on the earlier head's outputs.

The first interrupted attempt exposed a weighted histogram-binning edge case with entirely missing columns. Each specialist now selects columns with at least one finite training value; held-out values never determine that selection. No dependency patch or version change was required. The interrupted run remains separate from completed experiments.
