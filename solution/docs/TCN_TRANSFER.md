# Frozen TCN cross-provider transfer check — 3 November 2026

This checkpoint tests the current corrected-combined and final 50/50 TCN blend on Metrica's public Sample Games 1 and 2. Neither model was fitted on these fixtures. The old model's binary results were inspected before TCN development, so these are not untouched research holdouts. The new check freezes model weights, blend, decoder, corruption settings and reference rules before predictions. No competition inputs/references, training, model selection or threshold adjustments are part of it.

## References and what can be measured

Sources are the pinned [Metrica Sports sample repository](https://github.com/metrica-sports/sample-data/tree/e706dd506b360d69d9d123d5b8026e7294b13996) and its [event definitions](https://github.com/metrica-sports/sample-data/blob/e706dd506b360d69d9d123d5b8026e7294b13996/documentation/events-definitions.pdf). Credit Metrica Sports. Original files are SHA-256/Git-blob checked through the existing local manifest. No new data downloads are needed.

The publisher defines PASS as a same-team change of player possession. We interpret the interior of an eligible recorded pass as possession by the event's team. We exclude both endpoints, offside/referee-hit events, missing/self pass endpoints, mismatched halves, opposing possession actions inside the pass, H/A conflicts and conflicts with the existing binary reference. Team ownership is not extended into event gaps. These are conservative event-derived labels, not new manual complete possession annotations.

Dead references reuse the earlier accepted ball-out/shot-out/goal-to-explicit-restart intervals. The primary reference removes one second from either end of each eligible dead interval; unguarded results are also retained. All other frames are X in the reference and excluded from accuracy metrics. Predictions must remain H/A/D on active frames. Predicted D or the wrong team during a labelled pass counts as an ownership error.

The evaluation reports labelled-frame H/A/D accuracy, pass-team accuracy, dead precision/recall and conditional missing-ball/ghost/wrong-team subgroups. It does not calculate spellF1 or describe partial references as full-match accuracy or complete possession counts. Pass coverage is a selected subset that excludes many contested situations and turnovers.

## Fixed stress cases

Each game has ten input cases:

- Native tracking, including its natural missingness, seed 17.
- Ball gaps only, seeds 17/43/101: the already fixed moderate ball-dropout process; other input fields stay unchanged.
- Moderate combined noise, seeds 17/43/101: existing dropout, coordinate jitter, player loss, duplicates and within-team slot permutations.
- Contextual noise, seeds 17/43/101: the existing moderate process plus input-dependent dropout risk, ghost-ball tracks and wrong-team coordinate swaps.

The reused moderate generator has a nominal 0.407 ball-dropout duty fraction, lognormal gap median 10.72 s and sigma 0.95, player dropout 0.07, coordinate jitter 0.15 m, duplicate probability 0.01 and slot permutation period 30 s. Contextual settings remain in `solution/configs/nested_validation.json`: 45 s mean event spacing, 3 s median extra gaps, 0.6 s median ghost tracks offset by 5–20 m, and 2 s median wrong-team bursts. Realized rates are measured separately. These settings were reused, not fitted to the two samples. Artificial missingness overlaps native missingness, so its nominal rate is not the final observed missing fraction.

Corruption APIs receive tracking and random seeds only. Event references do not decide which frames to corrupt. Synthetic missing-ball accuracy specifically measures originally visible observations removed by the input-only generator. This creates labelled live-play occlusions without deriving reference status from ball visibility. It is still simulated noise, not a measurement of real challenge sensor errors. Ghosts may fill originally missing frames, so ghost exposure is tracked separately from visibility.

## Paired uncertainty

The same event-derived labelled intervals are sampled 1,000 times for both models within each game. Draws are shared across corruption seeds; seed deltas are averaged before interval endpoints are computed. Intervals preserve each labelled pass/dead stretch as a group. They are not complete possession spells, independently sampled matches, or full sequence re-evaluations. Only two games and selective partial references limit generalization claims. Per-game results are primary; no pooled percentage is used to hide a regression on one fixture.

## Reproduction and artifacts

From the project root:

```powershell
& '.\outputs\tcn_research\env\Scripts\python.exe' -u solution/tools/validate_tcn_transfer.py --output outputs/tcn_transfer/run_20261103
& '.\outputs\tcn_research\env\Scripts\python.exe' -u solution/tools/audit_tcn_transfer.py --output outputs/tcn_transfer/run_20261103
& '.\outputs\tcn_research\env\Scripts\python.exe' -m unittest discover -s solution/tests -q
```

The ignored output contains a frozen protocol, separate partial references, per-case prediction evidence, masks, realized noise, confusion matrices, paired draws, per-game reports and a final audit. The native current-model evidence must reproduce the previous hash-checked Metrica check exactly. Existing production and TCN bundles are read-only. No commit, push, publication or submission is performed.

## Completed results

The full joint blend is not a consistent cross-provider improvement. It improves ownership on artificially obscured pass interiors, but its dead-ball probabilities cause substantial stoppage regressions on the native samples. Both native labelled-frame accuracy intervals are below zero. This changes the interpretation of the earlier DFL gains: they do not justify a universal replacement claim.

| Native sample | Guarded reference coverage | Old partial accuracy | Joint partial accuracy | Old D recall | Joint D recall |
|---|---:|---:|---:|---:|---:|
| Game 1 | 36.18% | 89.63% | 84.20% | 87.01% | 76.36% |
| Game 2 | 42.60% | 90.35% | 87.68% | 88.79% | 83.02% |

Reference construction accepts 786 eligible passes / 56 stoppages in Game 1 and 951 passes / 59 stoppages in Game 2. Short stoppages may have no interior left after guarding. The native labelled pass frames all have observed ball positions, while virtually all labelled dead frames lack them. Thus native missing-ball ownership accuracy is undefined, not zero. Simulated input-only ball removal provides the following limited live-occlusion check.

| Game / stress | Old accuracy on newly missing pass frames | Joint-blend accuracy | Paired conditional 95% interval for change (percentage points) |
|---|---:|---:|---|
| 1 / ball_gaps | 82.63% | 88.31% | [+3.61, +7.79] |
| 1 / moderate | 79.88% | 83.93% | [+1.91, +6.27] |
| 1 / contextual | 79.37% | 82.75% | [+1.48, +5.34] |
| 2 / ball_gaps | 81.74% | 87.27% | [+3.76, +7.40] |
| 2 / moderate | 79.93% | 83.11% | [+1.70, +4.81] |
| 2 / contextual | 80.26% | 82.65% | [+0.94, +3.97] |

These conditional missing-pass gains are present on both samples in every stress condition, but do not compensate consistently for missed stoppages. Under moderate combined noise, overall partial accuracy is 88.79% to 88.89% on Game 1 and 89.32% to 89.30% on Game 2; both conditional intervals include zero. Under contextual noise, it is 87.83% to 87.41% and 88.46% to 87.79%, again with intervals crossing zero. In ghost-exposed labelled frames, Game 2 degrades from 78.05% to 70.44%; do not claim general false-detection improvement from this experiment.

Realized whole-input missing-ball fractions are 39.14% / 41.01% natively; 63.18% / 64.32% with ball gaps only; 64.22% / 63.37% with moderate noise; and 66.51% / 66.17% with contextual noise (Game 1 / Game 2, averaging seeds). Contextual ghost exposure is approximately 1.51% / 1.49% of frames. These samples already have substantial natural missingness.

## Diagnosis and exploratory ownership-only comparison

On native Game 1, the joint blend adds 2,766 dead-frame errors and 78 pass-team errors. On Game 2 it adds 1,713 dead-frame errors while removing 108 pass-team errors. The TCN dead head assigns less than 0.5 dead probability to 35.45% / 34.86% of guarded true-dead frames, versus 16.83% / 14.44% for the old dead head. Mixing these probabilities weakens stoppage evidence.

After observing that failure, a separate, explicitly exploratory diagnostic evaluates the existing DFL-prespecified ownership-only blend (50/50 conditional-home probabilities, original dead probabilities) and a dead-only blend. No weights, coefficients or decoder costs are fitted. The new diagnostic has its own frozen protocol and does not rewrite the primary transfer experiment. Its use of already inspected samples prevents treating it as an independent model-selection validation.

| Game / condition | Old partial accuracy | Full joint blend | Ownership-only (exploratory) | Ownership-only D recall |
|---|---:|---:|---:|---:|
| 1 / native | 89.63% | 84.20% | 89.51% | 87.07% |
| 1 / ball_gaps | 88.69% | 85.57% | 89.07% | 88.86% |
| 1 / moderate | 88.79% | 88.89% | 89.06% | 90.64% |
| 1 / contextual | 87.83% | 87.41% | 87.80% | 89.69% |
| 2 / native | 90.35% | 87.68% | 90.66% | 89.10% |
| 2 / ball_gaps | 89.15% | 88.52% | 89.52% | 90.81% |
| 2 / moderate | 89.32% | 89.30% | 89.47% | 91.72% |
| 2 / contextual | 88.46% | 87.79% | 88.48% | 90.61% |

Ownership-only largely avoids the native dead-recall regression: 87.07% / 89.10%, compared with the original 87.01% / 88.79%. It retains smaller improvements on synthetically missing pass frames: moderate 79.88% to 81.70% and 79.93% to 81.87%. It does not improve every subset or prove non-inferiority; native Game 1 partial accuracy is slightly lower than the old model.

The dead-only diagnostic drops native dead recall to 79.60% / 83.67% even while preserving old ownership probabilities. This implicates dead-emission fusion, with additional interaction through temporal decoding. Preserving dead probabilities does not force the decoded dead mask to be identical, because the global path also depends on ownership evidence.

The existing seven-DFL evaluation already measured ownership-only moderate spellF1 at 0.55523 versus old 0.51582 and joint 0.56117. Those earlier external results are supportive context, not a newly independent test.

Decision: retain all existing bundles; qualify the full joint blend as DFL-leading but transfer-limited. Prioritize freezing and testing ownership-only fusion on another external fixture before choosing the final submission. No deployed fusion, model file, submission CSV or threshold changed in this checkpoint. Real automatic-tag fusion, video recovery and complete independent possession references remain later work.

## Verification

All 136 tests pass. The primary transfer audit passes 88 hash checks, recalculates 480 subset score dictionaries and all stored paired draws, and reproduces two fresh full-pipeline moderate predictions exactly. Both native old-model replays match the previous check; every active prediction remains H/A/D. All model manifests and fitted files retain their recorded identities. The exploratory diagnostic passes 40 prediction hash checks and 120 score recalculations. These audits establish consistency, not hidden-match accuracy.

Exploratory reproduction:

```powershell
& '.\outputs\tcn_research\env\Scripts\python.exe' -u solution/tools/diagnose_transfer_fusion.py --source outputs/tcn_transfer/run_20261103 --output outputs/tcn_transfer/fusion_diagnostic_20261103
```
