# Learned possession-switch events

The subsequent runnable-pipeline checkpoint is complete in `INFERENCE.md`. It fits final all-seven external deployment heads and packages all three explicit variants; the seven-fold quality results below still refer to disjoint evaluation models, not those final fits.

Completed 1 November 2026 after the user's authorization to test direct switch-event learning. All seven outer models, 70 external cases, both fixed reward strengths and the independent audit are complete. Provider-label spellF1 improves, and the stronger reward raises moderate count passes from 2/7 to 4/7. Both variants regress on absorbed-under-3-second references, so the experiment establishes a provider-convention improvement rather than an unconditional new default or competition result. Existing heads/decoder artifacts remain intact.

## Separate the event task from ownership classification

The previous ownership classifier estimates P(home | alive). The new CPU gradient-boosted classifier estimates three mutually exclusive classes: no direct team switch, switch to home, and switch to away. It uses no true owner or true dead mask at inference. It supplements the existing evidence rather than replacing the original heads or their probability artifacts.

Supervised event targets mark a band of at most five native 25 Hz frames on either side of a direct H→A or A→H reference transition. Bands stay inside one true alive run and period; they cannot cross D, X or halftime. The nearest event wins overlapping bands, with the earlier event winning ties. Thus an event probability describes proximity to a handover, not a calibrated probability of a switch on that exact frame. True D and other graded non-event frames are negatives; X is excluded. Reference labels determine supervised targets and sampling only.

Training uses the existing clean/moderate/severe cases, with seeds 17,43,101 for noisy cases and clean once. Native 1 Hz rows are augmented with event-centre and ±5-frame samples where they remain inside that alive run. Class-stratified inverse sampling weights restore each match/condition's native full-frame no-event/to-home/to-away proportions. Matches and regimes receive equal mass, and noisy seeds share regime mass. This avoids silently interpreting oversampled event examples as their natural frequency. Calibration is not assumed perfect.

## Label-free evidence and cross-fitting

The model uses 114 channels: 74 existing physical/anchor features, four ownership/dead/confidence/distance-advantage features, and 36 past/future/change features at lags 5,12,25,50 frames. Physical features include team motion/formation summaries, observed/missing-ball context, ball speed and position, player-ball distance/motion, and geometric anchors. Additional history spans ±0.2, ±0.48, ±1 and ±2 seconds within halves. Existing geometric anchors can use more distant future observations, so the pipeline remains fully offline; the total lookahead is not limited to two seconds.

For outer holdout O and supervised training fixture T, both upstream probability heads were trained on the other five fixtures, excluding O and T. The new event model trains on six fixtures using those cross-fitted probabilities. At inference on O, it receives probabilities from the original six-match heads that exclude O. This prevents the new model from training on overly optimistic upstream predictions fitted to the same training fixture's labels. Training heads have five fixtures and inference heads six, which introduces a possible probability-distribution difference rather than label leakage. Every probability file, its exclusions and original upstream training membership are retained in the model/report lineage.

Tracking-only physical projections are cached separately, without opening reference arrays. Event feature extraction accepts physical values, names, upstream probabilities, periods and configuration; labels never enter that API. Feature shifts reset at halves and use NaN for unavailable context.

## Fixed model and integration

`configs/switch_events.json` freezes a three-class HistGradientBoostingClassifier with 150 iterations, learning rate 0.06, 15 leaves, minimum 40 samples per leaf, L2=1, seed 17 and no random-frame early stopping. Four processes use two numerical-library threads each. No GPU or new package is needed.

The existing H/A/D emission costs, published periods and previously nested-selected stop/restart penalties are unchanged. The H↔A costs become directional. For direction probability q above 0.1:

`directed cost = max(-4, original switch cost - strength × max(0, logit(q) - logit(0.1)))`

Strengths 2 and 4 are fixed ablations. If their same-frame A→H and H→A costs sum below 2, both are increased equally until the sum is 2. This prevents a negative same-frame round-trip incentive; it does not guarantee the absence of false switches across time. Negative individual penalties are allowed because a confident event may need to outweigh weak contradictory frame evidence. Exact Viterbi still finds a finite global optimum in the time-directed graph. There is no hard minimum possession duration or target count.

When strength is zero, costs and output must reproduce the original fixed decoder exactly. D-related transitions receive no event reward. Although dead evidence is unchanged, the final D mask can change through interactions with the globally decoded path; D precision/recall are reported, rather than assumed identical.

## Evaluation scope and limits

Seven outer models each train on the other six fixtures. This is a fixed training/integration ablation, not a new inner hyperparameter search: neither strength is selected using outer scores. Both are evaluated on 70 outer input cases, with the unseen contextual stress condition and alternate absorbed-under-3-second labels reserved for evaluation. All seven fixtures were previously opened, and earlier failure analysis motivated this experiment. Cross-fitting does not turn it into a new independent final test or erase historical adaptation.

Report spellF1, macroF1, D metrics, counts/durations, graded-frame X, recovery by true possession-duration bin, and direct-switch precision/recall/F1 with ±12-frame (0.48 s) one-to-one typed matching. False-switch counts must be considered with counts: adding incorrect transitions is not recovered possession. Frame-level event confusion matrices are also retained, but no-event accuracy alone would be misleading because events are rare.

Paired within-match intervals use 1,000 shared reference-possession draws with fixed matching/prediction counts and seed averaging, conditional on fitted models. Descriptive seven-match bootstrap intervals resample match differences. Neither refits models, removes overlapping-training dependence or measures generalization to a new annotation convention. All numerical results are external-data numbers; no competition fixture enters training or evaluation.

## Provider-label results

Means weight fixtures equally, averaging noisy seeds within each fixture. Both strengths were specified before fitting/evaluation; neither was selected using outer results.

| Condition | Original spellF1 | Reward 2 | Reward 4 | Count passes, original / reward 2 / reward 4 |
|---|---:|---:|---:|---:|
| Clean | 0.6014 | 0.6123 | 0.6205 | 3/7 / 5/7 / 6/7 |
| Moderate | 0.4916 | 0.5006 | 0.5046 | 2/7 / 2/7 / 4/7 |
| Severe | 0.4317 | 0.4368 | 0.4376 | 0/7 / 0/7 / 2/7 |
| Contextual | 0.4718 | 0.4798 | 0.4837 | 2/7 / 2/7 / 4/7 |

Count passes use the approximate ±20% heuristic on each fixture's mean count. Reward 2 improves provider spellF1 point estimates on every fixture under all four conditions. Reward 4 improves all fixtures under clean, moderate and contextual noise, and six under severe noise; severe J03WQQ decreases by 0.00035. It is not correct to claim that every per-fixture gain is statistically separated from zero.

For moderate provider labels, reward 2's mean improvement is +0.00905 with descriptive paired seven-match bootstrap interval [0.00608, 0.01201]. Reward 4 improves +0.01299 [0.00902, 0.01740]. Within-match conditional intervals exclude zero on 5/7 moderate fixtures for reward 2 and 3/7 for reward 4. Reward 4 contextual improvement is +0.01192 [0.00725, 0.01698]. These exploratory intervals do not correct for historical experimentation or constitute an independent final test.

| Moderate outer fixture | True spells | Original predicted | Reward 4 predicted | Original spellF1 | Reward 4 spellF1 |
|---|---:|---:|---:|---:|---:|
| J03WPY, Düsseldorf–Nürnberg | 421 | 304.0 | 359.0 | 0.5157 | 0.5263 |
| J03WOH, Düsseldorf–Regensburg | 402 | 305.0 | 339.7 | 0.5335 | 0.5488 |
| J03WOY, Düsseldorf–Rostock | 456 | 321.0 | 362.7 | 0.4914 | 0.5148 |
| J03WMX, Köln–Bayern | 511 | 315.3 | 349.0 | 0.4375 | 0.4546 |
| J03WQQ, Düsseldorf–St. Pauli | 401 | 333.7 | 365.0 | 0.5299 | 0.5348 |
| J03WR9, Düsseldorf–Kaiserslautern | 370 | 304.0 | 351.0 | 0.5095 | 0.5170 |
| J03WN1, Bochum–Leverkusen | 500 | 341.3 | 393.7 | 0.4237 | 0.4359 |

Mean moderate predicted count rises from 317.8 to 360.0 against 437.3 true; mean predicted duration falls from 9.90 to 8.74 s against 7.34 s true. Remaining count failures are Rostock, Bayern and Leverkusen. Rostock and Leverkusen are near the boundary but remain failures under the frozen ±20% definition; the tolerance was not changed to make them pass.

Dead precision/recall remain essentially unchanged because only team-switch costs receive rewards: moderate D precision is about 95.46% and recall 96.44% for both original and reward 4. They are not bit-identical masks: report-level values retain their small differences. All 70 cases for both strengths emit zero X on published playing-period frames and graded frames.

## Recovered short possessions and added false switches

| Moderate provider spell duration | Original matched at IoU 0.5 | Reward 4 matched | Reference spell evaluations |
|---|---:|---:|---:|
| Below 1 s | 41 | 105 | 1,152 |
| 1–3 s | 592 | 750 | 2,802 |
| 3–10 s | 1,693 | 1,805 | 3,312 |
| At least 10 s | 1,591 | 1,584 | 1,917 |

The table sums all 21 moderate fixture/seed cases, so each underlying reference possession appears three times. These are repeated evaluations, not 9,183 independent real possessions. One-to-three-second recovery increases from 21.1% to 26.8%, while some long possessions are lost. In clean data, one-to-three-second matches rise from 307/934 to 407/934 and sub-second matches from 37/384 to 79/384.

| Moderate direct H↔A transitions, ±12 frames | Original | Reward 2 | Reward 4 |
|---|---:|---:|---:|
| Mean precision | 0.2784 | 0.3094 | 0.3271 |
| Mean recall | 0.1836 | 0.2165 | 0.2562 |
| Mean F1 | 0.2204 | 0.2537 | 0.2861 |
| Mean predicted switches | 219.1 | 232.9 | 261.3 |
| Mean unmatched/spurious switches | 158.1 | 161.0 | 176.0 |

Precision and recall both improve, but the absolute number of spurious switches rises. Reward 4 adds about 42.2 predicted direct switches per fixture, including about 24.4 additional matches and 17.9 additional unmatched switches under this strict timing metric. This is useful provider-label progress, not precise recovery of every short possession. Frame argmax event confusion matrices and threshold-trigger coverage remain in each outer report; raw no-event accuracy is not offered as success evidence.

## Label-convention tradeoff and outcome

The absorbed-label references remove brief opposite-team interruptions, so rewarding those interruptions can hurt their score:

| Condition | Original alternate spellF1 | Reward 2 | Reward 4 |
|---|---:|---:|---:|
| Clean | 0.6497 | 0.6356 | 0.6052 |
| Moderate | 0.5509 | 0.5449 | 0.5250 |
| Severe | 0.4994 | 0.4938 | 0.4850 |
| Contextual | 0.5312 | 0.5275 | 0.5105 |

Moderate alternate count passes change from 6/7 to 5/7 (reward 2) or 1/7 (reward 4). Reward 4's alternate moderate spellF1 difference is −0.02585 [−0.03658, −0.01619], with no improved fixture. Reward 2's is −0.00597 [−0.00930, −0.00236]. These are descriptive match-bootstrap intervals, with the same limitations as the provider comparisons.

The experiment therefore succeeds at its provider-label objective more convincingly than the earlier physical-control boost or boundary-sampling change: both provider interval scores and count coverage improve with reward 4. It does not establish a universally better controlled-possession solver. The alternate set is a sensitivity convention, not authoritative challenge truth, so neither favorable provider results nor unfavorable alternate results alone determine challenge performance.

Retain the original reference and both fixed event variants explicitly. Do not silently select a new production default from outer results or describe this as winning/hidden-match performance. The next delivery task is reproducible end-to-end inference packaging with clear variant/provenance choices and these convention/count limitations documented. An independent controlled-possession reference would be needed to resolve the semantic tradeoff confidently. No final all-seven deployment event model is fitted in this checkpoint.

## Verification

All 90 tests pass, including event direction/overlap targets, dead/half barriers, native class-prior restoration, half-reset feature shifts, global optimality with negative rewards by exhaustive search, zero-reward reference equivalence and the directional round-trip guard. The independent audit verifies 651 artifact hashes, all 294 cross-fitted training probability traces, seven disjoint six-match event fits, seven exact held-out probability-reload probes, all 140 original comparator rows and exact repetition/format validation of 140 new CSVs. It also checks native zero-reward labels and path costs against the old decoder on all 70 cases.

Old feature/model/decoder sources, model artifacts and upstream probability arrays remain unchanged. No competition-data inference/evaluation, new package/download, commit, push or submission was performed. Generated models, caches, reports and external CSVs remain ignored by the single root `.gitignore`.

## Reproduction

From the project root in PowerShell:

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/train_switch_events.py --output outputs/switch_events/run_20261101
& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/audit_switch_events.py --output outputs/switch_events/run_20261101
```

The runner freezes configuration/source hashes before preparing new evidence or fitting. It reuses the completed nested experiment's immutable input/model/probability artifacts. Completed physical caches and outer models/reports can resume only when source hashes and declared training memberships match. The output directory contains `protocol.json`, 70 physical feature caches, seven `switch_model.joblib` evaluation fits with full cross-fitting lineage, 70 native event-probability arrays, 140 external prediction CSVs, seven outer reports and `validation_report.json`. The separate audit writes `integrity_audit.json` after checking lineage, exact held-out probability reloads, native zero-reward equivalence, original comparator reproduction and exact CSV repetition/format validity. Generated artifacts remain ignored under the single root policy.
