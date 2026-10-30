# Short-possession recovery

Completed 30 October 2026 after the seven-match count failures. Two targeted experiments are complete and audited. A count-balanced control boost raises moderate provider-label count passes from 2/7 to 3/7, with slightly lower mean spellF1. Boundary-aware training does not improve the original provider-label objective. Neither is promoted as an unconditional replacement; the original decoder/heads remain the preferred provider-score reference. These are adaptive experiments on opened external data, not an independent new test.

## Diagnosis

The diagnostic covers clean and moderate seed 17 on all seven external fixtures, using the already frozen six-match outer heads and nested decoder profiles. It is a failure analysis on opened data, not new validation or candidate selection. Inputs and oracle references are separate.

| Evidence | Clean mean spellF1 | Moderate seed-17 mean spellF1 |
|---|---:|---:|
| Current system | 0.6014 | 0.4813 |
| True dead mask, predicted ownership | 0.6090 | 0.5276 |
| True ownership evidence, predicted dead evidence | 0.9920 | 0.9297 |
| Both true labels | 1.0000 | 1.0000 |

The first oracle decodes learned ownership independently inside true alive blocks and inserts the exact reference D mask. The second supplies near-certain true H/A probabilities on true alive frames, retains learned ownership on true D frames, and decodes against the unchanged predicted dead probabilities. It is an evidence intervention, not a literal guarantee that the final team labels stay true when dead predictions are wrong. Both-true bypasses prediction and checks the scorer. Oracles never form deployable evidence.

| Provider spell duration | Number across seven fixtures | Clean matched at IoU 0.5 | Moderate seed 17 matched |
|---|---:|---:|---:|
| Below 1 s | 384 | 37 | 12 |
| 1–3 s | 934 | 307 | 181 |
| 3–10 s | 1,104 | 727 | 551 |
| At least 10 s | 639 | 585 | 521 |

For 1–3 s spells, the raw ownership posterior favors the true team on an average 65.5% of frames in clean data and 61.3% in moderate seed 17, averaging spell fractions within each fixture then fixtures. Many short events lack decisive evidence before decoding. Lowering penalties alone may add false switches; better local evidence is also worth testing. These diagnostics do not imply every brief provider spell is controlled possession under the challenge convention.

## Prespecified targeted changes

`configs/short_possession.json` freezes 12 candidates: owner-switch penalties 1, 2, 4 or 8, crossed with physical-control residual strength 0, 0.5 or 1. Learned heads, native head probabilities and each outer fold's existing nested stop/restart costs stay unchanged. The prior strength-0/switch-8 reference is included exactly. No model refitting, new download or dependency is required.

The new evidence uses existing anonymous tracking features only. It requires the ball within 2.5 m of the favored team, at least 1 m farther from the opponent, ball speed at most 15 m/s, and favored-player/ball speed difference at most 8 m/s. All necessary features must be observed and valid. At least three consecutive native frames support the same team, with confirmation backdated to their first frame. Runs cannot bridge missing evidence or halftime. The residual adds signed strength × logit(0.95) to learned P(home | alive) log odds. With strength zero or absent control evidence the original probabilities remain exactly unchanged.

This is close-control evidence, not proof of a touch or possession: aerial height is unavailable, opponents can be nearest without controlling, and wrong-team/ghost detections can create false evidence. Fast-ball filtering is only a physical heuristic. The contextual outer stress test remains essential. P(dead) is unchanged, but the final D mask can change because joint emission scores and temporal decoding interact with the revised team posterior.

## Selection and limits

The existing 21 five-match inner head pairs are reused. For each excluded outer fixture, every inner prediction comes from heads trained on five other fixtures, excluding both outer and inner holdouts. The original nested stop/restart penalties were already selected without that outer fixture. Each candidate is scored on six inner held-out matches under clean/moderate/severe noise, with seeds averaged and equal match/regime weighting.

Two selection rules are frozen before new outer evaluation:

- Score-first maximizes inner provider-label spellF1, with macroF1 tie-break.
- Count-balanced considers candidates within 0.005 mean inner spellF1 of the original decoder, maximizes the number of moderate inner fixture averages within ±20% count error, then minimizes mean absolute relative count error and maximizes spellF1. This is an explicit multi-objective tradeoff, not a guarantee of outer non-inferiority or a statistical non-inferiority test.

Both choices are reported on all 70 outer cases and both label conventions. Neither contextual stress nor absorbed labels enter selection. No outer score chooses between them. All seven fixtures were previously opened and their failure analysis motivated the experiment: nested selection here does not make this an independent new test or undo historical architecture adaptation. In particular, a provider-label count gain must not conceal oversegmentation under the alternate convention. Previous systems and artifacts remain intact.

## Postprocessing results

All 3,528 inner candidate scores completed before outer evaluation. Every score-first fold selected strength 0 / switch cost 8, exactly the original decoder. Count-balanced selection chose strength 0.5 / switch cost 8 for J03WPY, J03WOH, J03WOY, J03WQQ and J03WN1; the other two folds retained the original. No lower-switch-cost candidate won either selector.

| Condition | Original provider spellF1 | Count-balanced spellF1 | Provider count passes, original → balanced | Mean predicted spells, original → balanced |
|---|---:|---:|---:|---:|
| Clean | 0.6014 | 0.5976 | 3/7 → 5/7 | 340.6 → 364.4 |
| Moderate | 0.4916 | 0.4892 | 2/7 → 3/7 | 317.8 → 339.0 |
| Severe | 0.4317 | 0.4307 | 0/7 → 1/7 | 297.4 → 312.0 |
| Contextual | 0.4718 | 0.4679 | 2/7 → 3/7 | 317.5 → 336.7 |

Count passes use the ±20% heuristic on each fixture's seed-averaged count. Provider true counts average 437.3. Moderate provider spellF1 difference is −0.00237 with descriptive paired match-bootstrap interval [−0.00664, +0.00258]. The interval includes zero, so this does not establish a statistically reliable spellF1 loss or gain. Only one of seven matches improves moderate provider spellF1. Recovering counts has not established recovery of the correct intervals.

| Fixture | True provider spells | Moderate original | Moderate count-balanced |
|---|---:|---:|---:|
| J03WPY | 421 | 304.0 | 328.0 |
| J03WOH | 402 | 305.0 | 328.3 |
| J03WOY | 456 | 321.0 | 353.0 |
| J03WMX | 511 | 315.3 | 315.3 |
| J03WQQ | 401 | 333.7 | 355.3 |
| J03WR9 | 370 | 304.0 | 304.0 |
| J03WN1 | 500 | 341.3 | 389.0 |

Against absorbed-under-3-second references, count-balanced mean moderate spellF1 falls from 0.5509 to 0.5335 and count passes fall from 6/7 to 4/7; contextual spellF1 falls from 0.5312 to 0.5154. Adding brief possessions can therefore move the provider-count diagnostic in the desired direction while hurting the alternative convention. The count-balanced option remains an explicitly labeled experimental tradeoff. It is not adopted as the new best solver.

## Fixed boundary-sampling training experiment

The postprocessor results motivated a second protocol, frozen in `configs/boundary_sampling.json` before these new fits. The original 1 Hz samples represent about 94.41% of provider H/A spells across the 49 training cases, omitting 5.59% entirely. The new sampler retains the 1 Hz rows and adds the first/last frame, three-frame offsets inside both ends when available, and the midpoint of every H/A run. Rows are unique; every spell receives exactly equal total sampled weight within a match/condition. Every provider H/A spell is represented, including single-frame spells. Supervised labels select training rows/weights only, never feature values or inference observations.

The new training caches contain 256,461 rows across seven matches and the seven original corruption cases each. For each outer holdout, the unchanged 120-iteration/15-leaf ownership HGB is refitted on the other six fixtures (roughly 216k–223k rows), excluding the evaluated match. Dead probabilities, feature definitions and original nested decoder profiles stay frozen. Contextual stress and alternate labels remain evaluation-only. Two fixed inference options are evaluated: new head alone and an arithmetic 50/50 blend with the old head. Neither blend strength nor decoder costs are selected on the new outer results. No new inner search is claimed for this fixed training ablation.

| Condition | Original provider spellF1 | Half blend | New boundary head | Provider count passes: original / half / new |
|---|---:|---:|---:|---:|
| Clean | 0.6014 | 0.6016 | 0.5935 | 3 / 2 / 2 |
| Moderate | 0.4916 | 0.4903 | 0.4801 | 2 / 0 / 0 |
| Severe | 0.4317 | 0.4303 | 0.4234 | 0 / 0 / 0 |
| Contextual | 0.4718 | 0.4706 | 0.4576 | 2 / 1 / 0 |

The new head's moderate provider spellF1 change is −0.01149 [−0.01684, −0.00526] under descriptive match bootstrap, with only one improved match. Mean moderate predicted count falls from 317.8 to 293.4 (half blend: 303.2). This training change therefore fails the intended provider-count/score objective and is not promoted.

The result reverses partly under absorbed labels: half-blend mean spellF1 is 0.6595 clean / 0.5574 moderate / 0.5076 severe / 0.5383 contextual, versus original 0.6497 / 0.5509 / 0.4994 / 0.5312. Severe alternate count passes rise from 6/7 to 7/7. This is sensitivity evidence, not permission to switch reference conventions after observing results. It reinforces that more provider spells and better controlled-possession intervals need not be the same objective.

## Outcome and verification

The ±20% count rule is our diagnostic heuristic; the supplied challenge README identifies spellF1 as its ranking metric. These experiments do not establish a safe provider-score improvement over the existing solver. The count-balanced candidate improves count coverage modestly but has a convention-dependent tradeoff. The training sampler fixes missing training-spell coverage without fixing the prediction deficit. Keep the original as the score-first reference, retain all experimental candidates/reports, and do not manufacture extra transitions to meet a desired count.

The remaining issue needs stronger evidence or an event-level ownership objective, not merely less smoothing or more samples of ambiguous frames. A learned possession-switch/boundary head is a reasonable next research hypothesis, but it has not been implemented or shown to improve scores here. Low-confidence hidden events remain unidentifiable in some long gaps. End-to-end packaging remains separate; no claim of submission readiness is added.

All 83 tests pass. The postprocessor audit verifies 602 artifact hashes, all 140 original comparator rows and exact repetition/format validation of 140 new CSVs. The boundary audit verifies 301 hashes, seven disjoint six-match owner fits, seven exact held-out probability-reload probes, all 140 original comparator rows and exact repetition/format validation of its 140 new CSVs. All new predictions have zero playing-period X. Original sources, model artifacts and native probability files are unchanged; all outputs remain ignored. No competition-data inference/evaluation, new package/download, commit or push was performed.

## Reproduction and artifacts

From the project root in PowerShell:

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/diagnose_short_spells.py
& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/improve_short_possessions.py --output outputs/short_possession/followup_20261030
& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/audit_short_possessions.py --output outputs/short_possession/followup_20261030
& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/train_boundary_owner.py --output outputs/short_possession/boundary_20261030
& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/audit_boundary_owner.py --output outputs/short_possession/boundary_20261030
```

The diagnostic is saved to `outputs/short_possession/diagnosis_20261030.json`. The follow-up directory freezes source/config hashes in `protocol.json`, caches 70 tracking-only control vectors, saves 21 inner candidate reports and `selection.json`, then writes seven per-match evaluations, 140 selected external CSVs and `validation_report.json`. Reference/learned probabilities are not accepted as control-feature inputs. Input feature caches and probability artifacts are checked against their prior hashes, and original inner/outer decoder scores must reproduce exactly. CSVs use each external fixture's native row count and the supplied format validator. Outputs remain ignored under the single root policy. No competition input or reference is used.

The separate `boundary_20261030` directory contains its frozen `protocol.json`, seven reference-aware training sample caches, seven held-out `owner.joblib` fits and reports, 70 new-head probability arrays, 140 prediction CSVs, `validation_report.json` and `integrity_audit.json`. These are external evaluation fits, not a final deployment model or an ensemble adopted for inference.
