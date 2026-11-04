# Frozen ownership-only transfer check — 4 November 2026

This checkpoint compares the corrected-combined model with a fixed ownership-only TCN blend on Metrica Sample Game 3. The candidate configuration and three model identities were recorded before downloading this previously unscored external fixture. The adapter was implemented and tested before inference. Neither model was trained on any Metrica sample, and no Game 3 result determines a fitted coefficient, threshold, decoder cost or label convention.

## Fixed candidate and data

The candidate averages the original and TCN conditional-home probabilities 50/50, retaining the original dead probabilities and learned switch evidence. Both models use the existing 40/40/8 decoder costs and event reward 4. All fitted weights remain unchanged. Preserving dead probabilities does not force the decoded dead mask to remain identical: the global H/A/D path also depends on ownership evidence.

Credit [Metrica Sports public sample data](https://github.com/metrica-sports/sample-data/tree/e706dd506b360d69d9d123d5b8026e7294b13996/data/Sample_Game_3), pinned revision `e706dd506b360d69d9d123d5b8026e7294b13996`. The anonymized fixture's real clubs are not identified. Its EPTS tracking, XML metadata and JSON events are verified against pinned Git blobs and SHA-256 digests. No challenge match data is accessed in this checkpoint.

The input contains 143,761 frames at 25 Hz on a 105 × 68 m pitch. Metadata specifies two halves, frames 1–69,661 and 69,662–143,761. All frame rows exist. Eleven format ranges change player columns at substitutions; the adapter resolves these using player/channel/team metadata into 35 stable slots, with absent players represented by NaN. It also respects coordinate-axis order. Normalized positions are converted to metres; out-of-bounds observations are retained. The model receives only the same six tracking fields, with no identities, events or reference labels.

All 3,620 event start/end timestamps equal their frame IDs divided by 25, so no offset is fitted. Native ball missingness is 32.22% of all frames. This is supplied optical tracking with natural missingness, not noise-free ground truth.

## Partial reference and fixed stress protocol

References reuse the preceding transfer experiment's eligible same-team PASS interiors and unambiguous ball-out/shot-out/goal to explicit-restart dead intervals. The primary dead labels exclude one second at each end. Uncertain intervals remain unknown reference frames; they are excluded from scoring, not counted as model successes. The resulting guarded reference covers 67,553 frames (46.99%): 20,985 H, 19,890 A and 26,678 D. The unguarded reference contains 30,222 D frames.

JSON introduces CARRY events absent from the older CSV samples. They supply exclusions only: live blockers at both endpoints prevent assuming a dead interval through a carry; a carry also excludes overlapping dead or opposite-team pass labels. They do not create H/A ownership truth or fill event gaps. These conservative schema rules were fixed before predictions; the final overlap exclusion removes zero additional frames after the endpoint blockers.

The prespecified primary endpoint is the paired change in pass-team accuracy on originally visible ball observations removed by moderate input-only corruption. This tests simulated live-ball gaps without deriving ownership truth from ball availability. Native labelled-frame accuracy and dead recall, other gap cases, contextual noise and ghost-exposed subsets are preservation checks. There is no prespecified numerical non-inferiority margin.

Ten cases reuse the existing noise parameters: native seed 17, and ball gaps / moderate / contextual seeds 17, 43 and 101. Moderate nominal ball dropout is 40.7%, with lognormal gap median 10.72 s and sigma 0.95; player dropout is 7%, coordinate jitter 0.15 m, duplicates 1% and within-team permutations every 30 s. Contextual stress adds the previously fixed input-dependent gaps, ghost-ball detections and wrong-team bursts. Realized rates are recorded separately because simulated dropout overlaps naturally missing observations. See `TCN_TRANSFER.md` and the frozen noise JSON files for the full definitions.

Paired bootstrap draws resample event-derived labelled intervals 1,000 times, sharing draws across seeds and averaging seed deltas before taking percentiles. They are conditional on this fixture, partial reference and corruption scheme. They are neither full possession spells nor independently sampled matches. No full-match accuracy, spellF1 or complete possession counts can be established from these sparse references.

## Completed results

The frozen ownership-only candidate improves the primary endpoint on this fresh external fixture. Moderate accuracy on synthetically missing pass frames rises **84.13% → 87.00%**, a **+2.87 percentage-point** change with paired conditional 95% interval **[+1.40, +4.27]**. This supports the intended gap improvement without refitting on Game 3.

| Condition | Old partial H/A/D accuracy | Owner-only partial accuracy | Old pass-team accuracy | Owner-only pass-team accuracy | Old dead recall | Owner-only dead recall |
|---|---:|---:|---:|---:|---:|---:|
| Native | 87.65% | 87.99% | 91.55% | 91.93% | 81.66% | 81.97% |
| Ball gaps | 87.05% | 88.27% | 88.91% | 90.57% | 84.20% | 84.75% |
| Moderate | 87.96% | 88.86% | 88.39% | 89.69% | 87.30% | 87.57% |
| Contextual | 87.73% | 88.36% | 87.55% | 88.50% | 88.01% | 88.14% |

Native values use one input; stress values average three fixed seeds. All percentages are conditional on the guarded partial reference. Predicted dead precision on that reference is 99.81% → 99.83% native and 97.64% → 97.75% moderate; these are not precision estimates over unlabelled frames.

| Condition | Old newly missing pass accuracy | Owner-only accuracy | Paired 95% interval for change, percentage points |
|---|---:|---:|---|
| Ball gaps | 84.53% | 88.25% | [+2.11, +5.28] |
| Moderate, primary | 84.13% | 87.00% | [+1.40, +4.27] |
| Contextual | 83.51% | 85.94% | [+1.08, +3.65] |

The overall partial-accuracy change is +0.35 points native (interval [−0.54, +1.23]), +1.22 with ball gaps ([+0.45, +2.03]), +0.90 moderate ([+0.34, +1.56]), and +0.63 contextual ([+0.08, +1.23]). Native superiority or formal non-inferiority is not established. These intervals are conditional diagnostics, with no multiple-comparison adjustment or population-level match sampling.

The ghost-exposed labelled subset worsens **82.49% → 81.93%**, averaging 935 labelled frames per seed. Its dead recall falls 81.72% → 80.68%, despite the unchanged dead emissions. Ghost-exposed pass-team accuracy increases slightly (84.62% → 84.87%). This remains a weakness, and the experiment does not establish general false-ball robustness.

Reference construction accepts 1,110 eligible passes and 71 out-to-restart intervals; 11 passes are excluded. Native labelled pass frames have no naturally missing ball observations, so native missing-pass accuracy is undefined. Simulated dropout supplies the live-gap test; it cannot substitute for naturally occluded live-play truth. Final whole-input missing-ball fractions are 32.22% native, 57.12% ball gaps, 59.51% moderate and 63.07% contextual. Contextual ghost and wrong-team exposure average 1.45% and 4.48% of all frames.

Decision: ownership-only is the better-supported next candidate for packaging, combining prior DFL evidence with a prespecified fresh-fixture gap gain and no observed native stoppage regression here. This single partial-reference match does not settle complete controlled-possession accuracy or the final submission choice. Preserve the joint blend and original bundles. No runtime default, model weight, decoder setting or AOM submission file changes in this validation checkpoint. Complete independent possession references and improved false-ball/stoppage evidence remain unresolved.

## Verification

All 140 tests pass in the optional TCN environment, including four new adapter tests for reordered player/axis channels, sparse frames, unmodified out-of-bounds observations, malformed inputs, event clocks and carry exclusions. The integrity audit passes 48 hash checks, 240 subset score recalculations, 30 paired-draw recalculations and aggregate reconstruction. It independently verifies 33 raw rows across all eleven metadata formats and reconstructs every corruption/mask from raw tracking. Two fresh native/moderate pipeline replays reproduce all stored probabilities and predictions exactly, assembling ownership fusion directly in the audit. There are zero active prediction unknowns; all three model payloads and manifests remain unchanged. These checks establish integrity and repeatability, not accuracy on the hidden match.

## Reproduction

From the project root, using the existing optional CPU PyTorch environment:

```powershell
& '.\outputs\tcn_research\env\Scripts\python.exe' -u solution/tools/validate_owner_game3.py --output outputs/owner_transfer_game3/run_20261104
& '.\outputs\tcn_research\env\Scripts\python.exe' -u solution/tools/audit_owner_game3.py --output outputs/owner_transfer_game3/run_20261104
& '.\outputs\tcn_research\env\Scripts\python.exe' -m unittest discover -s solution/tests -q
```

The original pre-download choice is stored at `outputs/owner_transfer_game3/candidate_freeze.json`, SHA-256 `eb2b7b684cc5d1777d025a80e8961d6940c52847b628ee9bb294447b37d680ac`. The downloader requires this prior freeze and records its hash in the source manifest. It does not create a retrospective freeze. A new checkout needs the original local freeze and pinned model bundles to reproduce this historical run; do not manufacture a new claim of an unseen holdout after inspecting the report.

Ignored run artifacts include the protocol/source/model hashes, conversion audit, separate partial references, per-case probabilities/predictions/masks, confusion matrices, realized noise, bootstrap draws, summary and integrity audit. Existing production entry points, bundles and AOM submission files are preserved. The user handles commits and pushes.
