# Three-state temporal decoder

Completed 30 October 2026. Exact offline Viterbi decoding improves external possession segmentation over the frozen gap-aware system. Moderate counts now meet the approximate ±20% gate on both opened validation fixtures. The system still undercounts and misses boundaries; this is a completed decoder checkpoint, not a claim of competition readiness or ranking.

## Evidence and algorithm

The decoder consumes only native-frame P(dead), P(home | alive), and published period identifiers. It uses the previously selected full-feature/frame-weighted dead head and spell-weighted bidirectional ownership head unchanged. Reference labels enter evaluation and development cost selection only; the decoder API cannot accept them. No models were retrained, dependencies installed or competition inputs evaluated here.

Joint emission probabilities are H=(1-d)h, A=(1-d)(1-h), D=d. Costs are their negative logarithms, with h and d clipped to [0.000001, 0.999999]. These model outputs are scores, not a claim of perfectly calibrated probabilities. Dynamic programming finds the globally lowest sum of emission and transition costs within each half. Initial states have equal cost; halves reset independently. All direct transitions, including H↔A, are allowed. Backtracking places boundaries using the entire half's evidence, without a separate confirmation delay or first-touch refinement pass. Boundary positions still depend on the learned evidence and are not guaranteed first controlling touches.

The selected fixed penalties are 40 for H/A→D, 40 for D→H/A, and 8 for H↔A. Staying costs zero. Units are negative log-score penalties at 25 Hz, not seconds. There is no hard minimum event duration, duration distribution or possession-rate target. An event survives when accumulated evidence outweighs its entry/exit costs. This can suppress a weak short real event as well as a false event.

H/A/D cover every published playing-period frame; X appears only outside published periods. No reference mask decides output coverage. This is full output coverage, not proof that hidden possessions were recovered.

## Frozen development selection

The grid contains stop/restart pairs (8,8), (20,20), (40,40), (80,80), (20,40), (40,20), each with owner-switch cost 8 or 24, in two modes. Fixed costs give 12 candidates. Evidence-scaled costs give another 12: stop multiplies by 1.5-d, restart by 0.5+d, H→A by 0.5+h, and A→H by 1.5-h. These prespecified heuristics scale penalties between 0.5 and 1.5; no claim of superiority is assumed. A zero-cost joint frame-MAP candidate is also eligible.

All 25 candidates were scored on the existing 28 out-of-fold cases from four development matches. Both probability heads excluded the scored match from training. Selection averages seeds within each match/regime, then weights all 12 match/regime groups equally, maximizing spellF1 with macroF1 as tie-break. The winner is fixed_s40_r40_o8 (development spellF1 0.495068). The evidence-family winner is evidence_s20_r40_o8 (0.490561). Both were frozen before validation; the fixed winner remains selected even where the alternative does better on a validation case.

Excluding one fixture from cost selection retains the selected profile in three cases; excluding J03WOH selects fixed_s20_r40_o8. This is cost sensitivity, not nested validation: other matches' probability heads may have trained on the excluded fixture. Upstream architectures and ownership settings were also selected on development data. These development scores are selection scores, not unbiased generalization estimates. Neither opened validation fixture was used to tune this grid or select its winner. Repeated checkpoint use means these fixtures are no longer a fresh final test. All-seven nested outer-match validation remains necessary; J03WN1 remains absent and unopened here.

## External validation

J03WQQ is Düsseldorf–St. Pauli; J03WR9 is Düsseldorf–Kaiserslautern. Reference labels are provider BallStatus/BallPossession, as documented in EXTERNAL_DATA_SOURCES.md. Conditions and seeds are unchanged from validation.json: clean seed 17; moderate/severe seeds 17,43,101. Noisy results below average the three seed scores/counts/durations. All results in this document are external-data results.

| Fixture | Noise | Previous gap spellF1 | Decoder spellF1 | Predicted / true spells | Predicted / true mean seconds |
|---|---|---:|---:|---:|---:|
| J03WQQ | Clean | 0.5365 | 0.6078 | 344 / 401 | 9.79 / 8.41 |
| J03WQQ | Moderate | 0.3578 | 0.5163 | 337.3 / 401 | 10.02 / 8.41 |
| J03WQQ | Severe | 0.2286 | 0.4473 | 323.7 / 401 | 10.40 / 8.41 |
| J03WR9 | Clean | 0.5173 | 0.5950 | 313 / 370 | 10.85 / 9.22 |
| J03WR9 | Moderate | 0.3150 | 0.4969 | 307.3 / 370 | 10.95 / 9.22 |
| J03WR9 | Severe | 0.2118 | 0.4568 | 285.7 / 370 | 11.69 / 9.22 |

Moderate counts move from 611.3 to 337.3 on J03WQQ and 593.0 to 307.3 on J03WR9. Relative count errors become −15.9% / −16.9%, passing the approximate ±20% heuristic on seed averages. Five of six moderate cases pass individually; J03WQQ seed 17 has 318 versus 401 spells (−20.7%) and narrowly fails. Under severe noise J03WQQ averages −19.3%, while J03WR9 averages −22.8% and fails. Longer predicted durations and lower counts indicate remaining merging/omitted possessions, despite better interval overlap. Count agreement alone does not establish correct boundaries.

| Fixture | Noise | Decoder D precision | Decoder D recall | False alive islands wholly within reference D: previous → decoder |
|---|---|---:|---:|---:|
| J03WQQ | Clean | 99.27% | 99.58% | 5 → 0 |
| J03WQQ | Moderate | 95.29% | 95.27% | 177.3 → 0.3 |
| J03WQQ | Severe | 92.27% | 92.65% | 373.3 → 2.7 |
| J03WR9 | Clean | 99.08% | 99.70% | 3 → 0 |
| J03WR9 | Moderate | 94.47% | 96.32% | 112.3 → 0.7 |
| J03WR9 | Severe | 91.81% | 94.95% | 311.7 → 0.7 |

These islands are maximal predicted alive blocks inside a half that lie entirely within reference dead frames, not every erroneous alive frame. The decoder now changes the dead mask. Moderate D recall improves over the previous 93.92% / 94.78%, with slightly lower precision than the previous 95.45% / 94.65%. Zero X is maintained on all graded and published-period frames.

The evidence-dependent family scores 0.5103 / 0.4853 under moderate noise, versus the selected 0.5163 / 0.4969. Joint frame-MAP scores only 0.1196 / 0.1056, with about 2,718 / 2,678 predicted spells. Temporal penalties, rather than simply emitting every posterior argmax, are doing substantial work. This joint frame-MAP comparator also differs from thresholding d at 0.5: D wins whenever d exceeds both joint team probabilities.

Shared reference-possession bootstrap draws (1,000) give moderate decoder-minus-previous-gap spellF1 differences +0.1585 [0.1405, 0.1757] and +0.1819 [0.1588, 0.2016]. Against the old rule solver, differences are +0.2668 [0.2331, 0.3030] and +0.2323 [0.1925, 0.2685]. All six clean/moderate/severe improvements over previous_gap have positive conditional intervals. These are conditional intervals with fixed matching and prediction counts, averaging seed scores per draw; they do not refit models, rematch resampled timelines or measure uncertainty across unseen matches.

## Verification and local artifacts

All 73 tests pass. Decoder tests include exhaustive global-optimum comparison for both modes, half separation, weak-hole suppression, preservation of a strongly supported brief event, direct owner changes, input validation, repeatability and non-mutation. The brief-event test uses lower penalties to establish absence of a hard duration rule; it does not imply a single-frame event survives the selected 40/40 profile. Dependency checking passes.

All 56 previous-system comparator rows reproduce their frozen interval/frame scores, spell counts and noisy-input fingerprints exactly. All 14 validation probability files retain their hashes. The development runner also checks its 28 probability files before/after scoring. Upstream source hashes and fold-model hashes are verified; labels remain separate from decoder evidence. Selected predictions repeat exactly and all 14 external CSVs pass the supplied format validator using each external fixture's row count. Models, probabilities, CSVs and reports remain ignored under the single root policy. No commit, push or submission was made.

Decoder-only timing on a cached 146k-frame external fixture was approximately 0.15 s per mode in an isolated run; it excludes feature extraction, learned inference, I/O and evaluation, and is not an end-to-end performance claim. The selection run used four CPU workers and no GPU.

Local outputs:

- `outputs/decoder/development_20261030T162423164708Z/development_report.json`: 700 candidate scores, frozen selection, sensitivity and probability hashes.
- `outputs/decoder/validation_20261030T162644279645Z/validation_report.json`: 112 score rows, confusion matrices, boundary metrics, 30 paired comparisons, CSV hashes.
- The validation directory also contains 14 `*_decoded.csv` files and `integrity_audit.json`.

## Run this checkpoint

From the project root in PowerShell, with the already installed environment and cached prior ownership artifacts:

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/run_decoder.py --stage development --ownership-selection outputs/gap_ownership/development_20261030T153115114216Z/development_report.json

& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/run_decoder.py --stage validation --ownership-selection outputs/gap_ownership/development_20261030T153115114216Z/development_report.json --selection outputs/decoder/development_20261030T162423164708Z/development_report.json --validation-probabilities outputs/gap_ownership/validation_20261030T154041424173Z/validation_report.json
```

Each command creates a timestamped output directory. The second command above uses the recorded frozen selection; to validate a reproduced selection, replace its path with the first command's new report path. This runner decodes cached external probabilities. It is not yet an end-to-end learned challenge inference command or a graphical application. Previous runnable systems remain available. Next checkpoint: broader nested validation, label-convention sensitivity and stronger corruption scenarios, before final inference packaging.
