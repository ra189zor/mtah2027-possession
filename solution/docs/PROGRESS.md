# Execution progress

## Step 1 — complete, 21 October 2026

- Reused the user's existing project-root `.venv` (Python 3.11.9).
- Installed NumPy 2.4.3 and pandas 3.0.1, with python-dateutil 2.9.0.post0, six 1.17.0 and tzdata 2026.5.
- Saved exact runtime dependencies in `requirements-lock.txt`.
- `pip check`: no broken requirements.
- Official smooth50 CSV validator: OK, 199319 rows, frames 0 through 199318.
- Official public evaluation reproduced all four supplied scores, with 1000 bootstrap draws:

| Baseline | macroF1 | spellF1 |
|---|---:|---:|
| raw | 0.5972 | 0.0378 |
| smooth25 | 0.6200 | 0.2868 |
| smooth50 | 0.6207 | 0.3256 |
| smooth125 | 0.5812 | 0.4242 |

The execution reports remain outside this code repository, in the project-root `research/step1_evaluation/` directory. No baseline predictions were regenerated, no match assets were extracted and no solver was implemented in this step.

Step 1 initially used a local Git repository in `solution/`. Step 2 supersedes this layout: the root is the intended repository boundary, with one `.gitignore`. The old checkpoint metadata is preserved under ignored `.local-git/solution.git`. The user handles commits and pushes.

## Step 2 — complete, 21 October 2026

- Consolidated ignore policy into one project-root `.gitignore`; removed the solution and upstream nested ignore files.
- Prepared all nine canonical inputs: seven extracted assets and two video hard links. Originals retained.
- Verified both tracking streams record-by-record, metadata assignments and all homography rows.
- Verified 638 tactical tags and kickoff-window timing in both halves.
- Decoded key tactical and broadcast frames locally with OpenCV; tactical reference count matches exactly.
- Confirmed broadcast decoder-reported count is one smaller than tracking; broadcast fusion deferred without padding or alignment guesses.
- Verified rerun behavior, original tracking hashes and unchanged upstream implementation (except the explicitly removed ignore file).
- Updated dependency lock and checked installed package consistency.

Details and limits are in `DATA_VERIFICATION.md`. No baseline regeneration, algorithm training, commit or push was performed in Step 2.

## Step 3 — complete, 28 October 2026

- Rebuilt all four unchanged official baselines twice from tactical tracking into ignored output directories.
- Every generated frame index and label matches its supplied reference: zero differences across 199319 rows per file.
- All submission validators passed; repeated-run SHA-256 hashes match exactly.
- Official public scores reproduced with 1000 bootstrap draws; no parameter tuning or private evaluation.
- Recorded generation wall times of 2.643/2.718 s and worker peak memory of 43.69/43.55 MiB.
- Confirmed original source, reference files and raw inputs were unchanged.
- Added a reusable reproduction runner and documented Windows/reference newline differences.

Details: `BASELINE_REPRODUCTION.md`. No dependencies were added, and no commits or pushes were made.

## Step 4 — initial benchmark complete, 28 October 2026

- Froze a disjoint whole-match split across seven public Sportec/DFL fixtures before scoring.
- Downloaded and checksum-verified J03WPY (development), then J03WQQ (validation), after the first-match adapter passed.
- Streamed both fixtures at verified 25 Hz into anonymous input arrays and separate H/A/D/X reference targets.
- Built deterministic clean/moderate/severe tracking-noise scenarios without target access.
- Evaluated four official baseline windows across 56 match/condition/seed combinations, with 1000 bootstrap draws each.
- Repeated the development run with identical metrics and input-corruption fingerprints.
- Passed 11 tests covering leakage guards, split membership, temporal boundaries, sparse IDs, corruption and geometry.
- Left the sealed J03WN1 fixture untouched and kept all raw data/outputs ignored.

See `EXTERNAL_VALIDATION.md` for results and limits. Two fixtures are not broad cross-match evidence; additional development fixtures remain to be added before extensive tuning. No human labels on the supplied match, new dependencies, commits or pushes were introduced.

## Step 5 — complete, 30 October 2026

- Implemented 40 geometry/motion/missingness channels plus 12 optional automatic-tag channels, with explicit validity masks.
- Added label-free challenge and external input readers; external reference arrays are never opened by feature extraction.
- Checked all 199319 challenge frames, both clean external fixtures and development moderate/severe seed-17 stress inputs.
- Repeated extraction matched exactly; within-team slot reversal preserved masks and numerical values within the declared tolerance on all five caches. Inputs were unchanged.
- Passed all 24 tests, including synthetic geometry, motion, missingness, half boundaries, ambiguous identity and tag semantics.
- Kept caches ignored and the sealed fixture untouched. No training, parameter fitting, dependencies, commits or pushes.

Details and limitations: `FEATURE_EXTRACTION.md`. These are feature-correctness checks, not evidence of improved possession scores.

## Step 6 — complete, 30 October 2026

- Implemented deterministic owner confirmation/persistence with separate stoppage/restart state and optional weak automatic-tag evidence.
- Fixed three profiles before scoring; selected `responsive` on development only, with equal regime weighting.
- Evaluated that frozen winner once on J03WQQ. SpellF1 exceeds the ±50 baseline for clean/moderate/severe inputs, while noisy macroF1 and coverage remain weaknesses.
- Recomputed all four baselines; 56 baseline rows/fingerprints match the prior benchmark. Saved 84 total score rows, with 1000 bootstrap draws each.
- Passed all 39 tests and exact prediction-repeat/input-immutability checks.
- Generated a full 199319-row challenge CSV; the official validator passed. No challenge reference scoring or parameter tuning.
- Sealed fixture untouched; outputs ignored. No dependencies, commits, pushes or submissions.

See `FALLBACK_SOLVER.md` for results, coverage tradeoffs and failure taxonomy. Two fixtures and tracking-only external tests do not establish challenge performance or tag accuracy.

## Oracle diagnosis — complete, 30 October 2026

- Ran three principal external-only oracles plus two abstention/component sensitivity checks on clean and three moderate-noise replicas of J03WQQ.
- Both-true spellF1 and macroF1 equal 1.0 in every condition; all frozen solver/baseline results reproduce.
- Perfect dead evidence helps, but owner identity/timing and abstention recovery also impose a large interval-score limit.
- Added paired conditional possession-bootstrap differences, with scope and limitations documented.
- Passed all 47 tests; preserved frozen sources, configurations and source arrays. No new data, fitting, decoder, dependencies, commits or pushes.

Details: `ORACLE_DIAGNOSIS.md`.

## Learned dead-ball evidence — complete, 30 October 2026

- Expanded the original split to four prepared development and two validation fixtures; sealed J03WN1 remains untouched in this checkpoint.
- Added 275 label-free multi-second movement/formation/ball/missingness features and a 228-column ball-independent player ablation.
- Fixed model settings and three weighting/feature variants before fitting; used 12 development match-fold fits and 252 threshold/condition scores for selection.
- Selected full features/frame weighting at threshold 0.50. Trained three final candidates on four development fixtures, then evaluated them once on both validation fixtures (70 rows), without retuning.
- Learned moderate D recall is 93.9% / 94.8% on J03WQQ / J03WR9, with 95.4% / 94.6% precision. Ball-missing and player-only diagnostics confirm useful player evidence.
- Retained the original ownership rules for the integration diagnostic. Its noisy J03WR9 spellF1 trails the old rule solver despite better D detection; X and count gates remain unmet.
- All 55 tests pass; model serialization probes and 14 frozen baseline/solver comparisons match exactly. Pinned CPU dependencies; models/caches remain ignored by the single root ignore file.
- No competition-data inference/evaluation, new decoder, commit, push or submission.

Methods, results, calibration, paired intervals and failure analysis: `DEAD_BALL_MODEL.md`.

## Gap-aware ownership — complete, 30 October 2026

- Added 23 geometric/anchor context channels to the 275 existing inputs and built 42 label-free bidirectional feature cases.
- Compared bridge-only, frame-weighted and spell-weighted ownership across four development folds, three confirmation settings and seven corruption cases per fixture (252 rows).
- Refit the selected dead architecture on each fold's other three matches; all 28 out-of-fold dead scores reproduce. No final all-four-match dead predictions leaked into development selection.
- Selected spell-weighted ownership with 0.8 s confirmation on development, then evaluated fixed families on both opened validation fixtures (84 rows).
- Moderate spellF1 improves over the previous learned-dead overlay from 0.2740 to 0.3578 on J03WQQ and 0.2590 to 0.3150 on J03WR9; conditional paired improvement intervals exclude zero.
- Zero X inside published periods; frozen validation dead probabilities and mask are unchanged. Moderate counts still exceed reference by 52.5% / 60.3%, with many false-alive islands inside true stoppages.
- All 65 tests pass; 42 comparator score rows reproduce exactly. Retained fold/final models and out-of-fold probabilities for later decoding.
- No new decoder, dependencies, downloads, competition-data inference/evaluation, commit or push.

Details: `GAP_OWNERSHIP.md`.

## Three-state temporal decoding — complete, 30 October 2026

- Added exact offline H/A/D Viterbi with independent half resets and joint dead/conditional-team emissions; no hard minimum duration or possession-rate target.
- Scored 25 prespecified profiles on 28 cached development out-of-fold cases. Selected fixed stop/restart costs 40/40 and owner-switch cost 8, then froze selection before validation. No learned model or previous source was changed.
- Moderate external spellF1 improves from 0.3578 to 0.5163 on J03WQQ and 0.3150 to 0.4969 on J03WR9. Conditional paired improvement intervals exclude zero.
- Moderate counts fall from 611.3 / 593.0 to 337.3 / 307.3 against 401 / 370 truth, within the approximate ±20% gate on both fixture averages. Five of six moderate seeds pass individually; J03WQQ seed 17 narrowly fails at −20.7%. Some possessions still merge; severe J03WR9 averages −22.8% count error and fails.
- Zero X throughout published periods. False alive islands wholly inside true dead stretches are almost eliminated; D precision/recall tradeoffs and durations are reported.
- All 73 tests and dependency checking pass; 56 frozen comparator rows reproduce, 14 validation probability files remain unchanged, selected predictions repeat and 14 external CSVs validate.
- No new data, dependencies, competition-data inference/evaluation, commit, push or submission. Sealed fixture remains absent.

Details and reproduction: `TEMPORAL_DECODER.md`.

## Seven-match nested validation — complete, 30 October 2026

- Explicitly replaced the historical sealed-fixture role with seven outer folds in a new protocol; opened Bochum–Leverkusen at the existing pinned revision. Old split/guards and model sources remain unchanged.
- Completed 21 five-match inner head pairs, 7,350 decoder candidate scores and seven six-match outer fits. Decoder selection excludes the outer fixture; historically selected head architectures remain fixed and their adaptation limitation is documented.
- Evaluated 70 external input cases, four algorithms and two reference conventions (560 rows). Added input-only contextual dropout, ghost balls and wrong-team bursts as an unseen evaluation stress mixture.
- Mean provider spellF1 is 0.6014 clean / 0.4916 moderate / 0.4317 severe / 0.4718 contextual. Decoder beats gap composition on all seven fixtures under each condition; noisy paired conditional intervals exclude zero on all seven.
- Nested costs give negligible provider-label gain over the prior fixed decoder. Provider count gates pass only 3/7 clean, 2/7 moderate, 0/7 severe and 2/7 contextual fixture averages. Alternate labels change those gates materially; this does not resolve the convention mismatch.
- All 78 tests pass; 616 independent artifact hash checks, 28 serialized training-membership checks, seven exact probability-reload probes and 70 external CSV validations pass. No playing-period X.
- No competition-data inference/evaluation, new dependencies, commit, push or submission. No final all-seven deployment model was fitted.

Details, full limitations and commands: `NESTED_VALIDATION.md`.

## Short-possession recovery investigation — complete, 30 October 2026

- Diagnosed all seven fixtures on clean/moderate seed 17: true dead masks provide a small clean score gain, while true owner evidence nearly resolves clean errors. Short provider possessions remain weak before decoding.
- Froze 12 physical-control/switch-cost candidates and selected them from the existing disjoint inner predictions (3,528 scores), before 70-case outer evaluation under both reference conventions.
- Every score-first fold retains the original. Count-balanced control boosts improve moderate provider count passes from 2/7 to 3/7, with mean spellF1 0.4916 → 0.4892 and worse alternate-label segmentation. Retained as an explicit experimental tradeoff, not a best-solver replacement.
- Tested fixed reference-aware boundary sampling, covering all training spells versus 94.41% previously, with seven new six-match ownership fits and unchanged features/dead evidence/decoder costs. Provider count/score performance regresses; the new head is not promoted. Half blending helps absorbed-label scores, emphasizing convention sensitivity.
- All 83 tests pass. Postprocessor and training audits verify 602 / 301 hashes, exact comparator reproduction and exact repetition/validation of 140 / 140 external CSVs. Original sources, model artifacts and probabilities stay unchanged.
- No competition-data inference/evaluation, new package/download, commit or push. No demonstrated overall provider-score improvement is claimed.

Full results and reproduction: `SHORT_POSSESSIONS.md`.

## Learned switch-event model — complete, 1 November 2026

- Added a three-class no-switch/to-home/to-away model with 114 label-free physical and probability-context features. Event targets/sampling remain separate from feature extraction.
- Trained seven event models on six fixtures each. All 294 upstream training probability cases come from five-match heads excluding both the outer fixture and that training fixture; contextual noise and alternate labels remain evaluation-only.
- Added an exact event-aware Viterbi decoder with directed switch rewards, a finite negative-cost floor and a same-frame round-trip guard. Original emissions, stop/restart costs, heads, sources and probabilities are unchanged.
- Evaluated two prespecified reward strengths over 70 external cases and both label conventions (420 rows), without outer-score strength selection. Reward 4 improves moderate provider spellF1 0.4916 → 0.5046 and count passes 2/7 → 4/7. Contextual provider spellF1 improves 0.4718 → 0.4837. Some false switches and three moderate count failures remain.
- Stronger rewards regress alternate-label moderate spellF1 0.5509 → 0.5250 and count passes 6/7 → 1/7. This is a provider-convention improvement, not an unconditional production default or competition score.
- All 90 tests pass. Independent audit verifies 651 hashes, 294 cross-fitting traces, seven exact held-out probability probes, 140 unchanged comparator rows, 140 repeated/validated CSVs and 70 native zero-reward equivalence cases.
- No new package/download, competition-data inference/evaluation, final all-seven deployment model, commit, push or submission.

Details and reproduction: `SWITCH_EVENTS.md`.

## End-to-end inference packaging — complete, 1 November 2026

- Added `solution/predict.py` and a tracking-only inference module, with no research-runner or reference-loader imports. Requires an explicit baseline/event2/event4 variant and writes a validated CSV plus provenance/timing sidecar.
- Fitted final dead/owner heads on all seven external fixtures (280,720 / 154,818 sampled rows). Fitted the final switch head on 327,846 sampled rows using 49 six-match OOF upstream cases, each excluding the training fixture. No competition/contextual/alternate labels enter fitting.
- Kept the original fixed 40/40/8 deployment profile and all existing feature/model settings. Embedded configuration, exact source/package hashes, input/model schemas, data attribution and training membership in `models/possession-v1/manifest.json`.
- Inference needs only the bundle's manifest/model payload and matching repository, not research caches, targets, training checkpoints or evaluation reports. Source/weight/settings checks reject incompatible or altered bundles; existing outputs are protected.
- Exported separate six-match verification models excluding J03WN1. Nine raw-input runs across clean/moderate/contextual noise and all three variants reproduce frozen held-out CSVs byte for byte. Four final-bundle behavioral runs and a separate minimal two-file portability run pass with identical repeated predictions.
- All 14 subprocesses run from an unrelated directory under reference/research access guards. All 95 tests pass. The provided format validator accepts outputs; inference does not score quality or read labels.
- Implemented the local tactical `--challenge` adapter using the earlier verified metadata/raw-tracking loader, but did not execute full competition-input inference in this checkpoint. No competition numbers, new dependencies/downloads, commit, push or submission.

Commands, provenance and exact verification limits: `INFERENCE.md`.

## Final verification — complete, 1 November 2026

- Created a fresh isolated Python environment and installed all 12 locked dependencies offline from hash-recorded cached wheels. No system/user-site package inheritance; dependency checking passes.
- Verified a minimal current-candidate source snapshot with only runtime/validator/licence/requirements and two-file model bundles, from an unrelated working directory. This is a clean-source working-tree copy, not a committed Git clone or raw-to-model retraining claim.
- Found/fixed acceptance of complex-valued tracking/pitch and malformed FPS. Added regression/staging/version tests. All 98 tests pass in both environments. Learned weights/settings and AST logic outside input validation remain unchanged; old manifests/runtime source are archived and new manifests record revision 2.
- All 12 external raw-input CSVs reproduce frozen held-out predictions byte for byte across all variants and clean/moderate/severe/contextual conditions. Extreme missingness, repeatability, 12 rejection cases and access-guard self-tests pass.
- Full local challenge-input inference passes the standalone format checker and repeats exactly for every variant. Only tactical metadata/raw tracking are read; no reference labels or scoring. No competition match statistics or quality/resource numbers are reported.
- External worker measurements are 22.08–23.80 s wall time and 761.1–765.8 MiB peak working set on this host, within the declared engineering test budget. Cross-platform/hardware equivalence is not claimed.
- Added a root LF checkout policy for byte-stable runtime source hashes; the single root ignore policy and original learned-feature/decoder sources remain unchanged. Independent artifact/AST/environment/contract audit passes.
- No new package versions/downloads, model refit, commit, push, publication or submission.

Details, exact scope and reproduction: `FINAL_VERIFICATION.md`.

## Completed — draft submission and technical-document package

Prepared the technical/submission deliverables with code/runtime dependency/model provenance, known limitations, all three variant choices and exact artifact checks. A strict public runtime/model ZIP excludes match assets and predictions; private candidate folders hold unchanged CSVs and matching Markdown reports. Fresh-environment inference from the extracted ZIP reproduces each candidate byte for byte, with standalone format acceptance and no reference/scoring access. All 103 tests pass. The user explicitly deferred the registered team name and requested all variants remain available for later selection; temporary project-slug names are clearly marked draft. See `DELIVERY.md` and local `outputs/submission/package_20261101/HANDOFF.md`. No commit, push, publication or submission was performed. Count/boundary failures and label sensitivity remain quality limitations; input-only smoke acceptance is not accuracy evidence.

The user's subsequent improvement requests added the completed experiments above before packaging. The switch-event objective now has measured provider-label gains, but the semantic tradeoff still needs independent controlled-possession evidence before declaring a universal default. Keep both fixed event variants and the original reference explicit; do not increase predicted counts merely to pass the diagnostic heuristic or select the more favorable reference convention.

## Team-name update — 1 November 2026

Registered team name confirmed as **AOM**. Current package: `outputs/submission/AOM_20261101/`, with `AOM.csv` and `AOM.md` for each of the three preserved variants. This supersedes the earlier name deferral; primary variant selection remains pending. See `DELIVERY.md`.

## Outstanding administrative information

Member confirmation, signed terms review and external-benchmark review availability remain open. Team name is confirmed as AOM; primary variant remains pending. See `DECISIONS.md`. They were not needed for the environment installation and do not imply an uncompleted setup task.

## Plan update — 28 October 2026

Reviewed the user's friend's recommendations and checked current online rules. Revised the validation policy and schedule; see `VALIDATION_UPDATE_2026-10-28.md`. No annotation work had begun under the old policy. Steps 1–2 remain complete; Step 3 has not started. No packages, external datasets, commits or pushes were added during this plan update.

The later Step 3 completion entry above supersedes this plan update's historical execution status.
