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

## Next — learned evidence, not started

The user's revised order supersedes the decoder-first Step 7 proposal: learn dead-ball evidence from player behavior and available ball cues, then add bidirectional gap-aware ownership before the three-state decoder. Expand to a versioned seven-fixture outer-match validation protocol before selection. The old sealed role will not survive all-seven cross-validation; historical split files remain unchanged. No new model or decoder has been implemented in this oracle checkpoint.

## Outstanding administrative information

Team-name/member confirmation, signed terms review and external-benchmark review availability remain open. See `DECISIONS.md`. They were not needed for the environment installation and do not imply an uncompleted setup task.

## Plan update — 28 October 2026

Reviewed the user's friend's recommendations and checked current online rules. Revised the validation policy and schedule; see `VALIDATION_UPDATE_2026-10-28.md`. No annotation work had begun under the old policy. Steps 1–2 remain complete; Step 3 has not started. No packages, external datasets, commits or pushes were added during this plan update.

The later Step 3 completion entry above supersedes this plan update's historical execution status.
