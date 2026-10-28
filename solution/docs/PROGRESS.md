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

## Next — Step 4, not started

Build the external-match validation benchmark described in `VALIDATION_UPDATE_2026-10-28.md`, beginning with one audited Sportec/DFL match. Await the user's explicit instruction to continue.

## Outstanding administrative information

Team-name/member confirmation, signed terms review and external-benchmark review availability remain open. See `DECISIONS.md`. They were not needed for the environment installation and do not imply an uncompleted setup task.

## Plan update — 28 October 2026

Reviewed the user's friend's recommendations and checked current online rules. Revised the validation policy and schedule; see `VALIDATION_UPDATE_2026-10-28.md`. No annotation work had begun under the old policy. Steps 1–2 remain complete; Step 3 has not started. No packages, external datasets, commits or pushes were added during this plan update.

The later Step 3 completion entry above supersedes this plan update's historical execution status.
