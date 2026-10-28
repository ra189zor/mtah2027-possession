# Step 4: initial external validation benchmark

Completed **28 October 2026**. The benchmark infrastructure runs on one development fixture and one separate validation fixture. This is an initial benchmark, not a claim that our improved algorithm generalizes or that synthetic data reproduces the challenge exactly.

## Frozen match split

`solution/configs/validation.json` was created before scoring. It assigns whole matches, with no frames or halves shared between groups:

| Role | Matches | Current use |
|---|---|---|
| Development | J03WPY, J03WOH, J03WOY, J03WMX | J03WPY downloaded, audited and benchmarked; others reserved for expansion |
| Validation | J03WQQ, J03WR9 | J03WQQ audited and scored with fixed official baselines; J03WR9 reserved |
| Sealed final test | J03WN1 | Not downloaded, prepared, inspected or scored |

The scorer and adapter reject attempts to use the sealed fixture. J03WQQ is an opened validation fixture, not a sealed test: its initial baseline evaluation is recorded here and should not become an unrestricted tuning loop. A genuinely sealed fixture remains for final model selection. Additional development matches are needed before leave-one-match-out comparisons can support generalization claims.

BAR-ATH never enters these supervised splits. Our earlier plan to create human labels on the supplied fixture was superseded by the current rules before any such work began. See `VALIDATION_UPDATE_2026-10-28.md`.

## Public source and verified download

Sportec Open DFL / IDSSE, mirrored by PySport, is licensed under CC BY 4.0. Full attribution, links, pinned revision and transformation notes are in `EXTERNAL_DATA_SOURCES.md`. Only metadata and tracking for the two selected fixtures were downloaded, approximately 694 MiB total. No video, event files, hosted model or paid service was required.

The downloader pins the source revision and verifies file sizes and the publisher's SHA-256 for each large tracking file. Raw downloads and prepared arrays remain in ignored `data/external/sportec/`. No new Python package was installed.

## Actual data audit

| Property | J03WPY: development | J03WQQ: validation |
|---|---:|---:|
| Playing-data frames | 146211 | 142345 |
| Recorded periods | 2 | 2 |
| Native sample rate | 25 Hz, timestamp-verified | 25 Hz, timestamp-verified |
| Provider H frames | 51192 | 37030 |
| Provider A frames | 30268 | 47322 |
| Provider D frames | 64751 | 57993 |
| Provider possession spells | 421 | 401 |
| Natural ball-position availability | 100% | 100% |
| Mean observed player count | 22.00 | 21.63 |

These are approximately 192 minutes of recorded match time in total, with 822 provider-labelled possession intervals. They are two fixtures, not 822 independent match-level examples. Both have Fortuna Düsseldorf as the home team, so this initial sample has limited team and recording diversity.

Frame IDs are aligned individually, not assumed contiguous. Timestamp differences must agree with ID increments at 25 Hz. A synthetic sparse-ID test verifies that missing frames do not shift later observations. A single X separator is inserted between halves; unknown provider states remain X. The exact scorer handles each valid run independently, and the baseline smoother is reset at each half boundary.

Mapping: inactive ball becomes D; active ball with ownership 1 becomes H and ownership 2 becomes A. Unknown live ownership is ungraded, not guessed. This maps provider fields; it does not prove their definition of control matches Metrica's corrected labels.

## Input/target isolation

Prepared inputs contain only player XY, ball XY, team-of-slot, period, pitch dimensions and frame rate. Player identities, ball-status codes, ownership flags and events are excluded. Coordinates are converted from centred metres to corner-origin metres; raw speed fields are not used. Targets are saved separately.

Slots are anonymous track slots, not named-player features. They can still be temporally stable in the clean data, so any later learned model should pool by team or otherwise avoid learning an identity from slot index.

The corruption API accepts only the input schema. It never receives target labels. Before/after fingerprints ensure source arrays and targets remain unchanged. Prepared inputs/targets are checked against their audit hashes before scoring.

## Fixed stress scenarios

- **Clean:** preserve the supplied external positions.
- **Moderate:** nominal 40.7% ball dropout in bursts, player dropout, 0.15 m coordinate noise, occasional duplicated observations and within-team slot permutations every 30 seconds.
- **Severe:** nominal 60% ball dropout, longer bursts, more player dropout, 0.40 m noise, more duplicates and permutations every 10 seconds.

Seeds are fixed at 17, 43 and 101. Clean is evaluated once; both noisy scenarios use all three. Each input condition is scored with the four supplied baseline window settings, producing 28 score rows per match, 56 overall. No window or corruption parameter was chosen from the scores.

Actual missing-ball fractions vary because bursts have different lengths and the match is finite. Across these runs, moderate produced 35.1–43.7% missing observations; severe produced 57.9–63.4%. Gap duration and observation counts are logged for every condition. The challenge's measured gap aggregate includes stoppages, not just live occlusion. These scenarios therefore test robustness rather than claim calibrated simulation of every tracking failure.

## Baseline results

For the fixed official ±50-frame smoother:

| Input condition | J03WPY spellF1 | J03WQQ spellF1 |
|---|---:|---:|
| Clean | 0.3171 | 0.3608 |
| Moderate, mean of 3 seeds | 0.1839 | 0.2203 |
| Severe, mean of 3 seeds | 0.1359 | 0.1467 |

These are external-reference scores, not predictions of our official challenge rank. Provider conventions and error distributions differ.

The clean nearest-player baseline predicts no D when the ball/player geometry is always available, so its D F1 is zero. Introducing missing observations can accidentally improve frame macroF1 while damaging spellF1, because missing-ball=D sometimes coincides with a genuine stoppage by chance. That is a useful warning against optimizing frame metrics or treating missingness as a dead-ball detector.

Every condition uses the unchanged official `possession_metrics.py`, including 1000 bootstrap draws. The intervals describe within-match possession resampling. They do not measure uncertainty across leagues, teams or recording conditions. Future model comparisons should use matched corruptions, paired differences and whole-match reporting; no significance or model-improvement claim is made in this step.

The first development benchmark was executed twice: all 28 score rows and corrupted-input fingerprints reproduced exactly.

## Verification

Eleven tests passed. They cover H/A/D/X mapping and dead-state precedence, rejecting target fields in inputs, unchanged source arrays, deterministic corruption, clean-input preservation, half boundaries, exact-metric separation of same-team spells across halftime, sealed-match guards, disjoint match splits, metre-based team geometry, slot-permutation invariance and sparse external frame IDs.

Accepted local reports:

- `outputs/external_validation/J03WPY_20261028T163120688394Z/benchmark.json`
- `outputs/external_validation/J03WQQ_20261028T163158772856Z/benchmark.json`

Raw/transformed data and outputs are ignored by the single root `.gitignore`. Source/configuration, tests, attribution and this aggregate summary are suitable for the user's code commit. No commit or push was made.

## Reproduction

From the project root in PowerShell, after the existing environment is installed:

```powershell
& '.\.venv\Scripts\python.exe' solution/tools/fetch_sportec.py --match J03WPY
& '.\.venv\Scripts\python.exe' solution/tools/prepare_sportec.py --match J03WPY
& '.\.venv\Scripts\python.exe' solution/tools/benchmark_external.py --match J03WPY --bootstrap 1000
& '.\.venv\Scripts\python.exe' solution/tools/fetch_sportec.py --match J03WQQ
& '.\.venv\Scripts\python.exe' solution/tools/prepare_sportec.py --match J03WQQ
& '.\.venv\Scripts\python.exe' solution/tools/benchmark_external.py --match J03WQQ --split validation --bootstrap 1000
& '.\.venv\Scripts\python.exe' -m unittest discover -s solution/tests -v
```

Downloads verify and reuse existing complete files. Incomplete `.partial` downloads are preserved for inspection rather than silently overwritten. Benchmark outputs go into fresh timestamped directories. The scripts run from the existing environment without floodlight or kloppy installation.

## Next step

Step 5: implement team-level geometry, missingness and motion features for the challenge and external data. Broaden development-match coverage before extensive tuning or learned-model selection. The deterministic fallback, stronger decoder and any neural models are later steps. Await explicit instruction to continue.
