# Step 6: deterministic possession solver

Completed 30 October 2026. The first solver combines owner confirmation, limited persistence, separate stoppage detection and optional automatic-tag evidence. The selected `responsive` profile improved spellF1 over the official ±50-frame baseline on the two evaluated external fixtures, including their fixed stress cases. This is limited external evidence, not an estimate of challenge rank.

## Algorithm

`src/possession/state_machine.py` consumes feature values and validity masks, with published half IDs. It never receives reference labels. All temporal settings assume the validated native 25 Hz.

- Measured control requires a player within 4 m and at least a 1 m distance advantage over the opponent. The selected profile requires 0.4 s of consistent evidence for a new owner; shorter opponent touches preserve ownership. Uncertain geometry preserves the last owner for at most 4 s, then outputs X.
- Out-of-pitch evidence requires a signed boundary distance below -0.5 m for 0.4 s. Stationary evidence requires ball speed below 0.35 m/s and both teams' median speeds below 0.8 m/s for 1.5 s, with at least five valid motion observations per team. Missing ball alone cannot start D.
- Once D is established, restart confirmation requires 0.32 s of an in-pitch moving ball near a player, or sufficiently reliable motion from both teams (median at least 1.5 m/s). D persists until that evidence arrives. This can miss a restart during poor tracking.
- Optional automatic tags may supply an owner only when measured control is inconclusive. They require no conflicting team evidence, at least 80% vote share and 1 s confirmation. Restart/goal tag windows do not force D or a restart. The external fixtures lack this tag source, so the tag branch remains unvalidated externally.
- Confirmation refines the boundary back to the start of its evidence streak. This is an offline algorithm. Every half resets owner, stoppage and confirmation state; non-playing frames remain X.

Predictions carry an internal reason code for geometry, persistence, outside pitch, stationary evidence, dead-state persistence, tags or insufficient evidence. Invalid feature zeros are never treated as measured zeros.

## Prespecified selection

`configs/fallback.json` fixed three profiles before evaluation:

| Profile | Owner confirmation | Owner hold | Stationary confirmation |
|---|---:|---:|---:|
| responsive | 0.4 s | 4 s | 1.5 s |
| balanced | 0.8 s | 8 s | 2 s |
| persistent | 1.2 s | 16 s | 3 s |

Other thresholds were shared. Selection maximized development spellF1 with equal weight per match and per clean/moderate/severe regime; each noisy regime first averages its three fixed seeds. MacroF1 breaks ties. J03WPY was the sole development fixture in this small comparison. Objectives were 0.267679, 0.236887 and 0.198607 respectively, selecting `responsive`.

The selected profile was then evaluated once on J03WQQ, with no subsequent parameter changes. Source/configuration hashes in the development report must match before validation or challenge export. This guards accidental use of an old selection after an implementation change. J03WN1 remains sealed and absent locally. Broader development coverage is still needed before extensive tuning.

## Results

All scores use the unchanged official metric and 1000 bootstrap draws. Moderate/severe entries average seeds 17, 43 and 101. All four official baseline windows were recomputed; their 56 metric rows and corruption fingerprints matched the previous benchmark. The ±50 baseline is the primary published comparator; the full report retains every window.

| Fixture | Condition | Baseline ±50 spellF1 | Solver spellF1 | Difference |
|---|---|---:|---:|---:|
| J03WPY development | Clean | 0.3171 | 0.3957 | +0.0786 |
| J03WPY development | Moderate | 0.1839 | 0.2408 | +0.0569 |
| J03WPY development | Severe | 0.1359 | 0.1665 | +0.0307 |
| J03WQQ validation | Clean | 0.3608 | 0.3815 | +0.0207 |
| J03WQQ validation | Moderate | 0.2203 | 0.2495 | +0.0293 |
| J03WQQ validation | Severe | 0.1467 | 0.1717 | +0.0251 |

The measured improvement has material limits:

| Validation condition | Baseline macroF1 | Solver macroF1 | Baseline D F1 | Solver D F1 | Solver X on graded frames |
|---|---:|---:|---:|---:|---:|
| Clean | 0.4162 | 0.6717 | 0.0000 | 0.5474 | 9.7% |
| Moderate | 0.4726 | 0.4397 | 0.3671 | 0.1522 | 41.6% |
| Severe | 0.4228 | 0.3281 | 0.4641 | 0.0947 | 62.4% |

Clean development D F1 was 0.7642 and X coverage was 8.7%. Under noise the solver trades frame coverage and dead-ball recall for fewer invented ownership changes. The official temporal metric bridges X inside graded runs; higher spellF1 must therefore be read alongside abstention and frame metrics. It does not establish reliable recovery through long occlusions. Clean validation predicted 508 possessions against 401 references (baseline: 406); development predicted 420 against 421. Possession count alone is insufficient to assess interval alignment.

Clean validation 95% within-match spell bootstrap intervals overlap: baseline [0.3276, 0.3926], solver [0.3529, 0.4106]. These are not paired improvement intervals or evidence of significance across matches. Both fixtures share a home club and recording provider.

## Failure taxonomy and next work

1. Long occlusions exhaust the short owner hold, producing extensive X; the solver does not recover ownership from player formations.
2. Missing/implausible motion weakens stoppage detection. A previously established D can persist past a missed restart; a real stoppage can remain H/A/X when stationary evidence is absent.
3. Quiet live play can resemble a stoppage. Spatial motion correspondence is uncertain around close players and substitutions.
4. Noisy nearest-player evidence can still fragment ownership. Longer confirmation profiles oversmoothed on development, which is why they were not selected.
5. Challenge automatic tags can improve coverage but may be stale or wrong. External tracking-only gains do not validate them.

Step 7 will compare a structured temporal decoder with this preserved fallback, under the same external evaluation policy. It has not started.

## Full challenge artifact and verification

Generated `outputs/fallback/challenge_20261030T135142012882Z/submission.csv`, SHA-256 `fef823c6aafd7e8393d9be9584292ba11307cff2ea37ad2637c77494482d3ee8`. The unchanged official validator passed: 199319 rows, frames 0–199318, H/A/D/X only. Challenge reference labels were never opened or scored.

The CSV contains 75834 H, 49520 A, 7282 D and 66683 X. Of those X frames, 40462 are outside published playing periods and 26221 are insufficient-evidence frames inside periods. Format validity does not establish prediction quality. This is a working fallback artifact, not a final submission decision.

All 39 tests passed: 24 existing tests plus 15 state-machine/selection tests. New cases cover short opponent touches, sustained turnovers and boundary refinement, owner timeout without fabricated D, unknown initial state, both owner/dead resets at halftime, stationary-ball ambiguity, confirmed stoppages/restarts, brief glitches, conflicting tags, geometry precedence, masks and equal regime weighting. Every evaluated prediction reproduced exactly; feature/source input fingerprints remained unchanged.

Accepted ignored reports:

- `outputs/fallback/development_20261030T135023515007Z/report.json` (49 score rows).
- `outputs/fallback/validation_20261030T135114545044Z/report.json` (35 score rows).
- `outputs/fallback/challenge_20261030T135142012882Z/report.json` (export audit).

## Reproduce

From the project root, using existing prepared inputs:

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s solution/tests -v
& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/run_fallback.py --stage development --bootstrap 1000
```

Use the development report path printed by that run for `--selection` in these commands (the paths below identify the accepted run):

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/run_fallback.py --stage validation --selection outputs/fallback/development_20261030T135023515007Z/report.json --bootstrap 1000
& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/run_fallback.py --stage challenge --selection outputs/fallback/development_20261030T135023515007Z/report.json
```

Each execution creates a new ignored output directory. No packages, human annotations, challenge-label tuning, commits, pushes or external submissions were added in this step.
