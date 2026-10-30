# External oracle diagnosis and revised evidence plan

30 October 2026. This supersedes the proposal to make a decoder over existing evidence the immediate next implementation step. Only external fixture J03WQQ (Fortuna Düsseldorf–FC St. Pauli) was processed. No model was trained, thresholds changed, new fixtures opened or decoder implemented.

## Oracle definitions

The Step 6 production labels are a combined owner/dead decision. They do not expose a team on D or X frames, so simply replacing one component leaves an undefined other component. The diagnostic makes these conventions explicit:

1. **True dead + predicted team:** run the frozen ownership rules with only stoppage evidence disabled (invalidate ball-speed and signed-boundary features for this diagnostic branch). Ownership thresholds, confirmation and gap timeout remain unchanged; this branch receives no references. Set D on true-dead frames, otherwise use that branch's H/A/X. This intervention also removes owner resets caused by predicted stoppages.
2. **True team + predicted dead:** preserve every production D decision; elsewhere supply true H/A. Team identity during a true dead interval is undefined, so extend the preceding true alive team across it, backfilling only a leading interval from that half's first alive team. This prevents a false-alive frame from silently receiving a corrected D. No information crosses halftime. Production X is treated as not-D and receives the oracle team; this oracle repairs team abstention as well as team identity.
3. **Both true:** combine true dead status and the extended reference team; this must reconstruct every graded reference frame exactly.

Two sensitivity checks preserve more of the production output:

- **True dead + existing output:** set true D; retain original H/A/X on true-alive frames, changing a false D there to X.
- **True team + predicted dead, retaining X:** same true-team oracle as above, but original X remains X.

These are defined interventions, not uniquely defined or guaranteed mathematical upper bounds. The official temporal scorer still fills X within graded runs. Reference X and period separators remain excluded. Reference arrays are used only in the diagnostic/evaluator, never passed into the production solver or features.

## Results: spellF1

Moderate results average fixed seeds 17, 43 and 101. All use the unchanged official scorer with 1000 bootstrap draws.

| Diagnostic | Clean | Moderate |
|---|---:|---:|
| Official ±50 baseline | 0.3608 | 0.2203 |
| Frozen solver | 0.3815 | 0.2495 |
| True dead + predicted team | 0.4786 | 0.3643 |
| True team + predicted dead | 0.8056 | 0.7605 |
| Both true | 1.0000 | 1.0000 |
| True dead + existing output | 0.4751 | 0.3621 |
| True team + predicted dead, retaining X | 0.7596 | 0.5549 |

The true-dead/predicted-team oracle predicts 505 clean spells averaging 6.56 s and, under moderate noise, 350 spells averaging 8.41 s. Reference: 401 spells, 8.41 s. Even with perfect stoppages, the clean team stream remains fragmented and noisy gaps still miss changes. Equal average duration does not establish correct intervals.

The diagnosis is therefore broader than poor dead-ball recall alone. About 85% team accuracy conditional on committing to H/A does not establish accurate spell boundaries, and excludes frames where tracking degradation makes the solver abstain. Perfect dead evidence helps, but perfect team identity/timing helps considerably more in these interventions. Keep dead-evidence learning and bidirectional ownership recovery ahead of final decoder tuning.

## Paired uncertainty

The diagnostic uses the same resampled reference possessions for each compared prediction. Matching and total predicted count remain fixed, following the official conditional possession-bootstrap approach; this does not reconstruct and rescore block-resampled sequences. All three corruption seeds share each possession draw and their scores are averaged before taking percentiles, avoiding treatment of repeated truth as independent samples.

| Moderate comparison | SpellF1 difference | Conditional paired 95% interval |
|---|---:|---:|
| Solver minus ±50 baseline | +0.0293 | [+0.0076, +0.0485] |
| True dead + predicted team minus solver | +0.1147 | [+0.0944, +0.1355] |
| True team + predicted dead minus solver | +0.5110 | [+0.4625, +0.5553] |

Clean solver-minus-baseline interval: [-0.0098, +0.0478], point difference +0.0207. These are within-fixture, fixed-matching intervals, not evidence of significance across matches. The moderate positive interval is encouraging but does not resolve the broader validation gate.

## Verification and reproduction

Accepted report: `outputs/oracles/J03WQQ_20261030T141755188811Z/report.json` (28 score rows). All eight recomputed production/baseline metric rows and corruption fingerprints exactly matched the frozen Step 6 validation report. Both-true macroF1 and spellF1 were exactly 1.0 in all four conditions. Original input/target fingerprints were unchanged; frozen source/configuration hashes were verified before execution.

All 47 tests passed, including oracle-component separation, half-boundary handling, undefined-team rejection, source immutability, perfect-prediction bootstrap behavior and agreement of an unresampled bootstrap draw with the official spell score.

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s solution/tests -v
& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/run_oracles.py --match J03WQQ --selection outputs/fallback/development_20261030T135023515007Z/report.json --bootstrap 1000
```

`run_oracles.py` accepts approved external fixtures only; existing sealed-match guards remain active. Detailed results stay ignored. No dependencies, commits or pushes were added.

## Revised implementation order

The user's new sequence replaces the earlier decoder-first Step 7 proposal. The following implementation remains pending; this checkpoint completes the oracle work first.

1. **Learn dead-ball evidence.** Build multi-second player-motion summaries, slow-player fractions, centroid drift, spread and touchline/set-piece geometry. Add available ball geometry/motion, stationary duration and explicit missingness. Use a CPU gradient-boosted model, clean and degraded training replicas, and a player-only ablation. Test target/feature alignment and compare recall alongside precision and calibration. Temporal context features must derive from inputs or model predictions, never true alive/dead history.
2. **Recover ownership across gaps.** Preserve before/after team evidence, gap length and player motion; replace the fixed short timeout with bidirectional evidence. Infer a switch location without reference labels. Measure boundary error and full-interval matching, not only conditional frame accuracy.
3. **Decode stronger evidence.** Combine P(dead) and P(home|alive) in H/A/D Viterbi decoding, with separate stop, restart and owner-switch costs. Emit H/A/D throughout published playing periods; use no reference mask to decide where to emit X. Reset at halves. Preserve the frozen fallback as a comparator.
4. **Broaden validation before selection.** The user requests all seven DFL fixtures in leave-one-match-out evaluation, with three corruption seeds. That future protocol replaces the old sealed-fixture role: once all seven are used, none remains an untouched final holdout. Record this as a new versioned experiment policy rather than rewriting the historical split. Fit models and tune transition costs using only each outer fold's training fixtures (nested inner folds where needed); never tune on its held-out fixture. Report per-match results and matched-seed differences.
5. **Sensitivity and realistic corruption.** Keep provider labels primary. Add a separately reported, prespecified sensitivity reference absorbing sub-3-second opposite-team spells only when surrounded by the same team, without crossing dead intervals or halves. Compare ordinary versus spell/dead-stretch-balanced training with bounded weights. Add input-driven crowded-box/available-height dropout, ghost balls and wrong-team tracks; label aerial proxies as approximations if true height is unavailable. Avoid reference-dependent corruption and synthetic hints revealing the target.
6. **Later optional work.** Simulated tag priors with independent errors and leakage checks, then targeted video recovery only if diagnosed failures warrant it. Keep external-only numerical reporting.

Do not set transition/duration priors from the competition public window or deliberately bias them toward a guessed target-match duration. Use external raw/sensitivity references to report convention uncertainty rather than choosing a hidden-match possession rate.

The user's proposed gates are engineering targets, not official guarantees: substantially improved clean/moderate D recall (roughly 70% clean as a diagnostic), zero playing-period X, moderate count within about 20% of reference, and a positive paired spellF1 improvement interval. Report D precision, coverage, latency, per-match variation and the sensitivity-label results alongside these targets; do not optimize recall in isolation.
