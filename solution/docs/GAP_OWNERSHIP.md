# Bidirectional ownership recovery

Historical checkpoint: the later temporal decoder now reduces the fragmentation reported below. See `TEMPORAL_DECODER.md` for current results; the ownership evidence and this checkpoint's frozen artifacts remain unchanged.

Completed 30 October 2026. The selected gap-aware ownership model improves external spellF1 and eliminates playing-period X while keeping dead probabilities unchanged. Noisy possession counts still overfragment, so this checkpoint is not a finished submission system.

This checkpoint follows the learned dead-ball model. It adds future/past geometric control evidence and conditional ownership learning, keeping the selected dead-ball model and its threshold frozen. It does not implement Viterbi or smooth the dead mask. All experiments use external Sportec/DFL fixtures only.

## Inputs and gap context

`gap_ownership.py` adds 23 channels to the existing 275 dead-evidence channels, producing 298 float32 inputs. The new channels include each team's ball distance and nearest-player motion, previous/next confirmed geometric owner, endpoint ball position/speed, elapsed time from each endpoint, total anchor gap, relative gap position and endpoint agreement.

A geometric anchor requires the ball within 4 m of a player, at least a 1 m nearest-distance advantage over the opponent, and 0.4 s of consistent evidence. Confirmation backdates to the first frame of that supporting run. These anchors come from tracking, not reference ownership. Lookups and temporal features stop at halftime. Past/future anchors can span predicted stoppages; the final dead mask resets local owner confirmation, and the learned model receives endpoint distance/time rather than a ground-truth restart hint.

The simple `bridge` comparison carries the same owner between agreeing endpoints, interpolates between differing endpoints, or uses a single available endpoint. With no confirmed endpoint it uses available raw distance evidence, otherwise a neutral 0.5. Matching endpoints do not prove an unchanged possession during a long gap. Multiple unseen turnovers may remain unidentifiable.

The learned heads use movement, formations, observed geometry and anchor context to predict P(home | alive). They can revise the switch position relative to simple midpoint interpolation. There is no target-state history, reference event cue, corrected tracking, automatic tag or competition input in these features.

## Prespecified learning and selection

`configs/gap_ownership.json` fixes two learned variants (frame versus capped spell weights), the simple bridge ablation and confirmation durations of 0.4, 0.8 and 1.6 seconds. The CPU gradient-boosted ownership models use 120 iterations, learning rate 0.08, 15 leaves, minimum 60 samples per leaf, L2 regularization 1 and seed 17. No random-frame early-stopping split is used. Training samples about 1 Hz from true H/A frames only; native-frame inputs still form every temporal feature, and evaluation runs at 25 Hz. Each fixture/regime has equal training weight, shared across three corruption seeds.

Four development fixtures rotate as held-out matches. For each fold, the selected dead-model parameters are refitted on the other three fixtures; the final all-four-match dead model is never used to score its training fixtures for ownership selection. Ownership models also train on those three fixtures. No learned dead probability enters the ownership feature matrix, avoiding a supervised stacking input in the owner head. Development predictions are out of fold for both heads, though the dead architecture/threshold and candidate list were previously selected using development data; these are selection results, not an unbiased final performance estimate.

The nine family/confirmation combinations are ranked by development spellF1, weighting each match and clean/moderate/severe regime equally. MacroF1 breaks ties. The best confirmation for each family is frozen before validation. Final ownership models train on all four development fixtures; validation uses the existing all-four-match dead model unchanged. J03WQQ and J03WR9 are both opened validation fixtures. All-seven nested outer-match validation remains later work; J03WN1 stays untouched here.

## Composition and uncertainty

The local composition rule accepts a new team only after its posterior is at least 0.55 and supports the switch for the selected confirmation duration. It then refines the handover back to the first frame of that evidence streak. Predicted dead frames reset owner confirmation. Every candidate must preserve the exact frozen D mask inside published periods.

The system emits H/A/D throughout published playing periods, and X only outside those periods. At the start of an alive block, it uses the current posterior even if uncertain; a neutral 0.5 chooses home deterministically. This removes output abstention, not uncertainty. Low-confidence and no-anchor rates remain explicit diagnostics, and true-alive ball-missing accuracy measures whether gap completion is actually useful. No reference mask decides where X is allowed.

The previous rule solver, unchanged official ±50 baseline, and learned-dead overlay with the old ownership timeout remain comparators. Validation also retains all three frozen ownership families. Native probability arrays and development out-of-fold probabilities are saved for later decoder comparisons. Paired intervals use the documented shared-possession, fixed-matching conditional bootstrap, not between-match uncertainty.

## Selection and validation results

Four development folds produced 252 candidate/condition score rows. There were four dead refits and eight ownership fold fits, followed by two final ownership fits. Selection chose **spell weighting with 0.8 s confirmation** (mean development spellF1 0.3734, macroF1 0.8614). Best frame-weighted ownership used 0.4 s (spellF1 0.3701); best bridge-only used 1.6 s (0.3042). Final ownership models used 88135 augmented true-alive training rows each. Owner fold fits took about 21–22 s; final fits about 25 s on four CPU threads. Model serialization/reload preserved probe probabilities exactly.

Validation used both opened fixtures once with these frozen family settings. The table reports the selected model; moderate/severe values average seeds 17, 43 and 101.

| Fixture | Condition | Official ±50 spellF1 | Rule solver | Previous learned-dead overlay | Gap-aware selected |
|---|---|---:|---:|---:|---:|
| J03WQQ | Clean | 0.3608 | 0.3815 | 0.4250 | 0.5365 |
| J03WQQ | Moderate | 0.2203 | 0.2495 | 0.2740 | 0.3578 |
| J03WQQ | Severe | 0.1467 | 0.1717 | 0.1737 | 0.2286 |
| J03WR9 | Clean | 0.3152 | 0.3826 | 0.4342 | 0.5173 |
| J03WR9 | Moderate | 0.1860 | 0.2646 | 0.2590 | 0.3150 |
| J03WR9 | Severe | 0.1374 | 0.1929 | 0.1733 | 0.2118 |

Frame-weighted ownership has slightly higher moderate/severe validation spellF1 than the selected spell-weighted model: 0.3592/0.2367 on J03WQQ and 0.3180/0.2187 on J03WR9. It was not substituted after validation. Both learned families outperform the simple bridge under moderate/severe corruption; bridge-only moderate spellF1 is 0.2892/0.2507. Better held-out frame accuracy does not automatically select the best interval configuration.

Selected ownership-head accuracy on **all true-alive ball-missing frames** is 78.6% / 80.2% under moderate noise, versus 67.6% / 69.2% for bridge-only. This is a nonabstaining comparison, not accuracy conditioned on committing. Selected true-alive Brier scores are 0.1145 / 0.1033. The frame-weighted ablation has better raw ownership accuracy/Brier on these validation fixtures; weighting was selected for development interval scores, not presumed superior calibration.

## Coverage and remaining count failure

Every new family emits zero X inside published periods in every development/validation condition. D labels and probabilities are exactly unchanged from the frozen dead detector on validation. Moderate selected macroF1 is 0.8625 / 0.8561. Removing X does not establish confident recovery: 9.9% / 9.0% of predicted-alive frames are below the 0.55 ownership confidence threshold. All evaluated halves have at least one geometric anchor; the fully anchorless fallback is tested synthetically, not validated on a naturally anchorless half.

| Fixture | Condition | True count | Selected predicted count | Predicted mean duration |
|---|---|---:|---:|---:|
| J03WQQ | Clean | 401 | 411 | 8.18 s |
| J03WQQ | Moderate | 401 | 611.3 | 5.59 s |
| J03WQQ | Severe | 401 | 890.7 | 3.88 s |
| J03WR9 | Clean | 370 | 346 | 9.82 s |
| J03WR9 | Moderate | 370 | 593.0 | 5.75 s |
| J03WR9 | Severe | 370 | 868.0 | 3.92 s |

Counts/durations average the noisy seeds and use the official spell definition. Clean counts pass the proposed ±20% heuristic, but moderate counts are **+52.5% / +60.3%** and severe fragmentation is worse. The count gate remains unmet. Relative to the previous X-producing overlay, output completeness exposes additional false-alive intervals that the scorer previously bridged as X.

An audit of the unchanged dead mask gives the following moderate averages:

| Diagnostic | J03WQQ | J03WR9 |
|---|---:|---:|
| Predicted contiguous alive islands | 385.0 | 391.0 |
| Alive islands lasting at most 1 s | 253.0 | 248.0 |
| Alive islands entirely inside reference D | 177.3 | 112.3 |
| Predicted possession spells mostly inside reference D | 192.3 | 128.0 |

The short-island and wholly-dead counts are separate groups, not asserted to be identical. Under severe noise, wholly reference-dead alive islands rise to 373.3 / 311.7. High frame D F1 can coexist with many short errors that fragment interval predictions. Owner switches are not the only remaining cause. Do not fix this by post-hoc validation threshold changes or unconditional removal of every short spell.

## Paired uncertainty and limits

Moderate selected-minus-comparator spellF1 differences, with 95% conditional paired possession-bootstrap intervals:

| Fixture | Comparator | Difference | Conditional interval |
|---|---|---:|---|
| J03WQQ | Official ±50 | +0.1375 | [+0.1112, +0.1654] |
| J03WQQ | Previous rule solver | +0.1083 | [+0.0807, +0.1373] |
| J03WQQ | Previous learned-dead overlay | +0.0838 | [+0.0607, +0.1090] |
| J03WR9 | Official ±50 | +0.1291 | [+0.1000, +0.1577] |
| J03WR9 | Previous rule solver | +0.0505 | [+0.0180, +0.0809] |
| J03WR9 | Previous learned-dead overlay | +0.0560 | [+0.0319, +0.0786] |

These intervals are conditional on global matching and prediction counts; they share resampled reference possessions across compared methods and corruption seeds. They are not full sequence block-bootstrap intervals or evidence across independent recording domains. Both opened validation fixtures share the same home club/provider; simulation still lacks ghost balls and wrong-team tracks. Equal-endpoint gaps may conceal multiple turnovers, and pointwise ownership estimates can flicker even with local confirmation.

## Verification and artifacts

All 65 tests passed, including ten new tests for long-gap agreement, differing endpoint timing, leading gaps, half isolation, brief touches, exact D-mask preservation, zero playing-period X, low-confidence fallback, input immutability and feature schema alignment.

The integrity audit confirmed:

- All 28 regenerated development-fold dead scores match the original dead-model out-of-fold precision/recall/F1 exactly.
- All 14 validation dead-probability arrays match the prior frozen model outputs byte-for-byte at the array level.
- All 42 recomputed comparator metric/fingerprint rows match the previous accepted validation report.
- All source/selection hashes remain unchanged, held-out fixtures are excluded from fold training, and the selected system has zero playing-period X with the exact D mask in all 14 validation conditions.

Accepted ignored artifacts:

- `outputs/gap_ownership/development_20261030T153115114216Z/development_report.json`, four fold-model bundles, 28 out-of-fold probability files and two final ownership models.
- `outputs/gap_ownership/validation_20261030T154041424173Z/validation_report.json` (84 score rows), 14 probability files and `integrity_and_fragmentation_audit.json`.
- 42 label-free ownership feature cases under the six fixtures' ignored `owner_features/` caches.

No dependencies, downloads, human labels, competition-data inference/evaluation, commits, pushes or submissions were added in this checkpoint.

## Next

Implement three-state temporal decoding over the stronger dead/ownership evidence, comparing stop, restart and team-switch costs using development out-of-fold predictions. The stored probabilities avoid fitting against in-sample dead predictions. Keep all current systems as comparators. Diagnose short false-alive islands and boundary error explicitly; do not choose priors from competition-match totals. All-seven nested outer-match validation, stronger corruptions and secondary convention-sensitive labels remain pending.

## Reproduce

Use the existing environment, six prepared external fixtures and dead-feature caches. The accepted dead-model selection is `outputs/dead_model/development_20261030T150036779468Z/development_report.json`.

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s solution/tests -v
& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/train_gap_ownership.py --stage prepare --dead-selection outputs/dead_model/development_20261030T150036779468Z/development_report.json --matches J03WPY J03WOH J03WOY J03WMX J03WQQ J03WR9
& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/train_gap_ownership.py --stage development --dead-selection outputs/dead_model/development_20261030T150036779468Z/development_report.json
```

For validation, use `--stage validation`, the same `--dead-selection`, and `--selection` pointing to the ownership development report printed by the previous command. Frozen source and model checksums must match. No packages or raw downloads are added in this checkpoint. Feature caches, local serialized models, probabilities and detailed reports remain ignored under the single root `.gitignore`.

Accepted validation command:

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/train_gap_ownership.py --stage validation --dead-selection outputs/dead_model/development_20261030T150036779468Z/development_report.json --selection outputs/gap_ownership/development_20261030T153115114216Z/development_report.json
```
