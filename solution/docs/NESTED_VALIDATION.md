# Seven-match nested external validation

Completed 30 October 2026. The protocol was frozen in `configs/nested_validation.json` before preparing J03WN1 or training these folds. All seven external outer folds, 70 input cases, both label conventions and the independent integrity audit are complete. The decoder's interval-score advantage generalizes across these fixtures, but the provider-label possession-count gate fails on five of seven moderate fixture averages. Nested cost selection adds negligible benefit over the previous fixed decoder. This is validation evidence, not competition performance or a finished submission.

## What this protocol tests

Each of the seven public Sportec/DFL fixtures is an outer holdout. The other six fixtures supply six inner folds: each head trains on five fixtures, predicts the sixth, and excludes the outer fixture throughout. The unchanged 25-profile decoder grid is selected by mean inner spellF1, averaging corruption seeds, then weighting inner matches and clean/moderate/severe regimes equally. MacroF1 breaks ties. Both learned heads then refit on all six training fixtures, and the selected decoder evaluates the outer match. An unordered pair of exclusions shares a fitted head pair, reducing 42 inner fits to 21 without changing the training data for either exclusion. Seven additional head pairs serve outer inference.

This nests decoder-cost selection and model refitting. Head architectures and feature definitions remain historically selected: full-feature/frame-weighted dead boosting, and full-feature/capped-spell-weighted ownership boosting. Their earlier selection saw results on six of these fixtures. Consequently this experiment does not erase historical researcher adaptation and cannot be called a wholly untouched seven-match final test. J03WN1 is newly opened for this authorized protocol; it no longer has the sealed role. The original split and all historical checkpoints remain unchanged, and their old guards still reject that fixture.

Supervised training samples about 1 Hz; features and predictions use all native 25 Hz frames. Source hashes, excluded/training identifiers, model hashes, input/cache hashes, fitted row counts and per-fold choices are recorded. Outer label arrays are not passed to model fitting or cost selection. The preparation stage creates separate supervised sample files for each match; fitting loads only the explicitly included training matches. Alternate labels never affect fitting or selection. No competition fixture participates.

Dead weights remain frame-based, following the historical model selection; ownership uses capped inverse sampled-spell frequency. Sampling and capping mean this is approximate spell balancing, not exact equal weight for every native-frame possession. This checkpoint evaluates that fixed learning procedure rather than adding a new dead-stretch weighting search.

## Frozen nested choices

All 21 inner pairs completed 350 candidate scores each (7,350 total), using 294 held-out probability cases. The seven selections were frozen before outer inference:

| Outer fixture | Stop cost | Restart cost | Owner-switch cost |
|---|---:|---:|---:|
| J03WPY | 80 | 80 | 8 |
| J03WOH | 20 | 40 | 8 |
| J03WOY | 40 | 40 | 8 |
| J03WMX | 80 | 80 | 8 |
| J03WQQ | 40 | 40 | 8 |
| J03WR9 | 80 | 80 | 8 |
| J03WN1 | 40 | 40 | 8 |

Every winner uses fixed costs. Three folds select the highest tested stop/restart pair, so this finite grid does not establish an unconstrained optimum. It was not expanded after observing these choices or outer results. Selections, full rankings and inner report paths are in `selection.json`.

## Two label conventions

Provider references map BallStatus and BallPossession to H/A/D/X, as in prior checkpoints. A second evaluation-only reference absorbs an opposite-team H/A run strictly shorter than 3 seconds when immediately flanked by the same team. The shortest eligible run is merged first, earliest on ties, iterating to stability within each half. Dead and ungraded frames never change, and merges cannot cross them or halftime. Exactly 3-second runs remain. Report both conventions; neither is claimed to reproduce the challenge's controlled-possession convention exactly. This sensitivity may substantially reduce spell counts and can create long merged runs; it is not new ground truth.

## Corruption conditions

The existing clean, moderate and severe corruption implementations/configurations stay unchanged. Clean has one seed; each noisy condition uses seeds 17, 43 and 101. Moderate targets 40.7% ball dropout with lognormal median gap 10.72 s (sigma 0.95); severe targets 60% and median 20 s. Realized rates and lengths differ by fixture/seed. Existing player dropout, coordinate noise, duplicate attempts and within-team slot permutations remain as documented in the earlier validation checkpoint. Slot permutations do not themselves create wrong-team tracks.

The additional `contextual` condition starts from the moderate corruption and adds independent burst starts, specified before evaluation:

- Extra dropout starts at base rate one per 45 s, with five times that rate for fast unsupported ball motion or crowded boxes. A fast ball exceeds 15 m/s and is farther than 4 m from every observed player. A crowded box uses at least four players within 8 m of the ball in a penalty-box region. Extra gaps have lognormal median 3 s, sigma 0.7; overlapping bursts merge.
- Ghost-ball bursts start at rate one per 45 s and have lognormal median 0.6 s, sigma 0.5. Ball tracks receive an offset of 5–20 m in a random direction. When original positions are absent, a synthetic field position is used. Ghosts may overwrite a missing ball with a false detection and can fall outside the pitch.
- Wrong-team bursts start at rate one per 45 s and have lognormal median 2 s, sigma 0.5. Coordinates of an observed home/away pair swap while team membership stays fixed; this differs from a within-team identity permutation. Swapping can also move later missing observations between teams.

Risk is derived from original model-facing x/y inputs, never possession or alive/dead labels. No ball height is available. Fast unsupported motion is an aerial-ball proxy, not measured flight. This is a plausible stress hypothesis, not an empirically fitted tracking-error simulator. Starts can overlap, be truncated at half boundaries, or fail for lack of observed players. Stored masks record attempted/applied burst coverage; realized ball missingness also reflects ghost detections filling gaps. This contextual condition is evaluation-only: heads train on the unchanged three original conditions, providing a test of an unseen corruption mixture.

## Comparisons and uncertainty

Each outer case compares the inner-selected decoder, the previous fixed 40/40/8 costs, previous 0.8 s gap composition, and official smooth50, on identical inputs. The first three use the same newly refitted six-match heads; therefore `prior_fixed_costs` and `gap_composition` are algorithm ablations, not reproductions of old four-match model scores. No outer result chooses a replacement profile.

Report frame metrics, interval scores, confusion matrices, dead precision/recall, counts, durations, X, transition errors and false alive islands. Seed-averaged paired reference-possession bootstrap differences share 1,000 draws within a fixture and condition, conditional on fixed matching and prediction counts. A descriptive paired match bootstrap resamples seven outer-match differences. Neither interval refits models or captures overlapping training-set dependence, historical adaptation, unseen error mechanisms or a different label provider. Seven related fixtures remain a small sample: five feature Düsseldorf, alongside Köln–Bayern and the newly opened Bochum–Leverkusen fixture, so team diversity is limited.

## Results

All results below are external Sportec/DFL results. Means weight matches equally, with noisy seed scores averaged within each match; clean is evaluated once because its inputs do not depend on the corruption seed.

| Condition | Official smooth50 spellF1 | Gap composition | Nested decoder | Prior fixed 40/40/8 | Nested decoder, alternate labels |
|---|---:|---:|---:|---:|---:|
| Clean | 0.3081 | 0.5514 | 0.6014 | 0.6014 | 0.6497 |
| Moderate | 0.1895 | 0.3581 | 0.4916 | 0.4918 | 0.5509 |
| Severe | 0.1353 | 0.2376 | 0.4317 | 0.4314 | 0.4994 |
| Contextual | 0.1710 | 0.3151 | 0.4718 | 0.4708 | 0.5312 |

Against gap composition, the decoder improves mean provider-label spellF1 on every match in every condition. Within-match paired conditional 95% intervals exclude zero for all seven moderate, severe and contextual comparisons, and six of seven clean comparisons. Descriptive seven-match bootstrap differences are +0.1335 [0.1163, 0.1496] for moderate and +0.1567 [0.1374, 0.1737] for contextual noise. The small sample and shared training data limit the interpretation of these intervals.

Nested cost selection does not provide a meaningful provider-label gain over prior fixed costs: moderate difference −0.00017 [−0.00076, 0.00050], severe +0.00029 [−0.00068, 0.00116], contextual +0.00096 [−0.00059, 0.00265], using descriptive match bootstrap intervals. Clean interval scores are identical. This checkpoint therefore does not establish a better replacement cost profile for the previously frozen decoder, and no outer result was used to substitute one.

| Fixture | Provider true spells | Alternate true spells | Moderate predicted spells | Moderate provider spellF1 | Contextual provider spellF1 |
|---|---:|---:|---:|---:|---:|
| J03WPY, Düsseldorf–Nürnberg | 421 | 281 | 304.0 | 0.5157 | 0.4933 |
| J03WOH, Düsseldorf–Regensburg | 402 | 268 | 305.0 | 0.5335 | 0.5081 |
| J03WOY, Düsseldorf–Rostock | 456 | 296 | 321.0 | 0.4914 | 0.4719 |
| J03WMX, Köln–Bayern | 511 | 297 | 315.3 | 0.4375 | 0.4260 |
| J03WQQ, Düsseldorf–St. Pauli | 401 | 255 | 333.7 | 0.5299 | 0.5076 |
| J03WR9, Düsseldorf–Kaiserslautern | 370 | 262 | 304.0 | 0.5095 | 0.4946 |
| J03WN1, Bochum–Leverkusen | 500 | 320 | 341.3 | 0.4237 | 0.4011 |

The ±20% count heuristic passes on 3/7 clean, 2/7 moderate, 0/7 severe and 2/7 contextual fixture averages under provider labels. The moderate passes are the two previously opened validation fixtures. Their earlier success did not generalize to the broader count gate. Mean predicted provider spells are 340.6 / 317.8 / 297.4 / 317.5 for clean/moderate/severe/contextual, against 437.3 true spells. Mean predicted durations are 9.29 / 9.90 / 10.54 / 9.86 s against 7.34 s true. These are averages of per-match counts/durations, not one concatenated artificial match.

Alternate references change the mean true count to 282.7 and mean true duration to 11.28 s. Their ±20% count gate passes on 4/7 clean and 6/7 of each noisy condition; J03WQQ overcounts in the noisy conditions. Predictions are identical under the two reference sets. Thus the count diagnosis is highly convention-sensitive: provider labels suggest substantial merging/omissions, while absorbed labels suggest some excess segmentation. The alternate label set does not prove either interpretation matches the challenge. It is not justified to declare the provider-count failures solved simply by choosing the more favorable reference.

Bochum–Leverkusen was newly opened here, with native 25 Hz, 141,561 graded frames and 500 provider possession spells. Clean/moderate/contextual provider spellF1 is 0.5435 / 0.4237 / 0.4011, lower than the seven-match mean. The sample is too small and confounded to attribute this difference to a specific team, league or tracking mechanism. No settings were changed after seeing it.

| Condition | Mean D precision | Mean D recall | Mean macroF1 | Graded-frame X |
|---|---:|---:|---:|---:|
| Clean | 99.55% | 99.74% | 0.9178 | 0 |
| Moderate | 95.46% | 96.44% | 0.8628 | 0 |
| Severe | 92.95% | 94.06% | 0.8284 | 0 |
| Contextual | 94.67% | 96.17% | 0.8541 | 0 |

All 70 selected predictions also have zero X throughout published playing periods. Strong binary D metrics coexist with incomplete possession-boundary recovery; they do not establish that the remaining errors are solely label convention or solely ownership.

## Realized corruption and verification

The audit recomputes missingness from the exact ball-observation masks consumed by feature extraction, including ghosts that fill missing frames. Values below average per-case summaries across all seven matches and three noisy seeds; the gap columns are means of case medians/p95s, not pooled percentiles.

| Condition | Missing-ball frames | Median missing gap, s | P95 missing gap, s |
|---|---:|---:|---:|
| Clean | 0% | 0 | 0 |
| Moderate | 39.19% | 9.91 | 52.80 |
| Severe | 58.05% | 18.42 | 90.52 |
| Contextual | 44.79% | 5.66 | 31.15 |

Contextual risk covers 7.02% of frames; extra-dropout bursts cover 10.15%, ghost bursts 1.50%, and wrong-team bursts 4.70%. These mask coverages overlap and are not additive. Wrong-team coverage means frames containing a swapped pair, not the fraction of all player observations mislabeled. Ghost bursts can split missing runs, explaining why contextual median gaps are shorter despite higher overall missingness. No claim is made that this mixture matches the competition's errors.

All 78 tests pass, including exhaustive decoder checks, input-only corruption reproducibility, exact 3-second sensitivity boundary behavior, half/D/X preservation, fold membership and a leakage test showing that modifying an outer fixture's reference scores cannot change its own inner selection. Dependency checking passes. The independent audit verifies 616 artifact hashes, all 21 inner and seven outer serialized training memberships, all 70 external CSVs, and exact float32 probability reproduction from reloaded heads on seven held-out probes. The main runner also checks 224 evidence/sample/probability artifacts before declaring completion. Original models, feature source files, split configuration and old reports remain unchanged.

No competition-data inference/evaluation, new dependency, commit, push or submission was performed. Only the seventh public match was newly downloaded, at the existing pinned mirror revision. No all-seven final deployment model was fitted: the seven outer models are evaluation models with different training sets, not an ensemble selected for submission.

## Reproduction

From the project root in PowerShell, with existing `.venv`, six prepared fixtures and the seventh public match downloaded by `fetch_sportec.py`:

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/run_nested_validation.py --output outputs/nested_validation/run_20261030 --stage all

& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/audit_nested_validation.py --output outputs/nested_validation/run_20261030
```

The command freezes source/config hashes in `protocol.json` before preparation. Completed preparation, model fits, folds and evaluations are resumable in the same directory; changed sources require a new output directory. Optional stages are `prepare`, `inner`, `outer`, `evaluate`. Inner selection must finish before outer inference. Four CPU workers each use two numerical-library threads; no GPU or new package is required. Intermediate artifacts remain ignored under the single root `.gitignore`.

Final artifacts under `outputs/nested_validation/run_20261030/` are `protocol.json`, `selection.json`, `validation_report.json`, `integrity_and_noise_audit.json`, 21 inner fold reports/models, seven outer fold reports/models, seven per-match evaluation reports and 70 selected external CSVs. CSVs use external native timelines and are format-checked against their fixture row count. This is a benchmark runner, not the final learned challenge inference package. Next delivery checkpoint is an end-to-end inference/reproduction package with a frozen configuration and explicit documentation of the unresolved count/boundary limitations; additional quality experiments must use a declared protocol rather than retuning on favorable outer cases.
