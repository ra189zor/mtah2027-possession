# Learned dead-ball evidence

Completed 30 October 2026. The selected model greatly improves external D recall and retains high precision, including when ball observations are absent. The unsmoothed integration overlay still fragments possessions and retains ownership unknowns; it is not a finished replacement for the fallback.

This checkpoint follows the external oracle diagnosis. It adds a CPU gradient-boosted dead-ball classifier while preserving the existing ownership rules. Bidirectional gap recovery and the structured decoder remain separate, later work. All numerical experiments in this document use external Sportec/DFL data.

## Data and selection contract

The original whole-match split is retained for this first learned component:

| Role | ID | Fixture |
|---|---|---|
| Development | J03WPY | Fortuna Düsseldorf–1. FC Nürnberg |
| Development | J03WOH | Fortuna Düsseldorf–SSV Jahn Regensburg |
| Development | J03WOY | Fortuna Düsseldorf–Hansa Rostock |
| Development | J03WMX | 1. FC Köln–FC Bayern München |
| Validation | J03WQQ | Fortuna Düsseldorf–FC St. Pauli |
| Validation | J03WR9 | Fortuna Düsseldorf–1. FC Kaiserslautern |

Four additional fixtures were downloaded from the existing pinned public mirror revision, verified against publisher hashes, and converted using the audited adapter. References remain provider BallStatus/BallPossession, with status inactive mapped to D. Six fixtures are now prepared; J03WN1 remains untouched in this checkpoint. This is not yet the requested all-seven outer-fold experiment; that will replace the historical sealed role explicitly when implemented. Three development fixtures and both validation fixtures share the same home club/provider, limiting diversity despite the added Bundesliga fixture.

| Fixture | Playing-data frames | Reference D share |
|---|---:|---:|
| J03WPY | 146211 | 44.29% |
| J03WOH | 137214 | 47.57% |
| J03WOY | 142536 | 46.48% |
| J03WMX | 145967 | 41.51% |
| J03WQQ | 142345 | 40.74% |
| J03WR9 | 146810 | 41.89% |

Each prepared fixture adds one ungraded half separator. All reference-period ball positions are observed before artificial corruption; these experiments do not test natural tracking ball gaps. The adapter retains 22 mean observed player slots except J03WQQ (21.63). J03WMX excludes 437901 nonroster/out-of-period player rows under the existing adapter policy while retaining 22 mean player observations; these excluded observations are not silently assigned to a team.

Every candidate is trained on three development matches and evaluated on the fourth, rotating across all four development matches. Threshold/model selection uses only those out-of-fold predictions. Final candidate models train on all four development matches; validation labels do not enter fitting, early stopping, weighting, threshold selection or feature extraction. Previously inspected J03WQQ is opened validation; J03WR9 adds a newly evaluated fixture, not an independent final test after this run.

## Features

`dead_features.py` produces 275 float32 channels. NaN means unavailable; the estimator handles missing values. Features include:

- Team counts, centroid position, RMS spread, spans, short-lag motion observations, median speed and moving fraction.
- Same-team spatial motion over 1 second, mean/median speed, fraction below 0.8 m/s, and motion coverage. This uses unique mutual nearest observations, not stable player-slot identity.
- Fractions near pitch boundaries, in either penalty area and within 3 m of another teammate; centroid drift over 2 and 5 seconds.
- Past and explicitly offline centered 2/5-second means and observation coverage of movement/formation summaries.
- Available ball position, pitch/boundary geometry, speed, observation flags, missing age/future observation interval, distance to the nearest player and set-piece spots, stationary streak and time since an input-derived alive-looking cue.

Windows reset at published half boundaries. Input-only alive cues are not reference alive labels. Player-only ablation columns have no ball dependency, verified by replacing all ball positions with missing values and asserting identical player features. No clock, fixture identity, target state, corrected tracking, event labels or automatic tags enter these features. Coarse centroid drift can be distorted by player dropout, and spatial correspondence does not establish identity; corruption training tests those weaknesses only approximately.

## Learning and prespecified comparisons

The backend is scikit-learn's [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html). Fixed settings in `configs/dead_model.json`: 150 iterations, learning rate 0.08, at most 15 leaves, at least 60 samples per leaf, L2 regularization 1, seed 17, and no automatic random-frame early-stopping split. Four CPU threads are used.

The three prespecified variants are full features/frame weighting, full features/spell weighting, and player-only/spell weighting. Training samples roughly one frame per second, with deterministic per-half sampling offsets. All native 25 Hz input observations still contribute to temporal features; inference and evaluation remain at 25 Hz.

Spell weighting equalizes sampled contiguous H/A/D stretches, including dead stretches, with an inverse-weight cap of 10. Stretches containing no sampled frame remain unrepresented. Each training fixture and each clean/moderate/severe regime receives equal total weight; three noisy seeds share their regime's weight. Frame weighting is a prespecified comparison, not an assumption that weighting by spell is better.

Candidate thresholds are 0.35, 0.50 and 0.65. The primary selection criterion is mean binary dead F1, weighted equally per development fixture and regime; recall breaks ties. Each ablation gets its own development-selected threshold. Validation reports the frozen winner and all three prespecified ablations without retuning.

Predicted probabilities are not assumed calibrated. Brier score, ten-bin reliability summaries and ECE are recorded for native frames; spell weighting changes the effective training distribution. No post-hoc calibration was fitted on validation.

## Evaluation and isolation

The binary detector reports D precision, recall, F1, specificity, Brier/ECE and a separate ball-missing subset. For an integration diagnostic only, its thresholded D mask is overlaid on the frozen owner-only branch (original ownership thresholds/hold, stoppage evidence disabled). Existing unknown ownership remains X. The original rule solver and official ±50-frame baseline are retained.

The overlay receives the unchanged official temporal scorer with 1000 bootstrap draws. A paired conditional possession-bootstrap comparison uses shared reference-spell draws across models and corruption seeds. Matching and prediction counts are fixed, as documented in the oracle report; intervals do not establish uncertainty across fixtures. Improved binary recall need not improve spellF1 before boundary refinement and gap recovery.

## Recorded results

Development selection used 12 fits (three variants times four held-out fixtures) and 252 threshold/condition score rows. The winner was **full features, frame weighting, threshold 0.50**: mean out-of-fold dead F1 0.9613, recall 0.9585 and precision 0.9643, with equal fixture/regime weighting. The best full/spell model used threshold 0.35 (F1 0.9574); the best player/spell model also used 0.35 (F1 0.8912). Spell weighting did not improve this first detector.

Three final models were fitted on all four development matches, each using 160129 augmented training rows. Full models use 275 features (167.98 MiB for the float32 training feature array); player-only uses 228 (139.27 MiB). These figures exclude learner overhead and are not peak RAM measurements. Individual CPU fits took approximately 26–54 seconds. Serialization/reload preserved probe probabilities exactly.

Validation produced 70 score rows. No thresholds or settings changed after validation. Moderate/severe values below are arithmetic means over the three fixed seeds.

| Fixture | Condition | Old rule D recall | Learned D recall | Learned D precision | Learned D F1 |
|---|---|---:|---:|---:|---:|
| J03WQQ | Clean | 0.3800 | 0.9944 | 0.9900 | 0.9922 |
| J03WQQ | Moderate | 0.0826 | 0.9392 | 0.9545 | 0.9468 |
| J03WQQ | Severe | 0.0500 | 0.9011 | 0.9245 | 0.9126 |
| J03WR9 | Clean | 0.5728 | 0.9965 | 0.9907 | 0.9936 |
| J03WR9 | Moderate | 0.3036 | 0.9478 | 0.9465 | 0.9471 |
| J03WR9 | Severe | 0.1433 | 0.9250 | 0.9210 | 0.9230 |

On moderate ball-missing frames specifically, learned D recall is **0.8913 / 0.8966** for J03WQQ / J03WR9, versus **0.0387 / 0.1878** for the old rule solver. Player-only moderate D recall is **0.8449 / 0.8794**, with precision **0.8929 / 0.8580**; on ball-missing frames its recall is **0.8530 / 0.8704**. Player behavior therefore supplies useful evidence independently of ball features on these fixtures. Full features still perform better.

The selected model's moderate Brier scores are **0.0337 / 0.0326**, and ten-bin ECE **0.0075 / 0.0090**. Clean Brier is 0.0050 / 0.0042; severe Brier 0.0533 / 0.0470. These are observed provider/noise-specific calibration diagnostics, not a guarantee of calibrated scores under a different tracking distribution.

## Integration limits and gates

| Fixture | Condition | Official ±50 spellF1 | Previous rule solver | Learned-mask overlay |
|---|---|---:|---:|---:|
| J03WQQ | Clean | 0.3608 | 0.3815 | 0.4250 |
| J03WQQ | Moderate | 0.2203 | 0.2495 | 0.2740 |
| J03WQQ | Severe | 0.1467 | 0.1717 | 0.1737 |
| J03WR9 | Clean | 0.3152 | 0.3826 | 0.4342 |
| J03WR9 | Moderate | 0.1860 | 0.2646 | 0.2590 |
| J03WR9 | Severe | 0.1374 | 0.1929 | 0.1733 |

The overlay improves over the official baseline in all six reported cells, but trails the previous rule solver on noisy J03WR9. Better frame detection alone does not solve interval alignment. The selected detector stays selected by the prespecified binary development criterion; no validation-driven switch of models or thresholds was made.

| Moderate condition | Reference possessions | Overlay possessions | Count difference | X on graded frames |
|---|---:|---:|---:|---:|
| J03WQQ | 401 | 541.7 | +35.1% | 20.0% |
| J03WR9 | 370 | 498.3 | +34.7% | 19.1% |

Counts average the three seeds and use the official X-bridging convention. Mean predicted durations are 5.17 s / 5.74 s under moderate noise. Both the zero-X and ±20% count gates remain unmet. Clean overlays also overfragment (592 / 495 possessions). The tree scores are thresholded directly in this diagnostic without a new temporal decoder or boundary filter; small false-dead intervals can split possessions despite high aggregate D F1. Ownership gaps and temporal boundaries are the next work, not an excuse to promote the current overlay as final.

Moderate paired conditional spellF1 differences versus official ±50 are:

- J03WQQ: +0.0537, 95% interval [+0.0324, +0.0726].
- J03WR9: +0.0730, 95% interval [+0.0496, +0.0968].

These meet a within-fixture conditional interval check against the official baseline; the broader all-seven outer-match gate remains pending. They do not establish significance over the previous rule solver or across recording domains.

## Integrity and retained artifacts

All 55 tests passed, including new feature target rejection, slot-order invariance, source immutability, ball-independent player ablation, missing-window handling, half resets, deterministic sampling, capped spell weighting, split guards and binary-metric accounting. All 14 J03WQQ production/baseline score rows and corruption fingerprints exactly matched the previously accepted benchmark. Frozen source/model-selection hashes match throughout validation.

As an additional plausibility check, shifting the clean reference labels by half a period within each half drops D F1 from 0.9922 to 0.4587 on J03WQQ and from 0.9936 to 0.3980 on J03WR9. This negative control is not a proof against every possible leak; it was not used for training or selection. Source schema checks and the label-free feature API are the primary separation controls.

Accepted local outputs:

- `outputs/dead_model/development_20261030T150036779468Z/development_report.json` and three checksum-recorded `.joblib` models.
- `outputs/dead_model/validation_20261030T151158159199Z/validation_report.json`, per-case probability arrays and `integrity_audit.json`.
- 42 label-free feature cases under each fixture's ignored `dead_features/` cache.

No production ownership thresholds were changed. No new decoder, competition-data inference/evaluation, annotation, commit, push or submission occurred in this checkpoint.

No new temporal decoder, ghost/wrong-team corruption, sensitivity reference, video recovery or simulated tagging is part of this checkpoint. Noise remains the previously frozen clean/moderate/severe definition, with seeds 17, 43 and 101. The zero-X and possession-count gates apply to the later integrated system, not to this detector alone.

## Reproduction

From the project root, install the pinned `solution/requirements-lock.txt` in the existing Python 3.11 environment. Download/prepare missing approved external fixtures with the existing `fetch_sportec.py` / `prepare_sportec.py` commands and their match IDs.

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s solution/tests -v
& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/train_dead_model.py --stage prepare --matches J03WPY J03WOH J03WOY J03WMX J03WQQ J03WR9
& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/train_dead_model.py --stage development
```

For validation, pass the `development_report.json` printed by that run with `--stage validation --selection <path>`. Sources/configuration and model checksums must still match the frozen report. Local model files are trusted, checksum-verified joblib outputs; do not load untrusted serialized models. Source inputs, feature caches, probability arrays, models and detailed results remain ignored by the single root `.gitignore`.

Accepted validation command:

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/train_dead_model.py --stage validation --selection outputs/dead_model/development_20261030T150036779468Z/development_report.json
```

New pinned dependencies are scikit-learn 1.9.1, SciPy 1.17.1, joblib 1.6.0, threadpoolctl 3.7.0, narwhals 2.28.0 and cloudpickle 3.1.2. Public-data attribution and transformation notices remain in `EXTERNAL_DATA_SOURCES.md`; no pretrained weights were downloaded.
