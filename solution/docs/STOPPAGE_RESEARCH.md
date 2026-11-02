# Stoppage evidence, temporal boundaries and external transfer

2 November 2026. This checkpoint follows the user's authorization to start the researched improvement plan. It separates evidence from decoder effects, adds two external provider samples, and tests a fixed learned stop/restart component. It does not train or select settings on the competition match or its public reference window.

## Current-model diagnosis

All 70 corrected-combined outer predictions reproduce exactly: seven DFL fixtures, one clean input and three seeds for moderate, severe and contextual corruption. The original nested-cost benchmark and the fixed 40/40/8 deployment-cost experiment are named separately. Deployment-cost experiments still use six-match held-out heads, not the all-seven fitted deployment weights.

Provider-reference spellF1, averaged equally across matches and then seeds:

| Intervention | Clean | Moderate | Severe | Contextual |
|---|---:|---:|---:|---:|
| Current combined, deployment costs | 0.63119 | 0.51582 | 0.44583 | 0.49071 |
| Raw dead threshold with independently decoded team | 0.59140 | 0.39435 | 0.26559 | 0.34474 |
| Zero stop/restart penalties | 0.58632 | 0.37746 | 0.25451 | 0.33316 |
| True dead probabilities, same decoder | 0.63866 | 0.56000 | 0.50843 | 0.54200 |
| True dead mask, independently decoded team | 0.62834 | 0.55145 | 0.50374 | 0.53502 |
| True team with the current predicted dead mask | 0.99219 | 0.91314 | 0.86726 | 0.89621 |
| Both true | 1.00000 | 1.00000 | 1.00000 | 1.00000 |

The current deployment-cost decoder's clean dead precision/recall are 99.54% / 99.74%; moderate precision/recall are 95.42% / 96.52%. On moderate inputs, stop and restart recall within one second are 67.68% / 76.54%; matching precision is 68.28% / 77.32%. Thus framewise dead detection can be strong while boundaries are displaced. Removing smoothing is harmful on these fixtures. Perfect ownership/timing has a much larger external spellF1 effect than perfect dead evidence.

Oracle interventions use references only inside the diagnostic evaluator. The owner-only stream is decoded with zero dead probability; its continuity differs from the original stream. Reference ownership during true dead periods is extended from the preceding owner within each half, with leading backfill. False predicted D is retained in the true-team intervention. Hard masks and probability interventions are different experiments, not interchangeable ceilings. Both-true reconstructs every provider reference exactly in all 70 cases. The absorbed-under-three-second convention is also reported separately; its both-true provider intervention need not be perfect against that altered reference.

The initial run completed all inference and fold reports, then hit a missing aggregate `accuracy` field. Reporting was fixed to derive accuracy from each confusion matrix. Only aggregation was rerun. The original inference protocol is preserved; the aggregate records its reporting-source hash and all input report hashes.

## Two additional public external matches

Downloaded Metrica Sports Sample Games 1 and 2 at revision `e706dd506b360d69d9d123d5b8026e7294b13996`. Every file is verified against its pinned Git blob and recorded SHA-256. These are anonymized public sample fixtures, not the challenge game. Model-facing inputs contain only tracking, team membership, pitch, frame rate and periods. Event files are isolated as evaluation references.

These samples do not contain complete framewise H/A/D truth. The fixed adapter labels successful-pass interiors alive. It labels a stoppage only from a BALL OUT or explicit shot OUT/GOAL end to the next explicit set piece/goal kick when no intervening live action occurs. It never infers dead truth from missing ball observations, fouls, event gaps alone or predictions. All uncertain frames remain excluded. We report both original intervals and dead interiors with a one-second boundary guard.

The unchanged final corrected-combined model was tested without fitting:

| External sample | Event-reference coverage before guard | Accepted stoppages | Guarded D recall | Guarded D precision |
|---|---:|---:|---:|---:|
| Game 1 | 38.40% | 56 | 87.01% | 99.991% |
| Game 2 | 44.97% | 59 | 88.79% | 99.954% |

These are partial, event-derived binary measurements, not complete possession scores or full-match accuracy. No conflicting reference frames were retained. Source processing strongly confounds missingness with status: every labelled pass frame has a visible ball, while nearly every labelled dead interior has no ball. This cannot validate live-play occlusion handling or visible-ball stoppages adequately. Same-vendor sample data are also not the same product as the challenge's automatic tracking. The sample results therefore cannot establish challenge generalization. Undefined subgroup metrics are null; reporting version 2 reuses exactly the same hash-checked prediction arrays as version 1.

## Learned boundary experiment

The diagnostic supports testing event timing before globally lowering stop/restart penalties. A fixed CPU gradient-boosted model learns three classes: no boundary, stoppage start and restart. It uses the existing 275 physical dead features plus dead-probability past/future/change context over 0.2, 0.48, 1, 2 and 5 seconds (291 columns). This is a targeted temporal-feature experiment, not a TCN implementation.

Each of seven outer folds trains on the other six fixtures. For training fixture T and outer fixture O, all upstream dead probabilities come from a five-match model excluding both O and T. Native one-second sampling plus boundary-centre/nearby rows restores full-frame class proportions using the audited sampler. Clean/moderate/severe regimes share match/regime weight; contextual noise and alternate conventions are evaluation-only. All settings and reward strength were fixed before fitting this candidate.

The new decoder preserves corrected-combined team probabilities, dead emissions, ownership-switch evidence and base costs. It reduces only stop/restart costs near learned events, with a nonnegative floor. Zero boundary reward must reproduce the previous event decoder. Exhaustive short-path tests verify global optimality, and temporal features reset at half boundaries.

The full run evaluates 280 score rows (70 inputs, two candidates, two conventions). Provider spellF1:

| Condition | Current deployment costs | Boundary candidate | Change | Descriptive paired-match 95% interval | Improved fixtures |
|---|---:|---:|---:|---|---:|
| Clean | 0.63119 | 0.63180 | +0.00062 | [-0.00012, +0.00135] | 4/7 |
| Moderate | 0.51582 | 0.51658 | +0.00076 | [-0.00041, +0.00168] | 5/7 |
| Severe | 0.44583 | 0.44740 | +0.00157 | [+0.00097, +0.00218] | 7/7 |
| Contextual | 0.49071 | 0.49094 | +0.00023 | [-0.00104, +0.00143] | 4/7 |

These comparisons use the same fixed deployment costs. Earlier corrected-combined results used per-fold nested costs (moderate 0.51590); that is not the exact comparator here. Across absorbed-three-second references, moderate spellF1 changes 0.53623 to 0.53648 and contextual spellF1 declines 0.51752 to 0.51682. The severe alternate-reference gain is +0.00141, with descriptive interval [+0.00072, +0.00203].

Moderate stop recall within one second improves 67.68% to 69.22%, restart recall 76.54% to 77.16%, and dead recall 96.52% to 96.58%. Possession-count passes remain 4/7 in moderate noise. Under the fixed deployment profile, clean/severe/contextual count passes remain 7/7, 2/7 and 4/7 respectively; do not substitute historical nested-profile count numbers.

Decision: retain this as an experimental result; do not promote it or refit an all-seven submission bundle. The primary moderate gain is small and its interval includes zero, the contextual alternate-reference score regresses, and count recovery is unchanged. Severe-noise gains are encouraging but do not establish an overall replacement. No candidate selection used the competition window. Both existing deployment bundles and their predictions remain unchanged. The new boundary candidate has not been evaluated on the two Metrica samples; those samples evaluated the unchanged current model only.

Uncertainty uses 1,000 paired match draws after averaging seeds. The stored conditional possession comparison uses 200 shared draws per fixture/convention, averaged over corruption seeds; only one of seven moderate provider intervals excludes zero. Neither calculation accounts fully for overlapping training sets or repeated architecture development on these fixtures.

## Verification

All 120 unit tests pass. The independent boundary audit passes 458 recorded hash checks, 294 cross-fitting lineage checks, seven exact boundary-model replays and seven exact original-decoder replays on moderate seed 17. Both packaged base/research bundles still pass their source, settings and model-hash checks. The diagnostic also reproduces 70 saved corrected-combined predictions exactly and verifies 70 both-true oracle cases. These are reproducibility checks, not hidden-match quality estimates.

The two Metrica reference/score-report versions contain byte-identical evidence NPZs. Version 2 only corrects undefined subgroup metrics to null; it does not rerun or alter model predictions. No new package versions, competition inference/evaluation, model promotion, Git commit/push or submission occurred in this checkpoint.

## Scope and remaining work

This is adaptive research on seven previously examined full-reference fixtures, plus a first frozen-model check on two additional partial-reference fixtures. It is not nine equivalent independent possession benchmarks. No new full controlled-possession reference set has been obtained.

The TCN challenger, calibrated fusion of real automatic tags, targeted tactical-video recovery, and additional complete cross-provider possession references remain pending. Existing automatic tag-feature extraction is available, but it has not been validated as a predictor or integrated into the final learned pipeline. These steps should be driven by the measured domain/timing gaps, not by a claimed universal benefit of a larger architecture.

## Artifacts and reproduction

Local detailed results are ignored:

- `outputs/stoppage_research/diagnosis_20261102/`
- `outputs/stoppage_research/metrica_transfer_20261102_v2/`
- `outputs/stoppage_research/boundaries_20261102/`
- `data/external/metrica_public/source_manifest.json`

From the project root, the installed environment is sufficient; no new packages were installed:

```powershell
& '.\.venv\Scripts\python.exe' -u solution/tools/diagnose_stoppages.py --output outputs/stoppage_research/diagnosis_repeat
& '.\.venv\Scripts\python.exe' -u solution/tools/fetch_metrica_external.py
& '.\.venv\Scripts\python.exe' -u solution/tools/validate_metrica_transfer.py --output outputs/stoppage_research/metrica_transfer_repeat
& '.\.venv\Scripts\python.exe' -u solution/tools/train_stoppage_boundaries.py --output outputs/stoppage_research/boundaries_20261102
& '.\.venv\Scripts\python.exe' -u solution/tools/audit_stoppage_research.py --output outputs/stoppage_research/boundaries_20261102
& '.\.venv\Scripts\python.exe' -m unittest discover -s solution/tests -q
```

Research runners require the earlier frozen feature/probability caches; they are not standalone deployment commands. Boundary training consumes the recorded original diagnosis folder. Completed runs resume only with matching source/configuration hashes. Downloading requires network access. No commit, push, publication or submission is performed by these commands.
