# Reliability research results — 1 November 2026

All numbers below are external-data estimates. Seven previously examined fixtures, clean once and three noisy seeds each; 70 inputs, six candidates and four reference conventions produce 1,680 metric records. This is adaptive research, not a new independent test or challenge score. Benchmark decoder profiles retain their previous inner-fold selections; the separate deployment bundle retains the earlier global 40/40/8 profile.

The main tables below use the corrected clean/moderate/severe quality detector and fully retrained gap/boundary specialists. The first specialist experiment is preserved separately; its moderate provider combined score was 0.51667. The corrected experiment is a subsequent adaptive comparison, not an independent confirmation.

## provider

| Condition | baseline | event4 | quality | gap | boundary | combined |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.60143 | 0.62054 | 0.62070 | 0.62054 | 0.63079 | 0.63119 |
| moderate | 0.49160 | 0.50459 | 0.50511 | 0.50923 | 0.51152 | 0.51590 |
| severe | 0.43173 | 0.43763 | 0.43759 | 0.44201 | 0.44227 | 0.44617 |
| contextual | 0.47180 | 0.48373 | 0.48443 | 0.48694 | 0.48815 | 0.49190 |

## absorbed_1s

| Condition | baseline | event4 | quality | gap | boundary | combined |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.63836 | 0.64429 | 0.64469 | 0.64429 | 0.65402 | 0.65446 |
| moderate | 0.52339 | 0.52829 | 0.52892 | 0.53382 | 0.53561 | 0.54104 |
| severe | 0.46372 | 0.46509 | 0.46505 | 0.47030 | 0.46934 | 0.47413 |
| contextual | 0.50235 | 0.50796 | 0.50882 | 0.51173 | 0.51253 | 0.51688 |

## absorbed_3s

| Condition | baseline | event4 | quality | gap | boundary | combined |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.64970 | 0.60519 | 0.60570 | 0.60519 | 0.60758 | 0.60808 |
| moderate | 0.55088 | 0.52502 | 0.52571 | 0.53123 | 0.53037 | 0.53641 |
| severe | 0.49935 | 0.48497 | 0.48493 | 0.49297 | 0.48549 | 0.49422 |
| contextual | 0.53118 | 0.51051 | 0.51167 | 0.51574 | 0.51362 | 0.51937 |

## absorbed_5s

| Condition | baseline | event4 | quality | gap | boundary | combined |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.59498 | 0.53872 | 0.53961 | 0.53872 | 0.54160 | 0.54194 |
| moderate | 0.51866 | 0.48272 | 0.48341 | 0.48959 | 0.48839 | 0.49526 |
| severe | 0.47565 | 0.45395 | 0.45386 | 0.46258 | 0.45401 | 0.46341 |
| contextual | 0.50069 | 0.47224 | 0.47339 | 0.47889 | 0.47580 | 0.48289 |

## Moderate provider changes versus event4

| Candidate | Change | Descriptive match bootstrap 95% interval | Improved fixtures | Count passes |
|---|---:|---|---:|---:|
| quality | +0.00051 | [-0.00003, +0.00118] | 5/7 | 4/7 |
| gap | +0.00463 | [+0.00331, +0.00580] | 7/7 | 4/7 |
| boundary | +0.00692 | [+0.00397, +0.01057] | 7/7 | 4/7 |
| combined | +0.01131 | [+0.00768, +0.01541] | 7/7 | 4/7 |

The count gate is a diagnostic ±20% heuristic. Bootstrap intervals do not account for all researcher adaptation or shared training sets.

## Missing-ball ownership and boundaries

| Candidate | Moderate missing-alive frame accuracy | Direct-switch precision | Direct-switch recall | Spurious direct switches |
|---|---:|---:|---:|---:|
| baseline | 0.73564 | 0.27836 | 0.18364 | 158.14 |
| event4 | 0.73558 | 0.32714 | 0.25618 | 176.00 |
| quality | 0.73558 | 0.32785 | 0.25617 | 175.38 |
| gap | 0.74659 | 0.33439 | 0.25780 | 171.48 |
| boundary | 0.73593 | 0.34492 | 0.26612 | 168.81 |
| combined | 0.74677 | 0.35142 | 0.26657 | 164.38 |

## Quality detector and clean-domain follow-up

The first quality model trained on moderate/severe synthetic corruption only. Its clean false alarms motivated a separately recorded adaptive follow-up adding clean-source corruption with seed 829. The follow-up changes only quality training and evaluates its quality-only decoder integration; it does not retroactively change the first specialist experiment.

| Quality training | Clean false-positive rate at 0.5 | Contextual average precision | Contextual ghost recall at 0.5 | Contextual false-positive rate |
|---|---:|---:|---:|---:|
| moderate/severe | 0.45420 | 0.81290 | 0.65883 | 0.00143 |
| clean/moderate/severe | 0.00508 | 0.81947 | 0.63533 | 0.00189 |

| Condition | Corrected quality-only provider spellF1 | Change versus event4 | 95% descriptive interval |
|---|---:|---:|---|
| clean | 0.62070 | +0.00016 | [-0.00114, +0.00123] |
| moderate | 0.50511 | +0.00051 | [-0.00003, +0.00118] |
| severe | 0.43759 | -0.00004 | [-0.00025, +0.00011] |
| contextual | 0.48443 | +0.00070 | [+0.00025, +0.00118] |

Quality metrics use known synthetic ghost masks, not human-labelled real tracking errors. Broader validation here means all seven fixtures, repeated corruptions and four conventions; no additional independent real fixtures were acquired. No variant is automatically promoted.

## Verification

All 108 tests pass. Independent audit passes 601 hash checks, 294 cross-fitting traces and 84 exact held-out prediction replays. Original deployment bundles retain their integrity.

The failed initial run is retained separately. The completed v2 run excludes columns that are entirely unobserved in each specialist training fold, avoiding a weighted histogram-binning edge case without consulting held-out values.

Main experiment: `outputs/reliability_research/run_20261101_clean`. Initial ablations: `outputs/reliability_research/run_20261101_v2/`. Quality-only follow-up: `outputs/reliability_research/quality_clean_domain_20261101/`. Methods: `RELIABILITY_RESEARCH.md`.
