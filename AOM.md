# AOM — ownership-only temporal possession model

**Candidate:** frozen ownership-only TCN blend  
**Prepared:** 4 November 2026; documentation updated 12 November 2026  
**Code repository:** https://github.com/ra189zor/mtah2027-possession  
**Status:** final ownership-only model selected and code/weights published; competition submission remains pending.

**Runtime release:** https://github.com/ra189zor/mtah2027-possession/releases/tag/aom-owner-v1

## Approach and provided assets

We estimate team possession from automatic tactical tracking using learned player/ball evidence and offline H/A/D temporal decoding. The candidate combines a corrected tree-based system with a temporal convolutional network (TCN). It averages their conditional-home probabilities equally, while retaining the tree system's dead-ball probabilities and switch-event evidence. This choice follows external experiments where joint home/dead fusion weakened stoppage detection on another provider; ownership-only fusion retained useful gap improvements.

The submitted labels are generated entirely by code. Inputs are the supplied tactical x/y player and ball coordinates, team metadata, pitch dimensions and published playing-period boundaries. Speeds are computed from coordinates. Missing observations remain NaN. The pipeline does not consume video pixels, broadcast tracking, homography, Smart Tagging or external human-produced data about the provided fixture. Earlier tag-feature experiments were not included in this learned pipeline. We made no manual tracking corrections or possession annotations on the provided game.

The official public five-minute reference was used earlier for scoped correctness comparisons of older candidates, without fitting or tuning. The TCN and ownership-only choice were developed using external matches. The verified run accessed no public or private possession references and calculates no competition accuracy score.

## Labelled data: source, derivation and amount

All learned components are fitted using **seven external Sportec/DFL matches**: J03WPY, J03WOH, J03WOY, J03WMX, J03WQQ, J03WR9 and J03WN1. Their provider-derived reference contains **1,002,644 graded native frames at 25 Hz**, approximately **11.14 hours**. DFL owns the data; Sportec Solutions collected the TRACAB tracking and supplied BallStatus/BallPossession fields. Individual annotators are not identified in the public data, and we do not claim to have created human labels. BallStatus=0 becomes D; alive ball with ownership=1/2 becomes H/A; unsupported references remain unknown. The reference convention is not assumed identical to the challenge's controlled possession.

The source is [Bassek et al., Scientific Data 12, 195 (2025)](https://doi.org/10.1038/s41597-025-04505-y), [dataset DOI](https://doi.org/10.6084/m9.figshare.28196177), released under CC BY 4.0. We use the [PySport mirror](https://huggingface.co/datasets/pysport/idsse-data) at revision `a715a38dfbaf5f58e431727c2b78d174101a703c`. The adapter converts centred coordinates into metres on the pitch, anonymizes players into team slots, and separates references from inference inputs. Seeded synthetic dropout, jitter, duplicates, false ball tracks and team swaps augment these seven matches; they are not additional independent matches. Ghost supervision comes from generated corruption masks, not human labels of challenge tracking.

Three anonymized [Metrica Sports sample matches](https://github.com/metrica-sports/sample-data) at revision `e706dd506b360d69d9d123d5b8026e7294b13996` supply **evaluation only**, never fitted training samples. The publisher supplies synchronized tracking/events; individual annotators are not named. We derive conservative pass-interior H/A and guarded out-to-explicit-restart D labels. Reference coverage is 36.18%, 42.60% and 46.99% for Games 1–3, with the remaining frames unscored. CARRY in Game 3 supplies exclusions only. The first two games informed diagnosis; the ownership-only choice was frozen before downloading the third. Credit Metrica Sports; no additional dataset licence is asserted.

## Features, architecture and possession decisions

The tree system learns dead-ball status from player motion, team compactness/formation, ball geometry/motion, boundaries and explicit missingness. Its ownership features include distance advantages and past/future gap anchors. Reliability, gap-ownership and boundary specialists adjust ownership/switch evidence. Upstream predictions used to train specialists are cross-fitted; evaluation folds exclude the target match.

The TCN has 85,494 parameters and processes 298 physical feature channels plus missingness indicators at 5 Hz within each half. It uses 48 hidden channels and four residual blocks with dilations 1/2/4/8, two convolutions per block, and training-only normalization. Training runs for a fixed 16 epochs with seed 1117 and no test-driven early stopping. Conditional-home/dead heads and an auxiliary boundary loss are trained; this candidate uses only the conditional-home output. Its final all-seven fit is not scored on its training labels. Network emissions are returned to the native tracking timeline for decoding.

With h equal to the 50/50 ownership blend and d the unchanged dead probability, frame emissions are H=(1−d)h, A=(1−d)(1−h), D=d. A three-state Viterbi decoder independently handles each published half. Fixed stop/restart/team-switch penalties are 40/40/8, with the existing directional event reward 4. Direct H↔A transitions are allowed. No desired possession count, hard minimum possession duration or game-specific handover times are imposed.

Loose balls and contested touches are resolved through learned control evidence, persistence and later observations. Restarts require a decoded D-to-team transition; there is no oracle restart tag. H/A/D cover playing periods, with X outside published periods. This forced coverage does not imply certainty through long gaps. The complete pipeline uses future context and is offline, not a causal live-video system.

## External evaluation and limitations

Seven whole-match excluded-fit evaluations use provider labels and fixed clean/moderate/severe/contextual noise. Noisy cases average seeds 17/43/101 before equally weighting matches. These fixtures were examined in earlier research; correct fold exclusions do not make repeated architecture work an untouched final test.

| External provider spellF1 | Corrected-combined | Ownership-only candidate |
|---|---:|---:|
| Clean | 0.63119 | 0.67483 |
| Moderate | 0.51582 | 0.55523 |
| Severe | 0.44583 | 0.48437 |
| Contextual | 0.49071 | 0.52913 |

In the separately frozen Game 3 check, moderate accuracy on synthetically obscured pass frames improves 84.13% to 87.00%; the paired conditional 95% improvement interval is [+1.40, +4.27] percentage points. Overall labelled-frame accuracy improves 87.96% to 88.86%. Native dead recall is 81.66% to 81.97%, but native overall improvement has uncertainty spanning zero. Ghost-exposed partial accuracy worsens 82.49% to 81.93%.

These selected partial labels cannot establish full-match accuracy or complete spellF1. Bootstrap intervals group labelled event intervals, not independently sampled matches. Unobserved turnovers, unreliable ball detections, omitted ball height, differing label conventions and offline boundary timing remain weaknesses. Alternative longer-possession conventions change comparative results. Video recovery and real tag fusion remain unimplemented. No result here is hidden-match accuracy or a guarantee of qualification.

## Reproduction and artifact verification

Use Python 3.11 and the exact dependency locks. On this PC the pinned CPU environment already exists. For a fresh environment, install CPU PyTorch 2.8.0 from the official CPU index and then the optional lock:

```powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
& '.\.venv\Scripts\python.exe' -m pip install -r solution/requirements-tcn-lock.txt
```

Unpack `AOM_owner_runtime_v1.zip` into a clean working directory, or copy its `models` directories into the matching repository checkout. Dependency binaries and licensed match assets are not in the archive. Point `MTAH_DATA_DIR` at the separately supplied data root with the expected tactical layout, then run from that working directory:

```powershell
$env:MTAH_DATA_DIR = 'D:\path\to\provided-data-root'
python solution/owner_predict.py --challenge --output outputs/AOM.csv
```

Use the pinned interpreter in place of `python` if its environment is not activated. Existing outputs are never overwritten. Source, weights, dependency versions and settings are checked when loading. The sidecar records exact component identities. This step verifies inference from supplied fitted models; a clean raw-data-to-model retraining demonstration is outside its scope. Cross-platform bit equality is untested.

The extracted package reproduced prior external predictions. The development repository passed 143 tests before delivery cleanup; the active delivery suite is a smaller subset. This draft is generated twice from the locally supplied tactical inputs, with byte-identical CSVs, official format acceptance and published-period coverage checks. Access guards block reference and research reads; input/model identities stay unchanged. This is reproducibility evidence, not accuracy scoring.

- CSV SHA-256: `0924f42f6626d7721f6ba25dc6416ca02fa06fd7c53b5855d689cb80e72c91cc`
- Ownership-only manifest SHA-256: `16c698cd96f134ea73a568cc49a3478081f55890d0d1009b5e73b80ee6d2a22c`
- Runtime archive SHA-256: `4373a111611d78e2734bae80897ba650e5109f0cba29facfccafd52e5cf6d6ee`

The selected runtime is publicly available in the linked release. Anonymous download and exact ZIP checksum were verified on 9 November 2026. Registered submission remains pending. Historical draft artifacts remain unchanged in the local audit archive. The [submission instructions](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md) and [rules](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md) were reviewed on 4 November 2026; the report discloses all labelled datasets and uses none from the provided game for fitting.
