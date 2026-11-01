# Tracking-based football possession segmentation

**Team:** {{TEAM_NAME}}  
**Candidate variant:** {{VARIANT}}  
**Code repository:** {{REPOSITORY_URL}}  
**Prepared:** 1 November 2026  
{{DRAFT_NOTICE}}

## Approach and assets

We infer possession from noisy tracking using learned evidence and offline temporal decoding. Two CPU gradient-boosted classifiers estimate P(dead) and P(home | alive); a third predicts no handover, handover to home, or handover to away. A three-state Viterbi decoder combines them. This separates stoppage evidence from team ownership and allows observations after a tracking gap to inform its earlier frames.

The final system reads the provided tactical automatic x/y tracking and metadata for teams, pitch and published halves. It does not use broadcast tracking/video, tactical video pixels, homography or Smart Tagging. Tag features were investigated earlier but excluded from the learned pipeline because equivalent external training tags were unavailable. We did not hand-label or hand-correct the provided match. Its public correctness window was used for earlier baseline checks, not supervised fitting or hyperparameter selection. No private reference or third-party human annotations of that match were used.

## Labelled training data and provenance

All supervised labels come from the seven public Sportec/DFL fixtures J03WPY, J03WOH, J03WOY, J03WMX, J03WQQ, J03WR9 and J03WN1: **1,002,644 graded native frames at 25 Hz**, approximately 11.14 hours. DFL is the data owner; Sportec Solutions collected the data using TRACAB optical tracking. The provider supplies BallStatus/BallPossession; we created no new human annotations and do not claim named individual annotators. The dataset and collection process are described by [Bassek et al., Scientific Data 12, 195 (2025)](https://doi.org/10.1038/s41597-025-04505-y), released with DFL authorization under CC BY 4.0.

References are derived algorithmically: BallStatus=0 maps to D; active ball with BallPossession=1/2 maps to H/A; unrecognized references remain X. The paper describes possession in terms of player control, but we do not assume exact equivalence to the challenge annotation convention. Downloads use the [PySport mirror](https://huggingface.co/datasets/pysport/idsse-data), pinned to revision `a715a38dfbaf5f58e431727c2b78d174101a703c`. Our adapter converts centred metres to corner-origin metres and separates targets from inputs. It does not retain player identities, source ball height Z or synchronized event XML as model inputs. Those last two are source assets, not evidence that the current 2D adapter uses.

Final dead/owner models use 280,720 / 154,818 sampled, corruption-augmented rows. The final event model uses 327,846 rows with cross-fitted upstream evidence: each training fixture's base probabilities come from heads trained on the other six. These are repeated/augmented samples, not independent extra matches. Contextual stress and alternate label conventions are evaluation-only. The deployment fit uses all seven external fixtures; its training-match smoke outputs are not new held-out performance estimates.

## Features, rules and temporal decisions

The dead head has 275 features covering team movement, formations, compactness, ball motion, boundaries and explicit missingness over multi-second windows. Ownership adds 23 channels, including nearby players and past/future geometric anchors, for 298 features. The event head uses 114 physical/probability-context features, with additional shifts out to two seconds. Existing anchors may look farther ahead; the complete method is offline, not a real-time claim. Motion is estimated from same-team spatial correspondence rather than trusting persistent slot IDs.

Ownership is not simply the nearest team. A geometric anchor requires proximity, a distance advantage and consistent evidence; learned posteriors combine it with motion and gap context. During loose balls or ambiguous challenges, Viterbi balances the current team evidence with persistence and future observations. Stoppages are learned from player/ball behavior rather than a stationary-ball rule alone. Restarts are D-to-team transitions chosen by the joint evidence; no referee signal or exact restart tag is assumed. No hard minimum possession duration or desired match possession count is imposed.

Joint frame probabilities are H=(1-d)h, A=(1-d)(1-h), D=d. Negative logarithms are emissions, with independent decoding for each published half. The deployment stop/restart/team-switch costs are 40/40/8, selected in earlier external development. Direct H↔A changes are allowed. H/A/D cover playing periods; X appears only outside published periods. Forced coverage does not imply certainty through long missing-ball gaps.

For event variants, directional handover probabilities can reduce team-switch costs. The reward subtracts strength times positive log-odds excess above probability 0.1, floors individual costs at −4, and enforces a same-frame round-trip cost of at least 2. Negative transition penalties are valid in the time-directed dynamic program. **This candidate uses {{VARIANT_DESCRIPTION}}.** Original emissions and D-related transition costs remain unchanged. All settings and model/source hashes are embedded in the bundle manifest.

## External evaluation and limitations

We used whole-match held-out predictions, seeded clean/moderate/severe corruption, then additional contextual dropout, ghost balls and wrong-team bursts. Event-head training probabilities exclude both the outer fixture and the fixture whose labels train the event model. The two event reward strengths were fixed ablations, not selected using outer outcomes. Earlier architecture research saw these fixtures, so this is adaptive research rather than a new independent final test. Five fixtures involve Düsseldorf, limiting team diversity. No challenge accuracy numbers are reported.

Mean external provider-label spellF1, weighting matches equally and averaging noisy seeds:

| Condition | Baseline | Event2 | Event4 |
|---|---:|---:|---:|
| Clean | 0.6014 | 0.6123 | 0.6205 |
| Moderate | 0.4916 | 0.5006 | 0.5046 |
| Severe | 0.4317 | 0.4368 | 0.4376 |
| Contextual | 0.4718 | 0.4798 | 0.4837 |

Under moderate noise, the approximate ±20% provider-count heuristic passes on 2/7, 2/7 and 4/7 fixtures respectively. Event4 improves both direct-switch precision and recall, but adds false switches; it does not recover every possession. More importantly, absorbing brief opposite-team spells into longer reference possessions changes the conclusion: moderate spellF1 becomes 0.5509 / 0.5449 / 0.5250. This alternative is a sensitivity analysis, not authoritative challenge truth. There is no universally best variant established across conventions.

Other limitations include long gaps with unobserved turnovers, missing ball height in the adapter, imperfect timing/proximity as evidence of control, provider annotation differences and unvalidated tag/video fusion. Conditional possession bootstraps do not capture all match-to-match or research-selection uncertainty. Detailed positive and negative experiments remain in the repository.

## Reproduction and verification

The public runtime archive contains source, pinned dependencies, fitted model payload, attribution and checksums, but no match data or frame-level predictions. The organizers supply their licensed data separately. Use Python 3.11 and the matching source/model versions:

```text
python -m pip install -r solution/requirements-lock.txt
python solution/verify_release.py
python solution/predict.py --bundle models/possession-v1 --challenge --variant {{VARIANT}} --output outputs/{{TEAM_STEM}}.csv --quiet
```

Set `MTAH_DATA_DIR` to the local provided dataset root when it is not in the provided repository's default data directory. The CSV uses the supplied tactical timeline and is checked by the official format validator. It is algorithm-generated with no manual output edits. The accompanying candidate manifest records its exact SHA-256 and variant.

All **98 tests** pass. A fresh isolated environment reproduces twelve held-out raw-input CSVs exactly; malformed inputs, extreme missingness and access guards pass. The input-only challenge path passes format/repeatability checks without reference access or quality scoring. External worker measurements on this Windows x64 host are approximately 22–24 s and 761–766 MiB peak working set; cross-platform bit equality has not been tested. The final verification found only a validator type-acceptance defect, fixed without changing learned weights, settings or inference mathematics. See the repository verification/provenance records for exact scope.
