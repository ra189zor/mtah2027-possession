# Integrating the external review — 28 October 2026

This updates the plan; it does not execute Step 3 or later implementation steps. The architecture remains evidence fusion plus structured temporal decoding. The main change is a stronger external-match validation benchmark.

## Current rules supersede the downloaded snapshot

The current [official rules](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md), checked on 28 October, prohibit human labels on the provided match for training or validation. The 21 October local snapshot allowed small samples. No such labels have been created. Validation now uses other fixtures; the public window remains a pipeline check.

## What to adopt

1. **External validation first.** Inspect one Sportec/DFL match before downloading a larger bundle. The [dataset paper](https://www.nature.com/articles/s41597-025-04505-y) documents seven matches, possession and ball-status fields. [floodlight's DFL reader](https://floodlight.readthedocs.io/en/latest/modules/io/dfl.html) exposes these codes. Audit definitions, unknown values, sample rate, time alignment and licences before treating them as references. D maps from an inactive ball; H/A map from provider team ownership only when alive. Unknown values stay ungraded, rather than becoming guessed labels.
2. **Split by whole match.** Assign development versus sealed holdout fixtures before experimentation. With sufficient matches, use grouped cross-validation on the development portion; retain a genuinely untouched final match. If the initial benchmark is too small, report that limitation rather than advertising cross-match reliability. Our official match never enters supervised fitting.
3. **Separate inputs from answers.** Possession and ball-status codes are target labels, not model features or inputs to corruption. Strip stable player identity from the model-facing representation. Normalize coordinate conventions and compute geometry in metres. Do not upsample 10 Hz data and claim new 25 Hz timing evidence.
4. **Create several corruption regimes.** Preserve clean reference labels, then apply seeded burst ball dropout, player dropout, duplicated observations, plausible coordinate noise and within-team tracklet fragmentation to model inputs. Match observed gap-length and player-count distributions where possible, not merely 41% missing ball. Use a separate tactical-only stress analysis on our provided automatic data without creating answer labels. Evaluate clean, moderate and severe corruption, plus natural missingness where available.
5. **Use the exact scorer.** Route all external predictions through the official metric implementation with documented frame-rate handling. Score match/half boundaries separately. Use match/block-aware paired comparisons; possession resampling alone misses between-match variability. Log per-IoU spellF1, macroF1, D errors, false/missed transitions, fragmentation and latency.
6. **Make stoppages explicit.** Compare separate alive/dead estimation against an integrated three-state model. Smart Tagging team priors, restart cues and geometry must be individually ablated. Do not equate every stationary ball or low team speed with a stoppage. Set-piece windows are not exact restart frames.
7. **Use bidirectional evidence.** Feed team/dead-state evidence into a three-state Viterbi decoder. Compare constant switch penalties against evidence-dependent penalties. Later observations may confirm a switch and revise the earlier boundary automatically; do not shift every transition by an arbitrary delay. Short real possessions must remain possible.
8. **Try spell-balanced training only as an ablation.** On external training labels, compare ordinary frame loss with weights normalized by possession length, with caps so very short/noisy spells do not dominate. Keep dead-ball learning explicit. Neither weighting scheme directly optimizes the discrete spellF1 matching; model selection still needs external interval scores.

## Dataset priority and limits

| Source | Priority | Reason / gate |
|---|---|---|
| Sportec Open DFL | First | Seven matches with possession and ball-state fields. Confirm parser mapping and label convention on one match before expansion. |
| Metrica public samples | Second | Convenient format and event/tracking examples; event labels require careful conversion and are not automatically frame-level H/A/D truth. |
| SkillCorner official open data | Optional | Different camera conditions; [official repository](https://github.com/SkillCorner/opendata) describes broadcast tracking at 10 fps. Audit label provenance and timing before using it as a reference. |

An external source is a useful comparison environment, not proof of accuracy on the hidden corrected match. Similar baseline scores on two datasets do not establish that their error distributions or model rankings are equivalent. Synthetic corruption cannot recreate every systematic tracking failure.

## Claims not adopted as evidence

- The review's competitor repository names, 0.4941 public score, LightGBM/BiLSTM numbers and degraded-data experiment lack actual links in the pasted text and were not independently verified here.
- Matching 17 predicted spells to 17 true public spells is not evidence of wrongdoing or sufficient proof of overfitting.
- The approximate 0.04 noise band in the supplied evaluation documentation is sample-dependent, not a universal decision threshold.
- X at an uncertain transition is a decision problem: preserving an old state may be wrong, but an uninformed forced switch may also be wrong. Compare decoding policies; normally emit the chosen H/A/D state on graded frames.
- Later-stage product/pitch planning stays out of the user's current scope.

## Execution order and revised dates

As of 28 October, Steps 1–2 are complete and Step 3 has not begun. The original calendar has slipped, so remove optional experiments before sacrificing validation or packaging.

| Window | Work |
|---|---|
| 28–29 October | Step 3: regenerate official baselines, compare every label and record runtime/hash. |
| 29 October–2 November | Step 4: external-data adapter, one-match label audit, grouped split and initial corruption benchmark. Expand only after this works. |
| 2–6 November | Features, deterministic fallback, alive/dead model and temporal decoder. |
| 6–10 November | Small prespecified ablations and cross-match comparisons; neural/video work only if time and evidence justify it. |
| 10–12 November | Freeze candidate, evaluate sealed external holdout, select final configuration. |
| 12–15 November | Clean reproduction, dependency/model provenance, technical document and exact-file checks. |
| 16 November | Internal submission target; user handles publication and submission authorization. |
| 17 November | Failure-recovery buffer. |

These are proposed work windows, not authorization to skip numbered-step approval. Next authorized action still requires the user's explicit instruction to start Step 3. No external datasets or additional packages were downloaded for this plan update.
