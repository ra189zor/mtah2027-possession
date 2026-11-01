# Ball-cleaning screening — 1 November 2026

User requested renewed accuracy work, including noisy-data cleaning. Tested one fixed, conservative isolated-spike repair on external J03WN1, using the six-match held-out bundle and event4. No competition inputs, production edits or threshold search. This previously examined fixture and seed 17 constitute adaptive screening, not independent or seven-match validation.

Repair a visible central ball point only when both one-frame velocities exceed 45 m/s, the two neighbours imply less than 20 m/s, and all three observations are finite within the same playing period. Replace the central point with its neighbours' midpoint. Missingness remains identical; player coordinates are untouched. These are experimental engineering thresholds, not inferred physical limits or challenge-tuned settings.

| Condition | Repaired frames | Original spellF1 | Cleaned spellF1 |
|---|---:|---:|---:|
| Clean | 0 | 0.570758 | 0.570758 |
| Moderate | 0 | 0.432090 | 0.432090 |
| Severe | 23 | 0.354167 | 0.354167 |
| Contextual | 0 | 0.403248 | 0.403248 |

Alternate absorbed-label scores also remain unchanged. No accuracy gain demonstrated; do not promote this cleaner. The result does not establish that all cleaning methods fail. Longer ghost tracks, absent-ball gaps and ambiguous controlled-possession boundaries are outside this rule's scope.

Protocol was saved before inference; model excludes J03WN1; input fingerprints and missingness preservation checked. Eight full external inference runs completed. Detailed metrics: `outputs/ball_cleaning/pilot_20261101/report.json`; reproduction script: `solution/tools/pilot_ball_cleaning.py` (requires a new output directory for a repeat).

Next research priorities: a tracking-quality estimator trained with degraded external inputs; gap-aware uncertainty rather than aggressive interpolation; richer ball-height/control evidence with matching training/deployment availability; and broader independently labelled external matches to reduce convention uncertainty. Evaluate fixed ablations across fixtures and noise seeds before promotion. Repeated experimentation on the same seven fixtures is adaptive and should not be presented as a fresh independent estimate. The existing AOM package remains the verified deployment candidate.
