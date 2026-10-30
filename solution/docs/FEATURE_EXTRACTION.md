# Step 5: feature extraction

Completed 30 October 2026. The pipeline now extracts 52 channels per frame without opening possession reference-label arrays. This step prepares model inputs; it does not predict possession, train a model or demonstrate a score improvement.

## Reproduce

Run from the project root using the existing environment and prepared inputs:

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s solution/tests -v
& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/extract_features.py
```

Use `--source challenge` or `--source external` for a subset. Each run creates a fresh ignored `outputs/features/<UTC timestamp>/` directory. No new dependencies are needed.

## Channel contract

Each NPZ contains `values` (frames by channels, float32), `valid` (same shape, boolean), and `names` (ordered channel names). Invalid entries are stored as zero only for serialization: consumers must use the validity mask. All channels are invalid outside published playing periods; temporal calculations reset at half boundaries.

| Group | Channels | Meaning |
|---|---:|---|
| Ball | 11 | Observation flag, time since previous observation, time until next observation, trailing coverage, XY, inside-pitch flag, signed distance to goal/touch lines, speed, rejected-motion flag |
| Each team | 14 | Player count, centroid XY, RMS spread, XY spans, motion observation count, median speed, moving fraction, nearest ball distance, count near ball, nearest player's speed, ball/player speed difference, closing speed |
| Team comparison | 1 | Away nearest distance minus home nearest distance |
| Automatic tags | 12 | Source availability, overlap count, home/away evidence counts, restart-window and team counts, goal-window count, unknown-team count, home evidence fraction, conflict flag |

Coordinates and distances are in metres, speeds in metres/second, and observation intervals in seconds. Counts and flags retain their natural units. Signed boundary distances are negative outside the pitch. Positive closing speed means the ball and nearest matched player approach each other. The future-observation channel is explicitly named `ball_next_observation_s_offline`; it cannot be used unchanged in a live streaming system.

Challenge tracking is converted from normalized coordinates using published pitch dimensions; source frame 1 maps to output frame 0. Published half boundaries come from the official game configuration. External arrays already use corner-origin metres. Inputs have a strict schema that excludes target labels.

## Motion and missingness

`solution/configs/features.json` fixes the motion lag at 5 frames (0.2 seconds at the validated 25 Hz), trailing ball coverage at 25 frames, player speed ceiling at 12 m/s, ball speed ceiling at 80 m/s, moving threshold at 1 m/s and near-ball radius at 2 metres. These are engineering settings, not parameters fitted to the supplied match's labels.

Player slots are not assumed to preserve identity. Motion pairs unique mutual nearest spatial observations within the same team across the lag. Ties, missing endpoints and implausible displacement are rejected. The method does not prove identity: close crossings, substitutions and gaps between endpoints can still yield an incorrect correspondence. Team motion summaries and valid-match counts expose the available evidence.

Ball velocity requires observations throughout the lag interval. Missing positions and rejected motion stay unknown. Missing-ball time is a feature, never a direct dead-ball label. Leading/trailing unknown intervals are not backfilled with fabricated positions.

## Automatic tags

Attack-phase tags vote weakly for the named team; defense-phase tags vote for its opponent. Overlaps accumulate, and opposing evidence sets a conflict flag. Set-piece intervals are restart windows, not precise restart instants. Unrecognized team names retain unknown-team evidence. Intervals are half-open and clipped to the published half.

External fixtures lack equivalent automatic tags. Their `tag_available` is a known zero and other tag channels are invalid. An available but empty source instead has known zero counts. External tracking checks therefore do not validate the accuracy or usefulness of the challenge tag branch.

## Verified results

Accepted local audit: `outputs/features/20261030T133908928645Z/verification.json`.

| Match/scenario | Frames | Channels | Values + validity MiB |
|---|---:|---:|---:|
| BAR-ATH automatic tags | 199319 | 52 | 49.42 |
| J03WPY clean | 146212 | 52 | 36.25 |
| J03WPY moderate, seed 17 | 146212 | 52 | 36.25 |
| J03WPY severe, seed 17 | 146212 | 52 | 36.25 |
| J03WQQ clean | 142346 | 52 | 35.30 |

On all five caches, extraction repeated exactly, input fingerprints remained unchanged, validity masks survived within-team slot reversal exactly, and feature values survived reversal within `rtol=1e-5, atol=2e-4`. All serialized values were finite float32 and no features were valid outside playing periods. These memory figures describe uncompressed values plus masks, not peak process memory or compressed file size. Per-match wall times, including reading, three extraction passes per scenario and compressed writes, were 16.8 s, 35.6 s and 11.4 s respectively on this machine.

All 24 tests passed: 11 existing external-validation tests and 13 new tests covering analytic geometry/motion, slot changes, missingness, half boundaries, implausible motion, duplicate positions, target rejection, tag semantics and challenge coordinate/frame conversion.

External input SHA-256 values are checked against their preparation manifests. The feature reader opens only `inputs.npz`, never `targets.npz`; challenge inputs use tracking, metadata, published timing and automatic tags. Sealed J03WN1 was neither downloaded nor processed. Feature caches and detailed reports remain ignored. No human annotation, fitting, training, public-window scoring, commit or push occurred in Step 5.

## Next

Step 6 will implement the first geometry/persistence/stoppage solver. Feature validity must remain part of its interface, and improvement must be measured through the existing external benchmark. Broader development-match coverage is still needed before extensive model selection.
