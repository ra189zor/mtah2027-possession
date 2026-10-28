# Step 2 data verification — 21 October 2026

Status: **tactical inputs ready**. Optional broadcast input has a documented frame-count discrepancy and is not approved for direct fusion.

## Canonical inputs

The official `games.asset_paths()` defaults now resolve to nine files inside the supplied source snapshot's `data/` directory:

| Source | Assets |
|---|---|
| tactical | video, ATD metadata, ATD raw data, homography, Smart Tagging |
| broadcast | video, ATD metadata, ATD raw data, homography |

No environment variable or change to `games.py` is required. Seven files were extracted from the original ZIPs through a fixed filename allowlist and CRC-checked reads. Repeated preparation checks existing content and refuses to replace different data. Tracking SHA-256 hashes match the original archive audit.

The canonical video names are hard links to the downloaded suffixed filenames, preserving originals without a second video copy. Do not edit either video alias: hard links share file contents.

## Executed checks

| Check | Tactical | Broadcast |
|---|---:|---:|
| Tracking records | 199319 | 164488 |
| IDs exactly 1 through count | pass | pass |
| Track slots and all coordinate/speed field counts | 46; pass | 51; pass |
| Ball present | 113488 | 83121 |
| Homography rows and exact zero-based indexing | 199319; pass | 164488; pass |
| Valid homographies | 199319 | 108248 |
| Finite, nonsingular valid matrices | pass | pass |
| Decoder-reported frames | 199319 | 164487 |
| Video rate / dimensions | 25 fps / 1920×1080 | 25 fps / 1920×1080 |

Tracking checks scan every raw record, verify metadata team assignments, reject infinite coordinates and preserve missing observations. Homography checks include validity flags, matrix structure and valid reprojection errors.

Smart Tagging: **638 instances**, with both published kickoff frames inside their corresponding tagged windows. Tags are weak automatic evidence, not ground truth.

OpenCV's FFmpeg backend successfully decoded tactical frame indices 0, 11185, 18684, 90651, 99659, 110808, 113742, 190197 and 199318. Broadcast probes 0, 82243 and 164486 also decoded. No raw frame images were exported or sent to a hosted model.

These are selected-frame decode checks, not a full sequential video decode or a guarantee that every intermediate frame is error-free. They do not establish synchronization between the two feeds.

## Broadcast discrepancy

The container audit and decoder agree on 164487 video frames, while the supplied tracking/homography contain 164488 rows. The location/cause of the difference is not established. No padding, shifting or trimming was applied. Use tactical-only inputs until any future broadcast adapter independently proves alignment.

## Repository and environment

The user requested one `.gitignore` at the project root. Nested source/solution ignore files were removed and their relevant exclusions consolidated. The official public ground-truth CSV, supplied baselines and source `data/README.md` remain eligible for tracking; actual assets and generated reports do not.

The earlier solution Git metadata is preserved at `.local-git/solution.git`, which is ignored. Root Git metadata was present initially during this step and then was no longer present; no repository was reinitialized, and no commit or push was made during Step 2. The user owns those actions.

Installed `opencv-python-headless==5.0.0.93` for local video checks. All environment packages are pinned in `solution/requirements-lock.txt`; `pip check` passed.

Local-only detailed artifacts: `research/local_data_manifest.json` and `research/step2_data_check.json`. They are excluded by the root ignore policy.

## Next step

Step 3: regenerate the supplied baselines into a separate ignored output directory, compare labels with the official reference CSVs and establish runtime/reproducibility. Await explicit user instruction.
