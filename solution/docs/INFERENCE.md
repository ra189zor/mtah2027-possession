# One-command learned inference

Completed 1 November 2026, including final all-seven external fitting and guarded raw-input verification. The pipeline loads a model bundle, extracts tracking features, predicts ownership/dead/event evidence, decodes H/A/D/X and writes an officially format-validated CSV. It does not import research runners, load targets or calculate quality scores. All 95 tests pass; nine held-out raw-input runs reproduce earlier prediction CSVs byte for byte.

This is end-to-end from supplied tracking to prediction CSV. It is not a graphical application or a raw-video ball/player detector. Video and Smart Tagging are not consumed by this learned pipeline; the external training fixtures did not validate an equivalent tag branch.

## Run it now

The existing `.venv` and local `models/possession-v1/` bundle are ready. From the project root in PowerShell, run this external example with the explicitly chosen stronger event variant:

```powershell
& '.\.venv\Scripts\python.exe' solution/predict.py --bundle models/possession-v1 --input data/external/sportec/J03WN1/prepared/inputs.npz --variant event4 --output outputs/my_prediction.csv
```

The result is `outputs/my_prediction.csv` and `outputs/my_prediction.run.json`. The CSV has `frame,label`, zero-based native frame numbers and H/A/D/X labels. The sidecar records model/runtime/input hashes, training membership, timing, variant and successful format validation. It is not a scored evaluation report. Existing outputs are protected; choose a new output name or explicitly add `--overwrite`.

| Variant | Behavior | Interpretation |
|---|---|---|
| `baseline` | Original learned dead/owner heads and fixed 40/40/8 Viterbi | Retained reference; stronger under the absorbed-short-spell convention in the recorded experiments |
| `event2` | Add the learned directional switch head at strength 2 | Smaller provider-label gains and smaller alternate-label regression |
| `event4` | Add it at strength 4 | Larger provider-label gains/count recovery and larger alternate-label regression |

`--variant` is required. No convention-dependent option is silently made the default. Recorded quality comparisons are in `SWITCH_EVENTS.md`; they concern disjoint outer models, not fresh held-out performance of the all-seven deployment fit. The external example above uses a training fixture for the deployment bundle and is an inference demonstration only.

All paths can be absolute, so inference also works from an unrelated working folder. Run `solution/predict.py --help` for flags. `--quiet` prints output paths without row/timing numbers.

## Local challenge tracking

The CLI also has an input-only tactical adapter:

```powershell
& '.\.venv\Scripts\python.exe' solution/predict.py --bundle models/possession-v1 --challenge --variant baseline --output outputs/submission_candidate.csv --quiet
```

This uses the provided `games.py` metadata/path/half tables and the existing verified raw-tracking parser. Its output timeline follows the supplied tactical source contract. The adapter opens local metadata and raw tracking only; it does not open reference possession files, video, Smart Tagging or score the candidate. `MTAH_DATA_DIR`, when set, controls the provided asset paths. Full competition-input inference is deferred to final verification; it has not been executed in this packaging checkpoint. No competition-data numbers are reported here.

## Tracking-only NPZ contract

Exactly these six fields are accepted; adding `labels`, targets or other fields is rejected:

| Field | Shape / meaning |
|---|---|
| `players_xy` | N × slots × 2 numeric coordinates in corner-origin metres; NaN means missing |
| `ball_xy` | N × 2, same coordinate/missingness convention |
| `team_of_slot` | One home=0 or away=1 value per anonymous player slot; both teams must have slots |
| `period` | N values: published first half=1, second half=2, outside periods=0 |
| `pitch_m` | Positive length/width in metres |
| `fps` | Scalar exactly 25 |

Each half must form one ordered published block. Missing observations inside halves remain NaN, rather than creating artificial period gaps. Do not crop or resample when producing a timeline-bound submission. The runtime preserves N rows, emits H/A/D throughout playing periods and X outside them, and refuses infinite coordinates or incompatible schemas. Real out-of-bounds coordinates are allowed because they carry dead-ball evidence. Periods are input metadata, not a true alive/dead mask.

## Frozen bundle and fitting

The deployable payload is `manifest.json` plus `models.joblib` in one directory, used alongside the matching repository source. Extra `build.json` and `base_heads.joblib` files are training/resume records, not inference dependencies. Runtime verifies the model-file hash before joblib loading, embedded settings against the fitted objects, model class/feature schemas, exact numerical-package versions and source hashes. Joblib bundles are trusted local code artifacts, not an untrusted interchange format. Runtime Python must match the recorded major/minor version; use the pinned requirements on a fresh machine.

The final bundle is `models/possession-v1/`, kind `final_all_external_matches`:

- Original full-feature/frame-weighted dead head: 275 features, 280,720 sampled/augmented training rows.
- Original full-feature/spell-weighted owner head: 298 features, 154,818 sampled/augmented alive rows.
- Three-class switch-event head: 114 features, 327,846 sampled/augmented rows.
- Final switch training uses 49 existing six-match out-of-fold probability cases: each training fixture is excluded from its upstream heads. Inference upstream heads train on all seven fixtures.
- The fixed deployment decoder remains the original development-selected 40/40/8 profile. The globally deployed profile was not chosen from opened outer-match results.

These are repeated corruption/training samples, not independent real frames or matches. Final fitting uses only the seven public external fixtures and the unchanged clean/moderate/severe training regimes. Contextual stress, alternate references and competition data do not enter these fits. Hyperparameters/features were already selected in earlier research. No new quality estimate is attached to fitting on all available external matches.

Rebuild on a machine containing the recorded audited research artifacts with:

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/build_inference_bundle.py --output models/possession-v1
```

Use a new directory for changed sources/settings. `--resume` verifies an identical existing build, or resumes its checked base-head checkpoint. The training builder requires earlier prepared feature/target caches, out-of-fold probabilities and model provenance reports. It is not yet a clean-room retraining orchestrator from downloads alone. Inference itself needs none of those caches/reports or target files; moving just the payload and matching repository is sufficient. Model files are ignored by Git, so a code-only clone does not include the local bundle; supply it separately or rebuild it. Final distribution/provenance packaging remains a later checkpoint.

The separate `models/heldout-J03WN1-v1/` bundle is for verification only. It exports the original six-match heads/event model excluding Bochum–Leverkusen, with that outer fold's frozen decoder profile. It is not the final all-seven deployment model. Reproduce it with:

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/build_inference_bundle.py --output models/heldout-J03WN1-v1 --held-out J03WN1
```

Dataset revision, CC-BY-4.0 attribution, all model-training memberships, feature settings, numerical versions, source hashes, training-cache hashes and event probability lineage are recorded in the bundle manifest and `EXTERNAL_DATA_SOURCES.md`. Bundle settings are embedded, so changing a repository JSON does not silently change already packaged behavior; rebuilding is explicit.

## Verification commands

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s solution/tests -q
& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/verify_inference_pipeline.py --heldout-bundle models/heldout-J03WN1-v1 --final-bundle models/possession-v1 --output outputs/inference_packaging/verification_20261101
```

The verifier uses raw external inputs, recreates clean/moderate/contextual seed-17 corruptions, and runs all three variants in fresh guarded processes from an unrelated working directory. Expected CSVs are earlier predictions, not reference possession labels. Final-bundle smoke runs are explicitly in-sample behavioral checks. Guarded inference raises on reference-artifact or research-runner access. The report separates exact held-out reproduction from final-model run/repeatability checks and does not calculate match-quality scores.

## Completed packaging checks

- Nine guarded held-out runs cover all three variants on Bochum–Leverkusen clean, moderate seed 17 and contextual seed 17. All nine CSV hashes exactly match the corresponding earlier validation predictions. This checks fresh raw tracking, feature extraction, model loading and temporal decoding together, rather than only decoding cached probabilities.
- Four guarded final-bundle runs exercise baseline, event2 and event4, with two separate event4 executions producing byte-identical CSVs. These use a fixture included in final fitting and establish runtime behavior only, not new quality performance.
- A further guarded event4 run uses a copied bundle containing exactly `manifest.json` and `models.joblib`. It reproduces the final-bundle CSV byte for byte from an unrelated working directory, proving the base-head checkpoint/build log and research outputs are not inference requirements.
- All 14 processes finish without reference-artifact or research-runner access. Input copies retain their hashes. Output sidecars record `reference_arrays_read=false` and `scored=false`; the supplied validator accepts all output schemas.
- All 95 unit tests and dependency checking pass. Runtime-specific tests cover tracking-only loading with no sibling target file, label-field rejection, exact frame-rate/period guards, model hash verification before deserialization, settings/runtime tampering, CSV formatting and output replacement protection.

The held-out instrumentation records about 12.6–13.8 seconds for feature/model/decoder stages on this external fixture, under two concurrent subprocesses. It excludes interpreter startup/model loading/output I/O and is not a hardware-independent or competition-input runtime guarantee. The final three-head model payload is approximately 2.1 MB; its manifest records exact bytes/hashes.

Local records are `outputs/inference_packaging/verification_20261101/report.json`, `outputs/inference_packaging/portability_report.json`, per-run guard/sidecar files, and each model bundle's manifest/build record. Model weights, copied inputs, predictions and verification artifacts remain ignored under the single root policy. No new package/download, competition-input inference/evaluation, commit, push or submission occurred.

This completes the runnable-pipeline checkpoint. Broader final verification, clean reproduction/distribution checks and the submission/technical-document package remain the next two main steps. Label-convention tradeoffs, remaining count failures, offline lookahead and the lack of a new independent quality score for all-seven fits remain explicit limitations.
