# Final verification

Completed 1 November 2026. A fresh isolated Python environment and minimal clean-source snapshot reproduce all frozen external prediction checks. Missing-data and rejection tests pass. The local challenge-input adapter produces structurally valid, byte-repeatable outputs for each explicit variant, without reference-label access or quality scoring. All 98 tests pass in both the existing and fresh environments. The submission/technical-document package is the remaining main step.

This verifies execution, integrity and output contracts. It does not resolve label-convention tradeoffs, prove competition accuracy, or provide a new independent quality estimate for the all-seven deployment fits. All reported match/resource numbers below are external-data numbers only.

## Clean-source and dependency reproduction

The isolated environment is `outputs/final_verification/run_20261101/fresh_env/`. It uses Python 3.11.9 with `include-system-site-packages=false`, disabled user-site imports and no inherited PYTHONPATH. Its NumPy import resolves inside that environment, and `pip check` passes.

All 12 pinned dependencies were recovered from existing local pip-cache wheels. Their package metadata/compatible wheel tags and SHA-256 values were recorded, then installed with `--no-index --require-hashes`. The install report points to the recorded local wheels; no download or network access was required. Versions remain those in `requirements-lock.txt`, with no dependency updates.

The source snapshot contains only runtime modules, the provided format checker/metadata module, package initializer, requirements, licence and checkout policy, plus the two-file final/held-out bundles. No research runners, training caches, reference arrays, `.git` directory or original dataset directory are copied. Inference runs from an unrelated directory under file-access guards. Input-only challenge tests are allowed to open only the original tactical metadata/raw-tracking files outside this snapshot.

The snapshot is a copy of current candidate working-tree bytes, not a fresh committed Git clone: the validator fix and verification tools are still for the user to commit. A new root `.gitattributes` preserves LF source bytes on future checkout, including the supplied runtime modules; `git check-attr` confirms the policy. This avoids source-hash mismatches caused solely by checkout line-ending conversion. The single root `.gitignore` remains unchanged.

## Defect found and fixed

The original inference validator accepted complex-valued coordinates, despite tracking coordinates being real metres. It now rejects complex coordinates/pitch, non-real or malformed FPS, and reports malformed field types/shapes as validation errors. Regression tests cover those cases. Existing output protection, failed-format staging cleanup and dependency mismatch handling are also tested.

This is an input-validation change only. The independent audit compares old/new runtime ASTs with `validate_tracking` removed and finds them identical. All learned feature functions, model parameters, decoder logic, settings and model weight bytes remain unchanged. Both model manifests are now runtime revision 2 and retain their original model/settings hashes. Exact prior manifests/model payloads/runtime source snapshots were archived under `prior_final_bundle/` and `prior_heldout_bundle/` in the verification directory, with a recorded validator-only refresh history. Existing user commands still use `models/possession-v1/` and `models/heldout-J03WN1-v1/`.

## Acceptance checks

| Check | Result |
|---|---|
| Unit suite | 98 pass in each environment |
| External raw-input reproduction | 12/12 exact frozen CSVs: all three variants on clean/moderate/severe/contextual Bochum–Leverkusen, noisy seed 17 |
| Extreme missingness | All-ball-missing, all-players-missing, both-missing across all variants; repeat check passes |
| Invalid inputs/artifacts | 12 expected rejections, without publishing predictions or replacing existing output |
| Access guards | Dummy reference/research probes blocked before access; ordinary inference has no forbidden access attempts |
| Local challenge-input contract | Standalone format checker accepts outputs; each variant repeats byte for byte |
| Reference scoring | Not performed |

The held-out bundle excludes Bochum–Leverkusen from both base-head and event-head training. Fresh external inputs are regenerated with the original frozen corruption code, and their fingerprints agree with the older experiment. The CSV comparisons use previous model predictions, never target possession files. Synthetic missing-data checks only establish finite valid execution/coverage; they do not claim correct possession inference when observations are absent.

Rejection cases cover an extra label field, complex coordinates, vector FPS, no published period, infinite coordinates, altered model bytes, mismatched numerical version, stale runtime hash, omitted variant, invalid automatic variant, missing input and existing output. The extra label field contains dummy test values, not reference possession labels. The existing output sentinel remains unchanged; failed-format unit tests leave prior CSV/sidecar bytes untouched and remove staging files.

The full local challenge-input path uses published tactical metadata and raw tracking only. Reference/scoring data, video and Smart Tagging are not read. The standalone supplied checker is run after the CSV is written; its descriptive prediction counts are suppressed. No challenge frame counts, possession counts, durations, resource numbers or accuracy scores are reported here. Candidate CSVs remain local ignored artifacts, not submissions or semantic-accuracy evidence. No model settings were tuned using these outputs.

## External runtime and memory

Across the twelve full external raw-input checks, instrumented worker wall time is 22.08–23.80 seconds and peak Windows working set is 761.1–765.8 MiB. Two workers run concurrently, with four model threads each. These worker timings include imports/model loading, inference, CSV writing and the standalone checker after the harness starts; they are not directly comparable to the earlier feature/model/decoder-only timings.

The verifier uses empirical local acceptance limits of 300 seconds and 2 GiB peak working set per full external run. These are engineering regression limits, not official competition compute rules. Measurements cover this Windows x64 host and pinned packages only; they do not establish cross-platform bit equality or another machine's runtime. No competition-input resource metrics are published.

## Independent artifact audit

`audit_final_verification.py` verifies source snapshot identity, identical learned settings/weights before and after the validator update, the AST restriction, local wheel/install hashes, isolated environment metadata, exact external outputs, CSV/sidecar identities, expected negative failures and the challenge output contract/repeats. It performs no target-label reads or quality evaluation.

Main records under `outputs/final_verification/run_20261101/`:

- `report.json`, `audit_report.json`, `snapshot.json`: acceptance, independent checks and candidate-source inventory.
- `fresh_environment.json`, `install_report.json`, `wheelhouse/wheelhouse.json`: dependency/environment provenance.
- `clean_source/runs/`: generated predictions, per-process guards, sidecars and private process logs.
- `prior_final_bundle/`, `prior_heldout_bundle/`: preserved pre-fix payloads/runtime source and refresh records.

Normal prediction records explicitly mark `scored=false` and `reference_arrays_read=false`. Guard self-tests intentionally attempt dummy protected filenames; those blocked probes are separated from ordinary inference. The private output tree, including challenge predictions, is ignored. No commit, push, publication or submission was performed.

## Reproduce

On this machine, the cached-wheel/fresh-environment steps were:

```powershell
& '.\.venv\Scripts\python.exe' solution/tools/collect_cached_wheels.py --output outputs/final_verification/run_20261101/wheelhouse
& '.\.venv\Scripts\python.exe' -m venv outputs/final_verification/run_20261101/fresh_env
& '.\outputs\final_verification\run_20261101\fresh_env\Scripts\python.exe' -m pip install --no-index --find-links outputs/final_verification/run_20261101/wheelhouse --require-hashes -r outputs/final_verification/run_20261101/wheelhouse/requirements-hashed.txt --report outputs/final_verification/run_20261101/install_report.json
```

The collector depends on locally cached pinned wheels; it fails explicitly if a cache is incomplete, rather than using another package version. This is clean inference reproduction with supplied model weights, not a raw-download-to-model retraining demonstration. Final model training still requires the recorded research artifacts described in `INFERENCE.md`.

The completed verifier/auditor commands were:

```powershell
& '.\.venv\Scripts\python.exe' solution/tools/final_verify.py --output outputs/final_verification/run_20261101 --bundle models/possession-v1 --heldout-bundle models/heldout-J03WN1-v1 --python outputs/final_verification/run_20261101/fresh_env/Scripts/python.exe
& '.\.venv\Scripts\python.exe' solution/tools/audit_final_verification.py --output outputs/final_verification/run_20261101
```

Use a new verification output directory for a new full run; the verifier refuses to overwrite a completed record. Prepare its wheelhouse/environment first. The manifest-refresh helper is deliberately limited to the declared validator-only update and records an archive; it is not a general bypass of model/source compatibility. Refit/rebuild appropriately for future learned-feature or numerical changes.
