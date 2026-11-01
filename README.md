# mtah2027-possession

Football possession segmentation from noisy tracking data and tactical events for More Than A Hack 2027.

## Current status

Steps 1–6, learned evidence, gap-aware ownership, temporal decoding, seven-match external validation and learned switch-event experiments are complete. The stronger fixed event reward improves mean moderate provider-label spellF1 from 0.4916 to 0.5046 and count passes from 2/7 to 4/7. Contextual-noise spellF1 improves from 0.4718 to 0.4837. All predictions have zero playing-period X; all 90 tests and the cross-fitting/repetition audit pass.

The event variants regress under absorbed-short-spell references, so they and the original remain explicit choices rather than a universal default. False switches and three moderate count failures remain. Methods and label-convention results are in `solution/docs/SWITCH_EVENTS.md`.

End-to-end inference is now runnable with a final all-seven external model bundle and an explicit variant. All 95 tests pass; nine guarded held-out raw-input runs reproduce earlier CSVs exactly. From the project root:

```powershell
& '.\.venv\Scripts\python.exe' solution/predict.py --bundle models/possession-v1 --input data/external/sportec/J03WN1/prepared/inputs.npz --variant event4 --output outputs/my_prediction.csv
```

Use `baseline`, `event2` or `event4`; see `solution/docs/INFERENCE.md` for inputs, the challenge adapter, model provenance, guard/repeatability results and limitations. The local model bundle is ignored by Git and must be supplied separately or rebuilt on another machine. Final verification and submission packaging remain pending; no competition quality score or submission readiness is claimed.

## Project layout

- `solution/`: our implementation, dependencies, data preparation tools and progress notes.
- `more-than-a-hack-2027-main repo/`: provided challenge source and official evaluation code. Its original nested ignore file was consolidated into the root policy; other supplied source files remain unchanged.
- `research/`: research report and aggregate audit notes; generated reports/local manifests are ignored.
- `.gitignore`: the single project ignore policy.

Original downloads, prepared data, the virtual environment and the previous local Git checkpoint are ignored. The user handles repository initialization, commits and pushes.

## Environment and data verification

Run from the project root in PowerShell:

```powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r solution/requirements-lock.txt
& '.\.venv\Scripts\python.exe' solution/tools/prepare_data.py
& '.\.venv\Scripts\python.exe' solution/tools/check_data.py
```

The existing local `.venv` is already installed; recreation is only needed on a new machine. Restricted downloads must be obtained separately through registration and placed in the original dataset directory. They are not distributed in this repository.

Preparation extracts seven non-video assets and creates two canonical video hard links. Inputs remain local. Treat hard-linked video files as read-only, since either filename refers to the same bytes.

Tactical inputs pass all implemented checks. The broadcast decoder reports one fewer video frame than its tracking row count. Broadcast fusion is deferred; the qualifying output always uses the tactical timeline. Video checks decode selected key frames rather than scanning every frame sequentially.

Detailed results: `solution/docs/DATA_VERIFICATION.md`. Current progress: `solution/docs/PROGRESS.md`.

## Official baseline reproduction

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/reproduce_baselines.py
```

This rebuilds all four baselines twice, validates them, checks labels and repeated-file hashes, then scores them on the official public correctness window. Output stays under ignored `outputs/baseline_reproduction/`. Summary and measurement limits: `solution/docs/BASELINE_REPRODUCTION.md`.

## External validation

The historical split and original noise scenarios remain in `solution/configs/validation.json`. The completed seven-match protocol in `solution/configs/nested_validation.json` explicitly replaces the old sealed role: all seven public Sportec/DFL fixtures are now opened outer-test fixtures. Inputs and reference labels remain separate. See `solution/docs/NESTED_VALIDATION.md` for current results and `solution/docs/EXTERNAL_DATA_SOURCES.md` for attribution. Historical experiments and their split guards remain unchanged.

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s solution/tests -v
```

## Feature extraction

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/extract_features.py
```

This creates ignored feature caches from tracking and available automatic tags without opening reference-label arrays. Each channel has an explicit validity mask. Details, checks and limitations: `solution/docs/FEATURE_EXTRACTION.md`.
