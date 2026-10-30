# mtah2027-possession

Football possession segmentation from noisy tracking data and tactical events for More Than A Hack 2027.

## Current status

Steps 1–6 and external oracle diagnosis are complete. A learned dead-ball model is now selected through four development match folds and evaluated on two validation fixtures. Dead recall improves substantially; the integration still has ownership unknowns and fragmented possessions. See `solution/docs/DEAD_BALL_MODEL.md`. The next step is gap-aware ownership before decoder work; the original fallback is preserved.

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

The frozen whole-match split and noise scenarios are in `solution/configs/validation.json`. Two public Sportec/DFL fixtures are prepared locally; the sealed final-test fixture is untouched. Inputs and reference labels remain separate. See `solution/docs/EXTERNAL_VALIDATION.md` for results, limitations and reproduction commands, and `solution/docs/EXTERNAL_DATA_SOURCES.md` for attribution.

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s solution/tests -v
```

## Feature extraction

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/extract_features.py
```

This creates ignored feature caches from tracking and available automatic tags without opening reference-label arrays. Each channel has an explicit validity mask. Details, checks and limitations: `solution/docs/FEATURE_EXTRACTION.md`.
