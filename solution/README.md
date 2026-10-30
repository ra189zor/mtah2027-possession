# More Than A Hack 2027 solution workspace

Status: Steps 1–6, learned evidence, ownership recovery, temporal decoding and seven-match nested validation are complete. Mean outer spellF1 is 0.4916 moderate / 0.4718 contextual; all predictions have zero playing-period X. Provider-label counts pass the approximate ±20% gate on only 2/7 moderate fixture averages, and label convention materially changes the count diagnosis. Nested cost selection adds negligible gain over previous fixed costs. See `docs/NESTED_VALIDATION.md` for results, limitations and reproduction. Final end-to-end inference packaging remains pending.

Short-possession recovery experiments are also complete: a control boost improves moderate count passes to 3/7 with mean spellF1 0.4892, while boundary-aware ownership training does not improve the provider objective. Original heads/decoder remain preferred; no experimental candidate is silently substituted. All 83 tests pass. See `docs/SHORT_POSSESSIONS.md` for methods, audits and both label conventions.

The project root is the intended Git repository root. This `solution/` directory contains our code, configuration and documentation. A single project-root `.gitignore` excludes the virtual environment, original downloads, prepared match inputs, generated reports and local annotations. Committing and pushing are handled by the user. The previous Step 1 local Git metadata is preserved under ignored `.local-git/solution.git`; `solution/` is now a normal directory.

## Environment

Python 3.11.9. The existing virtual environment is `../.venv`; activation is optional when its interpreter is called directly. From this directory in PowerShell:

```powershell
& '..\.venv\Scripts\python.exe' -m pip install -r requirements-lock.txt
& '..\.venv\Scripts\python.exe' -m pip check
```

To create an independent environment on another machine, use Python 3.11:

```powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-lock.txt
```

## Official scorer smoke check

From this directory, using the existing sibling environment:

```powershell
& '..\.venv\Scripts\python.exe' '..\more-than-a-hack-2027-main repo\evaluation\validate_submission.py' '..\more-than-a-hack-2027-main repo\evaluation\results\baseline_smooth50.csv'
& '..\.venv\Scripts\python.exe' '..\more-than-a-hack-2027-main repo\evaluation\run_evaluation.py' --no-detail --bootstrap 1000 --out '..\research\step1_evaluation'
```

Expected: validator OK, 199319 rows; public spellF1 0.3256 for smooth50 and 0.4242 for smooth125. These commands score the supplied baseline files.

To rebuild from raw tracking, run from the project root:

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/reproduce_baselines.py
```

This verifies both reference-label equality and byte-identical repeated executions. See `docs/BASELINE_REPRODUCTION.md` for recorded results and Windows newline handling.

## Prepare and check local data

From the project root in PowerShell:

```powershell
& '.\.venv\Scripts\python.exe' solution/tools/prepare_data.py
& '.\.venv\Scripts\python.exe' solution/tools/check_data.py
```

The preparation tool extracts only the seven expected non-video assets from the original ZIPs. It refuses to overwrite different existing data and verifies existing inputs on reruns. It creates hard links to the two original videos with the canonical filenames expected by `games.asset_paths()`, without a second large video copy. Treat both names as read-only: editing a hard-linked video also changes the original bytes.

Inputs resolve under `more-than-a-hack-2027-main repo/data/tactical/` and `data/broadcast/`. The default official loader paths work without setting `MTAH_DATA_DIR`. A local manifest and aggregate verification report are written under ignored research paths. Video checking decodes selected frames, including the last reported frame; it does not perform a full sequential decode or prove broadcast/tactical synchronization.

## Data and publication

Feature extraction runs from the project root with `& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/extract_features.py`. It writes ignored caches with values, validity masks and channel names. See `docs/FEATURE_EXTRACTION.md` for methods, verification and limits.

External validation tooling, the frozen match split and test scenarios are documented in `docs/EXTERNAL_VALIDATION.md`. Public Sportec/DFL source attribution and transformations are recorded in `docs/EXTERNAL_DATA_SOURCES.md`. The actual external downloads, arrays and generated results remain in ignored paths at the project root.

Do not commit restricted data, frame-level predictions, independent validation labels, video clips or reconstructive exports. Signed registration terms govern data use. Hosted LLMs must not receive restricted match materials or extracts. Public model dependencies need licence and provenance review.

The official source snapshot retains its own MIT licence. We have not copied its implementation into this repository. If code is copied later, preserve its attribution and licence.

See `docs/DECISIONS.md` for the approved starting strategy and remaining decisions. Work advances one numbered implementation step at a time, on explicit user instruction.
