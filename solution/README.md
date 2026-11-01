# More Than A Hack 2027 solution workspace

Status: Algorithm development, end-to-end inference, final verification and draft submission packaging are complete. Stronger event rewards improve recorded moderate provider spellF1 from 0.4916 to 0.5046 and count passes from 2/7 to 4/7, with zero playing-period X. Alternate-label regressions remain, so variant choice is explicit. All 103 tests pass; the earlier 98 runtime/research tests also passed in the fresh environment. The public runtime archive reproduces all three private candidates exactly. Team name and primary variant remain deferred at the user's request. See `docs/DELIVERY.md`; nothing has been published or submitted.

See `docs/SWITCH_EVENTS.md` for current results, cross-fitting, audits and commands. `docs/NESTED_VALIDATION.md` records the original broader benchmark; `docs/SHORT_POSSESSIONS.md` retains the earlier control/sampling tradeoffs and negative results. No competition performance or submission readiness is claimed.

Run `predict.py` from the project root using `--bundle models/possession-v1`, either `--input <tracking.npz>` or `--challenge`, an explicit `--variant baseline|event2|event4`, and `--output <prediction.csv>`. The final bundle is fitted on all seven external fixtures; its own smoke predictions are not a new held-out quality estimate. See `docs/INFERENCE.md` for commands and `docs/FINAL_VERIFICATION.md` for fresh-environment, missing-data, rejection, resource and unscored challenge acceptance checks. Model weights/settings are unchanged; manifests record the validator-only runtime revision.

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
