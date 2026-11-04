# More Than A Hack 2027 solution workspace

Latest research: the corrected reliability/gap/boundary candidate is implemented and runnable. Seven-match moderate provider spellF1 rises from 0.5046 to 0.5159, with improvement on all seven fixtures. All 108 tests pass in both environments; the corrected research runtime reproduces held-out outputs exactly. See `docs/RELIABILITY_RESULTS.md` and `docs/RESEARCH_INFERENCE.md`. The following delivery notes describe the preserved original AOM package; the new research candidate is a separate explicit choice.

Status: Algorithm development, end-to-end inference, final verification and draft submission packaging are complete. Stronger event rewards improve recorded moderate provider spellF1 from 0.4916 to 0.5046 and count passes from 2/7 to 4/7, with zero playing-period X. Alternate-label regressions remain, so variant choice is explicit. All 103 tests pass; the earlier 98 runtime/research tests also passed in the fresh environment. The public runtime archive reproduces all three private candidates exactly. The registered team name is AOM; primary variant selection remains deferred. See `docs/DELIVERY.md`; nothing has been published or submitted.

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

The 2 November stoppage diagnosis, learned stop/restart experiment and two additional public Metrica sample checks are documented in `docs/STOPPAGE_RESEARCH.md`. These research checks preserve the existing deployment bundles; the two added samples supply partial binary references, not complete possession ground truth.

The TCN experiment completed on 3 November with a stronger fixed joint blend on all seven provider-label fixtures in every noise condition. The final candidate runs through `solution/tcn_predict.py` in the optional PyTorch environment and writes one CSV. See `docs/TCN_RESEARCH.md` for commands, held-out scores, alternate-label count limitations and verification. The earlier submission archives remain historical candidates.

The subsequent frozen Metrica transfer check found better ownership through simulated ball gaps but worse native stoppage recall for the full joint blend. See `docs/TCN_TRANSFER.md` for the partial-reference results and the exploratory ownership-only alternative. The TCN blend is not established as a universal replacement; current bundles and submission files are preserved.

The 4 November fresh Game 3 check supports the frozen ownership-only alternative: moderate accuracy on synthetically missing pass frames improves 84.13% to 87.00%, with similar native stoppage recall. Ghost-exposed frames remain a weakness. See `docs/OWNER_TRANSFER_GAME3.md` for the partial-reference results, uncertainty, reproduction and audit. This validation does not change the runnable defaults or final submission choice.

The ownership-only candidate is now runnable through `solution/owner_predict.py`, using the existing optional CPU PyTorch environment and `models/tcn-owner-v1/manifest.json`. It writes one CSV and a provenance sidecar. See [ownership-only run instructions](docs/OWNER_INFERENCE.md). The local portable archive is `outputs/owner_candidate/AOM_owner_runtime_v1.zip`; it includes verified runtime code and weights, with no match data. The extracted archive reproduces three external cases exactly, passes byte-identical repeat and missing/invalid-input checks, and runs without original research/reference access. All 143 tests pass. Original entry points, model weights and submission files remain unchanged; this packaging step does not provide a new accuracy score.

Do not commit restricted data, frame-level predictions, independent validation labels, video clips or reconstructive exports. Signed registration terms govern data use. Hosted LLMs must not receive restricted match materials or extracts. Public model dependencies need licence and provenance review.

The official source snapshot retains its own MIT licence. We have not copied its implementation into this repository. If code is copied later, preserve its attribution and licence.

See `docs/DECISIONS.md` for the approved starting strategy and remaining decisions. Work advances one numbered implementation step at a time, on explicit user instruction.
