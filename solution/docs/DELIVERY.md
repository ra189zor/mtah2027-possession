# Draft delivery — 1 November 2026

The runtime, model archive, technical reports and three private candidates are prepared. The registered team name is **AOM**, confirmed by the user. All variants remain available for later selection. This is a reviewed draft delivery, with no primary candidate or publication/submission claim.

## Local artifacts

Current delivery artifacts remain under ignored `outputs/submission/AOM_20261101/`. The earlier unnamed package is retained as a historical draft:

- `HANDOFF.md`: final user actions and boundaries.
- `mtah2027-possession-runtime-20261101.zip`: public-source/model archive, excluding all prediction CSVs and match inputs.
- `private_candidates/baseline`, `private_candidates/event2`, `private_candidates/event4`: each contains an unchanged algorithm-generated `AOM.csv`, matching `AOM.md` technical report and candidate manifest. Do not publish these folders as part of the code archive.
- `PACKAGE_INDEX.json`: SHA-256 identities and pending administrative fields.
- `VERIFICATION.json`: independent archive and inference verification.

Public archive SHA-256: `62161858f581441154c37f7d346c918f39ff291a77554a2dd84e730c4d56c6e8`.

The archive includes the final two-file all-external model bundle, exact inference sources, pinned package list, a stdlib checksum verifier, provider attribution and a dependency metadata inventory. The upstream MIT notice is preserved; no new licence has been assigned to project-owned source. Dependency binaries are installed separately. Source/model hashes are checked by inference. The archive's public technical report uses baseline as an explicit reproduction example and explains that all three variants are available.

## Verification scope

The ZIP is checked against an explicit source inventory before safe extraction; unexpected files, duplicate entries and non-regular files are rejected. SHA-256 checks cover payload, manifest, runtime sources and fitted weights. The extracted stdlib verifier passes.

Each extracted variant runs in the previously verified fresh environment, using only the provided raw tracking and metadata. Each regenerated CSV is byte-identical to its private candidate and passes the standalone format checker. Access guards record no reference/research reads; no quality scoring is performed. No challenge possession statistics are reported. These checks establish runtime reproducibility with supplied fitted weights, not raw-data-to-model retraining or hidden-match accuracy.

All 103 tests pass, including five delivery safeguards covering archive determinism, traversal, private prediction exclusion, corruption and deferred team/variant choices. The earlier 98 runtime/research tests also passed in the fresh environment. The package tests do not refit or tune the models.

## Rebuild locally

Choose a new output directory; existing packages are never overwritten:

```powershell
& '.\.venv\Scripts\python.exe' solution/tools/package_submission.py --output outputs/submission/my_new_draft --team-name AOM
& '.\.venv\Scripts\python.exe' solution/tools/verify_submission_package.py --package outputs/submission/my_new_draft --python outputs/final_verification/run_20261101/fresh_env/Scripts/python.exe
```

The packaging helper expects the already verified local model bundle and final candidate source files. The team name is supplied explicitly; primary variant selection remains pending. Building uses working-tree bytes, not an assertion that those bytes have been committed. Deterministic ZIP timestamps and sorted entries make equivalent inputs reproducible.

## Remaining user decisions/actions

Select one candidate; its report and filenames already use AOM. Confirm member registration/signed terms and public repository access. Commit/push code and make the model archive available separately (weights are ignored by Git), then submit the selected files through the registered channel. Repository visibility has not been verified. No commit, push, upload, release or submission was made by this step.

The [official submission instructions](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md) accept a Markdown report and request exact reproduction from public code with supplied licensed inputs; the [rules](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md) and signed registration terms remain authoritative. Online instructions were checked on 1 November 2026; original provided snapshots remain unchanged.
