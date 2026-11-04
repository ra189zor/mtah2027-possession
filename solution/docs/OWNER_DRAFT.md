# AOM ownership-only full-input draft — 4 November 2026

The frozen ownership-only candidate has been run against the locally supplied full tactical tracking. Two independent executions from the verified runtime archive produce identical CSV bytes. Both pass the supplied standalone format checker and published-period coverage checks. No reference labels, video, Smart Tagging or other match assets enter inference; the only allowed match files are tactical tracking and metadata.

The private draft lives under `outputs/submission/AOM_owner_20261104/`:

- `private_candidate/AOM.csv`: unchanged algorithm output.
- `private_candidate/AOM.md`: matching technical document, including architecture, assets, every labelled external dataset, label derivation/amount, external results, limitations and reproduction.
- `private_candidate/AOM.run.json` and `candidate_manifest.json`: exact input/model/runtime/output identities.
- `verification.json`, `protocol.json` and `runs/`: repeat, format, period and access-guard records.
- `HANDOFF.md`: file locations and remaining publication/submission actions.

The model is the previously evaluated and packaged candidate; no new fitting, thresholds, blend settings, decoder costs or hand corrections were introduced. Original components and earlier candidates remain unchanged. All 143 tests pass. Checksums cover the final CSV/report/sidecar, archive, source, model identities and input files. The final private files are ignored by Git.

These checks establish reproducibility and format acceptance, not possession accuracy. No public-reference scoring or competition match statistics/resource measurements are reported. Quality evidence remains in `OWNER_TRANSFER_GAME3.md` and the external DFL experiments; the candidate still has false-ball and label-convention limitations.

## Reproduce this draft

From the project root with the existing pinned environment, choose a new output directory:

```powershell
& '.\outputs\tcn_research\env\Scripts\python.exe' solution/tools/prepare_owner_candidate.py --output outputs/submission/AOM_owner_new_draft
```

The helper consumes the verified runtime ZIP, its preceding external verification record, the frozen model identities, the provided tactical files and a report template. It runs the isolated worker twice, suppresses the standalone checker's descriptive match statistics, and only creates the final candidate after repeat/provenance/format checks pass. It refuses to overwrite an earlier draft. The worker allows PyTorch to inspect its isolated harness source while blocking original research tools, benchmark evidence and reference files; guard self-tests exercise those blocks.

## Remaining delivery boundary

This is a local draft of one candidate, not final selection or an uploaded submission. The configured remote is `https://github.com/ra189zor/mtah2027-possession`; public availability of the latest code and fitted archive has not been verified. Model weights are Git-ignored, so a code push alone does not make this candidate reproducible for organizers. The user handles commits/pushes, final candidate choice, release publication and registered submission. The runtime ZIP remains `outputs/owner_candidate/AOM_owner_runtime_v1.zip` and contains no match data or candidate CSVs.

The [submission instructions](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md) and [rules](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md) were checked online on 4 November 2026. The current report requirements explicitly include labelled-data source, annotator/provider and amount. The rules restrict human labels/corrections for the provided fixture and reserve the public reference window for correctness checks. This draft uses labels from other games only for fitting and performs no new correctness-window evaluation. Historical local source snapshots are preserved.
