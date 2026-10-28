# Step 3: official baseline reproduction

Completed **28 October 2026**. This verifies the provided algorithm and pipeline; it is not an improved model or a qualification result.

## What ran

The unchanged `starter/baseline.py` was executed twice against the prepared tactical ATD metadata/raw files. Each execution reread the raw tracking and produced all four variants in separate ignored output directories. No predictions were copied from the supplied reference CSVs, and ground truth was not used to generate predictions.

For each variant:

- Validated both generated files against the exact 199319-row contract.
- Compared every generated frame index and label against the provided reference.
- Ran the official submission-validator CLI on the first generated file.
- Compared SHA-256 hashes between the two independent executions.
- Scored the regenerated predictions with the official public scorer, using 1000 bootstrap draws.

Source/reference/input hashes were checked before and after execution. The original implementation, supplied predictions and tracking files were unchanged.

## Results

| Baseline | Rows | Label differences from reference | Repeat file bytes | Public macroF1 | Public spellF1 |
|---|---:|---:|---|---:|---:|
| raw | 199319 | 0 | identical | 0.5972 | 0.0378 |
| smooth25 | 199319 | 0 | identical | 0.6200 | 0.2868 |
| smooth50 | 199319 | 0 | identical | 0.6207 | 0.3256 |
| smooth125 | 199319 | 0 | identical | 0.5812 | 0.4242 |

All validators passed. The public correctness scores match the supplied regression values. No parameters were tuned and no private full-match evaluation was performed.

### Newline handling

The provided CSVs use LF line endings. The unchanged official writer produces CRLF on Windows. Therefore, regenerated files have different byte hashes from the downloaded CSVs, but their complete text matches after newline normalization and all frame labels match exactly. The two Windows executions produce byte-identical files. Both newline conventions are accepted by the official validator. Our eventual submission writer should explicitly fix encoding and line endings for portable reproduction.

## Performance on this machine

| Run | Wall time to generate all four CSVs | Peak baseline-worker working set |
|---|---:|---:|
| 1 | 2.643 s | 43.69 MiB |
| 2 | 2.718 s | 43.55 MiB |

Public scoring, including 1000 bootstrap draws for each submission, took 1.183 s. Measurements include Python startup for each command; disk caching and system load can affect them. These are local CPU measurements, not speed guarantees for other hardware.

Peak memory was measured from inside the real baseline Python worker using Windows `PeakWorkingSetSize`. An initial parent-process monitor measured only the virtual-environment launcher; those preliminary memory numbers were discarded. No additional package was required for measurement.

## Reproduce

From the project root in PowerShell:

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 solution/tools/reproduce_baselines.py
```

The tool creates a fresh timestamped directory under ignored `outputs/baseline_reproduction/`. It preserves existing runs and refuses to overwrite an explicitly supplied output directory. Detailed local results include `verification.json`, generation/validation logs, the regenerated CSVs and official evaluation reports.

The accepted execution is `outputs/baseline_reproduction/20261028T154445555356Z/verification.json`. Do not publish the generated full-match prediction files or raw input data. Summary results and tooling can be committed by the user.

## What we learned

The prepared tactical data reproduces the official algorithm exactly at the label level. Frame alignment, team mapping, CSV generation and local scoring work together. Baseline generation is inexpensive enough for CPU iteration.

The different public scores do not select our final smoothing window. Only 17 official possessions are public, and the organizers report a different winner on the full match. Keep smooth50 as the published reference; evaluate improvements on the external-match benchmark planned for Step 4.

## Next step

Step 4: audit and adapt one Sportec/DFL external match, map possession/ball-status fields, define whole-match development/holdout splits and build seeded noisy-input tests. This has not started. No human labels will be created on the provided challenge match.
