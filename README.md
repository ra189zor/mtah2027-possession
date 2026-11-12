# AOM — football possession from tracking

One final offline pipeline: ownership-only temporal fusion. It uses recorded tracking and future context, not live video.

## Reproduce

Use Python 3.11. Download and verify the frozen code-and-weights release:

```powershell
python solution/verify_release.py --download --extract outputs/aom-runtime
cd outputs/aom-runtime
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
& '.\.venv\Scripts\python.exe' -m pip install -r solution/requirements-tcn-lock.txt
$env:MTAH_DATA_DIR = 'D:\path\to\provided-data-root'
& '.\.venv\Scripts\python.exe' solution/owner_predict.py --challenge --output outputs/AOM.csv
```

The data root must contain `tactical/BAR-ATH_tactical_atd.xml` and `tactical/BAR-ATH_tactical_atd.txt`. Obtain licensed inputs separately. Windows needs the Microsoft Visual C++ runtime. Existing prediction files are never overwritten.

From the repository root, validate the result with the pinned interpreter:

```powershell
& 'outputs/aom-runtime/.venv/Scripts/python.exe' 'more-than-a-hack-2027-main repo/evaluation/validate_submission.py' outputs/aom-runtime/outputs/AOM.csv
```

The [public release](https://github.com/ra189zor/mtah2027-possession/releases/tag/aom-owner-v1) provides `AOM_owner_runtime_v1.zip`, checksum and selection record. Use the attached ZIP: GitHub's automatic source archives omit weights. The published ZIP remains unchanged; its historical documentation predates final selection. This checkout retains the same frozen runtime sources and consolidated current documentation.

## Documentation

- [AOM technical report](AOM.md): architecture, training labels and amounts, conventions, evidence, limitations and reproduction.
- [Input/output guide](solution/docs/INFERENCE.md).
- [Validation summary](solution/docs/VALIDATION.md).
- [Data sources](solution/docs/EXTERNAL_DATA_SOURCES.md) and [third-party notices](solution/docs/THIRD_PARTY_NOTICES.md).

Only `solution/owner_predict.py` is the supported prediction entry point. Older-named modules and all three weight payloads are internal dependencies of this single pipeline. Removing or renaming them invalidates frozen source checks. `solution/configs/final_candidate.json` identifies the release; other retained configs support tests, while deployment settings come from manifests.

## Verification and submission

```powershell
python solution/verify_release.py --archive path/to/AOM_owner_runtime_v1.zip
python -m unittest discover -s solution/tests -v
```

Run tests with the full pinned PyTorch environment. Training and historical benchmark tooling remain in Git history and the preserved local development snapshot; they are not needed for fitted-model inference. The public runtime contains no match data or possession CSVs.

Official deliverables: one `AOM.csv`, a public repository URL, and short `AOM.md` report. Selection and publication are complete; competition submission remains pending. Consult the current [submission instructions](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md) and [rules](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md). Reproducibility does not establish hidden-match accuracy.
