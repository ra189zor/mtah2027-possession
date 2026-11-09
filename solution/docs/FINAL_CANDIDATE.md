# AOM final candidate — 9 November 2026

Ownership-only (`tcn_owner_blend`, `solution/owner_predict.py`) is selected as AOM's final candidate at the user's request. Earlier variants remain preserved as research alternatives, not additional submissions.

This is the already trained and verified 50/50 conditional-ownership blend, retaining corrected-combined dead-ball and switch-event evidence with fixed decoding. No weights, thresholds, runtime code or predictions changed during selection.

The choice balances the seven-match DFL results with cross-provider transfer. On the frozen external Metrica Game 3 test, moderate newly missing pass accuracy increased from 84.13% to 87.00%, with a paired conditional 95% improvement interval of +1.40 to +4.27 percentage points. Joint fusion had degraded stoppage transfer on Games 1/2. Ownership-only remains imperfect: ghost-exposed partial accuracy declined slightly, label conventions affect rankings, and hidden-match accuracy is unknown. See `OWNER_TRANSFER_GAME3.md`.

## Frozen identity

- Runtime ZIP: `AOM_owner_runtime_v1.zip`
- ZIP SHA-256: `4373a111611d78e2734bae80897ba650e5109f0cba29facfccafd52e5cf6d6ee`
- Owner manifest SHA-256: `16c698cd96f134ea73a568cc49a3478081f55890d0d1009b5e73b80ee6d2a22c`
- Model-set SHA-256: `8167d736571462e8d4d4b552ea91c00648f1f14a98e9e7126ff106e2e9e63398`

The ZIP contains inference source, all three fitted weight payloads, four manifests, dependency locks and attribution. It contains no match data or predictions. Use this attached ZIP for exact reproduction; GitHub's automatically generated source archives do not contain ignored weights.

The selected local CSV/report are the verified artifacts under `outputs/submission/AOM_owner_20261104/private_candidate/`. Historical draft manifests remain unchanged to preserve their audit hashes; this dated selection supersedes their pending-selection status. Selection is not competition submission.

## Reproduce

Extract the ZIP into a new directory. From the extracted root, using Python 3.11:

```powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
& '.\.venv\Scripts\python.exe' -m pip install -r solution/requirements-tcn-lock.txt
$env:MTAH_DATA_DIR = 'D:\path\to\provided-data-root'
& '.\.venv\Scripts\python.exe' solution/owner_predict.py --challenge --output outputs/AOM.csv
```

Obtain licensed tracking inputs separately. For other matches use `--input path/to/tracking.npz` instead of `--challenge`; see the archive's `solution/docs/OWNER_INFERENCE.md` for the six required fields. This is offline tracking inference, not live video inference. Windows requires the Microsoft Visual C++ runtime. Exact repeated output was verified in the pinned local environment; cross-platform byte equality and fresh dependency installation are not claimed.

The historical archive documentation calls this a candidate because the ZIP predates final selection; this document records the subsequent decision without rewriting the verified package.
