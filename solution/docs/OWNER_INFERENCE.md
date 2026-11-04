# Runnable ownership-only candidate

`solution/owner_predict.py` runs the frozen ownership-only blend and writes one possession CSV plus a `.run.json` provenance sidecar. It combines the corrected-combined conditional-home probability with the TCN conditional-home probability at 50/50, retaining the old dead probabilities, learned switch evidence and 40/40/8 decoder with event reward 4. Existing fitted weights are reused, with no retraining or change to previous entry points.

This is one inference pipeline with several internal components, not multiple competing CSV submissions. The new manifest is `models/tcn-owner-v1/manifest.json`; it binds the base, specialist and TCN component manifests/weights, runtime source hashes, settings and training provenance. The three original bundles remain in `models/possession-v1`, `models/research-specialists-clean` and `models/tcn-blend-v1`. The TCN bundle's historical name does not select joint fusion: this entry point explicitly uses only its ownership head. Loading rejects changed sources, components or settings.

## Run in the existing workspace

Use the existing optional CPU PyTorch interpreter from the project root. Activation is unnecessary. After runtime verification, a local Game 3 tracking-only example is available:

```powershell
& '.\outputs\tcn_research\env\Scripts\python.exe' solution/owner_predict.py --input outputs/owner_candidate/runtime_verification_20261104_v2/inputs/native.npz --output outputs/owner_candidate/my_game3_test.csv
```

Choose a new output name for each run. Existing CSVs or sidecars are never overwritten. Input NPZs must contain exactly `players_xy`, `ball_xy`, `team_of_slot`, `period`, `pitch_m` and `fps`: anonymous team slots, metre coordinates, 25 Hz, NaN for missing tracking, and ordered published periods 1/2 with 0 outside them. Event/reference fields are rejected. Match video alone is not this interface.

The CSV contains `frame,label` with zero-based frame numbers and H/A/D predictions during published halves; X is reserved for period 0. The official format checker validates the written CSV without reference scoring. The sidecar records the input fingerprint, component identities, source hashes, frozen fusion settings and timing. Its `model_sha256` is the SHA-256 of the canonical mapping of the three component weight hashes, explicitly identified as a model-set digest rather than a fourth weights file.

The CLI also accepts `--challenge` in place of `--input`, using the existing local tactical tracking loader and official published periods. It does not read video, smart tags or reference labels. This packaging checkpoint verifies only external inputs; it does not run that flag, score the challenge or select/upload a submission. `--base-bundle`, `--research-bundle`, `--tcn-bundle` and `--bundle` permit relocating identical verified bundles, not silently replacing the frozen model.

## Portable runtime archive

The local archive `outputs/owner_candidate/AOM_owner_runtime_v1.zip` contains an explicit allowlist of runtime code, the four manifests, three fitted weight files, dependency locks, usage notes and attribution. It excludes match inputs, reference labels, predictions, training tools, research caches and installed dependency binaries. It is a local runtime package, not a competition submission upload.

Extract it into a new directory and use Python 3.11 with the pinned versions from `solution/requirements-lock.txt` and `solution/requirements-tcn-lock.txt`. The existing environment can also run the extracted script directly. CPU PyTorch is `2.8.0+cpu`; its wheel comes from the official PyTorch CPU index. Dependency installation is separate from extraction, and Windows needs the Microsoft Visual C++ runtime already installed on this PC. All paths in runtime code resolve against the extracted directory; original research folders are not needed for inference.

## Rebuild and verify

The build step consumes the original frozen Game 3 validation artifacts to bind the evaluated candidate. They are needed for rebuilding provenance, not for running the packaged model. It performs no fitting, parameter search or accuracy calculation:

```powershell
& '.\outputs\tcn_research\env\Scripts\python.exe' solution/tools/package_owner.py
& '.\outputs\tcn_research\env\Scripts\python.exe' solution/tools/export_owner_runtime.py --output outputs/owner_candidate/AOM_owner_runtime_v1.zip
& '.\outputs\tcn_research\env\Scripts\python.exe' solution/tools/verify_owner_runtime.py --archive outputs/owner_candidate/AOM_owner_runtime_v1.zip --output outputs/owner_candidate/runtime_verification_20261104_v2
```

Use new output paths to repeat export or verification. The verifier extracts the allowlisted archive, reconstructs external inputs, compares CLI predictions with the validated evidence, checks repeatability and probabilities, exercises missing/invalid inputs, and runs with a read guard blocking original research/reference dependencies. Its report distinguishes software verification from accuracy validation. It reuses the existing pinned environment rather than claiming a fresh dependency installation.

## Accuracy and remaining limits

See `OWNER_TRANSFER_GAME3.md` for the fixed fresh-fixture experiment. Moderate accuracy on synthetically obscured pass frames improved 84.13% to 87.00%, with a paired conditional interval above zero. Guarded references cover only 46.99% of one match. Native stoppage recall stayed similar; ghost-exposed partial accuracy slightly regressed. Earlier seven-match DFL results support ownership-only fusion but use a different label convention. Neither benchmark proves hidden-match accuracy or a guaranteed winning submission.

This pipeline uses future context and runs offline on recorded tracking. It is not a causal live-stream predictor, and it does not add video ball recovery or automatic-tag fusion. The original and joint-blend candidates remain available; final AOM submission choice is still separate.
