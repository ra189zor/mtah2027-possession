# Final model input and output

Install the pinned Python 3.11 CPU PyTorch environment using the root README. Run `solution/owner_predict.py` from the extracted release.

For other recorded matches:

```powershell
python solution/owner_predict.py --input path/to/tracking.npz --output outputs/prediction.csv
```

The NPZ contains exactly six fields: `players_xy`, `ball_xy`, `team_of_slot`, `period`, `pitch_m`, `fps`. Player coordinates have shape `(frames, slots, 2)` and ball coordinates `(frames, 2)`, in metres from a pitch corner. Team slots use home=0, away=1. Periods use 1/2 during published halves and 0 outside them. Pitch dimensions are length/width; fps is 25. Missing coordinates are NaN. Extra fields, including references, are rejected. Video alone is unsupported.

Output is an ordered, zero-based `frame,label` CSV plus a `.run.json` provenance sidecar. Playing-period labels are H/A/D; X is reserved for period 0. Existing outputs are rejected. Dependency versions, source hashes and weights are checked at load time.

The model uses equal conditional-home fusion, retained tree-based dead/event evidence and fixed 40/40/8 decoding costs with event reward 4. All three weight payloads and four manifests are required together. A source-hash error means the installation differs from the verified release; do not bypass it.

Repeated CSV bytes were verified in the pinned local environment. Cross-platform byte equality is untested. Future context makes this offline inference.
