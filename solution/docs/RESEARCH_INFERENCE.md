# Running the new specialist candidates

The research runner supports explicit `gap`, `boundary` and `combined` candidates. It uses the original base bundle plus a separate experimental specialist bundle. The existing AOM submission archive and its three original variants are preserved; this research delivery does not silently select or publish a new submission.

## Input and runtime

Use the same six-field 25 Hz tracking input or installed official tactical tracking as the original runner. This remains offline inference using future observations. Video alone and live streaming are not supported. The research path reuses the original feature/model code, then derives quality/gap features and applies the specialist evidence. It requires no research caches or reference labels at inference.

From the project root, test the corrected-quality specialist candidate with:

```powershell
& '.\.venv\Scripts\python.exe' solution/research_predict.py --base-bundle models/possession-v1 --research-bundle models/research-specialists-clean --challenge --variant combined --output outputs/AOM_research_combined.csv
```

The corrected detector and joint refit are tracked in `run_20261101_clean`. The initial `research-specialists-v2` bundle is preserved for comparison and retains the first quality detector's documented clean false alarms. Do not interchange the quality head and specialist heads between these experiments. `gap` and `boundary` remain available as explicit ablations; `combined` is the research candidate shown here, not a registered submission selection.

To reproduce a genuinely held-out external case from the corrected experiment:

```powershell
& '.\.venv\Scripts\python.exe' solution/research_predict.py --base-bundle models/heldout-J03WN1-v1 --research-bundle models/research-heldout-J03WN1-clean --input outputs/final_verification/run_20261101/clean_source/inputs/moderate.npz --variant combined --output outputs/research_heldout_test.csv
```

Choose a fresh output filename for each run. The CLI refuses to replace existing outputs. The CSV has an accompanying `.run.json` identifying the research bundle and base manifest, with no scoring/reference-read flags. Bundle loading checks the model payload, runtime source and exact base manifest identities. Base dependency/version checks remain active. Only load trusted model artifacts.

## Interpretation

Final research heads fit all seven external fixtures; their own training-match predictions are not independent accuracy evidence. Quality estimates come from seven separate held-out fits. The public provider reference convention and alternate absorbed-spell conventions remain different objectives. A model that improves event4 can still trail the original baseline under longer-possession references.

The seven-fold benchmark retains each fold's previously inner-selected decoder costs. The deployment runner retains the original global 40/40/8 stop/restart/switch profile, chosen in earlier development. These are explicitly different cost-selection scopes; the seven-fold scores describe that validation pipeline, not a measured hidden-match score for the final all-seven fitted model. No new decoder search was performed on these outer results.

Timing fields inherited from the base inference describe the base pass; they are not measurements of the full specialist path. No speed improvement is claimed. See `RELIABILITY_RESULTS.md` for quality metrics, false alarms, uncertainty and remaining count failures.

All research weights and generated outputs remain ignored by Git. Publishing/repackaging them is a separate delivery action after candidate selection; no upload or competition submission is performed by these commands.
