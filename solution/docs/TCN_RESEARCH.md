# External TCN challenger — 2 November 2026

This completed fixed offline temporal-convolution experiment compares against corrected-combined with the deployed 40/40/8 decoder profile. It uses only seven external DFL/Sportec fixtures. It does not score or tune on challenge inputs, the public five-minute reference, or Metrica sample references. The joint blend leads this DFL experiment and has a final all-seven runnable refit. The subsequent frozen Metrica transfer check found a stoppage regression; see `TCN_TRANSFER.md` before treating it as an overall replacement. Completed original results and limitations appear below.

## Architecture and training

The network receives the existing 298 physical tracking features at 5 Hz, plus 298 explicit finite/missing indicators. Each outer fold fits its own mean/scale using only the other six fixtures. Finite values are standardized and clipped to eight standard deviations; missing values become zero with a zero indicator. Normalization statistics use approximately 1 Hz samples from all training cases, without labels.

A 596-to-48 projection feeds four residual blocks with two three-tap convolutions each, at dilations 1, 2, 4 and 8. Layer normalization operates across channels independently at each time step. Two output logits represent P(home | alive) and P(dead). A four-class auxiliary head learns no boundary / enter H / enter A / enter D during training. It is not used as an extra decoder reward. Total learned parameters: 85,494.

The convolutional radius is 30 samples, or six seconds in each direction. Physical input features already contain temporal context, so this is not the entire raw-tracking receptive field. This is offline inference and cannot be described as a causal live model. Half boundaries reset windows and interpolation. Explicit zero/missing padding at half edges is identical during training and inference. Native 25 Hz probabilities are interpolated within each half; final trailing frames hold the final sample.

Training is fixed at 16 epochs, 100 batches per epoch, 16 windows per batch and 128 scored steps per window. AdamW uses learning rate 0.001 with a cosine schedule to 0.0001, weight decay 0.001 and gradient clipping at 5. A training fixture and clean/moderate/severe condition are selected uniformly; noisy seeds 17, 43 and 101 are selected uniformly. Contextual corruption is evaluation-only. Dead BCE has frame weighting, conditional-home BCE has capped inverse-possession weights, and auxiliary boundary cross entropy has weight 0.2. Labels near boundaries supply auxiliary supervision only; they never enter normalization, feature extraction or inference.

Configuration, source hashes, package versions and prior comparator hashes are frozen before fitting. Checkpoints preserve optimizer, schedule and random state at completed epochs. The final fixed epoch is evaluated without outer-driven early stopping or checkpoint choice. The seven models use the same training seed, 1117. The code does not search architectures, blend weights or decoder costs against the outer results.

## Prespecified comparisons

All candidates use identical existing corrected-combined ownership-event evidence, reward strength 4 and fixed stop/restart/switch costs 40/40/8:

- `current`: exact previous corrected-combined probabilities and predictions.
- `tcn`: replace both conditional-home and dead emissions with TCN outputs.
- `tcn_blend`: primary challenger, equal-weight mixture of old/new joint H/A/D probabilities, then factor back into conditional-home/dead probabilities.
- `tcn_owner_blend`: average conditional-home probabilities; preserve current dead emissions.

Thus the `tcn` candidate is a standalone emission replacement within the existing decoder, not a network-only prediction pipeline. The auxiliary boundary head does not replace the existing learned switch-event model.

Each of seven held-out matches has ten input cases: clean seed 17; moderate, severe and contextual seeds 17/43/101. Every candidate is scored against provider labels and an evaluation-only convention absorbing eligible opposite-team spells shorter than three seconds: 560 score rows. Current predictions must reproduce the saved fixed-profile comparator exactly on all 70 inputs. This avoids substituting historical nested-cost scores for the actual comparator.

Reported interval-score uncertainty uses 1,000 paired match draws after averaging noise seeds. Conditional paired possession draws are also stored for the primary blend on moderate and contextual inputs. These are descriptive intervals, not protection against repeated research on seven previously examined fixtures, overlapping training folds or training-seed variability. No per-match candidate selection is allowed. More complete independently labelled matches remain necessary.

## Environment and reproduction

The working `.venv` and production requirement lock remain unchanged. This experiment uses `outputs/tcn_research/env`, Python 3.11.9, CPU PyTorch 2.8.0+cpu, and the pinned optional dependencies in `solution/requirements-tcn-lock.txt`. The initial newer PyTorch imports failed with Windows DLL error 1114. Microsoft's signed x64 Visual C++ runtime installer completed successfully, updating MSVCP140 to 14.51.36247.0 without restarting. A compatible PyTorch 2.8 CPU build was installed during troubleshooting and passed the complete suite; that exact environment was frozen for training. Installer and pip provenance remain in ignored `outputs/tcn_research/`.

The runtime update source is [Microsoft's official Visual C++ redistributable page](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170). CPU wheels come from [PyTorch's official version instructions](https://pytorch.org/get-started/previous-versions/). This dependency choice is not an accuracy claim.

From the project root in PowerShell, with the research environment already installed:

```powershell
& '.\outputs\tcn_research\env\Scripts\python.exe' -u solution/tools/train_tcn.py --output outputs/tcn_research/run_20261102_v2
& '.\outputs\tcn_research\env\Scripts\python.exe' -u solution/tools/audit_tcn.py --output outputs/tcn_research/run_20261102_v2
& '.\outputs\tcn_research\env\Scripts\python.exe' -m unittest discover -s solution/tests -q
& '.\outputs\tcn_research\env\Scripts\python.exe' -m pip check
```

The runner requires earlier frozen external feature caches and corrected-combined diagnosis artifacts. It is a research reproduction command, not a standalone inference CLI. Completed work resumes only with identical protocol/source/environment hashes. No commit, push or submission is performed.

The first run completed two folds, then failed when a reused process attempted to set PyTorch's interop thread count a second time. The runner now sets that value only if it differs; a worker-reuse regression test passes. The original protocol, two models and original source snapshot are retained under `run_20261102`. Version 2 retrains all seven folds in a fresh directory with the same architecture, seed, epochs and evaluation protocol. No hyperparameters were changed in response to the partial results.

## Completed external results — 3 November 2026

All seven folds and 560 score rows completed. Seed means are averaged equally across matches. The primary joint blend improves provider spellF1 on all seven matches in every condition. The standalone TCN emission replacement is not a clear overall improvement.

| Condition | Current spellF1 | TCN emissions | Joint blend (primary) | Owner-only blend | Primary change and descriptive 95% interval |
|---|---:|---:|---:|---:|---|
| clean | 0.63119 | 0.60516 | 0.67314 | 0.67483 | +0.04195 [+0.03115, +0.05419] |
| moderate | 0.51582 | 0.51586 | 0.56117 | 0.55523 | +0.04535 [+0.03402, +0.05640] |
| severe | 0.44583 | 0.45612 | 0.48946 | 0.48437 | +0.04364 [+0.03243, +0.05296] |
| contextual | 0.49071 | 0.49005 | 0.53630 | 0.52913 | +0.04559 [+0.03675, +0.05372] |

| Condition | Current frame accuracy | Joint-blend frame accuracy | Current count passes | Joint-blend count passes |
|---|---:|---:|---:|---:|
| clean | 93.78% | 94.11% | 7/7 | 7/7 |
| moderate | 88.77% | 89.70% | 4/7 | 6/7 |
| severe | 85.43% | 86.57% | 2/7 | 5/7 |
| contextual | 87.89% | 88.76% | 4/7 | 6/7 |

Under moderate noise, accuracy on true-alive frames with missing ball increases from 74.57% to 77.83%. Mean predicted possession count moves from 353.3 to 389.2 against 437.3 true; mean predicted duration moves from 8.89 s to 8.11 s against 7.34 s true. Stop/restart recall within one second improves from 67.68% / 76.54% to 70.99% / 77.91%. Direct typed H/A switch recall within 12 frames improves from 26.65% to 31.59%, while precision increases from 35.23% to 35.79%. Timing remains difficult.

The primary moderate provider gain is +0.04535, approximately 8.79% relative, with descriptive paired-match interval [+0.03402, +0.05640]. All seven moderate conditional possession intervals also exclude zero. These intervals retain the limitations described above.

| Alternate absorbed-three-second references | Current spellF1 | Joint-blend spellF1 | Current count passes | Joint-blend count passes |
|---|---:|---:|---:|---:|
| clean | 0.60808 | 0.63198 | 0/7 | 0/7 |
| moderate | 0.53623 | 0.55345 | 2/7 | 0/7 |
| severe | 0.49325 | 0.50920 | 6/7 | 1/7 |
| contextual | 0.51752 | 0.53437 | 3/7 | 0/7 |

Alternate-label mean interval F1 improves in every condition, but not on every individual match. Its moderate gain is +0.01722, descriptive interval [+0.00134, +0.03184]; only four of seven conditional possession intervals exclude zero. Longer-label possession-count agreement worsens because predicted intervals are more fragmented than this convention expects. This prevents declaring a universal convention-independent winner.

Decision: advance the prespecified 50/50 joint blend as the leading new research candidate and fit a runnable all-seven candidate. Preserve the current bundles and previous private AOM submissions. Do not substitute the slightly stronger clean owner-only blend after seeing results; no retrospective blend-weight or per-match tuning occurred. A combination of internal models still produces a single final prediction CSV.

## Verification

The independent experiment audit passes 502 hash checks, seven exact training-only scaler recomputations, seven exact probability replays and 28 exact candidate decoder replays. All 280 prediction arrays have zero active-frame unknowns. All 70 current-model comparator predictions reproduce exactly. Two independently repeated training fits have identical weights and scalers across the worker-setting fix. The 130-test suite passes in the optional environment, including model-bundle round-trip and mismatch rejection. Original production bundles retain valid source/settings/model hashes.

## Runnable candidate

`models/tcn-blend-v1/` contains the final 16-epoch network trained on all seven external matches, its all-seven training-only scaler, history and hash-checked manifest. It binds to the exact existing base and corrected-specialist bundles. The all-seven fit is not evaluated against its training labels; quality numbers above come from the seven separate held-out fits. Inference uses three local bundle directories as one pipeline, with one output CSV. It requires no feature caches, training labels or research runner imports.

Run a downloaded, prepared external tracking match from the project root:

```powershell
& '.\outputs\tcn_research\env\Scripts\python.exe' solution/tcn_predict.py --input 'data/external/sportec/J03WPY/prepared/inputs.npz' --output 'outputs/tcn_test/AOM.csv'
```

The installed command defaults to `models/possession-v1`, `models/research-specialists-clean` and `models/tcn-blend-v1`. All must be retained. Choose a new output path on each run; existing files are not overwritten. The input must contain exactly the six validated tracking fields at 25 Hz; raw video alone is not supported. The optional `--challenge` source loads local official tracking without reference labels, but it was not executed in this checkpoint. Tags/video are not consumed. Final submission selection and the earlier private submission archives have not been updated automatically.

To reproduce the final fit and independently verify the runtime's raw-feature path:

```powershell
& '.\outputs\tcn_research\env\Scripts\python.exe' -u solution/tools/package_tcn.py --experiment outputs/tcn_research/run_20261102_v2 --output models/tcn-blend-v1
& '.\outputs\tcn_research\env\Scripts\python.exe' -u solution/tools/verify_tcn_runtime.py --experiment outputs/tcn_research/run_20261102_v2
```

The final refit uses the same sampler, loss, architecture, seed, optimizer and fixed epoch schedule as the outer experiment. Its training loop is separate to preserve the frozen experimental source hashes. The runtime verification reconstructs moderate seed-17 features directly from raw inputs for all seven excluded-match models, then reproduces their TCN probabilities and blended decoded labels exactly. It does not evaluate the final refit on its training labels.

Two fresh CLI invocations on an external tracking file copied into an input-only directory produce byte-identical 146,212-row CSVs and pass the official format validator. They read no reference arrays and perform no accuracy scoring. Full-pipeline runtime was 39.88 / 39.72 seconds while the checks ran concurrently on this CPU; these are engineering measurements, not a general hardware guarantee. Evidence: `outputs/tcn_research/final_smoke/verification.json`. The test CSVs are external examples, not competition submissions.

For a new research environment, create a Python 3.11 virtual environment, install `torch==2.8.0` from `https://download.pytorch.org/whl/cpu`, then install `solution/requirements-tcn-lock.txt` using that environment's interpreter. The already installed environment above is ready to use. The production `.venv` intentionally does not require PyTorch; it runs the 120 original tests and skips the ten optional TCN tests.
