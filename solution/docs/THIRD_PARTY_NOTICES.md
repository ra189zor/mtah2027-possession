# Attribution and dependency provenance

## Provided challenge source

The runtime uses the supplied `games.py`, `evaluation/possession_metrics.py` and `evaluation/validate_submission.py` from [Metrica Sports' More Than A Hack repository](https://github.com/metrica-sports/more-than-a-hack-2027). Their MIT notice and scope are retained verbatim in `more-than-a-hack-2027-main repo/LICENSE`. This does not license or redistribute the challenge match assets. No new licence is assigned here to project-owned source.

## External training/reference data

Credit Deutsche Fußball Liga (DFL), Sportec Solutions and Bassek et al. (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data 12, 195, [DOI 10.1038/s41597-025-04505-y](https://doi.org/10.1038/s41597-025-04505-y). The dataset is published under CC BY 4.0. We used the PySport mirror at revision `a715a38dfbaf5f58e431727c2b78d174101a703c`.

Our changes include coordinate conversion, anonymous team slots, derived H/A/D/X references from provider codes, separate model-facing input arrays, seeded input-only corruption, feature extraction and fitted models. Raw downloads, label arrays and frame-level external predictions are not included in the public runtime archive. Full transformation notes remain in `DATA_PROVENANCE.md` / `solution/docs/EXTERNAL_DATA_SOURCES.md`.

The 2 November external transfer experiment additionally uses [Metrica Sports public sample tracking/events](https://github.com/metrica-sports/sample-data) at revision `e706dd506b360d69d9d123d5b8026e7294b13996`. Credit Metrica Sports. Its README asks public users to acknowledge the source; this project does not assign an additional data licence. The two anonymized sample games and event-derived partial binary references remain local, ignored and excluded from deployment archives. They are not training data for the existing fitted models. See `solution/docs/EXTERNAL_DATA_SOURCES.md`.

## Python dependencies

Exact versions are in `solution/requirements-lock.txt`. `DEPENDENCIES.json` records installed distribution metadata and licence identifiers/hints for the pinned packages. Dependency wheels/binaries are not redistributed in this runtime archive; installation obtains them separately, with their own licences. The metadata inventory is provenance, not a blanket licence grant for the project or a substitute for individual package notices.

## Optional TCN research dependencies

The optional TCN experiment additionally uses CPU PyTorch and its dependencies in a separate research environment. `solution/requirements-tcn-lock.txt` pins its versions; `solution/docs/TCN_DEPENDENCIES.json` records the additional distributions' installed licence metadata and source URLs. Wheels come from the official PyTorch CPU index, with per-install reports retained locally. These research dependencies and checkpoints are not included in the existing runtime archive. See `TCN_RESEARCH.md` for experiment provenance and reproduction.

## Archive boundaries

The public runtime archive contains code, documentation and the externally trained model payload only. It excludes the licensed match assets, reference labels, videos, tracking exports, independent research caches, registration terms, credentials and all possession CSV candidates. Candidate CSVs and their matching technical reports are kept in separate local private folders for the registered submission process.
