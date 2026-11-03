# External validation data: source and attribution

The external benchmark uses the public **Sportec Open DFL / IDSSE dataset**, provided by Deutsche Fußball Liga and collected by Sportec Solutions with TRACAB optical tracking. It is distinct from the restricted Barcelona/Athletic challenge assets.

- Paper: Bassek et al. (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data 12, 195. DOI: [10.1038/s41597-025-04505-y](https://doi.org/10.1038/s41597-025-04505-y).
- Original data: [Figshare DOI 10.6084/m9.figshare.28196177](https://doi.org/10.6084/m9.figshare.28196177).
- Download mirror: [PySport IDSSE on Hugging Face](https://huggingface.co/datasets/pysport/idsse-data).
- Licence reported by the paper/mirror: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
- Pinned mirror revision: `a715a38dfbaf5f58e431727c2b78d174101a703c`.
- Schema references: [floodlight DFL documentation](https://floodlight.readthedocs.io/en/latest/modules/io/dfl.html) and [kloppy Sportec documentation](https://kloppy.pysport.org/user-guide/loading-data/sportec/).

Our adapter is independently implemented with Python's standard-library XML parser; neither floodlight nor kloppy is required for this benchmark. The metric and reference smoothing functions are imported from the supplied challenge repository, retaining its existing MIT licence and attribution.

## Changes made for this benchmark

We map provider ownership and ball-state codes to H/A/D/X, convert centred coordinates to corner-origin metres, arrange player observations into anonymous team slots, separate input arrays from target labels, and create seeded noise scenarios. Raw speed, identity, event and ball-state fields are not included as model inputs. The prepared arrays and their hashes are stored locally in ignored paths.

Ownership codes mean provider-recorded ownership; their convention is not assumed identical to Metrica's controlled possession. External score values must not be presented as hidden challenge scores.

The mirror's README lists two match IDs that differ from its actual file inventory. The fixed split uses the seven IDs present in the inventory and in kloppy's documented table: J03WMX, J03WN1, J03WPY, J03WOH, J03WOY, J03WQQ and J03WR9. The inventory and source revision are recorded locally; no file names are guessed from the stale card alone.

Raw external files, transformed arrays and generated benchmark reports are not committed. Keep this attribution with any future publication of results or code, and preserve the licence and modification notice if distributing derivatives later.

30 October 2026 expansion: the seven-match protocol opened J03WN1 (Bochum–Leverkusen) at the same pinned revision, replacing its historical sealed role without changing the old split file. It adds input-only contextual corruption and an evaluation-only short-opposition-spell label sensitivity; both transformations and their limitations are documented in `NESTED_VALIDATION.md`. Original provider references remain available and are reported alongside the alternate convention.

## Public Metrica sample transfer check — 2 November 2026

Credit [Metrica Sports public sample data](https://github.com/metrica-sports/sample-data), revision `e706dd506b360d69d9d123d5b8026e7294b13996`, Sample Games 1 and 2. The publisher describes the fixtures as anonymized, synchronized tracking/events on a 105 x 68 m pitch. Its README requests source acknowledgement for public use; no CC licence is assumed. Raw CSVs, the publisher's event-definition PDF and local derivatives remain ignored, with Git-blob and SHA-256 verification recorded in `data/external/metrica_public/source_manifest.json`.

These fixtures were not used for model fitting. Tracking is converted to anonymous team-slot arrays in metres at native 25 Hz. Sparse events yield explicitly partial binary references: pass interiors are alive; unambiguous ball-out/shot-out/goal to next explicit restart intervals are dead. Uncertain spans are excluded, and guarded dead interiors are reported separately. No full possession truth is invented. Nearly perfect status/missing-ball association in the labelled subset limits realism for live-play occlusion testing. See `STOPPAGE_RESEARCH.md` for coverage, derivation and results.

3 November 2026 extension: `transfer_validation.py` derives partial H/A ownership only inside eligible recorded same-team PASS interiors, excluding event endpoints, interruption/identity/period problems, opposing possession actions and conflicts. No ownership labels are filled across event gaps. The same two fixtures remain absent from training, but earlier binary results were already examined. Input-only synthetic dropout/ghost/wrong-team stress is evaluated separately from native tracking. These remain selective event-derived references, not complete controlled-possession truth; no complete spellF1 is calculated. See `TCN_TRANSFER.md`, including the distinction between the frozen primary comparison and the post-result exploratory diagnostic.
