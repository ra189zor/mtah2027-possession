# AOM code and weights publication — 9 November 2026

Ownership-only is the final selected candidate. Its matching source and fitted weights are publicly available in the [AOM v1 release](https://github.com/ra189zor/mtah2027-possession/releases/tag/aom-owner-v1).

Release assets:

- `AOM_owner_runtime_v1.zip`: verified inference code, three weight payloads, manifests, locks and attribution.
- `AOM_owner_runtime_v1.sha256`: ZIP checksum.
- `FINAL_CANDIDATE.md`: dated final selection and reproduction instructions.

The release tag points to the existing commit `6273060e9384da72747ca0a95ada6494f7b94d24`. No commit or git push was performed. Documentation/configuration changes in this workspace remain uncommitted. Use the attached runtime ZIP for reproduction; GitHub's automatic source downloads omit ignored model weights.

Verification passed: repository public visibility, non-draft release, all three assets uploaded, anonymous runtime download, and SHA-256 equality with the previously tested archive (`4373a111611d78e2734bae80897ba650e5109f0cba29facfccafd52e5cf6d6ee`). Before upload, every archived file hash and the exact allowlist were checked. The existing selected manifests and weights were unchanged.

Local evidence: `outputs/owner_candidate/published_release_20261109.json`, `published_release_20261109.jpg`, and `AOM_owner_runtime_v1.public_download.zip`. All remain under the existing root ignore policy. The downloaded bytes match the previously verified runtime, so this publication step did not repeat model training, inference or accuracy evaluation.

No licensed match data, reference labels or prediction CSV was uploaded. The selected private submission artifacts remain under `outputs/submission/AOM_owner_20261104/private_candidate/`; historical draft manifests and technical report remain unmodified for audit integrity. This publication record and `FINAL_CANDIDATE.md` supersede their pending-selection/publication statements. Registered competition submission has not been performed.
