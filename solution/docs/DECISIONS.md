# Project decisions

## Explicit public correctness evaluation — 1 November 2026

The user's subsequent comparison request also authorized scoring the frozen older baseline/event2/event4 predictions on the same public window. Results are in `outputs/public_correctness/model_comparison_20261101/`. The tiny observed public-window differences do not establish a full-match winner; no model/settings or registered submission choice were changed from these results.

The user requested measuring the new model on the official public five-minute reference. This is a scoped exception to the earlier external-results-only reporting instruction. Scored the existing `outputs/AOM_research_combined.csv` after verifying its frozen prediction/model hashes, using the supplied evaluator and 1,000 bootstrap draws. Kept the evaluation in `outputs/public_correctness/research_combined_20261101/`; no model, threshold, decoder setting, variant selection or output correction followed from these labels. Full private-match references remain unavailable and were not accessed. This is a correctness check, not training/tuning or a full-match accuracy claim.

## Four-direction accuracy work — 1 November 2026

The user explicitly requested tracking reliability, missing-ball learning, better boundaries and broader external validation. Complete fixed ablations rather than claim an untested architecture is best. Record the first detector's clean false-alarm failure, the clean-domain correction and the full corrected joint refit separately. Preserve original references and report four conventions; no label convention or decoder profile is selected to flatter new outer scores. The corrected combined candidate improves provider interval scores but does not settle the longer-possession semantic tradeoff. Keep the existing AOM archive and original variants, plus separate runnable research candidates. Team name AOM is confirmed; primary submission selection remains pending. No automatic publication/submission. See `RELIABILITY_RESULTS.md` and `RESEARCH_INFERENCE.md`.

## Team name confirmed — 1 November 2026

The user confirmed the registered team name as **AOM**, superseding the earlier deferral. Rebuilt the package at `outputs/submission/AOM_20261101/` with AOM in reports/manifests and `AOM.csv` / `AOM.md` filenames in every private variant folder. All three variants remain; no primary is selected. Earlier unnamed drafts are historical. No publication or submission is authorized by this name update.

## Draft delivery — 1 November 2026

The user explicitly requested leaving the forgotten registered team name pending and preserving all three variants for later selection. Use temporary project-slug filenames; do not invent a registered name or infer a primary from one reference convention. Prepared a public runtime/model ZIP and separate private CSV/report folders. Strict allowlist/checksum audits and fresh-environment extracted-runtime inference reproduce all candidates exactly without reference labels or scoring. All 103 tests pass. Public repository visibility/member registration/signed terms remain unconfirmed. This completes local implementation/draft delivery, not publication or actual submission. See `DELIVERY.md` for exact scope and final user actions. No commit, push, upload or submission was made.

## Step 1 authorization

The user approved Step 1 and authorized the agent to execute commands and install dependencies on 21 October 2026. The existing project-root `.venv` is reused. Step 1 initially created a local repository inside `solution/`; the subsequent Step 2 instruction supersedes that layout: use one root `.gitignore` and leave commits/pushes to the user. Previous local checkpoint metadata is preserved in ignored `.local-git/solution.git`.

## Challenge contract

- Tactical-video frame indices 0 through 199318, exactly 199319 CSV data rows.
- Labels H/A/D/X; primary ranking metric spellF1, first tie-break macroF1.
- Provided ATD frame IDs map to output frames by subtracting 1.
- Public five-minute ground truth is a contract/semantics check, not a reliable model-selection set.
- Initial strategy: tactical tracking and semantic tags with explicit alive/dead evidence and a structured temporal decoder.
- Fallback: geometry, possession persistence and a separate stoppage state machine.
- Primary system runs locally on CPU. Optional neural experiments may use the RTX 3050 after validation infrastructure exists.
- No solver implementation, asset extraction or baseline regeneration in Step 1.

## Validation policy — revised 28 October 2026

Use external fixtures with whole-match development/holdout splits and noisy-input stress tests. No human annotation of the supplied match. Current online rules supersede our older local snapshot; see `VALIDATION_UPDATE_2026-10-28.md` for sources and the full protocol.

The revised benchmark was implemented in Step 4 on J03WPY (development) and J03WQQ (validation). The seven-match split was fixed before scoring; sealed J03WN1 is untouched. Public challenge labels remain reserved for correctness checks. See `EXTERNAL_VALIDATION.md` for limitations before any model-selection claim.

## Administrative items still requiring user information

- Team name and confirmation of 3–5 members registered under that name by 10 November 2026.
- Review of the signed registration terms, which are not available in this workspace.
- Reviewer/time budget for external-data semantics and benchmark review.

These do not block the Step 1 environment checkpoint. Do not represent them as completed.

## Checkpoints and next step

The user handles all commits and pushes. Step 2 was explicitly authorized. Ignore policy is maintained only in the project-root `.gitignore`, including the provided source snapshot; its old ignore file was removed at the user's request. Prepared match assets are ignored. All other upstream source files should retain their original hashes.

No innovation-stage planning is in the current implementation scope. Proceed through the approved Initial Challenge steps only. Steps 3–4 were subsequently authorized and completed on 28 October: baseline reproduction and the initial external validation benchmark. Steps 5–6 were authorized and completed on 30 October; Step 7 requires the user's next instruction.

## Feature contract — 30 October 2026

Features use metres/seconds and carry validity masks; missing observations never become measured zeros. Motion uses same-team spatial correspondence rather than persistent slot IDs, with explicit ambiguity and speed filtering. This remains an estimate, especially around close players and replacements. Automatic tactical tags provide optional weak evidence; no equivalent tags are present on the external fixtures, so that branch still lacks external validation. Future-observation timing is explicitly offline. Fixed feature settings were not fitted to challenge labels. See `FEATURE_EXTRACTION.md`.

## First solver — 30 October 2026

Selected the `responsive` state-machine profile from three presets using development spellF1, weighted equally across clean/moderate/severe regimes. Validation used only the frozen winner, with no retuning. Keep the full valid challenge artifact as a fallback, alongside the unchanged official baseline. Improved temporal scores coexist with substantial noisy-input abstention and weaker noisy frame metrics; broader coverage and tag validation remain open. See `FALLBACK_SOLVER.md`. The structured decoder is Step 7 and has not started.

## Revised order after oracle diagnosis — 30 October 2026

The user's subsequent review supersedes the decoder-first next step. Oracle experiments are complete; learn stronger dead evidence and bidirectional ownership before decoder tuning. Both errors matter for spellF1 despite high team accuracy on committed frames. See `ORACLE_DIAGNOSIS.md` for definitions, results, gates and the new order. The requested all-seven leave-one-match-out protocol will replace the original sealed-fixture role when implemented; retain historical manifests and do nested training-fold selection. No public-window duration target becomes a model prior. External-only numeric reporting remains in force.

## First learned dead classifier — 30 October 2026

Completed the next authorized evidence checkpoint using four development fixtures and both original validation fixtures. Model/threshold selection used development match folds only; validation was frozen and no decoder was added. Selected frame weighting rather than spell weighting based on the prespecified development D-F1 criterion. High binary D accuracy does not make the unsmoothed ownership overlay a replacement for the fallback: it still overfragments and leaves ownership unknown, and noisy J03WR9 spellF1 regresses versus the previous rule solver. Preserve all candidates and the baseline; next work is gap-aware ownership. See `DEAD_BALL_MODEL.md`.

## Bidirectional ownership — 30 October 2026

Completed the next authorized checkpoint with labels excluded from gap-feature extraction. Selected spell-weighted conditional ownership and 0.8 s confirmation using development interval scores. All development folds used dead models trained on other fixtures, and validation retained the frozen all-four-match dead model. Zero playing-period X is achieved by making decisions with explicit low-confidence diagnostics, not by proving every hidden possession recovered. Noisy counts remain too high; audits identify many false-alive islands within reference stoppages. Next is temporal decoding over the stronger evidence. See `GAP_OWNERSHIP.md`; no validation-driven candidate substitution or threshold changes were made.

## Temporal decoding — 30 October 2026

Completed the next authorized checkpoint using cached out-of-fold probabilities and separate evaluation labels. Selected fixed Viterbi penalties 40/40/8 from the prespecified 25-profile grid on development spellF1. Retained the frozen winner after opened validation; the evidence-scaled ablation does not replace it on individual cases. No hard duration floor, public-window duration prior or new model fitting was introduced. Moderate interval scores and count gates improve, but undercounting and severe-noise merging remain. Cost sensitivity is explicitly not fully nested validation. Preserve previous systems; next work is broader nested validation and sensitivity testing, followed by final inference packaging. See `TEMPORAL_DECODER.md`. External-only numeric reporting remains in force.

## Seven-match validation — 30 October 2026

Completed a new, prespecified protocol that replaces the old sealed role with seven outer folds while preserving historical files and guards. Both heads exclude outer/inner holdouts during decoder-cost selection; fixed historical architecture choices remain a research-adaptation limitation. Contextual corruption and alternate labels are evaluation-only. Broader interval-score gains hold, but provider count gates fail on most fixtures and nested costs add negligible benefit over fixed 40/40/8. Do not claim count recovery from the more favorable absorbed labels or select a replacement based on individual outer cases. Next delivery is end-to-end inference/reproduction packaging with these limitations exposed, not a declaration of submission readiness. See `NESTED_VALIDATION.md`. No competition data or scores entered this checkpoint.

## Short-possession investigation — 30 October 2026

The user requested improvement before packaging. Completed two declared adaptive experiments: nested selection of physical-control residuals/lower switch costs on frozen heads, then a fixed boundary-sampling ownership-training ablation with six-match outer refits. The original remains the score-first winner; count-balanced control raises moderate count passes 2/7 → 3/7 with slightly lower mean spellF1 and worse absorbed-label scores. Boundary sampling fixes missing training-spell representation but worsens the provider objective, so it is not promoted. Keep all results, source hashes and experimental outputs; do not treat count-only progress as winning-score progress or claim a new independent test. See `SHORT_POSSESSIONS.md`. A switch-event learner remains an untested future hypothesis.

## Learned switch events — 1 November 2026

Completed the authorized event-learning checkpoint as a fixed seven-fold outer experiment. Train event models only on upstream probabilities cross-fitted to exclude both outer and training fixture; record and audit every exclusion. Preserve original ownership/dead evidence and use new predictions only for directional team-switch penalties. Report both fixed reward strengths, both label conventions, false switches and remaining count failures. Stronger rewards improve provider interval scores and moderate count coverage but regress absorbed-label scores, so do not substitute a universal default or claim hidden-match performance. Keep the original and both variants for explicit inference packaging; an independent controlled-possession reference remains needed to settle the semantic tradeoff. See `SWITCH_EVENTS.md`.

## Runnable inference bundle — 1 November 2026

Completed the authorized first of the three remaining delivery steps. Fit final heads on all seven external fixtures and train final switch evidence on six-match OOF upstream predictions excluding each training fixture. Keep the original development-selected global 40/40/8 profile; variant is a required CLI flag, not an outer-result-selected universal default. Store models/configuration/provenance in a hash/version-checked local bundle; inference accepts tracking only and does not depend on research outputs or target labels. Distinguish the held-out verification bundle from final all-seven deployment fits. Nine raw held-out CSV comparisons, final-bundle repeats, minimal-payload portability and all 95 tests pass. Full competition-input inference, broader final verification and submission distribution remain later work; the current checkpoint performs no competition inference/evaluation. See `INFERENCE.md`.

## Final acceptance — 1 November 2026

Completed the authorized verification step with a fresh isolated environment, fully pinned offline cached wheels and clean candidate-source copy. Fixed a genuine complex-coordinate/malformed-FPS input guard; no feature/model/decoder math or settings change. Preserve prior manifests/runtime source and record the input-only runtime update instead of silently bypassing compatibility checks. Fresh external CSVs reproduce exactly; negative/missing-data and full challenge-input format/repeatability checks pass without reference access/scoring. Report resource numbers only for external inputs. Keep explicit variant choices and all quality/convention limitations; final verification does not establish hidden accuracy or raw-to-model clean retraining. Only the technical/submission package remains, with no automatic publish/submit action. See `FINAL_VERIFICATION.md`.

## New improvement cycle — 2 November 2026

The user's "okay then start" authorizes implementation after the preceding research-only turn. Keep corrected-combined as the development base, without declaring a final submission winner. Completed current-model external oracle/decoder diagnosis, two new public Metrica sample transfer checks, and one fixed learned stop/restart-boundary experiment. The results support retaining smoothing and expose provider/timing limitations; they do not support globally lowering penalties or claiming a large architecture gain.

Do not promote the new boundary candidate: its primary moderate gain has an interval crossing zero, count passes are unchanged, and alternate contextual performance regresses. Preserve the measured severe-noise benefit as research evidence. The two additional fixtures have sparse event-derived binary truth and cannot be counted as complete controlled-possession benchmarks. No competition reference, input, model fitting or score is used in this cycle. See `STOPPAGE_RESEARCH.md` for results, provenance, tests and remaining TCN/tag/video work. AOM remains the registered team name; the user continues to handle commits/pushes and final submission.

## TCN evidence and candidate refit — 3 November 2026

Advance the prespecified equal joint-probability blend as the leading new research candidate. Its external provider spellF1 improves on all seven fixtures in all conditions, and the average absorbed-label score improves in all conditions. Keep the standalone TCN and owner-only blend as reported ablations; do not select a different blend per match or retune after inspecting outer scores. Count alignment on the longer-possession convention worsens and still requires independent controlled-possession evidence.

The audited candidate has been refit on all seven external fixtures and made runnable as one tracking-to-CSV pipeline. All accuracy claims refer to excluded-match fits, not final-refit training accuracy. Preserve existing bundle files and historical AOM submission archives; this is not automatic final submission selection or upload. No competition window was used in this architecture/training decision. See `TCN_RESEARCH.md`.

## Transfer result qualifies the TCN recommendation — 3 November 2026

The frozen two-Metrica-sample check finds significant native partial-accuracy regressions for the joint blend, largely from weaker dead emissions. Simulated missing-pass ownership improves, but this does not establish an overall cross-provider replacement. Retain the DFL evidence and runnable candidate; withdraw any universal-winner interpretation. No model or decoder setting changes in response to these samples.

A separately marked post-result ownership-only diagnostic mostly restores old dead recall and retains smaller gap gains. It uses the existing 50/50 DFL ablation, not a newly fitted coefficient, but selecting it after this inspection still requires fresh external evidence. Next checkpoint should freeze that alternative and test another external fixture; do not call the reused samples an independent validation of the choice. Preserve prior AOM files, the root ignore policy and the user's ownership of commits/pushes. See `TCN_TRANSFER.md`.

## Fresh ownership-only evidence — 4 November 2026

Completed the frozen Game 3 test authorized by the user's next-step request. The candidate and model identities were recorded before downloading this new external fixture; the metadata/event adapter was finalized before inference. No retraining or settings search occurred. The primary moderate missing-pass endpoint improves 84.13% to 87.00%, with a paired conditional interval excluding zero, while native dead recall remains similar. This strengthens the case for separately packaging ownership-only fusion as the next runnable candidate.

Do not describe the result as full-match accuracy, a complete-possession win, formal native non-inferiority or a final submission decision. Only 46.99% of frames have conservative reference labels, native partial-accuracy uncertainty crosses zero, and ghost-exposed partial accuracy slightly regresses. No model/default/submission file changes occur in this checkpoint. Preserve the original and full-joint candidates, avoid tuning on the now-inspected Game 3, and keep seeking independent complete labels. See `OWNER_TRANSFER_GAME3.md` for results and audited provenance.
