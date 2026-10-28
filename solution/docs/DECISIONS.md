# Project decisions

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

No innovation-stage planning is in the current implementation scope. Proceed through the approved Initial Challenge steps only. Steps 3–4 were subsequently authorized and completed on 28 October: baseline reproduction and the initial external validation benchmark. Step 5 (feature extraction) requires the user's explicit instruction.
