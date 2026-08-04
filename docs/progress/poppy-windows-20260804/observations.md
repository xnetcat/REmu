# Poppy Playtime Chapter 1 — Windows/D3D12 progress evidence

Captured on 2026-08-04 from Windows release builds on an NVIDIA GeForce GTX 1050. Diagnostic runs
set `REMU_AVPLAYER_WALL_CLOCK_CATCH_UP=1` to reach the post-intro workload quickly. Controlled input
sent Enter only to the focused `REMu | Direct3D 12` window.

## Captured stages

1. [`01-mob-entertainment-intro.png`](01-mob-entertainment-intro.png) — native D3D12 window and Mob
   Entertainment opening movie.
2. [`02-keyboard-dialog.png`](02-keyboard-dialog.png) — keyboard/common dialog over a live frame.
3. [`03-saved-data-dialog.png`](03-saved-data-dialog.png) — trophy notification and save-data prompt.
4. [`04-post-intro-last-frame.png`](04-post-intro-last-frame.png) — original device-removal boundary.
5. [`05-title-menu-loop-guard-diagnostic.png`](05-title-menu-loop-guard-diagnostic.png) — menu-shaped
   diagnostic frame after the first loop guard; text/content are missing, so this is not a working
   title-menu capture.
6. [`06-after-first-menu-selection-diagnostic.png`](06-after-first-menu-selection-diagnostic.png) —
   black diagnostic frame after one controlled selection; this is not gameplay evidence.
7. [`07-evaluator-loop-live-progress.png`](07-evaluator-loop-live-progress.png) and
   [`08-after-keyboard-dialog-clean-run.png`](08-after-keyboard-dialog-clean-run.png) — evaluator-loop
   investigation frames; dialog rendering works but the underlying game frame is corrupt.
8. [`09-authoritative-version-clean-run.png`](09-authoritative-version-clean-run.png) — clean common
   dialog in the run that first forwarded the complete reset/classifier/expander GPU version chain.
9. [`10-after-keyboard-authoritative-run.png`](10-after-keyboard-authoritative-run.png) and
   [`11-sustained-post-dialog-frame.png`](11-sustained-post-dialog-frame.png) — sustained post-dialog
   execution with valid 120- and 4,200-group indirect dispatches; cyan/incomplete presentation
   remains a graphics defect.
10. [`12-general-loop-watchdog-run.png`](12-general-loop-watchdog-run.png) — common dialog during the
    generalized compute-HLSL watchdog validation.

## Current boundary

These captures do **not** prove a usable title menu or gameplay. The first fixed run moved the DRED
boundary from compute 151 to a data-dependent unbounded loop at compute 1,688. Generalizing finite
loop bounds to all compute HLSL moved the next 180-second validation past that point to compute 1,865
and more than 1,000 draws, with no D3D12 device removal before the timeout. The formerly poisoned
indirect dimensions repeatedly resolve to 120 and 4,200 groups.

The remaining visible defect is presentation/rendering correctness: after dismissing the common
dialog, the title draws cyan surfaces and partial menu-shaped gradients without usable text or game
content. Gameplay screenshots must wait until that separate graphics path is corrected. Generated
HLSL, shader IR, binding traces, and DRED logs remain in ignored `.remu/logs/` and `target/`
directories because game-derived shader material must not be committed.
