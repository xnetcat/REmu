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
11. [`13-rebased-export-format-dialog.png`](13-rebased-export-format-dialog.png) — clean trophies and
    keyboard dialog after rebasing on `main` and carrying the exact guest color-export format into
    the D3D12 shader path. The black underlay is an improvement over the earlier corrupt teal frame,
    but it is not evidence that the game scene rendered.
12. [`14-rebased-export-format-post-dialog.png`](14-rebased-export-format-post-dialog.png) — sustained
    post-dialog execution from the same rebased build. The cyan surface and incomplete pale shape
    confirm that matching the export format did not by itself fix presentation correctness.
13. [`15-present-correlation-dialog.png`](15-present-correlation-dialog.png) — a timing-instrumented
    run with the common dialog composited over the corrupt cyan guest frame. The dialog remains
    readable, isolating the defect below the host UI compositor.
14. [`16-present-correlation-post-dialog.png`](16-present-correlation-post-dialog.png) — the same run
    immediately after one controlled Enter press. Partial pale UI geometry is present, but no text,
    title-menu controls, or game scene are readable.

## Current boundary

These captures do **not** prove a usable title menu or gameplay. The first fixed run moved the DRED
boundary from compute 151 to a data-dependent unbounded loop at compute 1,688. Generalizing finite
loop bounds to all compute HLSL moved the next 180-second validation past that point to compute 1,865
and more than 1,000 draws, with no D3D12 device removal before the timeout. The formerly poisoned
indirect dimensions repeatedly resolve to 120 and 4,200 groups.

The remaining visible defect is presentation/rendering correctness: after dismissing the common
dialog, the title draws cyan surfaces and partial menu-shaped gradients without usable text or game
content. The rebased exact-export-format run still reaches and cleanly renders the common dialog,
then returns to the same cyan/incomplete boundary. A later timing run confirmed that both registered
3840x2160 display buffers are actively redrawn and presented; this is not a stale DXGI back buffer.
That run completed 69 flips and 707 GPU submissions in 120 seconds without device removal while the
same evaluator-truncated compute shader continued to recur. Gameplay screenshots must wait until
that separate graphics path is corrected. Generated
HLSL, shader IR, binding traces, and DRED logs remain in ignored `.remu/logs/` and `target/`
directories because game-derived shader material must not be committed.
