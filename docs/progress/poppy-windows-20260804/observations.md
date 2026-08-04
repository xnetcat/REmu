# Poppy Playtime Chapter 1 — Windows/D3D12 launch evidence

Captured on 2026-08-04 from the Windows release build at commit `ddb390e` on an NVIDIA GeForce
GTX 1050. The accelerated evidence runs set `REMU_AVPLAYER_WALL_CLOCK_CATCH_UP=1` so the opening
movie could reach the post-intro workload quickly. A separate controlled-input run sent Enter to
the focused REmu window to dismiss the dialogs.

## Captured stages

1. [`01-mob-entertainment-intro.png`](01-mob-entertainment-intro.png) — the title creates a native
   `REMu | Direct3D 12` window and presents the Mob Entertainment opening movie.
2. [`02-keyboard-dialog.png`](02-keyboard-dialog.png) — REmu renders the title's keyboard/common
   dialog over the live game frame.
3. [`03-saved-data-dialog.png`](03-saved-data-dialog.png) — the trophy notification and save-data
   prompt render, accept host keyboard input, and return a result to the guest.
4. [`04-post-intro-last-frame.png`](04-post-intro-last-frame.png) — the last distinct frame after
   the dialogs and opening movie, immediately before the first substantial post-intro compute
   workload removes the D3D12 device.

## Current boundary

This capture does **not** prove a title menu or gameplay. Three fresh runs reproduced the same
boundary: compute sequence 151 executes a 120 x 1 x 1 group dispatch with a 16 x 16 x 1 local size,
then `GetDeviceRemovedReason` reports `DXGI_ERROR_DEVICE_HUNG` (`0x887A0006`). The generated shader
contains a guest-controlled loop with four texture samples per iteration; its loop bound comes from
a translated wave64 maximum reduction. The exact HLSL, shader IR, bindings, dispatch dimensions,
and DRED diagnostics are retained under ignored `target/` diagnostics and are not committed because
they are derived from proprietary game code.

The next validation milestone is a real Poppy title-menu screenshot after correcting that general
wave64/loop translation defect. Gameplay screenshots must wait until the menu can be crossed without
device removal.
