# PS5-like system UI overlay (2026-07-30)

## Goal

Replace macOS AppKit `NSPanel` dialogs/toasts with a present-path overlay that looks and behaves recognizably like PS5 system UI, without Swift/AppKit UI chrome.

## Decisions

- Composite over the guest scanout in the Metal present pass (approach 1).
- Recognizably PS5-like layout (dim + dark panel + blue accent), not pixel-perfect system fonts.
- Cover message, save-data, trophy/notification toasts, error, progress, IME, and generic system kinds.
- `REMU_AUTO_DISMISS_DIALOGS` / headless: no overlay; auto-cancel (unchanged).

## Architecture

1. `remu-gpu` owns kinds, poll state, and `rasterize_system_ui` (platform-neutral RGBA).
2. `remu-gpu-metal` `host_ui` stores active dialog/toast, polls pad/keyboard, rebuilds the framebuffer.
3. Present pass uploads the overlay texture when dirty and alpha-blends it after the guest blit.
4. HLE (`remu-runtime/dialog.rs`) publishes `ShowHostDialog` / `ShowHostToast` as before; IME open shows an IME panel.

## Out of scope

- Pixel-perfect PS5 assets/fonts
- Full IME text entry into guest buffers
- Injecting UI draws into guest GPU memory
