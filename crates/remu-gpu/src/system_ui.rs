// SPDX-License-Identifier: GPL-2.0-or-later

//! Platform-neutral PS5-like system UI scene and software rasterizer.
//!
//! Layout and pixels live here so every GPU backend can composite the same overlay. The Metal
//! presenter uploads the RGBA buffer and draws it after the guest scanout; headless / auto-dismiss
//! paths never call into this module.

use crate::{HostDialogButtons, HostDialogKind, HostDialogRequest, HostToastRequest};

/// Logical overlay resolution. Presenters scale this to the drawable.
pub const SYSTEM_UI_WIDTH: u32 = 1280;
pub const SYSTEM_UI_HEIGHT: u32 = 720;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SystemUiFramebuffer {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl SystemUiFramebuffer {
    #[must_use]
    pub fn empty() -> Self {
        let width = SYSTEM_UI_WIDTH;
        let height = SYSTEM_UI_HEIGHT;
        Self {
            width,
            height,
            rgba: vec![0; (width * height * 4) as usize],
        }
    }

    #[must_use]
    pub fn is_transparent(&self) -> bool {
        self.rgba.chunks_exact(4).all(|p| p[3] == 0)
    }
}

/// Rasterizes a modal dialog and optional toast into one overlay buffer.
#[must_use]
pub fn rasterize_system_ui(
    dialog: Option<&HostDialogRequest>,
    toast: Option<&HostToastRequest>,
) -> SystemUiFramebuffer {
    let mut fb = SystemUiFramebuffer::empty();
    if let Some(dialog) = dialog {
        draw_dialog(&mut fb, dialog);
    }
    if let Some(toast) = toast {
        draw_toast(&mut fb, toast);
    }
    fb
}

fn draw_dialog(fb: &mut SystemUiFramebuffer, request: &HostDialogRequest) {
    fill_rect(fb, 0, 0, fb.width, fb.height, [0, 0, 0, 170]);

    let panel_w = 640u32;
    let panel_h = match request.kind {
        HostDialogKind::Ime => 360,
        HostDialogKind::Progress => 220,
        _ => 280,
    };
    let panel_x = (fb.width - panel_w) / 2;
    let panel_y = (fb.height - panel_h) / 2;

    // Soft shadow
    fill_rounded_rect(
        fb,
        panel_x + 6,
        panel_y + 8,
        panel_w,
        panel_h,
        18,
        [0, 0, 0, 90],
    );
    fill_rounded_rect(
        fb,
        panel_x,
        panel_y,
        panel_w,
        panel_h,
        18,
        [32, 34, 40, 245],
    );
    // Top accent bar (PS-ish blue)
    fill_rect(fb, panel_x, panel_y, panel_w, 4, [0, 117, 207, 255]);

    let title = kind_title(request.kind, &request.title);
    draw_text(
        fb,
        panel_x + 36,
        panel_y + 28,
        &title,
        2,
        [240, 242, 245, 255],
    );
    draw_wrapped_text(
        fb,
        panel_x + 36,
        panel_y + 72,
        panel_w - 72,
        &request.message,
        2,
        [190, 194, 202, 255],
        6,
    );

    if matches!(request.kind, HostDialogKind::Progress) {
        let bar_x = panel_x + 36;
        let bar_y = panel_y + panel_h - 100;
        let bar_w = panel_w - 72;
        fill_rounded_rect(fb, bar_x, bar_y, bar_w, 12, 6, [18, 20, 24, 255]);
        fill_rounded_rect(fb, bar_x, bar_y, bar_w * 2 / 5, 12, 6, [0, 117, 207, 255]);
    }

    if matches!(request.kind, HostDialogKind::Ime) {
        let field_x = panel_x + 36;
        let field_y = panel_y + 150;
        let field_w = panel_w - 72;
        fill_rounded_rect(fb, field_x, field_y, field_w, 40, 8, [18, 20, 24, 255]);
        draw_text(
            fb,
            field_x + 14,
            field_y + 12,
            "Type on the host keyboard…",
            2,
            [120, 124, 132, 255],
        );
    }

    let hint = button_hint(request.buttons);
    draw_text(
        fb,
        panel_x + 36,
        panel_y + panel_h - 36,
        &hint,
        1,
        [140, 144, 152, 255],
    );

    draw_buttons(fb, panel_x, panel_y, panel_w, panel_h, request.buttons);
}

fn draw_toast(fb: &mut SystemUiFramebuffer, request: &HostToastRequest) {
    let toast_w = 520u32;
    let toast_h = 88u32;
    let toast_x = (fb.width - toast_w) / 2;
    let toast_y = 36u32;
    fill_rounded_rect(
        fb,
        toast_x + 4,
        toast_y + 4,
        toast_w,
        toast_h,
        14,
        [0, 0, 0, 80],
    );
    fill_rounded_rect(
        fb,
        toast_x,
        toast_y,
        toast_w,
        toast_h,
        14,
        [28, 30, 36, 235],
    );
    fill_rect(
        fb,
        toast_x,
        toast_y,
        6,
        toast_h,
        accent_for_kind(request.kind),
    );
    let title = kind_title(request.kind, &request.title);
    draw_text(
        fb,
        toast_x + 24,
        toast_y + 18,
        &title,
        2,
        [240, 242, 245, 255],
    );
    draw_text(
        fb,
        toast_x + 24,
        toast_y + 48,
        &truncate(&request.message, 48),
        1,
        [180, 184, 192, 255],
    );
}

fn draw_buttons(
    fb: &mut SystemUiFramebuffer,
    panel_x: u32,
    panel_y: u32,
    panel_w: u32,
    panel_h: u32,
    buttons: HostDialogButtons,
) {
    let labels: &[&str] = match buttons {
        HostDialogButtons::Ok => &["OK"],
        HostDialogButtons::OkCancel => &["Cancel", "OK"],
        HostDialogButtons::YesNo => &["No", "Yes"],
    };
    let btn_w = 120u32;
    let btn_h = 36u32;
    let gap = 16u32;
    let total = labels.len() as u32 * btn_w + (labels.len() as u32 - 1) * gap;
    let mut x = panel_x + panel_w - 36 - total;
    let y = panel_y + panel_h - 88;
    for (index, label) in labels.iter().enumerate() {
        let primary = index + 1 == labels.len();
        let fill = if primary {
            [0, 117, 207, 255]
        } else {
            [48, 52, 60, 255]
        };
        fill_rounded_rect(fb, x, y, btn_w, btn_h, 8, fill);
        let text_w = (label.len() as u32) * 8 * 2;
        let text_x = x + btn_w.saturating_sub(text_w) / 2;
        draw_text(fb, text_x, y + 10, label, 2, [255, 255, 255, 255]);
        x += btn_w + gap;
    }
}

fn kind_title(kind: HostDialogKind, fallback: &str) -> String {
    if !fallback.is_empty()
        && !matches!(
            kind,
            HostDialogKind::Message
                | HostDialogKind::SaveData
                | HostDialogKind::Error
                | HostDialogKind::Progress
                | HostDialogKind::Ime
        )
    {
        return fallback.to_owned();
    }
    let default = match kind {
        HostDialogKind::Message => "Message",
        HostDialogKind::SaveData => "Saved Data",
        HostDialogKind::Trophy => "Trophy",
        HostDialogKind::Notification => "Notification",
        HostDialogKind::Error => "Error",
        HostDialogKind::Progress => "Please Wait",
        HostDialogKind::Ime => "Keyboard",
        HostDialogKind::System => "System",
    };
    if fallback.is_empty() {
        default.to_owned()
    } else {
        fallback.to_owned()
    }
}

fn accent_for_kind(kind: HostDialogKind) -> [u8; 4] {
    match kind {
        HostDialogKind::Trophy => [212, 175, 55, 255],
        HostDialogKind::Error => [220, 64, 64, 255],
        HostDialogKind::Progress => [0, 117, 207, 255],
        _ => [0, 117, 207, 255],
    }
}

fn button_hint(buttons: HostDialogButtons) -> String {
    match buttons {
        HostDialogButtons::Ok => "Cross / Enter : OK".to_owned(),
        HostDialogButtons::OkCancel => "Cross / Enter : OK    Circle / Esc : Cancel".to_owned(),
        HostDialogButtons::YesNo => "Cross / Enter : Yes    Circle / Esc : No".to_owned(),
    }
}

fn truncate(text: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for (index, ch) in text.chars().enumerate() {
        if index >= max_chars {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out
}

fn fill_rect(fb: &mut SystemUiFramebuffer, x: u32, y: u32, w: u32, h: u32, rgba: [u8; 4]) {
    let x1 = x.min(fb.width);
    let y1 = y.min(fb.height);
    let x2 = x.saturating_add(w).min(fb.width);
    let y2 = y.saturating_add(h).min(fb.height);
    for py in y1..y2 {
        for px in x1..x2 {
            blend_pixel(fb, px, py, rgba);
        }
    }
}

fn fill_rounded_rect(
    fb: &mut SystemUiFramebuffer,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    radius: u32,
    rgba: [u8; 4],
) {
    let r = radius.min(w / 2).min(h / 2);
    let x1 = x.min(fb.width);
    let y1 = y.min(fb.height);
    let x2 = x.saturating_add(w).min(fb.width);
    let y2 = y.saturating_add(h).min(fb.height);
    let r2 = i64::from(r * r);
    for py in y1..y2 {
        for px in x1..x2 {
            let dx = if px < x + r {
                i64::from(x + r - px)
            } else if px >= x + w - r {
                i64::from(px - (x + w - r - 1))
            } else {
                0
            };
            let dy = if py < y + r {
                i64::from(y + r - py)
            } else if py >= y + h - r {
                i64::from(py - (y + h - r - 1))
            } else {
                0
            };
            if dx * dx + dy * dy <= r2 {
                blend_pixel(fb, px, py, rgba);
            }
        }
    }
}

fn blend_pixel(fb: &mut SystemUiFramebuffer, x: u32, y: u32, src: [u8; 4]) {
    let index = ((y * fb.width + x) * 4) as usize;
    if index + 3 >= fb.rgba.len() {
        return;
    }
    let dst = &mut fb.rgba[index..index + 4];
    let sa = u32::from(src[3]);
    if sa == 0 {
        return;
    }
    if sa == 255 {
        dst.copy_from_slice(&src);
        return;
    }
    let inv = 255 - sa;
    for i in 0..3 {
        dst[i] = ((u32::from(src[i]) * sa + u32::from(dst[i]) * inv) / 255) as u8;
    }
    let da = u32::from(dst[3]);
    dst[3] = (sa + da * inv / 255).min(255) as u8;
}

fn draw_wrapped_text(
    fb: &mut SystemUiFramebuffer,
    x: u32,
    y: u32,
    max_width: u32,
    text: &str,
    scale: u32,
    rgba: [u8; 4],
    max_lines: u32,
) {
    let char_w = 8 * scale;
    let char_h = 8 * scale + 4;
    let chars_per_line = (max_width / char_w).max(1) as usize;
    let mut line_y = y;
    let mut lines = 0u32;
    for line in wrap_text(text, chars_per_line) {
        if lines >= max_lines {
            break;
        }
        draw_text(fb, x, line_y, &line, scale, rgba);
        line_y += char_h;
        lines += 1;
    }
}

fn wrap_text(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if word.len() > width {
            if !current.is_empty() {
                lines.push(std::mem::take(&mut current));
            }
            for chunk in word.chars().collect::<Vec<_>>().chunks(width) {
                lines.push(chunk.iter().collect());
            }
            continue;
        }
        if current.is_empty() {
            current.push_str(word);
        } else if current.len() + 1 + word.len() <= width {
            current.push(' ');
            current.push_str(word);
        } else {
            lines.push(std::mem::take(&mut current));
            current.push_str(word);
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

fn draw_text(fb: &mut SystemUiFramebuffer, x: u32, y: u32, text: &str, scale: u32, rgba: [u8; 4]) {
    let mut cursor_x = x;
    for ch in text.chars() {
        let glyph = glyph_bitmap(ch);
        for row in 0..8u32 {
            let bits = glyph[row as usize];
            for col in 0..8u32 {
                // FONT8X8 stores the leftmost pixel in the LSB of each row byte.
                if bits & (1 << col) != 0 {
                    fill_rect(
                        fb,
                        cursor_x + col * scale,
                        y + row * scale,
                        scale,
                        scale,
                        rgba,
                    );
                }
            }
        }
        cursor_x += 8 * scale;
        if cursor_x >= fb.width {
            break;
        }
    }
}

/// Compact 8×8 ASCII glyphs (public-domain style bitmap font).
fn glyph_bitmap(ch: char) -> [u8; 8] {
    let code = ch as u32;
    if (32..127).contains(&code) {
        FONT8X8[(code - 32) as usize]
    } else {
        FONT8X8[0]
    }
}

// Space through tilde — each row is one byte, MSB = leftmost pixel.
const FONT8X8: [[u8; 8]; 95] = [
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // space
    [0x18, 0x3C, 0x3C, 0x18, 0x18, 0x00, 0x18, 0x00], // !
    [0x36, 0x36, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // "
    [0x36, 0x36, 0x7F, 0x36, 0x7F, 0x36, 0x36, 0x00], // #
    [0x0C, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x0C, 0x00], // $
    [0x00, 0x63, 0x33, 0x18, 0x0C, 0x66, 0x63, 0x00], // %
    [0x1C, 0x36, 0x1C, 0x6E, 0x3B, 0x33, 0x6E, 0x00], // &
    [0x06, 0x06, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00], // '
    [0x18, 0x0C, 0x06, 0x06, 0x06, 0x0C, 0x18, 0x00], // (
    [0x06, 0x0C, 0x18, 0x18, 0x18, 0x0C, 0x06, 0x00], // )
    [0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00], // *
    [0x00, 0x0C, 0x0C, 0x3F, 0x0C, 0x0C, 0x00, 0x00], // +
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x06], // ,
    [0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00, 0x00], // -
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x00], // .
    [0x60, 0x30, 0x18, 0x0C, 0x06, 0x03, 0x01, 0x00], // /
    [0x3E, 0x63, 0x73, 0x7B, 0x6F, 0x67, 0x3E, 0x00], // 0
    [0x0C, 0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x3F, 0x00], // 1
    [0x1E, 0x33, 0x30, 0x1C, 0x06, 0x33, 0x3F, 0x00], // 2
    [0x1E, 0x33, 0x30, 0x1C, 0x30, 0x33, 0x1E, 0x00], // 3
    [0x38, 0x3C, 0x36, 0x33, 0x7F, 0x30, 0x78, 0x00], // 4
    [0x3F, 0x03, 0x1F, 0x30, 0x30, 0x33, 0x1E, 0x00], // 5
    [0x1C, 0x06, 0x03, 0x1F, 0x33, 0x33, 0x1E, 0x00], // 6
    [0x3F, 0x33, 0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x00], // 7
    [0x1E, 0x33, 0x33, 0x1E, 0x33, 0x33, 0x1E, 0x00], // 8
    [0x1E, 0x33, 0x33, 0x3E, 0x30, 0x18, 0x0E, 0x00], // 9
    [0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x00], // :
    [0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x06], // ;
    [0x18, 0x0C, 0x06, 0x03, 0x06, 0x0C, 0x18, 0x00], // <
    [0x00, 0x00, 0x3F, 0x00, 0x00, 0x3F, 0x00, 0x00], // =
    [0x06, 0x0C, 0x18, 0x30, 0x18, 0x0C, 0x06, 0x00], // >
    [0x1E, 0x33, 0x30, 0x18, 0x0C, 0x00, 0x0C, 0x00], // ?
    [0x3E, 0x63, 0x7B, 0x7B, 0x7B, 0x03, 0x1E, 0x00], // @
    [0x0C, 0x1E, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x00], // A
    [0x3F, 0x66, 0x66, 0x3E, 0x66, 0x66, 0x3F, 0x00], // B
    [0x3C, 0x66, 0x03, 0x03, 0x03, 0x66, 0x3C, 0x00], // C
    [0x1F, 0x36, 0x66, 0x66, 0x66, 0x36, 0x1F, 0x00], // D
    [0x7F, 0x46, 0x16, 0x1E, 0x16, 0x46, 0x7F, 0x00], // E
    [0x7F, 0x46, 0x16, 0x1E, 0x16, 0x06, 0x0F, 0x00], // F
    [0x3C, 0x66, 0x03, 0x03, 0x73, 0x66, 0x7C, 0x00], // G
    [0x33, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x33, 0x00], // H
    [0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00], // I
    [0x78, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E, 0x00], // J
    [0x67, 0x66, 0x36, 0x1E, 0x36, 0x66, 0x67, 0x00], // K
    [0x0F, 0x06, 0x06, 0x06, 0x46, 0x66, 0x7F, 0x00], // L
    [0x63, 0x77, 0x7F, 0x7F, 0x6B, 0x63, 0x63, 0x00], // M
    [0x63, 0x67, 0x6F, 0x7B, 0x73, 0x63, 0x63, 0x00], // N
    [0x1C, 0x36, 0x63, 0x63, 0x63, 0x36, 0x1C, 0x00], // O
    [0x3F, 0x66, 0x66, 0x3E, 0x06, 0x06, 0x0F, 0x00], // P
    [0x1E, 0x33, 0x33, 0x33, 0x3B, 0x1E, 0x38, 0x00], // Q
    [0x3F, 0x66, 0x66, 0x3E, 0x36, 0x66, 0x67, 0x00], // R
    [0x1E, 0x33, 0x07, 0x0E, 0x38, 0x33, 0x1E, 0x00], // S
    [0x3F, 0x2D, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00], // T
    [0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x3F, 0x00], // U
    [0x33, 0x33, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00], // V
    [0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00], // W
    [0x63, 0x63, 0x36, 0x1C, 0x1C, 0x36, 0x63, 0x00], // X
    [0x33, 0x33, 0x33, 0x1E, 0x0C, 0x0C, 0x1E, 0x00], // Y
    [0x7F, 0x63, 0x31, 0x18, 0x4C, 0x66, 0x7F, 0x00], // Z
    [0x1E, 0x06, 0x06, 0x06, 0x06, 0x06, 0x1E, 0x00], // [
    [0x03, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x40, 0x00], // \
    [0x1E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x1E, 0x00], // ]
    [0x08, 0x1C, 0x36, 0x63, 0x00, 0x00, 0x00, 0x00], // ^
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF], // _
    [0x0C, 0x0C, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00], // `
    [0x00, 0x00, 0x1E, 0x30, 0x3E, 0x33, 0x6E, 0x00], // a
    [0x07, 0x06, 0x06, 0x3E, 0x66, 0x66, 0x3B, 0x00], // b
    [0x00, 0x00, 0x1E, 0x33, 0x03, 0x33, 0x1E, 0x00], // c
    [0x38, 0x30, 0x30, 0x3e, 0x33, 0x33, 0x6E, 0x00], // d
    [0x00, 0x00, 0x1E, 0x33, 0x3f, 0x03, 0x1E, 0x00], // e
    [0x1C, 0x36, 0x06, 0x0f, 0x06, 0x06, 0x0F, 0x00], // f
    [0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x1F], // g
    [0x07, 0x06, 0x36, 0x6E, 0x66, 0x66, 0x67, 0x00], // h
    [0x0C, 0x00, 0x0E, 0x0C, 0x0C, 0x0C, 0x1E, 0x00], // i
    [0x30, 0x00, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E], // j
    [0x07, 0x06, 0x66, 0x36, 0x1E, 0x36, 0x67, 0x00], // k
    [0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00], // l
    [0x00, 0x00, 0x33, 0x7F, 0x7F, 0x6B, 0x63, 0x00], // m
    [0x00, 0x00, 0x1F, 0x33, 0x33, 0x33, 0x33, 0x00], // n
    [0x00, 0x00, 0x1E, 0x33, 0x33, 0x33, 0x1E, 0x00], // o
    [0x00, 0x00, 0x3B, 0x66, 0x66, 0x3E, 0x06, 0x0F], // p
    [0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x78], // q
    [0x00, 0x00, 0x3B, 0x6E, 0x66, 0x06, 0x0F, 0x00], // r
    [0x00, 0x00, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x00], // s
    [0x08, 0x0C, 0x3E, 0x0C, 0x0C, 0x2C, 0x18, 0x00], // t
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x33, 0x6E, 0x00], // u
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00], // v
    [0x00, 0x00, 0x63, 0x6B, 0x7F, 0x7F, 0x36, 0x00], // w
    [0x00, 0x00, 0x63, 0x36, 0x1C, 0x36, 0x63, 0x00], // x
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x3E, 0x30, 0x1F], // y
    [0x00, 0x00, 0x3F, 0x19, 0x0C, 0x26, 0x3F, 0x00], // z
    [0x38, 0x0C, 0x0C, 0x07, 0x0C, 0x0C, 0x38, 0x00], // {
    [0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18, 0x00], // |
    [0x07, 0x0C, 0x0C, 0x38, 0x0C, 0x0C, 0x07, 0x00], // }
    [0x6E, 0x3B, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // ~
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dialog_raster_is_not_fully_transparent() {
        let fb = rasterize_system_ui(
            Some(&HostDialogRequest {
                kind: HostDialogKind::Message,
                title: "Message".into(),
                message: "Hello from REmu".into(),
                buttons: HostDialogButtons::OkCancel,
                sequence: 1,
            }),
            None,
        );
        assert!(!fb.is_transparent());
        assert_eq!(
            fb.rgba.len(),
            (SYSTEM_UI_WIDTH * SYSTEM_UI_HEIGHT * 4) as usize
        );
    }

    #[test]
    fn toast_draws_near_the_top() {
        let fb = rasterize_system_ui(
            None,
            Some(&HostToastRequest {
                kind: HostDialogKind::Trophy,
                title: "Trophy".into(),
                message: "Offline".into(),
                duration_ms: 1000,
            }),
        );
        // Sample a pixel inside the toast band.
        let index = ((40 * SYSTEM_UI_WIDTH + SYSTEM_UI_WIDTH / 2) * 4) as usize;
        assert!(fb.rgba[index + 3] > 0);
    }

    #[test]
    fn all_dialog_kinds_rasterize() {
        for kind in [
            HostDialogKind::Message,
            HostDialogKind::SaveData,
            HostDialogKind::Error,
            HostDialogKind::Progress,
            HostDialogKind::Ime,
            HostDialogKind::System,
        ] {
            let fb = rasterize_system_ui(
                Some(&HostDialogRequest {
                    kind,
                    title: String::new(),
                    message: "Sample body text for system UI.".into(),
                    buttons: HostDialogButtons::OkCancel,
                    sequence: 1,
                }),
                None,
            );
            assert!(!fb.is_transparent(), "{kind:?}");
        }
    }
}
