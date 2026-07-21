// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

/// Decodes a GFX10 inline-constant operand into its exact 32-bit register representation.
#[must_use]
pub fn decode_inline_constant(encoded: u32) -> Option<u32> {
    match encoded {
        125 => Some(0),
        128..=192 => Some(encoded - 128),
        193..=208 => Some((192 - i32::try_from(encoded).ok()?).cast_unsigned()),
        240 => Some(0.5_f32.to_bits()),
        241 => Some((-0.5_f32).to_bits()),
        242 => Some(1.0_f32.to_bits()),
        243 => Some((-1.0_f32).to_bits()),
        244 => Some(2.0_f32.to_bits()),
        245 => Some((-2.0_f32).to_bits()),
        246 => Some(4.0_f32.to_bits()),
        247 => Some((-4.0_f32).to_bits()),
        248 => Some((1.0_f32 / (2.0_f32 * std::f32::consts::PI)).to_bits()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::decode_inline_constant;

    #[test]
    fn decodes_integer_and_floating_point_inline_constants() {
        assert_eq!(decode_inline_constant(125), Some(0));
        assert_eq!(decode_inline_constant(128), Some(0));
        assert_eq!(decode_inline_constant(192), Some(64));
        assert_eq!(decode_inline_constant(193), Some((-1_i32).cast_unsigned()));
        assert_eq!(decode_inline_constant(208), Some((-16_i32).cast_unsigned()));
        assert_eq!(decode_inline_constant(242), Some(1.0_f32.to_bits()));
        assert_eq!(
            decode_inline_constant(248),
            Some((1.0_f32 / std::f32::consts::TAU).to_bits())
        );
        assert_eq!(decode_inline_constant(249), None);
    }
}
