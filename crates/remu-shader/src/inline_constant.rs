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

/// Decodes a GFX10 packed-f16 inline constant into the 32-bit value presented to a
/// `V_PK_*_F16` source operand.
///
/// Packed 16-bit instructions do not see the ordinary f32 bit patterns for floating-point inline
/// constants. The hardware supplies the corresponding f16 bits in the low half and zero in the
/// high half. Integer inline constants remain sign-extended 32-bit values. Operand selection is
/// applied to this value afterwards.
#[must_use]
pub fn decode_v2f16_inline_constant(encoded: u32) -> Option<u32> {
    match encoded {
        125 => Some(0),
        128..=192 => Some(encoded - 128),
        193..=208 => Some((192 - i32::try_from(encoded).ok()?).cast_unsigned()),
        240 => Some(0x3800),
        241 => Some(0xb800),
        242 => Some(0x3c00),
        243 => Some(0xbc00),
        244 => Some(0x4000),
        245 => Some(0xc000),
        246 => Some(0x4400),
        247 => Some(0xc400),
        248 => Some(0x3118),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{decode_inline_constant, decode_v2f16_inline_constant};

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

    #[test]
    fn decodes_packed_f16_inline_constants_in_their_hardware_representation() {
        for encoded in 128..=192 {
            assert_eq!(decode_v2f16_inline_constant(encoded), Some(encoded - 128));
        }
        for encoded in 193..=208 {
            assert_eq!(
                decode_v2f16_inline_constant(encoded),
                Some((192 - i32::try_from(encoded).expect("small encoding")).cast_unsigned())
            );
        }
        assert_eq!(decode_v2f16_inline_constant(240), Some(0x3800));
        assert_eq!(decode_v2f16_inline_constant(242), Some(0x3c00));
        assert_eq!(decode_v2f16_inline_constant(247), Some(0xc400));
        assert_eq!(decode_v2f16_inline_constant(248), Some(0x3118));
        assert_eq!(decode_v2f16_inline_constant(249), None);
    }
}
