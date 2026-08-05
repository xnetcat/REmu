//! Metal source preludes emitted ahead of the translated shader body.

use std::fmt::Write as _;

use remu_shader::decode_unified_format;

pub(crate) fn emit_data_share_prelude(source: &mut String) {
    source.push_str(
        r"inline uint remu_ds_inc(threadgroup atomic_uint* location, uint ceiling) {
    uint observed = atomic_load_explicit(location, memory_order_relaxed);
    while (true) {
        uint desired = observed >= ceiling ? 0u : observed + 1u;
        uint expected = observed;
        if (atomic_compare_exchange_weak_explicit(location, &expected, desired,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) return observed;
        observed = expected;
    }
}

inline uint remu_ds_dec(threadgroup atomic_uint* location, uint ceiling) {
    uint observed = atomic_load_explicit(location, memory_order_relaxed);
    while (true) {
        uint desired = (observed == 0u || observed > ceiling) ? ceiling : observed - 1u;
        uint expected = observed;
        if (atomic_compare_exchange_weak_explicit(location, &expected, desired,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) return observed;
        observed = expected;
    }
}

inline uint remu_ds_smin(threadgroup atomic_uint* location, uint value) {
    uint observed = atomic_load_explicit(location, memory_order_relaxed);
    while (true) {
        uint desired = as_type<int>(value) < as_type<int>(observed) ? value : observed;
        uint expected = observed;
        if (atomic_compare_exchange_weak_explicit(location, &expected, desired,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) return observed;
        observed = expected;
    }
}

inline uint remu_ds_smax(threadgroup atomic_uint* location, uint value) {
    uint observed = atomic_load_explicit(location, memory_order_relaxed);
    while (true) {
        uint desired = as_type<int>(value) > as_type<int>(observed) ? value : observed;
        uint expected = observed;
        if (atomic_compare_exchange_weak_explicit(location, &expected, desired,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) return observed;
        observed = expected;
    }
}

inline void remu_ds_insert(threadgroup atomic_uint* location, uint offset, uint bits,
                           uint value) {
    // DS_WRITE_B8/B16 replace one field of a dword. Two lanes may target different fields of the
    // same dword, so a plain load/modify/store would drop the other lane's bytes.
    uint observed = atomic_load_explicit(location, memory_order_relaxed);
    while (true) {
        uint desired = insert_bits(observed, value, offset, bits);
        uint expected = observed;
        if (atomic_compare_exchange_weak_explicit(location, &expected, desired,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) return;
        observed = expected;
    }
}

inline uint remu_ds_cmpst(threadgroup atomic_uint* location, uint comparator,
                         uint replacement) {
    uint observed = atomic_load_explicit(location, memory_order_relaxed);
    while (observed == comparator) {
        uint expected = observed;
        if (atomic_compare_exchange_weak_explicit(location, &expected, replacement,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) return observed;
        observed = expected;
    }
    return observed;
}

",
    );
}

#[allow(clippy::too_many_lines)]
pub(crate) fn emit_global_memory_prelude(source: &mut String) {
    source.push_str(
        r"inline void remu_store_u8(device uchar* data, uint offset, uint value) {
    data[offset] = (uchar)value;
}

inline void remu_store_u16(device uchar* data, uint offset, uint value) {
    data[offset] = (uchar)value;
    data[offset + 1u] = (uchar)(value >> 8u);
}

inline void remu_store_u32(device uchar* data, uint offset, uint value) {
    for (uint i = 0u; i < 4u; ++i) data[offset + i] = (uchar)(value >> (i * 8u));
}

inline device atomic_uint* remu_atomic_pointer(device uchar* data, uint offset) {
    return (device atomic_uint*)(data + offset);
}

inline uint remu_atomic_swap(device uchar* data, uint offset, uint value) {
    return atomic_exchange_explicit(remu_atomic_pointer(data, offset), value,
                                    memory_order_relaxed);
}

inline uint remu_atomic_add(device uchar* data, uint offset, uint value) {
    return atomic_fetch_add_explicit(remu_atomic_pointer(data, offset), value,
                                     memory_order_relaxed);
}

inline uint remu_atomic_sub(device uchar* data, uint offset, uint value) {
    return atomic_fetch_sub_explicit(remu_atomic_pointer(data, offset), value,
                                     memory_order_relaxed);
}

inline uint remu_atomic_and(device uchar* data, uint offset, uint value) {
    return atomic_fetch_and_explicit(remu_atomic_pointer(data, offset), value,
                                     memory_order_relaxed);
}

inline uint remu_atomic_or(device uchar* data, uint offset, uint value) {
    return atomic_fetch_or_explicit(remu_atomic_pointer(data, offset), value,
                                    memory_order_relaxed);
}

inline uint remu_atomic_xor(device uchar* data, uint offset, uint value) {
    return atomic_fetch_xor_explicit(remu_atomic_pointer(data, offset), value,
                                     memory_order_relaxed);
}

inline uint remu_atomic_cmpswap(device uchar* data, uint offset, uint replacement,
                                uint comparator) {
    device atomic_uint* location = remu_atomic_pointer(data, offset);
    uint observed = atomic_load_explicit(location, memory_order_relaxed);
    while (observed == comparator) {
        uint expected = observed;
        if (atomic_compare_exchange_weak_explicit(location, &expected, replacement,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) return observed;
        observed = expected;
    }
    return observed;
}

inline uint remu_atomic_umin(device uchar* data, uint offset, uint value) {
    device atomic_uint* location = remu_atomic_pointer(data, offset);
    uint observed = atomic_load_explicit(location, memory_order_relaxed);
    while (true) {
        uint expected = observed;
        uint desired = min(observed, value);
        if (atomic_compare_exchange_weak_explicit(location, &expected, desired,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) return observed;
        observed = expected;
    }
}

inline uint remu_atomic_umax(device uchar* data, uint offset, uint value) {
    device atomic_uint* location = remu_atomic_pointer(data, offset);
    uint observed = atomic_load_explicit(location, memory_order_relaxed);
    while (true) {
        uint expected = observed;
        uint desired = max(observed, value);
        if (atomic_compare_exchange_weak_explicit(location, &expected, desired,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) return observed;
        observed = expected;
    }
}

inline uint remu_atomic_smin(device uchar* data, uint offset, uint value) {
    device atomic_uint* location = remu_atomic_pointer(data, offset);
    uint observed = atomic_load_explicit(location, memory_order_relaxed);
    while (true) {
        uint expected = observed;
        uint desired = as_type<int>(value) < as_type<int>(observed) ? value : observed;
        if (atomic_compare_exchange_weak_explicit(location, &expected, desired,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) return observed;
        observed = expected;
    }
}

inline uint remu_atomic_smax(device uchar* data, uint offset, uint value) {
    device atomic_uint* location = remu_atomic_pointer(data, offset);
    uint observed = atomic_load_explicit(location, memory_order_relaxed);
    while (true) {
        uint expected = observed;
        uint desired = as_type<int>(value) > as_type<int>(observed) ? value : observed;
        if (atomic_compare_exchange_weak_explicit(location, &expected, desired,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) return observed;
        observed = expected;
    }
}

inline uint remu_atomic_inc(device uchar* data, uint offset, uint limit) {
    device atomic_uint* location = remu_atomic_pointer(data, offset);
    uint observed = atomic_load_explicit(location, memory_order_relaxed);
    while (true) {
        uint expected = observed;
        uint desired = observed >= limit ? 0u : observed + 1u;
        if (atomic_compare_exchange_weak_explicit(location, &expected, desired,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) return observed;
        observed = expected;
    }
}

inline uint remu_atomic_dec(device uchar* data, uint offset, uint limit) {
    device atomic_uint* location = remu_atomic_pointer(data, offset);
    uint observed = atomic_load_explicit(location, memory_order_relaxed);
    while (true) {
        uint expected = observed;
        uint desired = observed == 0u || observed > limit ? limit : observed - 1u;
        if (atomic_compare_exchange_weak_explicit(location, &expected, desired,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) return observed;
        observed = expected;
    }
}

",
    );
}

pub(crate) fn emit_vop3p_prelude(source: &mut String) {
    source.push_str(
        r"inline bool remu_f32_bits_is_nan(uint bits) {
    return (bits & 0x7FFFFFFFu) > 0x7F800000u;
}

inline uint remu_f16_to_f32_bits(uint half_bits) {
    half_bits &= 0xFFFFu;
    uint sign = (half_bits & 0x8000u) << 16;
    uint exponent = (half_bits >> 10) & 0x1Fu;
    uint mantissa = half_bits & 0x3FFu;
    uint normal = ((exponent + 112u) << 23) | (mantissa << 13);
    uint infinity_nan = 0x7F800000u | (mantissa << 13);
    uint high_bit = 31u - clz(mantissa | 1u);
    uint shift = 23u - high_bit;
    uint sub_fraction = (mantissa << shift) & 0x7FFFFFu;
    uint subnormal = mantissa != 0u ? ((high_bit + 103u) << 23) | sub_fraction : 0u;
    uint magnitude = exponent == 0u ? subnormal : (exponent == 31u ? infinity_nan : normal);
    return sign | magnitude;
}

inline uint remu_f32_to_f16_bits(uint bits) {
    uint sign = (bits >> 16) & 0x8000u;
    uint absolute = bits & 0x7FFFFFFFu;
    bool infinity_nan_input = absolute >= 0x7F800000u;
    bool nan_input = absolute > 0x7F800000u;
    uint infinity_nan = sign | 0x7C00u | (nan_input ? 0x200u : 0u);

    uint exponent = absolute >> 23;
    uint mantissa = absolute & 0x7FFFFFu;
    uint significand = mantissa | 0x800000u;

    uint round_bit = (significand >> 13) & 1u;
    uint rounded = (significand + 0xFFFu + round_bit) >> 13;
    uint half_exponent = exponent - 112u;
    uint normal_bits = (half_exponent << 10) + (rounded - 0x400u);
    uint normal = exponent >= 113u ? (normal_bits >= 0x7C00u ? 0x7C00u : normal_bits) : 0u;

    uint distance = 126u - exponent;
    uint subnormal_shift = min(max(distance, 1u), 25u);
    uint shift_mask = (1u << subnormal_shift) - 1u;
    uint halfway = 1u << (subnormal_shift - 1u);
    uint low_bits = significand & shift_mask;
    uint quotient = significand >> subnormal_shift;
    bool round_up = low_bits > halfway || (low_bits == halfway && (quotient & 1u) != 0u);
    uint subnormal = quotient + (round_up ? 1u : 0u);

    uint finite = exponent <= 112u ? subnormal : normal;
    return infinity_nan_input ? infinity_nan : (sign | finite);
}

inline uint remu_f32_to_f16_rtz_bits(uint bits) {
    uint sign = (bits >> 16) & 0x8000u;
    uint absolute = bits & 0x7FFFFFFFu;
    uint exponent = (absolute >> 23) & 0xFFu;
    uint mantissa = absolute & 0x7FFFFFu;

    if (exponent == 0xFFu) {
        if (mantissa == 0u) return sign | 0x7C00u;
        uint payload = (mantissa >> 13) | 0x200u;
        return sign | 0x7C00u | (payload & 0x3FFu);
    }

    // Round-toward-zero overflow produces the greatest finite value. A true infinity is
    // handled above and remains an infinity, matching RDNA2's FP16 overflow behavior.
    if (exponent > 142u) return sign | 0x7BFFu;

    if (exponent >= 113u) {
        return sign | ((exponent - 112u) << 10) | (mantissa >> 13);
    }

    if (exponent >= 103u) {
        uint significand = mantissa | 0x800000u;
        return sign | (significand >> (126u - exponent));
    }

    return sign;
}

inline uint remu_clamp_unit_f32_bits(uint bits) {
    uint magnitude = bits & 0x7FFFFFFFu;
    if (remu_f32_bits_is_nan(bits) || (bits & 0x80000000u) != 0u || magnitude == 0u) return 0u;
    return magnitude > 0x3F800000u ? 0x3F800000u : bits;
}

inline uint remu_f16_minmax_bits(float left, float right, bool maximum) {
    uint left_bits = as_type<uint>(left);
    uint right_bits = as_type<uint>(right);
    if (remu_f32_bits_is_nan(left_bits)) return right_bits;
    if (remu_f32_bits_is_nan(right_bits)) return left_bits;
    return as_type<uint>(maximum ? (left > right ? left : right) : (left < right ? left : right));
}

inline uint remu_f16_fma_rto_bits(float left, float right, float addend) {
#pragma clang fp reassociate(off) contract(off)
    float product = left * right;
    float sum = product + addend;
    float product_part = sum - addend;
    float addend_part = sum - product_part;
    float product_error = product - product_part;
    float addend_error = addend - addend_part;
    float residual = product_error + addend_error;

    uint sum_bits = as_type<uint>(sum);
    uint residual_bits = as_type<uint>(residual);
    uint residual_magnitude = residual_bits & 0x7FFFFFFFu;
    bool inexact = residual_magnitude != 0u && residual_magnitude < 0x7F800000u;
    bool adjust = inexact && (sum_bits & 1u) == 0u;
    bool toward_zero = ((sum_bits ^ residual_bits) & 0x80000000u) != 0u;
    uint stepped = toward_zero ? sum_bits - 1u : sum_bits + 1u;
    return adjust ? stepped : sum_bits;
}

",
    );
}

pub(crate) fn emit_div_fixup_f32_prelude(source: &mut String) {
    source.push_str(
        r"inline bool remu_div_fixup_f32_is_nan(uint bits) {
    return (bits & 0x7FFFFFFFu) > 0x7F800000u;
}

inline uint remu_div_fixup_f32_quiet_nan(uint bits) {
    return bits | 0x00400000u;
}

inline uint remu_div_fixup_f32_bits(uint quotient_bits, uint denominator_bits, uint numerator_bits) {
    uint denominator_magnitude = denominator_bits & 0x7FFFFFFFu;
    uint numerator_magnitude = numerator_bits & 0x7FFFFFFFu;
    uint output_sign = (denominator_bits ^ numerator_bits) & 0x80000000u;

    if (remu_div_fixup_f32_is_nan(numerator_bits)) {
        return remu_div_fixup_f32_quiet_nan(numerator_bits);
    }
    if (remu_div_fixup_f32_is_nan(denominator_bits)) {
        return remu_div_fixup_f32_quiet_nan(denominator_bits);
    }
    if (denominator_magnitude == 0u && numerator_magnitude == 0u) {
        return 0xFFC00000u;
    }
    if (denominator_magnitude == 0x7F800000u && numerator_magnitude == 0x7F800000u) {
        return 0xFFC00000u;
    }
    if (denominator_magnitude == 0u || numerator_magnitude == 0x7F800000u) {
        return output_sign | 0x7F800000u;
    }
    if (denominator_magnitude == 0x7F800000u || numerator_magnitude == 0u) {
        return output_sign;
    }

    int denominator_exponent = int((denominator_bits >> 23) & 0xFFu);
    int numerator_exponent = int((numerator_bits >> 23) & 0xFFu);
    if (numerator_exponent - denominator_exponent < -150 || denominator_exponent == 255) {
        float corrected = as_type<float>(numerator_bits) / as_type<float>(denominator_bits);
        return as_type<uint>(corrected);
    }
    return output_sign | (quotient_bits & 0x7FFFFFFFu);
}

",
    );
}

#[allow(clippy::too_many_lines)]
pub(crate) fn emit_format_prelude(source: &mut String) {
    source.push_str("static constant uint remu_gfx10_formats[128] = {\n    ");
    for unified in 0..128 {
        let packed =
            decode_unified_format(unified).map_or(0, |(data, number)| data | (number << 8));
        let _ = write!(source, "0x{packed:X}u,");
        if unified % 16 == 15 && unified != 127 {
            source.push_str("\n    ");
        } else {
            source.push(' ');
        }
    }
    source.push_str(
        "\n};\n\n\
         static inline void remu_format_layout(uint dfmt, uint component, thread uint& byte_off, thread uint& bit_off, thread uint& bits) {\n\
         \x20   byte_off = 0u; bit_off = 0u; bits = 0u;\n\
         \x20   switch (component * 16u + dfmt) {\n",
    );
    for &(component, format, bytes, bit_offset, bits) in FORMAT_COMPONENT_LAYOUTS {
        let key = component * 16 + format;
        let _ = writeln!(
            source,
            "    case {key}u: byte_off = {bytes}u; bit_off = {bit_offset}u; bits = {bits}u; break;"
        );
    }
    source.push_str(
        "    default: break;\n    }\n}\n\n\
         static inline uint remu_minifloat(uint raw, uint bits) {\n\
         \x20   uint mantissa_bits = bits - 5u;\n\
         \x20   uint mantissa = raw & ((1u << mantissa_bits) - 1u);\n\
         \x20   uint exponent = (raw >> mantissa_bits) & 0x1Fu;\n\
         \x20   uint shift = 23u - mantissa_bits;\n\
         \x20   if (exponent == 31u) return 0x7F800000u | (mantissa << shift);\n\
         \x20   if (exponent == 0u) {\n\
         \x20       float scale = mantissa_bits == 6u ? (1.0f / 1048576.0f) : (1.0f / 524288.0f);\n\
         \x20       return as_type<uint>((float)mantissa * scale);\n\
         \x20   }\n\
         \x20   return ((exponent + 112u) << 23) | (mantissa << shift);\n}\n\n\
         static inline uint remu_format_one(uint nfmt) {\n\
         \x20   return (nfmt == 4u || nfmt == 5u) ? 1u : 0x3F800000u;\n}\n\n\
         static inline float remu_srgb_decode(float encoded) {\n\
         \x20   encoded = clamp(encoded, 0.0f, 1.0f);\n\
         \x20   return encoded <= 0.04045f ? encoded / 12.92f : pow((encoded + 0.055f) / 1.055f, 2.4f);\n}\n\n\
         static inline float remu_srgb_encode(float linear) {\n\
         \x20   linear = clamp(linear, 0.0f, 1.0f);\n\
         \x20   return linear <= 0.0031308f ? linear * 12.92f : 1.055f * pow(linear, 1.0f / 2.4f) - 0.055f;\n}\n\n\
         static inline uint remu_format_convert(uint raw, uint bits, uint nfmt, uint dfmt, uint component) {\n\
         \x20   uint low_mask = bits >= 32u ? 0xFFFFFFFFu : ((1u << bits) - 1u);\n\
         \x20   int signed_raw = extract_bits(as_type<int>(raw), 0u, bits);\n\
         \x20   switch (nfmt) {\n\
         \x20   case 0u: return as_type<uint>((float)raw / (float)low_mask);\n\
         \x20   case 1u: return as_type<uint>(fmax((float)signed_raw / (float)(low_mask >> 1), -1.0f));\n\
         \x20   case 2u: return as_type<uint>((float)raw);\n\
         \x20   case 3u: return as_type<uint>((float)signed_raw);\n\
         \x20   case 5u: return (uint)signed_raw;\n\
         \x20   case 7u:\n\
         \x20       if (dfmt == 6u || dfmt == 7u) return remu_minifloat(raw, bits);\n\
         \x20       if (bits == 16u) return as_type<uint>((float)as_type<half>((ushort)(raw & 0xFFFFu)));\n\
         \x20       return raw;\n\
         \x20   case 9u: {\n\
         \x20       float normalized = (float)raw / (float)low_mask;\n\
         \x20       return as_type<uint>(component == 3u ? normalized : remu_srgb_decode(normalized));\n\
         \x20   }\n\
         \x20   default: return raw;\n\
         \x20   }\n}\n\n",
    );
    source.push_str("static inline uint remu_format_size(uint dfmt) {\n    switch (dfmt) {\n");
    for format in 1..=14_u32 {
        let size = FORMAT_COMPONENT_LAYOUTS
            .iter()
            .filter(|(_, candidate, _, _, _)| *candidate == format)
            .map(|(_, _, byte_offset, bit_offset, bits)| {
                byte_offset + (bit_offset + bits).div_ceil(8)
            })
            .max()
            .unwrap_or(0);
        let _ = writeln!(source, "    case {format}u: return {size}u;");
    }
    source.push_str(
        r"    default: return 0u;
    }
}

static inline uint remu_load_format_bits(const device uchar* data, uint address,
                                         uint byte_off, uint bit_off, uint bits) {
    uint value = 0u;
    for (uint bit = 0u; bit < bits; ++bit) {
        uint source_bit = bit_off + bit;
        uint source_byte = address + byte_off + source_bit / 8u;
        value |= (((uint)data[source_byte] >> (source_bit & 7u)) & 1u) << bit;
    }
    return value;
}

static inline void remu_store_format_bits(device uchar* data, uint address, uint byte_off,
                                          uint bit_off, uint bits, uint value) {
    for (uint bit = 0u; bit < bits; ++bit) {
        uint destination_bit = bit_off + bit;
        uint destination_byte = address + byte_off + destination_bit / 8u;
        uint mask = 1u << (destination_bit & 7u);
        uint byte_value = (uint)data[destination_byte];
        byte_value = (byte_value & ~mask) | (((value >> bit) & 1u) * mask);
        data[destination_byte] = (uchar)byte_value;
    }
}

static inline uint remu_format_minifloat_pack(uint value, uint bits) {
    float converted = fmax(as_type<float>(value), 0.0f);
    uint half_bits = (uint)as_type<ushort>(half(converted));
    uint exponent = (half_bits >> 10) & 0x1Fu;
    uint mantissa = half_bits & 0x3FFu;
    uint mantissa_bits = bits - 5u;
    uint shift = 10u - mantissa_bits;
    uint truncated = mantissa >> shift;
    uint remainder = mantissa & ((1u << shift) - 1u);
    uint halfway = 1u << (shift - 1u);
    if (remainder > halfway || (remainder == halfway && (truncated & 1u) != 0u)) {
        truncated += 1u;
        if (truncated == (1u << mantissa_bits)) {
            truncated = 0u;
            exponent = min(exponent + 1u, 31u);
        }
    }
    return (exponent << mantissa_bits) | truncated;
}

static inline uint remu_format_pack(uint value, uint bits, uint nfmt, uint dfmt, uint component) {
    uint mask = bits >= 32u ? 0xFFFFFFFFu : ((1u << bits) - 1u);
    float floating = as_type<float>(value);
    switch (nfmt) {
    case 0u:
        return (uint)rint(clamp(floating, 0.0f, 1.0f) * (float)mask) & mask;
    case 1u: {
        int maximum = (int)(mask >> 1);
        int packed = (int)rint(clamp(floating, -1.0f, 1.0f) * (float)maximum);
        return (uint)packed & mask;
    }
    case 2u:
        return (uint)rint(clamp(floating, 0.0f, (float)mask)) & mask;
    case 3u: {
        int maximum = (int)(mask >> 1);
        int minimum = -maximum - 1;
        return (uint)(int)rint(clamp(floating, (float)minimum, (float)maximum)) & mask;
    }
    case 4u: return value & mask;
    case 5u: return (uint)as_type<int>(value) & mask;
    case 7u:
        if (dfmt == 6u || dfmt == 7u) return remu_format_minifloat_pack(value, bits);
        if (bits == 16u) return (uint)as_type<ushort>(half(floating));
        return value;
    case 9u: {
        float encoded = component == 3u ? clamp(floating, 0.0f, 1.0f)
                                        : remu_srgb_encode(floating);
        return (uint)rint(encoded * (float)mask) & mask;
    }
    default: return value & mask;
    }
}

static inline uint remu_format_from_d16(uint value, uint nfmt) {
    value &= 0xFFFFu;
    if (nfmt == 4u) return value;
    if (nfmt == 5u) return (value & 0x8000u) != 0u ? value | 0xFFFF0000u : value;
    return as_type<uint>((float)as_type<half>((ushort)value));
}

static inline uint remu_format_to_d16(uint value, uint nfmt) {
    if (nfmt == 4u || nfmt == 5u) return value & 0xFFFFu;
    return (uint)as_type<ushort>(half(as_type<float>(value)));
}

",
    );
}

const FORMAT_COMPONENT_LAYOUTS: &[(u32, u32, u32, u32, u32)] = &[
    (0, 1, 0, 0, 8),
    (0, 2, 0, 0, 16),
    (0, 3, 0, 0, 8),
    (0, 4, 0, 0, 32),
    (0, 5, 0, 0, 16),
    (0, 6, 0, 0, 10),
    (0, 7, 0, 0, 11),
    (0, 8, 0, 0, 10),
    (0, 9, 0, 0, 2),
    (0, 10, 0, 0, 8),
    (0, 11, 0, 0, 32),
    (0, 12, 0, 0, 16),
    (0, 13, 0, 0, 32),
    (0, 14, 0, 0, 32),
    (1, 3, 1, 0, 8),
    (1, 5, 2, 0, 16),
    (1, 6, 0, 10, 11),
    (1, 7, 0, 11, 11),
    (1, 8, 0, 10, 10),
    (1, 9, 0, 2, 10),
    (1, 10, 1, 0, 8),
    (1, 11, 4, 0, 32),
    (1, 12, 2, 0, 16),
    (1, 13, 4, 0, 32),
    (1, 14, 4, 0, 32),
    (2, 6, 0, 21, 11),
    (2, 7, 0, 22, 10),
    (2, 8, 0, 20, 10),
    (2, 9, 0, 12, 10),
    (2, 10, 2, 0, 8),
    (2, 12, 4, 0, 16),
    (2, 13, 8, 0, 32),
    (2, 14, 8, 0, 32),
    (3, 8, 0, 30, 2),
    (3, 9, 0, 22, 10),
    (3, 10, 3, 0, 8),
    (3, 12, 6, 0, 16),
    (3, 14, 12, 0, 32),
];
