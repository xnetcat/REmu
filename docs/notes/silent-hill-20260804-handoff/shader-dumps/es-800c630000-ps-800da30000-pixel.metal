#include <metal_stdlib>
using namespace metal;

inline bool remu_range_valid(uint offset, uint size, uint length) {
    return offset <= length && size <= length - offset;
}

inline uint remu_load_u8(const device uchar* data, uint offset, uint length) {
    return remu_range_valid(offset, 1u, length) ? (uint)data[offset] : 0u;
}

inline uint remu_load_u16(const device uchar* data, uint offset, uint length) {
    if (!remu_range_valid(offset, 2u, length)) return 0u;
    return (uint)data[offset] | ((uint)data[offset + 1u] << 8u);
}

inline uint remu_load_u32(const device uchar* data, uint offset, uint length) {
    if (!remu_range_valid(offset, 4u, length)) return 0u;
    if ((offset & 3u) == 0u) {
        return *((const device uint*)(data + offset));
    }
    uint value = 0u;
    for (uint i = 0u; i < 4u; ++i) {
        value |= (uint)data[offset + i] << (i * 8u);
    }
    return value;
}

inline bool remu_f32_bits_is_nan(uint bits) {
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

struct RemuSamplers {
    sampler s0 [[id(0)]];
};

struct RemuPsIn {
    float4 frag_coord [[position]];
    float4 attr0 [[user(locn0), center_no_perspective]];
};

struct RemuPsOut {
};

fragment void remu_main(
    RemuPsIn remu_in [[stage_in]],
    const device uchar* b0 [[buffer(0)]],
    constant uint* remu_initial_s [[buffer(1)]],
    texture2d<float> tex0 [[texture(0)]],
    constant RemuSamplers& remu_samplers [[buffer(2)]]
) {
    uint remu_s_3 = remu_initial_s[3];
    uint remu_s_16 = remu_initial_s[16];
    uint remu_s_17 = remu_initial_s[17];
    uint remu_s_18 = remu_initial_s[18];
    uint remu_s_19 = remu_initial_s[19];
    uint remu_s_20 = remu_initial_s[20];
    uint remu_s_21 = remu_initial_s[21];
    uint remu_s_22 = remu_initial_s[22];
    uint remu_s_23 = remu_initial_s[23];
    uint remu_s_124 = remu_initial_s[124];
    uint remu_v_0 = 0u;
    uint remu_v_1 = 0u;
    uint remu_v_2 = 0u;
    uint remu_v_3 = 0u;
    uint remu_v_4 = 0u;
    uint remu_v_5 = 0u;
    uint remu_v_6 = 0u;
    uint remu_v_7 = 0u;
    uint remu_v_8 = 0u;
    uint remu_v_9 = 0u;
    uint remu_v_10 = 0u;
    uint remu_v_11 = 0u;
    uint remu_v_12 = 0u;
    uint remu_v_13 = 0u;
    uint remu_v_14 = 0u;
    uint remu_v_15 = 0u;
    uint remu_v_16 = 0u;
    uint remu_v_17 = 0u;
    uint remu_v_18 = 0u;
    uint remu_v_19 = 0u;
    uint remu_v_20 = 0u;
    uint remu_v_21 = 0u;
    uint remu_v_22 = 0u;
    bool exec = true;
    bool vcc = false;
    bool scc = false;
    uint remu_lane = simd_prefix_exclusive_sum(1u);
            remu_s_124 = remu_s_16;
            uint t0 = (0u + 0u);
            remu_s_16 = remu_load_u32(b0, t0 + 0u, remu_initial_s[130]);
            remu_s_17 = remu_load_u32(b0, t0 + 4u, remu_initial_s[130]);
            remu_s_18 = remu_load_u32(b0, t0 + 8u, remu_initial_s[130]);
            remu_s_19 = remu_load_u32(b0, t0 + 12u, remu_initial_s[130]);
            remu_s_20 = remu_load_u32(b0, t0 + 16u, remu_initial_s[130]);
            remu_s_21 = remu_load_u32(b0, t0 + 20u, remu_initial_s[130]);
            remu_s_22 = remu_load_u32(b0, t0 + 24u, remu_initial_s[130]);
            remu_s_23 = remu_load_u32(b0, t0 + 28u, remu_initial_s[130]);
            if (exec) remu_v_2 = as_type<uint>(remu_in.attr0[0]);
            if (exec) remu_v_3 = as_type<uint>((fma(as_type<float>(remu_s_16), as_type<float>(remu_v_2), as_type<float>(0xBF000000u))));
            if (exec) remu_v_0 = as_type<uint>(remu_in.attr0[1]);
            if (exec) remu_v_1 = as_type<uint>((fma(as_type<float>(remu_v_0), as_type<float>(remu_s_17), as_type<float>(0xBF000000u))));
            if (exec) remu_v_4 = as_type<uint>((floor(as_type<float>(remu_v_3))));
            if (exec) remu_v_8 = as_type<uint>((floor(as_type<float>(remu_v_1))));
            if (exec) remu_v_3 = as_type<uint>((as_type<float>(0x3F000000u) + as_type<float>(remu_v_4)));
            if (exec) remu_v_9 = as_type<uint>((as_type<float>(0x40200000u) + as_type<float>(remu_v_4)));
            if (exec) remu_v_6 = as_type<uint>((as_type<float>(0xBF000000u) + as_type<float>(remu_v_4)));
            if (exec) remu_v_1 = as_type<uint>((as_type<float>(0x3F000000u) + as_type<float>(remu_v_8)));
            if (exec) remu_v_12 = as_type<uint>((as_type<float>(0x40200000u) + as_type<float>(remu_v_8)));
            if (exec) remu_v_7 = as_type<uint>((fma(as_type<float>(remu_s_16), as_type<float>(remu_v_2), -(as_type<float>(remu_v_3)))));
            if (exec) remu_v_5 = as_type<uint>((as_type<float>(0xBF000000u) + as_type<float>(remu_v_8)));
            if (exec) remu_v_2 = as_type<uint>((as_type<float>(remu_s_18) * as_type<float>(remu_v_6)));
            if (exec) remu_v_4 = as_type<uint>((fma(as_type<float>(remu_v_0), as_type<float>(remu_s_17), -(as_type<float>(remu_v_1)))));
            if (exec) remu_v_12 = as_type<uint>((as_type<float>(remu_s_19) * as_type<float>(remu_v_12)));
            if (exec) remu_v_11 = as_type<uint>((as_type<float>(remu_v_7) * as_type<float>(remu_v_7)));
            if (exec) remu_v_5 = as_type<uint>((as_type<float>(remu_s_19) * as_type<float>(remu_v_5)));
            if (exec) remu_v_6 = as_type<uint>((as_type<float>(remu_s_18) * as_type<float>(remu_v_9)));
            if (exec) remu_v_0 = as_type<uint>((as_type<float>(remu_v_4) * as_type<float>(remu_v_4)));
            if (exec) remu_v_2 = as_type<uint>((fmin(as_type<float>(remu_s_22), as_type<float>(remu_v_2))));
            if (exec) remu_v_14 = as_type<uint>((as_type<float>(remu_v_7) * as_type<float>(remu_v_11)));
            if (exec) remu_v_10 = as_type<uint>((fma(as_type<float>(remu_v_7), as_type<float>(remu_v_11), as_type<float>(remu_v_7))));
            if (exec) remu_v_9 = as_type<uint>((fma(as_type<float>(remu_v_7), as_type<float>(remu_v_11), -(as_type<float>(remu_v_11)))));
            if (exec) remu_v_13 = as_type<uint>((as_type<float>(remu_v_4) * as_type<float>(remu_v_0)));
            if (exec) remu_v_8 = as_type<uint>((fma(as_type<float>(remu_v_4), as_type<float>(remu_v_0), as_type<float>(remu_v_4))));
            if (exec) remu_v_14 = as_type<uint>((fma(as_type<float>(0x3F000000u), as_type<float>(remu_v_14), as_type<float>(remu_v_14))));
            if (exec) remu_v_7 = as_type<uint>((fma(as_type<float>(0xBF000000u), as_type<float>(remu_v_10), as_type<float>(remu_v_11))));
            if (exec) remu_v_10 = as_type<uint>((fma(as_type<float>(remu_v_4), as_type<float>(remu_v_0), -(as_type<float>(remu_v_0)))));
            if (exec) remu_v_13 = as_type<uint>((fma(as_type<float>(0x3F000000u), as_type<float>(remu_v_13), as_type<float>(remu_v_13))));
            if (exec) remu_v_8 = as_type<uint>((fma(as_type<float>(0xBF000000u), as_type<float>(remu_v_8), as_type<float>(remu_v_0))));
            if (exec) remu_v_11 = as_type<uint>((fma(as_type<float>(remu_v_11), as_type<float>(0xC0200000u), as_type<float>(remu_v_14))));
            if (exec) remu_v_14 = as_type<uint>((fmin(as_type<float>(remu_s_23), as_type<float>(remu_v_5))));
            if (exec) remu_v_6 = as_type<uint>((fmin(as_type<float>(remu_s_22), as_type<float>(remu_v_6))));
            if (exec) remu_v_13 = as_type<uint>((fma(as_type<float>(remu_v_0), as_type<float>(0xC0200000u), as_type<float>(remu_v_13))));
            if (exec) remu_v_4 = as_type<uint>((as_type<float>(0x3F000000u) * as_type<float>(remu_v_9)));
            if (exec) remu_v_11 = as_type<uint>((as_type<float>(0x3F800000u) + as_type<float>(remu_v_11)));
            if (exec) remu_v_2 = as_type<uint>((fmax(as_type<float>(remu_s_20), as_type<float>(remu_v_2))));
            if (exec) remu_v_5 = as_type<uint>((as_type<float>(0x3F000000u) * as_type<float>(remu_v_10)));
            if (exec) remu_v_0 = as_type<uint>((fmax(as_type<float>(remu_s_20), as_type<float>(remu_v_6))));
            if (exec) remu_v_6 = as_type<uint>((fmax(as_type<float>(remu_s_21), as_type<float>(remu_v_14))));
            if (exec) remu_v_15 = as_type<uint>((fmin(as_type<float>(remu_s_23), as_type<float>(remu_v_12))));
            if (exec) remu_v_12 = as_type<uint>((as_type<float>(0x3F800000u) + as_type<float>(remu_v_13)));
            uint t1 = as_type<uint>((-(as_type<float>(remu_v_7)) + -(as_type<float>(remu_v_11))));
            if (exec) remu_v_13 = t1;
            if (exec) remu_v_15 = as_type<uint>((fmax(as_type<float>(remu_s_21), as_type<float>(remu_v_15))));
            uint t2 = as_type<uint>((-(as_type<float>(remu_v_8)) + -(as_type<float>(remu_v_12))));
            if (exec) remu_v_16 = t2;
            if (exec) remu_v_13 = as_type<uint>((fma(as_type<float>(0xBF000000u), as_type<float>(remu_v_9), as_type<float>(remu_v_13))));
            if (exec) remu_v_16 = as_type<uint>((fma(as_type<float>(0xBF000000u), as_type<float>(remu_v_10), as_type<float>(remu_v_16))));
            if (exec) remu_v_14 = as_type<uint>((as_type<float>(0x3F800000u) + as_type<float>(remu_v_13)));
            if (exec) remu_v_13 = as_type<uint>((as_type<float>(0x3F800000u) + as_type<float>(remu_v_16)));
            if (exec) remu_v_9 = as_type<uint>((as_type<float>(remu_v_11) + as_type<float>(remu_v_14)));
            if (exec) remu_v_10 = as_type<uint>((as_type<float>(remu_v_12) + as_type<float>(remu_v_13)));
            if (exec) remu_v_11 = as_type<uint>((1.0f / as_type<float>(remu_v_9)));
            if (exec) remu_v_20 = as_type<uint>((as_type<float>(remu_v_8) * as_type<float>(remu_v_9)));
            if (exec) remu_v_17 = as_type<uint>((as_type<float>(remu_v_5) * as_type<float>(remu_v_9)));
            if (exec) remu_v_8 = as_type<uint>((1.0f / as_type<float>(remu_v_10)));
            if (exec) remu_v_3 = as_type<uint>((fma(as_type<float>(remu_v_14), as_type<float>(remu_v_11), as_type<float>(remu_v_3))));
            if (exec) remu_v_18 = as_type<uint>((as_type<float>(remu_v_7) * as_type<float>(remu_v_10)));
            if (exec) remu_v_7 = as_type<uint>((fma(as_type<float>(remu_v_7), as_type<float>(remu_v_10), as_type<float>(remu_v_20))));
            if (exec) remu_v_16 = as_type<uint>((as_type<float>(remu_v_9) * as_type<float>(remu_v_10)));
            if (exec) remu_v_1 = as_type<uint>((fma(as_type<float>(remu_v_13), as_type<float>(remu_v_8), as_type<float>(remu_v_1))));
            if (exec) remu_v_3 = as_type<uint>((as_type<float>(remu_s_18) * as_type<float>(remu_v_3)));
            if (exec) remu_v_19 = as_type<uint>((as_type<float>(remu_v_4) * as_type<float>(remu_v_10)));
            if (exec) remu_v_7 = as_type<uint>((fma(as_type<float>(remu_v_10), as_type<float>(remu_v_9), as_type<float>(remu_v_7))));
            if (exec) remu_v_1 = as_type<uint>((as_type<float>(remu_s_19) * as_type<float>(remu_v_1)));
            if (exec) remu_v_3 = as_type<uint>((fmin(as_type<float>(remu_s_22), as_type<float>(remu_v_3))));
            if (exec) remu_v_7 = as_type<uint>((fma(as_type<float>(remu_v_10), as_type<float>(remu_v_4), as_type<float>(remu_v_7))));
            if (exec) remu_v_1 = as_type<uint>((fmin(as_type<float>(remu_s_23), as_type<float>(remu_v_1))));
            if (exec) remu_v_22 = as_type<uint>((fmax(as_type<float>(remu_s_20), as_type<float>(remu_v_3))));
            if (exec) remu_v_7 = as_type<uint>((fma(as_type<float>(remu_v_9), as_type<float>(remu_v_5), as_type<float>(remu_v_7))));
            if (exec) remu_v_1 = as_type<uint>((fmax(as_type<float>(remu_s_21), as_type<float>(remu_v_1))));
            if (exec) remu_v_21 = as_type<uint>((1.0f / as_type<float>(remu_v_7)));
            float2 t3 = float2(as_type<float>(remu_v_22), as_type<float>(remu_v_6));
            vec<float, 4> t4 = tex0.sample(remu_samplers.s0, t3, level(0.0f));
            uint t5 = (remu_s_3 & 0xFFFu) == 0u ? 0xFACu : (remu_s_3 & 0xFFFu);
            vec<float, 4> t6 = vec<float, 4>((((t5 >> 0u) & 7u) == 1u ? 1.0f : (((t5 >> 0u) & 7u) >= 4u && ((t5 >> 0u) & 7u) <= 7u ? t4[((t5 >> 0u) & 7u) - 4u] : 0.0f)), (((t5 >> 3u) & 7u) == 1u ? 1.0f : (((t5 >> 3u) & 7u) >= 4u && ((t5 >> 3u) & 7u) <= 7u ? t4[((t5 >> 3u) & 7u) - 4u] : 0.0f)), (((t5 >> 6u) & 7u) == 1u ? 1.0f : (((t5 >> 6u) & 7u) >= 4u && ((t5 >> 6u) & 7u) <= 7u ? t4[((t5 >> 6u) & 7u) - 4u] : 0.0f)), (((t5 >> 9u) & 7u) == 1u ? 1.0f : (((t5 >> 9u) & 7u) >= 4u && ((t5 >> 9u) & 7u) <= 7u ? t4[((t5 >> 9u) & 7u) - 4u] : 0.0f)));
            if (exec) remu_v_12 = as_type<uint>(t6[0]);
            if (exec) remu_v_13 = as_type<uint>(t6[1]);
            if (exec) remu_v_14 = as_type<uint>(t6[2]);
            float2 t7 = float2(as_type<float>(remu_v_2), as_type<float>(remu_v_1));
            vec<float, 4> t8 = tex0.sample(remu_samplers.s0, t7, level(0.0f));
            uint t9 = (remu_s_3 & 0xFFFu) == 0u ? 0xFACu : (remu_s_3 & 0xFFFu);
            vec<float, 4> t10 = vec<float, 4>((((t9 >> 0u) & 7u) == 1u ? 1.0f : (((t9 >> 0u) & 7u) >= 4u && ((t9 >> 0u) & 7u) <= 7u ? t8[((t9 >> 0u) & 7u) - 4u] : 0.0f)), (((t9 >> 3u) & 7u) == 1u ? 1.0f : (((t9 >> 3u) & 7u) >= 4u && ((t9 >> 3u) & 7u) <= 7u ? t8[((t9 >> 3u) & 7u) - 4u] : 0.0f)), (((t9 >> 6u) & 7u) == 1u ? 1.0f : (((t9 >> 6u) & 7u) >= 4u && ((t9 >> 6u) & 7u) <= 7u ? t8[((t9 >> 6u) & 7u) - 4u] : 0.0f)), (((t9 >> 9u) & 7u) == 1u ? 1.0f : (((t9 >> 9u) & 7u) >= 4u && ((t9 >> 9u) & 7u) <= 7u ? t8[((t9 >> 9u) & 7u) - 4u] : 0.0f)));
            if (exec) remu_v_9 = as_type<uint>(t10[0]);
            if (exec) remu_v_10 = as_type<uint>(t10[1]);
            if (exec) remu_v_11 = as_type<uint>(t10[2]);
            float2 t11 = float2(as_type<float>(remu_v_0), as_type<float>(remu_v_1));
            vec<float, 4> t12 = tex0.sample(remu_samplers.s0, t11, level(0.0f));
            uint t13 = (remu_s_3 & 0xFFFu) == 0u ? 0xFACu : (remu_s_3 & 0xFFFu);
            vec<float, 4> t14 = vec<float, 4>((((t13 >> 0u) & 7u) == 1u ? 1.0f : (((t13 >> 0u) & 7u) >= 4u && ((t13 >> 0u) & 7u) <= 7u ? t12[((t13 >> 0u) & 7u) - 4u] : 0.0f)), (((t13 >> 3u) & 7u) == 1u ? 1.0f : (((t13 >> 3u) & 7u) >= 4u && ((t13 >> 3u) & 7u) <= 7u ? t12[((t13 >> 3u) & 7u) - 4u] : 0.0f)), (((t13 >> 6u) & 7u) == 1u ? 1.0f : (((t13 >> 6u) & 7u) >= 4u && ((t13 >> 6u) & 7u) <= 7u ? t12[((t13 >> 6u) & 7u) - 4u] : 0.0f)), (((t13 >> 9u) & 7u) == 1u ? 1.0f : (((t13 >> 9u) & 7u) >= 4u && ((t13 >> 9u) & 7u) <= 7u ? t12[((t13 >> 9u) & 7u) - 4u] : 0.0f)));
            if (exec) remu_v_3 = as_type<uint>(t14[0]);
            if (exec) remu_v_4 = as_type<uint>(t14[1]);
            if (exec) remu_v_5 = as_type<uint>(t14[2]);
            float2 t15 = float2(as_type<float>(remu_v_22), as_type<float>(remu_v_1));
            vec<float, 4> t16 = tex0.sample(remu_samplers.s0, t15, level(0.0f));
            uint t17 = (remu_s_3 & 0xFFFu) == 0u ? 0xFACu : (remu_s_3 & 0xFFFu);
            vec<float, 4> t18 = vec<float, 4>((((t17 >> 0u) & 7u) == 1u ? 1.0f : (((t17 >> 0u) & 7u) >= 4u && ((t17 >> 0u) & 7u) <= 7u ? t16[((t17 >> 0u) & 7u) - 4u] : 0.0f)), (((t17 >> 3u) & 7u) == 1u ? 1.0f : (((t17 >> 3u) & 7u) >= 4u && ((t17 >> 3u) & 7u) <= 7u ? t16[((t17 >> 3u) & 7u) - 4u] : 0.0f)), (((t17 >> 6u) & 7u) == 1u ? 1.0f : (((t17 >> 6u) & 7u) >= 4u && ((t17 >> 6u) & 7u) <= 7u ? t16[((t17 >> 6u) & 7u) - 4u] : 0.0f)), (((t17 >> 9u) & 7u) == 1u ? 1.0f : (((t17 >> 9u) & 7u) >= 4u && ((t17 >> 9u) & 7u) <= 7u ? t16[((t17 >> 9u) & 7u) - 4u] : 0.0f)));
            if (exec) remu_v_6 = as_type<uint>(t18[0]);
            if (exec) remu_v_7 = as_type<uint>(t18[1]);
            if (exec) remu_v_8 = as_type<uint>(t18[2]);
            float2 t19 = float2(as_type<float>(remu_v_22), as_type<float>(remu_v_15));
            vec<float, 4> t20 = tex0.sample(remu_samplers.s0, t19, level(0.0f));
            uint t21 = (remu_s_3 & 0xFFFu) == 0u ? 0xFACu : (remu_s_3 & 0xFFFu);
            vec<float, 4> t22 = vec<float, 4>((((t21 >> 0u) & 7u) == 1u ? 1.0f : (((t21 >> 0u) & 7u) >= 4u && ((t21 >> 0u) & 7u) <= 7u ? t20[((t21 >> 0u) & 7u) - 4u] : 0.0f)), (((t21 >> 3u) & 7u) == 1u ? 1.0f : (((t21 >> 3u) & 7u) >= 4u && ((t21 >> 3u) & 7u) <= 7u ? t20[((t21 >> 3u) & 7u) - 4u] : 0.0f)), (((t21 >> 6u) & 7u) == 1u ? 1.0f : (((t21 >> 6u) & 7u) >= 4u && ((t21 >> 6u) & 7u) <= 7u ? t20[((t21 >> 6u) & 7u) - 4u] : 0.0f)), (((t21 >> 9u) & 7u) == 1u ? 1.0f : (((t21 >> 9u) & 7u) >= 4u && ((t21 >> 9u) & 7u) <= 7u ? t20[((t21 >> 9u) & 7u) - 4u] : 0.0f)));
            if (exec) remu_v_0 = as_type<uint>(t22[0]);
            if (exec) remu_v_1 = as_type<uint>(t22[1]);
            if (exec) remu_v_2 = as_type<uint>(t22[2]);
            if (exec) remu_v_12 = as_type<uint>((as_type<float>(remu_v_12) * as_type<float>(remu_v_20)));
            if (exec) remu_v_13 = as_type<uint>((as_type<float>(remu_v_13) * as_type<float>(remu_v_20)));
            if (exec) remu_v_14 = as_type<uint>((as_type<float>(remu_v_14) * as_type<float>(remu_v_20)));
            if (exec) remu_v_12 = as_type<uint>((fma(as_type<float>(remu_v_18), as_type<float>(remu_v_9), as_type<float>(remu_v_12))));
            if (exec) remu_v_13 = as_type<uint>((fma(as_type<float>(remu_v_18), as_type<float>(remu_v_10), as_type<float>(remu_v_13))));
            if (exec) remu_v_14 = as_type<uint>((fma(as_type<float>(remu_v_18), as_type<float>(remu_v_11), as_type<float>(remu_v_14))));
            if (exec) remu_v_12 = as_type<uint>((fma(as_type<float>(remu_v_16), as_type<float>(remu_v_6), as_type<float>(remu_v_12))));
            if (exec) remu_v_13 = as_type<uint>((fma(as_type<float>(remu_v_16), as_type<float>(remu_v_7), as_type<float>(remu_v_13))));
            if (exec) remu_v_14 = as_type<uint>((fma(as_type<float>(remu_v_16), as_type<float>(remu_v_8), as_type<float>(remu_v_14))));
            if (exec) remu_v_12 = as_type<uint>((fma(as_type<float>(remu_v_19), as_type<float>(remu_v_3), as_type<float>(remu_v_12))));
            if (exec) remu_v_13 = as_type<uint>((fma(as_type<float>(remu_v_19), as_type<float>(remu_v_4), as_type<float>(remu_v_13))));
            if (exec) remu_v_14 = as_type<uint>((fma(as_type<float>(remu_v_19), as_type<float>(remu_v_5), as_type<float>(remu_v_14))));
            if (exec) remu_v_12 = as_type<uint>((fma(as_type<float>(remu_v_17), as_type<float>(remu_v_0), as_type<float>(remu_v_12))));
            if (exec) remu_v_13 = as_type<uint>((fma(as_type<float>(remu_v_17), as_type<float>(remu_v_1), as_type<float>(remu_v_13))));
            if (exec) remu_v_14 = as_type<uint>((fma(as_type<float>(remu_v_17), as_type<float>(remu_v_2), as_type<float>(remu_v_14))));
            if (exec) remu_v_0 = as_type<uint>((as_type<float>(remu_v_21) * as_type<float>(remu_v_12)));
            if (exec) remu_v_1 = as_type<uint>((as_type<float>(remu_v_21) * as_type<float>(remu_v_13)));
            if (exec) remu_v_2 = as_type<uint>((as_type<float>(remu_v_21) * as_type<float>(remu_v_14)));
            uint t23 = as_type<uint>(as_type<float>(remu_v_0));
            uint t24 = as_type<uint>(as_type<float>(remu_v_1));
            if (exec) remu_v_0 = (remu_f32_to_f16_rtz_bits(t23) | (remu_f32_to_f16_rtz_bits(t24) << 16));
            uint t25 = as_type<uint>(as_type<float>(remu_v_2));
            uint t26 = as_type<uint>(as_type<float>(0u));
            if (exec) remu_v_1 = (remu_f32_to_f16_rtz_bits(t25) | (remu_f32_to_f16_rtz_bits(t26) << 16));
    if (!exec) discard_fragment();
}
