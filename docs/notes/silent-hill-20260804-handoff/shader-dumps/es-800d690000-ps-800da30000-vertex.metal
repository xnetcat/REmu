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

static constant uint remu_gfx10_formats[128] = {
    0x0u, 0x1u, 0x101u, 0x201u, 0x301u, 0x401u, 0x501u, 0x2u, 0x102u, 0x202u, 0x302u, 0x402u, 0x502u, 0x702u, 0x3u, 0x103u,
    0x203u, 0x303u, 0x403u, 0x503u, 0x404u, 0x504u, 0x704u, 0x5u, 0x105u, 0x205u, 0x305u, 0x405u, 0x505u, 0x705u, 0x0u, 0x0u,
    0x0u, 0x0u, 0x0u, 0x0u, 0x706u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x707u, 0x8u, 0x108u, 0x0u, 0x0u,
    0x408u, 0x508u, 0x9u, 0x109u, 0x209u, 0x309u, 0x409u, 0x509u, 0xAu, 0x10Au, 0x20Au, 0x30Au, 0x40Au, 0x50Au, 0x40Bu, 0x50Bu,
    0x70Bu, 0xCu, 0x10Cu, 0x20Cu, 0x30Cu, 0x40Cu, 0x50Cu, 0x70Cu, 0x40Du, 0x50Du, 0x70Du, 0x40Eu, 0x50Eu, 0x70Eu, 0x0u, 0x0u,
    0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u,
    0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u,
    0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 0x0u, 
};

static inline void remu_format_layout(uint dfmt, uint component, thread uint& byte_off, thread uint& bit_off, thread uint& bits) {
    byte_off = 0u; bit_off = 0u; bits = 0u;
    switch (component * 16u + dfmt) {
    case 1u: byte_off = 0u; bit_off = 0u; bits = 8u; break;
    case 2u: byte_off = 0u; bit_off = 0u; bits = 16u; break;
    case 3u: byte_off = 0u; bit_off = 0u; bits = 8u; break;
    case 4u: byte_off = 0u; bit_off = 0u; bits = 32u; break;
    case 5u: byte_off = 0u; bit_off = 0u; bits = 16u; break;
    case 6u: byte_off = 0u; bit_off = 0u; bits = 10u; break;
    case 7u: byte_off = 0u; bit_off = 0u; bits = 11u; break;
    case 8u: byte_off = 0u; bit_off = 0u; bits = 10u; break;
    case 9u: byte_off = 0u; bit_off = 0u; bits = 2u; break;
    case 10u: byte_off = 0u; bit_off = 0u; bits = 8u; break;
    case 11u: byte_off = 0u; bit_off = 0u; bits = 32u; break;
    case 12u: byte_off = 0u; bit_off = 0u; bits = 16u; break;
    case 13u: byte_off = 0u; bit_off = 0u; bits = 32u; break;
    case 14u: byte_off = 0u; bit_off = 0u; bits = 32u; break;
    case 19u: byte_off = 1u; bit_off = 0u; bits = 8u; break;
    case 21u: byte_off = 2u; bit_off = 0u; bits = 16u; break;
    case 22u: byte_off = 0u; bit_off = 10u; bits = 11u; break;
    case 23u: byte_off = 0u; bit_off = 11u; bits = 11u; break;
    case 24u: byte_off = 0u; bit_off = 10u; bits = 10u; break;
    case 25u: byte_off = 0u; bit_off = 2u; bits = 10u; break;
    case 26u: byte_off = 1u; bit_off = 0u; bits = 8u; break;
    case 27u: byte_off = 4u; bit_off = 0u; bits = 32u; break;
    case 28u: byte_off = 2u; bit_off = 0u; bits = 16u; break;
    case 29u: byte_off = 4u; bit_off = 0u; bits = 32u; break;
    case 30u: byte_off = 4u; bit_off = 0u; bits = 32u; break;
    case 38u: byte_off = 0u; bit_off = 21u; bits = 11u; break;
    case 39u: byte_off = 0u; bit_off = 22u; bits = 10u; break;
    case 40u: byte_off = 0u; bit_off = 20u; bits = 10u; break;
    case 41u: byte_off = 0u; bit_off = 12u; bits = 10u; break;
    case 42u: byte_off = 2u; bit_off = 0u; bits = 8u; break;
    case 44u: byte_off = 4u; bit_off = 0u; bits = 16u; break;
    case 45u: byte_off = 8u; bit_off = 0u; bits = 32u; break;
    case 46u: byte_off = 8u; bit_off = 0u; bits = 32u; break;
    case 56u: byte_off = 0u; bit_off = 30u; bits = 2u; break;
    case 57u: byte_off = 0u; bit_off = 22u; bits = 10u; break;
    case 58u: byte_off = 3u; bit_off = 0u; bits = 8u; break;
    case 60u: byte_off = 6u; bit_off = 0u; bits = 16u; break;
    case 62u: byte_off = 12u; bit_off = 0u; bits = 32u; break;
    default: break;
    }
}

static inline uint remu_minifloat(uint raw, uint bits) {
    uint mantissa_bits = bits - 5u;
    uint mantissa = raw & ((1u << mantissa_bits) - 1u);
    uint exponent = (raw >> mantissa_bits) & 0x1Fu;
    uint shift = 23u - mantissa_bits;
    if (exponent == 31u) return 0x7F800000u | (mantissa << shift);
    if (exponent == 0u) {
        float scale = mantissa_bits == 6u ? (1.0f / 1048576.0f) : (1.0f / 524288.0f);
        return as_type<uint>((float)mantissa * scale);
    }
    return ((exponent + 112u) << 23) | (mantissa << shift);
}

static inline uint remu_format_one(uint nfmt) {
    return (nfmt == 4u || nfmt == 5u) ? 1u : 0x3F800000u;
}

static inline float remu_srgb_decode(float encoded) {
    encoded = clamp(encoded, 0.0f, 1.0f);
    return encoded <= 0.04045f ? encoded / 12.92f : pow((encoded + 0.055f) / 1.055f, 2.4f);
}

static inline float remu_srgb_encode(float linear) {
    linear = clamp(linear, 0.0f, 1.0f);
    return linear <= 0.0031308f ? linear * 12.92f : 1.055f * pow(linear, 1.0f / 2.4f) - 0.055f;
}

static inline uint remu_format_convert(uint raw, uint bits, uint nfmt, uint dfmt, uint component) {
    uint low_mask = bits >= 32u ? 0xFFFFFFFFu : ((1u << bits) - 1u);
    int signed_raw = extract_bits(as_type<int>(raw), 0u, bits);
    switch (nfmt) {
    case 0u: return as_type<uint>((float)raw / (float)low_mask);
    case 1u: return as_type<uint>(fmax((float)signed_raw / (float)(low_mask >> 1), -1.0f));
    case 2u: return as_type<uint>((float)raw);
    case 3u: return as_type<uint>((float)signed_raw);
    case 5u: return (uint)signed_raw;
    case 7u:
        if (dfmt == 6u || dfmt == 7u) return remu_minifloat(raw, bits);
        if (bits == 16u) return as_type<uint>((float)as_type<half>((ushort)(raw & 0xFFFFu)));
        return raw;
    case 9u: {
        float normalized = (float)raw / (float)low_mask;
        return as_type<uint>(component == 3u ? normalized : remu_srgb_decode(normalized));
    }
    default: return raw;
    }
}

static inline uint remu_format_size(uint dfmt) {
    switch (dfmt) {
    case 1u: return 1u;
    case 2u: return 2u;
    case 3u: return 2u;
    case 4u: return 4u;
    case 5u: return 4u;
    case 6u: return 4u;
    case 7u: return 4u;
    case 8u: return 4u;
    case 9u: return 4u;
    case 10u: return 4u;
    case 11u: return 8u;
    case 12u: return 8u;
    case 13u: return 12u;
    case 14u: return 16u;
    default: return 0u;
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

struct RemuVsOut {
    float4 position [[position]];
    float4 param0 [[user(locn0)]];
};

vertex RemuVsOut remu_main(
    const device uchar* b0 [[buffer(0)]],
    const device uchar* b1 [[buffer(1)]],
    const device uchar* b2 [[buffer(2)]],
    const device uchar* b3 [[buffer(3)]],
    constant uint* remu_initial_s [[buffer(4)]],
    uint remu_vertex_id [[vertex_id]],
    uint remu_instance_id [[instance_id]]
) {
    uint remu_s_0 = remu_initial_s[0];
    uint remu_s_1 = remu_initial_s[1];
    uint remu_s_2 = remu_initial_s[2];
    uint remu_s_3 = remu_initial_s[3];
    uint remu_s_4 = remu_initial_s[4];
    uint remu_s_5 = remu_initial_s[5];
    uint remu_s_6 = remu_initial_s[6];
    uint remu_s_7 = remu_initial_s[7];
    uint remu_s_8 = remu_initial_s[8];
    uint remu_s_9 = remu_initial_s[9];
    uint remu_s_10 = remu_initial_s[10];
    uint remu_s_11 = remu_initial_s[11];
    uint remu_s_12 = remu_initial_s[12];
    uint remu_s_13 = remu_initial_s[13];
    uint remu_s_14 = remu_initial_s[14];
    uint remu_s_15 = remu_initial_s[15];
    uint remu_s_16 = remu_initial_s[16];
    uint remu_s_17 = remu_initial_s[17];
    uint remu_s_18 = remu_initial_s[18];
    uint remu_s_19 = remu_initial_s[19];
    uint remu_s_20 = remu_initial_s[20];
    uint remu_s_21 = remu_initial_s[21];
    uint remu_s_22 = remu_initial_s[22];
    uint remu_s_23 = remu_initial_s[23];
    uint remu_s_24 = remu_initial_s[24];
    uint remu_s_25 = remu_initial_s[25];
    uint remu_s_26 = remu_initial_s[26];
    uint remu_s_27 = remu_initial_s[27];
    uint remu_s_106 = remu_initial_s[106];
    uint remu_s_107 = remu_initial_s[107];
    uint remu_s_124 = remu_initial_s[124];
    uint remu_s_126 = remu_initial_s[126];
    uint remu_s_127 = remu_initial_s[127];
    uint remu_v_0 = 0u;
    uint remu_v_1 = 0u;
    uint remu_v_2 = 0u;
    uint remu_v_3 = 0u;
    uint remu_v_4 = 0u;
    uint remu_v_5 = 0u;
    uint remu_v_8 = 0u;
    bool exec = true;
    bool vcc = false;
    bool scc = false;
    uint remu_lane = simd_prefix_exclusive_sum(1u);
    RemuVsOut remu_out = {};
    remu_v_5 = remu_vertex_id;
    remu_v_8 = remu_instance_id;
    remu_s_106 = 0u;
    remu_s_107 = 0u;
    remu_s_126 = (uint)(uint64_t)simd_ballot(true);
    remu_s_127 = 0u;
            uint t0 = 0x80008u & 31u;
            uint t1 = min((0x80008u >> 16) & 0x7Fu, 32u - t0);
            uint t2 = t1 == 0u ? 0u : extract_bits(remu_s_3, t0, t1);
            uint t3 = t2;
            remu_s_107 = t3;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t2 != 0u;
            uint t4 = (remu_s_3 & 0xFFu);
            remu_s_0 = t4;
            scc = t4 != 0u;
            uint t5 = (remu_s_107 << (0xCu & 31u));
            uint t6 = t5;
            remu_s_106 = t6;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t5 != 0u;
            uint t7 = (remu_s_0 | remu_s_106);
            remu_s_124 = t7;
            scc = t7 != 0u;
            uint t8 = (0x40u - remu_s_107);
            uint t9 = t8;
            remu_s_106 = t9;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = (((0x40u ^ remu_s_107) & (0x40u ^ t8)) >> 31) != 0u;
            ulong t10 = (0xFFFFFFFFFFFFFFFFul >> (remu_s_106 & 63u));
            remu_s_126 = (uint)t10;
            remu_s_127 = (uint)(t10 >> 32);
            exec = (((remu_s_126) >> remu_lane) & 1u) != 0u;
            scc = t10 != 0ul;
            uint t11 = (0x40u - remu_s_0);
            uint t12 = t11;
            remu_s_106 = t12;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = (((0x40u ^ remu_s_0) & (0x40u ^ t11)) >> 31) != 0u;
            ulong t13 = (0xFFFFFFFFFFFFFFFFul >> (remu_s_106 & 63u));
            remu_s_126 = (uint)t13;
            remu_s_127 = (uint)(t13 >> 32);
            exec = (((remu_s_126) >> remu_lane) & 1u) != 0u;
            scc = t13 != 0ul;
            uint t14 = (0u + 0u);
            remu_s_20 = remu_load_u32(b0, t14 + 0u, remu_initial_s[130]);
            remu_s_21 = remu_load_u32(b0, t14 + 4u, remu_initial_s[130]);
            remu_s_14 = 0x22C0204u;
            remu_s_15 = 0xFAC03ACu;
            uint t15 = (0u + 0u);
            remu_s_0 = remu_load_u32(b1, t15 + 0u, remu_initial_s[134]);
            remu_s_1 = remu_load_u32(b1, t15 + 4u, remu_initial_s[134]);
            remu_s_2 = remu_load_u32(b1, t15 + 8u, remu_initial_s[134]);
            remu_s_3 = remu_load_u32(b1, t15 + 12u, remu_initial_s[134]);
            remu_s_4 = remu_load_u32(b1, t15 + 16u, remu_initial_s[134]);
            remu_s_5 = remu_load_u32(b1, t15 + 20u, remu_initial_s[134]);
            remu_s_6 = remu_load_u32(b1, t15 + 24u, remu_initial_s[134]);
            remu_s_7 = remu_load_u32(b1, t15 + 28u, remu_initial_s[134]);
            uint t16 = (0u + 0x20u);
            remu_s_16 = remu_load_u32(b1, t16 + 0u, remu_initial_s[134]);
            remu_s_17 = remu_load_u32(b1, t16 + 4u, remu_initial_s[134]);
            remu_s_18 = remu_load_u32(b1, t16 + 8u, remu_initial_s[134]);
            remu_s_19 = remu_load_u32(b1, t16 + 12u, remu_initial_s[134]);
            uint t17 = (remu_s_20 << (4u & 31u));
            uint t18 = t17;
            remu_s_107 = t18;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t17 != 0u;
            uint t19 = 0x1001Au & 31u;
            uint t20 = min((0x1001Au >> 16) & 0x7Fu, 32u - t19);
            uint t21 = t20 == 0u ? 0u : extract_bits(remu_s_20, t19, t20);
            uint t22 = t21;
            remu_s_106 = t22;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t21 != 0u;
            uint t23 = (remu_s_107 & 0x1F0u);
            uint t24 = t23;
            remu_s_107 = t24;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t23 != 0u;
            scc = as_type<int>(0u) == as_type<int>(remu_s_106);
            uint t25 = (remu_s_107 + 0u);
            remu_s_8 = remu_load_u32(b2, t25 + 0u, remu_initial_s[138]);
            remu_s_9 = remu_load_u32(b2, t25 + 4u, remu_initial_s[138]);
            remu_s_10 = remu_load_u32(b2, t25 + 8u, remu_initial_s[138]);
            remu_s_11 = remu_load_u32(b2, t25 + 12u, remu_initial_s[138]);
            ulong t26 = (scc ? ((ulong)remu_s_126 | ((ulong)remu_s_127 << 32)) : 0x0000000000000000ul);
            remu_s_22 = (uint)t26;
            remu_s_23 = (uint)(t26 >> 32);
            uint t27 = 0x70007u & 31u;
            uint t28 = min((0x70007u >> 16) & 0x7Fu, 32u - t27);
            uint t29 = t28 == 0u ? 0u : extract_bits(remu_s_20, t27, t28);
            remu_s_26 = t29;
            scc = t29 != 0u;
            uint t30 = 0x20005u & 31u;
            uint t31 = min((0x20005u >> 16) & 0x7Fu, 32u - t30);
            uint t32 = t31 == 0u ? 0u : extract_bits(remu_s_20, t30, t31);
            uint t33 = t32;
            remu_s_106 = t33;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t32 != 0u;
            uint t34 = (remu_s_26 << (0xCu & 31u));
            remu_s_27 = t34;
            scc = t34 != 0u;
            uint t35 = ((remu_s_106 << 4) + 0xC0000u);
            uint t36 = t35;
            remu_s_107 = t36;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            uint t37 = (remu_s_11 & 0xFFF80000u);
            uint t38 = t37;
            remu_s_106 = t38;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t37 != 0u;
            uint t39 = remu_s_107 & 63u;
            uint t40 = min((remu_s_107 >> 16) & 0x7Fu, 64u - t39);
            ulong t41 = t40 >= 64u ? 0xFFFFFFFFFFFFFFFFul : ((1ul << t40) - 1ul);
            ulong t42 = (((ulong)remu_s_14 | ((ulong)remu_s_15 << 32)) >> t39) & t41;
            remu_s_24 = (uint)t42;
            remu_s_25 = (uint)(t42 >> 32);
            scc = t42 != 0ul;
            uint t43 = (remu_s_106 | remu_s_27);
            uint t44 = t43;
            remu_s_106 = t44;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t43 != 0u;
            if (exec) remu_v_0 = ((((remu_s_22) >> remu_lane) & 1u) != 0u) ? (remu_v_5) : (remu_v_8);
            uint t45 = (remu_s_106 | remu_s_24);
            uint t46 = t45;
            remu_s_106 = t46;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t45 != 0u;
            uint t47 = 0xC000Eu & 31u;
            uint t48 = min((0xC000Eu >> 16) & 0x7Fu, 32u - t47);
            uint t49 = t48 == 0u ? 0u : extract_bits(remu_s_20, t47, t48);
            uint t50 = t49;
            remu_s_107 = t50;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t49 != 0u;
            scc = (remu_s_26) == (0u);
            uint t51 = (scc ? remu_s_11 : remu_s_106);
            remu_s_11 = t51;
            uint t52 = (remu_s_21 << (4u & 31u));
            uint t53 = t52;
            remu_s_106 = t53;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t52 != 0u;
            ulong t54 = (ulong)remu_s_8 | (((ulong)remu_s_9 & 0xFFFFul) << 32);
            uint t55 = (uint)(t54 & 3ul);
            uint t56 = (remu_s_9 >> 16) & 0x3FFFu;
            bool t57 = (remu_s_9 & 0x80000000u) != 0u;
            bool t58 = (remu_s_11 & 0x00800000u) != 0u;
            uint t59 = remu_v_0 + (t58 ? (remu_lane & 63u) : 0u);
            uint t60 = 0u + 0u;
            uint t61 = 8u << ((remu_s_11 >> 21) & 3u);
            uint t62 = t59 / t61;
            uint t63 = t59 % t61;
            uint t64 = t60 / 4u;
            uint t65 = t60 % 4u;
            uint t66 = t59 * t56 + t60;
            uint t67 = (t62 * t56 + t64 * 4u) * t61 + t63 * 4u + t65;
            uint t68 = t57 ? t67 : t66;
            uint t69 = t55 + remu_s_107 + t68;
            uint t70 = remu_gfx10_formats[(remu_s_11 >> 12) & 0x7Fu];
            uint t71 = t70 & 0xFFu;
            uint t72 = remu_format_size(t71);
            uint t73 = (remu_s_11 >> 28) & 3u;
            uint t74 = remu_s_10;
            bool t75 = (remu_s_8 | remu_s_9 | remu_s_10 | remu_s_11) == 0u;
            uint t76 = t74 > remu_s_107 ? t74 - remu_s_107 : 0u;
            bool t77 = t75 || (t73 == 0u ? (t59 >= t74 || t60 >= t56) : t73 == 1u ? (t59 >= t74) : t73 == 2u ? (t74 == 0u) : (t57 && t56 != 0u ? (t59 >= t74 || !remu_range_valid(t60, t72, t56)) : !remu_range_valid(t68, t72, t76)));
            uint t78 = remu_s_11;
            uint t79 = remu_gfx10_formats[(t78 >> 12) & 0x7Fu];
            uint t80 = t79 & 0xFFu;
            uint t81 = (t79 >> 8) & 0xFFu;
            uint t82 = remu_format_size(t80);
            bool t83 = t77 || t82 == 0u || !remu_range_valid(t69, t82, remu_initial_s[142]);
            uint t84 = 0u;
            uint t85 = 0u;
            uint t86 = 0u;
            remu_format_layout(t80, 0u, t84, t85, t86);
            uint t87 = t83 || t86 == 0u ? 0u : remu_load_format_bits(b3, t69, t84, t85, t86);
            uint t88 = t86 == 0u ? 0u : remu_format_convert(t87, t86, t81, t80, 0u);
            uint t89 = 0u;
            uint t90 = 0u;
            uint t91 = 0u;
            remu_format_layout(t80, 1u, t89, t90, t91);
            uint t92 = t83 || t91 == 0u ? 0u : remu_load_format_bits(b3, t69, t89, t90, t91);
            uint t93 = t91 == 0u ? 0u : remu_format_convert(t92, t91, t81, t80, 1u);
            uint t94 = 0u;
            uint t95 = 0u;
            uint t96 = 0u;
            remu_format_layout(t80, 2u, t94, t95, t96);
            uint t97 = t83 || t96 == 0u ? 0u : remu_load_format_bits(b3, t69, t94, t95, t96);
            uint t98 = t96 == 0u ? 0u : remu_format_convert(t97, t96, t81, t80, 2u);
            uint t99 = 0u;
            uint t100 = 0u;
            uint t101 = 0u;
            remu_format_layout(t80, 3u, t99, t100, t101);
            uint t102 = t83 || t101 == 0u ? 0u : remu_load_format_bits(b3, t69, t99, t100, t101);
            uint t103 = t101 == 0u ? remu_format_one(t81) : remu_format_convert(t102, t101, t81, t80, 3u);
            uint t104 = (t78 >> 0u) & 7u;
            uint t105 = t104 == 1u ? remu_format_one(t81) : t83 ? 0u : t104 == 4u ? t88 : t104 == 5u ? t93 : t104 == 6u ? t98 : t104 == 7u ? t103 : 0u;
            uint t106 = (t78 >> 3u) & 7u;
            uint t107 = t106 == 1u ? remu_format_one(t81) : t83 ? 0u : t106 == 4u ? t88 : t106 == 5u ? t93 : t106 == 6u ? t98 : t106 == 7u ? t103 : 0u;
            uint t108 = (t78 >> 6u) & 7u;
            uint t109 = t108 == 1u ? remu_format_one(t81) : t83 ? 0u : t108 == 4u ? t88 : t108 == 5u ? t93 : t108 == 6u ? t98 : t108 == 7u ? t103 : 0u;
            uint t110 = (t78 >> 9u) & 7u;
            uint t111 = t110 == 1u ? remu_format_one(t81) : t83 ? 0u : t110 == 4u ? t88 : t110 == 5u ? t93 : t110 == 6u ? t98 : t110 == 7u ? t103 : 0u;
            if (exec) remu_v_0 = t105;
            if (exec) remu_v_1 = t107;
            if (exec) remu_v_2 = t109;
            if (exec) remu_v_3 = t111;
            uint t112 = (remu_s_106 & 0x1F0u);
            uint t113 = t112;
            remu_s_106 = t113;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t112 != 0u;
            uint t114 = 0x1001Au & 31u;
            uint t115 = min((0x1001Au >> 16) & 0x7Fu, 32u - t114);
            uint t116 = t115 == 0u ? 0u : extract_bits(remu_s_21, t114, t115);
            uint t117 = t116;
            remu_s_107 = t117;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t116 != 0u;
            uint t118 = (remu_s_106 + 0u);
            remu_s_8 = remu_load_u32(b2, t118 + 0u, remu_initial_s[138]);
            remu_s_9 = remu_load_u32(b2, t118 + 4u, remu_initial_s[138]);
            remu_s_10 = remu_load_u32(b2, t118 + 8u, remu_initial_s[138]);
            remu_s_11 = remu_load_u32(b2, t118 + 12u, remu_initial_s[138]);
            scc = as_type<int>(0u) == as_type<int>(remu_s_107);
            ulong t119 = (scc ? ((ulong)remu_s_126 | ((ulong)remu_s_127 << 32)) : 0x0000000000000000ul);
            remu_s_12 = (uint)t119;
            remu_s_13 = (uint)(t119 >> 32);
            uint t120 = 0x70007u & 31u;
            uint t121 = min((0x70007u >> 16) & 0x7Fu, 32u - t120);
            uint t122 = t121 == 0u ? 0u : extract_bits(remu_s_21, t120, t121);
            remu_s_23 = t122;
            scc = t122 != 0u;
            uint t123 = 0x20005u & 31u;
            uint t124 = min((0x20005u >> 16) & 0x7Fu, 32u - t123);
            uint t125 = t124 == 0u ? 0u : extract_bits(remu_s_21, t123, t124);
            uint t126 = t125;
            remu_s_107 = t126;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t125 != 0u;
            uint t127 = (remu_s_11 & 0xFFF80000u);
            remu_s_22 = t127;
            scc = t127 != 0u;
            uint t128 = (remu_s_23 << (0xCu & 31u));
            uint t129 = t128;
            remu_s_106 = t129;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t128 != 0u;
            uint t130 = ((remu_s_107 << 4) + 0xC0000u);
            uint t131 = t130;
            remu_s_107 = t131;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            uint t132 = (remu_s_22 | remu_s_106);
            remu_s_22 = t132;
            scc = t132 != 0u;
            uint t133 = remu_s_107 & 63u;
            uint t134 = min((remu_s_107 >> 16) & 0x7Fu, 64u - t133);
            ulong t135 = t134 >= 64u ? 0xFFFFFFFFFFFFFFFFul : ((1ul << t134) - 1ul);
            ulong t136 = (((ulong)remu_s_14 | ((ulong)remu_s_15 << 32)) >> t133) & t135;
            remu_s_106 = (uint)t136;
            remu_s_107 = (uint)(t136 >> 32);
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t136 != 0ul;
            if (exec) remu_v_4 = ((((remu_s_12) >> remu_lane) & 1u) != 0u) ? (remu_v_5) : (remu_v_8);
            uint t137 = (remu_s_22 | remu_s_106);
            uint t138 = t137;
            remu_s_107 = t138;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t137 != 0u;
            uint t139 = 0xC000Eu & 31u;
            uint t140 = min((0xC000Eu >> 16) & 0x7Fu, 32u - t139);
            uint t141 = t140 == 0u ? 0u : extract_bits(remu_s_21, t139, t140);
            uint t142 = t141;
            remu_s_106 = t142;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t141 != 0u;
            scc = (remu_s_23) == (0u);
            uint t143 = (scc ? remu_s_11 : remu_s_107);
            remu_s_11 = t143;
            ulong t144 = (ulong)remu_s_8 | (((ulong)remu_s_9 & 0xFFFFul) << 32);
            uint t145 = (uint)(t144 & 3ul);
            uint t146 = (remu_s_9 >> 16) & 0x3FFFu;
            bool t147 = (remu_s_9 & 0x80000000u) != 0u;
            bool t148 = (remu_s_11 & 0x00800000u) != 0u;
            uint t149 = remu_v_4 + (t148 ? (remu_lane & 63u) : 0u);
            uint t150 = 0u + 0u;
            uint t151 = 8u << ((remu_s_11 >> 21) & 3u);
            uint t152 = t149 / t151;
            uint t153 = t149 % t151;
            uint t154 = t150 / 4u;
            uint t155 = t150 % 4u;
            uint t156 = t149 * t146 + t150;
            uint t157 = (t152 * t146 + t154 * 4u) * t151 + t153 * 4u + t155;
            uint t158 = t147 ? t157 : t156;
            uint t159 = t145 + remu_s_106 + t158;
            uint t160 = remu_gfx10_formats[(remu_s_11 >> 12) & 0x7Fu];
            uint t161 = t160 & 0xFFu;
            uint t162 = remu_format_size(t161);
            uint t163 = (remu_s_11 >> 28) & 3u;
            uint t164 = remu_s_10;
            bool t165 = (remu_s_8 | remu_s_9 | remu_s_10 | remu_s_11) == 0u;
            uint t166 = t164 > remu_s_106 ? t164 - remu_s_106 : 0u;
            bool t167 = t165 || (t163 == 0u ? (t149 >= t164 || t150 >= t146) : t163 == 1u ? (t149 >= t164) : t163 == 2u ? (t164 == 0u) : (t147 && t146 != 0u ? (t149 >= t164 || !remu_range_valid(t150, t162, t146)) : !remu_range_valid(t158, t162, t166)));
            uint t168 = remu_s_11;
            uint t169 = remu_gfx10_formats[(t168 >> 12) & 0x7Fu];
            uint t170 = t169 & 0xFFu;
            uint t171 = (t169 >> 8) & 0xFFu;
            uint t172 = remu_format_size(t170);
            bool t173 = t167 || t172 == 0u || !remu_range_valid(t159, t172, remu_initial_s[142]);
            uint t174 = 0u;
            uint t175 = 0u;
            uint t176 = 0u;
            remu_format_layout(t170, 0u, t174, t175, t176);
            uint t177 = t173 || t176 == 0u ? 0u : remu_load_format_bits(b3, t159, t174, t175, t176);
            uint t178 = t176 == 0u ? 0u : remu_format_convert(t177, t176, t171, t170, 0u);
            uint t179 = 0u;
            uint t180 = 0u;
            uint t181 = 0u;
            remu_format_layout(t170, 1u, t179, t180, t181);
            uint t182 = t173 || t181 == 0u ? 0u : remu_load_format_bits(b3, t159, t179, t180, t181);
            uint t183 = t181 == 0u ? 0u : remu_format_convert(t182, t181, t171, t170, 1u);
            uint t184 = 0u;
            uint t185 = 0u;
            uint t186 = 0u;
            remu_format_layout(t170, 2u, t184, t185, t186);
            uint t187 = t173 || t186 == 0u ? 0u : remu_load_format_bits(b3, t159, t184, t185, t186);
            uint t188 = t186 == 0u ? 0u : remu_format_convert(t187, t186, t171, t170, 2u);
            uint t189 = 0u;
            uint t190 = 0u;
            uint t191 = 0u;
            remu_format_layout(t170, 3u, t189, t190, t191);
            uint t192 = t173 || t191 == 0u ? 0u : remu_load_format_bits(b3, t159, t189, t190, t191);
            uint t193 = t191 == 0u ? remu_format_one(t171) : remu_format_convert(t192, t191, t171, t170, 3u);
            uint t194 = (t168 >> 0u) & 7u;
            uint t195 = t194 == 1u ? remu_format_one(t171) : t173 ? 0u : t194 == 4u ? t178 : t194 == 5u ? t183 : t194 == 6u ? t188 : t194 == 7u ? t193 : 0u;
            uint t196 = (t168 >> 3u) & 7u;
            uint t197 = t196 == 1u ? remu_format_one(t171) : t173 ? 0u : t196 == 4u ? t178 : t196 == 5u ? t183 : t196 == 6u ? t188 : t196 == 7u ? t193 : 0u;
            if (exec) remu_v_4 = t195;
            if (exec) remu_v_5 = t197;
            if (exec) remu_v_1 = as_type<uint>((fma(as_type<float>(remu_v_1), as_type<float>(remu_s_1), as_type<float>(remu_s_3))));
            if (exec) remu_v_0 = as_type<uint>((fma(as_type<float>(remu_v_0), as_type<float>(remu_s_0), as_type<float>(remu_s_2))));
            if (exec) remu_v_1 = as_type<uint>((as_type<float>(remu_s_17) * as_type<float>(remu_v_1)));
            if (exec) remu_v_0 = as_type<uint>((as_type<float>(remu_s_16) * as_type<float>(remu_v_0)));
            if (exec) remu_v_1 = as_type<uint>((fma(as_type<float>(0x40000000u), as_type<float>(remu_v_1), as_type<float>(0xBF800000u))));
            if (exec) remu_v_0 = as_type<uint>((fma(as_type<float>(0x40000000u), as_type<float>(remu_v_0), as_type<float>(0xBF800000u))));
            uint t198 = as_type<uint>((-(as_type<float>(remu_v_1))));
            if (exec) remu_v_1 = t198;
            if (exec) remu_out.position = float4(as_type<float>(remu_v_0), as_type<float>(remu_v_1), as_type<float>(remu_v_2), as_type<float>(remu_v_3));
            if (exec) remu_v_5 = as_type<uint>((fma(as_type<float>(remu_s_5), as_type<float>(remu_v_5), as_type<float>(remu_s_7))));
            if (exec) remu_v_4 = as_type<uint>((fma(as_type<float>(remu_s_4), as_type<float>(remu_v_4), as_type<float>(remu_s_6))));
            if (exec) remu_v_5 = as_type<uint>((as_type<float>(remu_s_19) * as_type<float>(remu_v_5)));
            if (exec) remu_v_4 = as_type<uint>((as_type<float>(remu_s_18) * as_type<float>(remu_v_4)));
            if (exec) remu_out.param0 = float4(as_type<float>(remu_v_4), as_type<float>(remu_v_5), as_type<float>(remu_v_0), as_type<float>(remu_v_1));
    return remu_out;
}
