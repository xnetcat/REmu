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
    const device uchar* b4 [[buffer(4)]],
    constant uint* remu_initial_s [[buffer(5)]],
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
    uint remu_s_20 = remu_initial_s[20];
    uint remu_s_21 = remu_initial_s[21];
    uint remu_s_22 = remu_initial_s[22];
    uint remu_s_23 = remu_initial_s[23];
    uint remu_s_24 = remu_initial_s[24];
    uint remu_s_25 = remu_initial_s[25];
    uint remu_s_26 = remu_initial_s[26];
    uint remu_s_27 = remu_initial_s[27];
    uint remu_s_28 = remu_initial_s[28];
    uint remu_s_29 = remu_initial_s[29];
    uint remu_s_30 = remu_initial_s[30];
    uint remu_s_31 = remu_initial_s[31];
    uint remu_s_32 = remu_initial_s[32];
    uint remu_s_33 = remu_initial_s[33];
    uint remu_s_34 = remu_initial_s[34];
    uint remu_s_35 = remu_initial_s[35];
    uint remu_s_36 = remu_initial_s[36];
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
    uint remu_v_6 = 0u;
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
            remu_s_36 = remu_load_u32(b0, t14 + 0u, remu_initial_s[130]);
            uint t15 = (0u + 0u);
            remu_s_0 = remu_load_u32(b1, t15 + 0u, remu_initial_s[134]);
            remu_s_1 = remu_load_u32(b1, t15 + 4u, remu_initial_s[134]);
            remu_s_2 = remu_load_u32(b1, t15 + 8u, remu_initial_s[134]);
            remu_s_3 = remu_load_u32(b1, t15 + 12u, remu_initial_s[134]);
            remu_s_8 = 0x22C0204u;
            remu_s_9 = 0xFAC03ACu;
            uint t16 = (0u + 0u);
            remu_s_20 = remu_load_u32(b2, t16 + 0u, remu_initial_s[138]);
            remu_s_21 = remu_load_u32(b2, t16 + 4u, remu_initial_s[138]);
            remu_s_22 = remu_load_u32(b2, t16 + 8u, remu_initial_s[138]);
            remu_s_23 = remu_load_u32(b2, t16 + 12u, remu_initial_s[138]);
            remu_s_24 = remu_load_u32(b2, t16 + 16u, remu_initial_s[138]);
            remu_s_25 = remu_load_u32(b2, t16 + 20u, remu_initial_s[138]);
            remu_s_26 = remu_load_u32(b2, t16 + 24u, remu_initial_s[138]);
            remu_s_27 = remu_load_u32(b2, t16 + 28u, remu_initial_s[138]);
            remu_s_28 = remu_load_u32(b2, t16 + 32u, remu_initial_s[138]);
            remu_s_29 = remu_load_u32(b2, t16 + 36u, remu_initial_s[138]);
            remu_s_30 = remu_load_u32(b2, t16 + 40u, remu_initial_s[138]);
            remu_s_31 = remu_load_u32(b2, t16 + 44u, remu_initial_s[138]);
            remu_s_32 = remu_load_u32(b2, t16 + 48u, remu_initial_s[138]);
            remu_s_33 = remu_load_u32(b2, t16 + 52u, remu_initial_s[138]);
            remu_s_34 = remu_load_u32(b2, t16 + 56u, remu_initial_s[138]);
            remu_s_35 = remu_load_u32(b2, t16 + 60u, remu_initial_s[138]);
            uint t17 = (remu_s_36 << (4u & 31u));
            uint t18 = t17;
            remu_s_107 = t18;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t17 != 0u;
            uint t19 = 0x1001Au & 31u;
            uint t20 = min((0x1001Au >> 16) & 0x7Fu, 32u - t19);
            uint t21 = t20 == 0u ? 0u : extract_bits(remu_s_36, t19, t20);
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
            remu_s_4 = remu_load_u32(b3, t25 + 0u, remu_initial_s[142]);
            remu_s_5 = remu_load_u32(b3, t25 + 4u, remu_initial_s[142]);
            remu_s_6 = remu_load_u32(b3, t25 + 8u, remu_initial_s[142]);
            remu_s_7 = remu_load_u32(b3, t25 + 12u, remu_initial_s[142]);
            ulong t26 = (scc ? ((ulong)remu_s_126 | ((ulong)remu_s_127 << 32)) : 0x0000000000000000ul);
            remu_s_10 = (uint)t26;
            remu_s_11 = (uint)(t26 >> 32);
            uint t27 = 0x70007u & 31u;
            uint t28 = min((0x70007u >> 16) & 0x7Fu, 32u - t27);
            uint t29 = t28 == 0u ? 0u : extract_bits(remu_s_36, t27, t28);
            remu_s_12 = t29;
            scc = t29 != 0u;
            uint t30 = 0x20005u & 31u;
            uint t31 = min((0x20005u >> 16) & 0x7Fu, 32u - t30);
            uint t32 = t31 == 0u ? 0u : extract_bits(remu_s_36, t30, t31);
            uint t33 = t32;
            remu_s_106 = t33;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t32 != 0u;
            uint t34 = (remu_s_12 << (0xCu & 31u));
            remu_s_13 = t34;
            scc = t34 != 0u;
            uint t35 = ((remu_s_106 << 4) + 0xC0000u);
            uint t36 = t35;
            remu_s_107 = t36;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            uint t37 = (remu_s_7 & 0xFFF80000u);
            uint t38 = t37;
            remu_s_106 = t38;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t37 != 0u;
            uint t39 = remu_s_107 & 63u;
            uint t40 = min((remu_s_107 >> 16) & 0x7Fu, 64u - t39);
            ulong t41 = t40 >= 64u ? 0xFFFFFFFFFFFFFFFFul : ((1ul << t40) - 1ul);
            ulong t42 = (((ulong)remu_s_8 | ((ulong)remu_s_9 << 32)) >> t39) & t41;
            remu_s_8 = (uint)t42;
            remu_s_9 = (uint)(t42 >> 32);
            scc = t42 != 0ul;
            uint t43 = (remu_s_106 | remu_s_13);
            uint t44 = t43;
            remu_s_106 = t44;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t43 != 0u;
            if (exec) remu_v_0 = ((((remu_s_10) >> remu_lane) & 1u) != 0u) ? (remu_v_5) : (remu_v_8);
            uint t45 = (remu_s_106 | remu_s_8);
            uint t46 = t45;
            remu_s_107 = t46;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t45 != 0u;
            uint t47 = 0xC000Eu & 31u;
            uint t48 = min((0xC000Eu >> 16) & 0x7Fu, 32u - t47);
            uint t49 = t48 == 0u ? 0u : extract_bits(remu_s_36, t47, t48);
            uint t50 = t49;
            remu_s_106 = t50;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t49 != 0u;
            scc = (remu_s_12) == (0u);
            uint t51 = (scc ? remu_s_7 : remu_s_107);
            remu_s_7 = t51;
            ulong t52 = (ulong)remu_s_4 | (((ulong)remu_s_5 & 0xFFFFul) << 32);
            uint t53 = (uint)(t52 & 3ul);
            uint t54 = (remu_s_5 >> 16) & 0x3FFFu;
            bool t55 = (remu_s_5 & 0x80000000u) != 0u;
            bool t56 = (remu_s_7 & 0x00800000u) != 0u;
            uint t57 = remu_v_0 + (t56 ? (remu_lane & 63u) : 0u);
            uint t58 = 0u + 0u;
            uint t59 = 8u << ((remu_s_7 >> 21) & 3u);
            uint t60 = t57 / t59;
            uint t61 = t57 % t59;
            uint t62 = t58 / 4u;
            uint t63 = t58 % 4u;
            uint t64 = t57 * t54 + t58;
            uint t65 = (t60 * t54 + t62 * 4u) * t59 + t61 * 4u + t63;
            uint t66 = t55 ? t65 : t64;
            uint t67 = t53 + remu_s_106 + t66;
            uint t68 = remu_gfx10_formats[(remu_s_7 >> 12) & 0x7Fu];
            uint t69 = t68 & 0xFFu;
            uint t70 = remu_format_size(t69);
            uint t71 = (remu_s_7 >> 28) & 3u;
            uint t72 = remu_s_6;
            bool t73 = (remu_s_4 | remu_s_5 | remu_s_6 | remu_s_7) == 0u;
            uint t74 = t72 > remu_s_106 ? t72 - remu_s_106 : 0u;
            bool t75 = t73 || (t71 == 0u ? (t57 >= t72 || t58 >= t54) : t71 == 1u ? (t57 >= t72) : t71 == 2u ? (t72 == 0u) : (t55 && t54 != 0u ? (t57 >= t72 || !remu_range_valid(t58, t70, t54)) : !remu_range_valid(t66, t70, t74)));
            uint t76 = remu_s_7;
            uint t77 = remu_gfx10_formats[(t76 >> 12) & 0x7Fu];
            uint t78 = t77 & 0xFFu;
            uint t79 = (t77 >> 8) & 0xFFu;
            uint t80 = remu_format_size(t78);
            bool t81 = t75 || t80 == 0u || !remu_range_valid(t67, t80, remu_initial_s[146]);
            uint t82 = 0u;
            uint t83 = 0u;
            uint t84 = 0u;
            remu_format_layout(t78, 0u, t82, t83, t84);
            uint t85 = t81 || t84 == 0u ? 0u : remu_load_format_bits(b4, t67, t82, t83, t84);
            uint t86 = t84 == 0u ? 0u : remu_format_convert(t85, t84, t79, t78, 0u);
            uint t87 = 0u;
            uint t88 = 0u;
            uint t89 = 0u;
            remu_format_layout(t78, 1u, t87, t88, t89);
            uint t90 = t81 || t89 == 0u ? 0u : remu_load_format_bits(b4, t67, t87, t88, t89);
            uint t91 = t89 == 0u ? 0u : remu_format_convert(t90, t89, t79, t78, 1u);
            uint t92 = 0u;
            uint t93 = 0u;
            uint t94 = 0u;
            remu_format_layout(t78, 2u, t92, t93, t94);
            uint t95 = t81 || t94 == 0u ? 0u : remu_load_format_bits(b4, t67, t92, t93, t94);
            uint t96 = t94 == 0u ? 0u : remu_format_convert(t95, t94, t79, t78, 2u);
            uint t97 = 0u;
            uint t98 = 0u;
            uint t99 = 0u;
            remu_format_layout(t78, 3u, t97, t98, t99);
            uint t100 = t81 || t99 == 0u ? 0u : remu_load_format_bits(b4, t67, t97, t98, t99);
            uint t101 = t99 == 0u ? remu_format_one(t79) : remu_format_convert(t100, t99, t79, t78, 3u);
            uint t102 = (t76 >> 0u) & 7u;
            uint t103 = t102 == 1u ? remu_format_one(t79) : t81 ? 0u : t102 == 4u ? t86 : t102 == 5u ? t91 : t102 == 6u ? t96 : t102 == 7u ? t101 : 0u;
            uint t104 = (t76 >> 3u) & 7u;
            uint t105 = t104 == 1u ? remu_format_one(t79) : t81 ? 0u : t104 == 4u ? t86 : t104 == 5u ? t91 : t104 == 6u ? t96 : t104 == 7u ? t101 : 0u;
            uint t106 = (t76 >> 6u) & 7u;
            uint t107 = t106 == 1u ? remu_format_one(t79) : t81 ? 0u : t106 == 4u ? t86 : t106 == 5u ? t91 : t106 == 6u ? t96 : t106 == 7u ? t101 : 0u;
            if (exec) remu_v_0 = t103;
            if (exec) remu_v_1 = t105;
            if (exec) remu_v_2 = t107;
            if (exec) remu_v_3 = as_type<uint>((fma(as_type<float>(remu_v_2), as_type<float>(remu_s_3), as_type<float>(remu_s_2))));
            if (exec) remu_v_1 = as_type<uint>((fma(as_type<float>(remu_v_1), as_type<float>(remu_s_3), as_type<float>(remu_s_1))));
            if (exec) remu_v_0 = as_type<uint>((fma(as_type<float>(remu_v_0), as_type<float>(remu_s_3), as_type<float>(remu_s_0))));
            if (exec) remu_v_4 = as_type<uint>((fma(as_type<float>(remu_s_31), as_type<float>(remu_v_3), as_type<float>(remu_s_35))));
            if (exec) remu_v_5 = as_type<uint>((fma(as_type<float>(remu_s_30), as_type<float>(remu_v_3), as_type<float>(remu_s_34))));
            if (exec) remu_v_2 = as_type<uint>((fma(as_type<float>(remu_s_29), as_type<float>(remu_v_3), as_type<float>(remu_s_33))));
            if (exec) remu_v_6 = as_type<uint>((fma(as_type<float>(remu_s_28), as_type<float>(remu_v_3), as_type<float>(remu_s_32))));
            if (exec) remu_v_3 = as_type<uint>((fma(as_type<float>(remu_s_27), as_type<float>(remu_v_1), as_type<float>(remu_v_4))));
            if (exec) remu_v_4 = as_type<uint>((fma(as_type<float>(remu_s_26), as_type<float>(remu_v_1), as_type<float>(remu_v_5))));
            if (exec) remu_v_2 = as_type<uint>((fma(as_type<float>(remu_s_25), as_type<float>(remu_v_1), as_type<float>(remu_v_2))));
            if (exec) remu_v_6 = as_type<uint>((fma(as_type<float>(remu_s_24), as_type<float>(remu_v_1), as_type<float>(remu_v_6))));
            if (exec) remu_v_3 = as_type<uint>((fma(as_type<float>(remu_s_23), as_type<float>(remu_v_0), as_type<float>(remu_v_3))));
            if (exec) remu_v_4 = as_type<uint>((fma(as_type<float>(remu_s_22), as_type<float>(remu_v_0), as_type<float>(remu_v_4))));
            if (exec) remu_v_2 = as_type<uint>((fma(as_type<float>(remu_s_21), as_type<float>(remu_v_0), as_type<float>(remu_v_2))));
            if (exec) remu_v_6 = as_type<uint>((fma(as_type<float>(remu_s_20), as_type<float>(remu_v_0), as_type<float>(remu_v_6))));
            if (exec) remu_out.position = float4(as_type<float>(remu_v_6), as_type<float>(remu_v_2), as_type<float>(remu_v_4), as_type<float>(remu_v_3));
    return remu_out;
}
