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
    float clip_distances [[clip_distance]] [8];
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
    uint remu_s_28 = remu_initial_s[28];
    uint remu_s_29 = remu_initial_s[29];
    uint remu_s_30 = remu_initial_s[30];
    uint remu_s_31 = remu_initial_s[31];
    uint remu_s_32 = remu_initial_s[32];
    uint remu_s_33 = remu_initial_s[33];
    uint remu_s_34 = remu_initial_s[34];
    uint remu_s_35 = remu_initial_s[35];
    uint remu_s_36 = remu_initial_s[36];
    uint remu_s_37 = remu_initial_s[37];
    uint remu_s_38 = remu_initial_s[38];
    uint remu_s_39 = remu_initial_s[39];
    uint remu_s_40 = remu_initial_s[40];
    uint remu_s_41 = remu_initial_s[41];
    uint remu_s_42 = remu_initial_s[42];
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
    uint remu_v_7 = 0u;
    uint remu_v_8 = 0u;
    uint remu_v_9 = 0u;
    uint remu_v_10 = 0u;
    uint remu_v_11 = 0u;
    uint remu_v_12 = 0u;
    uint remu_v_13 = 0u;
    uint remu_v_14 = 0u;
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
            remu_s_37 = remu_load_u32(b0, t14 + 4u, remu_initial_s[130]);
            uint t15 = (0u + 0x460u);
            remu_s_0 = remu_load_u32(b1, t15 + 0u, remu_initial_s[134]);
            remu_s_1 = remu_load_u32(b1, t15 + 4u, remu_initial_s[134]);
            remu_s_2 = remu_load_u32(b1, t15 + 8u, remu_initial_s[134]);
            remu_s_3 = remu_load_u32(b1, t15 + 12u, remu_initial_s[134]);
            remu_s_18 = 0x22C0204u;
            remu_s_19 = 0xFAC03ACu;
            uint t16 = (0u + 0x7F0u);
            remu_s_4 = remu_load_u32(b1, t16 + 0u, remu_initial_s[134]);
            remu_s_5 = remu_load_u32(b1, t16 + 4u, remu_initial_s[134]);
            remu_s_6 = remu_load_u32(b1, t16 + 8u, remu_initial_s[134]);
            remu_s_7 = remu_load_u32(b1, t16 + 12u, remu_initial_s[134]);
            uint t17 = (0u + 0u);
            remu_s_20 = remu_load_u32(b1, t17 + 0u, remu_initial_s[134]);
            remu_s_21 = remu_load_u32(b1, t17 + 4u, remu_initial_s[134]);
            remu_s_22 = remu_load_u32(b1, t17 + 8u, remu_initial_s[134]);
            remu_s_23 = remu_load_u32(b1, t17 + 12u, remu_initial_s[134]);
            remu_s_24 = remu_load_u32(b1, t17 + 16u, remu_initial_s[134]);
            remu_s_25 = remu_load_u32(b1, t17 + 20u, remu_initial_s[134]);
            remu_s_26 = remu_load_u32(b1, t17 + 24u, remu_initial_s[134]);
            remu_s_27 = remu_load_u32(b1, t17 + 28u, remu_initial_s[134]);
            remu_s_28 = remu_load_u32(b1, t17 + 32u, remu_initial_s[134]);
            remu_s_29 = remu_load_u32(b1, t17 + 36u, remu_initial_s[134]);
            remu_s_30 = remu_load_u32(b1, t17 + 40u, remu_initial_s[134]);
            remu_s_31 = remu_load_u32(b1, t17 + 44u, remu_initial_s[134]);
            remu_s_32 = remu_load_u32(b1, t17 + 48u, remu_initial_s[134]);
            remu_s_33 = remu_load_u32(b1, t17 + 52u, remu_initial_s[134]);
            remu_s_34 = remu_load_u32(b1, t17 + 56u, remu_initial_s[134]);
            remu_s_35 = remu_load_u32(b1, t17 + 60u, remu_initial_s[134]);
            uint t18 = (remu_s_37 << (4u & 31u));
            uint t19 = t18;
            remu_s_107 = t19;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t18 != 0u;
            uint t20 = 0x1001Au & 31u;
            uint t21 = min((0x1001Au >> 16) & 0x7Fu, 32u - t20);
            uint t22 = t21 == 0u ? 0u : extract_bits(remu_s_37, t20, t21);
            uint t23 = t22;
            remu_s_106 = t23;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t22 != 0u;
            uint t24 = (remu_s_107 & 0x1F0u);
            uint t25 = t24;
            remu_s_107 = t25;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t24 != 0u;
            scc = as_type<int>(0u) == as_type<int>(remu_s_106);
            uint t26 = (remu_s_107 + 0u);
            remu_s_12 = remu_load_u32(b2, t26 + 0u, remu_initial_s[138]);
            remu_s_13 = remu_load_u32(b2, t26 + 4u, remu_initial_s[138]);
            remu_s_14 = remu_load_u32(b2, t26 + 8u, remu_initial_s[138]);
            remu_s_15 = remu_load_u32(b2, t26 + 12u, remu_initial_s[138]);
            ulong t27 = (scc ? ((ulong)remu_s_126 | ((ulong)remu_s_127 << 32)) : 0x0000000000000000ul);
            remu_s_38 = (uint)t27;
            remu_s_39 = (uint)(t27 >> 32);
            uint t28 = 0x70007u & 31u;
            uint t29 = min((0x70007u >> 16) & 0x7Fu, 32u - t28);
            uint t30 = t29 == 0u ? 0u : extract_bits(remu_s_37, t28, t29);
            remu_s_3 = t30;
            scc = t30 != 0u;
            uint t31 = 0x20005u & 31u;
            uint t32 = min((0x20005u >> 16) & 0x7Fu, 32u - t31);
            uint t33 = t32 == 0u ? 0u : extract_bits(remu_s_37, t31, t32);
            uint t34 = t33;
            remu_s_106 = t34;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t33 != 0u;
            uint t35 = (remu_s_3 << (0xCu & 31u));
            remu_s_42 = t35;
            scc = t35 != 0u;
            uint t36 = ((remu_s_106 << 4) + 0xC0000u);
            uint t37 = t36;
            remu_s_107 = t37;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            uint t38 = (remu_s_15 & 0xFFF80000u);
            uint t39 = t38;
            remu_s_106 = t39;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t38 != 0u;
            uint t40 = remu_s_107 & 63u;
            uint t41 = min((remu_s_107 >> 16) & 0x7Fu, 64u - t40);
            ulong t42 = t41 >= 64u ? 0xFFFFFFFFFFFFFFFFul : ((1ul << t41) - 1ul);
            ulong t43 = (((ulong)remu_s_18 | ((ulong)remu_s_19 << 32)) >> t40) & t42;
            remu_s_40 = (uint)t43;
            remu_s_41 = (uint)(t43 >> 32);
            scc = t43 != 0ul;
            uint t44 = (remu_s_106 | remu_s_42);
            uint t45 = t44;
            remu_s_106 = t45;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t44 != 0u;
            if (exec) remu_v_0 = ((((remu_s_38) >> remu_lane) & 1u) != 0u) ? (remu_v_5) : (remu_v_8);
            uint t46 = (remu_s_106 | remu_s_40);
            uint t47 = t46;
            remu_s_106 = t47;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t46 != 0u;
            uint t48 = 0xC000Eu & 31u;
            uint t49 = min((0xC000Eu >> 16) & 0x7Fu, 32u - t48);
            uint t50 = t49 == 0u ? 0u : extract_bits(remu_s_37, t48, t49);
            uint t51 = t50;
            remu_s_107 = t51;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t50 != 0u;
            scc = (remu_s_3) == (0u);
            uint t52 = (scc ? remu_s_15 : remu_s_106);
            remu_s_15 = t52;
            uint t53 = (remu_s_36 << (4u & 31u));
            uint t54 = t53;
            remu_s_106 = t54;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t53 != 0u;
            ulong t55 = (ulong)remu_s_12 | (((ulong)remu_s_13 & 0xFFFFul) << 32);
            uint t56 = (uint)(t55 & 3ul);
            uint t57 = (remu_s_13 >> 16) & 0x3FFFu;
            bool t58 = (remu_s_13 & 0x80000000u) != 0u;
            bool t59 = (remu_s_15 & 0x00800000u) != 0u;
            uint t60 = remu_v_0 + (t59 ? (remu_lane & 63u) : 0u);
            uint t61 = 0u + 0u;
            uint t62 = 8u << ((remu_s_15 >> 21) & 3u);
            uint t63 = t60 / t62;
            uint t64 = t60 % t62;
            uint t65 = t61 / 4u;
            uint t66 = t61 % 4u;
            uint t67 = t60 * t57 + t61;
            uint t68 = (t63 * t57 + t65 * 4u) * t62 + t64 * 4u + t66;
            uint t69 = t58 ? t68 : t67;
            uint t70 = t56 + remu_s_107 + t69;
            uint t71 = remu_gfx10_formats[(remu_s_15 >> 12) & 0x7Fu];
            uint t72 = t71 & 0xFFu;
            uint t73 = remu_format_size(t72);
            uint t74 = (remu_s_15 >> 28) & 3u;
            uint t75 = remu_s_14;
            bool t76 = (remu_s_12 | remu_s_13 | remu_s_14 | remu_s_15) == 0u;
            uint t77 = t75 > remu_s_107 ? t75 - remu_s_107 : 0u;
            bool t78 = t76 || (t74 == 0u ? (t60 >= t75 || t61 >= t57) : t74 == 1u ? (t60 >= t75) : t74 == 2u ? (t75 == 0u) : (t58 && t57 != 0u ? (t60 >= t75 || !remu_range_valid(t61, t73, t57)) : !remu_range_valid(t69, t73, t77)));
            uint t79 = remu_s_15;
            uint t80 = remu_gfx10_formats[(t79 >> 12) & 0x7Fu];
            uint t81 = t80 & 0xFFu;
            uint t82 = (t80 >> 8) & 0xFFu;
            uint t83 = remu_format_size(t81);
            bool t84 = t78 || t83 == 0u || !remu_range_valid(t70, t83, remu_initial_s[134]);
            uint t85 = 0u;
            uint t86 = 0u;
            uint t87 = 0u;
            remu_format_layout(t81, 0u, t85, t86, t87);
            uint t88 = t84 || t87 == 0u ? 0u : remu_load_format_bits(b1, t70, t85, t86, t87);
            uint t89 = t87 == 0u ? 0u : remu_format_convert(t88, t87, t82, t81, 0u);
            uint t90 = 0u;
            uint t91 = 0u;
            uint t92 = 0u;
            remu_format_layout(t81, 1u, t90, t91, t92);
            uint t93 = t84 || t92 == 0u ? 0u : remu_load_format_bits(b1, t70, t90, t91, t92);
            uint t94 = t92 == 0u ? 0u : remu_format_convert(t93, t92, t82, t81, 1u);
            uint t95 = 0u;
            uint t96 = 0u;
            uint t97 = 0u;
            remu_format_layout(t81, 2u, t95, t96, t97);
            uint t98 = t84 || t97 == 0u ? 0u : remu_load_format_bits(b1, t70, t95, t96, t97);
            uint t99 = t97 == 0u ? 0u : remu_format_convert(t98, t97, t82, t81, 2u);
            uint t100 = 0u;
            uint t101 = 0u;
            uint t102 = 0u;
            remu_format_layout(t81, 3u, t100, t101, t102);
            uint t103 = t84 || t102 == 0u ? 0u : remu_load_format_bits(b1, t70, t100, t101, t102);
            uint t104 = t102 == 0u ? remu_format_one(t82) : remu_format_convert(t103, t102, t82, t81, 3u);
            uint t105 = (t79 >> 0u) & 7u;
            uint t106 = t105 == 1u ? remu_format_one(t82) : t84 ? 0u : t105 == 4u ? t89 : t105 == 5u ? t94 : t105 == 6u ? t99 : t105 == 7u ? t104 : 0u;
            if (exec) remu_v_3 = t106;
            uint t107 = (remu_s_106 & 0x1F0u);
            uint t108 = t107;
            remu_s_107 = t108;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t107 != 0u;
            uint t109 = 0x1001Au & 31u;
            uint t110 = min((0x1001Au >> 16) & 0x7Fu, 32u - t109);
            uint t111 = t110 == 0u ? 0u : extract_bits(remu_s_36, t109, t110);
            uint t112 = t111;
            remu_s_106 = t112;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t111 != 0u;
            uint t113 = (remu_s_107 + 0u);
            remu_s_12 = remu_load_u32(b2, t113 + 0u, remu_initial_s[138]);
            remu_s_13 = remu_load_u32(b2, t113 + 4u, remu_initial_s[138]);
            remu_s_14 = remu_load_u32(b2, t113 + 8u, remu_initial_s[138]);
            remu_s_15 = remu_load_u32(b2, t113 + 12u, remu_initial_s[138]);
            scc = as_type<int>(0u) == as_type<int>(remu_s_106);
            ulong t114 = (scc ? ((ulong)remu_s_126 | ((ulong)remu_s_127 << 32)) : 0x0000000000000000ul);
            remu_s_16 = (uint)t114;
            remu_s_17 = (uint)(t114 >> 32);
            uint t115 = 0x70007u & 31u;
            uint t116 = min((0x70007u >> 16) & 0x7Fu, 32u - t115);
            uint t117 = t116 == 0u ? 0u : extract_bits(remu_s_36, t115, t116);
            remu_s_38 = t117;
            scc = t117 != 0u;
            uint t118 = 0x20005u & 31u;
            uint t119 = min((0x20005u >> 16) & 0x7Fu, 32u - t118);
            uint t120 = t119 == 0u ? 0u : extract_bits(remu_s_36, t118, t119);
            uint t121 = t120;
            remu_s_107 = t121;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t120 != 0u;
            uint t122 = (remu_s_15 & 0xFFF80000u);
            remu_s_37 = t122;
            scc = t122 != 0u;
            uint t123 = (remu_s_38 << (0xCu & 31u));
            uint t124 = t123;
            remu_s_106 = t124;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t123 != 0u;
            uint t125 = ((remu_s_107 << 4) + 0xC0000u);
            uint t126 = t125;
            remu_s_107 = t126;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            uint t127 = (remu_s_37 | remu_s_106);
            remu_s_37 = t127;
            scc = t127 != 0u;
            uint t128 = remu_s_107 & 63u;
            uint t129 = min((remu_s_107 >> 16) & 0x7Fu, 64u - t128);
            ulong t130 = t129 >= 64u ? 0xFFFFFFFFFFFFFFFFul : ((1ul << t129) - 1ul);
            ulong t131 = (((ulong)remu_s_18 | ((ulong)remu_s_19 << 32)) >> t128) & t130;
            remu_s_106 = (uint)t131;
            remu_s_107 = (uint)(t131 >> 32);
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t131 != 0ul;
            if (exec) remu_v_0 = ((((remu_s_16) >> remu_lane) & 1u) != 0u) ? (remu_v_5) : (remu_v_8);
            uint t132 = (remu_s_37 | remu_s_106);
            uint t133 = t132;
            remu_s_107 = t133;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t132 != 0u;
            uint t134 = 0xC000Eu & 31u;
            uint t135 = min((0xC000Eu >> 16) & 0x7Fu, 32u - t134);
            uint t136 = t135 == 0u ? 0u : extract_bits(remu_s_36, t134, t135);
            uint t137 = t136;
            remu_s_106 = t137;
            vcc = (((remu_s_106) >> remu_lane) & 1u) != 0u;
            scc = t136 != 0u;
            scc = (remu_s_38) == (0u);
            uint t138 = (scc ? remu_s_15 : remu_s_107);
            remu_s_15 = t138;
            ulong t139 = (ulong)remu_s_12 | (((ulong)remu_s_13 & 0xFFFFul) << 32);
            uint t140 = (uint)(t139 & 3ul);
            uint t141 = (remu_s_13 >> 16) & 0x3FFFu;
            bool t142 = (remu_s_13 & 0x80000000u) != 0u;
            bool t143 = (remu_s_15 & 0x00800000u) != 0u;
            uint t144 = remu_v_0 + (t143 ? (remu_lane & 63u) : 0u);
            uint t145 = 0u + 0u;
            uint t146 = 8u << ((remu_s_15 >> 21) & 3u);
            uint t147 = t144 / t146;
            uint t148 = t144 % t146;
            uint t149 = t145 / 4u;
            uint t150 = t145 % 4u;
            uint t151 = t144 * t141 + t145;
            uint t152 = (t147 * t141 + t149 * 4u) * t146 + t148 * 4u + t150;
            uint t153 = t142 ? t152 : t151;
            uint t154 = t140 + remu_s_106 + t153;
            uint t155 = remu_gfx10_formats[(remu_s_15 >> 12) & 0x7Fu];
            uint t156 = t155 & 0xFFu;
            uint t157 = remu_format_size(t156);
            uint t158 = (remu_s_15 >> 28) & 3u;
            uint t159 = remu_s_14;
            bool t160 = (remu_s_12 | remu_s_13 | remu_s_14 | remu_s_15) == 0u;
            uint t161 = t159 > remu_s_106 ? t159 - remu_s_106 : 0u;
            bool t162 = t160 || (t158 == 0u ? (t144 >= t159 || t145 >= t141) : t158 == 1u ? (t144 >= t159) : t158 == 2u ? (t159 == 0u) : (t142 && t141 != 0u ? (t144 >= t159 || !remu_range_valid(t145, t157, t141)) : !remu_range_valid(t153, t157, t161)));
            uint t163 = remu_s_15;
            uint t164 = remu_gfx10_formats[(t163 >> 12) & 0x7Fu];
            uint t165 = t164 & 0xFFu;
            uint t166 = (t164 >> 8) & 0xFFu;
            uint t167 = remu_format_size(t165);
            bool t168 = t162 || t167 == 0u || !remu_range_valid(t154, t167, remu_initial_s[142]);
            uint t169 = 0u;
            uint t170 = 0u;
            uint t171 = 0u;
            remu_format_layout(t165, 0u, t169, t170, t171);
            uint t172 = t168 || t171 == 0u ? 0u : remu_load_format_bits(b3, t154, t169, t170, t171);
            uint t173 = t171 == 0u ? 0u : remu_format_convert(t172, t171, t166, t165, 0u);
            uint t174 = 0u;
            uint t175 = 0u;
            uint t176 = 0u;
            remu_format_layout(t165, 1u, t174, t175, t176);
            uint t177 = t168 || t176 == 0u ? 0u : remu_load_format_bits(b3, t154, t174, t175, t176);
            uint t178 = t176 == 0u ? 0u : remu_format_convert(t177, t176, t166, t165, 1u);
            uint t179 = 0u;
            uint t180 = 0u;
            uint t181 = 0u;
            remu_format_layout(t165, 2u, t179, t180, t181);
            uint t182 = t168 || t181 == 0u ? 0u : remu_load_format_bits(b3, t154, t179, t180, t181);
            uint t183 = t181 == 0u ? 0u : remu_format_convert(t182, t181, t166, t165, 2u);
            uint t184 = 0u;
            uint t185 = 0u;
            uint t186 = 0u;
            remu_format_layout(t165, 3u, t184, t185, t186);
            uint t187 = t168 || t186 == 0u ? 0u : remu_load_format_bits(b3, t154, t184, t185, t186);
            uint t188 = t186 == 0u ? remu_format_one(t166) : remu_format_convert(t187, t186, t166, t165, 3u);
            uint t189 = (t163 >> 0u) & 7u;
            uint t190 = t189 == 1u ? remu_format_one(t166) : t168 ? 0u : t189 == 4u ? t173 : t189 == 5u ? t178 : t189 == 6u ? t183 : t189 == 7u ? t188 : 0u;
            uint t191 = (t163 >> 3u) & 7u;
            uint t192 = t191 == 1u ? remu_format_one(t166) : t168 ? 0u : t191 == 4u ? t173 : t191 == 5u ? t178 : t191 == 6u ? t183 : t191 == 7u ? t188 : 0u;
            uint t193 = (t163 >> 6u) & 7u;
            uint t194 = t193 == 1u ? remu_format_one(t166) : t168 ? 0u : t193 == 4u ? t173 : t193 == 5u ? t178 : t193 == 6u ? t183 : t193 == 7u ? t188 : 0u;
            if (exec) remu_v_0 = t190;
            if (exec) remu_v_1 = t192;
            if (exec) remu_v_2 = t194;
            if (exec) remu_v_3 = (remu_v_3 << (2u & 31u));
            if (exec) remu_v_5 = ((remu_v_3 << (3u & 31u)) + remu_v_3);
            if (exec) remu_v_4 = (1u + remu_v_5);
            if (exec) remu_v_3 = (2u + remu_v_5);
            if (exec) remu_v_6 = (3u + remu_v_5);
            ulong t195 = (ulong)remu_s_8 | (((ulong)remu_s_9 & 0xFFFFul) << 32);
            uint t196 = (uint)(t195 & 3ul);
            uint t197 = (remu_s_9 >> 16) & 0x3FFFu;
            bool t198 = (remu_s_9 & 0x80000000u) != 0u;
            bool t199 = (remu_s_11 & 0x00800000u) != 0u;
            uint t200 = remu_v_5 + (t199 ? (remu_lane & 63u) : 0u);
            uint t201 = 0u + 0u;
            uint t202 = 8u << ((remu_s_11 >> 21) & 3u);
            uint t203 = t200 / t202;
            uint t204 = t200 % t202;
            uint t205 = t201 / 4u;
            uint t206 = t201 % 4u;
            uint t207 = t200 * t197 + t201;
            uint t208 = (t203 * t197 + t205 * 4u) * t202 + t204 * 4u + t206;
            uint t209 = t198 ? t208 : t207;
            uint t210 = t196 + 0u + t209;
            uint t211 = (remu_s_11 >> 28) & 3u;
            uint t212 = remu_s_10;
            bool t213 = (remu_s_8 | remu_s_9 | remu_s_10 | remu_s_11) == 0u;
            uint t214 = t212 > 0u ? t212 - 0u : 0u;
            bool t215 = t213 || (t211 == 0u ? (t200 >= t212 || t201 >= t197) : t211 == 1u ? (t200 >= t212) : t211 == 2u ? (t212 == 0u) : (t198 && t197 != 0u ? (t200 >= t212 || !remu_range_valid(t201, 4u, t197)) : !remu_range_valid(t209, 4u, t214)));
            bool t216 = t213 || (t211 == 0u ? (t200 >= t212 || t201 >= t197) : t211 == 1u ? (t200 >= t212) : t211 == 2u ? (t212 == 0u) : (t198 && t197 != 0u ? (t200 >= t212 || !remu_range_valid(t201, 4u, t197)) : !remu_range_valid(t209, 4u, t214)));
            bool t217 = t213 || (t211 == 0u ? (t200 >= t212 || t201 + 4u >= t197) : t211 == 1u ? (t200 >= t212) : t211 == 2u ? (t212 == 0u) : (t198 && t197 != 0u ? (t200 >= t212 || !remu_range_valid(t201 + 4u, 4u, t197)) : !remu_range_valid(t209 + 4u, 4u, t214)));
            bool t218 = t213 || (t211 == 0u ? (t200 >= t212 || t201 + 8u >= t197) : t211 == 1u ? (t200 >= t212) : t211 == 2u ? (t212 == 0u) : (t198 && t197 != 0u ? (t200 >= t212 || !remu_range_valid(t201 + 8u, 4u, t197)) : !remu_range_valid(t209 + 8u, 4u, t214)));
            uint t219 = t210 & ~3u;
            uint t220 = !t216 && remu_range_valid(t219, 4u, remu_initial_s[146]) ? remu_load_u32(b4, t219, remu_initial_s[146]) : 0u;
            uint t221 = t220;
            if (exec) remu_v_12 = t221;
            uint t222 = !t217 && remu_range_valid(t219 + 4u, 4u, remu_initial_s[146]) ? remu_load_u32(b4, t219 + 4u, remu_initial_s[146]) : 0u;
            uint t223 = t222;
            if (exec) remu_v_13 = t223;
            uint t224 = !t218 && remu_range_valid(t219 + 8u, 4u, remu_initial_s[146]) ? remu_load_u32(b4, t219 + 8u, remu_initial_s[146]) : 0u;
            uint t225 = t224;
            if (exec) remu_v_14 = t225;
            ulong t226 = (ulong)remu_s_8 | (((ulong)remu_s_9 & 0xFFFFul) << 32);
            uint t227 = (uint)(t226 & 3ul);
            uint t228 = (remu_s_9 >> 16) & 0x3FFFu;
            bool t229 = (remu_s_9 & 0x80000000u) != 0u;
            bool t230 = (remu_s_11 & 0x00800000u) != 0u;
            uint t231 = remu_v_4 + (t230 ? (remu_lane & 63u) : 0u);
            uint t232 = 0u + 0u;
            uint t233 = 8u << ((remu_s_11 >> 21) & 3u);
            uint t234 = t231 / t233;
            uint t235 = t231 % t233;
            uint t236 = t232 / 4u;
            uint t237 = t232 % 4u;
            uint t238 = t231 * t228 + t232;
            uint t239 = (t234 * t228 + t236 * 4u) * t233 + t235 * 4u + t237;
            uint t240 = t229 ? t239 : t238;
            uint t241 = t227 + 0u + t240;
            uint t242 = (remu_s_11 >> 28) & 3u;
            uint t243 = remu_s_10;
            bool t244 = (remu_s_8 | remu_s_9 | remu_s_10 | remu_s_11) == 0u;
            uint t245 = t243 > 0u ? t243 - 0u : 0u;
            bool t246 = t244 || (t242 == 0u ? (t231 >= t243 || t232 >= t228) : t242 == 1u ? (t231 >= t243) : t242 == 2u ? (t243 == 0u) : (t229 && t228 != 0u ? (t231 >= t243 || !remu_range_valid(t232, 4u, t228)) : !remu_range_valid(t240, 4u, t245)));
            bool t247 = t244 || (t242 == 0u ? (t231 >= t243 || t232 >= t228) : t242 == 1u ? (t231 >= t243) : t242 == 2u ? (t243 == 0u) : (t229 && t228 != 0u ? (t231 >= t243 || !remu_range_valid(t232, 4u, t228)) : !remu_range_valid(t240, 4u, t245)));
            bool t248 = t244 || (t242 == 0u ? (t231 >= t243 || t232 + 4u >= t228) : t242 == 1u ? (t231 >= t243) : t242 == 2u ? (t243 == 0u) : (t229 && t228 != 0u ? (t231 >= t243 || !remu_range_valid(t232 + 4u, 4u, t228)) : !remu_range_valid(t240 + 4u, 4u, t245)));
            bool t249 = t244 || (t242 == 0u ? (t231 >= t243 || t232 + 8u >= t228) : t242 == 1u ? (t231 >= t243) : t242 == 2u ? (t243 == 0u) : (t229 && t228 != 0u ? (t231 >= t243 || !remu_range_valid(t232 + 8u, 4u, t228)) : !remu_range_valid(t240 + 8u, 4u, t245)));
            uint t250 = t241 & ~3u;
            uint t251 = !t247 && remu_range_valid(t250, 4u, remu_initial_s[146]) ? remu_load_u32(b4, t250, remu_initial_s[146]) : 0u;
            uint t252 = t251;
            if (exec) remu_v_9 = t252;
            uint t253 = !t248 && remu_range_valid(t250 + 4u, 4u, remu_initial_s[146]) ? remu_load_u32(b4, t250 + 4u, remu_initial_s[146]) : 0u;
            uint t254 = t253;
            if (exec) remu_v_10 = t254;
            uint t255 = !t249 && remu_range_valid(t250 + 8u, 4u, remu_initial_s[146]) ? remu_load_u32(b4, t250 + 8u, remu_initial_s[146]) : 0u;
            uint t256 = t255;
            if (exec) remu_v_11 = t256;
            ulong t257 = (ulong)remu_s_8 | (((ulong)remu_s_9 & 0xFFFFul) << 32);
            uint t258 = (uint)(t257 & 3ul);
            uint t259 = (remu_s_9 >> 16) & 0x3FFFu;
            bool t260 = (remu_s_9 & 0x80000000u) != 0u;
            bool t261 = (remu_s_11 & 0x00800000u) != 0u;
            uint t262 = remu_v_3 + (t261 ? (remu_lane & 63u) : 0u);
            uint t263 = 0u + 0u;
            uint t264 = 8u << ((remu_s_11 >> 21) & 3u);
            uint t265 = t262 / t264;
            uint t266 = t262 % t264;
            uint t267 = t263 / 4u;
            uint t268 = t263 % 4u;
            uint t269 = t262 * t259 + t263;
            uint t270 = (t265 * t259 + t267 * 4u) * t264 + t266 * 4u + t268;
            uint t271 = t260 ? t270 : t269;
            uint t272 = t258 + 0u + t271;
            uint t273 = (remu_s_11 >> 28) & 3u;
            uint t274 = remu_s_10;
            bool t275 = (remu_s_8 | remu_s_9 | remu_s_10 | remu_s_11) == 0u;
            uint t276 = t274 > 0u ? t274 - 0u : 0u;
            bool t277 = t275 || (t273 == 0u ? (t262 >= t274 || t263 >= t259) : t273 == 1u ? (t262 >= t274) : t273 == 2u ? (t274 == 0u) : (t260 && t259 != 0u ? (t262 >= t274 || !remu_range_valid(t263, 4u, t259)) : !remu_range_valid(t271, 4u, t276)));
            bool t278 = t275 || (t273 == 0u ? (t262 >= t274 || t263 >= t259) : t273 == 1u ? (t262 >= t274) : t273 == 2u ? (t274 == 0u) : (t260 && t259 != 0u ? (t262 >= t274 || !remu_range_valid(t263, 4u, t259)) : !remu_range_valid(t271, 4u, t276)));
            bool t279 = t275 || (t273 == 0u ? (t262 >= t274 || t263 + 4u >= t259) : t273 == 1u ? (t262 >= t274) : t273 == 2u ? (t274 == 0u) : (t260 && t259 != 0u ? (t262 >= t274 || !remu_range_valid(t263 + 4u, 4u, t259)) : !remu_range_valid(t271 + 4u, 4u, t276)));
            bool t280 = t275 || (t273 == 0u ? (t262 >= t274 || t263 + 8u >= t259) : t273 == 1u ? (t262 >= t274) : t273 == 2u ? (t274 == 0u) : (t260 && t259 != 0u ? (t262 >= t274 || !remu_range_valid(t263 + 8u, 4u, t259)) : !remu_range_valid(t271 + 8u, 4u, t276)));
            uint t281 = t272 & ~3u;
            uint t282 = !t278 && remu_range_valid(t281, 4u, remu_initial_s[146]) ? remu_load_u32(b4, t281, remu_initial_s[146]) : 0u;
            uint t283 = t282;
            if (exec) remu_v_3 = t283;
            uint t284 = !t279 && remu_range_valid(t281 + 4u, 4u, remu_initial_s[146]) ? remu_load_u32(b4, t281 + 4u, remu_initial_s[146]) : 0u;
            uint t285 = t284;
            if (exec) remu_v_4 = t285;
            uint t286 = !t280 && remu_range_valid(t281 + 8u, 4u, remu_initial_s[146]) ? remu_load_u32(b4, t281 + 8u, remu_initial_s[146]) : 0u;
            uint t287 = t286;
            if (exec) remu_v_5 = t287;
            ulong t288 = (ulong)remu_s_8 | (((ulong)remu_s_9 & 0xFFFFul) << 32);
            uint t289 = (uint)(t288 & 3ul);
            uint t290 = (remu_s_9 >> 16) & 0x3FFFu;
            bool t291 = (remu_s_9 & 0x80000000u) != 0u;
            bool t292 = (remu_s_11 & 0x00800000u) != 0u;
            uint t293 = remu_v_6 + (t292 ? (remu_lane & 63u) : 0u);
            uint t294 = 0u + 0u;
            uint t295 = 8u << ((remu_s_11 >> 21) & 3u);
            uint t296 = t293 / t295;
            uint t297 = t293 % t295;
            uint t298 = t294 / 4u;
            uint t299 = t294 % 4u;
            uint t300 = t293 * t290 + t294;
            uint t301 = (t296 * t290 + t298 * 4u) * t295 + t297 * 4u + t299;
            uint t302 = t291 ? t301 : t300;
            uint t303 = t289 + 0u + t302;
            uint t304 = (remu_s_11 >> 28) & 3u;
            uint t305 = remu_s_10;
            bool t306 = (remu_s_8 | remu_s_9 | remu_s_10 | remu_s_11) == 0u;
            uint t307 = t305 > 0u ? t305 - 0u : 0u;
            bool t308 = t306 || (t304 == 0u ? (t293 >= t305 || t294 >= t290) : t304 == 1u ? (t293 >= t305) : t304 == 2u ? (t305 == 0u) : (t291 && t290 != 0u ? (t293 >= t305 || !remu_range_valid(t294, 4u, t290)) : !remu_range_valid(t302, 4u, t307)));
            bool t309 = t306 || (t304 == 0u ? (t293 >= t305 || t294 >= t290) : t304 == 1u ? (t293 >= t305) : t304 == 2u ? (t305 == 0u) : (t291 && t290 != 0u ? (t293 >= t305 || !remu_range_valid(t294, 4u, t290)) : !remu_range_valid(t302, 4u, t307)));
            bool t310 = t306 || (t304 == 0u ? (t293 >= t305 || t294 + 4u >= t290) : t304 == 1u ? (t293 >= t305) : t304 == 2u ? (t305 == 0u) : (t291 && t290 != 0u ? (t293 >= t305 || !remu_range_valid(t294 + 4u, 4u, t290)) : !remu_range_valid(t302 + 4u, 4u, t307)));
            bool t311 = t306 || (t304 == 0u ? (t293 >= t305 || t294 + 8u >= t290) : t304 == 1u ? (t293 >= t305) : t304 == 2u ? (t305 == 0u) : (t291 && t290 != 0u ? (t293 >= t305 || !remu_range_valid(t294 + 8u, 4u, t290)) : !remu_range_valid(t302 + 8u, 4u, t307)));
            uint t312 = t303 & ~3u;
            uint t313 = !t309 && remu_range_valid(t312, 4u, remu_initial_s[146]) ? remu_load_u32(b4, t312, remu_initial_s[146]) : 0u;
            uint t314 = t313;
            if (exec) remu_v_6 = t314;
            uint t315 = !t310 && remu_range_valid(t312 + 4u, 4u, remu_initial_s[146]) ? remu_load_u32(b4, t312 + 4u, remu_initial_s[146]) : 0u;
            uint t316 = t315;
            if (exec) remu_v_7 = t316;
            uint t317 = !t311 && remu_range_valid(t312 + 8u, 4u, remu_initial_s[146]) ? remu_load_u32(b4, t312 + 8u, remu_initial_s[146]) : 0u;
            uint t318 = t317;
            if (exec) remu_v_8 = t318;
            if (exec) remu_v_13 = as_type<uint>((as_type<float>(remu_v_13) * as_type<float>(remu_v_0)));
            if (exec) remu_v_7 = as_type<uint>((as_type<float>(remu_s_1) + as_type<float>(remu_v_7)));
            if (exec) remu_v_14 = as_type<uint>((as_type<float>(remu_v_14) * as_type<float>(remu_v_0)));
            if (exec) remu_v_8 = as_type<uint>((as_type<float>(remu_s_2) + as_type<float>(remu_v_8)));
            if (exec) remu_v_0 = as_type<uint>((as_type<float>(remu_v_12) * as_type<float>(remu_v_0)));
            if (exec) remu_v_13 = as_type<uint>((fma(as_type<float>(remu_v_1), as_type<float>(remu_v_10), as_type<float>(remu_v_13))));
            if (exec) remu_v_14 = as_type<uint>((fma(as_type<float>(remu_v_1), as_type<float>(remu_v_11), as_type<float>(remu_v_14))));
            if (exec) remu_v_0 = as_type<uint>((fma(as_type<float>(remu_v_1), as_type<float>(remu_v_9), as_type<float>(remu_v_0))));
            if (exec) remu_v_13 = as_type<uint>((fma(as_type<float>(remu_v_2), as_type<float>(remu_v_4), as_type<float>(remu_v_13))));
            if (exec) remu_v_14 = as_type<uint>((fma(as_type<float>(remu_v_2), as_type<float>(remu_v_5), as_type<float>(remu_v_14))));
            if (exec) remu_v_5 = as_type<uint>((as_type<float>(remu_s_0) + as_type<float>(remu_v_6)));
            if (exec) remu_v_0 = as_type<uint>((fma(as_type<float>(remu_v_2), as_type<float>(remu_v_3), as_type<float>(remu_v_0))));
            if (exec) remu_v_2 = as_type<uint>((as_type<float>(remu_v_13) + as_type<float>(remu_v_7)));
            if (exec) remu_v_3 = as_type<uint>((as_type<float>(remu_v_14) + as_type<float>(remu_v_8)));
            if (exec) remu_v_0 = as_type<uint>((as_type<float>(remu_v_0) + as_type<float>(remu_v_5)));
            if (exec) remu_v_8 = as_type<uint>((fma(as_type<float>(remu_s_5), as_type<float>(remu_v_2), as_type<float>(remu_s_7))));
            if (exec) remu_v_5 = as_type<uint>((fma(as_type<float>(remu_s_28), as_type<float>(remu_v_3), as_type<float>(remu_s_32))));
            if (exec) remu_v_6 = as_type<uint>((fma(as_type<float>(remu_s_31), as_type<float>(remu_v_3), as_type<float>(remu_s_35))));
            if (exec) remu_v_7 = as_type<uint>((fma(as_type<float>(remu_s_30), as_type<float>(remu_v_3), as_type<float>(remu_s_34))));
            if (exec) remu_v_8 = as_type<uint>((fma(as_type<float>(remu_s_5), -(as_type<float>(remu_s_1)), as_type<float>(remu_v_8))));
            if (exec) remu_v_4 = as_type<uint>((fma(as_type<float>(remu_s_29), as_type<float>(remu_v_3), as_type<float>(remu_s_33))));
            if (exec) remu_v_1 = as_type<uint>((as_type<float>(remu_v_0) - as_type<float>(remu_s_0)));
            if (exec) remu_v_9 = as_type<uint>((fma(as_type<float>(remu_s_24), as_type<float>(remu_v_2), as_type<float>(remu_v_5))));
            if (exec) remu_v_5 = as_type<uint>((fma(as_type<float>(remu_s_27), as_type<float>(remu_v_2), as_type<float>(remu_v_6))));
            if (exec) remu_v_8 = as_type<uint>((fma(as_type<float>(remu_s_6), as_type<float>(remu_v_3), as_type<float>(remu_v_8))));
            if (exec) remu_v_6 = as_type<uint>((fma(as_type<float>(remu_s_26), as_type<float>(remu_v_2), as_type<float>(remu_v_7))));
            if (exec) remu_v_2 = as_type<uint>((fma(as_type<float>(remu_s_25), as_type<float>(remu_v_2), as_type<float>(remu_v_4))));
            if (exec) remu_v_9 = as_type<uint>((fma(as_type<float>(remu_s_20), as_type<float>(remu_v_0), as_type<float>(remu_v_9))));
            if (exec) remu_v_5 = as_type<uint>((fma(as_type<float>(remu_s_23), as_type<float>(remu_v_0), as_type<float>(remu_v_5))));
            if (exec) remu_v_8 = as_type<uint>((fma(as_type<float>(remu_s_6), -(as_type<float>(remu_s_2)), as_type<float>(remu_v_8))));
            if (exec) remu_v_6 = as_type<uint>((fma(as_type<float>(remu_s_22), as_type<float>(remu_v_0), as_type<float>(remu_v_6))));
            if (exec) remu_v_2 = as_type<uint>((fma(as_type<float>(remu_s_21), as_type<float>(remu_v_0), as_type<float>(remu_v_2))));
            if (exec) remu_v_8 = as_type<uint>((fma(as_type<float>(remu_s_4), as_type<float>(remu_v_1), as_type<float>(remu_v_8))));
            if (exec) remu_out.position = float4(as_type<float>(remu_v_9), as_type<float>(remu_v_2), as_type<float>(remu_v_6), as_type<float>(remu_v_5));
            if (exec) remu_out.clip_distances[0] = as_type<float>(remu_v_8);
    return remu_out;
}
