//! NID ↔ `u64` conversion for link-time route tables (plan C1 / B13).
//!
//! An 11-character PS5 NID is the first eleven base64 digits of the big-endian encoding of an
//! 8-byte value (with `/` rewritten as `-`). That value is losslessly a `u64`, so dispatch can
//! `match` on integers instead of length-11 string compares.
//!
//! Provenance: `scripts/generate-nid-names.py` `nid_of` /
//! `~/Documents/REmu-research/abi-inventory/sdk10-nid-map.tsv` (`.scenid` u64 parallel to `.dynsym`).

/// Decodes an 11-character NID string into its underlying `u64`.
///
/// Panics if `nid` is not exactly 11 characters of the NID alphabet (`A–Z a–z 0–9 + -`).
#[must_use]
pub const fn decode_nid(nid: &str) -> u64 {
    assert!(nid.len() == 11, "PS5 NIDs are exactly 11 characters");
    let bytes = nid.as_bytes();
    let mut value = 0_u64;
    let mut index = 0;
    while index < 10 {
        let six = nid_digit(bytes[index]);
        value = (value << 6) | (six as u64);
        index += 1;
    }
    // The eleventh character contributes the high 4 of its 6 bits; the trailing 2 bits were the
    // padding nibble dropped when the encoder truncated the 12-character base64 form.
    let last = nid_digit(bytes[10]);
    value = (value << 4) | ((last >> 2) as u64);
    value
}

/// Encodes a NID `u64` back to the 11-character string form (tests / diagnostics).
#[cfg_attr(not(test), allow(dead_code))]
#[must_use]
pub fn encode_nid(value: u64) -> String {
    let mut raw = [0_u8; 12];
    let be = value.to_be_bytes();
    // Standard base64 of 8 bytes → 12 characters including one `=`.
    const TABLE: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut bits: u32 = 0;
    let mut bit_count = 0_u32;
    let mut out = 0_usize;
    for byte in be {
        bits = (bits << 8) | u32::from(byte);
        bit_count += 8;
        while bit_count >= 6 {
            bit_count -= 6;
            raw[out] = TABLE[((bits >> bit_count) & 0x3f) as usize];
            out += 1;
        }
    }
    if bit_count > 0 {
        raw[out] = TABLE[((bits << (6 - bit_count)) & 0x3f) as usize];
        out += 1;
    }
    while out < 12 {
        raw[out] = b'=';
        out += 1;
    }
    let mut encoded = String::with_capacity(11);
    for byte in &raw[..11] {
        encoded.push(if *byte == b'/' {
            '-'
        } else {
            char::from(*byte)
        });
    }
    encoded
}

const fn nid_digit(byte: u8) -> u8 {
    match byte {
        b'A'..=b'Z' => byte - b'A',
        b'a'..=b'z' => byte - b'a' + 26,
        b'0'..=b'9' => byte - b'0' + 52,
        b'+' => 62,
        b'-' | b'/' => 63,
        _ => panic!("NID character outside the PS5 base64 alphabet"),
    }
}

/// Compile-time NID literal → `u64` for const contexts and equality checks.
#[macro_export]
macro_rules! nid {
    ($literal:literal) => {
        $crate::nid_u64::decode_nid($literal)
    };
}

#[cfg(test)]
mod tests {
    use super::{decode_nid, encode_nid};
    use crate::nid_table::known_nid_name;

    #[test]
    fn decode_round_trips_every_hand_catalog_nid() {
        // known_nid_name is a large match; walk a representative set plus pinned AGC NIDs.
        const SAMPLES: &[&str] = &[
            "23LRUSvYu1M", // sceAgcInit (pinned)
            "-KRzWekV120", // sceAgcDcbSetIndexSize (pinned)
            "V++UgBtQhn0", // sceAgcGetDataPacketPayloadAddress (pinned)
            "qj7QZpgr9Uw", // sceAgcDcbContextStateOp (pinned)
            "dolOmWH+huQ", // sceAgcGetFusedShaderSize (pinned)
            "fd5Bp5tGTgo", // sceAgcFuseShaderHalves (pinned)
            "1G3lF1Gg1k8", // sceKernelOpen
            "eV9wAD2riIA", // sceKernelStat
            "aI+OeCz8xrQ", // scePthreadSelf
            "T72hz6ffq08", // scePthreadYield
            "uMei1W9uyNo", // exit
            "4J2sUJmuHZQ", // sceKernelGetProcessTime
        ];
        for nid in SAMPLES {
            assert!(
                known_nid_name(nid).is_some() || matches!(*nid, "23LRUSvYu1M" | "-KRzWekV120"),
                "sample {nid} missing from hand catalog"
            );
            let value = decode_nid(nid);
            assert_eq!(encode_nid(value), *nid, "round-trip failed for {nid}");
        }
    }

    #[test]
    fn decode_matches_python_endianness_for_known_bytes() {
        // value 0x0123456789abcdef → base64 big-endian pack → first 11 chars with /→-
        let value = 0x0123_4567_89ab_cdef;
        let encoded = encode_nid(value);
        assert_eq!(decode_nid(&encoded), value);
        assert_eq!(encoded.len(), 11);
    }

    #[test]
    fn nid_macro_equals_decode() {
        const OPEN: u64 = crate::nid!("1G3lF1Gg1k8");
        assert_eq!(OPEN, decode_nid("1G3lF1Gg1k8"));
    }
}
