// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use crate::ajm_engine::SCE_AJM_ERROR_INVALID_PARAMETER;
use crate::{
    process_guest_range_is_readable, read_guest_bytes, sce_kernel_error, write_guest_bytes,
};

use super::guest_object_is_reachable;

/// Bytes one `SceAjmDecMp3ParseFrame` occupies.
pub(super) const MP3_PARSE_FRAME_SIZE: usize = 40;
/// Bytes of a caller's bitstream this build searches for a frame header.
pub(super) const MP3_SEARCH_WINDOW: u64 = 8192;
/// Bitrates, in kilobits per second, indexed by the header's bitrate field.
pub(super) const MP3_BITRATES_VERSION1: [u32; 16] = [
    0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0,
];
pub(super) const MP3_BITRATES_VERSION2: [u32; 16] = [
    0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0,
];
/// Sampling rates of the first version, which the later ones halve and quarter.
pub(super) const MP3_SAMPLE_RATES: [u32; 3] = [44_100, 48_000, 32_000];
/// Samples one frame carries in each version.
pub(super) const MP3_VERSION1_FRAME_SAMPLES: u32 = 1152;
pub(super) const MP3_VERSION2_FRAME_SAMPLES: u32 = 576;
/// Layer field value that selects layer III; this entry point parses no other layer.
pub(super) const MP3_LAYER_III: u8 = 1;
/// Channel mode that selects a single channel.
pub(super) const MP3_MODE_MONO: u8 = 3;
/// Original-file-length header kinds a frame may carry.
pub(super) const MP3_OFL_TYPE_NONE: u32 = 0;
pub(super) const MP3_OFL_TYPE_LAME: u32 = 1;
pub(super) const MP3_OFL_TYPE_VBRI: u32 = 2;
/// Byte offset of a `VBRI` header from the start of the frame.
pub(super) const MP3_VBRI_OFFSET: usize = 36;

/// One parsed frame header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Mp3Frame {
    pub(super) size: u32,
    pub(super) channels: u32,
    pub(super) samples: u32,
    pub(super) bitrate: u32,
    pub(super) sample_rate: u32,
    pub(super) side_info: usize,
}

/// Decodes a frame header, or returns `None` if the four bytes do not open a layer III frame.
pub(super) fn mp3_parse_header(header: &[u8]) -> Option<Mp3Frame> {
    const SYNC_MASK: u8 = 0xe0;
    const VERSION_RESERVED: u8 = 1;
    const VERSION_1: u8 = 3;
    const VERSION_2: u8 = 2;

    if header.len() < 4 || header[0] != 0xff || header[1] & SYNC_MASK != SYNC_MASK {
        return None;
    }
    let version = (header[1] >> 3) & 0x3;
    let layer = (header[1] >> 1) & 0x3;
    if version == VERSION_RESERVED || layer != MP3_LAYER_III {
        return None;
    }
    let bitrate_index = usize::from((header[2] >> 4) & 0xf);
    let rate_index = usize::from((header[2] >> 2) & 0x3);
    let padding = u32::from((header[2] >> 1) & 0x1);
    let mono = (header[3] >> 6) & 0x3 == MP3_MODE_MONO;
    if rate_index >= MP3_SAMPLE_RATES.len() {
        return None;
    }
    let first_version = version == VERSION_1;
    let bitrate = if first_version {
        MP3_BITRATES_VERSION1[bitrate_index]
    } else {
        MP3_BITRATES_VERSION2[bitrate_index]
    };
    if bitrate == 0 {
        return None;
    }
    let sample_rate = MP3_SAMPLE_RATES[rate_index]
        / match version {
            VERSION_1 => 1,
            VERSION_2 => 2,
            _ => 4,
        };
    let samples = if first_version {
        MP3_VERSION1_FRAME_SAMPLES
    } else {
        MP3_VERSION2_FRAME_SAMPLES
    };
    let bitrate = bitrate * 1000;
    let side_info = match (first_version, mono) {
        (true, false) => 32,
        (true, true) | (false, false) => 17,
        (false, true) => 9,
    };
    Some(Mp3Frame {
        size: samples / 8 * bitrate / sample_rate + padding,
        channels: if mono { 1 } else { 2 },
        samples,
        bitrate,
        sample_rate,
        side_info,
    })
}

/// Original-file-length data recovered from a frame's ancillary bytes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct Mp3OriginalFileLength {
    pub(super) kind: u32,
    pub(super) encoder_delay: u32,
    pub(super) frames: u32,
}

/// Reads a big-endian word from a frame at `offset`, if the frame carries one there.
pub(super) fn mp3_be_u32(frame: &[u8], offset: usize) -> Option<u32> {
    let bytes = frame.get(offset..offset.checked_add(4)?)?;
    Some(u32::from_be_bytes(bytes.try_into().ok()?))
}

/// Recovers the original-file-length header a frame carries, if any.
pub(super) fn mp3_parse_original_file_length(
    frame: &[u8],
    header: Mp3Frame,
) -> Mp3OriginalFileLength {
    const XING_FRAMES_FLAG: u32 = 1 << 0;
    const XING_BYTES_FLAG: u32 = 1 << 1;
    const XING_TOC_FLAG: u32 = 1 << 2;
    const XING_QUALITY_FLAG: u32 = 1 << 3;
    const XING_TOC_SIZE: usize = 100;
    /// Distance from the encoder tag's magic to the twelve-bit encoder delay.
    const LAME_DELAY_OFFSET: usize = 21;
    const VBRI_DELAY_OFFSET: usize = 6;
    const VBRI_FRAMES_OFFSET: usize = 14;

    let xing = 4 + header.side_info;
    if matches!(frame.get(xing..xing + 4), Some(b"Xing" | b"Info")) {
        let flags = mp3_be_u32(frame, xing + 4).unwrap_or(0);
        let mut cursor = xing + 8;
        let mut frames = 0;
        if flags & XING_FRAMES_FLAG != 0 {
            frames = mp3_be_u32(frame, cursor).unwrap_or(0);
            cursor += 4;
        }
        for (flag, length) in [
            (XING_BYTES_FLAG, 4),
            (XING_TOC_FLAG, XING_TOC_SIZE),
            (XING_QUALITY_FLAG, 4),
        ] {
            if flags & flag != 0 {
                cursor += length;
            }
        }
        let encoder_delay = frame
            .get(cursor + LAME_DELAY_OFFSET..cursor + LAME_DELAY_OFFSET + 2)
            .map_or(0, |bytes| {
                (u32::from(bytes[0]) << 4) | u32::from(bytes[1] >> 4)
            });
        return Mp3OriginalFileLength {
            kind: MP3_OFL_TYPE_LAME,
            encoder_delay,
            frames,
        };
    }
    if frame.get(MP3_VBRI_OFFSET..MP3_VBRI_OFFSET + 4) == Some(b"VBRI") {
        return Mp3OriginalFileLength {
            kind: MP3_OFL_TYPE_VBRI,
            encoder_delay: frame
                .get(MP3_VBRI_OFFSET + VBRI_DELAY_OFFSET..MP3_VBRI_OFFSET + VBRI_DELAY_OFFSET + 2)
                .map_or(0, |bytes| {
                    u32::from(u16::from_be_bytes([bytes[0], bytes[1]]))
                }),
            frames: mp3_be_u32(frame, MP3_VBRI_OFFSET + VBRI_FRAMES_OFFSET).unwrap_or(0),
        };
    }
    Mp3OriginalFileLength::default()
}

/// Serializes one parsed frame.
pub(super) fn mp3_frame_bytes(
    header: Mp3Frame,
    original: Mp3OriginalFileLength,
) -> [u8; MP3_PARSE_FRAME_SIZE] {
    let mut parsed = [0_u8; MP3_PARSE_FRAME_SIZE];
    parsed[..8].copy_from_slice(&u64::from(header.size).to_le_bytes());
    parsed[8..12].copy_from_slice(&header.channels.to_le_bytes());
    parsed[12..16].copy_from_slice(&header.samples.to_le_bytes());
    parsed[16..20].copy_from_slice(&header.bitrate.to_le_bytes());
    parsed[20..24].copy_from_slice(&header.sample_rate.to_le_bytes());
    parsed[24..28].copy_from_slice(&original.encoder_delay.to_le_bytes());
    parsed[28..32].copy_from_slice(&original.frames.to_le_bytes());
    parsed[32..36].copy_from_slice(&original.frames.saturating_mul(header.samples).to_le_bytes());
    parsed[36..40].copy_from_slice(&original.kind.to_le_bytes());
    parsed
}

/// Reports the geometry of the first frame a bitstream carries.
pub(super) fn ajm_dec_mp3_parse_frame(arguments: [u64; 6]) -> u64 {
    let (bitstream, length, parse_original, out_frame) =
        (arguments[0], arguments[1], arguments[2], arguments[3]);
    if bitstream == 0
        || length == 0
        || !guest_object_is_reachable(out_frame, MP3_PARSE_FRAME_SIZE as u64)
    {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    }
    let window = usize::try_from(length.min(MP3_SEARCH_WINDOW)).unwrap_or(0);
    if !process_guest_range_is_readable(bitstream, window) {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    }
    let mut bytes = vec![0_u8; window];
    read_guest_bytes(bitstream, &mut bytes);
    let Some((start, header)) = (0..window.saturating_sub(3))
        .find_map(|offset| mp3_parse_header(&bytes[offset..]).map(|header| (offset, header)))
    else {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    };
    let original = if parse_original == 0 {
        Mp3OriginalFileLength {
            kind: MP3_OFL_TYPE_NONE,
            ..Mp3OriginalFileLength::default()
        }
    } else {
        mp3_parse_original_file_length(&bytes[start..], header)
    };
    write_guest_bytes(out_frame, &mp3_frame_bytes(header, original));
    0
}
