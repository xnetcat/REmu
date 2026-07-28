// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use crate::{
    low_u32, process_guest_range_is_readable, read_guest_u16, read_guest_u32, sce_kernel_error,
    write_guest_bytes,
};

use super::{
    NGS2_ERROR_INVALID_NUM_CHANNELS, NGS2_ERROR_INVALID_OUT_ADDRESS,
    NGS2_ERROR_INVALID_WAVEFORM_ADDRESS, NGS2_ERROR_INVALID_WAVEFORM_DATA,
    NGS2_ERROR_INVALID_WAVEFORM_FORMAT, NGS2_ERROR_INVALID_WAVEFORM_TYPE,
    NGS2_ERROR_UNKNOWN_WAVEFORM_FORMAT, NGS2_MAX_SAMPLE_RATE, NGS2_MAX_VOICE_CHANNELS,
    guest_object_is_reachable,
};

/// Bytes one `SceNgs2WaveformBlock` occupies.
pub(super) const NGS2_WAVEFORM_BLOCK_SIZE: usize = 40;
/// Bytes one `SceNgs2WaveformInfo` occupies, and one `SceNgs2WaveformFormat` inside it.
pub(super) const NGS2_WAVEFORM_INFO_SIZE: usize = 232;
pub(super) const NGS2_WAVEFORM_FORMAT_SIZE: usize = 24;
/// Byte offset of the first block inside a waveform information block.
pub(super) const NGS2_WAVEFORM_INFO_BLOCK_OFFSET: usize = 72;

/// Waveform types this build can address by sample position.
pub(super) const NGS2_WAVEFORM_TYPE_PCM_I8: u32 = 0x10;
pub(super) const NGS2_WAVEFORM_TYPE_PCM_U8: u32 = 0x11;
pub(super) const NGS2_WAVEFORM_TYPE_PCM_I16L: u32 = 0x12;
pub(super) const NGS2_WAVEFORM_TYPE_PCM_I16B: u32 = 0x13;
pub(super) const NGS2_WAVEFORM_TYPE_PCM_I24L: u32 = 0x14;
pub(super) const NGS2_WAVEFORM_TYPE_PCM_I24B: u32 = 0x15;
pub(super) const NGS2_WAVEFORM_TYPE_PCM_I32L: u32 = 0x16;
pub(super) const NGS2_WAVEFORM_TYPE_PCM_I32B: u32 = 0x17;
pub(super) const NGS2_WAVEFORM_TYPE_PCM_F32L: u32 = 0x18;
pub(super) const NGS2_WAVEFORM_TYPE_PCM_F32B: u32 = 0x19;
pub(super) const NGS2_WAVEFORM_TYPE_PCM_F64L: u32 = 0x1a;
pub(super) const NGS2_WAVEFORM_TYPE_PCM_F64B: u32 = 0x1b;
pub(super) const NGS2_WAVEFORM_TYPE_VAG: u32 = 0x1c;
/// Samples one compressed `VAG` unit carries, and the bytes it occupies per channel.
pub(super) const NGS2_VAG_UNIT_SAMPLES: u32 = 28;
pub(super) const NGS2_VAG_UNIT_BYTES: u32 = 16;

/// Returns the bytes and samples one audio unit of a sample-addressed waveform carries.
///
/// Codec waveforms carry their frame geometry in the format's configuration word, which this
/// build does not decode; they are refused rather than answered with an offset into the wrong
/// frame.
pub(super) fn ngs2_audio_unit(waveform_type: u32, channels: u32) -> Option<(u32, u32)> {
    let bytes_per_channel = match waveform_type {
        NGS2_WAVEFORM_TYPE_PCM_I8 | NGS2_WAVEFORM_TYPE_PCM_U8 => 1,
        NGS2_WAVEFORM_TYPE_PCM_I16L | NGS2_WAVEFORM_TYPE_PCM_I16B => 2,
        NGS2_WAVEFORM_TYPE_PCM_I24L | NGS2_WAVEFORM_TYPE_PCM_I24B => 3,
        NGS2_WAVEFORM_TYPE_PCM_I32L
        | NGS2_WAVEFORM_TYPE_PCM_I32B
        | NGS2_WAVEFORM_TYPE_PCM_F32L
        | NGS2_WAVEFORM_TYPE_PCM_F32B => 4,
        NGS2_WAVEFORM_TYPE_PCM_F64L | NGS2_WAVEFORM_TYPE_PCM_F64B => 8,
        NGS2_WAVEFORM_TYPE_VAG => {
            return Some((
                NGS2_VAG_UNIT_BYTES.checked_mul(channels)?,
                NGS2_VAG_UNIT_SAMPLES,
            ));
        }
        _ => return None,
    };
    Some((bytes_per_channel * channels, 1))
}

/// Serializes one waveform block.
pub(super) fn ngs2_waveform_block_bytes(
    data_offset: u64,
    data_size: u64,
    skip_samples: u32,
    samples: u32,
) -> [u8; NGS2_WAVEFORM_BLOCK_SIZE] {
    let mut block = [0_u8; NGS2_WAVEFORM_BLOCK_SIZE];
    block[..8].copy_from_slice(&data_offset.to_le_bytes());
    block[8..16].copy_from_slice(&data_size.to_le_bytes());
    block[16..20].copy_from_slice(&1_u32.to_le_bytes());
    block[20..24].copy_from_slice(&skip_samples.to_le_bytes());
    block[24..28].copy_from_slice(&samples.to_le_bytes());
    block
}

/// Converts a sample range into the block that addresses it.
pub(super) fn ngs2_calc_waveform_block(arguments: [u64; 6]) -> u64 {
    let (format, sample_position, samples, out_block) = (
        arguments[0],
        low_u32(arguments[1]),
        low_u32(arguments[2]),
        arguments[3],
    );
    if !guest_object_is_reachable(out_block, NGS2_WAVEFORM_BLOCK_SIZE as u64) {
        return sce_kernel_error(NGS2_ERROR_INVALID_OUT_ADDRESS);
    }
    if format == 0 || !process_guest_range_is_readable(format, NGS2_WAVEFORM_FORMAT_SIZE) {
        return sce_kernel_error(NGS2_ERROR_INVALID_WAVEFORM_ADDRESS);
    }
    let waveform_type = read_guest_u32(format);
    let channels = read_guest_u32(format + 4);
    if channels == 0 || channels > NGS2_MAX_VOICE_CHANNELS {
        return sce_kernel_error(NGS2_ERROR_INVALID_NUM_CHANNELS);
    }
    let Some((unit_size, unit_samples)) = ngs2_audio_unit(waveform_type, channels) else {
        return sce_kernel_error(NGS2_ERROR_INVALID_WAVEFORM_TYPE);
    };
    let unit_index = u64::from(sample_position / unit_samples);
    let skip_samples = sample_position % unit_samples;
    let units = u64::from(skip_samples)
        .saturating_add(u64::from(samples))
        .div_ceil(u64::from(unit_samples));
    let block = ngs2_waveform_block_bytes(
        unit_index.saturating_mul(u64::from(unit_size)),
        units.saturating_mul(u64::from(unit_size)),
        skip_samples,
        samples,
    );
    write_guest_bytes(out_block, &block);
    0
}

/// Chunk identifiers of the container this build recognizes, little-endian as they appear.
pub(super) const RIFF_MAGIC: u32 = 0x4646_4952;
pub(super) const RIFF_WAVE_MAGIC: u32 = 0x4556_4157;
pub(super) const RIFF_FORMAT_CHUNK: u32 = 0x2074_6d66;
pub(super) const RIFF_DATA_CHUNK: u32 = 0x6174_6164;
/// Bytes the container's leading identifiers occupy.
pub(super) const RIFF_HEADER_SIZE: u64 = 12;
/// Chunks the walk visits before the container is treated as malformed.
pub(super) const RIFF_CHUNK_LIMIT: usize = 64;
/// Sample encodings a format chunk may name.
pub(super) const RIFF_FORMAT_PCM: u16 = 1;
pub(super) const RIFF_FORMAT_FLOAT: u16 = 3;
pub(super) const RIFF_FORMAT_EXTENSIBLE: u16 = 0xfffe;

/// One recognized container, reduced to what a waveform information block needs.
#[derive(Clone, Copy, Debug)]
pub(super) struct RiffWaveform {
    pub(super) waveform_type: u32,
    pub(super) channels: u32,
    pub(super) sample_rate: u32,
    pub(super) unit_size: u32,
    pub(super) data_offset: u32,
    pub(super) data_size: u32,
}

/// Reads a bounded word from a guest blob, or `None` if it lies outside the blob.
pub(super) fn blob_u32(data: u64, size: u64, offset: u64) -> Option<u32> {
    let end = offset.checked_add(4)?;
    let address = data.checked_add(offset)?;
    (end <= size && process_guest_range_is_readable(address, 4)).then(|| read_guest_u32(address))
}

/// Reads a bounded half-word from a guest blob, or `None` if it lies outside the blob.
pub(super) fn blob_u16(data: u64, size: u64, offset: u64) -> Option<u16> {
    let end = offset.checked_add(2)?;
    let address = data.checked_add(offset)?;
    (end <= size && process_guest_range_is_readable(address, 2)).then(|| read_guest_u16(address))
}

/// Returns the waveform type a sample encoding and width name.
pub(super) fn riff_waveform_type(format_tag: u16, bits: u16) -> Option<u32> {
    match (format_tag, bits) {
        (RIFF_FORMAT_PCM, 8) => Some(NGS2_WAVEFORM_TYPE_PCM_U8),
        (RIFF_FORMAT_PCM, 16) => Some(NGS2_WAVEFORM_TYPE_PCM_I16L),
        (RIFF_FORMAT_PCM, 24) => Some(NGS2_WAVEFORM_TYPE_PCM_I24L),
        (RIFF_FORMAT_PCM, 32) => Some(NGS2_WAVEFORM_TYPE_PCM_I32L),
        (RIFF_FORMAT_FLOAT, 32) => Some(NGS2_WAVEFORM_TYPE_PCM_F32L),
        (RIFF_FORMAT_FLOAT, 64) => Some(NGS2_WAVEFORM_TYPE_PCM_F64L),
        _ => None,
    }
}

/// Reads the format chunk at `offset`, whose payload runs for `length` bytes.
pub(super) fn riff_format_chunk(
    data: u64,
    size: u64,
    offset: u64,
    length: u64,
) -> Result<(u32, u32, u32, u32), u32> {
    const EXTENSIBLE_SUBFORMAT_OFFSET: u64 = 24;
    const MINIMUM_FORMAT_CHUNK: u64 = 16;

    if length < MINIMUM_FORMAT_CHUNK {
        return Err(NGS2_ERROR_INVALID_WAVEFORM_FORMAT);
    }
    let (Some(tag), Some(channels), Some(rate), Some(bits)) = (
        blob_u16(data, size, offset),
        blob_u16(data, size, offset + 2),
        blob_u32(data, size, offset + 4),
        blob_u16(data, size, offset + 14),
    ) else {
        return Err(NGS2_ERROR_INVALID_WAVEFORM_DATA);
    };
    let tag = if tag == RIFF_FORMAT_EXTENSIBLE {
        blob_u16(data, size, offset + EXTENSIBLE_SUBFORMAT_OFFSET)
            .ok_or(NGS2_ERROR_INVALID_WAVEFORM_DATA)?
    } else {
        tag
    };
    let waveform_type = riff_waveform_type(tag, bits).ok_or(NGS2_ERROR_UNKNOWN_WAVEFORM_FORMAT)?;
    let channels = u32::from(channels);
    if channels == 0
        || channels > NGS2_MAX_VOICE_CHANNELS
        || rate == 0
        || rate > NGS2_MAX_SAMPLE_RATE
    {
        return Err(NGS2_ERROR_INVALID_WAVEFORM_FORMAT);
    }
    let unit_size = u32::from(bits / 8) * channels;
    if unit_size == 0 {
        return Err(NGS2_ERROR_INVALID_WAVEFORM_FORMAT);
    }
    Ok((waveform_type, channels, rate, unit_size))
}

/// Recognizes a container and reduces it to the fields a waveform information block reports.
pub(super) fn riff_parse_waveform(data: u64, size: u64) -> Result<RiffWaveform, u32> {
    if blob_u32(data, size, 0) != Some(RIFF_MAGIC)
        || blob_u32(data, size, 8) != Some(RIFF_WAVE_MAGIC)
    {
        return Err(NGS2_ERROR_UNKNOWN_WAVEFORM_FORMAT);
    }
    let mut format = None;
    let mut payload = None;
    let mut offset = RIFF_HEADER_SIZE;
    for _ in 0..RIFF_CHUNK_LIMIT {
        let (Some(id), Some(length)) = (
            blob_u32(data, size, offset),
            blob_u32(data, size, offset + 4),
        ) else {
            break;
        };
        let body = offset + 8;
        let length = u64::from(length);
        match id {
            RIFF_FORMAT_CHUNK => format = Some(riff_format_chunk(data, size, body, length)?),
            RIFF_DATA_CHUNK => payload = Some((body, length.min(size.saturating_sub(body)))),
            _ => {}
        }
        offset = body.saturating_add(length).saturating_add(length & 1);
        if offset >= size {
            break;
        }
    }
    let (Some((waveform_type, channels, sample_rate, unit_size)), Some((body, length))) =
        (format, payload)
    else {
        return Err(NGS2_ERROR_INVALID_WAVEFORM_DATA);
    };
    let (Ok(data_offset), Ok(data_size)) = (u32::try_from(body), u32::try_from(length)) else {
        return Err(NGS2_ERROR_INVALID_WAVEFORM_DATA);
    };
    Ok(RiffWaveform {
        waveform_type,
        channels,
        sample_rate,
        unit_size,
        data_offset,
        data_size,
    })
}

/// Serializes a recognized container into a waveform information block.
pub(super) fn ngs2_waveform_info_bytes(waveform: RiffWaveform) -> [u8; NGS2_WAVEFORM_INFO_SIZE] {
    let samples = waveform.data_size / waveform.unit_size;
    let mut info = [0_u8; NGS2_WAVEFORM_INFO_SIZE];
    let mut field = |offset: usize, value: u32| {
        info[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    };
    field(0, waveform.waveform_type);
    field(4, waveform.channels);
    field(8, waveform.sample_rate);
    field(24, waveform.data_offset);
    field(28, waveform.data_size);
    field(40, samples);
    field(44, waveform.unit_size);
    field(48, 1);
    field(52, 1);
    field(56, waveform.unit_size);
    field(60, 1);
    field(68, 1);
    let block = ngs2_waveform_block_bytes(0, u64::from(waveform.data_size), 0, samples);
    info[NGS2_WAVEFORM_INFO_BLOCK_OFFSET..NGS2_WAVEFORM_INFO_BLOCK_OFFSET + block.len()]
        .copy_from_slice(&block);
    info
}

/// Recognizes an in-memory waveform container and reports what it holds.
pub(super) fn ngs2_parse_waveform_data(arguments: [u64; 6]) -> u64 {
    let (data, size, out_info) = (arguments[0], arguments[1], arguments[2]);
    if !guest_object_is_reachable(out_info, NGS2_WAVEFORM_INFO_SIZE as u64) {
        return sce_kernel_error(NGS2_ERROR_INVALID_OUT_ADDRESS);
    }
    if data == 0 || size < RIFF_HEADER_SIZE {
        return sce_kernel_error(NGS2_ERROR_INVALID_WAVEFORM_DATA);
    }
    match riff_parse_waveform(data, size) {
        Ok(waveform) => {
            write_guest_bytes(out_info, &ngs2_waveform_info_bytes(waveform));
            0
        }
        Err(error) => sce_kernel_error(error),
    }
}
