// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Memory encoding families: DS, MUBUF/MTBUF, FLAT/GLOBAL and MIMG.

use crate::ir::{
    BufferMemoryControl, GlobalMemoryControl, ImageControl, InstructionControl, Opcode, Operand,
    RawOpcodeFamily, ShaderEncoding,
};

use super::{DecodeError, Frame, known_frame, push_vector_range, sign_extend};

pub(crate) fn decode_ds(pc: u32, word: u32) -> Result<Frame, DecodeError> {
    let code = (word >> 18) & 0xff;
    let name = match code {
        0x00 => Some("DsAddU32"),
        0x01 => Some("DsSubU32"),
        0x03 => Some("DsIncU32"),
        0x04 => Some("DsDecU32"),
        0x05 => Some("DsMinI32"),
        0x06 => Some("DsMaxI32"),
        0x07 => Some("DsMinU32"),
        0x08 => Some("DsMaxU32"),
        0x09 => Some("DsAndB32"),
        0x0a => Some("DsOrB32"),
        0x0b => Some("DsXorB32"),
        0x0d => Some("DsWriteB32"),
        0x0e => Some("DsWrite2B32"),
        0x0f => Some("DsWrite2St64B32"),
        0x10 => Some("DsCmpstB32"),
        0x20 => Some("DsAddRtnU32"),
        0x21 => Some("DsSubRtnU32"),
        0x23 => Some("DsIncRtnU32"),
        0x24 => Some("DsDecRtnU32"),
        0x25 => Some("DsMinRtnI32"),
        0x26 => Some("DsMaxRtnI32"),
        0x27 => Some("DsMinRtnU32"),
        0x28 => Some("DsMaxRtnU32"),
        0x29 => Some("DsAndRtnB32"),
        0x2a => Some("DsOrRtnB32"),
        0x2b => Some("DsXorRtnB32"),
        0x2d => Some("DsWrxchgRtnB32"),
        0x30 => Some("DsCmpstRtnB32"),
        0x35 => Some("DsSwizzleB32"),
        0x36 => Some("DsReadB32"),
        0x37 => Some("DsRead2B32"),
        0x38 => Some("DsRead2St64B32"),
        0x4d => Some("DsWriteB64"),
        // The 64-bit DS block mirrors the 32-bit block at +0x40 (0x0d/0x4d writes, 0x36/0x76
        // reads). Poppy Playtime's glyph-prep compute reads LDS pairs through 0x76.
        0x76 => Some("DsReadB64"),
        0xb0 => Some("DsWriteAddtidB32"),
        0xb1 => Some("DsReadAddtidB32"),
        0xde => Some("DsWriteB96"),
        0xdf => Some("DsWriteB128"),
        0xfe => Some("DsReadB96"),
        0xff => Some("DsReadB128"),
        _ => None,
    };
    known_frame(pc, word, ShaderEncoding::Ds, code, name, 2)
}

pub(crate) fn decode_mtbuf(word: u32, extra: u32) -> Frame {
    let code = ((extra >> 21) & 1) << 3 | ((word >> 16) & 0x7);
    let name = match code {
        0x00 => "TBufferLoadFormatX",
        0x01 => "TBufferLoadFormatXy",
        0x02 => "TBufferLoadFormatXyz",
        0x03 => "TBufferLoadFormatXyzw",
        0x04 => "TBufferStoreFormatX",
        0x05 => "TBufferStoreFormatXy",
        0x06 => "TBufferStoreFormatXyz",
        0x07 => "TBufferStoreFormatXyzw",
        0x08 => "TBufferLoadFormatD16X",
        0x09 => "TBufferLoadFormatD16Xy",
        0x0a => "TBufferLoadFormatD16Xyz",
        0x0b => "TBufferLoadFormatD16Xyzw",
        0x0c => "TBufferStoreFormatD16X",
        0x0d => "TBufferStoreFormatD16Xy",
        0x0e => "TBufferStoreFormatD16Xyz",
        0x0f => "TBufferStoreFormatD16Xyzw",
        _ => unreachable!("four-bit opcode"),
    };
    Frame {
        encoding: ShaderEncoding::Mtbuf,
        opcode: Opcode::from_static(name),
        dwords: 2 + usize::from(extra >> 24 == 0xff),
    }
}

pub(crate) fn decode_mubuf(word: u32, extra: u32) -> Frame {
    let code = (word >> 18) & 0xff;
    let name = match code {
        0x00 => Some("BufferLoadFormatX"),
        0x01 => Some("BufferLoadFormatXy"),
        0x02 => Some("BufferLoadFormatXyz"),
        0x03 => Some("BufferLoadFormatXyzw"),
        0x04 => Some("BufferStoreFormatX"),
        0x05 => Some("BufferStoreFormatXy"),
        0x06 => Some("BufferStoreFormatXyz"),
        0x07 => Some("BufferStoreFormatXyzw"),
        0x08 => Some("BufferLoadUbyte"),
        0x09 => Some("BufferLoadSbyte"),
        0x0a => Some("BufferLoadUshort"),
        0x0b => Some("BufferLoadSshort"),
        0x0c => Some("BufferLoadDword"),
        0x0d => Some("BufferLoadDwordx2"),
        0x0e => Some("BufferLoadDwordx4"),
        0x0f => Some("BufferLoadDwordx3"),
        0x18 => Some("BufferStoreByte"),
        0x19 => Some("BufferStoreByteD16Hi"),
        0x1a => Some("BufferStoreShort"),
        0x1b => Some("BufferStoreShortD16Hi"),
        0x1c => Some("BufferStoreDword"),
        0x1d => Some("BufferStoreDwordx2"),
        0x1e => Some("BufferStoreDwordx4"),
        0x1f => Some("BufferStoreDwordx3"),
        0x20 => Some("BufferLoadUbyteD16"),
        0x21 => Some("BufferLoadUbyteD16Hi"),
        0x22 => Some("BufferLoadSbyteD16"),
        0x23 => Some("BufferLoadSbyteD16Hi"),
        0x24 => Some("BufferLoadShortD16"),
        0x25 => Some("BufferLoadShortD16Hi"),
        0x26 => Some("BufferLoadFormatD16HiX"),
        0x27 => Some("BufferStoreFormatD16HiX"),
        0x30 => Some("BufferAtomicSwap"),
        0x31 => Some("BufferAtomicCmpswap"),
        0x32 => Some("BufferAtomicAdd"),
        0x33 => Some("BufferAtomicSub"),
        0x35 => Some("BufferAtomicSmin"),
        0x36 => Some("BufferAtomicUmin"),
        0x37 => Some("BufferAtomicSmax"),
        0x38 => Some("BufferAtomicUmax"),
        0x39 => Some("BufferAtomicAnd"),
        0x3a => Some("BufferAtomicOr"),
        0x3b => Some("BufferAtomicXor"),
        0x3c => Some("BufferAtomicInc"),
        0x3d => Some("BufferAtomicDec"),
        0x80 => Some("BufferLoadFormatD16X"),
        0x81 => Some("BufferLoadFormatD16Xy"),
        0x82 => Some("BufferLoadFormatD16Xyz"),
        0x83 => Some("BufferLoadFormatD16Xyzw"),
        0x84 => Some("BufferStoreFormatD16X"),
        0x85 => Some("BufferStoreFormatD16Xy"),
        0x86 => Some("BufferStoreFormatD16Xyz"),
        0x87 => Some("BufferStoreFormatD16Xyzw"),
        _ => None,
    };
    Frame {
        encoding: ShaderEncoding::Mubuf,
        opcode: name.map_or_else(
            || Opcode::raw(RawOpcodeFamily::Mubuf, code as u16),
            Opcode::from_static,
        ),
        dwords: 2 + usize::from(extra >> 24 == 0xff),
    }
}

pub(crate) fn decode_flat(pc: u32, word: u32) -> Result<Frame, DecodeError> {
    let segment = (word >> 14) & 0x3;
    let code = (word >> 18) & 0x7f;
    if segment != 2 {
        return Err(DecodeError::FlatSegment {
            pc,
            segment,
            opcode: code,
        });
    }
    let name = match code {
        0x08 => Some("GlobalLoadUbyte"),
        0x09 => Some("GlobalLoadSbyte"),
        0x0a => Some("GlobalLoadUshort"),
        0x0b => Some("GlobalLoadSshort"),
        0x0c => Some("GlobalLoadDword"),
        0x0d => Some("GlobalLoadDwordx2"),
        0x0e => Some("GlobalLoadDwordx4"),
        0x0f => Some("GlobalLoadDwordx3"),
        0x18 => Some("GlobalStoreByte"),
        0x19 => Some("GlobalStoreByteD16Hi"),
        0x1a => Some("GlobalStoreShort"),
        0x1b => Some("GlobalStoreShortD16Hi"),
        0x1c => Some("GlobalStoreDword"),
        0x1d => Some("GlobalStoreDwordx2"),
        0x1e => Some("GlobalStoreDwordx4"),
        0x1f => Some("GlobalStoreDwordx3"),
        0x20 => Some("GlobalLoadUbyteD16"),
        0x21 => Some("GlobalLoadUbyteD16Hi"),
        0x22 => Some("GlobalLoadSbyteD16"),
        0x23 => Some("GlobalLoadSbyteD16Hi"),
        0x24 => Some("GlobalLoadShortD16"),
        0x25 => Some("GlobalLoadShortD16Hi"),
        0x32 => Some("GlobalAtomicAdd"),
        0x38 => Some("GlobalAtomicUMax"),
        _ => None,
    };
    known_frame(pc, word, ShaderEncoding::Flat, code, name, 2)
}

pub(crate) fn decode_mimg(word: u32) -> Frame {
    // MIMG is the one RDNA2 encoding whose opcode is split: OP[6:0] lives in
    // bits 24:18 while OP[7] lives in bit 0. Bit 0 must not be mistaken for
    // part of NSA; dropping it aliases IMAGE_MSAA_LOAD and later operations to
    // unrelated low-numbered image operations.
    let code = ((word >> 18) & 0x7f) | ((word & 1) << 7);
    let name = match code {
        0x00 => Some("ImageLoad"),
        0x01 => Some("ImageLoadMip"),
        0x08 => Some("ImageStore"),
        0x09 => Some("ImageStoreMip"),
        0x0e => Some("ImageGetResinfo"),
        0x0f => Some("ImageAtomicSwap"),
        0x10 => Some("ImageAtomicCmpswap"),
        0x11 => Some("ImageAtomicAdd"),
        0x12 => Some("ImageAtomicSub"),
        0x14 => Some("ImageAtomicSmin"),
        0x15 => Some("ImageAtomicUmin"),
        0x16 => Some("ImageAtomicSmax"),
        0x17 => Some("ImageAtomicUmax"),
        0x18 => Some("ImageAtomicAnd"),
        0x19 => Some("ImageAtomicOr"),
        0x1a => Some("ImageAtomicXor"),
        0x1b => Some("ImageAtomicInc"),
        0x1c => Some("ImageAtomicDec"),
        0x20 => Some("ImageSample"),
        0x22 => Some("ImageSampleD"),
        0x24 => Some("ImageSampleL"),
        0x25 => Some("ImageSampleB"),
        0x27 => Some("ImageSampleLz"),
        0x2f => Some("ImageSampleCLz"),
        0x30 => Some("ImageSampleO"),
        0x34 => Some("ImageSampleLO"),
        0x37 => Some("ImageSampleLzO"),
        0x40 => Some("ImageGather4"),
        0x47 => Some("ImageGather4Lz"),
        0x48 => Some("ImageGather4C"),
        0x4e => Some("ImageGather4CBCl"),
        0x57 => Some("ImageGather4LzO"),
        0x5f => Some("ImageGather4CLzO"),
        _ => None,
    };
    Frame {
        encoding: ShaderEncoding::Mimg,
        opcode: name.map_or_else(
            || {
                Opcode::raw(
                    RawOpcodeFamily::Mimg,
                    u16::try_from(code).expect("eight-bit MIMG opcode fits u16"),
                )
            },
            Opcode::from_static,
        ),
        dwords: 2 + ((word >> 1) & 3) as usize,
    }
}

pub(crate) fn decode_data_share(
    _word: u32,
    extra: u32,
    opcode: Opcode,
    sources: &mut Vec<Operand>,
    destinations: &mut Vec<Operand>,
) {
    let vector_address = extra & 0xff;
    let vector_data0 = (extra >> 8) & 0xff;
    let vector_data1 = (extra >> 16) & 0xff;
    let vector_destination = (extra >> 24) & 0xff;
    sources.push(Operand::vector(vector_address));
    if opcode.is("DsWriteAddtidB32") {
        sources.clear();
        sources.push(Operand::vector(vector_data0));
    } else if opcode.is("DsReadAddtidB32") {
        sources.clear();
    } else if opcode.is("DsWriteB32") {
        sources.push(Operand::vector(vector_data0));
    } else if opcode.is("DsWriteB64") {
        push_vector_range(sources, vector_data0, 2);
    } else if opcode.is("DsWriteB96") {
        push_vector_range(sources, vector_data0, 3);
    } else if opcode.is("DsWriteB128") {
        push_vector_range(sources, vector_data0, 4);
    } else if opcode.is("DsWrite2B32") || opcode.is("DsWrite2St64B32") {
        sources.extend([Operand::vector(vector_data0), Operand::vector(vector_data1)]);
    } else if opcode.is("DsSwizzleB32") {
        sources.clear();
        sources.push(Operand::vector(vector_data0));
    } else if opcode.is("DsCmpstB32") || opcode.is("DsCmpstRtnB32") {
        sources.extend([Operand::vector(vector_data0), Operand::vector(vector_data1)]);
    } else if is_data_share_atomic(opcode) {
        sources.push(Operand::vector(vector_data0));
    }

    if opcode.is("DsReadB32") || opcode.is("DsReadAddtidB32") || opcode.is("DsSwizzleB32") {
        destinations.push(Operand::vector(vector_destination));
    } else if opcode.is("DsRead2B32") || opcode.is("DsRead2St64B32") || opcode.is("DsReadB64") {
        push_vector_range(destinations, vector_destination, 2);
    } else if opcode.is("DsReadB96") {
        push_vector_range(destinations, vector_destination, 3);
    } else if opcode.is("DsReadB128") {
        push_vector_range(destinations, vector_destination, 4);
    } else if is_data_share_atomic(opcode) && opcode.contains("Rtn") {
        destinations.push(Operand::vector(vector_destination));
    }
}

pub(crate) fn decode_global_memory(
    word: u32,
    extra: u32,
    opcode: Opcode,
    sources: &mut Vec<Operand>,
    destinations: &mut Vec<Operand>,
) -> InstructionControl {
    let vector_address = extra & 0xff;
    let vector_data = (extra >> 8) & 0xff;
    let scalar_address_field = (extra >> 16) & 0x7f;
    let scalar_address = (scalar_address_field != 0x7f).then_some(scalar_address_field);
    let vector_destination = extra >> 24;
    let dword_count = memory_dword_count(opcode);
    sources.push(Operand::vector(vector_address));
    if let Some(scalar_address) = scalar_address {
        sources.push(Operand::scalar(scalar_address));
    } else {
        sources.push(Operand::vector(vector_address.saturating_add(1)));
    }
    let lds = word >> 13 & 1 != 0;
    if opcode.starts_with("GlobalLoad") {
        if !lds {
            push_vector_range(destinations, vector_destination, dword_count);
        }
    } else {
        push_vector_range(sources, vector_data, dword_count);
        if opcode.starts_with("GlobalAtomic") && word >> 16 & 1 != 0 {
            destinations.push(Operand::vector(vector_destination));
        }
    }
    InstructionControl::GlobalMemory(GlobalMemoryControl {
        dword_count,
        vector_address,
        vector_data,
        vector_destination,
        scalar_address,
        offset_bytes: sign_extend(word & 0xfff, 12),
        dlc: word >> 12 & 1 != 0,
        lds,
        glc: word >> 16 & 1 != 0,
        slc: word >> 17 & 1 != 0,
    })
}

pub(crate) fn decode_buffer_memory(
    word: u32,
    extra: u32,
    opcode: Opcode,
    literal: Option<u32>,
    sources: &mut Vec<Operand>,
    destinations: &mut Vec<Operand>,
) -> InstructionControl {
    let vector_address = extra & 0xff;
    let vector_data = (extra >> 8) & 0xff;
    let scalar_resource = ((extra >> 16) & 0x1f) * 4;
    let scalar_offset = (extra >> 24) & 0xff;
    let dword_count = memory_dword_count(opcode);
    sources.extend([
        Operand::vector(vector_address),
        Operand::scalar(scalar_resource),
        Operand::source(scalar_offset, literal),
    ]);
    let lds = !opcode.starts_with("TBuffer") && word >> 16 & 1 != 0;
    let tfe = extra >> 23 & 1 != 0;
    if opcode.contains("Load") {
        if !lds {
            push_vector_range(destinations, vector_data, dword_count);
            if tfe {
                destinations.push(Operand::vector(vector_data.saturating_add(dword_count)));
            }
        }
    } else {
        push_vector_range(sources, vector_data, dword_count);
        if opcode.starts_with("BufferAtomic") && word >> 14 & 1 != 0 {
            destinations.push(Operand::vector(vector_data));
        }
    }
    InstructionControl::BufferMemory(BufferMemoryControl {
        dword_count,
        vector_address,
        vector_data,
        scalar_resource,
        instruction_format: opcode.starts_with("TBuffer").then_some((word >> 19) & 0x7f),
        offset_bytes: i32::try_from(word & 0xfff).expect("12-bit buffer offset fits i32"),
        index_enabled: word >> 13 & 1 != 0,
        offset_enabled: word >> 12 & 1 != 0,
        dlc: word >> 15 & 1 != 0,
        lds,
        glc: word >> 14 & 1 != 0,
        slc: extra >> 22 & 1 != 0,
        tfe,
    })
}

pub(crate) fn decode_image(
    word: u32,
    words: &[u32],
    opcode: Opcode,
    sources: &mut Vec<Operand>,
    destinations: &mut Vec<Operand>,
) -> InstructionControl {
    let extra = words[1];
    let vector_address = extra & 0xff;
    let vector_data = (extra >> 8) & 0xff;
    let scalar_resource = ((extra >> 16) & 0x1f) * 4;
    let scalar_sampler = ((extra >> 21) & 0x1f) * 4;
    let mut address_registers = Vec::with_capacity(1 + words.len().saturating_sub(2) * 4);
    address_registers.push(vector_address);
    for extension in &words[2..] {
        for shift in [0, 8, 16, 24] {
            address_registers.push((extension >> shift) & 0xff);
        }
    }
    sources.extend(address_registers.iter().copied().map(Operand::vector));
    sources.extend([
        Operand::scalar(scalar_resource),
        Operand::scalar(scalar_sampler),
    ]);
    let result_count = (word >> 8 & 0xf).count_ones().max(1);
    let d16 = extra >> 31 & 1 != 0;
    let data_register_count = if d16 {
        result_count.div_ceil(2)
    } else {
        result_count
    };
    if opcode.starts_with("ImageStore") || opcode.starts_with("ImageAtomic") {
        let data_count = if opcode.is("ImageAtomicCmpswap") {
            2
        } else {
            data_register_count
        };
        push_vector_range(sources, vector_data, data_count);
        if opcode.starts_with("ImageAtomic") && word >> 13 & 1 != 0 {
            destinations.push(Operand::vector(vector_data));
        }
    } else {
        push_vector_range(destinations, vector_data, data_register_count);
    }
    let dimension = (word >> 3) & 7;
    InstructionControl::Image(ImageControl {
        dmask: (word >> 8) & 0xf,
        vector_address,
        address_registers,
        vector_data,
        scalar_resource,
        scalar_sampler,
        dimension,
        is_array: matches!(dimension, 4 | 5 | 7),
        glc: word >> 13 & 1 != 0,
        slc: word >> 25 & 1 != 0,
        a16: extra >> 30 & 1 != 0,
        d16,
        unorm: word >> 12 & 1 != 0,
    })
}

pub(crate) fn memory_dword_count(opcode: Opcode) -> u32 {
    let Some(name) = opcode.known_name() else {
        return 0;
    };
    if name.contains("FormatD16") {
        if name.ends_with("Xyz") || name.ends_with("Xyzw") {
            2
        } else {
            1
        }
    } else if name.ends_with("Xyzw") || name.ends_with("Dwordx4") {
        4
    } else if name.ends_with("Xyz") || name.ends_with("Dwordx3") {
        3
    } else if name.ends_with("Xy") || name.ends_with("Dwordx2") || name.ends_with("Cmpswap") {
        2
    } else {
        1
    }
}

pub(crate) fn is_data_share_atomic(opcode: Opcode) -> bool {
    matches!(
        opcode.known_name(),
        Some(
            "DsAddU32"
                | "DsSubU32"
                | "DsIncU32"
                | "DsDecU32"
                | "DsMinI32"
                | "DsMaxI32"
                | "DsMinU32"
                | "DsMaxU32"
                | "DsAndB32"
                | "DsOrB32"
                | "DsXorB32"
                | "DsCmpstB32"
                | "DsAddRtnU32"
                | "DsSubRtnU32"
                | "DsIncRtnU32"
                | "DsDecRtnU32"
                | "DsMinRtnI32"
                | "DsMaxRtnI32"
                | "DsMinRtnU32"
                | "DsMaxRtnU32"
                | "DsAndRtnB32"
                | "DsOrRtnB32"
                | "DsXorRtnB32"
                | "DsWrxchgRtnB32"
                | "DsCmpstRtnB32"
        )
    )
}
