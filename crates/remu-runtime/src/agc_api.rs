use crate::{
    ProbeDispatchState, ProcessGuestMemory, agc_command_abi, agc_shader_code_fingerprint,
    agc_shader_prefix_snapshot, guest_calloc, guest_free, guest_memcpy, low_i32, low_u32,
    patch_agc_shader_program_registers, read_guest_u8, read_guest_u16, read_guest_u32,
    read_guest_u64, relocate_guest_pointer, trace_agc_programs, trace_agc_registers,
    write_guest_u8, write_guest_u32, write_guest_u64,
};

pub(crate) fn agc_register_defaults(
    version: u64,
    internal: bool,
    state: &mut ProbeDispatchState,
) -> u64 {
    if !matches!(low_u32(version), 7 | 8 | 10 | 13) {
        return 0;
    }

    let cached = if internal {
        &mut state.agc_internal_register_defaults
    } else {
        &mut state.agc_primary_register_defaults
    };
    if *cached != 0 {
        return *cached;
    }

    let layout = if internal {
        crate::agc_register_defaults::RegisterDefaultsLayout::internal()
    } else {
        crate::agc_register_defaults::RegisterDefaultsLayout::primary()
    };
    let blob_length = layout.blob_length();
    let Ok(blob_length_u64) = u64::try_from(blob_length) else {
        return 0;
    };
    let address = guest_calloc(1, blob_length_u64);
    if address == 0 {
        return 0;
    }
    let Some(blob) = layout.build(address) else {
        guest_free(address);
        return 0;
    };
    debug_assert_eq!(blob.len(), blob_length);
    // SAFETY: `address` owns a live allocation of exactly `blob_length` bytes and the source is a
    // separate Vec allocation. The descriptor remains cached for the lifetime of the probe.
    unsafe {
        std::ptr::copy_nonoverlapping(blob.as_ptr(), address as *mut u8, blob.len());
    }
    *cached = address;
    eprintln!(
        "[REMu][AGC] register defaults internal={internal} version={} groups={} address=0x{address:016x}",
        low_u32(version),
        layout.group_count()
    );
    address
}

pub(crate) const AGC_SHADER_FILE_HEADER: u32 = 0x3433_3231;
pub(crate) const AGC_SHADER_VERSION: u32 = 0x18;
const AGC_SHADER_USER_DATA_OFFSET: u64 = 0x08;
const AGC_SHADER_CODE_OFFSET: u64 = 0x10;
const AGC_SHADER_CX_REGISTERS_OFFSET: u64 = 0x18;
pub(crate) const AGC_SHADER_SH_REGISTERS_OFFSET: u64 = 0x20;
pub(crate) const AGC_SHADER_SPECIALS_OFFSET: u64 = 0x28;
pub(crate) const AGC_SHADER_INPUT_SEMANTICS_OFFSET: u64 = 0x30;
pub(crate) const AGC_SHADER_OUTPUT_SEMANTICS_OFFSET: u64 = 0x38;
pub(crate) const AGC_SHADER_SIZE_OFFSET: u64 = 0x44;
pub(crate) const AGC_SHADER_TYPE_OFFSET: u64 = 0x5a;
pub(crate) const AGC_SHADER_SH_REGISTER_COUNT_OFFSET: u64 = 0x5c;
pub(crate) const AGC_SHADER_INPUT_SEMANTICS_COUNT_OFFSET: u64 = 0x50;
pub(crate) const AGC_SHADER_OUTPUT_SEMANTICS_COUNT_OFFSET: u64 = 0x56;

const AGC_COMMAND_BUFFER_CURSOR_UP_OFFSET: u64 = 0x10;
const AGC_COMMAND_BUFFER_CURSOR_DOWN_OFFSET: u64 = 0x18;
const AGC_COMMAND_BUFFER_RESERVED_DWORDS_OFFSET: u64 = 0x30;

pub(crate) const AGC_IT_NOP: u32 = 0x10;
pub(crate) const AGC_IT_SET_BASE: u32 = 0x11;
pub(crate) const AGC_IT_CLEAR_STATE: u32 = 0x12;
pub(crate) const AGC_IT_INDEX_BUFFER_SIZE: u32 = 0x13;
pub(crate) const AGC_IT_DISPATCH_DIRECT: u32 = 0x15;
pub(crate) const AGC_IT_DISPATCH_INDIRECT: u32 = 0x16;
pub(crate) const AGC_IT_SET_PREDICATION: u32 = 0x20;
pub(crate) const AGC_IT_COND_EXEC: u32 = 0x22;
pub(crate) const AGC_IT_DRAW_INDIRECT: u32 = 0x24;
pub(crate) const AGC_IT_DRAW_INDEX_INDIRECT: u32 = 0x25;
pub(crate) const AGC_IT_INDEX_BASE: u32 = 0x26;
pub(crate) const AGC_IT_DRAW_INDEX_2: u32 = 0x27;
pub(crate) const AGC_IT_INDEX_TYPE: u32 = 0x2a;
pub(crate) const AGC_IT_DRAW_INDEX_AUTO: u32 = 0x2d;
pub(crate) const AGC_IT_NUM_INSTANCES: u32 = 0x2f;
pub(crate) const AGC_IT_DRAW_INDEX_OFFSET: u32 = 0x35;
pub(crate) const AGC_IT_WRITE_DATA: u32 = 0x37;
pub(crate) const AGC_IT_DRAW_INDEX_INDIRECT_MULTI: u32 = 0x38;
pub(crate) const AGC_IT_WAIT_REG_MEM: u32 = 0x3c;
pub(crate) const AGC_IT_INDIRECT_BUFFER: u32 = 0x3f;
pub(crate) const AGC_IT_PFP_SYNC_ME: u32 = 0x42;
pub(crate) const AGC_IT_EVENT_WRITE: u32 = 0x46;
pub(crate) const AGC_IT_RELEASE_MEM: u32 = 0x49;
pub(crate) const AGC_IT_DMA_DATA: u32 = 0x50;
pub(crate) const AGC_IT_SET_SH_REG_INDIRECT: u32 = 0x63;
pub(crate) const AGC_IT_SET_UCONFIG_REG_INDIRECT: u32 = 0x64;
pub(crate) const AGC_IT_SET_CONTEXT_REG: u32 = 0x69;
pub(crate) const AGC_IT_SET_SH_REG: u32 = 0x76;
pub(crate) const AGC_IT_SET_UCONFIG_REG: u32 = 0x79;
pub(crate) const AGC_IT_SET_UCONFIG_REG_INDEX: u32 = 0x7a;
pub(crate) const AGC_IT_SET_CONTEXT_REG_INDIRECT: u32 = 0x9f;
const AGC_R_ZERO: u32 = 0x00;
pub(crate) const AGC_R_DRAW_INDEX_AUTO: u32 = 0x04;
pub(crate) const AGC_R_DRAW_RESET: u32 = 0x05;
pub(crate) const AGC_R_ACB_RESET: u32 = 0x09;
pub(crate) const AGC_R_WAIT_MEM_32: u32 = 0x0a;
pub(crate) const AGC_R_SH_REGS_INDIRECT: u32 = 0x11;
pub(crate) const AGC_R_CX_REGS_INDIRECT: u32 = 0x12;
pub(crate) const AGC_R_UC_REGS_INDIRECT: u32 = 0x13;
pub(crate) const AGC_R_ACQUIRE_MEM: u32 = 0x14;
pub(crate) const AGC_R_WRITE_DATA: u32 = 0x15;
pub(crate) const AGC_R_WAIT_MEM_64: u32 = 0x16;
pub(crate) const AGC_R_FLIP: u32 = 0x17;
pub(crate) const AGC_R_RELEASE_MEM: u32 = 0x18;
pub(crate) const AGC_R_DMA_DATA: u32 = 0x19;
const AGC_CB_SET_SH_REGISTER_RANGE_MARKER: u32 = 0x6875_000d;

pub(crate) const AGC_SPI_PS_INPUT_CNTL_0: u32 = 0x191;
pub(crate) const AGC_VGT_PRIMITIVE_TYPE: u32 = 0x242;
pub(crate) const AGC_VGT_INDEX_TYPE: u32 = 0x243;
const AGC_VGT_GS_OUT_PRIM_TYPE: u32 = 0x29b;
pub(crate) const AGC_SPI_SHADER_PGM_LO_PS: u32 = 0x008;
pub(crate) const AGC_SPI_SHADER_PGM_HI_PS: u32 = 0x009;
pub(crate) const AGC_SPI_SHADER_PGM_LO_ES: u32 = 0x0c8;
pub(crate) const AGC_SPI_SHADER_PGM_HI_ES: u32 = 0x0c9;
pub(crate) const AGC_CB_COLOR0_BASE: u32 = 0x318;
pub(crate) const AGC_CB_COLOR0_BASE_EXT: u32 = 0x390;
pub(crate) const AGC_CB_TARGET_MASK: u32 = 0x08e;
pub(crate) const AGC_CB_SHADER_MASK: u32 = 0x08f;
pub(crate) const AGC_CB_COLOR_REGISTER_STRIDE: u32 = 15;
pub(crate) const AGC_CB_COLOR0_VIEW: u32 = 0x31b;
pub(crate) const AGC_CB_COLOR0_INFO: u32 = 0x31c;
pub(crate) const AGC_CB_COLOR0_ATTRIB: u32 = 0x31d;
pub(crate) const AGC_CB_COLOR0_ATTRIB2: u32 = 0x3b0;
pub(crate) const AGC_CB_COLOR0_ATTRIB3: u32 = 0x3b8;
pub(crate) const AGC_COLOR_TARGET_COUNT: u32 = 8;
pub(crate) const AGC_DB_RENDER_CONTROL: u32 = 0x000;
pub(crate) const AGC_DB_DEPTH_SIZE_XY: u32 = 0x007;
pub(crate) const AGC_DB_Z_INFO: u32 = 0x010;
pub(crate) const AGC_DB_Z_READ_BASE: u32 = 0x012;
pub(crate) const AGC_DB_Z_WRITE_BASE: u32 = 0x014;
pub(crate) const AGC_DB_Z_READ_BASE_HI: u32 = 0x01a;
pub(crate) const AGC_DB_Z_WRITE_BASE_HI: u32 = 0x01c;
pub(crate) const AGC_DB_DEPTH_CONTROL: u32 = 0x200;
pub(crate) const AGC_COMPUTE_USER_DATA_0: u32 = 0x240;

const AGC_SHADER_SPECIAL_GE_CNTL_OFFSET: u64 = 0x00;
const AGC_SHADER_SPECIAL_VGT_SHADER_STAGES_EN_OFFSET: u64 = 0x08;
const AGC_SHADER_SPECIAL_VGT_GS_OUT_PRIM_TYPE_OFFSET: u64 = 0x20;
const AGC_SHADER_SPECIAL_GE_USER_VGPR_EN_OFFSET: u64 = 0x28;

pub(crate) fn agc_create_shader(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let destination = arguments[0];
    let header = arguments[1];
    let code = arguments[2];
    if header == 0 || code == 0 {
        return u64::MAX;
    }
    if read_guest_u32(header) != AGC_SHADER_FILE_HEADER
        || read_guest_u32(header + 4) != AGC_SHADER_VERSION
    {
        eprintln!(
            "[REMu][AGC] rejected shader header at {header:#x}: magic={:#x} version={:#x}",
            read_guest_u32(header),
            read_guest_u32(header + 4)
        );
        return u64::MAX;
    }

    for offset in [
        AGC_SHADER_CX_REGISTERS_OFFSET,
        AGC_SHADER_SH_REGISTERS_OFFSET,
        AGC_SHADER_USER_DATA_OFFSET,
        AGC_SHADER_SPECIALS_OFFSET,
        AGC_SHADER_INPUT_SEMANTICS_OFFSET,
        AGC_SHADER_OUTPUT_SEMANTICS_OFFSET,
    ] {
        relocate_guest_pointer(header + offset);
    }
    write_guest_u64(header + AGC_SHADER_CODE_OFFSET, code);

    let user_data = read_guest_u64(header + AGC_SHADER_USER_DATA_OFFSET);
    if user_data != 0 {
        for offset in [0, 8, 0x10, 0x18, 0x20] {
            relocate_guest_pointer(user_data + offset);
        }
    }

    if !patch_agc_shader_program_registers(header, code) {
        return u64::MAX;
    }
    let code_size = read_guest_u32(header + AGC_SHADER_SIZE_OFFSET);
    let mut registration_code_fingerprint = None;
    if trace_agc_programs() {
        let input_semantics_address = read_guest_u64(header + AGC_SHADER_INPUT_SEMANTICS_OFFSET);
        let input_semantics_count = u32::from(read_guest_u16(
            header + AGC_SHADER_INPUT_SEMANTICS_COUNT_OFFSET,
        ));
        let output_semantics_address = read_guest_u64(header + AGC_SHADER_OUTPUT_SEMANTICS_OFFSET);
        let output_semantics_count = u32::from(read_guest_u16(
            header + AGC_SHADER_OUTPUT_SEMANTICS_COUNT_OFFSET,
        ));
        let input_semantics = (0..input_semantics_count.min(32))
            .filter(|_| input_semantics_address != 0)
            .map(|index| read_guest_u32(input_semantics_address + u64::from(index) * 4))
            .collect::<Vec<_>>();
        let output_semantics = (0..output_semantics_count.min(32))
            .filter(|_| output_semantics_address != 0)
            .map(|index| read_guest_u32(output_semantics_address + u64::from(index) * 4))
            .collect::<Vec<_>>();
        let code_prefix = agc_shader_prefix_snapshot(code);
        registration_code_fingerprint =
            agc_shader_code_fingerprint(&ProcessGuestMemory, code, code_size);
        eprintln!(
            "[REMu][AGC][SHADER] registered code={code:#x} size={code_size} fingerprint={registration_code_fingerprint:x?} prefix={code_prefix:x?} header={header:#x} type={} user_data={:#x} specials={:#x} inputs={input_semantics_count}@{input_semantics_address:#x}/{input_semantics:x?} outputs={output_semantics_count}@{output_semantics_address:#x}/{output_semantics:x?}",
            read_guest_u8(header + AGC_SHADER_TYPE_OFFSET),
            read_guest_u64(header + AGC_SHADER_USER_DATA_OFFSET),
            read_guest_u64(header + AGC_SHADER_SPECIALS_OFFSET),
        );
    }
    state.agc_shader_registry.headers.insert(code, header);
    state.agc_shader_registry.code_sizes.insert(code, code_size);
    match registration_code_fingerprint {
        Some(fingerprint) => {
            state
                .agc_shader_registry
                .registration_code_fingerprints
                .insert(code, fingerprint);
        }
        None => {
            state
                .agc_shader_registry
                .registration_code_fingerprints
                .remove(&code);
        }
    }
    state.agc_shader_registry.snapshots.remove(&code);
    state.agc_shader_registry.decode_failures.remove(&code);
    if destination != 0 {
        write_guest_u64(destination, header);
    }
    0
}

pub(crate) fn agc_pm4(length_dwords: u32, opcode: u32, register: u32) -> u32 {
    0xc000_0000
        | ((length_dwords.wrapping_sub(2) & 0x3fff) << 16)
        | ((opcode & 0xff) << 8)
        | ((register & 0x3f) << 2)
}

/// Returns the encoded PM4 packet size, including AGC's one-dword uncounted NOP.
///
/// A Type-3 count field normally stores `dwords - 2`. The all-ones count paired with NOP is a
/// special filler encoding, however, and occupies only its header dword. Treating it as the
/// ordinary maximum-length packet desynchronizes command-buffer scans or rejects a valid stream
/// as a 16,385-dword packet.
pub(crate) const fn agc_packet_dword_count(header: u32) -> Option<u32> {
    match header >> 30 {
        2 => Some(1),
        3 if header & 0x3fff_ff00 == 0x3fff_1000 => Some(1),
        3 => Some(((header >> 16) & 0x3fff) + 2),
        _ => None,
    }
}

pub(crate) fn agc_allocate_command_dwords(command_buffer: u64, size_dwords: u32) -> Option<u64> {
    if command_buffer == 0 || size_dwords == 0 {
        return None;
    }
    let cursor_up = read_guest_u64(command_buffer + AGC_COMMAND_BUFFER_CURSOR_UP_OFFSET);
    let cursor_down = read_guest_u64(command_buffer + AGC_COMMAND_BUFFER_CURSOR_DOWN_OFFSET);
    let reserved_dwords =
        read_guest_u32(command_buffer + AGC_COMMAND_BUFFER_RESERVED_DWORDS_OFFSET);
    let available_dwords = cursor_down
        .checked_sub(cursor_up)
        .map_or(0, |bytes| bytes / 4);
    let usable_dwords = available_dwords.saturating_sub(u64::from(reserved_dwords));
    if u64::from(size_dwords) > usable_dwords {
        eprintln!(
            "[REMu][AGC] command buffer full at {command_buffer:#x}: need={size_dwords} available={usable_dwords}"
        );
        return None;
    }
    let size_bytes = u64::from(size_dwords).checked_mul(4)?;
    let next_cursor = cursor_up.checked_add(size_bytes)?;
    write_guest_u64(
        command_buffer + AGC_COMMAND_BUFFER_CURSOR_UP_OFFSET,
        next_cursor,
    );
    agc_command_abi::track_graphics_command_allocation(
        command_buffer,
        cursor_up,
        size_dwords,
        false,
    );
    Some(cursor_up)
}

pub(crate) fn agc_dcb_reset_queue(arguments: [u64; 6]) -> u64 {
    let command_buffer = arguments[0];
    if low_u32(arguments[1]) != 0x3ff || low_u32(arguments[2]) != 0 {
        return 0;
    }
    let Some(command) = agc_allocate_command_dwords(command_buffer, 2) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(2, AGC_IT_NOP, AGC_R_DRAW_RESET));
    write_guest_u32(command + 4, 0);
    command
}

pub(crate) fn agc_dcb_set_registers_indirect(arguments: [u64; 6], packet_register: u32) -> u64 {
    let command_buffer = arguments[0];
    let registers = arguments[1];
    let register_count = low_u32(arguments[2]);
    let Some(command) = agc_allocate_command_dwords(command_buffer, 4) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(4, AGC_IT_NOP, packet_register));
    write_guest_u32(command + 4, register_count);
    write_guest_u32(command + 8, low_u32(registers));
    write_guest_u32(command + 12, (registers >> 32) as u32);
    command
}

pub(crate) fn agc_set_indirect_patch_address(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let registers = arguments[1];
    if command == 0 || registers == 0 {
        return u64::MAX;
    }
    write_guest_u32(command + 8, low_u32(registers));
    write_guest_u32(command + 12, (registers >> 32) as u32);
    0
}

pub(crate) fn agc_add_indirect_patch_registers(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    if command == 0 {
        return u64::MAX;
    }
    let register_count = low_u32(arguments[1]);
    let current_count = read_guest_u32(command + 4);
    write_guest_u32(command + 4, current_count.wrapping_add(register_count));
    0
}

fn agc_copy_shader_register(source: u64, destination: u64) {
    write_guest_u32(destination, read_guest_u32(source));
    write_guest_u32(destination + 4, read_guest_u32(source + 4));
}

fn agc_primitive_type_to_gs_output(primitive_type: u32) -> u32 {
    match primitive_type {
        1 => 0,                    // point list -> points
        2 | 3 | 10 | 11 | 18 => 1, // line primitives -> lines
        7 => 3,                    // rectangle list -> 2D rectangle
        17 => 4,                   // legacy rectangle list
        _ => 2,                    // triangles
    }
}

pub(crate) fn agc_create_primitive_state(arguments: [u64; 6]) -> u64 {
    let cx_registers = arguments[0];
    let uc_registers = arguments[1];
    let hull_shader = arguments[2];
    let geometry_shader = arguments[3];
    let primitive_type = low_u32(arguments[4]);
    if cx_registers == 0 && uc_registers == 0 {
        return 0;
    }
    if geometry_shader == 0
        || read_guest_u8(geometry_shader + AGC_SHADER_TYPE_OFFSET) != SHADER_BINARY_TYPE_GS
        || (hull_shader != 0
            && read_guest_u8(hull_shader + AGC_SHADER_TYPE_OFFSET) != SHADER_BINARY_TYPE_HS)
    {
        return u64::MAX;
    }
    let geometry_specials = read_guest_u64(geometry_shader + AGC_SHADER_SPECIALS_OFFSET);
    let hull_specials = if hull_shader == 0 {
        0
    } else {
        read_guest_u64(hull_shader + AGC_SHADER_SPECIALS_OFFSET)
    };
    if geometry_specials == 0 || (hull_shader != 0 && hull_specials == 0) {
        return u64::MAX;
    }

    if cx_registers != 0 {
        agc_copy_shader_register(
            geometry_specials + AGC_SHADER_SPECIAL_VGT_SHADER_STAGES_EN_OFFSET,
            cx_registers,
        );
        if read_guest_u32(cx_registers + 4) & 0x20 != 0 {
            agc_copy_shader_register(
                geometry_specials + AGC_SHADER_SPECIAL_VGT_GS_OUT_PRIM_TYPE_OFFSET,
                cx_registers + 8,
            );
        } else {
            write_guest_u32(cx_registers + 8, AGC_VGT_GS_OUT_PRIM_TYPE);
            write_guest_u32(
                cx_registers + 12,
                agc_primitive_type_to_gs_output(primitive_type),
            );
        }
        if hull_specials != 0 {
            let combined_stages = read_guest_u32(cx_registers + 4)
                | read_guest_u32(
                    hull_specials + AGC_SHADER_SPECIAL_VGT_SHADER_STAGES_EN_OFFSET + 4,
                );
            write_guest_u32(cx_registers + 4, combined_stages);
            if combined_stages & 0x20 == 0 {
                agc_copy_shader_register(
                    hull_specials + AGC_SHADER_SPECIAL_VGT_GS_OUT_PRIM_TYPE_OFFSET,
                    cx_registers + 8,
                );
            }
        }
    }

    if uc_registers != 0 {
        agc_copy_shader_register(
            geometry_specials + AGC_SHADER_SPECIAL_GE_CNTL_OFFSET,
            uc_registers,
        );
        agc_copy_shader_register(
            if hull_specials == 0 {
                geometry_specials + AGC_SHADER_SPECIAL_GE_USER_VGPR_EN_OFFSET
            } else {
                hull_specials + AGC_SHADER_SPECIAL_GE_USER_VGPR_EN_OFFSET
            },
            uc_registers + 8,
        );
        write_guest_u32(uc_registers + 16, AGC_VGT_PRIMITIVE_TYPE);
        write_guest_u32(uc_registers + 20, primitive_type);
    }
    0
}

fn agc_apply_interpolant_default_value(mut value: u32, pixel_semantic: u32) -> u32 {
    value &= !0x0000_0300;
    value |= ((pixel_semantic >> 28) & 0x3) << 8;
    value
}

fn agc_apply_interpolant_default_value_hi(mut value: u32, pixel_semantic: u32) -> u32 {
    value &= !0x0060_0000;
    value |= ((pixel_semantic >> 30) & 0x3) << 21;
    value
}

fn agc_interpolant_mapping_value(
    mut value: u32,
    pixel_semantic: u32,
    geometry_semantic: u32,
) -> u32 {
    let flat_shade = if pixel_semantic & 0x0140_0000 != 0 {
        0x0000_0400
    } else {
        0
    };
    value &= !0x1f;
    value |= (geometry_semantic >> 8) & 0x1f;
    value &= !0x0000_0400;
    value |= flat_shade;
    agc_apply_interpolant_default_value(value, pixel_semantic)
}

fn agc_interpolant_default_value(mut value: u32, pixel_semantic: u32) -> u32 {
    value &= !0x1f;
    value &= !0x0000_0400;
    agc_apply_interpolant_default_value(value, pixel_semantic)
}

fn agc_interpolant_f16_value(pixel_semantic: u32, geometry_semantic: Option<u32>) -> u32 {
    let mut value = (pixel_semantic << 4) & 0x0300_0000;
    if let Some(geometry_semantic) = geometry_semantic {
        let common = pixel_semantic & geometry_semantic;
        value &= 0xfff7_ffdf;
        value |= (common >> 15) & 0x20;
        value ^= 0x0008_0020;
        value &= !0x0010_0000;
        value |= (!common >> 1) & 0x0010_0000;
    } else {
        value |= 0x0018_0020;
    }
    agc_apply_interpolant_default_value_hi(value, pixel_semantic)
}

fn agc_interpolant_non_f16_value(pixel_semantic: u32, geometry_semantic: Option<u32>) -> u32 {
    if pixel_semantic & 0x0100_0000 != 0 || geometry_semantic.is_none() {
        0x20
    } else {
        0
    }
}

fn agc_write_interpolant_register(registers: u64, index: u32, value: u32) {
    let destination = registers + u64::from(index) * 8;
    write_guest_u32(destination, AGC_SPI_PS_INPUT_CNTL_0 + index);
    write_guest_u32(destination + 4, value);
}

pub(crate) fn agc_create_interpolant_mapping(arguments: [u64; 6]) -> u64 {
    let registers = arguments[0];
    let geometry_shader = arguments[1];
    let pixel_shader = arguments[2];
    if registers == 0 {
        return u64::MAX;
    }

    let input_count = if pixel_shader == 0 {
        0
    } else {
        read_guest_u32(pixel_shader + AGC_SHADER_INPUT_SEMANTICS_COUNT_OFFSET)
    };
    if input_count > 32 {
        return u64::MAX;
    }
    if input_count == 0 {
        for index in 0..32 {
            agc_write_interpolant_register(registers, index, index);
        }
        if trace_agc_registers() {
            eprintln!(
                "[REMu][AGC][INTERPOLANTS] registers={registers:#x} gs_header={geometry_shader:#x} ps_header={pixel_shader:#x} inputs=0 identity=true"
            );
        }
        return 0;
    }
    let input_semantics = read_guest_u64(pixel_shader + AGC_SHADER_INPUT_SEMANTICS_OFFSET);
    if input_semantics == 0 || geometry_shader == 0 {
        return u64::MAX;
    }
    let output_semantics = read_guest_u64(geometry_shader + AGC_SHADER_OUTPUT_SEMANTICS_OFFSET);
    let output_count = u32::from(read_guest_u16(
        geometry_shader + AGC_SHADER_OUTPUT_SEMANTICS_COUNT_OFFSET,
    ));
    if output_count != 0 && output_semantics == 0 {
        return u64::MAX;
    }

    let mut traced_mappings = Vec::new();
    for index in 0..input_count {
        let pixel_semantic = read_guest_u32(input_semantics + u64::from(index) * 4);
        let semantic = pixel_semantic & 0xff;
        let geometry_semantic = (0..output_count).find_map(|output_index| {
            let candidate = read_guest_u32(output_semantics + u64::from(output_index) * 4);
            (candidate & 0xff == semantic).then_some(candidate)
        });
        let value = if pixel_semantic & 0x0030_0000 != 0 {
            agc_interpolant_f16_value(pixel_semantic, geometry_semantic)
        } else {
            agc_interpolant_non_f16_value(pixel_semantic, geometry_semantic)
        };
        let value = geometry_semantic.map_or_else(
            || agc_interpolant_default_value(value, pixel_semantic),
            |geometry_semantic| {
                agc_interpolant_mapping_value(value, pixel_semantic, geometry_semantic)
            },
        );
        agc_write_interpolant_register(registers, index, value);
        if trace_agc_registers() {
            traced_mappings.push((index, pixel_semantic, geometry_semantic, value));
        }
    }
    for index in input_count..32 {
        agc_write_interpolant_register(registers, index, index);
    }
    if trace_agc_registers() {
        eprintln!(
            "[REMu][AGC][INTERPOLANTS] registers={registers:#x} gs_header={geometry_shader:#x} gs_code={:#x} gs_outputs={} ps_header={pixel_shader:#x} ps_code={:#x} ps_inputs={} mappings={traced_mappings:x?}",
            read_guest_u64(geometry_shader + AGC_SHADER_CODE_OFFSET),
            output_count,
            read_guest_u64(pixel_shader + AGC_SHADER_CODE_OFFSET),
            input_count,
        );
    }
    0
}

pub(crate) fn agc_cb_set_sh_register_range_direct(arguments: [u64; 6]) -> u64 {
    let command_buffer = arguments[0];
    let offset = low_u32(arguments[1]);
    let values = arguments[2];
    let value_count = low_u32(arguments[3]);
    if command_buffer == 0 || offset == 0 || offset > 0x3ff || value_count == 0 {
        return 0;
    }
    let Some(marker) = agc_allocate_command_dwords(command_buffer, 2) else {
        return 0;
    };
    write_guest_u32(marker, agc_pm4(2, AGC_IT_NOP, AGC_R_ZERO));
    write_guest_u32(marker + 4, AGC_CB_SET_SH_REGISTER_RANGE_MARKER);
    let Some(command) = agc_allocate_command_dwords(command_buffer, value_count.saturating_add(2))
    else {
        return 0;
    };
    write_guest_u32(
        command,
        agc_pm4(value_count.saturating_add(2), AGC_IT_SET_SH_REG, 0),
    );
    write_guest_u32(command + 4, offset);
    for index in 0..value_count {
        let value = if values == 0 {
            0
        } else {
            read_guest_u32(values + u64::from(index) * 4)
        };
        write_guest_u32(command + 8 + u64::from(index) * 4, value);
    }
    command
}

pub(crate) fn agc_get_data_packet_payload_address(arguments: [u64; 6]) -> u64 {
    let output = arguments[0];
    let command = arguments[1];
    if output == 0 || command == 0 {
        return u64::MAX;
    }
    let payload = if low_i32(arguments[2]) == 0 {
        let header = read_guest_u32(command);
        if header & 0x3fff_0000 == 0x3fff_0000 {
            0
        } else {
            command + 4
        }
    } else {
        command + 8
    };
    write_guest_u64(output, payload);
    0
}

pub(crate) fn agc_dcb_event_write(arguments: [u64; 6]) -> u64 {
    let event_type = low_u32(arguments[1]) & 0xff;
    if event_type > 0x3f || arguments[2] != 0 {
        return 0;
    }
    let Some(command) = agc_allocate_command_dwords(arguments[0], 2) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(2, AGC_IT_EVENT_WRITE, 0));
    write_guest_u32(command + 4, event_type);
    command
}

/// Gen5 shader header field offsets, as the toolchain emits them.
pub(crate) const SHADER_HEADER_CODE_OFFSET: u64 = 0x10;
pub(crate) const SHADER_HEADER_SH_REGISTERS_OFFSET: u64 = 0x20;
pub(crate) const SHADER_HEADER_TYPE_OFFSET: u64 = 0x5a;
pub(crate) const SHADER_HEADER_NUM_SH_REGISTERS_OFFSET: u64 = 0x5c;
pub(crate) const SHADER_HEADER_SIZE: u64 = 0x60;
/// One `(offset, value)` pair of a shader register.
const SHADER_REGISTER_SIZE: u64 = 8;

const SHADER_BINARY_TYPE_GS: u8 = 2;
const SHADER_BINARY_TYPE_HS: u8 = 3;
pub(crate) const SHADER_BINARY_TYPE_GS_FRONT: u8 = 4;
const SHADER_BINARY_TYPE_HS_FRONT: u8 = 5;
pub(crate) const SHADER_BINARY_TYPE_GS_BACK: u8 = 6;
const SHADER_BINARY_TYPE_HS_BACK: u8 = 7;

const SPI_SHADER_PGM_CHKSUM_GS: u32 = 0x80;
pub(crate) const SPI_SHADER_PGM_LO_ES: u32 = 0xc8;
const SPI_SHADER_PGM_LO_LS: u32 = 0x148;
const GRAPHICS_ERROR_INVALID_SHADER_HALVES: u64 = 0x8a6c_000b;

/// Confirms two shader halves form a fusable merged pair and reports which stage they produce.
fn fused_shader_stage(front: u64, back: u64) -> Option<u8> {
    if front == 0 || back == 0 {
        return None;
    }
    let front_type = read_guest_u8(front + SHADER_HEADER_TYPE_OFFSET);
    let back_type = read_guest_u8(back + SHADER_HEADER_TYPE_OFFSET);
    match (front_type, back_type) {
        (SHADER_BINARY_TYPE_GS_FRONT, SHADER_BINARY_TYPE_GS_BACK) => Some(SHADER_BINARY_TYPE_GS),
        (SHADER_BINARY_TYPE_HS_FRONT, SHADER_BINARY_TYPE_HS_BACK) => Some(SHADER_BINARY_TYPE_HS),
        _ => None,
    }
}

/// Reports the scratch size needed to fuse two merged shader halves.
///
/// The fused stage keeps the back half's shader registers, so the caller must supply storage for
/// exactly that many register pairs.
pub(crate) fn agc_get_fused_shader_size(arguments: [u64; 6]) -> u64 {
    let (output, front, back) = (arguments[0], arguments[1], arguments[2]);
    if output == 0 || fused_shader_stage(front, back).is_none() {
        return GRAPHICS_ERROR_INVALID_SHADER_HALVES;
    }
    let registers = u64::from(read_guest_u8(back + SHADER_HEADER_NUM_SH_REGISTERS_OFFSET));
    write_guest_u64(output, registers * SHADER_REGISTER_SIZE);
    write_guest_u64(output + 8, 4);
    0
}

/// Finds a shader register by its hardware offset, skipping earlier occurrences.
fn find_shader_register(registers: u64, count: u64, offset: u32, occurrence: u32) -> Option<u64> {
    let mut seen = 0;
    for index in 0..count {
        let entry = registers + index * SHADER_REGISTER_SIZE;
        if read_guest_u32(entry) == offset {
            if seen == occurrence {
                return Some(entry);
            }
            seen += 1;
        }
    }
    None
}

/// Fuses a merged shader's two halves into one stage description.
///
/// The fused stage is the back half with the front half's entry point: its register block is
/// copied into caller-supplied scratch, the front half's checksum registers are carried over, and
/// the program address registers are repointed at the front half's code.
pub(crate) fn agc_fuse_shader_halves(arguments: [u64; 6]) -> u64 {
    let (fused, front, back, scratch) = (arguments[0], arguments[1], arguments[2], arguments[3]);
    let Some(stage) = fused_shader_stage(front, back) else {
        if trace_agc_programs() {
            eprintln!(
                "[REMu][AGC][FUSE] rejected fused={fused:#x} front={front:#x}/type={} back={back:#x}/type={} scratch={scratch:#x}",
                if front == 0 {
                    0
                } else {
                    read_guest_u8(front + SHADER_HEADER_TYPE_OFFSET)
                },
                if back == 0 {
                    0
                } else {
                    read_guest_u8(back + SHADER_HEADER_TYPE_OFFSET)
                },
            );
        }
        return GRAPHICS_ERROR_INVALID_SHADER_HALVES;
    };
    if fused == 0 {
        return GRAPHICS_ERROR_INVALID_SHADER_HALVES;
    }
    if trace_agc_programs() {
        eprintln!(
            "[REMu][AGC][FUSE] begin fused={fused:#x} scratch={scratch:#x} stage={stage} \
             front={front:#x}/type={}/code={:#x}/size={}/sh={:#x}/regs={}/user={:#x}/specials={:#x} \
             back={back:#x}/type={}/code={:#x}/size={}/sh={:#x}/regs={}/user={:#x}/specials={:#x}",
            read_guest_u8(front + SHADER_HEADER_TYPE_OFFSET),
            read_guest_u64(front + SHADER_HEADER_CODE_OFFSET),
            read_guest_u32(front + AGC_SHADER_SIZE_OFFSET),
            read_guest_u64(front + SHADER_HEADER_SH_REGISTERS_OFFSET),
            read_guest_u8(front + SHADER_HEADER_NUM_SH_REGISTERS_OFFSET),
            read_guest_u64(front + AGC_SHADER_USER_DATA_OFFSET),
            read_guest_u64(front + AGC_SHADER_SPECIALS_OFFSET),
            read_guest_u8(back + SHADER_HEADER_TYPE_OFFSET),
            read_guest_u64(back + SHADER_HEADER_CODE_OFFSET),
            read_guest_u32(back + AGC_SHADER_SIZE_OFFSET),
            read_guest_u64(back + SHADER_HEADER_SH_REGISTERS_OFFSET),
            read_guest_u8(back + SHADER_HEADER_NUM_SH_REGISTERS_OFFSET),
            read_guest_u64(back + AGC_SHADER_USER_DATA_OFFSET),
            read_guest_u64(back + AGC_SHADER_SPECIALS_OFFSET),
        );
    }
    guest_memcpy(fused, back, SHADER_HEADER_SIZE);
    write_guest_u8(fused + SHADER_HEADER_TYPE_OFFSET, stage);

    let back_registers = read_guest_u64(back + SHADER_HEADER_SH_REGISTERS_OFFSET);
    let register_count = u64::from(read_guest_u8(back + SHADER_HEADER_NUM_SH_REGISTERS_OFFSET));
    let fused_registers = if scratch != 0 && back_registers != 0 && register_count != 0 {
        guest_memcpy(
            scratch,
            back_registers,
            register_count * SHADER_REGISTER_SIZE,
        );
        write_guest_u64(fused + SHADER_HEADER_SH_REGISTERS_OFFSET, scratch);
        scratch
    } else {
        back_registers
    };

    let front_registers = read_guest_u64(front + SHADER_HEADER_SH_REGISTERS_OFFSET);
    let front_count = u64::from(read_guest_u8(front + SHADER_HEADER_NUM_SH_REGISTERS_OFFSET));
    if stage == SHADER_BINARY_TYPE_GS && fused_registers != 0 && front_registers != 0 {
        // Both checksum halves belong to the front program.
        for occurrence in 0..2 {
            let destination = find_shader_register(
                fused_registers,
                register_count,
                SPI_SHADER_PGM_CHKSUM_GS,
                occurrence,
            );
            let source = find_shader_register(
                front_registers,
                front_count,
                SPI_SHADER_PGM_CHKSUM_GS,
                occurrence,
            );
            if let (Some(destination), Some(source)) = (destination, source) {
                write_guest_u32(destination + 4, read_guest_u32(source + 4));
            }
        }
    }

    // The merged stage begins at the front half's code, so its address registers name that
    // program rather than the back half's.
    let code = read_guest_u64(front + SHADER_HEADER_CODE_OFFSET);
    let address_register = if stage == SHADER_BINARY_TYPE_GS {
        SPI_SHADER_PGM_LO_ES
    } else {
        SPI_SHADER_PGM_LO_LS
    };
    if fused_registers != 0 && code != 0 {
        if let Some(entry) =
            find_shader_register(fused_registers, register_count, address_register, 0)
        {
            write_guest_u32(entry + 4, low_u32(code >> 8));
        }
        if let Some(entry) =
            find_shader_register(fused_registers, register_count, address_register + 1, 0)
        {
            let previous = read_guest_u32(entry + 4);
            write_guest_u32(
                entry + 4,
                (previous & 0xffff_ff00) | (low_u32(code >> 40) & 0xff),
            );
        }
    }
    if trace_agc_programs() {
        eprintln!(
            "[REMu][AGC][FUSE] end fused={fused:#x}/type={}/code={:#x}/size={}/sh={:#x}/regs={}/user={:#x}/specials={:#x}",
            read_guest_u8(fused + SHADER_HEADER_TYPE_OFFSET),
            read_guest_u64(fused + SHADER_HEADER_CODE_OFFSET),
            read_guest_u32(fused + AGC_SHADER_SIZE_OFFSET),
            read_guest_u64(fused + SHADER_HEADER_SH_REGISTERS_OFFSET),
            read_guest_u8(fused + SHADER_HEADER_NUM_SH_REGISTERS_OFFSET),
            read_guest_u64(fused + AGC_SHADER_USER_DATA_OFFSET),
            read_guest_u64(fused + AGC_SHADER_SPECIALS_OFFSET),
        );
    }
    0
}

pub(crate) fn agc_cb_nop(arguments: [u64; 6]) -> u64 {
    let dword_count = low_u32(arguments[1]);
    if !(2..=0x4001).contains(&dword_count) {
        return 0;
    }
    let Some(command) = agc_allocate_command_dwords(arguments[0], dword_count) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(dword_count, AGC_IT_NOP, AGC_R_ZERO));
    for index in 1..dword_count {
        write_guest_u32(command + u64::from(index) * 4, 0);
    }
    command
}

pub(crate) fn agc_dcb_dma_data(arguments: [u64; 6], stack_arguments: [u64; 6]) -> u64 {
    let byte_count = low_u32(stack_arguments[2]);
    if arguments[0] == 0 || byte_count == 0 || byte_count & 3 != 0 {
        return 0;
    }
    let Some(command) = agc_allocate_command_dwords(arguments[0], 8) else {
        return 0;
    };
    let destination = low_u32(arguments[1]) & 0xff;
    let destination_cache_policy = low_u32(arguments[2]) & 0xff;
    let source = low_u32(arguments[3]) & 0xff;
    let source_cache_policy = low_u32(arguments[5]) & 0xff;
    let control4 = low_u32(stack_arguments[0]) & 0xff;
    let source_address = stack_arguments[1];
    let control7 = low_u32(stack_arguments[3]) & 0xff;
    let control8 = low_u32(stack_arguments[4]) & 0xff;
    let control9 = low_u32(stack_arguments[5]) & 0xff;
    write_guest_u32(command, agc_pm4(8, AGC_IT_NOP, AGC_R_DMA_DATA));
    write_guest_u32(
        command + 4,
        destination
            | (destination_cache_policy << 8)
            | (source << 16)
            | (source_cache_policy << 24),
    );
    write_guest_u32(
        command + 8,
        control4 | (control7 << 8) | (control8 << 16) | (control9 << 24),
    );
    write_guest_u32(command + 12, byte_count);
    write_guest_u64(command + 16, arguments[4]);
    write_guest_u64(command + 24, source_address);
    command
}

pub(crate) fn agc_cb_release_mem(arguments: [u64; 6], stack_arguments: [u64; 6]) -> u64 {
    let destination = low_u32(arguments[3]) & 0xff;
    let data_selection = low_u32(stack_arguments[0]) & 0xff;
    let gds_offset = low_u32(stack_arguments[2]) & 0xffff;
    let gds_size = low_u32(stack_arguments[3]) & 0xffff;
    let interrupt = low_u32(stack_arguments[4]) & 0xff;
    if arguments[0] == 0
        || destination > 1
        || data_selection > 3
        || gds_offset != 0
        || gds_size > 2
        || interrupt > 3
    {
        return 0;
    }
    let Some(command) = agc_allocate_command_dwords(arguments[0], 8) else {
        return 0;
    };
    let action = low_u32(arguments[1]) & 0xff;
    let gcr_control = low_u32(arguments[2]) & 0xffff;
    let cache_policy = low_u32(arguments[4]) & 0xff;
    let destination_address = arguments[5];
    let data = stack_arguments[1];
    write_guest_u32(command, agc_pm4(8, AGC_IT_NOP, AGC_R_RELEASE_MEM));
    write_guest_u32(command + 4, action | (cache_policy << 8));
    write_guest_u32(
        command + 8,
        gcr_control | (data_selection << 16) | (interrupt << 24),
    );
    write_guest_u32(command + 12, low_u32(destination_address));
    write_guest_u32(command + 16, (destination_address >> 32) as u32);
    write_guest_u32(command + 20, low_u32(data));
    write_guest_u32(command + 24, (data >> 32) as u32);
    write_guest_u32(command + 28, low_u32(stack_arguments[5]));
    command
}

pub(crate) fn agc_dcb_wait_reg_mem(arguments: [u64; 6], stack_arguments: [u64; 6]) -> u64 {
    let size = low_u32(arguments[1]) & 0xff;
    let compare_function = low_u32(arguments[2]) & 0xff;
    let operation = low_u32(arguments[3]) & 0xff;
    let cache_policy = low_u32(arguments[4]) & 0xff;
    if arguments[0] == 0 || size > 1 || compare_function > 7 || operation > 4 || cache_policy > 3 {
        return 0;
    }
    let standard_wait = matches!(operation, 2 | 3);
    let packet_dwords = if standard_wait {
        7
    } else if size == 0 {
        7
    } else {
        9
    };
    let Some(command) = agc_allocate_command_dwords(arguments[0], packet_dwords) else {
        return 0;
    };
    let address = arguments[5];
    let reference = stack_arguments[0];
    let mask = stack_arguments[1];
    let poll_cycles = low_u32(stack_arguments[2]);
    if standard_wait {
        write_guest_u32(command, agc_pm4(packet_dwords, AGC_IT_WAIT_REG_MEM, 0));
        write_guest_u32(command + 4, compare_function | ((operation & 1) << 8));
        write_guest_u32(command + 8, low_u32(address));
        write_guest_u32(command + 12, (address >> 32) as u32);
        write_guest_u32(command + 16, low_u32(reference));
        write_guest_u32(command + 20, low_u32(mask));
        write_guest_u32(command + 24, poll_cycles / 40);
    } else {
        let packet_register = if size == 0 {
            AGC_R_WAIT_MEM_32
        } else {
            AGC_R_WAIT_MEM_64
        };
        write_guest_u32(command, agc_pm4(packet_dwords, AGC_IT_NOP, packet_register));
        write_guest_u32(command + 4, low_u32(address));
        write_guest_u32(command + 8, (address >> 32) as u32);
        write_guest_u32(command + 12, low_u32(mask));
        if size == 0 {
            write_guest_u32(command + 16, low_u32(reference));
            write_guest_u32(
                command + 20,
                0x10 | compare_function
                    | ((operation & 0x3) << 8)
                    | ((operation & 0xc) << 4)
                    | (cache_policy << 25),
            );
            write_guest_u32(command + 24, (poll_cycles >> 4).min(u32::from(u16::MAX)));
        } else {
            write_guest_u32(command + 16, (mask >> 32) as u32);
            write_guest_u32(command + 20, low_u32(reference));
            write_guest_u32(command + 24, (reference >> 32) as u32);
            write_guest_u32(
                command + 28,
                0x10 | compare_function
                    | ((operation & 0x1) << 8)
                    | ((operation & 0x6) << 5)
                    | (cache_policy << 25),
            );
            write_guest_u32(command + 32, (poll_cycles >> 4).min(u32::from(u16::MAX)));
        }
    }
    command
}

fn agc_packet_identity(command: u64) -> Option<(u32, u32, u32)> {
    if command == 0 {
        return None;
    }
    let header = read_guest_u32(command);
    if header >> 30 != 3 {
        return None;
    }
    Some((
        ((header >> 8) & 0xff),
        (header >> 2) & 0x3f,
        ((header >> 16) & 0x3fff) + 2,
    ))
}

pub(crate) fn agc_dma_data_patch_destination(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let destination = arguments[1];
    let Some((opcode, register, length)) = agc_packet_identity(command) else {
        return u64::MAX;
    };
    if opcode != AGC_IT_NOP || register != AGC_R_DMA_DATA {
        return u64::MAX;
    }
    let destination_offset = if length == 7 { 4 } else { 16 };
    write_guest_u64(command + destination_offset, destination);
    0
}

pub(crate) fn agc_wait_reg_mem_patch_address(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let address = arguments[1];
    let Some((opcode, register, _)) = agc_packet_identity(command) else {
        return u64::MAX;
    };
    let offset = if opcode == AGC_IT_WAIT_REG_MEM {
        8
    } else if opcode == AGC_IT_NOP && matches!(register, AGC_R_WAIT_MEM_32 | AGC_R_WAIT_MEM_64) {
        4
    } else {
        return u64::MAX;
    };
    write_guest_u64(command + offset, address);
    0
}

pub(crate) fn agc_release_mem_patch_address(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let Some((opcode, register, _)) = agc_packet_identity(command) else {
        return u64::MAX;
    };
    if opcode != AGC_IT_NOP || register != AGC_R_RELEASE_MEM {
        return u64::MAX;
    }
    write_guest_u64(command + 12, arguments[1]);
    0
}

pub(crate) fn agc_dcb_acquire_mem(arguments: [u64; 6], stack_arguments: [u64; 6]) -> u64 {
    let command_buffer = arguments[0];
    let engine = low_u32(arguments[1]) & 0xff;
    let cb_db_operation = low_u32(arguments[2]);
    let gcr_control = low_u32(arguments[3]);
    let base_address = arguments[4];
    let size_bytes = arguments[5];
    let no_size = size_bytes == u64::MAX;
    if command_buffer == 0
        || engine > 1
        || (!no_size && (size_bytes & 0xff != 0 || size_bytes >> 40 != 0))
        || base_address & 0xff != 0
        || base_address >> 40 != 0
    {
        return 0;
    }
    let Some(command) = agc_allocate_command_dwords(command_buffer, 8) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(8, AGC_IT_NOP, AGC_R_ACQUIRE_MEM));
    write_guest_u32(command + 4, (engine << 31) | cb_db_operation);
    write_guest_u32(
        command + 8,
        if no_size { 0 } else { low_u32(size_bytes >> 8) },
    );
    write_guest_u32(command + 12, 0);
    write_guest_u32(command + 16, low_u32(base_address >> 8));
    write_guest_u32(command + 20, 0);
    write_guest_u32(command + 24, low_u32(stack_arguments[0]) / 40);
    write_guest_u32(command + 28, gcr_control);
    command
}
