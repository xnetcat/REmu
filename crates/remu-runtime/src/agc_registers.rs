use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use remu_shader::GuestMemoryReader;
use thiserror::Error;

use crate::{
    AGC_CB_COLOR_REGISTER_STRIDE, AGC_CB_COLOR0_ATTRIB, AGC_CB_COLOR0_ATTRIB2,
    AGC_CB_COLOR0_ATTRIB3, AGC_CB_COLOR0_BASE, AGC_CB_COLOR0_BASE_EXT, AGC_CB_COLOR0_INFO,
    AGC_CB_COLOR0_VIEW, AGC_CB_TARGET_MASK, AGC_COLOR_TARGET_COUNT, AGC_DB_DEPTH_CONTROL,
    AGC_DB_DEPTH_SIZE_XY, AGC_DB_RENDER_CONTROL, AGC_DB_Z_INFO, AGC_DB_Z_READ_BASE,
    AGC_DB_Z_READ_BASE_HI, AGC_DB_Z_WRITE_BASE, AGC_DB_Z_WRITE_BASE_HI, AGC_IT_NOP,
    AGC_IT_SET_CONTEXT_REG, AGC_IT_SET_CONTEXT_REG_INDIRECT, AGC_IT_SET_SH_REG,
    AGC_IT_SET_SH_REG_INDIRECT, AGC_IT_SET_UCONFIG_REG, AGC_IT_SET_UCONFIG_REG_INDEX,
    AGC_IT_SET_UCONFIG_REG_INDIRECT, AGC_R_CX_REGS_INDIRECT, AGC_R_SH_REGS_INDIRECT,
    AGC_R_UC_REGS_INDIRECT, AGC_SHADER_SPECIALS_OFFSET, AGC_SPI_PS_INPUT_CNTL_0,
    AGC_SPI_SHADER_PGM_HI_ES, AGC_SPI_SHADER_PGM_HI_PS, AGC_SPI_SHADER_PGM_LO_ES,
    AGC_SPI_SHADER_PGM_LO_PS, AGC_VGT_INDEX_TYPE, AGC_VGT_PRIMITIVE_TYPE, AgcGraphicsState,
    AgcShaderRegistry, ProcessGuestMemory, agc_read_mapped_u64, read_guest_u32, read_guest_u64,
    trace_agc_programs, trace_agc_registers, try_read_process_guest_u32,
};

pub(crate) fn agc_apply_submission_registers(
    packet_address: u64,
    length: u32,
    opcode: u32,
    register: u32,
    graphics: &mut AgcGraphicsState,
) -> Result<(), AgcSubmissionError> {
    if opcode == AGC_IT_SET_UCONFIG_REG_INDEX {
        if length != 3 {
            return Err(AgcSubmissionError::IndexedUconfigRegisterPacketLength { actual: length });
        }
        let register = read_guest_u32(packet_address + 4) & !0x7000_0000;
        let value = read_guest_u32(packet_address + 8);
        if trace_agc_registers() && register == AGC_VGT_PRIMITIVE_TYPE {
            eprintln!(
                "[REMu][AGC][REG] primitive={value:#x} source=indexed packet={packet_address:#x}"
            );
        }
        graphics.user_config_registers.insert(register, value);
        if register == AGC_VGT_INDEX_TYPE {
            graphics.index_size = value & 0x3;
        }
        return Ok(());
    }

    if matches!(
        opcode,
        AGC_IT_SET_CONTEXT_REG | AGC_IT_SET_SH_REG | AGC_IT_SET_UCONFIG_REG
    ) {
        if length < 3 {
            return Ok(());
        }
        // SET_UCONFIG_REG_INDEX masks selector bits out of the register dword; plain
        // SET_UCONFIG_REG / SET_SH_REG packets from UE/AGC use the same high bits
        // (e.g. 0x2000_0243 for VGT_INDEX_TYPE, 0x2000_0207 for COMPUTE_NUM_THREAD_X).
        // Without masking, index_size stays on the 16-bit CLEAR_STATE default while the
        // buffer is 32-bit (`[0,1,2]` → `[0,0,1,0,2,0]`), and compute threadgroup size
        // silently falls back to `[1,1,1]` via `.max(1)` when NUM_THREAD is missing.
        let first_register = if matches!(opcode, AGC_IT_SET_UCONFIG_REG | AGC_IT_SET_SH_REG) {
            read_guest_u32(packet_address + 4) & !0x7000_0000
        } else {
            read_guest_u32(packet_address + 4)
        };
        let destination = match opcode {
            AGC_IT_SET_CONTEXT_REG => &mut graphics.context_registers,
            AGC_IT_SET_SH_REG => &mut graphics.shader_registers,
            _ => &mut graphics.user_config_registers,
        };
        let mut index_type_value = None;
        for index in 0..length - 2 {
            let register = first_register + index;
            let value = read_guest_u32(packet_address + u64::from(index + 2) * 4);
            if trace_agc_registers()
                && opcode == AGC_IT_SET_UCONFIG_REG
                && register == AGC_VGT_PRIMITIVE_TYPE
            {
                eprintln!(
                    "[REMu][AGC][REG] primitive={value:#x} source=direct packet={packet_address:#x}"
                );
            }
            destination.insert(register, value);
            if opcode == AGC_IT_SET_UCONFIG_REG && register == AGC_VGT_INDEX_TYPE {
                index_type_value = Some(value);
            }
        }
        if let Some(value) = index_type_value {
            graphics.index_size = value & 0x3;
        }
        if trace_agc_registers()
            && opcode == AGC_IT_SET_CONTEXT_REG
            && first_register < AGC_SPI_PS_INPUT_CNTL_0 + 32
            && first_register.saturating_add(length - 2) > AGC_SPI_PS_INPUT_CNTL_0
        {
            let first = first_register.max(AGC_SPI_PS_INPUT_CNTL_0);
            let last = first_register
                .saturating_add(length - 2)
                .min(AGC_SPI_PS_INPUT_CNTL_0 + 32);
            let entries = (first..last)
                .filter_map(|offset| {
                    destination
                        .get(&offset)
                        .copied()
                        .map(|value| (offset, value))
                })
                .collect::<Vec<_>>();
            eprintln!(
                "[REMu][AGC][REG] pixel inputs direct packet={packet_address:#x} first={first_register:#x} count={} entries={entries:x?}",
                length - 2,
            );
        }
        if (trace_agc_registers() || trace_agc_programs())
            && opcode == AGC_IT_SET_SH_REG
            && [
                AGC_SPI_SHADER_PGM_LO_PS,
                AGC_SPI_SHADER_PGM_HI_PS,
                AGC_SPI_SHADER_PGM_LO_ES,
                AGC_SPI_SHADER_PGM_HI_ES,
            ]
            .into_iter()
            .any(|register| {
                register >= first_register && register < first_register.saturating_add(length - 2)
            })
        {
            eprintln!(
                "[REMu][AGC][REG] shader programs direct packet={packet_address:#x} first={first_register:#x} count={} ps={:#x}/{:#x} es={:#x}/{:#x}",
                length - 2,
                graphics
                    .shader_registers
                    .get(&AGC_SPI_SHADER_PGM_LO_PS)
                    .copied()
                    .unwrap_or_default(),
                graphics
                    .shader_registers
                    .get(&AGC_SPI_SHADER_PGM_HI_PS)
                    .copied()
                    .unwrap_or_default(),
                graphics
                    .shader_registers
                    .get(&AGC_SPI_SHADER_PGM_LO_ES)
                    .copied()
                    .unwrap_or_default(),
                graphics
                    .shader_registers
                    .get(&AGC_SPI_SHADER_PGM_HI_ES)
                    .copied()
                    .unwrap_or_default(),
            );
        }
        if (trace_agc_registers() || trace_agc_programs())
            && opcode == AGC_IT_SET_SH_REG
            && first_register <= 0x013
            && first_register.saturating_add(length - 2) > 0x00c
        {
            let pixel_user_data = std::array::from_fn::<_, 8, _>(|index| {
                graphics
                    .shader_registers
                    .get(&(0x00c + u32::try_from(index).expect("pixel scalar index fits")))
                    .copied()
            });
            eprintln!(
                "[REMu][AGC][REG] pixel user-data direct packet={packet_address:#x} first={first_register:#x} count={} values={pixel_user_data:x?}",
                length - 2,
            );
        }
        return Ok(());
    }

    if matches!(
        opcode,
        AGC_IT_SET_CONTEXT_REG_INDIRECT
            | AGC_IT_SET_SH_REG_INDIRECT
            | AGC_IT_SET_UCONFIG_REG_INDIRECT
    ) {
        if length != 5 {
            return Err(AgcSubmissionError::NativeIndirectRegisterPacketLength {
                opcode,
                actual: length,
            });
        }
        let registers_address = u64::from(read_guest_u32(packet_address + 4) & !0x3)
            | (u64::from(read_guest_u32(packet_address + 8)) << 32);
        let register_count = read_guest_u32(packet_address + 16) & 0x3fff;
        let (destination, normalize_selector) = match opcode {
            AGC_IT_SET_CONTEXT_REG_INDIRECT => (&mut graphics.context_registers, false),
            AGC_IT_SET_SH_REG_INDIRECT => (&mut graphics.shader_registers, true),
            _ => (&mut graphics.user_config_registers, true),
        };
        if let Some(value) = agc_apply_indirect_register_table(
            opcode,
            registers_address,
            register_count,
            normalize_selector,
            destination,
        )? {
            graphics.index_size = value & 0x3;
        }
        return Ok(());
    }

    if opcode != AGC_IT_NOP
        || !matches!(
            register,
            AGC_R_SH_REGS_INDIRECT | AGC_R_CX_REGS_INDIRECT | AGC_R_UC_REGS_INDIRECT
        )
        || length < 4
    {
        return Ok(());
    }
    let register_count = read_guest_u32(packet_address + 4).min(4_096);
    let registers_address = read_guest_u64(packet_address + 8);
    let (destination, normalize_selector) = match register {
        AGC_R_CX_REGS_INDIRECT => (&mut graphics.context_registers, false),
        // HLE SH/UC tables use the same 0x2000_/0x4000_ selector high bits as the native
        // indirect packets; context tables keep bare Gen5 offsets.
        AGC_R_SH_REGS_INDIRECT => (&mut graphics.shader_registers, true),
        _ => (&mut graphics.user_config_registers, true),
    };
    if let Some(value) = agc_apply_indirect_register_table(
        opcode,
        registers_address,
        register_count,
        normalize_selector,
        destination,
    )? {
        graphics.index_size = value & 0x3;
    }
    Ok(())
}

/// Applies an indirect register table. When a user-config write targets `VGT_INDEX_TYPE`, returns
/// that raw register value so callers can mirror it into `graphics.index_size`.
fn agc_apply_indirect_register_table(
    opcode: u32,
    registers_address: u64,
    register_count: u32,
    normalize_selector: bool,
    destination: &mut BTreeMap<u32, u32>,
) -> Result<Option<u32>, AgcSubmissionError> {
    if register_count == 0 {
        return Ok(None);
    }
    if registers_address == 0 {
        return Err(AgcSubmissionError::NullIndirectRegisterTable {
            opcode,
            register_count,
        });
    }
    let byte_count =
        usize::try_from(register_count).expect("14-bit indirect register count fits usize") * 8;
    let mut bytes = vec![0_u8; byte_count];
    if registers_address
        .checked_add(u64::try_from(byte_count).expect("register table length fits u64"))
        .is_none()
        || !ProcessGuestMemory.read(registers_address, &mut bytes)
    {
        return Err(AgcSubmissionError::UnreadableIndirectRegister(
            registers_address,
        ));
    }
    let mut traced_shader_registers = Vec::new();
    let mut traced_raw_registers = Vec::new();
    let mut traced_pixel_inputs = Vec::new();
    let mut index_type_value = None;
    for pair in bytes.chunks_exact(8) {
        let raw_offset = u32::from_le_bytes(pair[..4].try_into().expect("register offset dword"));
        if raw_offset == u32::MAX {
            continue;
        }
        let offset = if normalize_selector {
            raw_offset & !0x7000_0000
        } else {
            raw_offset
        };
        let value = u32::from_le_bytes(pair[4..].try_into().expect("register value dword"));
        if trace_agc_registers() && offset == AGC_VGT_PRIMITIVE_TYPE {
            eprintln!(
                "[REMu][AGC][REG] primitive={value:#x} source=indirect opcode={opcode:#x} table={registers_address:#x}"
            );
        }
        if (trace_agc_registers() || trace_agc_programs())
            && opcode == AGC_IT_SET_SH_REG_INDIRECT
            && matches!(
                offset,
                AGC_SPI_SHADER_PGM_LO_PS
                    | AGC_SPI_SHADER_PGM_HI_PS
                    | AGC_SPI_SHADER_PGM_LO_ES
                    | AGC_SPI_SHADER_PGM_HI_ES
            )
        {
            traced_shader_registers.push((offset, value));
        }
        if trace_agc_programs() && opcode == AGC_IT_SET_SH_REG_INDIRECT && register_count <= 32 {
            traced_raw_registers.push((raw_offset, offset, value));
        }
        if trace_agc_registers()
            && opcode == AGC_IT_SET_CONTEXT_REG_INDIRECT
            && (AGC_SPI_PS_INPUT_CNTL_0..AGC_SPI_PS_INPUT_CNTL_0 + 32).contains(&offset)
        {
            traced_pixel_inputs.push((raw_offset, offset, value));
        }
        destination.insert(offset, value);
        if offset == AGC_VGT_INDEX_TYPE {
            index_type_value = Some(value);
        }
    }
    if !traced_shader_registers.is_empty() {
        eprintln!(
            "[REMu][AGC][REG] shader indirect table={registers_address:#x} count={register_count} entries={traced_shader_registers:x?}"
        );
    }
    if !traced_raw_registers.is_empty() {
        eprintln!(
            "[REMu][AGC][REG] shader indirect raw table={registers_address:#x} count={register_count} entries={traced_raw_registers:x?}"
        );
    }
    if !traced_pixel_inputs.is_empty() {
        eprintln!(
            "[REMu][AGC][REG] pixel inputs indirect table={registers_address:#x} count={register_count} entries={traced_pixel_inputs:x?}"
        );
    }
    Ok(index_type_value)
}

pub(crate) fn agc_program_address(registers: &BTreeMap<u32, u32>, low: u32, high: u32) -> u64 {
    let Some((&low, &high)) = registers.get(&low).zip(registers.get(&high)) else {
        return 0;
    };
    (u64::from(high & 0xff) << 40) | (u64::from(low) << 8)
}

#[derive(Debug)]
pub(crate) enum AgcColorTargetState {
    Absent,
    Disabled,
    Programmed(Vec<remu_gpu::GpuColorTarget>),
    Malformed { slot: u32 },
}

pub(crate) fn agc_color_target_state(registers: &BTreeMap<u32, u32>) -> AgcColorTargetState {
    let target_mask = registers.get(&AGC_CB_TARGET_MASK).copied();
    let mut targets = Vec::new();
    for slot in 0..AGC_COLOR_TARGET_COUNT {
        if target_mask.is_some_and(|mask| (mask >> (slot * 4)).trailing_zeros() >= 4) {
            continue;
        }
        let primary = slot * AGC_CB_COLOR_REGISTER_STRIDE;
        let base = registers.get(&(AGC_CB_COLOR0_BASE + primary)).copied();
        let base_ext = registers.get(&(AGC_CB_COLOR0_BASE_EXT + slot)).copied();
        let address = base
            .zip(base_ext)
            .map(|(low, high)| (u64::from(high & 0xff) << 40) | (u64::from(low) << 8));
        let explicitly_enabled =
            target_mask.is_some_and(|mask| (mask >> (slot * 4)).trailing_zeros() < 4);
        if address.is_none_or(|address| address == 0) {
            if explicitly_enabled {
                return AgcColorTargetState::Malformed { slot };
            }
            continue;
        }
        let Some(view) = registers.get(&(AGC_CB_COLOR0_VIEW + primary)).copied() else {
            return AgcColorTargetState::Malformed { slot };
        };
        let Some(info) = registers.get(&(AGC_CB_COLOR0_INFO + primary)).copied() else {
            return AgcColorTargetState::Malformed { slot };
        };
        let Some(attrib) = registers.get(&(AGC_CB_COLOR0_ATTRIB + primary)).copied() else {
            return AgcColorTargetState::Malformed { slot };
        };
        let Some(attrib2) = registers.get(&(AGC_CB_COLOR0_ATTRIB2 + slot)).copied() else {
            return AgcColorTargetState::Malformed { slot };
        };
        let Some(attrib3) = registers.get(&(AGC_CB_COLOR0_ATTRIB3 + slot)).copied() else {
            return AgcColorTargetState::Malformed { slot };
        };
        targets.push(remu_gpu::GpuColorTarget {
            slot,
            address: address.expect("nonzero color target address was checked"),
            view,
            info,
            attrib,
            attrib2,
            attrib3,
        });
    }
    if targets.is_empty() {
        if target_mask.is_some() {
            AgcColorTargetState::Disabled
        } else {
            AgcColorTargetState::Absent
        }
    } else {
        AgcColorTargetState::Programmed(targets)
    }
}

pub(crate) fn agc_has_active_depth_state(registers: &BTreeMap<u32, u32>) -> bool {
    let enabled = registers
        .get(&AGC_DB_DEPTH_CONTROL)
        .is_some_and(|control| control & 0x6 != 0)
        || registers
            .get(&AGC_DB_RENDER_CONTROL)
            .is_some_and(|control| control & 1 != 0);
    if !enabled
        || registers
            .get(&AGC_DB_Z_INFO)
            .is_none_or(|info| info.trailing_zeros() >= 2)
        || !registers.contains_key(&AGC_DB_DEPTH_SIZE_XY)
    {
        return false;
    }
    let read_address = (u64::from(
        registers
            .get(&AGC_DB_Z_READ_BASE_HI)
            .copied()
            .unwrap_or_default()
            & 0xff,
    ) << 40)
        | (u64::from(
            registers
                .get(&AGC_DB_Z_READ_BASE)
                .copied()
                .unwrap_or_default(),
        ) << 8);
    let write_address = (u64::from(
        registers
            .get(&AGC_DB_Z_WRITE_BASE_HI)
            .copied()
            .unwrap_or_default()
            & 0xff,
    ) << 40)
        | (u64::from(
            registers
                .get(&AGC_DB_Z_WRITE_BASE)
                .copied()
                .unwrap_or_default(),
        ) << 8);
    read_address != 0 || write_address != 0
}

pub(crate) fn agc_register_snapshot(registers: &BTreeMap<u32, u32>) -> Vec<remu_gpu::GpuRegister> {
    registers
        .iter()
        .map(|(&offset, &value)| remu_gpu::GpuRegister { offset, value })
        .collect()
}

pub(crate) fn agc_shader_prefix_snapshot(address: u64) -> [u32; 4] {
    if address == 0 {
        return [0; 4];
    }
    std::array::from_fn(|index| {
        read_guest_u32(address + u64::try_from(index).expect("shader prefix index fits") * 4)
    })
}

pub(crate) fn agc_shader_code_fingerprint(
    memory: &impl GuestMemoryReader,
    address: u64,
    code_size: u32,
) -> Option<u64> {
    let mut bytes = vec![0_u8; usize::try_from(code_size).ok()?];
    if bytes.is_empty() || !memory.read(address, &mut bytes) {
        return None;
    }
    Some(bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    }))
}

const SHADER_DECODE_FAILURE_WINDOW_BYTES: u32 = 128;
const SHADER_DECODE_FAILURE_PREFIX_BYTES: u32 = 64;

#[derive(Debug, Eq, PartialEq)]
struct ShaderDecodeFailureWindow {
    offset: u32,
    bytes: Vec<u8>,
}

fn shader_decode_error_pc(error: &remu_shader::DecodeError) -> Option<u32> {
    match error {
        remu_shader::DecodeError::Truncated { pc, .. }
        | remu_shader::DecodeError::UnknownEncoding { pc, .. }
        | remu_shader::DecodeError::UnknownOpcode { pc, .. }
        | remu_shader::DecodeError::FlatSegment { pc, .. }
        | remu_shader::DecodeError::AddressOverflow { pc }
        | remu_shader::DecodeError::MemoryFault { pc, .. } => Some(*pc),
        remu_shader::DecodeError::Empty
        | remu_shader::DecodeError::Unterminated { .. }
        | remu_shader::DecodeError::InvalidCodeSize { .. }
        | remu_shader::DecodeError::ProgramCounterOverflow => None,
    }
}

fn shader_decode_context_pc(
    error: &remu_shader::DecodeError,
    code_size: Option<u32>,
) -> Option<u32> {
    shader_decode_error_pc(error).or_else(|| {
        matches!(error, remu_shader::DecodeError::Unterminated { .. })
            .then(|| code_size?.checked_sub(4))
            .flatten()
    })
}

fn shader_decode_failure_window(
    memory: &impl GuestMemoryReader,
    address: u64,
    code_size: Option<u32>,
    error: &remu_shader::DecodeError,
) -> Option<ShaderDecodeFailureWindow> {
    let pc = shader_decode_context_pc(error, code_size)?;
    let offset = pc.saturating_sub(SHADER_DECODE_FAILURE_PREFIX_BYTES) & !3;
    let requested_end = offset.saturating_add(SHADER_DECODE_FAILURE_WINDOW_BYTES);
    let end = code_size.map_or(requested_end, |size| requested_end.min(size));
    if end <= offset {
        return None;
    }

    let mut bytes = Vec::with_capacity(usize::try_from(end - offset).ok()?);
    let mut word_offset = offset;
    while word_offset.saturating_add(4) <= end {
        let word_address = address.checked_add(u64::from(word_offset))?;
        let mut word = [0_u8; 4];
        if !memory.read(word_address, &mut word) {
            break;
        }
        bytes.extend_from_slice(&word);
        word_offset = word_offset.saturating_add(4);
    }
    (!bytes.is_empty()).then_some(ShaderDecodeFailureWindow { offset, bytes })
}

fn trace_shader_decode_failure(
    memory: &impl GuestMemoryReader,
    address: u64,
    header_address: u64,
    code_size: Option<u32>,
    registration_code_fingerprint: Option<u64>,
    error: &remu_shader::DecodeError,
) {
    let dump_directory = std::env::var_os("REMU_DUMP_MSL_DIR").map(PathBuf::from);
    if !trace_agc_programs() && dump_directory.is_none() {
        return;
    }
    let Some(window) = shader_decode_failure_window(memory, address, code_size, error) else {
        if trace_agc_programs() {
            eprintln!(
                "[REMu][AGC][SHADER][DECODE-CONTEXT] code={address:#x} header={header_address:#x} registered_code_size={code_size:x?} window=unavailable"
            );
        }
        return;
    };

    if trace_agc_programs() {
        let current_code_fingerprint =
            code_size.and_then(|size| agc_shader_code_fingerprint(memory, address, size));
        let fingerprint_changed = registration_code_fingerprint
            .zip(current_code_fingerprint)
            .map(|(registered, current)| registered != current);
        let words = window
            .bytes
            .chunks_exact(4)
            .enumerate()
            .map(|(index, bytes)| {
                let offset = window.offset
                    + u32::try_from(index).expect("bounded shader window index fits u32") * 4;
                (
                    offset,
                    u32::from_le_bytes(bytes.try_into().expect("dword chunk has four bytes")),
                )
            })
            .collect::<Vec<_>>();
        eprintln!(
            "[REMu][AGC][SHADER][DECODE-CONTEXT] code={address:#x} header={header_address:#x} registered_code_size={code_size:x?} registration_fingerprint={registration_code_fingerprint:x?} current_fingerprint={current_code_fingerprint:x?} fingerprint_changed={fingerprint_changed:?} window_offset={:#x} window_bytes={} words={words:x?}",
            window.offset,
            window.bytes.len(),
        );
    }

    let Some(directory) = dump_directory else {
        return;
    };
    if let Err(error) = std::fs::create_dir_all(&directory) {
        eprintln!("[REMu][AGC][SHADER] could not create decode failure dump directory: {error}");
        return;
    }
    let pc = shader_decode_context_pc(error, code_size).unwrap_or_default();
    let stem = format!("{address:016x}-failed-pc-{pc:08x}");
    if let Err(error) = std::fs::write(directory.join(format!("{stem}-window.bin")), &window.bytes)
    {
        eprintln!("[REMu][AGC][SHADER] could not write decode failure window: {error}");
        return;
    }
    let metadata = format!(
        "code={address:#x}\nheader={header_address:#x}\nregistered_code_size={code_size:x?}\nerror_pc={pc:#x}\nwindow_offset={:#x}\nwindow_address={:#x}\nwindow_bytes={}\nerror={error}\n",
        window.offset,
        address.saturating_add(u64::from(window.offset)),
        window.bytes.len(),
    );
    if let Err(error) = std::fs::write(directory.join(format!("{stem}-window.txt")), metadata) {
        eprintln!("[REMu][AGC][SHADER] could not write decode failure metadata: {error}");
    }
}

pub(crate) fn agc_shader_snapshot(
    shaders: &mut AgcShaderRegistry,
    address: u64,
) -> Option<remu_gpu::ShaderSnapshot> {
    if address == 0 || shaders.decode_failures.contains(&address) {
        return None;
    }
    if let Some(snapshot) = shaders.snapshots.get(&address) {
        return Some(snapshot.clone());
    }
    let fused_continuation = match shaders.fused_continuations.get(&address) {
        None => None,
        Some(continuations) if continuations.len() == 1 => continuations.first().copied(),
        Some(continuations) => {
            shaders.decode_failures.insert(address);
            eprintln!(
                "[REMu][AGC][SHADER] refusing ambiguous fused entry={address:#x} continuations={continuations:x?}"
            );
            return None;
        }
    };
    if let Some(continuation) = fused_continuation
        && (shaders.headers.get(&address).copied() != Some(continuation.front_header_address)
            || shaders
                .headers
                .get(&continuation.back_program_address)
                .copied()
                != Some(continuation.back_header_address)
            || shaders
                .code_sizes
                .get(&continuation.back_program_address)
                .copied()
                != Some(continuation.back_code_size))
    {
        shaders.decode_failures.insert(address);
        eprintln!(
            "[REMu][AGC][SHADER] refusing stale fused entry={address:#x} continuation={continuation:x?}"
        );
        return None;
    }
    let memory = ProcessGuestMemory;
    let programs = match shaders.code_sizes.get(&address).copied() {
        Some(code_size) => usize::try_from(code_size)
            .map_err(|_| remu_shader::DecodeError::InvalidCodeSize { bytes: usize::MAX })
            .and_then(|code_size| {
                let leading = remu_shader::read_program_bounded(&memory, address, code_size)?;
                Ok(
                    remu_shader::read_program_segments_bounded(&memory, address, code_size)
                        .unwrap_or_else(|_| vec![leading]),
                )
            }),
        None => remu_shader::read_program(&memory, address).map(|program| vec![program]),
    };
    let mut programs = match programs {
        Ok(programs) => programs,
        Err(error) => {
            shaders.decode_failures.insert(address);
            eprintln!("[REMu][AGC][SHADER] decode failed at {address:#x}: {error}");
            trace_shader_decode_failure(
                &memory,
                address,
                shaders.headers.get(&address).copied().unwrap_or_default(),
                shaders.code_sizes.get(&address).copied(),
                shaders
                    .registration_code_fingerprints
                    .get(&address)
                    .copied(),
                &error,
            );
            return None;
        }
    };
    let continuation_program = if let Some(continuation) = fused_continuation {
        let back_program = usize::try_from(continuation.back_code_size)
            .map_err(|_| remu_shader::DecodeError::InvalidCodeSize { bytes: usize::MAX })
            .and_then(|code_size| {
                remu_shader::read_program_bounded(
                    &memory,
                    continuation.back_program_address,
                    code_size,
                )
            });
        match back_program {
            // Only the exact leading entry point selected by this fuse operation is a transfer
            // target. Trailing entry points in the back allocation remain unrelated.
            Ok(back_program) => Some(back_program),
            Err(error) => {
                shaders.decode_failures.insert(address);
                eprintln!(
                    "[REMu][AGC][SHADER] fused continuation decode failed entry={address:#x} continuation={:#x}: {error}",
                    continuation.back_program_address,
                );
                trace_shader_decode_failure(
                    &memory,
                    continuation.back_program_address,
                    continuation.back_header_address,
                    Some(continuation.back_code_size),
                    shaders
                        .registration_code_fingerprints
                        .get(&continuation.back_program_address)
                        .copied(),
                    &error,
                );
                return None;
            }
        }
    } else {
        None
    };
    let leading_program = programs.remove(0);
    let program = if let Some(back_program) = continuation_program {
        let linked_program = match remu_shader::link_program_continuation(
            &leading_program,
            &back_program,
        ) {
            Ok(program) => program,
            Err(error) => {
                shaders.decode_failures.insert(address);
                eprintln!(
                    "[REMu][AGC][SHADER] fused continuation link failed entry={address:#x} continuation={:#x}: {error}",
                    back_program.address,
                );
                return None;
            }
        };
        // Keep the raw continuation after the front allocation's independent supplemental
        // segments for diagnostics. Execution and cache identity use only the linked composite.
        programs.push(back_program);
        Arc::new(linked_program)
    } else {
        Arc::new(leading_program)
    };
    let supplemental_programs: Arc<[remu_shader::Program]> = programs.into();
    // AGC's fused object is the back descriptor with only its program registers repointed at the
    // front entry point. User-data metadata and output semantics therefore belong to the back
    // header even though `program` begins at the front code address.
    let header_address = fused_continuation.map_or_else(
        || shaders.headers.get(&address).copied().unwrap_or_default(),
        |continuation| continuation.back_header_address,
    );
    let metadata = (header_address != 0)
        .then(|| remu_shader::read_shader_metadata(&memory, header_address))
        .transpose()
        .map_err(|error| {
            eprintln!("[REMu][AGC][SHADER] metadata failed at header {header_address:#x}: {error}");
        })
        .ok()
        .flatten()
        .map(Arc::new);
    if !supplemental_programs.is_empty() {
        static LOGGED_MERGED_PROGRAMS: AtomicUsize = AtomicUsize::new(0);
        let sequence = LOGGED_MERGED_PROGRAMS.fetch_add(1, Ordering::Relaxed) + 1;
        if sequence <= 64 || sequence.is_power_of_two() {
            let segments = supplemental_programs
                .iter()
                .map(|program| {
                    (
                        program.address,
                        program.instructions.len(),
                        program
                            .instructions
                            .iter()
                            .filter_map(|instruction| {
                                let remu_shader::InstructionControl::Export(export) =
                                    instruction.control.as_ref()?
                                else {
                                    return None;
                                };
                                Some((export.target, export.enable_mask, export.done))
                            })
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>();
            eprintln!(
                "[REMu][AGC][MERGED-PROGRAM] sequence={sequence} code={address:#x} leading_instructions={} supplemental={segments:x?}",
                program.instructions.len(),
            );
        }
    }
    if trace_agc_programs() {
        let specials = (header_address != 0)
            .then(|| agc_read_mapped_u64(header_address + AGC_SHADER_SPECIALS_OFFSET))
            .flatten()
            .unwrap_or_default();
        let user_data_range = (specials != 0)
            .then(|| {
                try_read_process_guest_u32(specials + 0x14)
                    .map(|packed| (packed & u32::from(u16::MAX), packed >> u16::BITS))
            })
            .flatten();
        let prefix = program
            .instructions
            .iter()
            .take(12)
            .map(|instruction| {
                (
                    instruction.pc,
                    instruction.opcode,
                    instruction.words.clone(),
                    instruction.sources.clone(),
                    instruction.destinations.clone(),
                )
            })
            .collect::<Vec<_>>();
        let supplemental = supplemental_programs
            .iter()
            .map(|program| {
                (
                    program.address,
                    program.instructions.len(),
                    program
                        .instructions
                        .iter()
                        .filter_map(|instruction| {
                            let remu_shader::InstructionControl::Export(export) =
                                instruction.control.as_ref()?
                            else {
                                return None;
                            };
                            Some((export.target, export.enable_mask, export.done))
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        eprintln!(
            "[REMu][AGC][SHADER] decoded code={address:#x} header={header_address:#x} fused_continuation={fused_continuation:x?} user_data_range={user_data_range:x?} metadata={metadata:x?} supplemental={supplemental:x?} prefix={prefix:x?}"
        );
    }
    let snapshot = remu_gpu::ShaderSnapshot {
        header_address,
        program,
        supplemental_programs,
        metadata,
    };
    shaders.snapshots.insert(address, snapshot.clone());
    Some(snapshot)
}

#[derive(Debug, Error)]
pub(crate) enum AgcComputeDispatchError {
    #[error(transparent)]
    Decode(#[from] remu_gpu::ComputeDispatchDecodeError),
    #[error("base-relative DISPATCH_INDIRECT has no active compute indirect-argument base")]
    MissingIndirectBase,
    #[error("compute indirect dimensions address overflows the guest address space")]
    IndirectAddressOverflow,
    #[error("compute indirect dimensions at 0x{0:016X} are not readable")]
    UnreadableIndirectDimensions(u64),
    #[error("compute program registers COMPUTE_PGM_LO/HI are incomplete")]
    MissingProgramAddress,
    #[error("compute program address is null")]
    NullProgramAddress,
    #[error("compute shader at 0x{0:016X} could not be decoded")]
    ShaderDecode(u64),
    #[error("compute shader at 0x{address:016X} could not be evaluated: {reason}")]
    ShaderEvaluation { address: u64, reason: String },
}

#[derive(Debug, Error)]
pub(crate) enum AgcSubmissionError {
    #[error(
        "DMA packet-time snapshot budget exceeded: requested {requested_bytes} bytes, limit {limit_bytes} bytes"
    )]
    MemorySnapshotBudget {
        requested_bytes: usize,
        limit_bytes: usize,
    },
    #[error("unsupported PM4 packet type {packet_type} at dword {dword} (header 0x{header:08X})")]
    UnsupportedPacketType {
        dword: u32,
        packet_type: u32,
        header: u32,
    },
    #[error(
        "PM4 packet at dword {dword} ({address:#x}, header {header:#010x}) declares {length} dwords with {remaining} remaining"
    )]
    InvalidPacketLength {
        dword: u32,
        address: u64,
        header: u32,
        length: u32,
        remaining: u32,
    },
    #[error("indirect command-buffer nesting exceeds {maximum} levels")]
    IndirectBufferNesting { maximum: u32 },
    #[error("INDIRECT_BUFFER packet has {actual} dwords, expected 4 or 14")]
    IndirectBufferPacketLength { actual: u32 },
    #[error(
        "INDIRECT_BUFFER names unreadable command range 0x{address:016X} with {dword_count} dwords"
    )]
    InvalidIndirectBuffer { address: u64, dword_count: u32 },
    #[error("SET_PREDICATION packet has {actual} dwords, expected at least 3")]
    PredicationPacketLength { actual: u32 },
    #[error("SET_PREDICATION operation {0} is not implemented")]
    UnsupportedPredicationOperation(u32),
    #[error("SET_PREDICATION names unreadable value at 0x{0:016X}")]
    UnreadablePredicationValue(u64),
    #[error("COND_EXEC packet has {actual} dwords, expected 5")]
    CondExecPacketLength { actual: u32 },
    #[error("COND_EXEC control has unsupported value 0x{0:08X}")]
    UnsupportedCondExecControl(u32),
    #[error("COND_EXEC names unreadable value at 0x{0:016X}")]
    UnreadableCondExecValue(u64),
    #[error(
        "COND_EXEC at 0x{packet_address:016X} skips {skip_dwords} dwords beyond the command stream"
    )]
    InvalidCondExecRange {
        packet_address: u64,
        skip_dwords: u32,
    },
    #[error("Gen5 branch uses unsupported mode {0}")]
    UnsupportedBranchMode(u32),
    #[error("Gen5 branch uses unsupported comparison function {0}")]
    UnsupportedBranchComparison(u32),
    #[error("Gen5 branch names unreadable comparison value at 0x{0:016X}")]
    UnreadableBranchValue(u64),
    #[error("SET_BASE packet has {actual} dwords, expected 4")]
    SetBasePacketLength { actual: u32 },
    #[error("SET_BASE selects unsupported base index {0}")]
    UnsupportedBaseIndex(u32),
    #[error("SET_BASE selects unsupported shader type {0}")]
    UnsupportedSetBaseShaderType(u32),
    #[error("indirect graphics draw has no active graphics indirect-argument base")]
    MissingGraphicsIndirectBase,
    #[error(
        "indirect graphics draw packet opcode 0x{opcode:02X} has {actual} dwords, expected {expected}"
    )]
    IndirectDrawPacketLength {
        opcode: u32,
        actual: u32,
        expected: u32,
    },
    #[error("indirect graphics draw argument address overflows the guest address space")]
    IndirectDrawAddressOverflow,
    #[error("indirect graphics draw arguments at 0x{0:016X} are not readable")]
    UnreadableIndirectDrawArguments(u64),
    #[error("indirect graphics draw packet opcode 0x{0:02X} is not implemented")]
    UnsupportedIndirectDrawPacket(u32),
    #[error("CLEAR_STATE packet has {actual} dwords, expected 2")]
    ClearStatePacketLength { actual: u32 },
    #[error("CLEAR_STATE control has unsupported bits 0x{0:08X}")]
    UnsupportedClearStateControl(u32),
    #[error("PFP_SYNC_ME packet has {actual} dwords, expected 2")]
    PfpSyncMePacketLength { actual: u32 },
    #[error("EVENT_WRITE packet has {actual} dwords, expected at least 2")]
    EventWritePacketLength { actual: u32 },
    #[error("partial-flush EVENT_WRITE control has unsupported bits 0x{0:08X}")]
    UnsupportedEventWriteControl(u32),
    #[error(
        "partial-flush EVENT_WRITE type 0x{event_type:02X} uses event index {actual}, expected 4"
    )]
    PartialFlushEventIndex { event_type: u32, actual: u32 },
    #[error(
        "EVENT_WRITE type 0x{event_type:02X} index {event_index} with {length} dwords is not implemented"
    )]
    UnsupportedEventWrite {
        event_type: u32,
        event_index: u32,
        length: u32,
    },
    #[error("PIXEL_PIPE_STAT_DUMP names invalid qword address 0x{0:016X}")]
    InvalidPixelPipeStatAddress(u64),
    #[error("native indirect-register opcode 0x{opcode:02X} has {actual} dwords, expected 5")]
    NativeIndirectRegisterPacketLength { opcode: u32, actual: u32 },
    #[error("indexed SET_UCONFIG_REG packet has {actual} dwords, expected 3")]
    IndexedUconfigRegisterPacketLength { actual: u32 },
    #[error(
        "indirect-register opcode 0x{opcode:02X} has null table address with {register_count} entries"
    )]
    NullIndirectRegisterTable { opcode: u32, register_count: u32 },
    #[error("indirect register entry at 0x{0:016X} is not readable")]
    UnreadableIndirectRegister(u64),
    #[error("ACB descriptor has address 0x{address:016X} and {dword_count} dwords")]
    InvalidAcbDescriptor { address: u64, dword_count: u32 },
    #[error("compute dispatch at 0x{packet_address:016X}: {source}")]
    Compute {
        packet_address: u64,
        #[source]
        source: AgcComputeDispatchError,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SliceMemory {
        base: u64,
        bytes: Vec<u8>,
    }

    impl GuestMemoryReader for SliceMemory {
        fn read(&self, address: u64, output: &mut [u8]) -> bool {
            let Some(offset) = address.checked_sub(self.base) else {
                return false;
            };
            let Ok(offset) = usize::try_from(offset) else {
                return false;
            };
            let Some(end) = offset.checked_add(output.len()) else {
                return false;
            };
            let Some(source) = self.bytes.get(offset..end) else {
                return false;
            };
            output.copy_from_slice(source);
            true
        }
    }

    #[test]
    fn shader_decode_failure_window_includes_branch_tail_and_obeys_registered_extent() {
        let base = 0x0080_db60_0000;
        let code_size = 0x16d0_u32;
        let mut bytes = vec![0_u8; usize::try_from(code_size).expect("test code size fits")];
        bytes[0x16a8..0x16ac].copy_from_slice(&0xdead_beef_u32.to_le_bytes());
        bytes[0x16b4..0x16b8].copy_from_slice(&0x0000_023e_u32.to_le_bytes());
        let memory = SliceMemory { base, bytes };
        let error = remu_shader::DecodeError::UnknownOpcode {
            pc: 0x16b4,
            encoding: remu_shader::ShaderEncoding::Vop2,
            opcode: 0,
            word: 0x0000_023e,
        };

        let window = shader_decode_failure_window(&memory, base, Some(code_size), &error)
            .expect("failure pc has a readable diagnostic window");

        assert_eq!(window.offset, 0x1674);
        assert_eq!(window.bytes.len(), 0x5c);
        assert_eq!(&window.bytes[0x34..0x38], &0xdead_beef_u32.to_le_bytes());
        assert_eq!(&window.bytes[0x40..0x44], &0x0000_023e_u32.to_le_bytes());
        assert_eq!(window.bytes.last(), Some(&0));
    }

    #[test]
    fn unterminated_shader_decode_failure_window_captures_registered_tail() {
        let base = 0x0080_4eaa_0000;
        let code_size = 0xb90_u32;
        let bytes = vec![0xff_u8; usize::try_from(code_size).expect("test code size fits")];
        let memory = SliceMemory { base, bytes };
        let error = remu_shader::DecodeError::Unterminated {
            inspected_instructions: 370,
        };

        let window = shader_decode_failure_window(&memory, base, Some(code_size), &error)
            .expect("registered extent gives an unterminated shader a tail window");

        assert_eq!(
            shader_decode_context_pc(&error, Some(code_size)),
            Some(0xb8c)
        );
        assert_eq!(window.offset, 0xb4c);
        assert_eq!(window.bytes.len(), 0x44);
        assert!(window.bytes.iter().all(|byte| *byte == 0xff));
    }

    #[test]
    fn shader_code_fingerprint_covers_the_registered_tail() {
        let base = 0x0080_4eaa_0000;
        let code_size = 0xb90_u32;
        let mut first = vec![0xff_u8; usize::try_from(code_size).expect("test code size fits")];
        let first_memory = SliceMemory {
            base,
            bytes: first.clone(),
        };
        let first_fingerprint = agc_shader_code_fingerprint(&first_memory, base, code_size)
            .expect("registered shader extent is readable");

        *first.last_mut().expect("shader extent is nonempty") = 0;
        let changed_memory = SliceMemory { base, bytes: first };
        let changed_fingerprint = agc_shader_code_fingerprint(&changed_memory, base, code_size)
            .expect("changed shader extent is readable");

        assert_ne!(first_fingerprint, changed_fingerprint);
        assert_eq!(agc_shader_code_fingerprint(&changed_memory, base, 0), None);
    }
}
