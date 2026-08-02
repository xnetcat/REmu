// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    GRAPHICS5_ERROR_INVALID_PACKET, IT_COND_EXEC, IT_DMA_DATA, IT_EVENT_WRITE_EOP,
    IT_INDIRECT_BUFFER, IT_NOP, IT_RELEASE_MEM, IT_WRITE_DATA, IndirectRegisterKind, R_RELEASE_MEM,
    R_WAIT_MEM_32, R_WAIT_MEM_64, is_packet, low_u8, low_u16, low_u32, packet_opcode,
    packet_register, read_u32, write_u32, write_u64,
};

/// Names why an indirect-patch import declined its packet. The 0x8a6c000c return is visible in
/// `[REMu][FAIL]`, but several distinct causes collapse into that one constant — and at least one
/// title (Octopath) reaches these imports with a null descriptor as apparently ordinary guest
/// behaviour, so telling a null descriptor from a wrong opcode is what makes the line readable.
fn decline_indirect_patch(reason: &str, command: u64, expected_opcode: u32) -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static DECLINED: AtomicU64 = AtomicU64::new(0);
    let sequence = DECLINED.fetch_add(1, Ordering::Relaxed) + 1;
    if crate::trace_logarithmic_sequence(sequence, 16) {
        let header = (command != 0 && command.is_multiple_of(4))
            .then(|| crate::try_read_process_guest_u32(command))
            .flatten()
            .map_or_else(|| "unreadable".to_owned(), |header| format!("{header:#010x}"));
        eprintln!(
            "[REMu][AGC] declined indirect-register patch sequence={sequence} \
             command={command:#x} header={header} expected-opcode={expected_opcode:#x} \
             reason={reason}"
        );
    }
    GRAPHICS5_ERROR_INVALID_PACKET
}

pub(super) fn patch_indirect_address(arguments: [u64; 6], kind: IndirectRegisterKind) -> u64 {
    let command = arguments[0];
    let address = arguments[1];
    if address == 0 || address & 3 != 0 {
        return decline_indirect_patch("unusable-register-block-address", command, kind.opcode());
    }
    if !is_packet(command, kind.opcode(), None) {
        return decline_indirect_patch("packet-mismatch", command, kind.opcode());
    }
    write_u32(
        command + 4,
        (read_u32(command + 4) & 3) | (low_u32(address) & !3),
    );
    write_u32(command + 8, low_u32(address >> 32));
    0
}

pub(super) fn patch_indirect_count(
    arguments: [u64; 6],
    kind: IndirectRegisterKind,
    add: bool,
) -> u64 {
    let command = arguments[0];
    if !is_packet(command, kind.opcode(), None) {
        return decline_indirect_patch("packet-mismatch", command, kind.opcode());
    }
    let count = low_u32(arguments[1]);
    if count > 0x3fff {
        return decline_indirect_patch("register-count-too-large", command, kind.opcode());
    }
    let previous = read_u32(command + 16);
    let next = if add {
        (previous & 0x3fff).wrapping_add(count) & 0x3fff
    } else {
        count
    };
    write_u32(command + 16, (previous & !0x3fff) | next);
    0
}

pub(super) fn patch_dma_address(arguments: [u64; 6], source: bool) -> u64 {
    let command = arguments[0];
    if !is_packet(command, IT_DMA_DATA, None) {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    }
    write_u64(command + if source { 8 } else { 16 }, arguments[1]);
    0
}

pub(super) fn patch_jump_target(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let target = arguments[1];
    let count = low_u32(arguments[2]);
    if target == 0
        || target & 3 != 0
        || count > 0x000f_ffff
        || !is_packet(command, IT_INDIRECT_BUFFER, None)
    {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    }
    write_u32(command + 4, low_u32(target));
    write_u32(
        command + 8,
        (read_u32(command + 8) & 0xffff_0000) | (low_u32(target >> 32) & 0xffff),
    );
    write_u32(command + 12, (read_u32(command + 12) & 0xfff0_0000) | count);
    0
}

pub(super) fn patch_cond_exec_address(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let address = arguments[1];
    if address == 0 || address & 3 != 0 || !is_packet(command, IT_COND_EXEC, None) {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    }
    write_u32(
        command + 4,
        (read_u32(command + 4) & 3) | (low_u32(address) & !3),
    );
    write_u32(command + 8, low_u32(address >> 32));
    0
}

pub(super) fn patch_cond_exec_end(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let end = arguments[1];
    if end == 0 || end & 3 != 0 || !is_packet(command, IT_COND_EXEC, None) {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    }
    let Some(packet_end) = command.checked_add(20) else {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    };
    let Some(bytes) = end.checked_sub(packet_end) else {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    };
    let count = bytes / 4;
    if count > 0x3fff {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    }
    write_u32(
        command + 16,
        (read_u32(command + 16) & !0x3fff) | low_u32(count),
    );
    0
}

pub(super) fn wait_kind(command: u64) -> Option<(u64, u64, u64)> {
    if is_packet(command, IT_NOP, Some(R_WAIT_MEM_32)) {
        Some((20, 16, 12))
    } else if is_packet(command, IT_NOP, Some(R_WAIT_MEM_64)) {
        Some((28, 20, 12))
    } else {
        None
    }
}

pub(super) fn patch_wait_address(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let address = arguments[1];
    let header = crate::try_read_process_guest_u32(command);
    let is_wait_32 = header.is_some_and(|header| {
        (header >> 8) & 0xff == IT_NOP && (header >> 2) & 0x3f == R_WAIT_MEM_32
    });
    let is_wait_64 = header.is_some_and(|header| {
        (header >> 8) & 0xff == IT_NOP && (header >> 2) & 0x3f == R_WAIT_MEM_64
    });
    let alignment = if is_wait_32 {
        4
    } else if is_wait_64 {
        8
    } else {
        // Newer Gen5 SDKs pass (address-or-offset, pointer-to-address-field). The first argument
        // may be a GPU-only virtual address in the host's PROT_NONE aperture, so never probe it as
        // a packet. The second argument is the CPU-writable field returned by the packet builder.
        if command == 0
            || command & 3 != 0
            || address == 0
            || !address.is_multiple_of(4)
            || !crate::process_guest_range_is_readable(address, 8)
        {
            return GRAPHICS5_ERROR_INVALID_PACKET;
        }
        write_u64(address, command);
        return 0;
    };
    if address == 0 || address & (alignment - 1) != 0 || address >> 50 != 0 {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    }
    let alignment = u32::try_from(alignment).expect("wait alignment fits in u32");
    write_u32(command + 4, low_u32(address) & !(alignment - 1));
    write_u32(command + 8, low_u32(address >> 32) & 0x3_ffff);
    0
}

pub(super) fn patch_wait_compare(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let compare = low_u8(arguments[1]);
    let Some((control_offset, _, _)) = wait_kind(command) else {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    };
    if compare > 7 {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    }
    write_u32(
        command + control_offset,
        (read_u32(command + control_offset) & !7) | compare,
    );
    0
}

pub(super) fn patch_wait_reference(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let Some((_, reference_offset, _)) = wait_kind(command) else {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    };
    if is_packet(command, IT_NOP, Some(R_WAIT_MEM_32)) {
        write_u32(command + reference_offset, low_u32(arguments[1]));
    } else {
        write_u64(command + reference_offset, arguments[1]);
    }
    0
}

pub(super) fn patch_wait_mask(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let Some((_, _, mask_offset)) = wait_kind(command) else {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    };
    if is_packet(command, IT_NOP, Some(R_WAIT_MEM_32)) {
        write_u32(command + mask_offset, low_u32(arguments[1]));
    } else {
        write_u64(command + mask_offset, arguments[1]);
    }
    0
}

pub(super) fn release_packet_kind(command: u64) -> Option<u32> {
    let opcode = packet_opcode(command)?;
    ((opcode == IT_NOP && packet_register(command) == R_RELEASE_MEM)
        || matches!(opcode, IT_RELEASE_MEM | IT_EVENT_WRITE_EOP))
    .then_some(opcode)
}

pub(super) fn patch_release_address(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let address = arguments[1];
    let Some(opcode) = release_packet_kind(command) else {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    };
    if opcode != IT_NOP && address & 3 != 0 {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    }
    if opcode == IT_EVENT_WRITE_EOP {
        write_u32(command + 8, low_u32(address));
        write_u32(
            command + 12,
            (read_u32(command + 12) & 0xffff_0000) | (low_u32(address >> 32) & 0xffff),
        );
    } else {
        write_u64(command + 12, address);
    }
    0
}

pub(super) fn patch_release_gcr(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let gcr = low_u16(arguments[1]);
    if !is_packet(command, IT_NOP, Some(R_RELEASE_MEM)) {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    }
    write_u32(command + 8, (read_u32(command + 8) & !0x0000_ffff) | gcr);
    0
}

pub(super) fn patch_release_data(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let Some(opcode) = release_packet_kind(command) else {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    };
    let mut data = arguments[3];
    if opcode == IT_NOP
        && packet_register(command) == R_RELEASE_MEM
        && low_u32(arguments[1]) > 1
        && low_u32(arguments[2]) == 1
    {
        data = (u64::from(low_u32(arguments[1]) - 2) << 24) | (data & 0x00ff_ffff);
    }
    write_u64(
        command + if opcode == IT_EVENT_WRITE_EOP { 16 } else { 20 },
        data,
    );
    0
}

pub(super) fn patch_release_type(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let data_selection = low_u8(arguments[1]);
    if data_selection > 3 || !is_packet(command, IT_NOP, Some(R_RELEASE_MEM)) {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    }
    write_u32(
        command + 8,
        (read_u32(command + 8) & !0x00ff_0000) | (data_selection << 16),
    );
    0
}

pub(super) fn patch_write_data_address(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    if !is_packet(command, IT_WRITE_DATA, None) || arguments[1] & 3 != 0 {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    }
    write_u64(command + 8, arguments[1]);
    0
}
