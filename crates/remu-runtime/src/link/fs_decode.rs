// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Prefix-aware seeded decode for guest `fs:` virtualization (Phase 1B / F3).
//!
//! Under Rosetta the guest has no writable FS base, so every `fs:` memory operand must be
//! rewritten or fault-emulated against the guest thread pointer. Discovery is a seeded linear
//! decode (entry, `DT_INIT*`, reloc targets, `.eh_frame` FDE starts) — never a byte-pattern
//! substring match. The rewrite table is keyed by instruction shape.

use iced_x86::{
    Decoder, DecoderOptions, FlowControl, Instruction, MemorySize, Mnemonic, OpKind, Register,
};
use remu_core::GuestAddress;
use std::collections::{BTreeSet, VecDeque};

/// Shape of one authenticated `fs:` memory operand at an instruction boundary.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum FsAccessKind {
    /// `mov rN, fs:[disp]` with no base/index (absolute FS form).
    LoadGpr {
        destination: u8,
        width_bytes: u8,
        displacement: i32,
    },
    /// `mov fs:[disp], imm` with no base/index.
    StoreImm {
        width_bytes: u8,
        displacement: i32,
        immediate: i64,
    },
    /// `mov fs:[disp], rN` with no base/index.
    StoreGpr {
        source: u8,
        width_bytes: u8,
        displacement: i32,
    },
    /// `lea rN, fs:[disp]` with no base/index — address formation only.
    LeaGpr { destination: u8, displacement: i32 },
    /// Any other FS memory operand (RMW, indexed form, non-mov). AOT leaves these for the fault
    /// backstop; discovery still records them so misses are never silent.
    Other,
}

/// One authenticated `fs:` access at an instruction boundary.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct FsAccessSite {
    /// Start of the complete decoded instruction, including redundant prefixes.
    pub(crate) offset: usize,
    /// Complete decoded instruction length. The replacement must cover all of it so no prefix
    /// can leak onto a helper call.
    pub(crate) length: usize,
    pub(crate) kind: FsAccessKind,
}

/// Seeded linear decode: walk from each seed, collect instruction starts and authenticated FS sites.
pub(crate) fn seeded_fs_access_sites(
    bytes: &[u8],
    seeds: &[usize],
) -> (BTreeSet<usize>, Vec<FsAccessSite>) {
    let mut pending = VecDeque::new();
    for &seed in seeds {
        if seed < bytes.len() {
            pending.push_back(seed);
        }
    }
    let mut visited = BTreeSet::new();
    let mut boundaries = BTreeSet::new();
    let mut sites = Vec::new();
    while let Some(mut offset) = pending.pop_front() {
        while offset < bytes.len() && visited.insert(offset) {
            boundaries.insert(offset);
            let mut decoder =
                Decoder::with_ip(64, &bytes[offset..], offset as u64, DecoderOptions::NONE);
            let instruction = decoder.decode();
            if instruction.is_invalid() || instruction.len() == 0 {
                break;
            }
            let next = offset.saturating_add(instruction.len());
            if next > bytes.len() {
                break;
            }
            if let Some(site) = decoded_fs_access_site(&instruction, offset) {
                sites.push(site);
            }
            match instruction.flow_control() {
                FlowControl::Next | FlowControl::IndirectCall => {
                    offset = next;
                }
                FlowControl::Call => {
                    if let Some(target) =
                        in_buffer_target(instruction.near_branch_target(), bytes.len())
                    {
                        pending.push_back(target);
                    }
                    offset = next;
                }
                FlowControl::ConditionalBranch => {
                    if let Some(target) =
                        in_buffer_target(instruction.near_branch_target(), bytes.len())
                    {
                        pending.push_back(target);
                    }
                    offset = next;
                }
                FlowControl::UnconditionalBranch => {
                    if let Some(target) =
                        in_buffer_target(instruction.near_branch_target(), bytes.len())
                    {
                        pending.push_back(target);
                    }
                    break;
                }
                FlowControl::Return
                | FlowControl::IndirectBranch
                | FlowControl::Interrupt
                | FlowControl::XbeginXabortXend
                | FlowControl::Exception => break,
            }
        }
    }
    sites.sort_unstable();
    sites.dedup();
    (boundaries, sites)
}

fn decoded_fs_access_site(instruction: &Instruction, offset: usize) -> Option<FsAccessSite> {
    if !instruction_has_fs_memory(instruction) {
        return None;
    }
    let length = instruction.len();
    let kind = classify_fs_access(instruction).unwrap_or(FsAccessKind::Other);
    Some(FsAccessSite {
        offset,
        length,
        kind,
    })
}

fn instruction_has_fs_memory(instruction: &Instruction) -> bool {
    // Ordinary memory ops expose FS through `memory_segment()`. `lea` ignores the segment for the
    // effective address but compilers still emit `fs:` overrides for TLS address formation; those
    // are visible only via the segment-override prefix.
    if instruction.has_segment_prefix() && instruction.segment_prefix() == Register::FS {
        return true;
    }
    for operand in 0..instruction.op_count() {
        if instruction.op_kind(operand) == OpKind::Memory
            && instruction.memory_segment() == Register::FS
        {
            return true;
        }
    }
    false
}

fn classify_fs_access(instruction: &Instruction) -> Option<FsAccessKind> {
    // Absolute FS form only for AOT shapes; indexed/base forms stay Other for the fault path.
    if instruction.memory_base() != Register::None || instruction.memory_index() != Register::None {
        return Some(FsAccessKind::Other);
    }
    let displacement = fs_displacement(instruction)?;
    match instruction.mnemonic() {
        Mnemonic::Mov if instruction.op_count() == 2 => {
            let width_bytes = memory_width_bytes(instruction.memory_size())?;
            match (instruction.op0_kind(), instruction.op1_kind()) {
                (OpKind::Register, OpKind::Memory) => {
                    let destination = gpr_number(instruction.op0_register())?;
                    if destination == 4 {
                        return Some(FsAccessKind::Other);
                    }
                    Some(FsAccessKind::LoadGpr {
                        destination,
                        width_bytes,
                        displacement,
                    })
                }
                (OpKind::Memory, OpKind::Register) => {
                    let source = gpr_number(instruction.op1_register())?;
                    Some(FsAccessKind::StoreGpr {
                        source,
                        width_bytes,
                        displacement,
                    })
                }
                (
                    OpKind::Memory,
                    OpKind::Immediate8
                    | OpKind::Immediate8to16
                    | OpKind::Immediate8to32
                    | OpKind::Immediate8to64
                    | OpKind::Immediate16
                    | OpKind::Immediate32
                    | OpKind::Immediate32to64
                    | OpKind::Immediate64,
                ) => Some(FsAccessKind::StoreImm {
                    width_bytes,
                    displacement,
                    immediate: instruction.immediate(1) as i64,
                }),
                _ => Some(FsAccessKind::Other),
            }
        }
        Mnemonic::Lea if instruction.op_count() == 2 => {
            // `lea` has no memory operand size; the destination register width is irrelevant to
            // address formation (the effective address is always 64-bit in long mode).
            if instruction.op0_kind() != OpKind::Register || instruction.op1_kind() != OpKind::Memory
            {
                return Some(FsAccessKind::Other);
            }
            let destination = gpr_number(instruction.op0_register())?;
            if destination == 4 {
                return Some(FsAccessKind::Other);
            }
            Some(FsAccessKind::LeaGpr {
                destination,
                displacement,
            })
        }
        _ => Some(FsAccessKind::Other),
    }
}

fn fs_displacement(instruction: &Instruction) -> Option<i32> {
    i32::try_from(instruction.memory_displacement64() as i64).ok()
}

fn memory_width_bytes(size: MemorySize) -> Option<u8> {
    let bytes = size.size();
    u8::try_from(bytes).ok().filter(|width| *width != 0)
}

fn gpr_number(register: Register) -> Option<u8> {
    match register {
        Register::AL | Register::AX | Register::EAX | Register::RAX => Some(0),
        Register::CL | Register::CX | Register::ECX | Register::RCX => Some(1),
        Register::DL | Register::DX | Register::EDX | Register::RDX => Some(2),
        Register::BL | Register::BX | Register::EBX | Register::RBX => Some(3),
        Register::SPL | Register::SP | Register::ESP | Register::RSP => Some(4),
        Register::BPL | Register::BP | Register::EBP | Register::RBP => Some(5),
        Register::SIL | Register::SI | Register::ESI | Register::RSI => Some(6),
        Register::DIL | Register::DI | Register::EDI | Register::RDI => Some(7),
        Register::R8L | Register::R8W | Register::R8D | Register::R8 => Some(8),
        Register::R9L | Register::R9W | Register::R9D | Register::R9 => Some(9),
        Register::R10L | Register::R10W | Register::R10D | Register::R10 => Some(10),
        Register::R11L | Register::R11W | Register::R11D | Register::R11 => Some(11),
        Register::R12L | Register::R12W | Register::R12D | Register::R12 => Some(12),
        Register::R13L | Register::R13W | Register::R13D | Register::R13 => Some(13),
        Register::R14L | Register::R14W | Register::R14D | Register::R14 => Some(14),
        Register::R15L | Register::R15W | Register::R15D | Register::R15 => Some(15),
        _ => None,
    }
}

fn in_buffer_target(target: u64, length: usize) -> Option<usize> {
    let offset = usize::try_from(target).ok()?;
    (offset < length).then_some(offset)
}

/// Collects absolute guest seed addresses for one linked image (entry, `DT_INIT*`, reloc targets).
///
/// FDE function starts are supplied by the caller (from `.eh_frame_hdr`) so this stays independent
/// of unwind registration.
pub(crate) fn collect_fs_decode_seed_addresses(
    image_base: GuestAddress,
    executable: &remu_loader::LoadedExecutable,
    fde_function_starts: &[GuestAddress],
) -> Vec<GuestAddress> {
    let mut seeds = BTreeSet::new();
    let Some(elf) = executable.info.elf.as_ref() else {
        return fde_function_starts.to_vec();
    };
    let rebase = |address: GuestAddress| -> GuestAddress {
        if address.get() >= image_base.get() {
            address
        } else {
            GuestAddress::new(image_base.get().saturating_add(address.get()))
        }
    };
    seeds.insert(rebase(elf.header.entry));
    if let Some(dynamic) = elf.dynamic.as_ref() {
        if dynamic.init_address != 0 {
            seeds.insert(rebase(GuestAddress::new(dynamic.init_address)));
        }
        for relocation in dynamic
            .relocations
            .iter()
            .chain(dynamic.jump_relocations.iter())
        {
            match relocation.relocation_type {
                // R_X86_64_RELATIVE / IRELATIVE: the addend names a code/data VA in the module.
                8 | 37 => {
                    if relocation.addend >= 0 {
                        seeds.insert(rebase(GuestAddress::new(relocation.addend as u64)));
                    }
                }
                _ => {
                    if let Some(symbol) = dynamic.symbols.get(relocation.symbol_index as usize)
                        && !symbol.is_undefined()
                        && symbol.symbol_type() == 2
                        && symbol.value.get() != 0
                    {
                        seeds.insert(rebase(symbol.value));
                    }
                }
            }
        }
    }
    for start in fde_function_starts {
        seeds.insert(*start);
    }
    seeds.into_iter().collect()
}

/// Filters absolute seeds down to offsets within one contiguous byte window.
pub(crate) fn seed_offsets_in_region(
    seeds: &[GuestAddress],
    region_start: GuestAddress,
    region_len: usize,
) -> Vec<usize> {
    let start = region_start.get();
    let end = start.saturating_add(region_len as u64);
    seeds
        .iter()
        .filter_map(|seed| {
            let address = seed.get();
            if address >= start && address < end {
                usize::try_from(address - start).ok()
            } else {
                None
            }
        })
        .collect()
}

/// Decodes one instruction at `bytes` and returns its FS access shape when the encoding is an
/// FS memory operand. Used by the fault-driven backstop (async-signal-safe caller supplies a
/// bounded stack buffer of instruction bytes).
#[cfg_attr(
    not(any(test, all(target_os = "macos", target_arch = "x86_64"))),
    allow(dead_code)
)]
pub(crate) fn classify_fs_instruction_bytes(bytes: &[u8]) -> Option<FsAccessSite> {
    if bytes.is_empty() {
        return None;
    }
    let mut decoder = Decoder::with_ip(64, bytes, 0, DecoderOptions::NONE);
    let instruction = decoder.decode();
    if instruction.is_invalid() || instruction.len() == 0 || instruction.len() > bytes.len() {
        return None;
    }
    decoded_fs_access_site(&instruction, 0)
}

#[cfg(test)]
mod tests {
    use super::{FsAccessKind, FsAccessSite, classify_fs_instruction_bytes, seeded_fs_access_sites};

    const FS_LOAD_RAX: [u8; 9] = [0x64, 0x48, 0x8b, 0x04, 0x25, 0, 0, 0, 0];
    // mov rax, fs:[0x28]
    const FS_LOAD_RAX_DISP28: [u8; 9] = [0x64, 0x48, 0x8b, 0x04, 0x25, 0x28, 0, 0, 0];
    // mov eax, fs:[0]
    const FS_LOAD_EAX: [u8; 8] = [0x64, 0x8b, 0x04, 0x25, 0, 0, 0, 0];
    // mov dword fs:[0x28], imm32
    const FS_STORE_IMM: [u8; 12] = [0x64, 0xc7, 0x04, 0x25, 0x28, 0, 0, 0, 0x0b, 0, 0x02, 0xa0];
    // mov fs:[0x10], rax
    const FS_STORE_RAX: [u8; 9] = [0x64, 0x48, 0x89, 0x04, 0x25, 0x10, 0, 0, 0];
    // lea rcx, fs:[0x30]
    const FS_LEA_RCX: [u8; 9] = [0x64, 0x48, 0x8d, 0x0c, 0x25, 0x30, 0, 0, 0];
    // add rax, fs:[0] — non-mov FS op
    const FS_ADD_RAX: [u8; 9] = [0x64, 0x48, 0x03, 0x04, 0x25, 0, 0, 0, 0];

    #[test]
    fn seeded_decode_covers_displacement_and_widths() {
        let mut code = Vec::new();
        code.extend_from_slice(&FS_LOAD_RAX);
        code.extend_from_slice(&FS_LOAD_RAX_DISP28);
        code.extend_from_slice(&FS_LOAD_EAX);
        code.extend_from_slice(&FS_STORE_IMM);
        code.extend_from_slice(&FS_STORE_RAX);
        code.extend_from_slice(&FS_LEA_RCX);
        code.extend_from_slice(&FS_ADD_RAX);
        code.push(0xc3);
        let (_boundaries, sites) = seeded_fs_access_sites(&code, &[0]);
        assert!(sites.iter().any(|s| {
            matches!(
                s.kind,
                FsAccessKind::LoadGpr {
                    destination: 0,
                    width_bytes: 8,
                    displacement: 0
                }
            )
        }));
        assert!(sites.iter().any(|s| {
            matches!(
                s.kind,
                FsAccessKind::LoadGpr {
                    destination: 0,
                    width_bytes: 8,
                    displacement: 0x28
                }
            )
        }));
        assert!(sites.iter().any(|s| {
            matches!(
                s.kind,
                FsAccessKind::LoadGpr {
                    destination: 0,
                    width_bytes: 4,
                    displacement: 0
                }
            )
        }));
        assert!(sites.iter().any(|s| {
            matches!(
                s.kind,
                FsAccessKind::StoreImm {
                    width_bytes: 4,
                    displacement: 0x28,
                    // iced_x86 zero-extends the Immediate32; AOT must keep the low 32 bits.
                    immediate: 0x0000_0000_a002_000b,
                }
            )
        }));
        assert!(sites.iter().any(|s| {
            matches!(
                s.kind,
                FsAccessKind::StoreGpr {
                    source: 0,
                    width_bytes: 8,
                    displacement: 0x10
                }
            )
        }));
        assert!(sites.iter().any(|s| {
            matches!(
                s.kind,
                FsAccessKind::LeaGpr {
                    destination: 1,
                    displacement: 0x30
                }
            )
        }));
        assert!(
            sites
                .iter()
                .any(|s| matches!(s.kind, FsAccessKind::Other))
        );
    }

    #[test]
    fn mid_instruction_pattern_is_not_an_fs_site() {
        // `movabs rax, imm64` whose immediate begins with the fs:[0] load pattern.
        let mut code = vec![0x48, 0xb8];
        code.extend_from_slice(&FS_LOAD_RAX);
        code.push(0xc3);
        let (_boundaries, sites) = seeded_fs_access_sites(&code, &[0]);
        assert!(sites.is_empty());
    }

    #[test]
    fn seeded_decode_follows_a_direct_call_into_a_leaf() {
        let mut code = vec![0xe8, 1, 0, 0, 0, 0xc3];
        code.extend_from_slice(&FS_LOAD_RAX_DISP28);
        code.push(0xc3);
        assert_eq!(
            seeded_fs_access_sites(&code, &[0]).1,
            vec![FsAccessSite {
                offset: 6,
                length: 9,
                kind: FsAccessKind::LoadGpr {
                    destination: 0,
                    width_bytes: 8,
                    displacement: 0x28,
                }
            }]
        );
    }

    #[test]
    fn redundant_prefixes_remain_part_of_one_authenticated_fs_load() {
        let mut code = vec![0x66, 0x66, 0x66];
        code.extend_from_slice(&FS_LOAD_RAX);
        code.push(0xc3);
        assert_eq!(
            seeded_fs_access_sites(&code, &[0]).1,
            vec![FsAccessSite {
                offset: 0,
                length: 12,
                kind: FsAccessKind::LoadGpr {
                    destination: 0,
                    width_bytes: 8,
                    displacement: 0,
                }
            }]
        );
    }

    #[test]
    fn without_seeds_the_decoder_confirms_nothing() {
        let mut code = FS_LOAD_RAX.to_vec();
        code.push(0xc3);
        assert!(seeded_fs_access_sites(&code, &[]).1.is_empty());
    }

    #[test]
    fn fault_classifier_accepts_stack_guard_load() {
        let site = classify_fs_instruction_bytes(&FS_LOAD_RAX_DISP28).expect("fs site");
        assert_eq!(
            site.kind,
            FsAccessKind::LoadGpr {
                destination: 0,
                width_bytes: 8,
                displacement: 0x28,
            }
        );
    }
}
