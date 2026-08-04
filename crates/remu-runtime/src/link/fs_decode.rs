// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Seeded linear decode for guest `fs:` virtualization (F3).
//!
//! The historical load rewriter scanned every executable byte for the 9-byte
//! `mov r64, fs:[0]` pattern. That scan can match inside immediates and corrupt an
//! unrelated instruction stream. This module walks code from known entry points
//! (ELF entry, `DT_INIT*`, reloc targets, `.eh_frame` FDE starts) and reports every
//! byte-scan hit that is not an instruction boundary before any rewrite runs.

use iced_x86::{Decoder, DecoderOptions, FlowControl, Instruction, Mnemonic, OpKind, Register};
use remu_core::GuestAddress;
use std::collections::{BTreeSet, VecDeque};

/// Nine-byte encoding of `mov r64, fs:[0]` with a ModRM sib-disp32 form.
pub(crate) const FS_LOAD_ENCODING_LEN: usize = 9;

/// Summary of comparing the legacy byte scan against a seeded linear decode.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct FsLoadBoundaryReport {
    /// Sites where the raw byte scan found the nine-byte pattern.
    pub(crate) byte_scan_hits: usize,
    /// Byte-scan hits that are not instruction starts under seeded decode.
    pub(crate) boundary_mismatches: usize,
    /// Authentic `mov r64, fs:[0]` instructions found on decode boundaries.
    pub(crate) decoder_confirmed_loads: usize,
}

/// One authenticated `mov r64, fs:[0]` at an instruction boundary.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct FsLoadSite {
    /// Start of the complete decoded instruction, including redundant prefixes.
    pub(crate) offset: usize,
    /// Complete decoded instruction length. The replacement must cover all of it so no prefix
    /// can leak onto the helper call.
    pub(crate) length: usize,
    /// Offset where the canonical nine-byte `fs:` encoding begins inside the instruction.
    pub(crate) encoding_offset: usize,
    /// Destination register number in the AMD64 encoding (0=rax … 15=r15).
    pub(crate) destination: u8,
}

/// Returns every offset where the nine-byte `mov r64, fs:[0]` pattern occurs.
pub(crate) fn byte_scan_fs_load_offsets(bytes: &[u8]) -> Vec<usize> {
    let mut hits = Vec::new();
    if bytes.len() < FS_LOAD_ENCODING_LEN {
        return hits;
    }
    for offset in 0..=bytes.len() - FS_LOAD_ENCODING_LEN {
        if let Some(destination) =
            fs_load_destination(&bytes[offset..offset + FS_LOAD_ENCODING_LEN])
            && destination != 4
        {
            hits.push(offset);
        }
    }
    hits
}

/// Seeded linear decode: walk from each seed, collect instruction starts and authenticated loads.
pub(crate) fn seeded_fs_load_sites(
    bytes: &[u8],
    seeds: &[usize],
) -> (BTreeSet<usize>, Vec<FsLoadSite>) {
    let mut pending = VecDeque::new();
    for &seed in seeds {
        if seed < bytes.len() {
            pending.push_back(seed);
        }
    }
    let mut visited = BTreeSet::new();
    let mut boundaries = BTreeSet::new();
    let mut loads = Vec::new();
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
            if let Some(site) = decoded_fs_load_site(&instruction, bytes, offset, next) {
                loads.push(site);
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
    loads.sort_unstable();
    loads.dedup();
    (boundaries, loads)
}

/// Compares the legacy byte scan to seeded decode. Boundary mismatches are the loud diagnostic
/// that must ship before any rewrite trusts a scan hit.
#[cfg_attr(not(test), allow(dead_code))] // F3 diagnostic; unit tests exercise it.
pub(crate) fn diagnose_fs_load_boundary_mismatches(
    bytes: &[u8],
    seeds: &[usize],
) -> FsLoadBoundaryReport {
    let byte_hits = byte_scan_fs_load_offsets(bytes);
    let (_boundaries, confirmed) = seeded_fs_load_sites(bytes, seeds);
    let confirmed_encodings = confirmed
        .iter()
        .map(|site| site.encoding_offset)
        .collect::<BTreeSet<_>>();
    let mismatches = byte_hits
        .iter()
        .filter(|offset| !confirmed_encodings.contains(offset))
        .count();
    FsLoadBoundaryReport {
        byte_scan_hits: byte_hits.len(),
        boundary_mismatches: mismatches,
        decoder_confirmed_loads: confirmed.len(),
    }
}

/// Logs every byte-scan hit the decoder rejects as a non-boundary, then returns the report.
pub(crate) fn report_fs_load_boundary_mismatches(
    region_label: &str,
    region_base: u64,
    bytes: &[u8],
    seeds: &[usize],
) -> FsLoadBoundaryReport {
    let byte_hits = byte_scan_fs_load_offsets(bytes);
    let (_boundaries, confirmed) = seeded_fs_load_sites(bytes, seeds);
    let confirmed_encodings = confirmed
        .iter()
        .map(|site| site.encoding_offset)
        .collect::<BTreeSet<_>>();
    let mut mismatches = 0_usize;
    for offset in &byte_hits {
        if confirmed_encodings.contains(offset) {
            continue;
        }
        mismatches += 1;
        let address = region_base.saturating_add(*offset as u64);
        let encoding = &bytes[*offset..*offset + FS_LOAD_ENCODING_LEN];
        eprintln!(
            "[REMu][TLS][FS] boundary-mismatch region={region_label} addr={address:#x} offset={offset:#x} encoding={encoding:02x?} — byte-scan hit is not an instruction boundary under seeded decode; refusing silent rewrite"
        );
    }
    let report = FsLoadBoundaryReport {
        byte_scan_hits: byte_hits.len(),
        boundary_mismatches: mismatches,
        decoder_confirmed_loads: confirmed.len(),
    };
    if report.byte_scan_hits != 0 || report.decoder_confirmed_loads != 0 {
        eprintln!(
            "[REMu][TLS][FS] decode-summary region={region_label} base={region_base:#x} byte_scan_hits={} boundary_mismatches={} decoder_confirmed_loads={}",
            report.byte_scan_hits, report.boundary_mismatches, report.decoder_confirmed_loads
        );
    }
    report
}

fn fs_load_destination(encoding: &[u8]) -> Option<u8> {
    if encoding.len() < FS_LOAD_ENCODING_LEN {
        return None;
    }
    if encoding[0] != 0x64 || !(0x48..=0x4f).contains(&encoding[1]) {
        return None;
    }
    let rex = encoding[1];
    if rex & 0x08 == 0 || encoding[2] != 0x8b {
        return None;
    }
    let mod_rm = encoding[3];
    if mod_rm >> 6 != 0 || mod_rm & 7 != 4 || encoding[4] != 0x25 {
        return None;
    }
    if encoding[5..9] != [0, 0, 0, 0] {
        return None;
    }
    Some(((mod_rm >> 3) & 7) | ((rex & 4) << 1))
}

fn decoded_fs_load_site(
    instruction: &Instruction,
    bytes: &[u8],
    offset: usize,
    next: usize,
) -> Option<FsLoadSite> {
    if instruction.mnemonic() != Mnemonic::Mov
        || instruction.op_count() != 2
        || instruction.op0_kind() != OpKind::Register
        || instruction.op1_kind() != OpKind::Memory
        || instruction.memory_segment() != Register::FS
        || instruction.memory_base() != Register::None
        || instruction.memory_index() != Register::None
        || instruction.memory_displacement64() != 0
    {
        return None;
    }
    let decoded_destination = gpr64_number(instruction.op0_register())?;
    let last_encoding_start = next.checked_sub(FS_LOAD_ENCODING_LEN)?;
    for encoding_offset in offset..=last_encoding_start {
        let Some(destination) =
            fs_load_destination(&bytes[encoding_offset..encoding_offset + FS_LOAD_ENCODING_LEN])
        else {
            continue;
        };
        if destination == decoded_destination && destination != 4 {
            return Some(FsLoadSite {
                offset,
                length: instruction.len(),
                encoding_offset,
                destination,
            });
        }
    }
    None
}

fn gpr64_number(register: Register) -> Option<u8> {
    match register {
        Register::RAX => Some(0),
        Register::RCX => Some(1),
        Register::RDX => Some(2),
        Register::RBX => Some(3),
        Register::RSP => Some(4),
        Register::RBP => Some(5),
        Register::RSI => Some(6),
        Register::RDI => Some(7),
        Register::R8 => Some(8),
        Register::R9 => Some(9),
        Register::R10 => Some(10),
        Register::R11 => Some(11),
        Register::R12 => Some(12),
        Register::R13 => Some(13),
        Register::R14 => Some(14),
        Register::R15 => Some(15),
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
        // Init-array slots themselves are data; the pointers they hold are code seeds once the
        // image is mapped. Callers that already expanded the array should pass those addresses
        // via `fde_function_starts`/`extra`. Relative/symbolic reloc targets that land in code
        // are still useful before constructors run.
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

#[cfg(test)]
mod tests {
    use super::{
        FsLoadSite, byte_scan_fs_load_offsets, diagnose_fs_load_boundary_mismatches,
        seeded_fs_load_sites,
    };

    const FS_LOAD_RAX: [u8; 9] = [0x64, 0x48, 0x8b, 0x04, 0x25, 0, 0, 0, 0];

    #[test]
    fn authentic_fs_load_is_not_a_boundary_mismatch() {
        let mut code = FS_LOAD_RAX.to_vec();
        code.push(0xc3);
        let report = diagnose_fs_load_boundary_mismatches(&code, &[0]);
        assert_eq!(report.byte_scan_hits, 1);
        assert_eq!(report.boundary_mismatches, 0);
        assert_eq!(report.decoder_confirmed_loads, 1);
        assert_eq!(
            seeded_fs_load_sites(&code, &[0]).1,
            vec![FsLoadSite {
                offset: 0,
                length: 9,
                encoding_offset: 0,
                destination: 0
            }]
        );
    }

    #[test]
    fn mid_instruction_byte_scan_hit_is_a_loud_boundary_mismatch() {
        // `movabs rax, imm64` whose immediate begins with the fs:[0] load pattern; the ninth
        // pattern byte sits immediately after the imm64 so a raw scan still matches at offset 2.
        let mut code = vec![0x48, 0xb8];
        code.extend_from_slice(&FS_LOAD_RAX);
        code.push(0xc3);
        assert_eq!(byte_scan_fs_load_offsets(&code), vec![2]);
        let report = diagnose_fs_load_boundary_mismatches(&code, &[0]);
        assert_eq!(report.byte_scan_hits, 1);
        assert_eq!(report.boundary_mismatches, 1);
        assert_eq!(report.decoder_confirmed_loads, 0);
        assert!(seeded_fs_load_sites(&code, &[0]).1.is_empty());
    }

    #[test]
    fn seeded_decode_reaches_fs_load_past_a_nop_from_entry() {
        let mut code = vec![0x90];
        code.extend_from_slice(&FS_LOAD_RAX);
        code.push(0xc3);
        assert_eq!(
            seeded_fs_load_sites(&code, &[0]).1,
            vec![FsLoadSite {
                offset: 1,
                length: 9,
                encoding_offset: 1,
                destination: 0
            }]
        );
        let report = diagnose_fs_load_boundary_mismatches(&code, &[0]);
        assert_eq!(report.boundary_mismatches, 0);
        assert_eq!(report.decoder_confirmed_loads, 1);
    }

    #[test]
    fn seeded_decode_follows_a_direct_call_into_a_leaf_without_its_own_seed() {
        // call +1 → offset 6; the fall-through path returns at offset 5. Small leaf functions do
        // not necessarily have an FDE/symbol of their own, so their direct call edge is the only
        // authenticated route to the TLS load.
        let mut code = vec![0xe8, 1, 0, 0, 0, 0xc3];
        code.extend_from_slice(&FS_LOAD_RAX);
        code.push(0xc3);
        assert_eq!(
            seeded_fs_load_sites(&code, &[0]).1,
            vec![FsLoadSite {
                offset: 6,
                length: 9,
                encoding_offset: 6,
                destination: 0
            }]
        );
    }

    #[test]
    fn redundant_prefixes_remain_part_of_one_authenticated_tls_load() {
        // Clang may align a hot-patchable instruction with redundant operand-size prefixes.
        // The `fs:` prefix is therefore not necessarily the first byte of the decoded MOV.
        let mut code = vec![0x66, 0x66, 0x66];
        code.extend_from_slice(&FS_LOAD_RAX);
        code.push(0xc3);
        let report = diagnose_fs_load_boundary_mismatches(&code, &[0]);
        assert_eq!(report.byte_scan_hits, 1);
        assert_eq!(report.boundary_mismatches, 0);
        assert_eq!(report.decoder_confirmed_loads, 1);
        assert_eq!(
            seeded_fs_load_sites(&code, &[0]).1,
            vec![FsLoadSite {
                offset: 0,
                length: 12,
                encoding_offset: 3,
                destination: 0
            }]
        );
    }

    #[test]
    fn without_seeds_the_decoder_confirms_nothing_even_when_bytes_match() {
        let mut code = FS_LOAD_RAX.to_vec();
        code.push(0xc3);
        let report = diagnose_fs_load_boundary_mismatches(&code, &[]);
        assert_eq!(report.byte_scan_hits, 1);
        assert_eq!(report.boundary_mismatches, 1);
        assert_eq!(report.decoder_confirmed_loads, 0);
    }
}
