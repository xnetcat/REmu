// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Kraken/Oodle decompress stage for AMPR (plan P3), backed by the vendored ooz decoder
//! (`vendored/ooz/PROVENANCE.md`).
//!
//! The two-phase AMPR submit already accepts a decode-stage closure of signature
//! `FnMut(&std::fs::File, u32, u64, u64, u64, usize) -> Result<usize, u64>`
//! (`ampr_api.rs` — `ampr_submit_command_buffer_two_phase_with` / `ampr_execute_recorded_read`).
//! This module provides [`ampr_kraken_decode_stage`] matching that signature so the
//! integrator can swap it in for `ampr_read_file_to_guest` at the submit call site
//! (`ampr_submit_command_buffer_ordered`) without this Batch A2 touching `ampr_api.rs`.
//!
//! Seam (DO NOT edit ampr_api from this batch): the production submit currently passes
//! `ampr_read_file_to_guest` at the ordered-submit entry. The historical NOTES line
//! `ampr_api.rs:942` named an earlier anchor; at `b9d55a5` the live injection points are
//! the `F: FnMut(...)` bounds on the two-phase helper and the
//! `ampr_submit_command_buffer_ordered` call that supplies the default reader.
//!
//! These helpers are intentionally unused by production until the AMPR owner wires them;
//! `#[allow(dead_code)]` keeps the seam visible without inventing a fake call site.

#![allow(dead_code)]

use std::io::{Read, Seek, SeekFrom};
use std::os::raw::c_int;

unsafe extern "C" {
    fn ooz_kraken_decompress(
        src: *const u8,
        src_len: usize,
        dst: *mut u8,
        dst_len: usize,
    ) -> c_int;
}

/// Decompresses a Kraken/Oodle payload into `dst`. Returns the number of bytes written,
/// or `None` when ooz rejects the stream.
pub(crate) fn kraken_decompress(src: &[u8], dst: &mut [u8]) -> Option<usize> {
    if src.is_empty() || dst.is_empty() {
        return None;
    }
    // SAFETY: `src`/`dst` are valid host buffers; ooz reads `src_len` and writes at most
    // `dst_len` (plus an internal SAFE_SPACE scratch it allocates itself).
    let written = unsafe {
        ooz_kraken_decompress(src.as_ptr(), src.len(), dst.as_mut_ptr(), dst.len())
    };
    if written <= 0 {
        None
    } else {
        Some(written as usize)
    }
}

/// AMPR decode-stage closure body matching
/// `FnMut(&File, file_id, destination, size, file_offset, capacity) -> Result<usize, u64>`.
///
/// Framing (ooz tool convention): the file region begins with an 8-byte (or legacy 4-byte)
/// little-endian uncompressed size, followed by the Kraken bitstream. The guest `size` /
/// `capacity` is the destination byte count (= uncompressed size). On success the
/// decompressed bytes are written into guest memory at `destination` and the uncompressed
/// length is returned. A malformed stream fails this record only (`Err`), so the two-phase
/// submit can contain the failure per the existing per-read contract.
pub fn ampr_kraken_decode_stage(
    file: &std::fs::File,
    _file_id: u32,
    destination: u64,
    size: u64,
    file_offset: u64,
    capacity: usize,
) -> Result<usize, u64> {
    let Ok(want) = usize::try_from(size) else {
        return Err(crate::sce_kernel_error(crate::SCE_KERNEL_ERROR_INVALID_ARGUMENT));
    };
    if want == 0 {
        return Ok(0);
    }
    if want > capacity || destination == 0 {
        return Err(crate::sce_kernel_error(crate::SCE_KERNEL_ERROR_INVALID_ARGUMENT));
    }

    let mut file = file;
    if file.seek(SeekFrom::Start(file_offset)).is_err() {
        return Err(crate::sce_kernel_error(crate::SCE_KERNEL_ERROR_NOT_FOUND));
    }

    // Read the ooz size header (prefer 8-byte). Peek 8 bytes; fall back to 4 when the
    // high dword looks like a legacy header (same autodetection as ooz's CLI).
    let mut header = [0_u8; 8];
    if file.read_exact(&mut header).is_err() {
        // Short read — try 4-byte legacy header alone.
        if file.seek(SeekFrom::Start(file_offset)).is_err() {
            return Err(crate::sce_kernel_error(crate::SCE_KERNEL_ERROR_NOT_FOUND));
        }
        let mut legacy = [0_u8; 4];
        if file.read_exact(&mut legacy).is_err() {
            return Err(crate::sce_kernel_error(crate::SCE_KERNEL_ERROR_NOT_FOUND));
        }
        header[..4].copy_from_slice(&legacy);
        header[4..].fill(0);
    }
    let as_u64 = u64::from_le_bytes(header);
    let (hdrsize, unpacked) = if as_u64 >= 0x100_0000_0000 {
        (4_usize, u64::from(u32::from_le_bytes(header[0..4].try_into().unwrap())))
    } else {
        (8_usize, as_u64)
    };
    if unpacked == 0 || unpacked as usize > capacity || unpacked as usize != want {
        // Destination size must match the stream's declared uncompressed length — a mismatch
        // is a loud refusal, not a truncated invent.
        eprintln!(
            "[REMu][OOZ] kraken decode refused unpacked={unpacked} want={want} capacity={capacity}"
        );
        return Err(crate::sce_kernel_error(crate::SCE_KERNEL_ERROR_INVALID_ARGUMENT));
    }

    // Compressed payload: read until EOF from the post-header offset. Bound by a generous
    // multiple of the uncompressed size so a hostile file cannot force unbounded alloc.
    let max_compressed = want.saturating_mul(2).saturating_add(64 * 1024).max(want + 16);
    if file
        .seek(SeekFrom::Start(file_offset + hdrsize as u64))
        .is_err()
    {
        return Err(crate::sce_kernel_error(crate::SCE_KERNEL_ERROR_NOT_FOUND));
    }
    let mut compressed = Vec::new();
    {
        let mut take = file.take(max_compressed as u64);
        if take.read_to_end(&mut compressed).is_err() || compressed.is_empty() {
            return Err(crate::sce_kernel_error(crate::SCE_KERNEL_ERROR_NOT_FOUND));
        }
    }

    let mut plain = vec![0_u8; want];
    let Some(written) = kraken_decompress(&compressed, &mut plain) else {
        eprintln!("[REMu][OOZ] kraken decompress failed want={want} compressed={}", compressed.len());
        return Err(crate::sce_kernel_error(crate::SCE_KERNEL_ERROR_INVALID_ARGUMENT));
    };
    if written != want {
        eprintln!("[REMu][OOZ] kraken length mismatch written={written} want={want}");
        return Err(crate::sce_kernel_error(crate::SCE_KERNEL_ERROR_INVALID_ARGUMENT));
    }

    // SAFETY: caller validated the destination mapping before invoking the decode stage
    // (see `ampr_execute_recorded_read`); we copy exactly `want` bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(plain.as_ptr(), destination as *mut u8, want);
    }
    remu_memory::record_guest_write(
        remu_memory::GuestWriteSite::AmprCompletion,
        destination,
        want as u64,
        &plain[..want.min(4096)],
        0,
    );
    Ok(want)
}
