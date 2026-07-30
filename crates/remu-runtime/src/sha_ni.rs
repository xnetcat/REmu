// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Software emulation of Intel SHA-NI (SHA-1) instructions.
//!
//! PS5 titles compile SHA-1 with SHA-NI. Apple Silicon Rosetta does not execute those opcodes and
//! raises `EXC_BAD_INSTRUCTION` / SIGILL. Recover by decoding the faulting instruction, applying
//! the Intel SDM / Bochs semantics to the live XMM state, and advancing RIP.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Number of SHA-NI instructions recovered by the SIGILL handler.
static RECOVERED: AtomicU64 = AtomicU64::new(0);

static INSTALL: OnceLock<Result<(), i32>> = OnceLock::new();

/// Installs the process-wide SIGILL recovery for SHA-NI opcodes. Safe to call repeatedly.
pub(crate) fn install_sha_ni_sigill_recovery() -> Result<(), std::io::Error> {
    let result = INSTALL.get_or_init(|| {
        // SAFETY: installs a process-wide SA_SIGINFO handler once; the previous action is retained
        // for chaining when the fault is not a SHA-NI opcode we recognize.
        unsafe { install_sigill_handler_once() }
    });
    result.map_err(std::io::Error::from_raw_os_error)
}

pub(crate) fn recovered_sha_ni_count() -> u64 {
    RECOVERED.load(Ordering::Relaxed)
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
unsafe fn install_sigill_handler_once() -> Result<(), i32> {
    let mut action = unsafe { std::mem::zeroed::<libc::sigaction>() };
    action.sa_sigaction = sha_ni_sigill_handler as *const () as usize;
    action.sa_flags = libc::SA_SIGINFO;
    unsafe { libc::sigemptyset(&raw mut action.sa_mask) };
    // SAFETY: PREVIOUS_SIGILL is uninitialized writable storage reserved for the prior action.
    if unsafe { libc::sigaction(libc::SIGILL, &raw const action, PREVIOUS_SIGILL.as_mut_ptr()) } != 0
    {
        return Err(std::io::Error::last_os_error()
            .raw_os_error()
            .unwrap_or(1));
    }
    Ok(())
}

#[cfg(not(all(target_os = "macos", target_arch = "x86_64")))]
unsafe fn install_sigill_handler_once() -> Result<(), i32> {
    Ok(())
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
struct PreviousAction(std::cell::UnsafeCell<std::mem::MaybeUninit<libc::sigaction>>);

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
// SAFETY: written once during install, then only read from the signal handler.
unsafe impl Sync for PreviousAction {}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
impl PreviousAction {
    const fn new() -> Self {
        Self(std::cell::UnsafeCell::new(std::mem::MaybeUninit::uninit()))
    }

    const fn as_mut_ptr(&self) -> *mut libc::sigaction {
        self.0.get().cast()
    }

    unsafe fn get(&self) -> &libc::sigaction {
        unsafe { &*self.as_mut_ptr() }
    }
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
static PREVIOUS_SIGILL: PreviousAction = PreviousAction::new();

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
extern "C" fn sha_ni_sigill_handler(
    signal: libc::c_int,
    information: *mut libc::siginfo_t,
    context: *mut std::ffi::c_void,
) {
    if try_recover_sha_ni(context) {
        RECOVERED.fetch_add(1, Ordering::Relaxed);
        return;
    }
    unsafe { chain_previous_sigill(signal, information, context) };
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn try_recover_sha_ni(context: *mut std::ffi::c_void) -> bool {
    if context.is_null() {
        return false;
    }
    // SAFETY: SA_SIGINFO delivers a live ucontext_t for the duration of the handler.
    let ucontext = unsafe { &mut *context.cast::<libc::ucontext_t>() };
    if ucontext.uc_mcontext.is_null() {
        return false;
    }
    // SAFETY: uc_mcontext points at the faulting thread's Darwin mcontext for this signal.
    let mcontext = unsafe { &mut *ucontext.uc_mcontext };
    let rip = mcontext.__ss.__rip;
    if rip == 0 {
        return false;
    }
    // SAFETY: guest code is mapped into the process address space; RIP points at the faulting insn.
    let bytes = unsafe { std::slice::from_raw_parts(rip as *const u8, 8) };
    let Some(decoded) = decode_sha1_ni(bytes) else {
        return false;
    };
    match decoded.op {
        Sha1Op::Msg1 => {
            let mut dst = read_xmm(mcontext, decoded.dest);
            let src = read_xmm(mcontext, decoded.src);
            sha1_msg1(&mut dst, src);
            write_xmm(mcontext, decoded.dest, dst);
        }
        Sha1Op::Msg2 => {
            let mut dst = read_xmm(mcontext, decoded.dest);
            let src = read_xmm(mcontext, decoded.src);
            sha1_msg2(&mut dst, src);
            write_xmm(mcontext, decoded.dest, dst);
        }
        Sha1Op::Nexte => {
            let src1 = read_xmm(mcontext, decoded.dest);
            let mut dst = read_xmm(mcontext, decoded.src);
            sha1_nexte_apply_src1(&mut dst, src1);
            write_xmm(mcontext, decoded.dest, dst);
        }
        Sha1Op::Rnds4 { imm } => {
            let mut dst = read_xmm(mcontext, decoded.dest);
            let src = read_xmm(mcontext, decoded.src);
            sha1_rnds4(&mut dst, src, imm);
            write_xmm(mcontext, decoded.dest, dst);
        }
    }
    mcontext.__ss.__rip = rip.wrapping_add(u64::from(decoded.length));
    true
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn read_xmm(mcontext: &libc::__darwin_mcontext64, index: u8) -> [u32; 4] {
    let xmm = xmm_reg(mcontext, index);
    [
        u32::from_le_bytes(xmm[0..4].try_into().expect("4 bytes")),
        u32::from_le_bytes(xmm[4..8].try_into().expect("4 bytes")),
        u32::from_le_bytes(xmm[8..12].try_into().expect("4 bytes")),
        u32::from_le_bytes(xmm[12..16].try_into().expect("4 bytes")),
    ]
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn write_xmm(mcontext: &mut libc::__darwin_mcontext64, index: u8, value: [u32; 4]) {
    let xmm = xmm_reg_mut(mcontext, index);
    for (lane, word) in value.into_iter().enumerate() {
        xmm[lane * 4..lane * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn xmm_reg(mcontext: &libc::__darwin_mcontext64, index: u8) -> &[u8; 16] {
    // Darwin stores __fpu_xmm0..__fpu_xmm15 contiguously as 16-byte records.
    let base = std::ptr::from_ref(&mcontext.__fs.__fpu_xmm0).cast::<u8>();
    // SAFETY: index is 0..15 from ModRM+REX decode; each slot is 16 bytes.
    unsafe { &*base.add(usize::from(index) * 16).cast::<[u8; 16]>() }
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn xmm_reg_mut(mcontext: &mut libc::__darwin_mcontext64, index: u8) -> &mut [u8; 16] {
    let base = std::ptr::from_mut(&mut mcontext.__fs.__fpu_xmm0).cast::<u8>();
    // SAFETY: index is 0..15 from ModRM+REX decode; each slot is 16 bytes.
    unsafe { &mut *base.add(usize::from(index) * 16).cast::<[u8; 16]>() }
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
unsafe fn chain_previous_sigill(
    signal: libc::c_int,
    information: *mut libc::siginfo_t,
    context: *mut std::ffi::c_void,
) {
    let previous = unsafe { PREVIOUS_SIGILL.get() };
    let handler = previous.sa_sigaction;
    if handler == libc::SIG_DFL || handler == libc::SIG_IGN || handler == 0 {
        // SAFETY: restore default disposition and re-raise so the process still crashes observably.
        unsafe {
            libc::signal(signal, libc::SIG_DFL);
            libc::raise(signal);
        }
        return;
    }
    if previous.sa_flags & libc::SA_SIGINFO != 0 {
        let callback: extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut std::ffi::c_void) =
            unsafe { std::mem::transmute(handler) };
        callback(signal, information, context);
    } else {
        let callback: extern "C" fn(libc::c_int) = unsafe { std::mem::transmute(handler) };
        callback(signal);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Sha1Op {
    Msg1,
    Msg2,
    Nexte,
    Rnds4 { imm: u8 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DecodedSha1 {
    op: Sha1Op,
    dest: u8,
    src: u8,
    length: u8,
}

/// Decodes a register-form SHA-1 NI instruction at `bytes`.
fn decode_sha1_ni(bytes: &[u8]) -> Option<DecodedSha1> {
    if bytes.len() < 4 {
        return None;
    }
    let mut index = 0_usize;
    let mut rex = 0_u8;
    if (0x40..=0x4f).contains(&bytes[index]) {
        rex = bytes[index];
        index += 1;
    }
    if index + 3 >= bytes.len() {
        return None;
    }
    if bytes[index] == 0x0f && bytes[index + 1] == 0x38 {
        let opcode = bytes[index + 2];
        let modrm = bytes[index + 3];
        if modrm >> 6 != 0b11 {
            return None;
        }
        let op = match opcode {
            0xc8 => Sha1Op::Nexte,
            0xc9 => Sha1Op::Msg1,
            0xca => Sha1Op::Msg2,
            _ => return None,
        };
        let mut dest = (modrm >> 3) & 0b111;
        let mut src = modrm & 0b111;
        if rex & 0x04 != 0 {
            dest |= 8;
        }
        if rex & 0x01 != 0 {
            src |= 8;
        }
        return Some(DecodedSha1 {
            op,
            dest,
            src,
            length: (index + 4) as u8,
        });
    }
    if bytes[index] == 0x0f && bytes[index + 1] == 0x3a && bytes[index + 2] == 0xcc {
        if index + 4 >= bytes.len() {
            return None;
        }
        let modrm = bytes[index + 3];
        if modrm >> 6 != 0b11 {
            return None;
        }
        let mut dest = (modrm >> 3) & 0b111;
        let mut src = modrm & 0b111;
        if rex & 0x04 != 0 {
            dest |= 8;
        }
        if rex & 0x01 != 0 {
            src |= 8;
        }
        let imm = bytes[index + 4] & 0b11;
        return Some(DecodedSha1 {
            op: Sha1Op::Rnds4 { imm },
            dest,
            src,
            length: (index + 5) as u8,
        });
    }
    None
}

/// `SHA1MSG1 xmm1, xmm2` — Bochs / Intel SDM dword-lane semantics (`lane0` = bits\[31:0\]).
pub(crate) fn sha1_msg1(dst: &mut [u32; 4], src: [u32; 4]) {
    dst[3] ^= dst[1];
    dst[2] ^= dst[0];
    dst[1] ^= src[3];
    dst[0] ^= src[2];
}

/// `SHA1MSG2 xmm1, xmm2` — Bochs evaluation order (lane 0 XORs the updated lane 3).
pub(crate) fn sha1_msg2(dst: &mut [u32; 4], src: [u32; 4]) {
    dst[3] = (dst[3] ^ src[2]).rotate_left(1);
    dst[2] = (dst[2] ^ src[1]).rotate_left(1);
    dst[1] = (dst[1] ^ src[0]).rotate_left(1);
    dst[0] = (dst[0] ^ dst[3]).rotate_left(1);
}

/// Applies SRC1 into a DEST that already holds SRC2 for `SHA1NEXTE`.
fn sha1_nexte_apply_src1(dest_from_src2: &mut [u32; 4], src1: [u32; 4]) {
    dest_from_src2[3] = dest_from_src2[3].wrapping_add(src1[3].rotate_left(30));
}

/// `SHA1NEXTE xmm1, xmm2`: `dst` is SRC1 on entry; result replaces `dst`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn sha1_nexte(dst: &mut [u32; 4], src: [u32; 4]) {
    let src1 = *dst;
    *dst = src;
    sha1_nexte_apply_src1(dst, src1);
}

fn sha_f(b: u32, c: u32, d: u32, imm: u8) -> u32 {
    match imm & 0b11 {
        0 => (b & c) ^ (!b & d),
        1 | 3 => b ^ c ^ d,
        2 => (b & c) ^ (b & d) ^ (c & d),
        _ => unreachable!("imm masked to 2 bits"),
    }
}

/// `SHA1RNDS4 xmm1, xmm2, imm8`.
pub(crate) fn sha1_rnds4(dst: &mut [u32; 4], src: [u32; 4], imm: u8) {
    const K: [u32; 4] = [0x5A82_7999, 0x6ED9_EBA1, 0x8F1B_BCDC, 0xCA62_C1D6];
    let imm = imm & 0b11;
    let k = K[usize::from(imm)];
    let mut a = dst[3];
    let mut b = dst[2];
    let mut c = dst[1];
    let mut d = dst[0];
    let mut e = 0_u32;
    let w = [src[3], src[2], src[1], src[0]];
    for word in w {
        let a_next = sha_f(b, c, d, imm)
            .wrapping_add(a.rotate_left(5))
            .wrapping_add(word)
            .wrapping_add(e)
            .wrapping_add(k);
        e = d;
        d = c;
        c = b.rotate_left(30);
        b = a;
        a = a_next;
    }
    dst[3] = a;
    dst[2] = b;
    dst[1] = c;
    dst[0] = d;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_poppy_fault_site() {
        // `sha1msg1 xmm7, xmm1` at the Poppy SIGILL site.
        let decoded = decode_sha1_ni(&[0x0f, 0x38, 0xc9, 0xf9]).unwrap();
        assert_eq!(decoded.op, Sha1Op::Msg1);
        assert_eq!(decoded.dest, 7);
        assert_eq!(decoded.src, 1);
        assert_eq!(decoded.length, 4);
    }

    #[test]
    fn decodes_rex_b_nexte() {
        let decoded = decode_sha1_ni(&[0x41, 0x0f, 0x38, 0xc8, 0xc8]).unwrap();
        assert_eq!(decoded.op, Sha1Op::Nexte);
        assert_eq!(decoded.dest, 1);
        assert_eq!(decoded.src, 8);
        assert_eq!(decoded.length, 5);
    }

    #[test]
    fn decodes_rnds4() {
        let decoded = decode_sha1_ni(&[0x0f, 0x3a, 0xcc, 0xf5, 0x00]).unwrap();
        assert_eq!(decoded.op, Sha1Op::Rnds4 { imm: 0 });
        assert_eq!(decoded.dest, 6);
        assert_eq!(decoded.src, 5);
        assert_eq!(decoded.length, 5);
    }

    #[test]
    fn msg1_matches_bochs_lane_xor() {
        let mut dst = [0x1111_1111, 0x2222_2222, 0x3333_3333, 0x4444_4444];
        let src = [0x5555_5555, 0x6666_6666, 0x7777_7777, 0x8888_8888];
        sha1_msg1(&mut dst, src);
        assert_eq!(dst[3], 0x4444_4444 ^ 0x2222_2222);
        assert_eq!(dst[2], 0x3333_3333 ^ 0x1111_1111);
        assert_eq!(dst[1], 0x2222_2222 ^ 0x8888_8888);
        assert_eq!(dst[0], 0x1111_1111 ^ 0x7777_7777);
    }

    #[test]
    fn msg2_uses_updated_high_lane_for_low_lane() {
        let mut dst = [0x10, 0x20, 0x30, 0x40];
        let src = [0x01, 0x02, 0x03, 0x04];
        sha1_msg2(&mut dst, src);
        let high = (0x40_u32 ^ 0x03).rotate_left(1);
        assert_eq!(dst[3], high);
        assert_eq!(dst[2], (0x30_u32 ^ 0x02).rotate_left(1));
        assert_eq!(dst[1], (0x20_u32 ^ 0x01).rotate_left(1));
        assert_eq!(dst[0], (0x10_u32 ^ high).rotate_left(1));
    }

    #[test]
    fn nexte_adds_rotated_src1_high_dword() {
        let mut dst = [0, 0, 0, 5];
        sha1_nexte(&mut dst, [9, 8, 7, 6]);
        assert_eq!(dst[3], 6_u32.wrapping_add(5_u32.rotate_left(30)));
        assert_eq!(&dst[..3], &[9, 8, 7]);
    }

    #[test]
    fn rnds4_changes_abcd_state() {
        let mut dst = [0x1032_5476, 0x98BA_DCFE, 0xEFCD_AB89, 0x6745_2301];
        let before = dst;
        sha1_rnds4(&mut dst, [1, 2, 3, 4], 0);
        assert_ne!(dst, before);
    }
}
