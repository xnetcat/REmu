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
        Sha1Op::Sha256Msg1 => {
            let mut dst = read_xmm(mcontext, decoded.dest);
            let src = read_xmm(mcontext, decoded.src);
            sha256_msg1(&mut dst, src);
            write_xmm(mcontext, decoded.dest, dst);
        }
        Sha1Op::Sha256Msg2 => {
            let mut dst = read_xmm(mcontext, decoded.dest);
            let src = read_xmm(mcontext, decoded.src);
            sha256_msg2(&mut dst, src);
            write_xmm(mcontext, decoded.dest, dst);
        }
        Sha1Op::Sha256Rnds2 => {
            let mut dst = read_xmm(mcontext, decoded.dest);
            let src = read_xmm(mcontext, decoded.src);
            // The round constants are an implicit XMM0 operand, not encoded in the ModRM.
            let constants = read_xmm(mcontext, 0);
            sha256_rnds2(&mut dst, src, constants);
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
    /// `SHA256RNDS2 xmm1, xmm2` — two SHA-256 rounds; the round constants come from XMM0.
    Sha256Rnds2,
    /// `SHA256MSG1 xmm1, xmm2` — message schedule, sigma0 stage.
    Sha256Msg1,
    /// `SHA256MSG2 xmm1, xmm2` — message schedule, sigma1 stage.
    Sha256Msg2,
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
            0xcb => Sha1Op::Sha256Rnds2,
            0xcc => Sha1Op::Sha256Msg1,
            0xcd => Sha1Op::Sha256Msg2,
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
/// SHA-256 `sigma0`: the message-schedule mixer applied to W[i+1].
const fn sha256_small_sigma0(value: u32) -> u32 {
    value.rotate_right(7) ^ value.rotate_right(18) ^ (value >> 3)
}

/// SHA-256 `sigma1`: the message-schedule mixer applied to W[i+14].
const fn sha256_small_sigma1(value: u32) -> u32 {
    value.rotate_right(17) ^ value.rotate_right(19) ^ (value >> 10)
}

/// SHA-256 `Sigma0`, the round mixer applied to state word A.
const fn sha256_big_sigma0(value: u32) -> u32 {
    value.rotate_right(2) ^ value.rotate_right(13) ^ value.rotate_right(22)
}

/// SHA-256 `Sigma1`, the round mixer applied to state word E.
const fn sha256_big_sigma1(value: u32) -> u32 {
    value.rotate_right(6) ^ value.rotate_right(11) ^ value.rotate_right(25)
}

const fn sha256_choose(e: u32, f: u32, g: u32) -> u32 {
    (e & f) ^ (!e & g)
}

const fn sha256_majority(a: u32, b: u32, c: u32) -> u32 {
    (a & b) ^ (a & c) ^ (b & c)
}

/// `SHA256MSG1 xmm1, xmm2` — Intel SDM: dword lane `i` becomes `W[i] + sigma0(W[i+1])`, where
/// `W[4]` is the low lane of the source.
pub(crate) fn sha256_msg1(dst: &mut [u32; 4], src: [u32; 4]) {
    let w4 = src[0];
    let next = [dst[1], dst[2], dst[3], w4];
    for lane in 0..4 {
        dst[lane] = dst[lane].wrapping_add(sha256_small_sigma0(next[lane]));
    }
}

/// `SHA256MSG2 xmm1, xmm2` — Intel SDM: the sigma1 stage, whose two upper results feed the two
/// lower ones, so the lanes must be evaluated in order.
pub(crate) fn sha256_msg2(dst: &mut [u32; 4], src: [u32; 4]) {
    let w14 = src[2];
    let w15 = src[3];
    let w16 = dst[0].wrapping_add(sha256_small_sigma1(w14));
    let w17 = dst[1].wrapping_add(sha256_small_sigma1(w15));
    let w18 = dst[2].wrapping_add(sha256_small_sigma1(w16));
    let w19 = dst[3].wrapping_add(sha256_small_sigma1(w17));
    *dst = [w16, w17, w18, w19];
}

/// `SHA256RNDS2 xmm1, xmm2, <XMM0>` — two SHA-256 rounds.
///
/// Intel SDM operand mapping: the destination carries `{C, D, G, H}` from high to low lane and
/// the source `{A, B, E, F}`; the pre-added `W + K` pair arrives in the low two lanes of the
/// implicit XMM0. The result is `{A, B, E, F}` after two rounds.
pub(crate) fn sha256_rnds2(dst: &mut [u32; 4], src: [u32; 4], constants: [u32; 4]) {
    let (mut a, mut b) = (src[3], src[2]);
    let (mut c, mut d) = (dst[3], dst[2]);
    let (mut e, mut f) = (src[1], src[0]);
    let (mut g, mut h) = (dst[1], dst[0]);
    for round_constant in [constants[0], constants[1]] {
        let common = sha256_choose(e, f, g)
            .wrapping_add(sha256_big_sigma1(e))
            .wrapping_add(round_constant)
            .wrapping_add(h);
        let next_a = common
            .wrapping_add(sha256_majority(a, b, c))
            .wrapping_add(sha256_big_sigma0(a));
        let next_e = common.wrapping_add(d);
        h = g;
        g = f;
        f = e;
        e = next_e;
        d = c;
        c = b;
        b = a;
        a = next_a;
    }
    *dst = [f, e, b, a];
}

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
    fn sha256_ni_opcodes_decode_with_their_operands() {
        // The three SHA-256 opcodes share the 0f 38 map with SHA-1.
        for (opcode, expected) in [
            (0xcb_u8, Sha1Op::Sha256Rnds2),
            (0xcc, Sha1Op::Sha256Msg1),
            (0xcd, Sha1Op::Sha256Msg2),
        ] {
            let decoded = decode_sha1_ni(&[0x0f, 0x38, opcode, 0xd1]).expect("decodes");
            assert_eq!(decoded.op, expected);
            assert_eq!(decoded.dest, 2);
            assert_eq!(decoded.src, 1);
            assert_eq!(decoded.length, 4);
        }
        // REX.RB reaches the upper eight registers.
        let decoded = decode_sha1_ni(&[0x45, 0x0f, 0x38, 0xcc, 0xc1]).expect("decodes");
        assert_eq!(decoded.op, Sha1Op::Sha256Msg1);
        assert_eq!(decoded.dest, 8);
        assert_eq!(decoded.src, 9);
        assert_eq!(decoded.length, 5);
    }

    #[test]
    fn sha256_ni_primitives_hash_the_empty_string_correctly() {
        // Drive a full SHA-256 compression of the one-block padded empty message using only the
        // emulated instructions, and compare against the published digest. Any lane-order or
        // rotation error changes the digest, so this pins the semantics rather than the shape.
        let mut w = [0_u32; 64];
        w[0] = 0x8000_0000;
        for index in 16..64 {
            // Build the schedule with the same primitives the instructions use.
            let s0 = sha256_small_sigma0(w[index - 15]);
            let s1 = sha256_small_sigma1(w[index - 2]);
            w[index] = w[index - 16]
                .wrapping_add(s0)
                .wrapping_add(w[index - 7])
                .wrapping_add(s1);
        }
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let initial: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];
        // SHA256RNDS2 lane layout: dst = {H, G, D, C}, src = {F, E, B, A}.
        let mut abef = [initial[5], initial[4], initial[1], initial[0]];
        let mut cdgh = [initial[7], initial[6], initial[3], initial[2]];
        for round in (0..64).step_by(2) {
            let constants = [
                w[round].wrapping_add(K[round]),
                w[round + 1].wrapping_add(K[round + 1]),
                0,
                0,
            ];
            let previous_abef = abef;
            sha256_rnds2(&mut cdgh, previous_abef, constants);
            std::mem::swap(&mut abef, &mut cdgh);
        }
        let digest = [
            abef[3].wrapping_add(initial[0]),
            abef[2].wrapping_add(initial[1]),
            cdgh[3].wrapping_add(initial[2]),
            cdgh[2].wrapping_add(initial[3]),
            abef[1].wrapping_add(initial[4]),
            abef[0].wrapping_add(initial[5]),
            cdgh[1].wrapping_add(initial[6]),
            cdgh[0].wrapping_add(initial[7]),
        ];
        assert_eq!(
            digest,
            [
                0xe3b0c442, 0x98fc1c14, 0x9afbf4c8, 0x996fb924, 0x27ae41e4, 0x649b934c, 0xa495991b,
                0x7852b855
            ],
            "SHA-256 of the empty string"
        );
    }

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
