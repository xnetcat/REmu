// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! `libSceRandom` — cryptographically secure random bytes for the guest.

use crate::{process_guest_range_has_read_access, sce_kernel_error, write_guest_bytes};

#[cfg(windows)]
const BCRYPT_USE_SYSTEM_PREFERRED_RNG: u32 = 0x2;

#[cfg(windows)]
#[link(name = "bcrypt")]
unsafe extern "system" {
    fn BCryptGenRandom(
        algorithm: *mut std::ffi::c_void,
        buffer: *mut u8,
        byte_count: u32,
        flags: u32,
    ) -> i32;
}

/// Maximum request size accepted by `sceRandomGetRandomNumber`.
pub(crate) const SCE_RANDOM_MAX_SIZE: usize = 64;
/// Returned when the buffer is null (with nonzero size) or `size` exceeds [`SCE_RANDOM_MAX_SIZE`].
pub(crate) const SCE_RANDOM_ERROR_INVALID: u32 = 0x817c_0016;
/// Returned when the host CSPRNG cannot satisfy the request.
pub(crate) const SCE_RANDOM_ERROR_OUT_OF_RESOURCES: u32 = 0x817c_001c;

/// `int sceRandomGetRandomNumber(void *buf, size_t size)`.
///
/// Fills `buf` with up to 64 cryptographically secure random bytes. The SDK documents this as the
/// platform entropy source used by libsodium and other crypto stacks on PS4/PS5; using `rand()`
/// would be incorrect.
pub(crate) fn sce_random_get_random_number(arguments: [u64; 6]) -> u64 {
    let buffer = arguments[0];
    let size = arguments[1] as usize;
    if size > SCE_RANDOM_MAX_SIZE {
        return sce_kernel_error(SCE_RANDOM_ERROR_INVALID);
    }
    if size == 0 {
        return 0;
    }
    if buffer == 0 || !process_guest_range_has_read_access(buffer, size) {
        return sce_kernel_error(SCE_RANDOM_ERROR_INVALID);
    }

    let mut bytes = vec![0_u8; size];
    if let Err(error) = fill_secure_random(&mut bytes) {
        eprintln!("[REMu][RANDOM] getentropy failed size={size}: {error}");
        return sce_kernel_error(SCE_RANDOM_ERROR_OUT_OF_RESOURCES);
    }
    write_guest_bytes(buffer, &bytes);
    0
}

fn fill_secure_random(bytes: &mut [u8]) -> Result<(), String> {
    #[cfg(unix)]
    {
        // SAFETY: `bytes` is a valid mutable host buffer; getentropy writes at most buflen bytes.
        let status =
            unsafe { libc::getentropy(bytes.as_mut_ptr().cast::<libc::c_void>(), bytes.len()) };
        if status == 0 {
            return Ok(());
        }
        Err(std::io::Error::last_os_error().to_string())
    }
    #[cfg(windows)]
    {
        let byte_count = u32::try_from(bytes.len())
            .map_err(|_| "secure random request is too large".to_owned())?;
        // SAFETY: the system-preferred provider requires a null algorithm handle and writes at
        // most byte_count bytes into the live slice.
        let status = unsafe {
            BCryptGenRandom(
                std::ptr::null_mut(),
                bytes.as_mut_ptr(),
                byte_count,
                BCRYPT_USE_SYSTEM_PREFERRED_RNG,
            )
        };
        if status == 0 {
            return Ok(());
        }
        Err(format!("BCryptGenRandom failed with NTSTATUS {status:#x}"))
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = bytes;
        Err("secure random is unavailable on this host".to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::read_guest_bytes;

    #[test]
    fn rejects_oversized_requests() {
        assert_eq!(
            sce_random_get_random_number([0x1000, (SCE_RANDOM_MAX_SIZE + 1) as u64, 0, 0, 0, 0]),
            sce_kernel_error(SCE_RANDOM_ERROR_INVALID)
        );
    }

    #[test]
    fn zero_length_succeeds_without_touching_memory() {
        assert_eq!(sce_random_get_random_number([0, 0, 0, 0, 0, 0]), 0);
    }

    #[test]
    fn null_buffer_with_nonzero_size_is_invalid() {
        assert_eq!(
            sce_random_get_random_number([0, 16, 0, 0, 0, 0]),
            sce_kernel_error(SCE_RANDOM_ERROR_INVALID)
        );
    }

    #[test]
    fn fills_guest_buffer_with_non_constant_bytes() {
        let mut storage = vec![0_u8; 64];
        let guest = storage.as_mut_ptr() as u64;
        assert_eq!(sce_random_get_random_number([guest, 32, 0, 0, 0, 0]), 0);
        let mut out = vec![0_u8; 32];
        read_guest_bytes(guest, &mut out);
        // A constant-zero fill would indicate the CSPRNG path failed silently.
        assert!(out.iter().any(|byte| *byte != 0));
        let mut second = vec![0_u8; 32];
        assert_eq!(sce_random_get_random_number([guest, 32, 0, 0, 0, 0]), 0);
        read_guest_bytes(guest, &mut second);
        assert_ne!(out, second);
        assert_eq!(storage.len(), 64);
    }
}
