// SPDX-License-Identifier: GPL-2.0-or-later

use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TranslationState {
    Native,
    Translated,
    Unknown,
}

impl fmt::Display for TranslationState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Native => formatter.write_str("native"),
            Self::Translated => formatter.write_str("translated"),
            Self::Unknown => formatter.write_str("unknown"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostInfo {
    pub operating_system: &'static str,
    pub architecture: &'static str,
    pub process_translation: TranslationState,
}

#[must_use]
pub fn host_info() -> HostInfo {
    HostInfo {
        operating_system: std::env::consts::OS,
        architecture: std::env::consts::ARCH,
        process_translation: process_translation_state(),
    }
}

#[cfg(target_os = "macos")]
fn process_translation_state() -> TranslationState {
    use std::ffi::c_void;
    use std::mem::size_of;

    let mut translated: libc::c_int = 0;
    let mut translated_size = size_of::<libc::c_int>();
    let name = c"sysctl.proc_translated";

    // SAFETY: `translated` and `translated_size` point to valid writable storage, the sysctl name
    // is NUL-terminated, and no new value is supplied.
    let result = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&raw mut translated).cast::<c_void>(),
            &raw mut translated_size,
            std::ptr::null_mut(),
            0,
        )
    };

    if result == 0 {
        if translated == 1 {
            TranslationState::Translated
        } else {
            TranslationState::Native
        }
    } else if std::env::consts::ARCH == "aarch64" {
        TranslationState::Native
    } else {
        TranslationState::Unknown
    }
}

#[cfg(not(target_os = "macos"))]
const fn process_translation_state() -> TranslationState {
    TranslationState::Native
}
