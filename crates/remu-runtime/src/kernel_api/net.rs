// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use crate::{
    low_i32, low_u32, read_guest_bytes, read_guest_cstring, signed_i32_return, write_guest_bytes,
};
use std::net::{Ipv4Addr, Ipv6Addr};

use super::{guest_range_is_readable, set_guest_errno};

/// Address families the textual address conversions understand.
pub(super) const AF_INET: i32 = 2;

pub(super) const AF_INET6: i32 = 28;

/// Bytes an address of each family occupies in its binary form.
pub(super) const IPV4_ADDRESS_BYTES: usize = 4;

pub(super) const IPV6_ADDRESS_BYTES: usize = 16;

/// Formats a binary internet address into the caller's text buffer.
///
/// Returns the buffer it wrote on success and a null pointer otherwise.
pub(super) fn inet_ntop(arguments: [u64; 6]) -> u64 {
    let (source, destination) = (arguments[1], arguments[2]);
    let Ok(capacity) = usize::try_from(low_u32(arguments[3])) else {
        set_guest_errno(libc::ENOSPC);
        return 0;
    };
    let family = low_i32(arguments[0]);
    if source == 0 || destination == 0 {
        set_guest_errno(libc::EFAULT);
        return 0;
    }
    let text = match family {
        AF_INET => {
            if !guest_range_is_readable(source, IPV4_ADDRESS_BYTES) {
                set_guest_errno(libc::EFAULT);
                return 0;
            }
            let mut octets = [0_u8; IPV4_ADDRESS_BYTES];
            read_guest_bytes(source, &mut octets);
            Ipv4Addr::from(octets).to_string()
        }
        AF_INET6 => {
            if !guest_range_is_readable(source, IPV6_ADDRESS_BYTES) {
                set_guest_errno(libc::EFAULT);
                return 0;
            }
            let mut octets = [0_u8; IPV6_ADDRESS_BYTES];
            read_guest_bytes(source, &mut octets);
            Ipv6Addr::from(octets).to_string()
        }
        _ => {
            set_guest_errno(libc::EAFNOSUPPORT);
            return 0;
        }
    };
    if text.len() + 1 > capacity {
        set_guest_errno(libc::ENOSPC);
        return 0;
    }
    let mut bytes = text.into_bytes();
    bytes.push(0);
    write_guest_bytes(destination, &bytes);
    destination
}

/// Parses a textual internet address into its binary form.
///
/// Reports one for an address it accepted, zero for text that is not an address of that family,
/// and minus one for a family it does not know.
pub(super) fn inet_pton(arguments: [u64; 6]) -> u64 {
    let (source, destination) = (arguments[1], arguments[2]);
    let family = low_i32(arguments[0]);
    if !matches!(family, AF_INET | AF_INET6) {
        set_guest_errno(libc::EAFNOSUPPORT);
        return signed_i32_return(-1);
    }
    if source == 0 || destination == 0 {
        set_guest_errno(libc::EFAULT);
        return signed_i32_return(-1);
    }
    let Some(text) = read_guest_cstring(source) else {
        set_guest_errno(libc::EFAULT);
        return signed_i32_return(-1);
    };
    if family == AF_INET {
        let Ok(address) = text.parse::<Ipv4Addr>() else {
            return 0;
        };
        write_guest_bytes(destination, &address.octets());
    } else {
        let Ok(address) = text.parse::<Ipv6Addr>() else {
            return 0;
        };
        write_guest_bytes(destination, &address.octets());
    }
    1
}
