// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Static linking of a bundled PS5 process image.
//!
//! Mapping, relocation, export collection and import resolution all happen once, before the guest
//! runs, so nothing here sits on a hot path. What it produces does: every unresolved guest import
//! becomes a stub that jumps into the runtime's dispatcher, and the addresses those stubs carry
//! are fixed here.

use std::path::PathBuf;

use remu_core::GuestAddress;
use remu_loader::{InspectError, ModuleDependencyGraphError};
use remu_memory::{GuestImage, MappingError};
use thiserror::Error;

mod exports;
pub(crate) mod fs_decode;
mod image;
mod imports;
mod modules;
mod native_symbols;
mod relocation;

pub use self::image::{GuestTls, LinkedImage, ProbeOutcome};
pub use self::imports::{
    ImportResolutionKind, ImportResolutionRecord, ImportResolutionStats, ImportStub,
};
pub use self::modules::ModuleLinkReport;
pub use self::relocation::RelocationStats;

#[cfg(test)]
pub(crate) use self::imports::{IMPORT_STUB_SIZE_USIZE, write_import_stub};
pub(crate) use self::modules::{
    LinkedGuestModule, MACOS_GUEST_PAGE_ALIGNMENT, PT_TLS, collect_initializer_array,
    discover_runtime_module_paths,
};
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub(crate) use self::native_symbols::native_cxx_runtime;
pub(crate) use self::relocation::runtime_address_at;

#[cfg(test)]
use self::exports::encode_sce_id;
use self::exports::{
    ProcessSymbolKey, collect_process_exports, main_base_export_input, module_exports_by_nid,
    process_symbol_key,
};
use self::imports::{IMPORT_STUB_SIZE, collect_import_resolutions, create_import_stubs};
use self::modules::{
    LinkedModuleInput, append_object_initializers, calculate_process_auxiliary_reserve,
    map_bundled_modules, resolve_process_param,
};
use self::native_symbols::{native_import_target, synthetic_data_import_target};
use self::relocation::{
    TlsRelocationContext, add_relocation_stats, apply_relocations_at, collect_tls_symbols,
    defined_symbol_address_at,
};
pub(crate) use self::relocation::PendingIfunc;

#[derive(Debug, Error)]
pub enum LinkError {
    #[error(transparent)]
    Mapping(#[from] MappingError),
    #[error(transparent)]
    Inspect(#[from] InspectError),
    #[error(transparent)]
    ModuleGraph(#[from] ModuleDependencyGraphError),
    #[error("executable does not contain an ELF image")]
    MissingElf,
    #[error("executable does not contain dynamic-linker metadata")]
    MissingDynamicMetadata,
    #[error("import stub allocation size overflow")]
    StubAllocationOverflow,
    #[error("relocation references missing symbol {index}")]
    MissingSymbol { index: u32 },
    #[error("unresolved non-weak symbol {index} has no import stub")]
    MissingImportStub { index: u32 },
    #[error("dynamic symbol {symbol:?} refers to an unknown Sony library or module namespace")]
    InvalidSymbolNamespace { symbol: String },
    #[error("unsupported x86-64 relocation type {relocation_type} at {offset}")]
    UnsupportedRelocation {
        relocation_type: u32,
        offset: GuestAddress,
    },
    #[error("guest address arithmetic overflow")]
    AddressOverflow,
    #[error("guest TLS alignment 0x{0:x} is not a power of two")]
    InvalidTlsAlignment(u64),
    #[error(
        "guest static TLS requires 0x{required:x} bytes, exceeding the 0x{available:x} startup reservation"
    )]
    StaticTlsTooLarge { required: u64, available: u64 },
    #[error("executable region size cannot be represented on this host")]
    ExecutableRegionTooLarge,
    #[error("TLS helper at {helper} is outside rel32 range of guest instruction {instruction}")]
    TlsHelperOutOfRange {
        instruction: GuestAddress,
        helper: GuestAddress,
    },
    #[error("native guest probing requires an x86-64 process")]
    ProbeRequiresX86_64,
    #[error("process image name contains an interior NUL byte")]
    InvalidProcessImageName,
    #[error("unsupported guest .eh_frame pointer encoding 0x{0:02x}")]
    UnsupportedEhFrameEncoding(u8),
    #[error("malformed guest .eh_frame data: {0}")]
    MalformedEhFrame(&'static str),
    #[error("could not discover runtime modules under {path}: {source}")]
    RuntimeModuleDiscovery {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("guest {kind} byte size 0x{size:x} is not a multiple of eight")]
    InvalidInitializerArraySize { kind: &'static str, size: u64 },
    #[error("guest initializer address {address} is outside the process image")]
    InitializerOutsideImage { address: GuestAddress },
    #[error("native macOS C++ unwind runtime is unavailable")]
    NativeCxxRuntimeUnavailable,
    #[error("native macOS unwind symbol {0} is unavailable")]
    MissingNativeUnwindSymbol(&'static str),
}

pub(crate) fn add_address(address: GuestAddress, offset: u64) -> Result<GuestAddress, LinkError> {
    Ok(GuestAddress::new(
        address
            .get()
            .checked_add(offset)
            .ok_or(LinkError::AddressOverflow)?,
    ))
}

pub(crate) fn subtract_address(
    address: GuestAddress,
    offset: u64,
) -> Result<GuestAddress, LinkError> {
    Ok(GuestAddress::new(
        address
            .get()
            .checked_sub(offset)
            .ok_or(LinkError::AddressOverflow)?,
    ))
}

pub(crate) fn write_u64(
    memory: &mut GuestImage,
    address: GuestAddress,
    value: u64,
) -> Result<(), LinkError> {
    memory.write(address, &value.to_le_bytes())?;
    Ok(())
}

#[cfg(test)]
mod linker_metadata_tests;
