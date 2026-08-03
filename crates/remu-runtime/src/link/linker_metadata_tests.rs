use std::collections::BTreeMap;

use remu_core::GuestAddress;
use remu_loader::{DynamicInfo, Elf64Rela, Elf64Symbol, SceLibraryReference, SceModuleReference};

use super::relocation::{
    R_X86_64_DTPMOD64, R_X86_64_DTPOFF64, R_X86_64_TPOFF64, TlsRelocationContext,
    TlsSymbolResolution, resolve_tls_reference, tls_relocation_value,
};
use super::{ProcessSymbolKey, encode_sce_id, process_symbol_key};
use crate::sce_symbol_nid;

#[test]
fn encodes_sony_namespace_ids_in_the_dynamic_symbol_alphabet() {
    assert_eq!(encode_sce_id(0), "A");
    assert_eq!(encode_sce_id(63), "-");
    assert_eq!(encode_sce_id(64), "BA");
    assert_eq!(encode_sce_id(0x0fff), "--");
    assert_eq!(encode_sce_id(0x1000), "BAA");
    assert_eq!(encode_sce_id(u16::MAX), "P--");
}

#[test]
fn hashes_runtime_symbol_names_to_sony_nids() {
    assert_eq!(sce_symbol_nid("sceKernelOpen"), "1G3lF1Gg1k8");
    assert_eq!(sce_symbol_nid("sceKernelClose"), "UK2Tl2DWUns");
}

#[test]
fn resolves_symbols_by_namespace_versions_and_type_not_local_encoded_ids() {
    let imported = dynamic_namespace(false, 4, 3, 7, 2, 1);
    let exported = dynamic_namespace(true, 11, 10, 7, 2, 1);
    let import_symbol = symbol("someNid#E#D", 0x12, 0);
    let export_symbol = symbol("someNid#L#K", 0x12, 1);

    let import_key = process_symbol_key(&imported, &import_symbol).unwrap();
    let export_key = process_symbol_key(&exported, &export_symbol).unwrap();
    assert_eq!(import_key, export_key);
    assert!(matches!(import_key, ProcessSymbolKey::Scoped { .. }));

    let different_version = dynamic_namespace(true, 11, 10, 8, 2, 1);
    assert_ne!(
        process_symbol_key(&different_version, &export_symbol).unwrap(),
        export_key
    );
    let object_export = symbol("someNid#L#K", 0x11, 1);
    assert_ne!(
        process_symbol_key(&exported, &object_export).unwrap(),
        export_key
    );
    assert!(process_symbol_key(&exported, &symbol("someNid#Z#Z", 0x12, 1)).is_err());
}

#[test]
fn tls_relocation_values_are_module_offset_and_thread_pointer_relative() {
    let resolution = TlsSymbolResolution {
        module_id: 3,
        tls_offset: 0x40,
        static_offset: 0x100,
    };
    // DTPMOD64 names the module and ignores the addend.
    assert_eq!(tls_relocation_value(R_X86_64_DTPMOD64, &resolution, 0x1000), 3);
    // DTPOFF64 is the offset within the module's own TLS block.
    assert_eq!(tls_relocation_value(R_X86_64_DTPOFF64, &resolution, 8), 0x48);
    // TPOFF64 is that offset expressed below the thread pointer (variant II).
    assert_eq!(
        tls_relocation_value(R_X86_64_TPOFF64, &resolution, 8),
        0x48_u64.wrapping_sub(0x100)
    );
}

#[test]
fn a_cross_module_tls_reference_resolves_through_its_defining_module() {
    // A module references an undefined TLS symbol whose storage lives in another module (id 5).
    let mut referencing = dynamic_namespace(false, 4, 3, 7, 2, 1);
    referencing.symbols = vec![symbol("", 0, 0), symbol("tlsNid#E#D", 0x16, 0)];
    let key = process_symbol_key(&referencing, &referencing.symbols[1]).unwrap();
    let mut tls_symbols = BTreeMap::new();
    tls_symbols.insert(
        key,
        TlsSymbolResolution {
            module_id: 5,
            tls_offset: 0x20,
            static_offset: 0x200,
        },
    );
    let context = TlsRelocationContext {
        module_id: 2,
        module_static_offset: 0x80,
        tls_symbols: &tls_symbols,
    };

    let cross = resolve_tls_reference(&referencing, &context, &tls_relocation(R_X86_64_DTPMOD64, 1, 0))
        .unwrap();
    // The defining module (5), not the referencing module (2), names the DTPMOD64 slot.
    assert_eq!(cross.module_id, 5);
    assert_eq!(tls_relocation_value(R_X86_64_DTPMOD64, &cross, 0), 5);

    // A symbol defined in the referencing module keeps its own module id and static offset.
    let mut defined = dynamic_namespace(false, 4, 3, 7, 2, 1);
    defined.symbols = vec![symbol("", 0, 0), symbol("localTls#E#D", 0x16, 7)];
    let local =
        resolve_tls_reference(&defined, &context, &tls_relocation(R_X86_64_TPOFF64, 1, 0x10)).unwrap();
    assert_eq!(local.module_id, 2);
    assert_eq!(local.tls_offset, 0x100);
    assert_eq!(local.static_offset, 0x80);
    assert_eq!(
        tls_relocation_value(R_X86_64_TPOFF64, &local, 0x10),
        (0x100_u64 + 0x10).wrapping_sub(0x80)
    );
}

fn tls_relocation(relocation_type: u32, symbol_index: u32, addend: i64) -> Elf64Rela {
    Elf64Rela {
        offset: GuestAddress::new(0x1000),
        symbol_index,
        relocation_type,
        addend,
    }
}

fn symbol(name: &str, info: u8, section_index: u16) -> Elf64Symbol {
    Elf64Symbol {
        name_offset: 0,
        info,
        other: 0,
        section_index,
        value: GuestAddress::new(0x100),
        size: 8,
        name: name.to_owned(),
    }
}

fn dynamic_namespace(
    exports: bool,
    library_id: u16,
    module_id: u16,
    library_version: u16,
    module_version_major: u8,
    module_version_minor: u8,
) -> DynamicInfo {
    let library = SceLibraryReference {
        name: "GuestLibrary".to_owned(),
        id: library_id,
        version: library_version,
    };
    let module = SceModuleReference {
        name: "GuestModule".to_owned(),
        id: module_id,
        version_major: module_version_major,
        version_minor: module_version_minor,
    };
    let (required_modules, exported_modules, imported_libraries, exported_libraries) = if exports {
        (Vec::new(), vec![module], Vec::new(), vec![library])
    } else {
        (vec![module], Vec::new(), vec![library], Vec::new())
    };
    DynamicInfo {
        entries: Vec::new(),
        needed_libraries: Vec::new(),
        original_filename: None,
        required_modules,
        exported_modules,
        imported_libraries,
        exported_libraries,
        string_table_address: 0,
        string_table_size: 0,
        symbol_table_address: 0,
        symbol_table_size: 0,
        rela_address: 0,
        rela_size: 0,
        jump_relocation_address: 0,
        jump_relocation_size: 0,
        plt_got_address: 0,
        init_address: 0,
        init_array_address: 0,
        init_array_size: 0,
        preinit_array_address: 0,
        preinit_array_size: 0,
        symbols: Vec::new(),
        relocations: Vec::new(),
        jump_relocations: Vec::new(),
        imports: Vec::new(),
    }
}
