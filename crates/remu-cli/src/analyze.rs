// SPDX-License-Identifier: GPL-2.0-or-later

//! Subcommands that read an executable without executing guest code.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::ExitCode;

use iced_x86::{Decoder, DecoderOptions, Formatter, IntelFormatter};

fn reject_directory(path: &Path) -> Option<ExitCode> {
    if !path.is_dir() {
        return None;
    }
    let parent = path.parent().unwrap_or(path);
    let eboot_hint = parent.join("eboot.bin");
    eprintln!(
        "error: `{}` is a directory; pass eboot.bin or a single .prx/.elf (for example `{}`)",
        path.display(),
        eboot_hint.display()
    );
    // sce_module itself is never the right target; if they passed that, suggest the title root.
    if path
        .file_name()
        .is_some_and(|name| name == "sce_module" || name == "sce_modules")
    {
        eprintln!(
            "hint: to inspect a shipped library, pass a file inside it, e.g. `{}`",
            path.join("libc.prx").display()
        );
    }
    Some(ExitCode::FAILURE)
}

pub(crate) fn inspect(path: &Path) -> ExitCode {
    if let Some(code) = reject_directory(path) {
        return code;
    }
    match remu_loader::inspect_path(path) {
        Ok(info) => {
            println!("path={}", path.display());
            println!("kind={}", info.kind);
            println!("file_size=0x{:x}", info.file_size);
            if let Some(container) = info.container {
                let header = container.header;
                println!("container.version={}", header.version);
                println!("container.mode={}", header.mode);
                println!("container.endian={}", header.endian);
                println!("container.attributes=0x{:02x}", header.attributes);
                println!("container.key_type=0x{:08x}", header.key_type);
                println!("container.header_size=0x{:x}", header.header_size);
                println!("container.metadata_size=0x{:x}", header.metadata_size);
                println!(
                    "container.declared_file_size=0x{:x}",
                    header.declared_file_size
                );
                println!("container.flags=0x{:04x}", header.flags);
                println!(
                    "container.embedded_elf_offset=0x{:x}",
                    header.embedded_elf_offset
                );
                for (index, segment) in container.segments.iter().enumerate() {
                    println!(
                        "container.segment[{index}]=program:{} properties:0x{:x} offset:0x{:x} stored:0x{:x} expanded:0x{:x} encrypted:{} blocks:{}",
                        segment.properties.program_index(),
                        segment.properties.bits(),
                        segment.file_offset,
                        segment.stored_size,
                        segment.expanded_size,
                        segment.properties.is_encrypted(),
                        segment.properties.has_blocks(),
                    );
                }
            }
            if let Some(elf) = info.elf {
                let header = elf.header;
                println!("elf.file_offset=0x{:x}", elf.file_offset);
                println!("elf.os_abi={}", header.os_abi);
                println!("elf.abi_version={}", header.abi_version);
                println!("elf.object_type=0x{:04x}", header.object_type);
                println!("elf.machine=0x{:04x}", header.machine);
                println!("elf.entry={}", header.entry);
                println!("elf.program_headers={}", header.program_header_count);
                println!(
                    "elf.program_header_offset=0x{:x}",
                    header.program_header_offset
                );
                println!("elf.section_headers={}", header.section_header_count);
                println!(
                    "elf.section_header_offset=0x{:x}",
                    header.section_header_offset
                );
                for (index, program) in elf.program_headers.iter().enumerate() {
                    println!(
                        "elf.program[{index}]=type:0x{:08x} flags:0x{:x} offset:0x{:x} vaddr:{} filesz:0x{:x} memsz:0x{:x} align:0x{:x}",
                        program.program_type,
                        program.flags,
                        program.file_offset,
                        program.virtual_address,
                        program.file_size,
                        program.memory_size,
                        program.alignment,
                    );
                }
                if let Some(dynamic) = elf.dynamic {
                    print_dynamic_info(dynamic);
                }
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {}: {error}", path.display());
            ExitCode::FAILURE
        }
    }
}

fn print_dynamic_info(dynamic: remu_loader::DynamicInfo) {
    println!("dynamic.entries={}", dynamic.entries.len());
    println!(
        "dynamic.string_table=0x{:x}:0x{:x}",
        dynamic.string_table_address, dynamic.string_table_size
    );
    println!(
        "dynamic.symbol_table=0x{:x}:0x{:x}",
        dynamic.symbol_table_address, dynamic.symbol_table_size
    );
    println!(
        "dynamic.rela=0x{:x}:0x{:x}",
        dynamic.rela_address, dynamic.rela_size
    );
    println!(
        "dynamic.jump_relocations=0x{:x}:0x{:x}",
        dynamic.jump_relocation_address, dynamic.jump_relocation_size
    );
    println!("dynamic.init=0x{:x}", dynamic.init_address);
    println!(
        "dynamic.init_array=0x{:x}:0x{:x}",
        dynamic.init_array_address, dynamic.init_array_size
    );
    println!(
        "dynamic.preinit_array=0x{:x}:0x{:x}",
        dynamic.preinit_array_address, dynamic.preinit_array_size
    );
    println!("dynamic.symbols={}", dynamic.symbols.len());
    println!("dynamic.relocations={}", dynamic.relocations.len());
    println!("dynamic.plt_relocations={}", dynamic.jump_relocations.len());
    println!("dynamic.imports={}", dynamic.imports.len());
    for library in dynamic.needed_libraries {
        println!("dynamic.needed={library}");
    }
}

pub(crate) fn imports(path: &Path) -> ExitCode {
    if let Some(code) = reject_directory(path) {
        return code;
    }
    let info = match remu_loader::inspect_path(path) {
        Ok(info) => info,
        Err(error) => {
            eprintln!("error: {}: {error}", path.display());
            return ExitCode::FAILURE;
        }
    };
    let Some(dynamic) = info.elf.and_then(|elf| elf.dynamic) else {
        eprintln!("error: {} has no dynamic-linker metadata", path.display());
        return ExitCode::FAILURE;
    };

    println!("path={}", path.display());
    println!("dependencies={}", dynamic.needed_libraries.len());
    for library in dynamic.needed_libraries {
        println!("dependency={library}");
    }
    println!("symbols={}", dynamic.symbols.len());
    println!(
        "relocations={}",
        dynamic.relocations.len() + dynamic.jump_relocations.len()
    );
    let mut relocation_types = BTreeMap::<u32, usize>::new();
    for relocation in dynamic.relocations.iter().chain(&dynamic.jump_relocations) {
        *relocation_types
            .entry(relocation.relocation_type)
            .or_default() += 1;
    }
    for (relocation_type, count) in relocation_types {
        println!("relocation_type[{relocation_type}]={count}");
    }
    println!("plt_entries={}", dynamic.jump_relocations.len());
    for (index, relocation) in dynamic.jump_relocations.iter().enumerate() {
        let symbol_name = usize::try_from(relocation.symbol_index)
            .ok()
            .and_then(|index| dynamic.symbols.get(index))
            .map_or("<missing>", |symbol| symbol.name.as_str());
        let nid = symbol_name
            .split_once('#')
            .map_or(symbol_name, |(nid, _)| nid);
        let resolved_name = remu_runtime::nid_display_name(nid).unwrap_or("unknown");
        println!(
            "plt[{index}]=offset:{} symbol:{} nid:{} resolved:{} name:{}",
            relocation.offset, relocation.symbol_index, nid, resolved_name, symbol_name
        );
    }
    let exports = dynamic
        .symbols
        .iter()
        .enumerate()
        .filter(|(_, symbol)| !symbol.is_undefined())
        .collect::<Vec<_>>();
    println!("exports={}", exports.len());
    for (index, symbol) in exports {
        println!(
            "export={} symbol={} value:{} size:0x{:x}",
            symbol.name, index, symbol.value, symbol.size
        );
    }
    println!("imports={}", dynamic.imports.len());
    for import in dynamic.imports {
        let relocation_types = import
            .relocation_types
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "import={} symbol={} weak={} relocations={} types={}",
            import.name,
            import.symbol_index,
            import.weak,
            import.relocation_count,
            relocation_types
        );
    }
    ExitCode::SUCCESS
}

pub(crate) fn disassemble(path: &Path, address: u64, length: usize) -> ExitCode {
    if length == 0 || length > 4096 {
        eprintln!("error: disassembly length must be between 1 and 4096 bytes");
        return ExitCode::FAILURE;
    }
    let executable = match remu_loader::load_path(path) {
        Ok(executable) => executable,
        Err(error) => {
            eprintln!("error: {}: {error}", path.display());
            return ExitCode::FAILURE;
        }
    };
    let image_base = remu_memory::PS5_MAIN_IMAGE_BASE.get();
    let virtual_address = address.checked_sub(image_base).unwrap_or(address);
    let Some(segment) = executable.load_segments.iter().find(|segment| {
        let start = segment.virtual_address.get();
        let end = start.saturating_add(u64::try_from(segment.data.len()).unwrap_or(u64::MAX));
        virtual_address >= start && virtual_address < end
    }) else {
        eprintln!("error: address {address:#x} is not backed by a file load segment");
        return ExitCode::FAILURE;
    };
    let offset = usize::try_from(virtual_address - segment.virtual_address.get())
        .expect("the selected segment offset fits usize");
    let bytes = &segment.data[offset..segment.data.len().min(offset.saturating_add(length))];
    let mapped_address = image_base.saturating_add(virtual_address);
    let mut decoder = Decoder::with_ip(64, bytes, mapped_address, DecoderOptions::NONE);
    let mut formatter = IntelFormatter::new();
    let mut text = String::new();
    while decoder.can_decode() {
        let instruction = decoder.decode();
        text.clear();
        formatter.format(&instruction, &mut text);
        let byte_offset = usize::try_from(instruction.ip() - mapped_address)
            .expect("decoded byte offset fits usize");
        let instruction_bytes = &bytes[byte_offset..][..instruction.len()];
        let hex = instruction_bytes
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<Vec<_>>()
            .join(" ");
        println!("{:#018x} {hex:<44} {text}", instruction.ip());
    }
    ExitCode::SUCCESS
}
