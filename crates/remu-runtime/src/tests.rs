use std::io::Cursor;
use std::sync::Arc;

use remu_core::GuestAddress;
use remu_loader::load_reader;

use super::{
    LinkedImage, host_protection_from_guest, process_mapping_contains,
    set_current_guest_thread_pointer,
};

type TlsHelper = unsafe extern "C" fn() -> u64;

type TlsStoreHelper = unsafe extern "C" fn();

#[test]
fn av_player_nv12_layout_matches_linear_gpu_texture_storage_extent() {
    // The extended-ABI consumers derive the chroma plane at pData + pitch*height with the
    // height used raw (ghidra-avplayer-abi.md), so the layout is tightly packed.
    let layout_1440 = super::av_player_nv12_layout(1_440, 1_080, true).expect("layout");
    assert_eq!(layout_1440.pitch, 1_536);
    assert_eq!(layout_1440.frame_bytes, 2_488_320);
    let layout_1920 = super::av_player_nv12_layout(1_920, 1_080, true).expect("layout");
    assert_eq!(layout_1920.pitch, 2_048);
    assert_eq!(layout_1920.frame_bytes, 3_317_760);
}

#[test]
fn shader_evaluation_cache_buckets_vertex_record_extents_safely() {
    assert_eq!(super::canonical_vertex_record_count(None), None);
    assert_eq!(super::canonical_vertex_record_count(Some(0)), Some(1));
    assert_eq!(super::canonical_vertex_record_count(Some(1)), Some(1));
    assert_eq!(super::canonical_vertex_record_count(Some(513)), Some(1024));
    assert_eq!(super::canonical_vertex_record_count(Some(700)), Some(1024));
    assert_eq!(
        super::canonical_vertex_record_count(Some(u32::MAX)),
        Some(u32::MAX)
    );
}

#[test]
fn shader_evaluation_cache_key_ignores_unreferenced_user_data() {
    let state = remu_shader::ShaderState {
        program: Arc::new(remu_shader::Program::new(0x1000, Vec::new())),
        user_data: vec![10, 20, 30, 40].into(),
        compute_system_registers: None,
        // Empty programs do not observe any incoming scalar register values.
        user_data_scalar_register_base: 104,
    };
    assert!(
        super::shader_evaluation_user_data_key(&state)
            .as_slice()
            .is_empty()
    );
}

#[test]
fn shader_evaluation_cache_key_indexes_user_data_relative_to_scalar_base() {
    let program = remu_shader::Program::new(
        0x1000,
        vec![remu_shader::Instruction {
            pc: 0,
            encoding: remu_shader::ShaderEncoding::Smem,
            opcode: remu_shader::Opcode::END_PROGRAM,
            words: vec![0],
            sources: vec![remu_shader::Operand::scalar(104)],
            destinations: Vec::new(),
            control: Some(remu_shader::InstructionControl::ScalarMemory(
                remu_shader::ScalarMemoryControl {
                    destination_count: 4,
                    immediate_offset_bytes: 0,
                    dynamic_offset_register: None,
                },
            )),
        }],
    );
    let state = remu_shader::ShaderState {
        program: Arc::new(program),
        user_data: vec![10, 20, 30, 40].into(),
        compute_system_registers: None,
        user_data_scalar_register_base: 104,
    };

    assert_eq!(
        super::shader_evaluation_user_data_key(&state).as_slice(),
        [10, 20, 30, 40]
    );
}

#[test]
fn recognizes_process_image_permission_changes_without_claiming_external_vmas() {
    assert!(process_mapping_contains(
        0x8000_0000,
        0x8002_0000,
        0x8000_4000,
        0x8000_8000
    ));
    assert!(!process_mapping_contains(
        0x8000_0000,
        0x8002_0000,
        0x7fff_c000,
        0x8000_4000
    ));
    assert!(!process_mapping_contains(
        0x8000_0000,
        0x8002_0000,
        0x8001_c000,
        0x8002_4000
    ));
    assert_eq!(
        host_protection_from_guest(0x12),
        libc::PROT_READ | libc::PROT_WRITE
    );
    assert_eq!(host_protection_from_guest(0x11), libc::PROT_READ);
}

#[test]
fn links_relative_and_import_relocations_into_fixed_memory() {
    let mut bytes = vec![0_u8; 0x400];
    write_elf_header(&mut bytes);
    write_program_header(&mut bytes, 0, 1, 7, 0x100, 0x4000, 0x200, 0x300);
    write_program_header(&mut bytes, 1, 2, 6, 0x100, 0x4000, 0x70, 0x70);
    write_program_header(&mut bytes, 2, 7, 4, 0x260, 0x4160, 2, 0x20);
    bytes[0xe0..0xe8].copy_from_slice(&0x10_u64.to_le_bytes());
    write_dynamic(&mut bytes[0x100..]);
    bytes[0x180..0x18d].copy_from_slice(b"\0testNid#A#A\0");
    bytes[0x1d8..0x1dc].copy_from_slice(&1_u32.to_le_bytes());
    bytes[0x1dc] = 0x12;
    bytes[0x1f0..0x1f4].copy_from_slice(&13_u32.to_le_bytes());
    bytes[0x1f4] = 0x12;
    bytes[0x1f6..0x1f8].copy_from_slice(&1_u16.to_le_bytes());
    bytes[0x1f8..0x200].copy_from_slice(&0x4150_u64.to_le_bytes());
    bytes[0x200..0x208].copy_from_slice(&13_u64.to_le_bytes());
    write_relocation(&mut bytes[0x210..], 0, 0x4180, 8, 0x1234);
    write_relocation(&mut bytes[0x210..], 1, 0x4188, (1_u64 << 32) | 7, 0);
    bytes[0x240..0x249].copy_from_slice(&[0x64, 0x48, 0x8b, 0x04, 0x25, 0, 0, 0, 0]);
    bytes[0x250..0x25c]
        .copy_from_slice(&[0x64, 0xc7, 0x04, 0x25, 0x28, 0, 0, 0, 0x0b, 0, 0x02, 0xa0]);
    bytes[0x25c] = 0xc3;
    bytes[0x260..0x262].copy_from_slice(&[0xaa, 0xbb]);
    bytes[0x290..0x29c]
        .copy_from_slice(&[0x64, 0xc7, 0x04, 0x25, 0x28, 0, 0, 0, 0x0b, 0, 0x02, 0xa0]);
    let executable = load_reader(&mut Cursor::new(bytes)).unwrap();

    let linked = LinkedImage::link_main(&executable).unwrap();
    assert_eq!(linked.relocation_stats.relative, 1);
    assert_eq!(linked.relocation_stats.jump_slots, 1);
    assert_eq!(linked.import_stubs.len(), 1);
    assert_eq!(linked.tls.static_block_size, 0x20);
    assert_eq!(linked.tls.patched_loads, 1);
    assert_eq!(linked.tls.patched_stores, 1);
    let base = linked.memory.image_base().get();
    let mut relative = [0_u8; 8];
    linked
        .memory
        .read(GuestAddress::new(base + 0x4180), &mut relative)
        .unwrap();
    assert_eq!(u64::from_le_bytes(relative), base + 0x1234);
    let mut import = [0_u8; 8];
    linked
        .memory
        .read(GuestAddress::new(base + 0x4188), &mut import)
        .unwrap();
    assert_eq!(
        u64::from_le_bytes(import),
        linked.import_stubs[0].address.get()
    );
    let mut patched_tls_load = [0_u8; 9];
    linked
        .memory
        .read(GuestAddress::new(base + 0x4140), &mut patched_tls_load)
        .unwrap();
    assert_eq!(patched_tls_load[0], 0xe8);
    assert_eq!(&patched_tls_load[5..], &[0x90; 4]);
    let displacement = i32::from_le_bytes(
        patched_tls_load[1..5]
            .try_into()
            .expect("fixed-width call displacement"),
    );
    let helper_address = (base + 0x4145).wrapping_add_signed(i64::from(displacement));
    set_current_guest_thread_pointer(0x1234_5678_9abc_def0);
    // SAFETY: The linker emitted and finalized the tested helper as executable x86-64 code.
    let helper: TlsHelper = unsafe { std::mem::transmute(helper_address) };
    // SAFETY: The helper takes no arguments and returns the configured pointer in RAX.
    assert_eq!(unsafe { helper() }, 0x1234_5678_9abc_def0);
    set_current_guest_thread_pointer(0);
    let mut patched_tls_store = [0_u8; 12];
    linked
        .memory
        .read(GuestAddress::new(base + 0x4150), &mut patched_tls_store)
        .unwrap();
    assert_eq!(
        &patched_tls_store[..6],
        &[0x48, 0x8d, 0x64, 0x24, 0x80, 0xe8]
    );
    assert_eq!(&patched_tls_store[10..], &[0x90; 2]);
    set_current_guest_thread_pointer(linked.tls.thread_pointer.get());
    // SAFETY: The test fixture emits a TLS store followed by RET, and the linker finalized it.
    let store_helper: TlsStoreHelper = unsafe { std::mem::transmute(base + 0x4150) };
    // SAFETY: The rewritten fixture takes no arguments and stores through the configured guest TP.
    unsafe { store_helper() };
    set_current_guest_thread_pointer(0);
    let mut tls_errno = [0_u8; 4];
    linked
        .memory
        .read(
            GuestAddress::new(linked.tls.thread_pointer.get() + 0x28),
            &mut tls_errno,
        )
        .unwrap();
    assert_eq!(u32::from_le_bytes(tls_errno), 0xa002_000b);
    let mut embedded_tls_pattern = [0_u8; 12];
    linked
        .memory
        .read(GuestAddress::new(base + 0x4190), &mut embedded_tls_pattern)
        .unwrap();
    assert_eq!(
        embedded_tls_pattern,
        [0x64, 0xc7, 0x04, 0x25, 0x28, 0, 0, 0, 0x0b, 0, 0x02, 0xa0]
    );
    let mut tls_template = [0_u8; 2];
    linked
        .memory
        .read(
            GuestAddress::new(linked.tls.thread_pointer.get() - 0x20),
            &mut tls_template,
        )
        .unwrap();
    assert_eq!(tls_template, [0xaa, 0xbb]);
}

fn write_elf_header(bytes: &mut [u8]) {
    bytes[..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
    bytes[4] = 2;
    bytes[5] = 1;
    bytes[0x12..0x14].copy_from_slice(&62_u16.to_le_bytes());
    bytes[0x18..0x20].copy_from_slice(&0x4000_u64.to_le_bytes());
    bytes[0x20..0x28].copy_from_slice(&64_u64.to_le_bytes());
    bytes[0x34..0x36].copy_from_slice(&64_u16.to_le_bytes());
    bytes[0x36..0x38].copy_from_slice(&56_u16.to_le_bytes());
    bytes[0x38..0x3a].copy_from_slice(&3_u16.to_le_bytes());
}

#[allow(clippy::too_many_arguments)]
fn write_program_header(
    bytes: &mut [u8],
    index: usize,
    program_type: u32,
    flags: u32,
    file_offset: u64,
    virtual_address: u64,
    file_size: u64,
    memory_size: u64,
) {
    let offset = 64 + index * 56;
    bytes[offset..offset + 4].copy_from_slice(&program_type.to_le_bytes());
    bytes[offset + 4..offset + 8].copy_from_slice(&flags.to_le_bytes());
    bytes[offset + 8..offset + 16].copy_from_slice(&file_offset.to_le_bytes());
    bytes[offset + 16..offset + 24].copy_from_slice(&virtual_address.to_le_bytes());
    bytes[offset + 32..offset + 40].copy_from_slice(&file_size.to_le_bytes());
    bytes[offset + 40..offset + 48].copy_from_slice(&memory_size.to_le_bytes());
    bytes[offset + 48..offset + 56].copy_from_slice(&0x4000_u64.to_le_bytes());
}

fn write_dynamic(bytes: &mut [u8]) {
    let entries = [
        (5_i64, 0x4080_u64),
        (10, 0x40),
        (6, 0x40c0),
        (0x6100_003f, 0x48),
        (7, 0x4110),
        (8, 0x30),
        (0, 0),
    ];
    for (index, (tag, value)) in entries.into_iter().enumerate() {
        let offset = index * 16;
        bytes[offset..offset + 8].copy_from_slice(&tag.to_le_bytes());
        bytes[offset + 8..offset + 16].copy_from_slice(&value.to_le_bytes());
    }
}

fn write_relocation(bytes: &mut [u8], index: usize, offset: u64, info: u64, addend: i64) {
    let entry = index * 24;
    bytes[entry..entry + 8].copy_from_slice(&offset.to_le_bytes());
    bytes[entry + 8..entry + 16].copy_from_slice(&info.to_le_bytes());
    bytes[entry + 16..entry + 24].copy_from_slice(&addend.to_le_bytes());
}
