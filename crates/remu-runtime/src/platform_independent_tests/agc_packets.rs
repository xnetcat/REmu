// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    AGC_IT_NOP, AGC_IT_SET_CONTEXT_REG_INDIRECT, AGC_IT_SET_SH_REG, AGC_IT_SET_SH_REG_INDIRECT,
    AGC_IT_SET_UCONFIG_REG, AGC_IT_SET_UCONFIG_REG_INDEX, AGC_IT_SET_UCONFIG_REG_INDIRECT,
    AGC_VGT_INDEX_TYPE, AgcGraphicsState, AgcShaderRegistry, GPU_EVENT_SINK_TEST, VideoOutState,
    agc_apply_submission_registers, agc_pm4, compute_graphics_state, decode_agc_dma_data,
    decode_agc_memory_acquire, decode_agc_memory_wait, low_u32,
};

fn append_native_memory_dma(
    commands: &mut Vec<u32>,
    source_address: u64,
    destination_address: u64,
    byte_count: u32,
) {
    commands.extend([
        agc_pm4(7, crate::AGC_IT_DMA_DATA, 0),
        (3 << 29) | (3 << 20),
        low_u32(source_address),
        low_u32(source_address >> 32),
        low_u32(destination_address),
        low_u32(destination_address >> 32),
        byte_count,
    ]);
}

fn append_native_indexed_draw(commands: &mut Vec<u32>, index_address: u64, index_count: u32) {
    commands.extend([
        agc_pm4(6, crate::AGC_IT_DRAW_INDEX_2, 0),
        index_count,
        low_u32(index_address),
        low_u32(index_address >> 32),
        index_count,
        2,
    ]);
}

fn capture_submission_events(
    commands: &[u32],
    graphics: &mut AgcGraphicsState,
    shaders: &mut AgcShaderRegistry,
) -> Vec<remu_gpu::GpuEvent> {
    let mut submission_count = 0;
    let (result, events) = crate::capture_agc_gpu_events(|| {
        crate::agc_parse_submission(
            commands.as_ptr().addr() as u64,
            u32::try_from(commands.len()).expect("command dword count"),
            &mut crate::AgcSubmissionContext {
                graphics,
                shaders,
                video_out: &VideoOutState::default(),
                submission_count: &mut submission_count,
                flips: &mut Vec::new(),
                event_writes: &mut Vec::new(),
            },
        )
    });
    result.expect("ordered DMA submission");
    assert_eq!(submission_count, 1);
    events
}

#[test]
fn submission_parser_skips_the_one_dword_uncounted_nop() {
    let events = capture_submission_events(
        &[0xffff_1000],
        &mut AgcGraphicsState::default(),
        &mut AgcShaderRegistry::default(),
    );

    assert!(events.is_empty());
}

#[test]
fn pfp_sync_me_becomes_an_ordered_command_processor_barrier() {
    let packet = [agc_pm4(2, crate::AGC_IT_PFP_SYNC_ME, 0), 0];
    let events = capture_submission_events(
        &packet,
        &mut AgcGraphicsState::default(),
        &mut AgcShaderRegistry::default(),
    );

    assert_eq!(events, [remu_gpu::GpuEvent::CommandProcessorBarrier]);
}

#[test]
fn pfp_sync_me_rejects_a_non_native_packet_length() {
    let packet = [agc_pm4(3, crate::AGC_IT_PFP_SYNC_ME, 0), 0, 0xfeed_face];
    let mut submission_count = 0;
    let result = crate::agc_parse_submission(
        packet.as_ptr().addr() as u64,
        u32::try_from(packet.len()).expect("command dword count"),
        &mut crate::AgcSubmissionContext {
            graphics: &mut AgcGraphicsState::default(),
            shaders: &mut AgcShaderRegistry::default(),
            video_out: &VideoOutState::default(),
            submission_count: &mut submission_count,
            flips: &mut Vec::new(),
            event_writes: &mut Vec::new(),
        },
    );

    assert!(matches!(
        result,
        Err(crate::AgcSubmissionError::PfpSyncMePacketLength { actual: 3 })
    ));
    assert_eq!(submission_count, 0);
}

#[test]
fn decodes_gen5_memory_acquire_ranges_and_controls() {
    let packet = [
        agc_pm4(8, AGC_IT_NOP, crate::AGC_R_ACQUIRE_MEM),
        0x8000_1234,
        0x3456_789a,
        0x0000_00ab,
        0x7654_3210,
        0x0000_cdef,
        0xabcd_2468,
        0xfff5_4321,
    ];
    let decoded = decode_agc_memory_acquire(packet.as_ptr().addr() as u64);

    assert_eq!(decoded.engine, 1);
    assert_eq!(decoded.cb_db_control, 0x1234);
    assert_eq!(decoded.base_address, 0x00cd_ef76_5432_1000);
    assert_eq!(decoded.size_bytes, 0xab34_5678_9a00);
    assert_eq!(decoded.poll_interval, 0x2468);
    assert_eq!(decoded.gcr_control, 0x5_4321);
}

#[test]
fn decodes_native_32_bit_wait_reference_before_its_control_word() {
    let mut label = 0_u32;
    let label_address = (&raw mut label).addr() as u64;
    let packet = [
        agc_pm4(7, AGC_IT_NOP, crate::AGC_R_WAIT_MEM_32),
        low_u32(label_address),
        low_u32(label_address >> 32),
        u32::MAX,
        1,
        0x113,
        0,
    ];

    assert_eq!(
        decode_agc_memory_wait(
            packet.as_ptr().addr() as u64,
            crate::AGC_R_WAIT_MEM_32,
            false
        ),
        Some(remu_gpu::GpuMemoryWait {
            address: label_address,
            reference: 1,
            mask: u64::from(u32::MAX),
            compare_function: 3,
            is_64_bit: false,
        })
    );
}

#[test]
fn decodes_native_dma_immediate_fill_without_treating_the_value_as_an_address() {
    let mut destination = [0_u32; 2];
    let destination_address = destination.as_mut_ptr().addr() as u64;
    let control = 1 | (3 << 13) | (3 << 20) | (3 << 25) | (2 << 29) | (1 << 31);
    let packet = [
        agc_pm4(7, crate::AGC_IT_DMA_DATA, 0),
        control,
        1,
        0,
        low_u32(destination_address),
        low_u32(destination_address >> 32),
        8,
    ];

    assert_eq!(
        decode_agc_dma_data(packet.as_ptr().addr() as u64, 7, true),
        Some(remu_gpu::GpuMemoryCopy {
            source_address: 1,
            destination_address,
            byte_count: 8,
            source_selection: 2,
            destination_selection: 3,
            source_snapshot: None,
        })
    );
}

#[test]
fn draw_before_dma_keeps_the_pre_copy_index_snapshot_and_event_order() {
    let source = [9_u16, 8, 7];
    let destination = [1_u16, 2, 3];
    let source_address = source.as_ptr().addr() as u64;
    let destination_address = destination.as_ptr().addr() as u64;
    let mut commands = Vec::new();
    append_native_indexed_draw(&mut commands, destination_address, 3);
    append_native_memory_dma(
        &mut commands,
        source_address,
        destination_address,
        u32::try_from(std::mem::size_of_val(&source)).expect("source size"),
    );

    let events = capture_submission_events(
        &commands,
        &mut AgcGraphicsState::default(),
        &mut AgcShaderRegistry::default(),
    );

    assert_eq!(
        destination,
        [1, 2, 3],
        "parsing must not execute a later DMA packet"
    );
    let [
        remu_gpu::GpuEvent::IndexedDraw(draw),
        remu_gpu::GpuEvent::MemoryCopy(copy),
    ] = events.as_slice()
    else {
        panic!("draw and following DMA must retain command order: {events:?}");
    };
    assert_eq!(draw.index_snapshot.as_deref(), Some([1, 2, 3].as_slice()));
    assert_eq!(copy.source_address, source_address);
    assert_eq!(copy.destination_address, destination_address);
}

#[test]
fn dma_before_draw_shadows_indices_without_executing_the_copy_early() {
    let mut source = [9_u16, 8, 7];
    let destination = [1_u16, 2, 3];
    let source_address = source.as_ptr().addr() as u64;
    let destination_address = destination.as_ptr().addr() as u64;
    let mut commands = Vec::new();
    append_native_memory_dma(
        &mut commands,
        source_address,
        destination_address,
        u32::try_from(std::mem::size_of_val(&source)).expect("source size"),
    );
    append_native_indexed_draw(&mut commands, destination_address, 3);

    let events = capture_submission_events(
        &commands,
        &mut AgcGraphicsState::default(),
        &mut AgcShaderRegistry::default(),
    );

    assert_eq!(
        destination,
        [1, 2, 3],
        "the backend event is the only authoritative guest-memory mutation"
    );
    let [
        remu_gpu::GpuEvent::MemoryCopy(copy),
        remu_gpu::GpuEvent::IndexedDraw(draw),
    ] = events.as_slice()
    else {
        panic!("DMA and following draw must retain command order: {events:?}");
    };
    assert_eq!(copy.destination_address, destination_address);
    let expected_source_bytes = [9_u16, 8, 7]
        .into_iter()
        .flat_map(u16::to_ne_bytes)
        .collect::<Vec<_>>();
    assert_eq!(
        copy.source_snapshot.as_deref(),
        Some(expected_source_bytes.as_slice())
    );
    source.fill(0xffff);
    assert_eq!(
        copy.source_snapshot.as_deref(),
        Some(expected_source_bytes.as_slice()),
        "the queued event must retain packet-time bytes after the rotating source is reused"
    );
    assert_eq!(
        draw.index_snapshot.as_deref(),
        Some([9, 8, 7].as_slice()),
        "following CPU-side draw decoding must observe the staged DMA result"
    );
}

#[test]
fn chained_dma_snapshot_includes_earlier_submission_shadow_writes() {
    let source = [0x11_u8, 0x22, 0x33, 0x44];
    let intermediate = [0_u8; 4];
    let destination = [0_u8; 4];
    let mut commands = Vec::new();
    append_native_memory_dma(
        &mut commands,
        source.as_ptr().addr() as u64,
        intermediate.as_ptr().addr() as u64,
        4,
    );
    append_native_memory_dma(
        &mut commands,
        intermediate.as_ptr().addr() as u64,
        destination.as_ptr().addr() as u64,
        4,
    );

    let events = capture_submission_events(
        &commands,
        &mut AgcGraphicsState::default(),
        &mut AgcShaderRegistry::default(),
    );
    let [
        remu_gpu::GpuEvent::MemoryCopy(first),
        remu_gpu::GpuEvent::MemoryCopy(second),
    ] = events.as_slice()
    else {
        panic!("two ordered DMA events expected: {events:?}");
    };
    assert_eq!(first.source_snapshot.as_deref(), Some(source.as_slice()));
    assert_eq!(
        second.source_snapshot.as_deref(),
        Some(source.as_slice()),
        "the second packet-time read must include the first DMA's staged bytes"
    );
    assert_eq!(intermediate, [0; 4]);
    assert_eq!(destination, [0; 4]);
}

#[test]
fn dma_shadow_reaches_an_indirect_buffer_dispatch_without_reordering_events() {
    let program = super::AlignedComputeProgram([0; 4]);
    let program_address = program.0.as_ptr().addr() as u64;
    let (mut graphics, mut shaders) = compute_graphics_state(program_address);
    let source_dimensions = [3_u32, 2, 1];
    let destination_dimensions = [0_u32; 3];
    let source_address = source_dimensions.as_ptr().addr() as u64;
    let destination_address = destination_dimensions.as_ptr().addr() as u64;
    graphics.indirect_args_address = destination_address;

    let indirect = [agc_pm4(3, crate::AGC_IT_DISPATCH_INDIRECT, 0), 0, 0x8041];
    let indirect_address = indirect.as_ptr().addr() as u64;
    let mut commands = Vec::new();
    append_native_memory_dma(
        &mut commands,
        source_address,
        destination_address,
        u32::try_from(std::mem::size_of_val(&source_dimensions)).expect("dimension size"),
    );
    commands.extend([
        agc_pm4(4, crate::AGC_IT_INDIRECT_BUFFER, 0),
        low_u32(indirect_address),
        low_u32(indirect_address >> 32),
        u32::try_from(indirect.len()).expect("indirect dword count"),
    ]);

    let events = capture_submission_events(&commands, &mut graphics, &mut shaders);

    assert_eq!(
        destination_dimensions,
        [0, 0, 0],
        "nested parsing must not commit DMA to guest memory"
    );
    let [
        remu_gpu::GpuEvent::MemoryCopy(copy),
        remu_gpu::GpuEvent::ComputeDispatch(dispatch),
    ] = events.as_slice()
    else {
        panic!("DMA and nested indirect dispatch must retain command order: {events:?}");
    };
    assert_eq!(copy.destination_address, destination_address);
    assert_eq!(
        dispatch.source,
        remu_gpu::ComputeDispatchSource::Indirect {
            dimensions_address: destination_address,
        }
    );
    assert_eq!(
        dispatch.launch.grid,
        remu_gpu::ComputeGrid::Threadgroups(source_dimensions)
    );
}

#[test]
fn dma_shadow_reaches_indirect_draw_arguments_without_executing_the_copy_early() {
    let source_arguments = [6_u32, 2, 1, (-3_i32) as u32, 9];
    let destination_arguments = [0_u32; 5];
    let source_address = source_arguments.as_ptr().addr() as u64;
    let destination_address = destination_arguments.as_ptr().addr() as u64;
    let indices = [10_u16, 11, 12, 13, 14, 15, 16, 17];
    let index_address = indices.as_ptr().addr() as u64;
    let mut graphics = AgcGraphicsState {
        indirect_args_address: destination_address,
        index_buffer_address: index_address,
        index_resource_address: index_address,
        index_buffer_count: u32::try_from(indices.len()).expect("index count"),
        index_size: 1,
        ..AgcGraphicsState::default()
    };
    let mut commands = Vec::new();
    append_native_memory_dma(
        &mut commands,
        source_address,
        destination_address,
        u32::try_from(std::mem::size_of_val(&source_arguments)).expect("argument size"),
    );
    commands.extend([
        agc_pm4(5, crate::AGC_IT_DRAW_INDEX_INDIRECT, 0),
        0,
        0x08d_008c,
        0x08e,
        2,
    ]);

    let events =
        capture_submission_events(&commands, &mut graphics, &mut AgcShaderRegistry::default());

    assert_eq!(
        destination_arguments, [0; 5],
        "the backend event is the only authoritative guest-memory mutation"
    );
    let [
        remu_gpu::GpuEvent::MemoryCopy(copy),
        remu_gpu::GpuEvent::IndexedDraw(draw),
    ] = events.as_slice()
    else {
        panic!("DMA and following indirect draw must retain command order: {events:?}");
    };
    assert_eq!(copy.destination_address, destination_address);
    assert_eq!(draw.index_count, 6);
    assert_eq!(draw.instance_count, 2);
    assert_eq!(draw.index_offset, 1);
    assert_eq!(draw.base_vertex, -3);
    assert_eq!(draw.indirect_arguments_address, Some(destination_address));
    assert_eq!(
        draw.shader_registers
            .iter()
            .find(|register| register.offset == 0x08c)
            .map(|register| register.value),
        Some(0)
    );
    assert_eq!(graphics.shader_registers.get(&0x08c), None);
    assert_eq!(graphics.shader_registers.get(&0x08d), None);
    assert_eq!(graphics.shader_registers.get(&0x08e), None);
}

#[test]
fn emits_ordered_gen5_release_and_wait_label_operations() {
    let _event_guard = GPU_EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut label = 0_u64;
    let label_address = (&raw mut label).addr() as u64;
    let commands = [
        agc_pm4(8, AGC_IT_NOP, crate::AGC_R_RELEASE_MEM),
        0,
        2 << 16,
        low_u32(label_address),
        low_u32(label_address >> 32),
        0x89ab_cdef,
        0x0123_4567,
        0,
        agc_pm4(9, AGC_IT_NOP, crate::AGC_R_WAIT_MEM_64),
        low_u32(label_address),
        low_u32(label_address >> 32),
        u32::MAX,
        u32::MAX,
        0x89ab_cdef,
        0x0123_4567,
        3,
        0,
    ];
    let (sender, receiver) = remu_gpu::gpu_event_channel();
    remu_gpu::install_event_sink(sender);
    let mut graphics = AgcGraphicsState::default();
    let mut shaders = AgcShaderRegistry::default();
    let mut submission_count = 0;
    crate::agc_parse_submission(
        commands.as_ptr().addr() as u64,
        u32::try_from(commands.len()).expect("command dword count"),
        &mut crate::AgcSubmissionContext {
            graphics: &mut graphics,
            shaders: &mut shaders,
            video_out: &VideoOutState::default(),
            submission_count: &mut submission_count,
            flips: &mut Vec::new(),
            event_writes: &mut Vec::new(),
        },
    )
    .expect("release/wait submission");
    remu_gpu::clear_event_sink();

    assert_eq!(
        receiver.recv().expect("release event"),
        remu_gpu::GpuEvent::MemoryRelease(remu_gpu::GpuMemoryRelease {
            address: label_address,
            value: remu_gpu::GpuMemoryReleaseValue::Immediate64(0x0123_4567_89ab_cdef),
        })
    );
    assert_eq!(
        receiver.recv().expect("wait event"),
        remu_gpu::GpuEvent::MemoryWait(remu_gpu::GpuMemoryWait {
            address: label_address,
            reference: 0x0123_4567_89ab_cdef,
            mask: u64::MAX,
            compare_function: 3,
            is_64_bit: true,
        })
    );
    assert_eq!(submission_count, 1);
    assert!(receiver.try_recv().is_err());
}

#[test]
fn executes_indirect_command_buffer_events_before_following_waits() {
    let _event_guard = GPU_EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut label = 0_u32;
    let label_address = (&raw mut label).addr() as u64;
    let indirect = [
        agc_pm4(8, AGC_IT_NOP, crate::AGC_R_RELEASE_MEM),
        0,
        1 << 16,
        low_u32(label_address),
        low_u32(label_address >> 32),
        1,
        0,
        0,
    ];
    let indirect_address = indirect.as_ptr().addr() as u64;
    let commands = [
        agc_pm4(4, crate::AGC_IT_INDIRECT_BUFFER, 0),
        low_u32(indirect_address),
        low_u32(indirect_address >> 32),
        0x0f20_0000 | u32::try_from(indirect.len()).expect("indirect dword count"),
        agc_pm4(7, AGC_IT_NOP, crate::AGC_R_WAIT_MEM_32),
        low_u32(label_address),
        low_u32(label_address >> 32),
        u32::MAX,
        1,
        0x13,
        0,
    ];
    let (sender, receiver) = remu_gpu::gpu_event_channel();
    remu_gpu::install_event_sink(sender);
    let mut graphics = AgcGraphicsState::default();
    let mut shaders = AgcShaderRegistry::default();
    let mut submission_count = 0;
    crate::agc_parse_submission(
        commands.as_ptr().addr() as u64,
        u32::try_from(commands.len()).expect("command dword count"),
        &mut crate::AgcSubmissionContext {
            graphics: &mut graphics,
            shaders: &mut shaders,
            video_out: &VideoOutState::default(),
            submission_count: &mut submission_count,
            flips: &mut Vec::new(),
            event_writes: &mut Vec::new(),
        },
    )
    .expect("indirect release and direct wait submission");
    remu_gpu::clear_event_sink();

    assert_eq!(
        receiver.recv().expect("indirect release event"),
        remu_gpu::GpuEvent::MemoryRelease(remu_gpu::GpuMemoryRelease {
            address: label_address,
            value: remu_gpu::GpuMemoryReleaseValue::Immediate32(1),
        })
    );
    assert_eq!(
        receiver.recv().expect("following wait event"),
        remu_gpu::GpuEvent::MemoryWait(remu_gpu::GpuMemoryWait {
            address: label_address,
            reference: 1,
            mask: u64::from(u32::MAX),
            compare_function: 3,
            is_64_bit: false,
        })
    );
    assert_eq!(submission_count, 1);
    assert!(receiver.try_recv().is_err());
}

#[test]
fn emits_ordered_standard_and_agc_write_data_operations() {
    let _event_guard = GPU_EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut labels = [0_u32; 4];
    let label_address = labels.as_mut_ptr().addr() as u64;
    let commands = [
        agc_pm4(5, crate::AGC_IT_WRITE_DATA, 0),
        5 << 8,
        low_u32(label_address),
        low_u32(label_address >> 32),
        0x1122_3344,
        agc_pm4(6, AGC_IT_NOP, crate::AGC_R_WRITE_DATA),
        4 | (1 << 16),
        low_u32(label_address + 8),
        low_u32((label_address + 8) >> 32),
        0x5566_7788,
        0x99aa_bbcc,
    ];
    let (sender, receiver) = remu_gpu::gpu_event_channel();
    remu_gpu::install_event_sink(sender);
    let mut graphics = AgcGraphicsState::default();
    let mut shaders = AgcShaderRegistry::default();
    let mut submission_count = 0;
    crate::agc_parse_submission(
        commands.as_ptr().addr() as u64,
        u32::try_from(commands.len()).expect("command dword count"),
        &mut crate::AgcSubmissionContext {
            graphics: &mut graphics,
            shaders: &mut shaders,
            video_out: &VideoOutState::default(),
            submission_count: &mut submission_count,
            flips: &mut Vec::new(),
            event_writes: &mut Vec::new(),
        },
    )
    .expect("write-data submission");
    remu_gpu::clear_event_sink();

    assert_eq!(
        receiver.recv().expect("standard write event"),
        remu_gpu::GpuEvent::MemoryWrite(remu_gpu::GpuMemoryWrite {
            address: label_address,
            values: vec![0x1122_3344],
            increment_address: true,
        })
    );
    assert_eq!(
        receiver.recv().expect("AGC write event"),
        remu_gpu::GpuEvent::MemoryWrite(remu_gpu::GpuMemoryWrite {
            address: label_address + 8,
            values: vec![0x5566_7788, 0x99aa_bbcc],
            increment_address: false,
        })
    );
    assert_eq!(submission_count, 1);
    assert!(receiver.try_recv().is_err());
}

#[test]
fn decodes_event_writes_without_inventing_completion_notifications() {
    for (event_type, wait_for) in [
        (0x07, remu_gpu::GpuPipelineStage::ComputeShader),
        (0x0f, remu_gpu::GpuPipelineStage::VertexShader),
        (0x10, remu_gpu::GpuPipelineStage::PixelShader),
    ] {
        let packet = [agc_pm4(2, crate::AGC_IT_EVENT_WRITE, 0), 0x400 | event_type];
        assert_eq!(
            crate::decode_agc_pipeline_barrier(packet.as_ptr().addr() as u64, 2)
                .expect("supported partial flush"),
            remu_gpu::GpuPipelineBarrier { wait_for }
        );
    }

    // The hardware packet processor uses the Type-3 header length to advance over the
    // packet, while EVENT_INDEX=4 consumes only EVENT_CONTROL. Native Gen5 streams may leave
    // two padding dwords in that packet; they must not turn a valid partial flush into a
    // fatal submission error.
    let padded_partial_flush = [
        agc_pm4(4, crate::AGC_IT_EVENT_WRITE, 0),
        0x410,
        0xfeed_face,
        0xcafe_babe,
    ];
    assert_eq!(
        crate::decode_agc_pipeline_barrier(padded_partial_flush.as_ptr().addr() as u64, 4)
            .expect("padded partial flush"),
        remu_gpu::GpuPipelineBarrier {
            wait_for: remu_gpu::GpuPipelineStage::PixelShader,
        }
    );

    let wrong_index = [agc_pm4(2, crate::AGC_IT_EVENT_WRITE, 0), 0x10];
    assert!(matches!(
        crate::decode_agc_pipeline_barrier(wrong_index.as_ptr().addr() as u64, 2),
        Err(crate::AgcSubmissionError::PartialFlushEventIndex {
            event_type: 0x10,
            actual: 0,
        })
    ));
    let unsupported = [agc_pm4(2, crate::AGC_IT_EVENT_WRITE, 0), 0x416];
    assert!(matches!(
        crate::decode_agc_pipeline_barrier(unsupported.as_ptr().addr() as u64, 2),
        Err(crate::AgcSubmissionError::UnsupportedEventWrite {
            event_type: 0x16,
            event_index: 4,
            length: 2,
        })
    ));
    let reserved = [agc_pm4(2, crate::AGC_IT_EVENT_WRITE, 0), 0x8000_0410];
    assert!(matches!(
        crate::decode_agc_pipeline_barrier(reserved.as_ptr().addr() as u64, 2),
        Err(crate::AgcSubmissionError::UnsupportedEventWriteControl(
            0x8000_0410
        ))
    ));

    let ordinary_event = [agc_pm4(2, crate::AGC_IT_EVENT_WRITE, 0), 0x02];
    let mut graphics = AgcGraphicsState::default();
    let mut notifications = Vec::new();
    crate::agc_handle_submission_packet(
        crate::AgcSubmissionPacket {
            address: ordinary_event.as_ptr().addr() as u64,
            length: 2,
            opcode: crate::AGC_IT_EVENT_WRITE,
            register: 0,
            predicated: false,
        },
        &mut graphics,
        &mut AgcShaderRegistry::default(),
        &VideoOutState::default(),
        &mut Vec::new(),
        &mut notifications,
        None,
    )
    .expect("ordinary cache event");
    assert!(
        notifications.is_empty(),
        "ordinary EVENT_WRITE must not fabricate a kernel interrupt"
    );

    let query_results = [0_u64; 16];
    let query_address = query_results.as_ptr().addr() as u64;
    let pixel_pipe_stat_dump = [
        agc_pm4(4, crate::AGC_IT_EVENT_WRITE, 0),
        0x139,
        low_u32(query_address),
        (query_address >> 32) as u32,
    ];
    assert_eq!(
        crate::decode_agc_event_write(pixel_pipe_stat_dump.as_ptr().addr() as u64, 4)
            .expect("addressed pixel-pipe statistic dump"),
        Some(crate::AgcEventWrite::PixelPipeStatDump {
            destination: query_address,
        })
    );
    let mut shaders = AgcShaderRegistry::default();
    let mut flips = Vec::new();
    let (result, query_events) = crate::capture_agc_gpu_events(|| {
        for _ in 0..2 {
            crate::agc_handle_submission_packet(
                crate::AgcSubmissionPacket {
                    address: pixel_pipe_stat_dump.as_ptr().addr() as u64,
                    length: 4,
                    opcode: crate::AGC_IT_EVENT_WRITE,
                    register: 0,
                    predicated: false,
                },
                &mut graphics,
                &mut shaders,
                &VideoOutState::default(),
                &mut flips,
                &mut notifications,
                None,
            )?;
        }
        Ok::<(), crate::AgcSubmissionError>(())
    });
    result.expect("pixel-pipe statistic dumps");
    assert_eq!(
        query_results, [0; 16],
        "query results must retire with preceding GPU work instead of changing during decoding"
    );
    assert_eq!(query_events.len(), 32);
    for (event_index, event) in query_events.into_iter().enumerate() {
        let dump_index = event_index / 16;
        let pair_index = event_index % 16;
        let expected_count = if dump_index == 0 { 0 } else { 0x02ff_ffff };
        assert_eq!(
            event,
            remu_gpu::GpuEvent::MemoryRelease(remu_gpu::GpuMemoryRelease {
                address: query_address
                    + u64::try_from(pair_index).expect("counter pair index") * 16,
                value: remu_gpu::GpuMemoryReleaseValue::Immediate64((1_u64 << 63) | expected_count,),
            })
        );
    }
    assert!(
        notifications.is_empty(),
        "occlusion query dumps are not kernel completion interrupts"
    );

    let malformed_query = [
        agc_pm4(4, crate::AGC_IT_EVENT_WRITE, 0),
        0x239,
        low_u32(query_address),
        (query_address >> 32) as u32,
    ];
    assert!(matches!(
        crate::decode_agc_event_write(malformed_query.as_ptr().addr() as u64, 4),
        Err(crate::AgcSubmissionError::UnsupportedEventWrite {
            event_type: 0x39,
            event_index: 2,
            length: 4,
        })
    ));
}

#[test]
fn applies_native_gen5_indirect_register_packets_to_each_register_space() {
    let context_registers = [0x318, 0x1234_5678, 0x390, 0x5a, u32::MAX, 0xfeed_face];
    let shader_registers = [
        0x2000_020c,
        0x7654_3210,
        0x2000_020d,
        0x44,
        0x2000_0213,
        0x388,
    ];
    let user_config_registers = [0x4000_0242, 4];
    let packet = |opcode, registers: &[u32]| {
        let address = registers.as_ptr().addr() as u64;
        [
            agc_pm4(5, opcode, 0),
            low_u32(address),
            (address >> 32) as u32,
            0x8000_0000,
            u32::try_from(registers.len() / 2).expect("register pair count"),
        ]
    };
    let mut graphics = AgcGraphicsState::default();

    for (opcode, registers) in [
        (AGC_IT_SET_CONTEXT_REG_INDIRECT, &context_registers[..]),
        (AGC_IT_SET_SH_REG_INDIRECT, &shader_registers[..]),
        (AGC_IT_SET_UCONFIG_REG_INDIRECT, &user_config_registers[..]),
    ] {
        let command = packet(opcode, registers);
        agc_apply_submission_registers(command.as_ptr().addr() as u64, 5, opcode, 0, &mut graphics)
            .expect("native indirect register packet");
    }

    assert_eq!(graphics.context_registers.get(&0x318), Some(&0x1234_5678));
    assert_eq!(graphics.context_registers.get(&0x390), Some(&0x5a));
    assert!(!graphics.context_registers.contains_key(&u32::MAX));
    assert_eq!(
        graphics
            .shader_registers
            .get(&remu_gpu::COMPUTE_PGM_LO_REGISTER),
        Some(&0x7654_3210)
    );
    assert_eq!(
        graphics
            .shader_registers
            .get(&remu_gpu::COMPUTE_PGM_HI_REGISTER),
        Some(&0x44)
    );
    assert_eq!(graphics.user_config_registers.get(&0x242), Some(&4));
}

#[test]
fn skips_predicated_packets_when_boolean_predication_is_false() {
    let predicate = 1_u64;
    let predicate_address = std::ptr::from_ref(&predicate).addr() as u64;
    let commands = [
        agc_pm4(4, crate::AGC_IT_SET_PREDICATION, 0),
        3 << 16,
        low_u32(predicate_address) & !0x0f,
        low_u32(predicate_address >> 32),
        agc_pm4(3, crate::AGC_IT_SET_SH_REG, 0) | 1,
        0x00c,
        0xdead_beef,
        agc_pm4(3, crate::AGC_IT_SET_SH_REG, 0),
        0x00d,
        0x1234_5678,
    ];
    let mut graphics = AgcGraphicsState::default();
    let mut shaders = AgcShaderRegistry::default();
    let mut submission_count = 0;

    crate::agc_parse_submission(
        commands.as_ptr().addr() as u64,
        u32::try_from(commands.len()).expect("command dword count"),
        &mut crate::AgcSubmissionContext {
            graphics: &mut graphics,
            shaders: &mut shaders,
            video_out: &VideoOutState::default(),
            submission_count: &mut submission_count,
            flips: &mut Vec::new(),
            event_writes: &mut Vec::new(),
        },
    )
    .expect("predicated submission");

    assert!(graphics.predicate_skip);
    assert!(!graphics.shader_registers.contains_key(&0x00c));
    assert_eq!(graphics.shader_registers.get(&0x00d), Some(&0x1234_5678));
}

#[test]
fn skipped_predicated_packet_is_not_rejected_during_predecode() {
    let predicate = 1_u64;
    let predicate_address = std::ptr::from_ref(&predicate).addr() as u64;
    let commands = [
        agc_pm4(4, crate::AGC_IT_SET_PREDICATION, 0),
        3 << 16,
        low_u32(predicate_address) & !0x0f,
        low_u32(predicate_address >> 32),
        // EVENT_INDEX=4 is unsupported when executed. Its predication bit is set, so the active
        // false predicate must skip it before payload validation.
        agc_pm4(2, crate::AGC_IT_EVENT_WRITE, 0) | 1,
        0x416,
        agc_pm4(3, crate::AGC_IT_SET_SH_REG, 0),
        0x00d,
        0x1234_5678,
    ];
    let mut graphics = AgcGraphicsState::default();
    let mut shaders = AgcShaderRegistry::default();
    let mut submission_count = 0;

    crate::agc_parse_submission(
        commands.as_ptr().addr() as u64,
        u32::try_from(commands.len()).expect("command dword count"),
        &mut crate::AgcSubmissionContext {
            graphics: &mut graphics,
            shaders: &mut shaders,
            video_out: &VideoOutState::default(),
            submission_count: &mut submission_count,
            flips: &mut Vec::new(),
            event_writes: &mut Vec::new(),
        },
    )
    .expect("skipped invalid packet must not reject the submission");

    assert_eq!(graphics.shader_registers.get(&0x00d), Some(&0x1234_5678));
}

#[test]
fn cond_exec_skips_packet_payload_validation_inside_its_guarded_range() {
    let predicate = 0_u32;
    let predicate_address = std::ptr::from_ref(&predicate).addr() as u64;
    let commands = [
        agc_pm4(5, crate::AGC_IT_COND_EXEC, 0),
        low_u32(predicate_address) & !3,
        low_u32(predicate_address >> 32),
        0,
        2,
        // EVENT_INDEX=4 would fail if its payload were decoded. The command processor only needs
        // its header length to advance across the zero-guarded COND_EXEC span.
        agc_pm4(2, crate::AGC_IT_EVENT_WRITE, 0),
        0x416,
        agc_pm4(3, crate::AGC_IT_SET_SH_REG, 0),
        0x00d,
        0x1234_5678,
    ];
    let mut graphics = AgcGraphicsState::default();
    let mut shaders = AgcShaderRegistry::default();
    let mut submission_count = 0;

    crate::agc_parse_submission(
        commands.as_ptr().addr() as u64,
        u32::try_from(commands.len()).expect("command dword count"),
        &mut crate::AgcSubmissionContext {
            graphics: &mut graphics,
            shaders: &mut shaders,
            video_out: &VideoOutState::default(),
            submission_count: &mut submission_count,
            flips: &mut Vec::new(),
            event_writes: &mut Vec::new(),
        },
    )
    .expect("conditionally skipped invalid packet must not reject the submission");

    assert_eq!(graphics.shader_registers.get(&0x00d), Some(&0x1234_5678));
}

#[test]
fn cond_exec_skips_exactly_its_zero_guarded_dword_range() {
    let predicate = 0_u32;
    let predicate_address = std::ptr::from_ref(&predicate).addr() as u64;
    let commands = [
        agc_pm4(5, crate::AGC_IT_COND_EXEC, 0),
        low_u32(predicate_address) & !3,
        low_u32(predicate_address >> 32),
        0,
        3,
        agc_pm4(3, crate::AGC_IT_SET_SH_REG, 0),
        0x00c,
        0xdead_beef,
        agc_pm4(3, crate::AGC_IT_SET_SH_REG, 0),
        0x00d,
        0x1234_5678,
    ];
    let mut graphics = AgcGraphicsState::default();
    let mut shaders = AgcShaderRegistry::default();
    let mut submission_count = 0;

    crate::agc_parse_submission(
        commands.as_ptr().addr() as u64,
        u32::try_from(commands.len()).expect("command dword count"),
        &mut crate::AgcSubmissionContext {
            graphics: &mut graphics,
            shaders: &mut shaders,
            video_out: &VideoOutState::default(),
            submission_count: &mut submission_count,
            flips: &mut Vec::new(),
            event_writes: &mut Vec::new(),
        },
    )
    .expect("conditional submission");

    assert!(!graphics.shader_registers.contains_key(&0x00c));
    assert_eq!(graphics.shader_registers.get(&0x00d), Some(&0x1234_5678));
}

#[test]
fn native_gen5_branch_executes_only_the_selected_indirect_buffer() {
    let _tracking = crate::agc_command_abi::graphics_tracking_test_guard();
    let comparison_value = 5_u64;
    let comparison_address = std::ptr::from_ref(&comparison_value).addr() as u64;
    let then_commands = [agc_pm4(3, crate::AGC_IT_SET_SH_REG, 0), 0x00c, 0x1111_1111];
    let else_commands = [agc_pm4(3, crate::AGC_IT_SET_SH_REG, 0), 0x00d, 0x2222_2222];
    let then_address = then_commands.as_ptr().addr() as u64;
    let else_address = else_commands.as_ptr().addr() as u64;
    crate::agc_command_abi::track_graphics_allocation(then_address, 3);
    let commands = [
        agc_pm4(14, crate::AGC_IT_INDIRECT_BUFFER, 0),
        2 | (3 << 8),
        low_u32(comparison_address) & !7,
        low_u32(comparison_address >> 32),
        u32::MAX,
        u32::MAX,
        5,
        0,
        low_u32(then_address) & !3,
        low_u32(then_address >> 32),
        3,
        low_u32(else_address) & !3,
        low_u32(else_address >> 32),
        3,
    ];
    let mut graphics = AgcGraphicsState::default();
    let mut shaders = AgcShaderRegistry::default();
    let mut submission_count = 0;

    crate::agc_parse_submission(
        commands.as_ptr().addr() as u64,
        u32::try_from(commands.len()).expect("command dword count"),
        &mut crate::AgcSubmissionContext {
            graphics: &mut graphics,
            shaders: &mut shaders,
            video_out: &VideoOutState::default(),
            submission_count: &mut submission_count,
            flips: &mut Vec::new(),
            event_writes: &mut Vec::new(),
        },
    )
    .expect("branched submission");

    assert_eq!(graphics.shader_registers.get(&0x00c), Some(&0x1111_1111));
    assert!(!graphics.shader_registers.contains_key(&0x00d));
    assert_eq!(
        crate::agc_command_abi::take_recorded_graphics_prefix(then_address + 12),
        None,
        "an already executed native branch target must not be reconstructed before an adjacent tail"
    );
}

#[test]
fn unconditional_gen5_branch_does_not_read_its_queue_label() {
    let then_commands = [agc_pm4(3, crate::AGC_IT_SET_SH_REG, 0), 0x00c, 0x8765_4321];
    let then_address = then_commands.as_ptr().addr() as u64;
    let commands = [
        agc_pm4(14, crate::AGC_IT_INDIRECT_BUFFER, 0),
        1,
        0x0000_1000,
        0,
        0,
        0,
        0,
        0,
        low_u32(then_address) & !3,
        low_u32(then_address >> 32),
        3,
        0,
        0,
        0,
    ];
    let mut graphics = AgcGraphicsState::default();
    let mut shaders = AgcShaderRegistry::default();
    let mut submission_count = 0;

    crate::agc_parse_submission(
        commands.as_ptr().addr() as u64,
        u32::try_from(commands.len()).expect("command dword count"),
        &mut crate::AgcSubmissionContext {
            graphics: &mut graphics,
            shaders: &mut shaders,
            video_out: &VideoOutState::default(),
            submission_count: &mut submission_count,
            flips: &mut Vec::new(),
            event_writes: &mut Vec::new(),
        },
    )
    .expect("unconditional branch");

    assert_eq!(graphics.shader_registers.get(&0x00c), Some(&0x8765_4321));
}

#[test]
fn applies_native_gen5_indexed_uconfig_packet() {
    let packet = [
        agc_pm4(3, AGC_IT_SET_UCONFIG_REG_INDEX, 0),
        0x2000_0243,
        0x400 | 1 | (2 << 6),
    ];
    let mut graphics = AgcGraphicsState::default();

    agc_apply_submission_registers(
        packet.as_ptr().addr() as u64,
        3,
        AGC_IT_SET_UCONFIG_REG_INDEX,
        0,
        &mut graphics,
    )
    .expect("native indexed user-config register packet");

    assert_eq!(graphics.user_config_registers.get(&0x243), Some(&0x481));
    assert_eq!(graphics.index_size, 1);
}

#[test]
fn applies_direct_uconfig_vgt_index_type_to_graphics_index_size() {
    // UE writes VGT_INDEX_TYPE through plain SET_UCONFIG_REG, often with the same selector
    // high bits as the indexed packet (0x2000_0243). Mask those bits and update index_size.
    let packet = [
        agc_pm4(3, AGC_IT_SET_UCONFIG_REG, 0),
        0x2000_0000 | AGC_VGT_INDEX_TYPE,
        0x400 | 1 | (2 << 6),
    ];
    let mut graphics = AgcGraphicsState::default();
    assert_eq!(graphics.index_size, 0);

    agc_apply_submission_registers(
        packet.as_ptr().addr() as u64,
        3,
        AGC_IT_SET_UCONFIG_REG,
        0,
        &mut graphics,
    )
    .expect("direct user-config VGT_INDEX_TYPE packet");

    assert_eq!(graphics.user_config_registers.get(&AGC_VGT_INDEX_TYPE), Some(&0x481));
    assert_eq!(graphics.index_size, 1);
}

#[test]
fn applies_direct_sh_reg_with_selector_high_bits() {
    // UE SET_SH_REG for compute threadgroup size uses the same 0x2000_ selector high bits as
    // SET_SH_REG_INDIRECT tables. Without masking, COMPUTE_NUM_THREAD_* never match decode.
    let packet = [
        agc_pm4(5, AGC_IT_SET_SH_REG, 0),
        0x2000_0000 | remu_gpu::COMPUTE_NUM_THREAD_X_REGISTER,
        64,
        1,
        1,
    ];
    let mut graphics = AgcGraphicsState::default();

    agc_apply_submission_registers(
        packet.as_ptr().addr() as u64,
        5,
        AGC_IT_SET_SH_REG,
        0,
        &mut graphics,
    )
    .expect("direct shader NUM_THREAD packet");

    assert_eq!(
        graphics
            .shader_registers
            .get(&remu_gpu::COMPUTE_NUM_THREAD_X_REGISTER),
        Some(&64)
    );
    assert_eq!(
        graphics
            .shader_registers
            .get(&remu_gpu::COMPUTE_NUM_THREAD_Y_REGISTER),
        Some(&1)
    );
    assert_eq!(
        graphics
            .shader_registers
            .get(&remu_gpu::COMPUTE_NUM_THREAD_Z_REGISTER),
        Some(&1)
    );
    assert!(!graphics.shader_registers.contains_key(&0x2000_0207));
}

#[test]
fn clear_state_resets_complete_queue_state_without_discarding_residency() {
    let address = 0x0070_1234_5600_u64;
    let target = remu_gpu::GpuColorTarget {
        slot: 0,
        address,
        view: 0,
        info: 10 << 2,
        attrib: 0,
        attrib2: ((1_280 - 1) << 14) | (720 - 1),
        attrib3: (27 << 14) | (1 << 24),
    };
    let packet = [agc_pm4(2, crate::AGC_IT_CLEAR_STATE, 0), 0x0f];
    let mut graphics = AgcGraphicsState::default();
    graphics.context_registers.insert(0x318, 1);
    graphics.shader_registers.insert(0x20c, 2);
    graphics.user_config_registers.insert(0x242, 3);
    graphics.index_buffer_address = 0x1000;
    graphics.index_resource_address = 0x2000;
    graphics.index_buffer_count = 9;
    graphics.index_size = 1;
    graphics.instance_count = 7;
    graphics.draw_index_offset = 5;
    graphics.indirect_args_address = 0x4000;
    graphics.emitted_draws = 11;
    graphics.known_color_targets.insert(address, target);

    crate::agc_handle_submission_packet(
        crate::AgcSubmissionPacket {
            address: packet.as_ptr().addr() as u64,
            length: 2,
            opcode: crate::AGC_IT_CLEAR_STATE,
            register: 0,
            predicated: false,
        },
        &mut graphics,
        &mut AgcShaderRegistry::default(),
        &VideoOutState::default(),
        &mut Vec::new(),
        &mut Vec::new(),
        None,
    )
    .expect("valid CLEAR_STATE packet");

    assert!(graphics.context_registers.is_empty());
    assert!(graphics.shader_registers.is_empty());
    assert!(graphics.user_config_registers.is_empty());
    assert_eq!(graphics.index_buffer_address, 0);
    assert_eq!(graphics.index_resource_address, 0);
    assert_eq!(graphics.index_buffer_count, 0);
    assert_eq!(graphics.index_size, 0);
    assert_eq!(graphics.instance_count, 1);
    assert_eq!(graphics.draw_index_offset, 0);
    assert_eq!(graphics.indirect_args_address, 0);
    assert_eq!(graphics.emitted_draws, 11);
    assert_eq!(graphics.known_color_targets.get(&address), Some(&target));
}

#[test]
fn clear_state_rejects_non_native_layouts() {
    let invalid_control = [agc_pm4(2, crate::AGC_IT_CLEAR_STATE, 0), 0x10];
    let mut graphics = AgcGraphicsState::default();
    let mut shaders = AgcShaderRegistry::default();
    let video_out = VideoOutState::default();
    let mut flips = Vec::new();
    let mut event_writes = Vec::new();

    let mut invoke = |address, length, graphics: &mut AgcGraphicsState| {
        crate::agc_handle_submission_packet(
            crate::AgcSubmissionPacket {
                address,
                length,
                opcode: crate::AGC_IT_CLEAR_STATE,
                register: 0,
                predicated: false,
            },
            graphics,
            &mut shaders,
            &video_out,
            &mut flips,
            &mut event_writes,
            None,
        )
    };
    assert!(matches!(
        invoke(invalid_control.as_ptr().addr() as u64, 2, &mut graphics),
        Err(crate::AgcSubmissionError::UnsupportedClearStateControl(
            0x10
        ))
    ));
    assert!(matches!(
        invoke(invalid_control.as_ptr().addr() as u64, 3, &mut graphics),
        Err(crate::AgcSubmissionError::ClearStatePacketLength { actual: 3 })
    ));
}
