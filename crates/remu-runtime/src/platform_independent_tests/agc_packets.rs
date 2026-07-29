// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::collections::BTreeMap;

use super::{
    AGC_IT_NOP, AGC_IT_SET_CONTEXT_REG_INDIRECT, AGC_IT_SET_SH_REG_INDIRECT,
    AGC_IT_SET_UCONFIG_REG_INDEX, AGC_IT_SET_UCONFIG_REG_INDIRECT, AgcGraphicsState,
    AgcShaderRegistry, GPU_EVENT_SINK_TEST, VideoOutState, agc_apply_submission_registers, agc_pm4,
    decode_agc_dma_data, decode_agc_memory_acquire, decode_agc_memory_wait, low_u32,
};

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
            cpu_applied: false,
        })
    );
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
            index_resources: &BTreeMap::new(),
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
            index_resources: &BTreeMap::new(),
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
            index_resources: &BTreeMap::new(),
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
fn decodes_partial_flush_events_without_inventing_completion_notifications() {
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
        &BTreeMap::new(),
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
        None
    );
    crate::agc_handle_submission_packet(
        crate::AgcSubmissionPacket {
            address: pixel_pipe_stat_dump.as_ptr().addr() as u64,
            length: 4,
            opcode: crate::AGC_IT_EVENT_WRITE,
            register: 0,
            predicated: false,
        },
        &mut graphics,
        &mut AgcShaderRegistry::default(),
        &BTreeMap::new(),
        &VideoOutState::default(),
        &mut Vec::new(),
        &mut notifications,
        None,
    )
    .expect("pixel-pipe statistic dump");
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
            index_resources: &BTreeMap::new(),
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
            index_resources: &BTreeMap::new(),
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
            index_resources: &BTreeMap::new(),
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
            index_resources: &BTreeMap::new(),
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
    graphics.compute_indirect_args_address = 0x3000;
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
        &BTreeMap::new(),
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
    assert_eq!(graphics.compute_indirect_args_address, 0);
    assert_eq!(graphics.emitted_draws, 11);
    assert_eq!(graphics.known_color_targets.get(&address), Some(&target));
}

#[test]
fn clear_state_rejects_non_native_layouts() {
    let invalid_control = [agc_pm4(2, crate::AGC_IT_CLEAR_STATE, 0), 0x10];
    let mut graphics = AgcGraphicsState::default();
    let mut shaders = AgcShaderRegistry::default();
    let index_resources = BTreeMap::new();
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
            &index_resources,
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
