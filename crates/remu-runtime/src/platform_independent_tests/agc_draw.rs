// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::collections::BTreeMap;

use std::sync::Arc;

use super::{
    AGC_IT_DISPATCH_DIRECT, AGC_IT_DISPATCH_INDIRECT, AGC_IT_DRAW_INDEX_INDIRECT,
    AGC_IT_DRAW_INDIRECT, AGC_IT_INDEX_BASE, AGC_IT_INDEX_BUFFER_SIZE, AGC_IT_NOP,
    AGC_IT_NUM_INSTANCES, AGC_IT_SET_BASE, AGC_IT_SET_SH_REG_INDIRECT, AGC_R_ACB_RESET,
    AGC_R_DRAW_INDEX_AUTO, AgcColorTargetState, AgcComputeDispatchError, AgcGraphicsState,
    AgcShaderRegistry, AgcSubmitPacket, AlignedComputeProgram, AlignedIndirectArguments,
    ProbeDispatchState, VideoOutBufferAttributes, VideoOutState, agc_apply_set_base,
    agc_color_target_state, agc_dcb_draw_index_auto, agc_dcb_set_index_buffer,
    agc_dcb_set_num_instances, agc_decode_compute_dispatch, agc_driver_submit_acb, agc_pm4,
    compute_graphics_state, draw_required_vertex_record_count, low_u32, read_guest_u64,
    targetless_barrier_acquire_flip_commands, targetless_scanout_test_draw,
    targetless_scanout_test_shader_state, targetless_scanout_test_video, write_guest_u32,
    write_guest_u64,
};

#[test]
fn base_vertex_expands_the_required_indexed_vertex_resource_range() {
    let mut draw = targetless_scanout_test_draw();
    draw.indexed = true;
    draw.index_snapshot = Some(vec![3, 7, 5].into());
    draw.base_vertex = 11;
    assert_eq!(draw_required_vertex_record_count(&draw), Some(19));

    draw.base_vertex = -3;
    assert_eq!(draw_required_vertex_record_count(&draw), Some(5));
}

#[test]
fn primitive_restart_does_not_expand_the_required_vertex_resource_range() {
    let mut draw = targetless_scanout_test_draw();
    draw.indexed = true;
    draw.index_size = 0;
    draw.index_snapshot = Some(vec![3, u32::from(u16::MAX), 7].into());
    draw.context_registers = vec![
        remu_gpu::GpuRegister {
            offset: 0x103,
            value: u32::MAX,
        },
        remu_gpu::GpuRegister {
            offset: 0x2a5,
            value: 1,
        },
    ]
    .into();
    assert_eq!(draw_required_vertex_record_count(&draw), Some(8));
}

#[test]
fn distinguishes_absent_disabled_programmed_and_malformed_color_target_state() {
    let address = 0x0070_1234_5600_u64;
    let mut registers = BTreeMap::from([
        (crate::AGC_CB_COLOR0_BASE, low_u32(address >> 8)),
        (crate::AGC_CB_COLOR0_BASE_EXT, low_u32(address >> 40)),
        (crate::AGC_CB_COLOR0_VIEW, 0),
        (crate::AGC_CB_COLOR0_INFO, 10 << 2),
        (crate::AGC_CB_COLOR0_ATTRIB, 0),
        (
            crate::AGC_CB_COLOR0_ATTRIB2,
            ((1_280 - 1) << 14) | (720 - 1),
        ),
        (crate::AGC_CB_COLOR0_ATTRIB3, (27 << 14) | (1 << 24)),
    ]);

    let AgcColorTargetState::Programmed(targets) = agc_color_target_state(&registers) else {
        panic!("a complete target remains enabled when CB_TARGET_MASK is absent");
    };
    assert_eq!(targets.len(), 1);
    assert_eq!(targets[0].address, address);

    registers.remove(&crate::AGC_CB_COLOR0_VIEW);
    assert!(matches!(
        agc_color_target_state(&registers),
        AgcColorTargetState::Malformed { slot: 0 }
    ));

    registers.insert(crate::AGC_CB_TARGET_MASK, 0);
    assert!(matches!(
        agc_color_target_state(&registers),
        AgcColorTargetState::Disabled
    ));
}

#[test]
fn explicit_zero_target_mask_does_not_enable_targetless_scanout_fallback() {
    let target = crate::AgcVideoOutTarget {
        address: 0x0070_1234_5600,
        attributes: VideoOutBufferAttributes {
            width: 1_280,
            height: 720,
            pitch: 1_280,
            pixel_format: 0x8000_0000_2200_0000,
            tiling_mode: 0,
        },
    };
    let mut graphics = AgcGraphicsState::default();
    graphics
        .context_registers
        .insert(crate::AGC_CB_TARGET_MASK, 0);
    let (_, color_target_absent) = crate::agc_prepare_draw_color_targets(&mut graphics);
    assert!(!color_target_absent);

    let mut draw = targetless_scanout_test_draw();
    draw.pixel_shader_evaluation = Some(Arc::new(remu_shader::ShaderEvaluation {
        initial_scalar_registers: Vec::new().into(),
        scalar_registers: Vec::new().into(),
        scalar_memory_reads: Vec::new(),
        image_bindings: Vec::new(),
        global_memory_bindings: Vec::new(),
        compute_system_registers: None,
        vertex_inputs: Vec::new(),
    }).into());
    crate::agc_resolve_targetless_scanout(
        &mut graphics,
        &mut draw,
        color_target_absent,
        Some(target),
    );
    assert_eq!(draw.scanout_target, None);
}

#[test]
fn scanout_lookahead_is_submission_bounded_and_not_limited_by_channel_capacity() {
    let address = 0x0070_1234_5600_u64;
    let video_out = VideoOutState {
        is_open: true,
        display_buffers: vec![address],
        display_buffer_attributes: vec![Some(VideoOutBufferAttributes {
            width: 1_280,
            height: 720,
            pitch: 1_280,
            pixel_format: 0x8000_0000_2200_0000,
            tiling_mode: 0,
        })],
        ..VideoOutState::default()
    };
    let flip = [agc_pm4(6, AGC_IT_NOP, crate::AGC_R_FLIP), 1, 0, 0, 0, 0];
    let draw_packet = || crate::AgcSubmissionPacket {
        address: 0,
        length: 3,
        opcode: crate::AGC_IT_DRAW_INDEX_AUTO,
        register: 0,
        predicated: false,
    };
    let mut packets = vec![draw_packet()];
    packets.extend((0..(remu_gpu::GPU_EVENT_QUEUE_CAPACITY + 17)).map(|_| {
        crate::AgcSubmissionPacket {
            address: 0,
            length: 2,
            opcode: AGC_IT_NOP,
            register: 0,
            predicated: false,
        }
    }));
    let second_draw = packets.len();
    packets.push(draw_packet());
    packets.push(crate::AgcSubmissionPacket {
        address: flip.as_ptr().addr() as u64,
        length: 6,
        opcode: AGC_IT_NOP,
        register: crate::AGC_R_FLIP,
        predicated: false,
    });

    let future = crate::agc_future_scanout_targets(&packets, &video_out);
    assert_eq!(future[0].map(|target| target.address), Some(address));
    assert_eq!(
        future[second_draw].map(|target| target.address),
        Some(address)
    );

    let boundary_packets = [
        draw_packet(),
        crate::AgcSubmissionPacket {
            address: 0,
            length: 2,
            opcode: crate::AGC_IT_CLEAR_STATE,
            register: 0,
            predicated: false,
        },
        draw_packet(),
        *packets.last().expect("flip packet"),
    ];
    let boundary_future = crate::agc_future_scanout_targets(&boundary_packets, &video_out);
    assert_eq!(boundary_future[0], None);
    assert_eq!(
        boundary_future[2].map(|target| target.address),
        Some(address)
    );
}

#[test]
fn executes_native_draw_index_2_with_embedded_index_base_and_count() {
    let mut indices = [0_u16, 1, 2, 2, 3, 0];
    let index_address = indices.as_ptr().addr() as u64;
    let commands = [
        agc_pm4(6, crate::AGC_IT_DRAW_INDEX_2, 0),
        6,
        low_u32(index_address),
        low_u32(index_address >> 32),
        6,
        2,
    ];
    let (sender, receiver) = remu_gpu::gpu_event_channel();
    remu_gpu::install_thread_local_event_sink(sender);
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
    .expect("native DRAW_INDEX_2 submission");
    remu_gpu::clear_thread_local_event_sink();

    let remu_gpu::GpuEvent::IndexedDraw(draw) = receiver.recv().expect("indexed draw event") else {
        panic!("DRAW_INDEX_2 emitted a non-draw event");
    };
    assert!(draw.indexed);
    assert_eq!(draw.index_buffer_address, index_address);
    assert_eq!(draw.index_resource_address, index_address);
    assert_eq!(draw.index_buffer_count, 6);
    assert_eq!(draw.index_offset, 0);
    assert_eq!(draw.base_vertex, 0);
    assert_eq!(draw.index_count, 6);
    indices.fill(u16::MAX);
    assert_eq!(
        draw.index_snapshot.as_deref(),
        Some([0, 1, 2, 2, 3, 0].as_slice())
    );
    assert_eq!(draw.draw_flags, 2);
    assert_eq!(submission_count, 1);
    assert!(receiver.try_recv().is_err());
}

#[test]
fn executes_nonindexed_indirect_draw_from_a_compute_typed_set_base() {
    let mut arguments = AlignedIndirectArguments([6, 3, 4, 7, 0, 0, 0, 0]);
    let arguments_address = arguments.0.as_ptr().addr() as u64;
    let set_base = [
        agc_pm4(4, AGC_IT_SET_BASE, 0) | (1 << 1),
        1,
        low_u32(arguments_address),
        low_u32(arguments_address >> 32),
    ];
    let packet = [agc_pm4(5, AGC_IT_DRAW_INDIRECT, 0), 0, 0x08c, 0x08d, 2];
    let mut graphics = AgcGraphicsState::default();
    let mut shaders = AgcShaderRegistry::default();
    let mut flips = Vec::new();
    let mut event_writes = Vec::new();
    agc_apply_set_base(set_base.as_ptr().addr() as u64, 4, &mut graphics)
        .expect("compute-routed shared indirect base");

    let (result, events) = crate::capture_agc_gpu_events(|| {
        crate::agc_handle_submission_packet(
            crate::AgcSubmissionPacket {
                address: packet.as_ptr().addr() as u64,
                length: 5,
                opcode: AGC_IT_DRAW_INDIRECT,
                register: 0,
                predicated: false,
            },
            &mut graphics,
            &mut shaders,
            &VideoOutState::default(),
            &mut flips,
            &mut event_writes,
            None,
        )
    });
    result.expect("nonindexed indirect draw");

    let [remu_gpu::GpuEvent::IndexedDraw(draw)] = events.as_slice() else {
        panic!("indirect draw emitted unexpected events: {events:?}");
    };
    assert!(!draw.indexed);
    assert_eq!(draw.index_count, 6);
    assert_eq!(draw.instance_count, 3);
    assert_eq!(draw.index_offset, 4);
    assert_eq!(draw.base_vertex, 0);
    assert_eq!(draw.draw_flags, 2);
    assert_eq!(draw.indirect_arguments_address, Some(arguments_address));
    assert_eq!(
        draw.shader_registers
            .iter()
            .find(|register| register.offset == 0x08c)
            .map(|register| register.value),
        Some(0)
    );
    assert_eq!(
        draw.shader_registers
            .iter()
            .find(|register| register.offset == 0x08d)
            .map(|register| register.value),
        Some(0)
    );
    // Neutralized SH remaps restore the seeded AGC defaults (0), not an empty map.
    assert_eq!(graphics.shader_registers.get(&0x08c), Some(&0));
    assert_eq!(graphics.shader_registers.get(&0x08d), Some(&0));

    arguments.0[..4].fill(0);
    let (result, events) = crate::capture_agc_gpu_events(|| {
        crate::agc_handle_submission_packet(
            crate::AgcSubmissionPacket {
                address: packet.as_ptr().addr() as u64,
                length: 5,
                opcode: AGC_IT_DRAW_INDIRECT,
                register: 0,
                predicated: false,
            },
            &mut graphics,
            &mut shaders,
            &VideoOutState::default(),
            &mut flips,
            &mut event_writes,
            None,
        )
    });
    result.expect("zero-snapshot nonindexed indirect draw");
    let [remu_gpu::GpuEvent::IndexedDraw(draw)] = events.as_slice() else {
        panic!("zero-snapshot indirect draw must reach the ordered backend");
    };
    assert_eq!(draw.indirect_arguments_address, Some(arguments_address));
    assert_eq!(draw.index_count, 0);
    assert_eq!(draw.instance_count, 0);
}

#[test]
fn executes_indexed_indirect_draw_with_all_five_argument_fields() {
    #[repr(C, align(8))]
    struct OffsetIndirectArguments {
        padding: [u32; 10],
        draw: [u32; 5],
    }
    let arguments = OffsetIndirectArguments {
        padding: [0; 10],
        draw: [6, 2, 1, (-3_i32) as u32, 9],
    };
    let arguments_address = arguments.padding.as_ptr().addr() as u64;
    let indices = [10_u16, 11, 12, 13, 14, 15, 16, 17];
    let index_address = indices.as_ptr().addr() as u64;
    let set_base = [
        agc_pm4(4, AGC_IT_SET_BASE, 0),
        1,
        low_u32(arguments_address),
        low_u32(arguments_address >> 32),
    ];
    let packet = [
        agc_pm4(5, AGC_IT_DRAW_INDEX_INDIRECT, 0),
        0x28,
        0x08d_008c,
        0x08e,
        2,
    ];
    let mut graphics = AgcGraphicsState {
        index_buffer_address: index_address,
        index_resource_address: index_address,
        index_buffer_count: u32::try_from(indices.len()).expect("index count"),
        index_size: 1,
        ..AgcGraphicsState::default()
    };
    graphics
        .context_registers
        .insert(crate::AGC_CB_TARGET_MASK, 0xf);
    let mut shaders = AgcShaderRegistry::default();
    let mut flips = Vec::new();
    let mut event_writes = Vec::new();
    agc_apply_set_base(set_base.as_ptr().addr() as u64, 4, &mut graphics)
        .expect("graphics indirect base");

    let (result, events) = crate::capture_agc_gpu_events(|| {
        crate::agc_handle_submission_packet(
            crate::AgcSubmissionPacket {
                address: packet.as_ptr().addr() as u64,
                length: 5,
                opcode: AGC_IT_DRAW_INDEX_INDIRECT,
                register: 0,
                predicated: false,
            },
            &mut graphics,
            &mut shaders,
            &VideoOutState::default(),
            &mut flips,
            &mut event_writes,
            None,
        )
    });
    result.expect("indexed indirect draw");

    let [remu_gpu::GpuEvent::IndexedDraw(draw)] = events.as_slice() else {
        panic!("indexed indirect draw emitted unexpected events: {events:?}");
    };
    assert!(draw.indexed);
    assert_eq!(draw.index_count, 6);
    assert_eq!(draw.instance_count, 2);
    assert_eq!(draw.index_offset, 1);
    assert_eq!(draw.base_vertex, -3);
    assert_eq!(draw.draw_flags, 2);
    assert_eq!(
        draw.indirect_arguments_address,
        Some(arguments_address + 0x28)
    );
    assert_eq!(
        draw.shader_registers
            .iter()
            .find(|register| register.offset == 0x08c)
            .map(|register| register.value),
        Some(0)
    );
    // Neutralized SH remaps restore the seeded AGC defaults (0), not an empty map.
    assert_eq!(graphics.shader_registers.get(&0x08c), Some(&0));
    assert_eq!(graphics.shader_registers.get(&0x08d), Some(&0));
    assert_eq!(graphics.shader_registers.get(&0x08e), Some(&0));
}

#[test]
fn targetless_scanout_resolution_requires_complete_unambiguous_draw_state() {
    let target = crate::AgcVideoOutTarget {
        address: 0x0070_1234_5600,
        attributes: VideoOutBufferAttributes {
            width: 1_280,
            height: 720,
            pitch: 1_280,
            pixel_format: 0x8000_0000_2200_0000,
            tiling_mode: 0,
        },
    };
    let mut graphics = AgcGraphicsState::default();
    let mut draw = targetless_scanout_test_draw();

    crate::agc_resolve_targetless_scanout(&mut graphics, &mut draw, true, Some(target));
    assert_eq!(
        draw.scanout_target, None,
        "failed resource evaluation is not guessed"
    );

    draw.pixel_shader_evaluation = Some(Arc::new(remu_shader::ShaderEvaluation {
        initial_scalar_registers: Vec::new().into(),
        scalar_registers: Vec::new().into(),
        scalar_memory_reads: Vec::new(),
        image_bindings: Vec::new(),
        global_memory_bindings: Vec::new(),
        compute_system_registers: None,
        vertex_inputs: Vec::new(),
    }).into());
    crate::agc_resolve_targetless_scanout(&mut graphics, &mut draw, true, Some(target));
    assert_eq!(
        draw.scanout_target, None,
        "a constant export with no sampled source is not a display composite"
    );

    draw.pixel_shader_evaluation = Some(Arc::new(remu_shader::ShaderEvaluation {
        initial_scalar_registers: Vec::new().into(),
        scalar_registers: Vec::new().into(),
        scalar_memory_reads: Vec::new(),
        image_bindings: vec![remu_shader::ImageBinding { descriptor_index_is_runtime: false,
            pc: 0,
            opcode: remu_shader::Opcode::Known("ImageSample"),
            control: remu_shader::ImageControl {
                dmask: 0xf,
                vector_address: 0,
                address_registers: Vec::new(),
                vector_data: 0,
                scalar_resource: 0,
                scalar_sampler: 0,
                dimension: 2,
                is_array: false,
                glc: false,
                slc: false,
                a16: false,
                d16: false,
                unorm: false,
            },
            resource_descriptor: vec![1, 0, 0, 0].into(),
            sampler_descriptor: vec![0; 4].into(),
            mip_level: None,
        }],
        global_memory_bindings: Vec::new(),
        compute_system_registers: None,
        vertex_inputs: Vec::new(),
    }).into());
    crate::agc_resolve_targetless_scanout(&mut graphics, &mut draw, true, Some(target));
    assert_eq!(
        draw.scanout_target.map(|scanout| scanout.address),
        Some(target.address),
        "a side-effect-free sampled draw may be the display composite"
    );

    draw.scanout_target = None;
    graphics
        .context_registers
        .insert(crate::AGC_DB_DEPTH_CONTROL, 0x2);
    crate::agc_resolve_targetless_scanout(&mut graphics, &mut draw, true, Some(target));
    assert_eq!(
        draw.scanout_target.map(|scanout| scanout.address),
        Some(target.address),
        "stale depth control without a bound depth surface does not suppress scanout"
    );

    draw.scanout_target = None;
    graphics.context_registers.insert(crate::AGC_DB_Z_INFO, 3);
    graphics
        .context_registers
        .insert(crate::AGC_DB_DEPTH_SIZE_XY, 1_279 | (719 << 16));
    graphics
        .context_registers
        .insert(crate::AGC_DB_Z_READ_BASE, 0x0012_3456);
    crate::agc_resolve_targetless_scanout(&mut graphics, &mut draw, true, Some(target));
    assert_eq!(
        draw.scanout_target, None,
        "a complete active depth target is not inferred as scanout"
    );

    graphics.context_registers.clear();
    crate::agc_resolve_targetless_scanout(&mut graphics, &mut draw, true, None);
    assert_eq!(
        draw.scanout_target, None,
        "association never crosses a missing flip"
    );

    let pixel = draw.pixel_shader.as_mut().expect("test pixel shader");
    let mut instructions = pixel.program.instructions.clone();
    instructions.push(remu_shader::Instruction {
        pc: 8,
        encoding: remu_shader::ShaderEncoding::Exp,
        opcode: remu_shader::Opcode::EXPORT,
        words: vec![0, 0],
        sources: Vec::new(),
        destinations: Vec::new(),
        control: Some(remu_shader::InstructionControl::Export(
            remu_shader::ExportControl {
                target: 1,
                enable_mask: 0xf,
                compressed: false,
                done: true,
                valid_mask: false,
            },
        )),
    });
    pixel.program = Arc::new(remu_shader::Program::new(
        pixel.program.address,
        instructions,
    ));
    Arc::make_mut(&mut draw.context_registers)
        .iter_mut()
        .find(|register| register.offset == crate::AGC_CB_SHADER_MASK)
        .expect("test shader mask")
        .value = 0xff;
    crate::agc_resolve_targetless_scanout(&mut graphics, &mut draw, true, Some(target));
    assert_eq!(
        draw.scanout_target, None,
        "multiple color exports are ambiguous"
    );
}

#[test]
fn constant_targetless_draw_does_not_overwrite_ordered_scanout() {
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
    let pixel_code = AlignedComputeProgram([0; 4]);
    let pixel_address = pixel_code.0.as_ptr().addr() as u64;
    let (mut graphics, mut shaders) = targetless_scanout_test_shader_state(pixel_address, target);
    let video_out = targetless_scanout_test_video(address);
    let (commands, expected_acquire) = targetless_barrier_acquire_flip_commands();
    let (sender, receiver) = remu_gpu::gpu_event_channel();
    remu_gpu::install_thread_local_event_sink(sender);

    let mut submission_count = 0;
    let mut flips = Vec::new();
    let mut event_writes = Vec::new();
    crate::agc_parse_submission(
        commands.as_ptr().addr() as u64,
        u32::try_from(commands.len()).expect("command dword count"),
        &mut crate::AgcSubmissionContext {
            graphics: &mut graphics,
            shaders: &mut shaders,
            video_out: &video_out,
            submission_count: &mut submission_count,
            flips: &mut flips,
            event_writes: &mut event_writes,
        },
    )
    .expect("ordered targetless submission");
    remu_gpu::clear_thread_local_event_sink();

    let remu_gpu::GpuEvent::IndexedDraw(draw) = receiver.recv().expect("constant draw event")
    else {
        panic!("draw must precede flip");
    };
    assert_eq!(
        draw.scanout_target, None,
        "an evaluated constant export is not redirected into VideoOut"
    );
    assert_eq!(
        receiver.recv().expect("pixel partial flush"),
        remu_gpu::GpuEvent::PipelineBarrier(remu_gpu::GpuPipelineBarrier {
            wait_for: remu_gpu::GpuPipelineStage::PixelShader,
        })
    );
    assert_eq!(
        receiver.recv().expect("memory acquire"),
        remu_gpu::GpuEvent::MemoryAcquire(expected_acquire)
    );
    let remu_gpu::GpuEvent::Flip {
        address: flipped_address,
        width,
        height,
        ..
    } = receiver.recv().expect("ordered flip event")
    else {
        panic!("flip must follow draw");
    };
    assert_eq!(flipped_address, address);
    assert_eq!((width, height), (1_280, 720));
    assert_eq!(flips, vec![(1, 0, 0x1234)]);
    assert!(receiver.try_recv().is_err());
}

#[test]
fn submits_descriptor_wrapped_acb_work_on_an_owner_scoped_compute_queue() {
    let program = AlignedComputeProgram([0, 0, 0, 0]);
    let program_address = program.0.as_ptr().addr() as u64;
    let (_, shaders) = compute_graphics_state(program_address);
    let register_pairs = [
        remu_gpu::COMPUTE_PGM_LO_REGISTER,
        low_u32(program_address >> 8),
        remu_gpu::COMPUTE_PGM_HI_REGISTER,
        ((program_address >> 40) & 0xff) as u32,
        remu_gpu::COMPUTE_PGM_RSRC2_REGISTER,
        0,
        remu_gpu::COMPUTE_NUM_THREAD_X_REGISTER,
        64,
        remu_gpu::COMPUTE_NUM_THREAD_Y_REGISTER,
        1,
        remu_gpu::COMPUTE_NUM_THREAD_Z_REGISTER,
        1,
    ];
    let registers_address = register_pairs.as_ptr().addr() as u64;
    let commands = [
        agc_pm4(2, AGC_IT_NOP, AGC_R_ACB_RESET),
        0,
        agc_pm4(5, AGC_IT_SET_SH_REG_INDIRECT, 0),
        low_u32(registers_address),
        (registers_address >> 32) as u32,
        0x8000_0000,
        6,
        agc_pm4(5, AGC_IT_DISPATCH_DIRECT, 0),
        3,
        2,
        1,
        0x8045,
    ];
    let command_address = commands.as_ptr().addr() as u64;
    let descriptor = [
        low_u32(command_address),
        (command_address >> 32) as u32,
        u32::try_from(commands.len()).expect("command dword count"),
        0,
        0x5533_ccaa,
    ];
    let packet = AgcSubmitPacket {
        address: descriptor.as_ptr().addr() as u64,
        dword_count: 5,
        flags: [0; 4],
    };
    let (sender, receiver) = remu_gpu::gpu_event_channel();
    remu_gpu::install_thread_local_event_sink(sender);
    let mut state = ProbeDispatchState::default();
    crate::agc_dispatch_state::with_agc_mut(|agc| {
        agc.shader_registry = shaders;
    });

    let result = agc_driver_submit_acb(
        [7, (&raw const packet).addr() as u64, 0, 0, 0, 0],
        &mut state,
    );
    let submission_event = receiver.recv_timeout(std::time::Duration::from_secs(1));
    remu_gpu::clear_thread_local_event_sink();

    assert_eq!(result, 0);
    let remu_gpu::GpuEvent::QueueSubmission { queue_id, events } =
        submission_event.expect("compute queue submission")
    else {
        panic!("ACB did not preserve its compute queue");
    };
    assert_eq!(queue_id, 8);
    let [
        remu_gpu::GpuEvent::ComputeDispatch(dispatch),
        remu_gpu::GpuEvent::SubmissionBoundary(1),
    ] = events.as_slice()
    else {
        panic!("ACB submission did not retain compute dispatch and completion order");
    };
    assert_eq!(dispatch.source, remu_gpu::ComputeDispatchSource::Direct);
    assert_eq!(
        dispatch.launch.grid,
        remu_gpu::ComputeGrid::Threadgroups([3, 2, 1])
    );
    assert_eq!(dispatch.shader_address, program_address);
    // Graphics SH is seeded from AGC defaults; ACB work must not publish compute PGM into it.
    assert_ne!(
        crate::agc_dispatch_state::with_agc(|agc| agc
            .graphics
            .shader_registers
            .get(&remu_gpu::COMPUTE_PGM_LO_REGISTER)
            .copied()),
        Some(low_u32(program_address >> 8))
    );
    assert_eq!(
        crate::agc_dispatch_state::with_agc(|agc| {
            agc.compute_queues.get(&7).and_then(|queue| {
                queue
                    .shader_registers
                    .get(&remu_gpu::COMPUTE_PGM_LO_REGISTER)
                    .copied()
            })
        }),
        Some(low_u32(program_address >> 8))
    );
    assert_eq!(
        crate::event_state::with_event_state(|e| e
            .pending_agc_completions
            .iter()
            .copied()
            .collect::<Vec<_>>()),
        vec![]
    );
    assert_eq!(
        crate::event_state::with_event_state(|e| e
            .pending_agc_user_interrupts
            .iter()
            .copied()
            .collect::<Vec<_>>()),
        vec![1]
    );
}

#[test]
fn preserves_zero_snapshot_indirect_compute_for_ordered_gpu_resolution() {
    let program = AlignedComputeProgram([0; 4]);
    let program_address = program.0.as_ptr().addr() as u64;
    let (mut graphics, mut shaders) = compute_graphics_state(program_address);
    let dimensions = AlignedIndirectArguments([0, 8, 1, 0, 0, 0, 0, 0]);
    graphics.indirect_args_address = dimensions.0.as_ptr().addr() as u64;
    let packet = [agc_pm4(3, AGC_IT_DISPATCH_INDIRECT, 0), 0, 0x8041];
    let mut flips = Vec::new();
    let mut event_writes = Vec::new();

    let (result, gpu_events) = crate::capture_agc_gpu_events(|| {
        crate::agc_handle_submission_packet(
            crate::AgcSubmissionPacket {
                address: packet.as_ptr().addr() as u64,
                length: 3,
                opcode: AGC_IT_DISPATCH_INDIRECT,
                register: 0,
                predicated: false,
            },
            &mut graphics,
            &mut shaders,
            &VideoOutState::default(),
            &mut flips,
            &mut event_writes,
            None,
        )
    });

    result.expect("zero-snapshot indirect dispatch remains valid");
    let [remu_gpu::GpuEvent::ComputeDispatch(dispatch)] = gpu_events.as_slice() else {
        panic!("indirect dispatch must reach the backend for ordered argument resolution");
    };
    assert_eq!(
        dispatch.source,
        remu_gpu::ComputeDispatchSource::Indirect {
            dimensions_address: dimensions.0.as_ptr().addr() as u64,
        }
    );
    assert!(
        dispatch.launch.is_empty(),
        "packet-time snapshot stays diagnostic"
    );
    assert_eq!(graphics.emitted_compute_dispatches, 1);
}

#[test]
fn decodes_and_evaluates_a_complete_direct_compute_dispatch_snapshot() {
    let program = AlignedComputeProgram([0xbeef_cafe, 0x1234_5678, 0, 0]);
    let program_address = program.0.as_ptr().addr() as u64;
    let (mut graphics, mut shaders) = compute_graphics_state(program_address);
    let packet = [agc_pm4(5, AGC_IT_DISPATCH_DIRECT, 0), 10, 4, 2, 0x8045];

    let dispatch = agc_decode_compute_dispatch(
        packet.as_ptr().addr() as u64,
        5,
        AGC_IT_DISPATCH_DIRECT,
        &mut graphics,
        &mut shaders,
    )
    .expect("complete direct compute dispatch");

    assert_eq!(dispatch.sequence, 1);
    assert_eq!(dispatch.source, remu_gpu::ComputeDispatchSource::Direct);
    assert_eq!(
        dispatch.launch.grid,
        remu_gpu::ComputeGrid::Threadgroups([10, 4, 2])
    );
    assert_eq!(dispatch.launch.threads_per_group, [64, 2, 1]);
    assert_eq!(dispatch.launch.wave_lane_count, 32);
    assert_eq!(dispatch.shader_address, program_address);
    assert_eq!(dispatch.shader_prefix, program.0);
    let evaluation = dispatch
        .shader_evaluation
        .as_ref()
        .expect("compute evaluation");
    assert_eq!(evaluation.initial_scalar_registers[0], 0x1234_5678);
    assert_eq!(evaluation.initial_scalar_registers[1], 0x9abc_def0);
    let system = evaluation
        .compute_system_registers
        .expect("compute system registers");
    assert_eq!(system.work_group_x, Some(2));
    assert_eq!(system.work_group_y, None);
    assert_eq!(system.work_group_z, Some(3));
    assert_eq!(system.thread_group_size, Some(4));
}

#[test]
fn resolves_base_relative_and_absolute_compute_indirect_dimensions() {
    let program = AlignedComputeProgram([0, 0, 0, 0]);
    let program_address = program.0.as_ptr().addr() as u64;
    let (mut graphics, mut shaders) = compute_graphics_state(program_address);
    let arguments = AlignedIndirectArguments([0, 0, 7, 3, 1, 11, 5, 2]);
    let arguments_address = arguments.0.as_ptr().addr() as u64;
    let compute_set_base = [
        agc_pm4(4, AGC_IT_SET_BASE, 0) | (1 << 1),
        1,
        low_u32(arguments_address),
        (arguments_address >> 32) as u32,
    ];
    agc_apply_set_base(compute_set_base.as_ptr().addr() as u64, 4, &mut graphics)
        .expect("compute indirect base");
    assert_eq!(graphics.indirect_args_address, arguments_address);

    let graphics_arguments = AlignedIndirectArguments([13, 17, 19, 23, 29, 31, 37, 41]);
    let graphics_arguments_address = graphics_arguments.0.as_ptr().addr() as u64;
    let graphics_set_base = [
        agc_pm4(4, AGC_IT_SET_BASE, 0),
        1,
        low_u32(graphics_arguments_address),
        (graphics_arguments_address >> 32) as u32,
    ];
    agc_apply_set_base(graphics_set_base.as_ptr().addr() as u64, 4, &mut graphics)
        .expect("graphics indirect base");
    assert_eq!(graphics.indirect_args_address, graphics_arguments_address);

    let relative = [agc_pm4(3, AGC_IT_DISPATCH_INDIRECT, 0), 8, 0x8041];
    let relative_dispatch = agc_decode_compute_dispatch(
        relative.as_ptr().addr() as u64,
        3,
        AGC_IT_DISPATCH_INDIRECT,
        &mut graphics,
        &mut shaders,
    )
    .expect("base-relative compute dispatch");
    assert_eq!(
        relative_dispatch.source,
        remu_gpu::ComputeDispatchSource::Indirect {
            dimensions_address: graphics_arguments_address + 8,
        }
    );
    assert_eq!(
        relative_dispatch.launch.grid,
        remu_gpu::ComputeGrid::Threadgroups([19, 23, 29])
    );

    let absolute_address = arguments_address + 20;
    let absolute = [
        agc_pm4(4, AGC_IT_DISPATCH_INDIRECT, 0),
        low_u32(absolute_address),
        (absolute_address >> 32) as u32,
        0x8041,
    ];
    let absolute_dispatch = agc_decode_compute_dispatch(
        absolute.as_ptr().addr() as u64,
        4,
        AGC_IT_DISPATCH_INDIRECT,
        &mut graphics,
        &mut shaders,
    )
    .expect("absolute compute dispatch");
    assert_eq!(
        absolute_dispatch.source,
        remu_gpu::ComputeDispatchSource::Indirect {
            dimensions_address: absolute_address,
        }
    );
    assert_eq!(
        absolute_dispatch.launch.grid,
        remu_gpu::ComputeGrid::Threadgroups([11, 5, 2])
    );
}

#[test]
fn rejects_indirect_compute_without_a_set_base_packet() {
    let program = AlignedComputeProgram([0; 4]);
    let (mut graphics, mut shaders) = compute_graphics_state(program.0.as_ptr().addr() as u64);
    let packet = [agc_pm4(3, AGC_IT_DISPATCH_INDIRECT, 0), 0x80, 0x8041];
    assert!(matches!(
        agc_decode_compute_dispatch(
            packet.as_ptr().addr() as u64,
            3,
            AGC_IT_DISPATCH_INDIRECT,
            &mut graphics,
            &mut shaders,
        ),
        Err(AgcComputeDispatchError::MissingIndirectBase)
    ));
}

#[test]
fn builds_the_three_argument_gen5_index_buffer_packets() {
    let mut command_buffer = vec![0_u8; 0x40];
    let mut commands = vec![0_u32; 16];
    let indices = [0_u32, 1, 2, 1, 2, 3];
    let command_buffer_address =
        u64::try_from(command_buffer.as_mut_ptr().addr()).expect("command buffer address");
    let commands_address =
        u64::try_from(commands.as_mut_ptr().addr()).expect("command storage address");
    let index_buffer_address =
        u64::try_from(indices.as_ptr().addr()).expect("index buffer address");
    write_guest_u64(command_buffer_address + 0x10, commands_address);
    write_guest_u64(command_buffer_address + 0x18, commands_address + 64);
    write_guest_u32(command_buffer_address + 0x30, 0);

    let command =
        agc_dcb_set_index_buffer([command_buffer_address, index_buffer_address, 16, 0, 0, 0]);

    assert_eq!(command, commands_address);
    assert_eq!(commands[0], agc_pm4(3, AGC_IT_INDEX_BASE, 0));
    assert_eq!(commands[1], low_u32(index_buffer_address));
    assert_eq!(commands[2], low_u32(index_buffer_address >> 32));
    assert_eq!(commands[3], agc_pm4(2, AGC_IT_INDEX_BUFFER_SIZE, 0));
    assert_eq!(commands[4], 16);
    assert_eq!(
        read_guest_u64(command_buffer_address + 0x10),
        commands_address + 20
    );
}

#[test]
fn builds_the_gen5_instanced_auto_draw_packets() {
    let mut command_buffer = vec![0_u8; 0x40];
    let mut commands = vec![0_u32; 16];
    let command_buffer_address =
        u64::try_from(command_buffer.as_mut_ptr().addr()).expect("command buffer address");
    let commands_address =
        u64::try_from(commands.as_mut_ptr().addr()).expect("command storage address");
    write_guest_u64(command_buffer_address + 0x10, commands_address);
    write_guest_u64(command_buffer_address + 0x18, commands_address + 64);
    write_guest_u32(command_buffer_address + 0x30, 0);

    let instances = agc_dcb_set_num_instances([command_buffer_address, 3, 0, 0, 0, 0]);
    let draw = agc_dcb_draw_index_auto([command_buffer_address, 3, 0x4000_0000, 0, 0, 0]);

    assert_eq!(instances, commands_address);
    assert_eq!(commands[0], agc_pm4(2, AGC_IT_NUM_INSTANCES, 0));
    assert_eq!(commands[1], 3);
    assert_eq!(draw, commands_address + 8);
    assert_eq!(commands[2], agc_pm4(7, AGC_IT_NOP, AGC_R_DRAW_INDEX_AUTO));
    assert_eq!(commands[3], 3);
    assert_eq!(&commands[4..9], &[0; 5]);
    assert_eq!(
        read_guest_u64(command_buffer_address + 0x10),
        commands_address + 36
    );
}
