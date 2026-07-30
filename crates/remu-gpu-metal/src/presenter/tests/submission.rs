// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use metal::{
    Device, MTLClearColor, MTLLoadAction, MTLPixelFormat, MTLPrimitiveTopologyClass,
    MTLResourceOptions, MTLStorageMode, MTLStoreAction, MTLTextureType, MTLTextureUsage,
    RenderPassDescriptor, TextureDescriptor,
};
use remu_gpu::{
    GpuEvent, GpuMemoryAcquire, GpuMemoryRelease, GpuMemoryReleaseValue, GpuMemoryWait,
    GpuMemoryWrite, GpuPipelineBarrier, GpuPipelineStage, gpu_event_channel,
};
use remu_memory::{
    CpuWriteWatch, GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestProtection,
};
use remu_shader::{GlobalMemoryBinding, Program};
use std::time::Duration;

use super::{
    GLOBAL_UPLOAD_ALIGNMENT, GraphicsPipelineKey, GraphicsShaderResourceKey, GuestBatchFlushReason,
    GuestCommandBatch, GuestMemoryWaitKey, GuestMemoryWaitRecovery, GuestMemoryWaitStatus,
    GuestMemoryWaitTracker, GuestRenderPassKey, MetalPresenter, OpenGuestRenderPass,
    OrderedEventPoll, OrderedEventSource, OrderedGpuEventReceiver, PendingGlobalWriteback,
    PendingGuestMemoryWrite, QueueBlockReason, apply_pending_guest_memory_writes, indexed_draw,
    write_back_pending_globals,
};

#[test]
fn partial_pipeline_flushes_coalesce_without_reordering_submission_completion() {
    let (sender, receiver) = gpu_event_channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let queue = presenter
        .command_queue
        .as_ref()
        .expect("Metal queue")
        .clone();
    let texture_descriptor = TextureDescriptor::new();
    texture_descriptor.set_texture_type(MTLTextureType::D2);
    texture_descriptor.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
    texture_descriptor.set_width(4);
    texture_descriptor.set_height(4);
    texture_descriptor.set_storage_mode(MTLStorageMode::Private);
    texture_descriptor.set_usage(MTLTextureUsage::RenderTarget);
    let texture = device.new_texture(&texture_descriptor);
    let render_pass = RenderPassDescriptor::new();
    let attachment = render_pass
        .color_attachments()
        .object_at(0)
        .expect("color attachment zero");
    attachment.set_texture(Some(&texture));
    attachment.set_load_action(MTLLoadAction::Clear);
    attachment.set_store_action(MTLStoreAction::Store);
    attachment.set_clear_color(MTLClearColor::new(0.0, 0.0, 0.0, 1.0));

    let command_buffer = queue.new_command_buffer().to_owned();
    let original_command_buffer = command_buffer.clone();
    let encoder = command_buffer
        .new_render_command_encoder(render_pass)
        .to_owned();
    presenter.open_guest_batch = Some(GuestCommandBatch {
        command_buffer,
        work: Vec::new(),
        global_upload_chunks: Vec::new(),
        global_writebacks: Vec::new(),
        guest_write_ranges: Vec::new(),
        memory_writes: Vec::new(),
        render_pass: Some(OpenGuestRenderPass {
            key: GuestRenderPassKey {
                targets: Vec::new(),
                depth: None,
            },
            encoder,
        }),
        completion_sequences: vec![40],
    });
    sender
        .send(GpuEvent::QueueSubmission {
            queue_id: 7,
            events: vec![
                GpuEvent::PipelineBarrier(GpuPipelineBarrier {
                    wait_for: GpuPipelineStage::PixelShader,
                }),
                GpuEvent::SubmissionBoundary(41),
                GpuEvent::PipelineBarrier(GpuPipelineBarrier {
                    wait_for: GpuPipelineStage::ComputeShader,
                }),
                GpuEvent::SubmissionBoundary(42),
                GpuEvent::PipelineBarrier(GpuPipelineBarrier {
                    wait_for: GpuPipelineStage::VertexShader,
                }),
            ],
        })
        .expect("queue partial flushes");

    assert!(matches!(
        presenter.receive_events(),
        super::PresenterFlow::Continue
    ));
    let open = presenter
        .open_guest_batch
        .as_ref()
        .expect("partial flush retains the open Metal batch");
    assert!(
        open.render_pass.is_some(),
        "an in-pass barrier must not spill and reload unchanged render attachments"
    );
    assert!(std::ptr::eq(
        open.command_buffer.as_ref(),
        original_command_buffer.as_ref()
    ));
    assert_eq!(open.completion_sequences, [40, 41, 42]);
    assert!(presenter.committed_guest_batches.is_empty());
    assert_eq!(presenter.statistics.submitted_guest_batches, 0);
    assert_eq!(presenter.statistics.coalesced_pipeline_barriers, 3);

    let mut release_label = 0_u32;
    let release_address =
        u64::try_from((&raw mut release_label) as usize).expect("test label address fits u64");
    presenter
        .execute_guest_memory_release(GpuMemoryRelease {
            address: release_address,
            value: GpuMemoryReleaseValue::Immediate32(1),
        })
        .expect("a real release remains in the coalesced batch");
    let open = presenter
        .open_guest_batch
        .as_ref()
        .expect("release does not commit the open batch");
    assert!(open.render_pass.is_some());
    assert!(matches!(
        open.memory_writes.as_slice(),
        [PendingGuestMemoryWrite::Release(GpuMemoryRelease {
            address,
            value: GpuMemoryReleaseValue::Immediate32(1),
        })] if *address == release_address
    ));
    assert_eq!(release_label, 0, "the label remains deferred");
    assert!(presenter.committed_guest_batches.is_empty());
    assert_eq!(presenter.statistics.submitted_guest_batches, 0);
    assert_eq!(presenter.statistics.batch_flushes_for_guest_memory_write, 0);
    assert_eq!(presenter.statistics.render_pass_ends_for_batch_flush, 0);
    assert_eq!(presenter.statistics.render_pass_ends_for_compute, 0);
    assert_eq!(presenter.statistics.render_pass_ends_for_target_change, 0);

    presenter
        .flush_guest_command_batch(false, GuestBatchFlushReason::EventLoop)
        .expect("event loop commits deferred work");
    assert!(presenter.open_guest_batch.is_none());
    assert_eq!(presenter.statistics.submitted_guest_batches, 1);
    assert_eq!(presenter.statistics.batch_flushes_from_event_loop, 1);
    assert_eq!(presenter.statistics.render_pass_ends_for_batch_flush, 1);
    let committed = presenter
        .committed_guest_batches
        .pop_front()
        .expect("event loop retains one committed batch");
    assert_eq!(committed.completion_sequences, [40, 41, 42]);
    assert!(matches!(
        committed.memory_writes.as_slice(),
        [PendingGuestMemoryWrite::Release(GpuMemoryRelease {
            address,
            ..
        })] if *address == release_address
    ));
    committed.command_buffer.wait_until_completed();
    assert_eq!(
        committed.command_buffer.status(),
        metal::MTLCommandBufferStatus::Completed
    );
    apply_pending_guest_memory_writes(&committed.memory_writes)
        .expect("completed release publishes its label");
    assert_eq!(release_label, 1);
}

#[test]
fn queued_global_writer_buffer_is_forwarded_to_later_snapshot() {
    let (_sender, receiver) = gpu_event_channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let queue = presenter
        .command_queue
        .as_ref()
        .expect("Metal queue")
        .clone();
    presenter.ensure_open_guest_batch(&queue);

    let producer = device.new_buffer(128, MTLResourceOptions::StorageModeShared);
    presenter
        .open_guest_batch
        .as_mut()
        .expect("guest batch")
        .global_writebacks
        .push(PendingGlobalWriteback {
            buffer: producer.clone(),
            offset: 32,
            guest_address: 0x12_0000,
            byte_count: 64,
            mapping_identity: None,
            write_watch: None,
            original_bytes: vec![0; 64],
        });
    let binding = GlobalMemoryBinding {
        scalar_address: 4,
        base_address: 0x12_0010,
        instruction_pcs: Default::default(),
        byte_count: 16,
        data: vec![0xaa; 16],
        writable: false,
        write_back_to_guest: false,
    };

    let uploaded = presenter
        .upload_global_buffer(&device, &binding)
        .expect("queued writer is forwarded");

    assert!(std::ptr::eq(uploaded.buffer.as_ref(), producer.as_ref()));
    assert_eq!(uploaded.offset, 48);
    assert!(!uploaded.guest_memory);
}

#[test]
fn pending_label_writes_coalesce_until_a_blocked_wait_commits_them() {
    let (_sender, receiver) = gpu_event_channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
    let queue = presenter
        .command_queue
        .as_ref()
        .expect("Metal queue")
        .clone();
    presenter.open_guest_batch = Some(GuestCommandBatch {
        command_buffer: queue.new_command_buffer().to_owned(),
        work: Vec::new(),
        global_upload_chunks: Vec::new(),
        global_writebacks: Vec::new(),
        guest_write_ranges: Vec::new(),
        memory_writes: Vec::new(),
        render_pass: None,
        completion_sequences: Vec::new(),
    });

    let mut label = 0_u32;
    let address = u64::try_from((&raw mut label) as usize).expect("test label address fits u64");
    for value in [1, 2] {
        presenter
            .execute_guest_memory_release(GpuMemoryRelease {
                address,
                value: GpuMemoryReleaseValue::Immediate32(value),
            })
            .expect("label release coalesces");
    }
    assert_eq!(label, 0, "open-batch label writes remain deferred");
    assert_eq!(
        presenter.pending_guest_memory_value(address, false),
        Some(2)
    );
    assert_eq!(
        presenter
            .open_guest_batch
            .as_ref()
            .expect("label writes retain the open batch")
            .memory_writes
            .len(),
        2
    );
    assert!(presenter.committed_guest_batches.is_empty());
    assert_eq!(presenter.statistics.submitted_guest_batches, 0);

    let satisfied = GpuMemoryWait {
        address,
        reference: 2,
        mask: u64::from(u32::MAX),
        compare_function: 3,
        is_64_bit: false,
    };
    assert_eq!(
        presenter
            .execute_guest_memory_wait(OrderedEventSource::GuestQueue(77), None, satisfied)
            .expect("latest deferred value satisfies the ordered wait"),
        GuestMemoryWaitStatus::Satisfied
    );
    assert!(presenter.open_guest_batch.is_some());
    assert_eq!(presenter.statistics.submitted_guest_batches, 0);

    let blocked = GpuMemoryWait {
        reference: 3,
        ..satisfied
    };
    assert_eq!(
        presenter
            .execute_guest_memory_wait(OrderedEventSource::GuestQueue(77), None, blocked)
            .expect("nonmatching deferred value blocks"),
        GuestMemoryWaitStatus::Blocked
    );
    assert!(presenter.open_guest_batch.is_none());
    assert_eq!(presenter.committed_guest_batches.len(), 1);
    assert_eq!(presenter.statistics.submitted_guest_batches, 1);
    assert_eq!(presenter.statistics.batch_flushes_for_blocked_wait, 1);
    assert_eq!(label, 0, "blocked wait commits before CPU publication");
    assert_eq!(
        presenter.pending_guest_memory_value(address, false),
        Some(2)
    );

    presenter
        .wait_for_guest_command_batches()
        .expect("committed label batch completes");
    assert!(presenter.committed_guest_batches.is_empty());
    assert_eq!(label, 2, "deferred writes publish in exact packet order");
}

#[test]
fn terminal_gpu_failure_still_publishes_release_and_latches_ordered_wait() {
    let (_sender, receiver) = gpu_event_channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
    let mut label = 0_u32;
    let address = u64::try_from((&raw mut label) as usize).expect("test label address fits u64");
    let wait = GpuMemoryWait {
        address,
        reference: 7,
        mask: u64::from(u32::MAX),
        compare_function: 3,
        is_64_bit: false,
    };
    let key = GuestMemoryWaitKey::new(OrderedEventSource::GuestQueue(19), wait);
    assert_eq!(
        presenter.guest_memory_waits.observe_unsatisfied(key),
        GuestMemoryWaitRecovery::None
    );
    let pending = [PendingGuestMemoryWrite::Release(GpuMemoryRelease {
        address,
        value: GpuMemoryReleaseValue::Immediate32(7),
    })];

    let error = presenter
        .publish_terminal_guest_memory_writes(
            &pending,
            Err("Metal render pipeline rejected the ordered draw".to_owned()),
        )
        .expect_err("the original GPU failure must remain visible");

    assert_eq!(
        error, "Metal render pipeline rejected the ordered draw",
        "publishing the completion label must not erase the primary failure"
    );
    assert_eq!(
        label, 7,
        "terminal failure must not strand its following RELEASE_MEM"
    );
    assert_eq!(
        presenter.guest_memory_waits.observe_unsatisfied(key),
        GuestMemoryWaitRecovery::ProducerLatched(7),
        "the ordered producer must wake a suspended WAIT_REG_MEM"
    );
}

#[test]
fn synchronous_failure_drains_later_batches_and_their_completion_labels() {
    let (_sender, receiver) = gpu_event_channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
    let queue = presenter
        .command_queue
        .as_ref()
        .expect("Metal queue")
        .clone();
    let mut completion_label = 0_u32;
    let completion_address =
        u64::try_from((&raw mut completion_label) as usize).expect("test label address fits u64");
    for memory_writes in [
        vec![PendingGuestMemoryWrite::Release(GpuMemoryRelease {
            address: 2,
            value: GpuMemoryReleaseValue::Immediate32(1),
        })],
        vec![PendingGuestMemoryWrite::Release(GpuMemoryRelease {
            address: completion_address,
            value: GpuMemoryReleaseValue::Immediate32(9),
        })],
    ] {
        let command_buffer = queue.new_command_buffer().to_owned();
        command_buffer.commit();
        presenter
            .committed_guest_batches
            .push_back(GuestCommandBatch {
                command_buffer,
                work: Vec::new(),
                global_upload_chunks: Vec::new(),
                global_writebacks: Vec::new(),
                guest_write_ranges: Vec::new(),
                memory_writes,
                render_pass: None,
                completion_sequences: Vec::new(),
            });
    }

    let error = presenter
        .wait_for_guest_command_batches()
        .expect_err("the invalid first release must remain an error");

    assert!(error.contains("ordered guest memory publication"));
    assert!(error.contains("unaligned"));
    assert!(
        presenter.committed_guest_batches.is_empty(),
        "a failed terminal batch must not abandon later committed tickets"
    );
    assert_eq!(
        completion_label, 9,
        "the later batch's ordered completion label must still publish"
    );
}

#[test]
fn timed_out_guest_batch_stays_queued_until_metal_reaches_a_terminal_state() {
    let (_sender, receiver) = gpu_event_channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
    let queue = presenter
        .command_queue
        .as_ref()
        .expect("Metal queue")
        .clone();
    let device = presenter.device.as_ref().expect("Metal device");
    let event = device.new_shared_event();
    let command_buffer = queue.new_command_buffer().to_owned();
    command_buffer.encode_wait_for_event(&event, 1);
    command_buffer.commit();

    let mut completion_label = 0_u32;
    let completion_address =
        u64::try_from((&raw mut completion_label) as usize).expect("test label address fits u64");
    presenter
        .committed_guest_batches
        .push_back(GuestCommandBatch {
            command_buffer,
            work: Vec::new(),
            global_upload_chunks: Vec::new(),
            global_writebacks: Vec::new(),
            guest_write_ranges: Vec::new(),
            memory_writes: vec![PendingGuestMemoryWrite::Release(GpuMemoryRelease {
                address: completion_address,
                value: GpuMemoryReleaseValue::Immediate32(9),
            })],
            render_pass: None,
            completion_sequences: Vec::new(),
        });

    let error = presenter
        .wait_for_guest_command_batches_within(Duration::from_millis(200))
        .expect_err("the unsignaled Metal event must exhaust the bounded wait");
    assert!(error.contains("did not complete"));
    assert_eq!(
        presenter.committed_guest_batches.len(),
        1,
        "a timeout must retain the still-running command buffer"
    );
    assert_eq!(
        completion_label, 0,
        "a timeout must not publish ordered guest completion effects"
    );

    event.set_signaled_value(1);
    presenter
        .wait_for_guest_command_batches()
        .expect("the retained batch retires after Metal completes it");
    assert!(presenter.committed_guest_batches.is_empty());
    assert_eq!(completion_label, 9);
}

#[test]
fn guest_batch_global_uploads_are_aligned_and_write_back_from_their_slice() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let mut batch = GuestCommandBatch {
        command_buffer: queue.new_command_buffer().to_owned(),
        work: Vec::new(),
        global_upload_chunks: Vec::new(),
        global_writebacks: Vec::new(),
        guest_write_ranges: Vec::new(),
        memory_writes: Vec::new(),
        render_pass: None,
        completion_sequences: Vec::new(),
    };
    let (first_buffer, first_offset) = batch
        .upload_global_bytes(&device, &[1, 2, 3])
        .expect("first upload");
    let (second_buffer, second_offset) = batch
        .upload_global_bytes(&device, &[9, 8, 7, 6])
        .expect("second upload");
    assert_eq!(first_buffer.contents(), second_buffer.contents());
    assert_eq!(first_offset, 0);
    assert_eq!(second_offset, GLOBAL_UPLOAD_ALIGNMENT as u64);

    let mut destination = [0_u8; 4];
    let guest_address =
        u64::try_from(destination.as_mut_ptr() as usize).expect("host pointer fits u64");
    write_back_pending_globals(&[PendingGlobalWriteback {
        buffer: second_buffer,
        offset: second_offset,
        guest_address,
        byte_count: destination.len(),
        mapping_identity: None,
        write_watch: None,
        original_bytes: vec![0; destination.len()],
    }])
    .expect("sliced global writeback");
    assert_eq!(destination, [9, 8, 7, 6]);
}

#[test]
fn global_writeback_preserves_cpu_bytes_the_gpu_did_not_change() {
    let device = Device::system_default().expect("Metal device");
    let buffer = device.new_buffer_with_data(
        [1_u8, 9, 3, 8].as_ptr().cast(),
        4,
        MTLResourceOptions::StorageModeShared,
    );
    let mut destination = [7_u8; 4];
    let guest_address =
        u64::try_from(destination.as_mut_ptr() as usize).expect("host pointer fits u64");

    write_back_pending_globals(&[PendingGlobalWriteback {
        buffer,
        offset: 0,
        guest_address,
        byte_count: destination.len(),
        mapping_identity: None,
        write_watch: None,
        original_bytes: vec![1, 2, 3, 4],
    }])
    .expect("changed global bytes write back");

    assert_eq!(destination, [7, 9, 7, 8]);
}

#[test]
fn anonymous_global_writeback_rejects_replaced_guest_mapping() {
    let mut address_space =
        GuestAddressSpace::new(2 * GUEST_PAGE_SIZE).expect("guest address space");
    let address = address_space
        .map_anonymous(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
        )
        .expect("anonymous guest mapping");
    let guest_address = address.get();
    let watch = unsafe { CpuWriteWatch::new(guest_address, GUEST_PAGE_SIZE) }
        .expect("anonymous mapping write watch");

    let device = Device::system_default().expect("Metal device");
    let buffer = device.new_buffer_with_data(
        [5_u8, 6, 7, 8].as_ptr().cast(),
        4,
        MTLResourceOptions::StorageModeShared,
    );
    let pending = PendingGlobalWriteback {
        buffer,
        offset: 0,
        guest_address,
        byte_count: 4,
        mapping_identity: None,
        write_watch: Some(watch),
        original_bytes: vec![0; 4],
    };

    address_space
        .unmap(address, GUEST_PAGE_SIZE)
        .expect("unmap watched guest allocation");
    address_space
        .map_anonymous(
            GuestMapPlacement::Fixed(address),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
        )
        .expect("replacement guest mapping");

    let error = write_back_pending_globals(&[pending])
        .expect_err("writeback must reject the replacement mapping");
    assert!(error.contains("mapping changed before completion"));
    let replacement = unsafe { std::slice::from_raw_parts(guest_address as *const u8, 4) }.to_vec();
    assert_eq!(replacement, [0, 0, 0, 0]);
}

#[test]
fn anonymous_global_writeback_updates_the_pinned_guest_mapping() {
    let mut address_space =
        GuestAddressSpace::new(2 * GUEST_PAGE_SIZE).expect("guest address space");
    let address = address_space
        .map_anonymous(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
        )
        .expect("anonymous guest mapping");
    let guest_address = address.get();
    unsafe {
        std::ptr::copy_nonoverlapping([1_u8, 2, 3, 4].as_ptr(), guest_address as *mut u8, 4);
    }
    let watch = unsafe { CpuWriteWatch::new(guest_address, GUEST_PAGE_SIZE) }
        .expect("anonymous mapping write watch");

    let device = Device::system_default().expect("Metal device");
    let buffer = device.new_buffer_with_data(
        [1_u8, 9, 3, 8].as_ptr().cast(),
        4,
        MTLResourceOptions::StorageModeShared,
    );
    write_back_pending_globals(&[PendingGlobalWriteback {
        buffer,
        offset: 0,
        guest_address,
        byte_count: 4,
        mapping_identity: None,
        write_watch: Some(watch),
        original_bytes: vec![1, 2, 3, 4],
    }])
    .expect("pinned anonymous writeback");

    let destination = unsafe { std::slice::from_raw_parts(guest_address as *const u8, 4) }.to_vec();
    assert_eq!(destination, [1, 9, 3, 8]);
}

#[test]
fn ordered_guest_label_writes_publish_only_the_declared_values() {
    let mut release_value = 0_u64;
    let mut words = [0_u32; 3];
    let mut fill_words = [0_u32; 3];
    let release = PendingGuestMemoryWrite::Release(GpuMemoryRelease {
        address: std::ptr::addr_of_mut!(release_value) as u64,
        value: GpuMemoryReleaseValue::Immediate64(0x1122_3344_5566_7788),
    });
    let write = PendingGuestMemoryWrite::Write(GpuMemoryWrite {
        address: words.as_mut_ptr() as u64,
        values: vec![3, 5, 8],
        increment_address: true,
    });
    let fill = PendingGuestMemoryWrite::Fill {
        address: fill_words.as_mut_ptr() as u64,
        value: 0xaabb_ccdd,
        dword_count: 3,
    };
    assert_eq!(
        release.value_at(std::ptr::addr_of!(release_value) as u64, true),
        Some(0x1122_3344_5566_7788)
    );
    assert_eq!(write.value_at(words.as_ptr() as u64 + 8, false), Some(8));
    assert_eq!(
        fill.value_at(fill_words.as_ptr() as u64 + 4, false),
        Some(0xaabb_ccdd)
    );
    assert_eq!(
        fill.value_at(fill_words.as_ptr() as u64, true),
        Some(0xaabb_ccdd_aabb_ccdd)
    );
    apply_pending_guest_memory_writes(&[release, write, fill]).expect("ordered guest writes");
    assert_eq!(release_value, 0x1122_3344_5566_7788);
    assert_eq!(words, [3, 5, 8]);
    assert_eq!(fill_words, [0xaabb_ccdd; 3]);
}

#[test]
fn queue_local_compile_gate_runs_compute_without_overtaking_graphics_or_flip() {
    let (sender, receiver) = gpu_event_channel();
    let mut ordered = OrderedGpuEventReceiver::new(receiver);
    let key = GraphicsPipelineKey {
        export_shader_address: 0x1000,
        pixel_shader_address: 0x2000,
        export_program_identity: super::ProgramIdentity(std::sync::Arc::new(Program::new(
            0x1000,
            Vec::new(),
        ))),
        pixel_program_identity: super::ProgramIdentity(std::sync::Arc::new(Program::new(
            0x2000,
            Vec::new(),
        ))),
        export_resources: GraphicsShaderResourceKey {
            initial_scalar_count: 0,
            global_bindings: Vec::new(),
            image_bindings: Vec::new(),
            vertex_inputs: Vec::new(),
        },
        pixel_resources: GraphicsShaderResourceKey {
            initial_scalar_count: 0,
            global_bindings: Vec::new(),
            image_bindings: Vec::new(),
            vertex_inputs: Vec::new(),
        },
        export_allocated_vector_register_count: 256,
        pixel_allocated_vector_register_count: 256,
        vertex_output_control: 0,
        pixel_input_controls: std::array::from_fn(|index| {
            u32::try_from(index).expect("PS input index fits u32")
        }),
        targets: Vec::new(),
        depth: None,
        sample_count: 1,
        render_target_array_length: 1,
        primitive_topology_class: MTLPrimitiveTopologyClass::Triangle,
    };
    let mut draw = indexed_draw(0, 0, 0);
    draw.sequence = 17;
    let mut later_draw = indexed_draw(0, 0, 0);
    later_draw.sequence = 23;
    sender
        .send(GpuEvent::QueueSubmission {
            queue_id: 0,
            events: vec![
                GpuEvent::IndexedDraw(Box::new(draw)),
                GpuEvent::PipelineBarrier(remu_gpu::GpuPipelineBarrier {
                    wait_for: remu_gpu::GpuPipelineStage::PixelShader,
                }),
                GpuEvent::MemoryAcquire(GpuMemoryAcquire {
                    engine: 0,
                    cb_db_control: 0,
                    base_address: 0x2000,
                    size_bytes: 0x1000,
                    poll_interval: 0,
                    gcr_control: 0,
                }),
                GpuEvent::IndexedDraw(Box::new(later_draw)),
                GpuEvent::SubmissionBoundary(1),
            ],
        })
        .expect("queue graphics submission");
    sender
        .send(GpuEvent::QueueSubmission {
            queue_id: 33,
            events: vec![
                GpuEvent::MemoryRelease(GpuMemoryRelease {
                    address: 0x3000,
                    value: GpuMemoryReleaseValue::Immediate32(1),
                }),
                GpuEvent::SubmissionBoundary(2),
            ],
        })
        .expect("queue compute submission");
    sender
        .send(GpuEvent::Flip {
            address: 0x7000_0000,
            buffer_index: 1,
            flip_argument: 9,
            width: 1_280,
            height: 720,
            pitch: 1_280,
            pixel_format: 0x8000_0000_2200_0000,
            tiling_mode: 0,
        })
        .expect("queue later flip");

    let (graphics_source, blocked_draw) = match ordered.try_recv(|_| false) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::IndexedDraw(draw),
        } => {
            assert_eq!(draw.sequence, 17);
            (source, GpuEvent::IndexedDraw(draw))
        }
        other => panic!("expected first graphics draw, got {other:?}"),
    };
    ordered.block_event(
        graphics_source,
        blocked_draw,
        QueueBlockReason::GraphicsPipeline(key),
    );

    let compute_source = match ordered.try_recv(|_| false) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::MemoryRelease(release),
        } => {
            assert_eq!(release.address, 0x3000);
            source
        }
        other => panic!("expected independent compute release, got {other:?}"),
    };
    ordered.finish_event(compute_source);
    let compute_source = match ordered.try_recv(|_| false) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::SubmissionBoundary(2),
        } => source,
        other => panic!("expected compute boundary, got {other:?}"),
    };
    ordered.finish_event(compute_source);

    assert!(matches!(
        ordered.try_recv(|_| false),
        OrderedEventPoll::Blocked
    ));
    let prefetched = ordered.prefetch_draws(16, 16);
    assert_eq!(
        prefetched
            .iter()
            .map(|draw| draw.sequence)
            .collect::<Vec<_>>(),
        [17, 23]
    );

    match ordered.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::IndexedDraw(draw),
        } => {
            assert_eq!(draw.sequence, 17);
            ordered.finish_event(source);
        }
        other => panic!("expected the blocked draw exactly once, got {other:?}"),
    }
    let source = match ordered.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event:
                GpuEvent::PipelineBarrier(remu_gpu::GpuPipelineBarrier {
                    wait_for: remu_gpu::GpuPipelineStage::PixelShader,
                }),
        } => source,
        other => panic!("expected graphics barrier, got {other:?}"),
    };
    ordered.finish_event(source);
    let source = match ordered.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event:
                GpuEvent::MemoryAcquire(GpuMemoryAcquire {
                    base_address: 0x2000,
                    size_bytes: 0x1000,
                    ..
                }),
        } => source,
        other => panic!("expected graphics acquire, got {other:?}"),
    };
    ordered.finish_event(source);
    match ordered.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::IndexedDraw(draw),
        } => {
            assert_eq!(draw.sequence, 23);
            ordered.finish_event(source);
        }
        other => panic!("expected the prefetched draw in exact order, got {other:?}"),
    }
    let source = match ordered.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::SubmissionBoundary(1),
        } => source,
        other => panic!("expected graphics boundary, got {other:?}"),
    };
    ordered.finish_event(source);
    assert!(matches!(
        ordered.try_recv(|_| true),
        OrderedEventPoll::Event {
            event: GpuEvent::Flip {
                address: 0x7000_0000,
                ..
            },
            ..
        }
    ));
    assert!(matches!(
        ordered.try_recv(|_| true),
        OrderedEventPoll::Empty
    ));
}

#[test]
fn ordered_ingress_stops_at_the_decoded_event_backpressure_limit() {
    let (sender, receiver) = std::sync::mpsc::channel();
    for _ in
        0..(remu_gpu::GPU_PENDING_EVENT_CAPACITY / remu_gpu::GPU_EVENT_SUBMISSION_CHUNK_SIZE + 4)
    {
        sender
            .send(GpuEvent::QueueSubmission {
                queue_id: 0,
                events: vec![GpuEvent::Shutdown; remu_gpu::GPU_EVENT_SUBMISSION_CHUNK_SIZE],
            })
            .expect("queue decoded event chunk");
    }
    let mut ordered = OrderedGpuEventReceiver::new(receiver);

    ordered.ingest(usize::MAX);

    assert_eq!(
        ordered.pending_event_count(),
        remu_gpu::GPU_PENDING_EVENT_CAPACITY
    );
}

#[test]
fn blocked_wait_ingests_past_backpressure_only_until_a_runnable_queue_is_found() {
    let (sender, receiver) = std::sync::mpsc::channel();
    let wait = GpuMemoryWait {
        address: 0x4000,
        reference: 1,
        mask: u64::from(u32::MAX),
        compare_function: 3,
        is_64_bit: false,
    };
    sender
        .send(GpuEvent::QueueSubmission {
            queue_id: 0,
            events: vec![GpuEvent::MemoryWait(wait)],
        })
        .expect("queue graphics wait");
    for _ in 0..(remu_gpu::GPU_PENDING_EVENT_CAPACITY / remu_gpu::GPU_EVENT_SUBMISSION_CHUNK_SIZE) {
        sender
            .send(GpuEvent::QueueSubmission {
                queue_id: 0,
                events: vec![
                    GpuEvent::PipelineBarrier(remu_gpu::GpuPipelineBarrier {
                        wait_for: remu_gpu::GpuPipelineStage::ComputeShader,
                    });
                    remu_gpu::GPU_EVENT_SUBMISSION_CHUNK_SIZE
                ],
            })
            .expect("queue blocked graphics tail");
    }
    sender
        .send(GpuEvent::QueueSubmission {
            queue_id: 33,
            events: vec![GpuEvent::MemoryRelease(GpuMemoryRelease {
                address: wait.address,
                value: GpuMemoryReleaseValue::Immediate32(1),
            })],
        })
        .expect("queue independent producer");
    sender
        .send(GpuEvent::QueueSubmission {
            queue_id: 34,
            events: vec![GpuEvent::SubmissionBoundary(3)],
        })
        .expect("queue work after the producer");

    let mut ordered = OrderedGpuEventReceiver::new(receiver);
    let source = match ordered.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::MemoryWait(actual),
        } => {
            assert_eq!(actual, wait);
            source
        }
        other => panic!("expected graphics wait, got {other:?}"),
    };
    ordered.block_event(
        source,
        GpuEvent::MemoryWait(wait),
        QueueBlockReason::GuestMemoryWait(wait),
    );
    assert!(
        ordered.pending_event_count() <= remu_gpu::GPU_PENDING_EVENT_CAPACITY,
        "ordinary ingress must stop at its decoded-event cap"
    );

    let source = match ordered.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::MemoryRelease(release),
        } => {
            assert_eq!(release.address, wait.address);
            source
        }
        other => panic!("dependency ingress did not discover the producer queue: {other:?}"),
    };
    assert_eq!(source, OrderedEventSource::GuestQueue(33));
    assert!(
        ordered.pending_event_count() > remu_gpu::GPU_PENDING_EVENT_CAPACITY,
        "blocked queue tails must not hide the independent producer"
    );
    ordered.finish_event(source);

    assert!(
        !ordered.submissions.contains_key(&34),
        "ingress must stop again as soon as one guest queue is runnable"
    );
    assert!(matches!(
        ordered.blocked_queues.get(&0),
        Some(QueueBlockReason::GuestMemoryWait(actual)) if *actual == wait
    ));
}

#[test]
fn blocked_wait_dependency_ingress_has_a_finite_pending_event_bound() {
    let (sender, receiver) = std::sync::mpsc::channel();
    let wait = GpuMemoryWait {
        address: 0x5000,
        reference: 1,
        mask: u64::from(u32::MAX),
        compare_function: 3,
        is_64_bit: false,
    };
    sender
        .send(GpuEvent::QueueSubmission {
            queue_id: 0,
            events: vec![GpuEvent::MemoryWait(wait)],
        })
        .expect("queue graphics wait");
    for _ in 0..64 {
        sender
            .send(GpuEvent::QueueSubmission {
                queue_id: 0,
                events: vec![
                    GpuEvent::PipelineBarrier(remu_gpu::GpuPipelineBarrier {
                        wait_for: remu_gpu::GpuPipelineStage::ComputeShader,
                    });
                    remu_gpu::GPU_EVENT_SUBMISSION_CHUNK_SIZE
                ],
            })
            .expect("queue repeated blocked graphics tail");
    }

    let mut ordered = OrderedGpuEventReceiver::new(receiver);
    let source = match ordered.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::MemoryWait(actual),
        } => {
            assert_eq!(actual, wait);
            source
        }
        other => panic!("expected graphics wait, got {other:?}"),
    };
    ordered.block_event(
        source,
        GpuEvent::MemoryWait(wait),
        QueueBlockReason::GuestMemoryWait(wait),
    );
    ordered.ingest(usize::MAX);

    let dependency_pending_capacity = remu_gpu::GPU_PENDING_EVENT_CAPACITY
        + remu_gpu::GPU_EVENT_QUEUE_CAPACITY * remu_gpu::GPU_EVENT_SUBMISSION_CHUNK_SIZE * 2;
    assert!(
        ordered.pending_event_count() <= dependency_pending_capacity,
        "blocked-only ingress must retain a finite decoded-event bound"
    );
    assert!(
        !ordered.has_runnable_events(),
        "blocked queue tails must never become runnable ahead of their wait"
    );
    assert!(matches!(
        ordered.try_recv(|_| true),
        OrderedEventPoll::Blocked
    ));
}

#[test]
fn blocked_wait_diagnostics_report_both_edges_of_a_cross_queue_cycle() {
    let (sender, receiver) = std::sync::mpsc::channel();
    let graphics_wait = GpuMemoryWait {
        address: 0x4e0,
        reference: 1,
        mask: u64::from(u32::MAX),
        compare_function: 3,
        is_64_bit: false,
    };
    let compute_wait = GpuMemoryWait {
        address: 0x4c0,
        ..graphics_wait
    };
    for (queue_id, wait, producer_address) in [
        (0, graphics_wait, compute_wait.address),
        (33, compute_wait, graphics_wait.address),
    ] {
        sender
            .send(GpuEvent::QueueSubmission {
                queue_id,
                events: vec![
                    GpuEvent::MemoryWait(wait),
                    GpuEvent::MemoryRelease(GpuMemoryRelease {
                        address: producer_address,
                        value: GpuMemoryReleaseValue::Immediate32(1),
                    }),
                ],
            })
            .expect("queue cyclic wait and producer");
    }

    let mut ordered = OrderedGpuEventReceiver::new(receiver);
    for expected in [graphics_wait, compute_wait] {
        let (source, event) = match ordered.try_recv(|_| true) {
            OrderedEventPoll::Event {
                source,
                event: GpuEvent::MemoryWait(actual),
            } => {
                assert_eq!(actual, expected);
                (source, GpuEvent::MemoryWait(actual))
            }
            other => panic!("expected cyclic wait, got {other:?}"),
        };
        ordered.block_event(source, event, QueueBlockReason::GuestMemoryWait(expected));
    }

    let dependencies = ordered.blocked_wait_dependency_diagnostics();
    assert!(
        dependencies.contains("q0:wait@0x4e0")
            && dependencies.contains("q33/s0/e1:release@0x4e0=0x1")
    );
    assert!(
        dependencies.contains("q33:wait@0x4c0")
            && dependencies.contains("q0/s0/e1:release@0x4c0=0x1")
    );
    assert!(matches!(
        ordered.try_recv(|_| true),
        OrderedEventPoll::Blocked
    ));
}

#[test]
fn ordered_receiver_reports_only_dependency_runnable_work() {
    let (sender, receiver) = std::sync::mpsc::channel();
    let mut ordered = OrderedGpuEventReceiver::new(receiver);
    let wait = GpuMemoryWait {
        address: 0x4000,
        reference: 1,
        mask: u64::from(u32::MAX),
        compare_function: 3,
        is_64_bit: false,
    };
    assert!(!ordered.has_runnable_events());
    sender
        .send(GpuEvent::QueueSubmission {
            queue_id: 0,
            events: vec![GpuEvent::MemoryWait(wait)],
        })
        .expect("queue memory wait");
    ordered.ingest(1);
    assert!(ordered.has_runnable_events());

    let source = match ordered.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::MemoryWait(actual),
        } => {
            assert_eq!(actual, wait);
            source
        }
        other => panic!("expected memory wait, got {other:?}"),
    };
    ordered.block_event(
        source,
        GpuEvent::MemoryWait(wait),
        QueueBlockReason::GuestMemoryWait(wait),
    );
    assert!(!ordered.has_runnable_events());

    ordered.retry_guest_memory_waits();
    assert!(ordered.has_runnable_events());
    let source = match ordered.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::MemoryWait(actual),
        } => {
            assert_eq!(actual, wait);
            source
        }
        other => panic!("expected retried memory wait, got {other:?}"),
    };
    ordered.finish_event(source);
    assert!(!ordered.has_runnable_events());
}

#[test]
fn blocked_guest_wait_does_not_stall_independent_video_out_flip() {
    let (sender, receiver) = std::sync::mpsc::channel();
    let mut ordered = OrderedGpuEventReceiver::new(receiver);
    let wait = GpuMemoryWait {
        address: 0x4000,
        reference: 1,
        mask: u64::from(u32::MAX),
        compare_function: 3,
        is_64_bit: false,
    };
    sender
        .send(GpuEvent::QueueSubmission {
            queue_id: 0,
            events: vec![GpuEvent::MemoryWait(wait)],
        })
        .expect("queue blocked guest wait");
    let source = match ordered.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::MemoryWait(actual),
        } => {
            assert_eq!(actual, wait);
            source
        }
        other => panic!("expected guest wait, got {other:?}"),
    };
    ordered.block_event(
        source,
        GpuEvent::MemoryWait(wait),
        QueueBlockReason::GuestMemoryWait(wait),
    );

    sender
        .send(GpuEvent::Flip {
            address: 0x7000_0000,
            buffer_index: 2,
            flip_argument: 11,
            width: 1_920,
            height: 1_080,
            pitch: 1_920,
            pixel_format: 0x8000_0000_2200_0000,
            tiling_mode: 0,
        })
        .expect("queue independent VideoOut flip");

    assert!(matches!(
        ordered.try_recv(|_| true),
        OrderedEventPoll::Event {
            source: OrderedEventSource::Host { .. },
            event: GpuEvent::Flip {
                address: 0x7000_0000,
                flip_argument: 11,
                ..
            },
        }
    ));
    assert!(matches!(
        ordered.try_recv(|_| true),
        OrderedEventPoll::Blocked
    ));
}

#[test]
fn produced_label_retries_only_overlapping_blocked_waits() {
    let (sender, receiver) = std::sync::mpsc::channel();
    let mut ordered = OrderedGpuEventReceiver::new(receiver);
    let first = GpuMemoryWait {
        address: 0x4000,
        reference: 1,
        mask: u64::from(u32::MAX),
        compare_function: 3,
        is_64_bit: false,
    };
    let second = GpuMemoryWait {
        address: 0x8000,
        ..first
    };
    for (queue_id, wait) in [(0, first), (33, second)] {
        sender
            .send(GpuEvent::QueueSubmission {
                queue_id,
                events: vec![GpuEvent::MemoryWait(wait)],
            })
            .expect("queue guest wait");
    }
    for expected in [first, second] {
        let source = match ordered.try_recv(|_| true) {
            OrderedEventPoll::Event {
                source,
                event: GpuEvent::MemoryWait(actual),
            } => {
                assert_eq!(actual, expected);
                source
            }
            other => panic!("expected guest wait, got {other:?}"),
        };
        ordered.block_event(
            source,
            GpuEvent::MemoryWait(expected),
            QueueBlockReason::GuestMemoryWait(expected),
        );
    }

    ordered.retry_guest_memory_waits_overlapping(first.address, 4);
    let source = match ordered.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::MemoryWait(actual),
        } => {
            assert_eq!(actual, first);
            source
        }
        other => panic!("expected only overlapping wait to retry, got {other:?}"),
    };
    ordered.finish_event(source);
    assert!(matches!(
        ordered.try_recv(|_| true),
        OrderedEventPoll::Blocked
    ));
}

#[test]
fn video_out_flip_retains_ingress_order_against_runnable_guest_work() {
    let (sender, receiver) = std::sync::mpsc::channel();
    let mut ordered = OrderedGpuEventReceiver::new(receiver);
    sender
        .send(GpuEvent::QueueSubmission {
            queue_id: 0,
            events: vec![GpuEvent::MemoryWrite(GpuMemoryWrite {
                address: 0x5000,
                values: vec![9],
                increment_address: true,
            })],
        })
        .expect("queue runnable guest write");
    sender
        .send(GpuEvent::Flip {
            address: 0x7100_0000,
            buffer_index: 3,
            flip_argument: 12,
            width: 1_920,
            height: 1_080,
            pitch: 1_920,
            pixel_format: 0x8000_0000_2200_0000,
            tiling_mode: 0,
        })
        .expect("queue later VideoOut flip");

    let source = match ordered.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::MemoryWrite(write),
        } => {
            assert_eq!(write.address, 0x5000);
            source
        }
        other => panic!("expected older runnable guest work, got {other:?}"),
    };
    ordered.finish_event(source);
    assert!(matches!(
        ordered.try_recv(|_| true),
        OrderedEventPoll::Event {
            source: OrderedEventSource::Host { .. },
            event: GpuEvent::Flip {
                address: 0x7100_0000,
                flip_argument: 12,
                ..
            },
        }
    ));
}

#[test]
fn suspended_memory_wait_keeps_same_queue_tail_behind_it() {
    let (sender, receiver) = gpu_event_channel();
    let mut ordered = OrderedGpuEventReceiver::new(receiver);
    let wait = GpuMemoryWait {
        address: 0x4000,
        reference: 1,
        mask: u64::from(u32::MAX),
        compare_function: 3,
        is_64_bit: false,
    };
    sender
        .send(GpuEvent::QueueSubmission {
            queue_id: 0,
            events: vec![
                GpuEvent::MemoryWait(wait),
                GpuEvent::MemoryWrite(GpuMemoryWrite {
                    address: 0x5000,
                    values: vec![9],
                    increment_address: true,
                }),
                GpuEvent::SubmissionBoundary(1),
            ],
        })
        .expect("queue waiting graphics submission");
    sender
        .send(GpuEvent::QueueSubmission {
            queue_id: 33,
            events: vec![
                GpuEvent::MemoryRelease(GpuMemoryRelease {
                    address: wait.address,
                    value: GpuMemoryReleaseValue::Immediate32(1),
                }),
                GpuEvent::SubmissionBoundary(2),
            ],
        })
        .expect("queue satisfying compute submission");

    let source = match ordered.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::MemoryWait(actual),
        } => {
            assert_eq!(actual, wait);
            source
        }
        other => panic!("expected graphics wait, got {other:?}"),
    };
    ordered.block_event(
        source,
        GpuEvent::MemoryWait(wait),
        QueueBlockReason::GuestMemoryWait(wait),
    );

    let source = match ordered.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::MemoryRelease(release),
        } => {
            assert_eq!(release.address, wait.address);
            source
        }
        other => panic!("expected compute release, got {other:?}"),
    };
    ordered.finish_event(source);
    ordered.retry_guest_memory_waits();

    let source = match ordered.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::SubmissionBoundary(2),
        } => source,
        other => panic!("expected compute boundary, got {other:?}"),
    };
    ordered.finish_event(source);
    let source = match ordered.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::MemoryWait(actual),
        } => {
            assert_eq!(actual, wait);
            source
        }
        other => panic!("expected retried graphics wait, got {other:?}"),
    };
    ordered.finish_event(source);

    assert!(matches!(
        ordered.try_recv(|_| true),
        OrderedEventPoll::Event {
            event: GpuEvent::MemoryWrite(GpuMemoryWrite {
                address: 0x5000,
                ..
            }),
            ..
        }
    ));
}

#[test]
fn guest_wait_latches_a_real_producer_before_guest_memory_can_be_recycled() {
    let mut tracker = GuestMemoryWaitTracker::default();
    let wait = GpuMemoryWait {
        address: 0x7000_1000,
        reference: 1,
        mask: u64::from(u32::MAX),
        compare_function: 3,
        is_64_bit: false,
    };
    let key = GuestMemoryWaitKey::new(OrderedEventSource::GuestQueue(32), wait);
    assert_eq!(
        tracker.observe_unsatisfied(key),
        GuestMemoryWaitRecovery::None
    );

    tracker.record_produced(wait.address, false, 1);
    assert_eq!(
        tracker.observe_unsatisfied(key),
        GuestMemoryWaitRecovery::ProducerLatched(1)
    );
    assert!(!tracker.active.contains_key(&key));
}

#[test]
fn guest_wait_never_reuses_a_producer_from_before_its_registration() {
    let wait = GpuMemoryWait {
        address: 0x7000_2000,
        reference: 1,
        mask: u64::from(u32::MAX),
        compare_function: 3,
        is_64_bit: false,
    };
    let key = GuestMemoryWaitKey::new(OrderedEventSource::GuestQueue(0), wait);
    let mut tracker = GuestMemoryWaitTracker::default();
    tracker.record_produced(wait.address, false, 1);

    assert_eq!(
        tracker.observe_unsatisfied(key),
        GuestMemoryWaitRecovery::None
    );
    // Guest allocators recycle label addresses. Wall-clock age cannot turn a producer from an
    // earlier allocation generation into satisfaction for this wait.
    assert_eq!(
        tracker.observe_unsatisfied(key),
        GuestMemoryWaitRecovery::None
    );

    tracker.record_produced(wait.address, false, 1);
    assert_eq!(
        tracker.observe_unsatisfied(key),
        GuestMemoryWaitRecovery::ProducerLatched(1)
    );
}

#[test]
fn guest_wait_does_not_latch_missing_or_nonmatching_producers() {
    let wait = GpuMemoryWait {
        address: 0x7000_2000,
        reference: 1,
        mask: u64::from(u32::MAX),
        compare_function: 3,
        is_64_bit: false,
    };
    let key = GuestMemoryWaitKey::new(OrderedEventSource::GuestQueue(0), wait);
    let mut missing = GuestMemoryWaitTracker::default();
    assert_eq!(
        missing.observe_unsatisfied(key),
        GuestMemoryWaitRecovery::None
    );
    assert_eq!(
        missing.observe_unsatisfied(key),
        GuestMemoryWaitRecovery::None
    );

    let mut wrong = GuestMemoryWaitTracker::default();
    assert_eq!(
        wrong.observe_unsatisfied(key),
        GuestMemoryWaitRecovery::None
    );
    wrong.record_produced(wait.address, false, 2);
    assert_eq!(
        wrong.observe_unsatisfied(key),
        GuestMemoryWaitRecovery::None
    );
}
