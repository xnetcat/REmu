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
    CpuWriteWatch, GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
    GuestProtection,
};
use remu_shader::{GlobalMemoryBinding, Program};
use std::time::Duration;

use super::{
    GLOBAL_UPLOAD_ALIGNMENT, GraphicsDiagnosticPipelineState, GraphicsPipelineKey,
    GraphicsShaderResourceKey, GuestBatchFlushReason, GuestCommandBatch, GuestMemoryWaitKey,
    GuestMemoryWaitRecovery, GuestMemoryWaitStatus, GuestMemoryWaitTracker,
    GuestMemoryWriteOperation, GuestRenderPassKey, GuestTextureDescriptor, MetalPresenter,
    OpenGuestRenderPass, OrderedEventPoll, OrderedEventSource, OrderedGpuEventReceiver,
    PendingGlobalWriteback, PendingGuestMemoryWrite, QueueBlockReason,
    apply_pending_guest_memory_writes, indexed_draw, pin_guest_memory_write, read_texture_linear,
    write_back_pending_globals,
};

fn mapped_test_label() -> (GuestAddressSpace, u64) {
    let mut address_space =
        GuestAddressSpace::new(2 * GUEST_PAGE_SIZE).expect("guest label address space");
    let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).expect("guest label backing");
    let address = address_space
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("guest label mapping");
    (address_space, address.get())
}

fn read_test_label(address: u64) -> u32 {
    // SAFETY: every caller retains the `GuestAddressSpace` returned by `mapped_test_label`.
    unsafe { std::ptr::read_volatile(address as *const u32) }
}

#[test]
fn native_metal_draw_arguments_match_the_gen5_indirect_layouts() {
    use metal::{MTLDrawIndexedPrimitivesIndirectArguments, MTLDrawPrimitivesIndirectArguments};

    assert_eq!(std::mem::size_of::<MTLDrawPrimitivesIndirectArguments>(), 16);
    assert_eq!(
        std::mem::offset_of!(MTLDrawPrimitivesIndirectArguments, vertexCount),
        0
    );
    assert_eq!(
        std::mem::offset_of!(MTLDrawPrimitivesIndirectArguments, instanceCount),
        4
    );
    assert_eq!(
        std::mem::offset_of!(MTLDrawPrimitivesIndirectArguments, vertexStart),
        8
    );
    assert_eq!(
        std::mem::offset_of!(MTLDrawPrimitivesIndirectArguments, baseInstance),
        12
    );

    assert_eq!(
        std::mem::size_of::<MTLDrawIndexedPrimitivesIndirectArguments>(),
        20
    );
    assert_eq!(
        std::mem::offset_of!(MTLDrawIndexedPrimitivesIndirectArguments, indexCount),
        0
    );
    assert_eq!(
        std::mem::offset_of!(MTLDrawIndexedPrimitivesIndirectArguments, instanceCount),
        4
    );
    assert_eq!(
        std::mem::offset_of!(MTLDrawIndexedPrimitivesIndirectArguments, indexStart),
        8
    );
    assert_eq!(
        std::mem::offset_of!(MTLDrawIndexedPrimitivesIndirectArguments, baseVertex),
        12
    );
    assert_eq!(
        std::mem::offset_of!(MTLDrawIndexedPrimitivesIndirectArguments, baseInstance),
        16
    );
}

#[test]
fn busy_event_loop_retains_batches_but_idle_callbacks_flush() {
    assert!(!super::super::submission::event_loop_batch_requires_flush(
        true
    ));
    assert!(super::super::submission::event_loop_batch_requires_flush(
        false
    ));
}

#[test]
fn retained_busy_batch_has_a_finite_submission_completion_bound() {
    let capacity = super::super::GUEST_BATCH_COMPLETION_CAPACITY;
    assert!(!super::super::submission::guest_batch_completion_capacity_reached(
        capacity - 1
    ));
    assert!(
        super::super::submission::guest_batch_completion_capacity_reached(capacity)
    );
    assert!(
        super::super::submission::guest_batch_completion_capacity_reached(capacity + 1)
    );
}

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

    let (_release_space, release_address) = mapped_test_label();
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
        [PendingGuestMemoryWrite {
            operation: GuestMemoryWriteOperation::Release(GpuMemoryRelease {
                address,
                value: GpuMemoryReleaseValue::Immediate32(1),
            }),
            ..
        }] if *address == release_address
    ));
    assert_eq!(read_test_label(release_address), 0, "the label remains deferred");
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
        [PendingGuestMemoryWrite {
            operation: GuestMemoryWriteOperation::Release(GpuMemoryRelease { address, .. }),
            ..
        }] if *address == release_address
    ));
    committed.command_buffer.wait_until_completed();
    assert_eq!(
        committed.command_buffer.status(),
        metal::MTLCommandBufferStatus::Completed
    );
    apply_pending_guest_memory_writes(&committed.memory_writes)
        .expect("completed release publishes its label");
    assert_eq!(read_test_label(release_address), 1);
}

#[test]
fn synchronous_wait_finishes_the_active_render_encoder_before_readback() {
    let (_sender, receiver) = gpu_event_channel();
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
    attachment.set_clear_color(MTLClearColor::new(1.0, 0.0, 0.0, 1.0));

    let command_buffer = queue.new_command_buffer().to_owned();
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
        completion_sequences: Vec::new(),
    });

    presenter
        .wait_for_guest_command_batches()
        .expect("synchronous readback wait flushes and retires the active render pass");
    assert!(presenter.open_guest_batch.is_none());
    assert!(presenter.committed_guest_batches.is_empty());

    let bytes = read_texture_linear(
        &device,
        &queue,
        &texture,
        GuestTextureDescriptor {
            address: 0,
            width: 4,
            height: 4,
            pitch: 4,
            tile_mode: 0,
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            bytes_per_pixel: 4,
        },
    )
    .expect("completed render target is readable");
    assert!(
        bytes
            .chunks_exact(4)
            .all(|pixel| pixel == [u8::MAX, 0, 0, u8::MAX]),
        "target readback must observe the selected draw's completed render pass"
    );
}

#[test]
fn command_processor_barrier_commits_prior_work_without_a_cpu_wait() {
    let (sender, receiver) = gpu_event_channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
    let queue = presenter
        .command_queue
        .as_ref()
        .expect("Metal queue")
        .clone();
    let command_buffer = queue.new_command_buffer().to_owned();
    let original_command_buffer = command_buffer.clone();
    presenter.open_guest_batch = Some(GuestCommandBatch {
        command_buffer,
        work: Vec::new(),
        global_upload_chunks: Vec::new(),
        global_writebacks: Vec::new(),
        guest_write_ranges: Vec::new(),
        memory_writes: Vec::new(),
        render_pass: None,
        completion_sequences: vec![40],
    });
    sender
        .send(GpuEvent::QueueSubmission {
            queue_id: 7,
            events: vec![
                GpuEvent::CommandProcessorBarrier,
                GpuEvent::SubmissionBoundary(41),
            ],
        })
        .expect("queue command-processor barrier");

    assert!(matches!(
        presenter.receive_events(),
        super::PresenterFlow::Continue
    ));
    assert!(presenter.open_guest_batch.is_none());
    assert_eq!(presenter.statistics.submitted_guest_batches, 1);
    assert_eq!(
        presenter
            .statistics
            .batch_flushes_for_command_processor_barrier,
        1
    );
    let committed = presenter
        .committed_guest_batches
        .pop_front()
        .expect("barrier retains the nonblocking committed batch");
    assert!(std::ptr::eq(
        committed.command_buffer.as_ref(),
        original_command_buffer.as_ref()
    ));
    assert_eq!(committed.completion_sequences, [40, 41]);
    committed.command_buffer.wait_until_completed();
    assert_eq!(
        committed.command_buffer.status(),
        metal::MTLCommandBufferStatus::Completed
    );
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

    let indirect = presenter
        .upload_indirect_dispatch_arguments(&device, 0x12_0010)
        .expect("indirect arguments forward the queued GPU producer");
    assert!(std::ptr::eq(indirect.buffer.as_ref(), producer.as_ref()));
    assert_eq!(indirect.offset, 48);
    assert!(!indirect.guest_memory);

    let indexed_draw = presenter
        .upload_ordered_indirect_arguments(&device, 0x12_0010, 5 * std::mem::size_of::<u32>())
        .expect("indexed draw arguments forward the queued GPU producer");
    assert!(std::ptr::eq(
        indexed_draw.buffer.as_ref(),
        producer.as_ref()
    ));
    assert_eq!(indexed_draw.offset, 48);
    assert!(!indexed_draw.guest_memory);
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

    let (_label_space, address) = mapped_test_label();
    for value in [1, 2] {
        presenter
            .execute_guest_memory_release(GpuMemoryRelease {
                address,
                value: GpuMemoryReleaseValue::Immediate32(value),
            })
            .expect("label release coalesces");
    }
    assert_eq!(
        read_test_label(address),
        0,
        "open-batch label writes remain deferred"
    );
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
    assert_eq!(
        read_test_label(address),
        0,
        "blocked wait commits before CPU publication"
    );
    assert_eq!(
        presenter.pending_guest_memory_value(address, false),
        Some(2)
    );

    presenter
        .wait_for_guest_command_batches()
        .expect("committed label batch completes");
    assert!(presenter.committed_guest_batches.is_empty());
    assert_eq!(
        read_test_label(address),
        2,
        "deferred writes publish in exact packet order"
    );
}

#[test]
fn terminal_gpu_failure_still_publishes_release_and_latches_ordered_wait() {
    let (_sender, receiver) = gpu_event_channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
    let (_label_space, address) = mapped_test_label();
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
    let pending = [pin_guest_memory_write(GuestMemoryWriteOperation::Release(
        GpuMemoryRelease {
            address,
            value: GpuMemoryReleaseValue::Immediate32(7),
        },
    ))
    .expect("pin release label mapping")];

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
        read_test_label(address), 7,
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
    let (_completion_space, completion_address) = mapped_test_label();
    let invalid = PendingGuestMemoryWrite {
        operation: GuestMemoryWriteOperation::Release(GpuMemoryRelease {
            address: 2,
            value: GpuMemoryReleaseValue::Immediate32(1),
        }),
        mapping_identity: None,
        write_watch: None,
    };
    let completion = pin_guest_memory_write(GuestMemoryWriteOperation::Release(
        GpuMemoryRelease {
            address: completion_address,
            value: GpuMemoryReleaseValue::Immediate32(9),
        },
    ))
    .expect("pin completion label mapping");
    for memory_writes in [vec![invalid], vec![completion]] {
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
        read_test_label(completion_address), 9,
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

    let (_completion_space, completion_address) = mapped_test_label();
    presenter
        .committed_guest_batches
        .push_back(GuestCommandBatch {
            command_buffer,
            work: Vec::new(),
            global_upload_chunks: Vec::new(),
            global_writebacks: Vec::new(),
            guest_write_ranges: Vec::new(),
            memory_writes: vec![pin_guest_memory_write(
                GuestMemoryWriteOperation::Release(GpuMemoryRelease {
                    address: completion_address,
                    value: GpuMemoryReleaseValue::Immediate32(9),
                }),
            )
            .expect("pin completion label mapping")],
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
        read_test_label(completion_address), 0,
        "a timeout must not publish ordered guest completion effects"
    );

    event.set_signaled_value(1);
    presenter
        .wait_for_guest_command_batches()
        .expect("the retained batch retires after Metal completes it");
    assert!(presenter.committed_guest_batches.is_empty());
    assert_eq!(read_test_label(completion_address), 9);
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
    let (_label_space, release_address) = mapped_test_label();
    let words_address = release_address + 16;
    let fill_address = release_address + 32;
    let release = pin_guest_memory_write(GuestMemoryWriteOperation::Release(GpuMemoryRelease {
        address: release_address,
        value: GpuMemoryReleaseValue::Immediate64(0x1122_3344_5566_7788),
    }))
    .expect("pin release destination");
    let write = pin_guest_memory_write(GuestMemoryWriteOperation::Write(GpuMemoryWrite {
        address: words_address,
        values: vec![3, 5, 8],
        increment_address: true,
    }))
    .expect("pin write destination");
    let fill = pin_guest_memory_write(GuestMemoryWriteOperation::Fill {
        address: fill_address,
        value: 0xaabb_ccdd,
        dword_count: 3,
    })
    .expect("pin fill destination");
    assert_eq!(
        release.value_at(release_address, true),
        Some(0x1122_3344_5566_7788)
    );
    assert_eq!(write.value_at(words_address + 8, false), Some(8));
    assert_eq!(
        fill.value_at(fill_address + 4, false),
        Some(0xaabb_ccdd)
    );
    assert_eq!(
        fill.value_at(fill_address, true),
        Some(0xaabb_ccdd_aabb_ccdd)
    );
    apply_pending_guest_memory_writes(&[release, write, fill]).expect("ordered guest writes");
    assert_eq!(
        unsafe { std::ptr::read_volatile(release_address as *const u64) },
        0x1122_3344_5566_7788
    );
    assert_eq!(
        unsafe { std::slice::from_raw_parts(words_address as *const u32, 3) },
        [3, 5, 8]
    );
    assert_eq!(
        unsafe { std::slice::from_raw_parts(fill_address as *const u32, 3) },
        [0xaabb_ccdd; 3]
    );
}

#[test]
fn pixel_pipe_release_trace_follows_the_exact_deferred_write_to_publication() {
    let (_label_space, release_address) = mapped_test_label();
    let release = GpuMemoryRelease {
        address: release_address,
        value: GpuMemoryReleaseValue::Immediate64(0x8000_0000_02ff_ffff),
    };
    let trace = remu_gpu::PixelPipeStatReleaseTrace {
        dump_sequence: 47,
        pair_index: 15,
    };
    remu_gpu::record_pixel_pipe_stat_release_trace(release, trace);
    let pending = pin_guest_memory_write(GuestMemoryWriteOperation::Release(release))
        .expect("pin traced pixel-pipe release destination");

    apply_pending_guest_memory_writes(&[pending]).expect("publish traced pixel-pipe release");

    assert_eq!(
        unsafe { std::ptr::read_volatile(release_address as *const u64) },
        0x8000_0000_02ff_ffff
    );
    assert_eq!(
        remu_gpu::take_pixel_pipe_stat_release_trace(release),
        None,
        "publication consumes the exact trace provenance once"
    );
}

#[test]
fn pixel_pipe_query_completion_trace_follows_release_to_publication() {
    let (_label_space, release_address) = mapped_test_label();
    let payload_address = release_address - 0x100;
    remu_gpu::record_pixel_pipe_stat_dump_trace(2001, payload_address, false);
    remu_gpu::record_pixel_pipe_stat_dump_trace(2002, payload_address + 8, true);
    let release = GpuMemoryRelease {
        address: release_address,
        value: GpuMemoryReleaseValue::Immediate64(1),
    };
    let trace = remu_gpu::record_pixel_pipe_query_completion_release_trace(release)
        .expect("the matching query-completion release is traced");
    let pending = pin_guest_memory_write(GuestMemoryWriteOperation::Release(release))
        .expect("pin traced query-completion release destination");

    apply_pending_guest_memory_writes(&[pending]).expect("publish query completion");

    assert_eq!(
        unsafe { std::ptr::read_volatile(release_address as *const u64) },
        1
    );
    assert_eq!(
        remu_gpu::take_pixel_pipe_query_completion_release_trace(release),
        None,
        "publication consumes the exact query-completion provenance once"
    );
    assert_eq!(trace.payload_address, payload_address);
    assert_eq!(trace.completion_address, release_address);
}

#[test]
fn deferred_shared_guest_write_rejects_reused_mapping() {
    let mut address_space =
        GuestAddressSpace::new(2 * GUEST_PAGE_SIZE).expect("guest address space");
    let original_backing =
        GuestPhysicalBacking::new(GUEST_PAGE_SIZE).expect("original guest backing");
    let address = address_space
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &original_backing,
            0,
        )
        .expect("original shared mapping");
    let pending = pin_guest_memory_write(GuestMemoryWriteOperation::Release(GpuMemoryRelease {
        address: address.get(),
        value: GpuMemoryReleaseValue::Immediate32(0xdead_beef),
    }))
    .expect("pin original shared mapping");

    address_space
        .unmap(address, GUEST_PAGE_SIZE)
        .expect("unmap original shared mapping");
    let replacement_backing =
        GuestPhysicalBacking::new(GUEST_PAGE_SIZE).expect("replacement guest backing");
    address_space
        .map_shared(
            GuestMapPlacement::Fixed(address),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &replacement_backing,
            0,
        )
        .expect("replacement shared mapping");

    let error = apply_pending_guest_memory_writes(&[pending])
        .expect_err("stale deferred write must reject the replacement mapping");
    assert!(error.contains("mapping changed before completion"));
    let mut replacement = [0_u8; 4];
    address_space
        .read(address, &mut replacement)
        .expect("read replacement mapping");
    assert_eq!(replacement, [0; 4]);
}

#[test]
fn deferred_anonymous_guest_write_rejects_reused_mapping() {
    let mut address_space =
        GuestAddressSpace::new(2 * GUEST_PAGE_SIZE).expect("guest address space");
    let address = address_space
        .map_anonymous(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
        )
        .expect("original anonymous mapping");
    let pending = pin_guest_memory_write(GuestMemoryWriteOperation::Release(GpuMemoryRelease {
        address: address.get(),
        value: GpuMemoryReleaseValue::Immediate32(0xdead_beef),
    }))
    .expect("pin original anonymous mapping");

    address_space
        .unmap(address, GUEST_PAGE_SIZE)
        .expect("unmap original anonymous mapping");
    address_space
        .map_anonymous(
            GuestMapPlacement::Fixed(address),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
        )
        .expect("replacement anonymous mapping");

    let error = apply_pending_guest_memory_writes(&[pending])
        .expect_err("stale deferred write must reject the replacement mapping");
    assert!(error.contains("mapping changed before completion"));
    let mut replacement = [0_u8; 4];
    address_space
        .read(address, &mut replacement)
        .expect("read replacement mapping");
    assert_eq!(replacement, [0; 4]);
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
        pixel_depth_export: false,
        ngg_subgroup_limits: None,
        diagnostics: GraphicsDiagnosticPipelineState::default(),
        targets: Vec::new(),
        depth: None,
        sample_count: 1,
        render_target_array_length: 1,
        primitive_topology_class: MTLPrimitiveTopologyClass::Triangle,
    };
    let mut capture_key = key.clone();
    capture_key.diagnostics.capture_vertex_position = true;
    capture_key.diagnostics.capture_pixel_output = true;
    assert_ne!(
        key, capture_key,
        "capture hooks change both Metal function interfaces and must not reuse a hook-free pipeline"
    );
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
fn blocked_wait_dependencies_are_ingested_before_periodic_retry_clears_block_state() {
    let (sender, receiver) = std::sync::mpsc::channel();
    let wait = GpuMemoryWait {
        address: 0x4a00,
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
        .expect("queue compute producer behind the ordinary ingress cap");

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
        !ordered.submissions.contains_key(&33),
        "the producer starts behind the ordinary decoded-event cap"
    );

    // This is the order used by a presenter polling tick. Dependency ingress must observe the
    // authoritative block graph before periodic label polling makes the wait runnable again.
    ordered.retry_guest_memory_waits();
    assert!(
        ordered.submissions.contains_key(&33),
        "dependency ingress must discover the independent producer"
    );

    let source = match ordered.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::MemoryWait(actual),
        } => {
            assert_eq!(actual, wait);
            source
        }
        other => panic!("expected periodic wait retry, got {other:?}"),
    };
    ordered.block_event(
        source,
        GpuEvent::MemoryWait(wait),
        QueueBlockReason::GuestMemoryWait(wait),
    );
    assert!(matches!(
        ordered.try_recv(|_| true),
        OrderedEventPoll::Event {
            source: OrderedEventSource::GuestQueue(33),
            event: GpuEvent::MemoryRelease(GpuMemoryRelease {
                address: 0x4a00,
                value: GpuMemoryReleaseValue::Immediate32(1),
            }),
        }
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
