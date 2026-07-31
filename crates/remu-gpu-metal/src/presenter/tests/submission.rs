// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use metal::{
    Device, MTLClearColor, MTLLoadAction, MTLPixelFormat, MTLResourceOptions, MTLStorageMode,
    MTLStoreAction, MTLTextureType, MTLTextureUsage, RenderPassDescriptor, TextureDescriptor,
};
use remu_gpu::{
    ComputeDispatch, ComputeDispatchSource, ComputeGrid, ComputeLaunch, GpuEvent, GpuMemoryAcquire,
    GpuMemoryCopy, GpuMemoryRelease, GpuMemoryReleaseValue, GpuMemoryWait, GpuMemoryWrite,
    GpuPipelineBarrier, GpuPipelineStage, gpu_event_channel,
};
use remu_memory::{
    CpuWriteWatch, GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
    GuestProtection, guest_shared_backing_identity, guest_shared_backing_range,
};
use remu_shader::{GlobalMemoryBinding, Program, ShaderEvaluation};
use std::sync::Arc;
use std::time::Duration;

use super::super::UploadedGlobalBuffer;
use super::super::cache::metal_buffer_with_guest_backing_range;
use super::super::submission::{
    append_unique_global_writebacks, append_unique_guest_write_ranges,
    collect_pending_global_writebacks, collect_pending_guest_write_ranges,
};
use super::{
    ComputePipelineKey, GLOBAL_UPLOAD_ALIGNMENT, GuestBatchFlushReason, GuestCommandBatch,
    GuestMemoryWaitKey, GuestMemoryWaitRecovery, GuestMemoryWaitStatus, GuestMemoryWaitTracker,
    GuestMemoryWriteOperation, GuestRenderPassKey, GuestTextureDescriptor, MetalPresenter,
    OpenGuestRenderPass, OrderedEventPoll, OrderedEventSource, OrderedGpuEventReceiver,
    PendingGlobalWriteback, PendingGuestMemoryWrite, PipelineCompileCandidate, ProgramIdentity,
    QueueBlockReason, apply_pending_guest_memory_writes, indexed_draw, pin_guest_memory_write,
    read_texture_linear, write_back_pending_globals,
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
fn dma_uses_packet_time_cpu_snapshot_after_source_reuse() {
    let mut address_space =
        GuestAddressSpace::new(2 * GUEST_PAGE_SIZE).expect("guest DMA address space");
    let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).expect("guest DMA backing");
    let base = address_space
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("guest DMA mapping");
    let source = base.get();
    let destination_address = base.checked_add(128).expect("destination guest address");
    let destination = destination_address.get();
    address_space
        .write(base, &[1, 2, 3, 4])
        .expect("initial source");
    let snapshot = remu_gpu::GpuMemoryCopySnapshot::try_new(Arc::<[u8]>::from([1, 2, 3, 4]))
        .expect("snapshot budget");
    address_space
        .write(base, &[9, 9, 9, 9])
        .expect("recycled source");

    let (_sender, receiver) = gpu_event_channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
    presenter
        .execute_guest_memory_copy(GpuMemoryCopy {
            source_address: source,
            destination_address: destination,
            byte_count: 4,
            source_selection: 3,
            destination_selection: 3,
            source_snapshot: Some(snapshot),
        })
        .expect("snapshot DMA");
    let mut copied = [0_u8; 4];
    address_space
        .read(destination_address, &mut copied)
        .expect("copied destination");
    assert_eq!(copied, [1, 2, 3, 4]);
}

#[test]
fn dma_rejects_a_truncated_packet_time_snapshot() {
    let mut address_space =
        GuestAddressSpace::new(2 * GUEST_PAGE_SIZE).expect("guest DMA address space");
    let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).expect("guest DMA backing");
    let base = address_space
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("guest DMA mapping");
    let (_sender, receiver) = gpu_event_channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
    let error = presenter
        .execute_guest_memory_copy(GpuMemoryCopy {
            source_address: base.get(),
            destination_address: base.get() + 128,
            byte_count: 4,
            source_selection: 3,
            destination_selection: 3,
            source_snapshot: Some(
                remu_gpu::GpuMemoryCopySnapshot::try_new(Arc::<[u8]>::from([1, 2, 3]))
                    .expect("snapshot budget"),
            ),
        })
        .expect_err("truncated snapshot must fail");
    assert!(error.contains("snapshot length mismatch"), "{error}");
}

#[test]
fn dma_ignores_cpu_snapshot_when_an_ordered_gpu_source_is_authoritative() {
    let mut address_space =
        GuestAddressSpace::new(4 * GUEST_PAGE_SIZE).expect("guest DMA address space");
    let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).expect("guest DMA backing");
    let producer_alias = address_space
        .map_shared(
            GuestMapPlacement::Fixed(address_space.base()),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("GPU producer alias");
    let source_alias = address_space
        .map_shared(
            GuestMapPlacement::Fixed(
                address_space
                    .base()
                    .checked_add(2 * GUEST_PAGE_SIZE)
                    .expect("source alias address"),
            ),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("DMA source alias");
    let source = source_alias.get();
    let destination_address = source_alias
        .checked_add(128)
        .expect("destination guest address");
    address_space
        .write(source_alias, &[9, 9, 9, 9])
        .expect("GPU-current source bytes");

    let (_sender, receiver) = gpu_event_channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let binding = GlobalMemoryBinding {
        scalar_address: 0,
        base_address: producer_alias.get(),
        instruction_pcs: Default::default(),
        byte_count: 4,
        data: Vec::new(),
        writable: true,
        write_back_to_guest: false,
    };
    let _producer = record_zero_copy_guest_writer(&mut presenter, &device, &binding);
    presenter
        .execute_guest_memory_copy(GpuMemoryCopy {
            source_address: source,
            destination_address: destination_address.get(),
            byte_count: 4,
            source_selection: 3,
            destination_selection: 3,
            source_snapshot: Some(
                remu_gpu::GpuMemoryCopySnapshot::try_new(Arc::<[u8]>::from([1, 2, 3, 4]))
                    .expect("snapshot budget"),
            ),
        })
        .expect("GPU-authoritative DMA");
    let mut copied = [0_u8; 4];
    address_space
        .read(destination_address, &mut copied)
        .expect("copied destination");
    assert_eq!(
        copied,
        [9, 9, 9, 9],
        "an ordered GPU producer through a physical alias must win over the earlier CPU snapshot"
    );
}

fn evaluation_with_global(binding: GlobalMemoryBinding) -> ShaderEvaluation {
    ShaderEvaluation {
        initial_scalar_registers: Default::default(),
        scalar_registers: Default::default(),
        scalar_memory_reads: Vec::new(),
        image_bindings: Vec::new(),
        global_memory_bindings: vec![binding],
        compute_system_registers: None,
        vertex_inputs: Vec::new(),
    }
}

fn record_zero_copy_guest_writer(
    presenter: &mut MetalPresenter,
    device: &Device,
    binding: &GlobalMemoryBinding,
) -> UploadedGlobalBuffer {
    let queue = presenter
        .command_queue
        .as_ref()
        .expect("Metal queue")
        .clone();
    presenter.ensure_open_guest_batch(&queue);
    let byte_count = u64::try_from(binding.byte_count).expect("binding byte count fits u64");
    let backing_range = guest_shared_backing_range(binding.base_address, byte_count)
        .expect("synthetic zero-copy writer range resolves");
    let uploaded = UploadedGlobalBuffer {
        buffer: metal_buffer_with_guest_backing_range(device, &backing_range)
            .expect("synthetic zero-copy writer buffer"),
        offset: backing_range.resource_offset(),
        guest_memory: true,
        physical_identity: guest_shared_backing_identity(binding.base_address, byte_count),
    };
    let evaluation = evaluation_with_global(binding.clone());
    let mut pending = Vec::new();
    collect_pending_guest_write_ranges(std::slice::from_ref(&uploaded), &evaluation, &mut pending)
        .expect("zero-copy producer provenance is recorded");
    append_unique_guest_write_ranges(
        &mut presenter
            .open_guest_batch
            .as_mut()
            .expect("open guest batch")
            .guest_write_ranges,
        pending,
    );
    uploaded
}

#[test]
fn native_metal_draw_arguments_match_the_gen5_indirect_layouts() {
    use metal::{MTLDrawIndexedPrimitivesIndirectArguments, MTLDrawPrimitivesIndirectArguments};

    assert_eq!(
        std::mem::size_of::<MTLDrawPrimitivesIndirectArguments>(),
        16
    );
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
        committed_at: None,
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
    assert_eq!(
        read_test_label(release_address),
        0,
        "the label remains deferred"
    );
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
        committed_at: None,
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
        committed_at: None,
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
fn queued_copied_writer_forwards_across_a_shifted_physical_alias() {
    let mut address_space = GuestAddressSpace::new(5 * GUEST_PAGE_SIZE).unwrap();
    let base = address_space.base();
    let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).unwrap();
    let producer_alias = address_space
        .map_shared(
            GuestMapPlacement::Fixed(base),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    let consumer_alias = address_space
        .map_shared(
            GuestMapPlacement::Fixed(base.checked_add(2 * GUEST_PAGE_SIZE).unwrap()),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    let (_sender, receiver) = gpu_event_channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let queue = presenter
        .command_queue
        .as_ref()
        .expect("Metal queue")
        .clone();
    presenter.ensure_open_guest_batch(&queue);
    let producer = device.new_buffer(512, MTLResourceOptions::StorageModeShared);
    let producer_address = producer_alias.get() + 128;
    let producer_bytes = 256_u64;
    presenter
        .open_guest_batch
        .as_mut()
        .expect("guest batch")
        .global_writebacks
        .push(PendingGlobalWriteback {
            buffer: producer.clone(),
            offset: 32,
            guest_address: producer_address,
            byte_count: usize::try_from(producer_bytes).unwrap(),
            mapping_identity: guest_shared_backing_identity(producer_address, producer_bytes),
            write_watch: None,
            original_bytes: vec![0; usize::try_from(producer_bytes).unwrap()],
        });
    let consumer_binding = GlobalMemoryBinding {
        scalar_address: 0,
        base_address: consumer_alias.get() + 160,
        instruction_pcs: Default::default(),
        byte_count: 12,
        data: Vec::new(),
        writable: false,
        write_back_to_guest: false,
    };
    let consumer = presenter
        .upload_global_buffer(&device, &consumer_binding)
        .expect("copied producer is forwarded through its physical alias");
    assert!(!consumer.guest_memory);
    assert!(std::ptr::eq(producer.as_ref(), consumer.buffer.as_ref()));
    assert_eq!(consumer.offset, 64);
    assert_eq!(
        consumer.physical_identity,
        guest_shared_backing_identity(consumer_binding.base_address, 12)
    );
}

#[test]
fn disjoint_guest_aliases_share_one_coherent_physical_global_buffer() {
    let mut address_space = GuestAddressSpace::new(10 * GUEST_PAGE_SIZE).unwrap();
    let base = address_space.base();
    let backing = GuestPhysicalBacking::new(2 * GUEST_PAGE_SIZE).unwrap();
    let first = address_space
        .map_shared(
            GuestMapPlacement::Fixed(base),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("first disjoint alias");
    let second = address_space
        .map_shared(
            GuestMapPlacement::Fixed(
                base.checked_add(4 * GUEST_PAGE_SIZE)
                    .expect("second alias address"),
            ),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            GUEST_PAGE_SIZE,
        )
        .expect("second disjoint alias");
    let contiguous = address_space
        .map_shared(
            GuestMapPlacement::Fixed(
                base.checked_add(7 * GUEST_PAGE_SIZE)
                    .expect("contiguous alias address"),
            ),
            2 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("contiguous alias of the same physical range");

    let interior_offset = 128_u64;
    // SAFETY: the aliases stay live and writable throughout this test.
    unsafe {
        ((first.get() + interior_offset) as *mut u8).write(0x11);
        ((second.get() + interior_offset) as *mut u8).write(0x22);
    }
    let requested_bytes =
        usize::try_from(2 * GUEST_PAGE_SIZE - 2 * interior_offset).expect("request fits usize");
    let requested_bytes_u64 = u64::try_from(requested_bytes).expect("request fits u64");
    let disjoint_range =
        guest_shared_backing_range(first.get() + interior_offset, requested_bytes_u64)
            .expect("disjoint physical range resolves");
    let contiguous_range =
        guest_shared_backing_range(contiguous.get() + interior_offset, requested_bytes_u64)
            .expect("contiguous physical alias resolves");
    assert_eq!(
        disjoint_range.view_identity(),
        contiguous_range.view_identity(),
        "physical cache identity is independent of the selected guest alias"
    );
    assert_eq!(
        disjoint_range.view_byte_count(),
        contiguous_range.view_byte_count()
    );
    assert_eq!(disjoint_range.resource_offset(), interior_offset);

    let (_sender, receiver) = gpu_event_channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let buffer = metal_buffer_with_guest_backing_range(&device, &disjoint_range)
        .expect("create the physical no-copy Metal view");
    drop(disjoint_range);
    let buffer_offset = usize::try_from(interior_offset).expect("buffer offset fits usize");
    let page_byte_count = usize::try_from(GUEST_PAGE_SIZE).expect("guest page fits usize");
    // SAFETY: the Metal buffer contains both backing pages and owns its no-copy mmap.
    assert_eq!(
        unsafe { buffer.contents().cast::<u8>().add(buffer_offset).read() },
        0x11
    );
    assert_eq!(
        unsafe {
            buffer
                .contents()
                .cast::<u8>()
                .add(buffer_offset + page_byte_count)
                .read()
        },
        0x22
    );

    // CPU stores through any coherent alias are immediately visible to the shared Metal view.
    unsafe {
        ((second.get() + interior_offset) as *mut u8).write(0x7e);
    }
    assert_eq!(
        unsafe {
            buffer
                .contents()
                .cast::<u8>()
                .add(buffer_offset + page_byte_count)
                .read()
        },
        0x7e
    );

    // The creator's local mmap owner has already gone out of scope; Metal's deallocator block
    // retains the mapped storage for the Buffer's complete lifetime.
    assert_eq!(
        unsafe { buffer.contents().cast::<u8>().add(buffer_offset).read() },
        0x11
    );

    let disjoint_binding = GlobalMemoryBinding {
        scalar_address: 0,
        base_address: first.get() + interior_offset,
        instruction_pcs: Default::default(),
        byte_count: requested_bytes,
        data: Vec::new(),
        writable: false,
        write_back_to_guest: false,
    };
    let mut contiguous_binding = disjoint_binding.clone();
    contiguous_binding.base_address = contiguous.get() + interior_offset;
    let disjoint_upload = presenter
        .upload_global_buffer(&device, &disjoint_binding)
        .expect("disjoint aliases upload through one physical view");
    let contiguous_upload = presenter
        .upload_global_buffer(&device, &contiguous_binding)
        .expect("contiguous alias reuses the physical view");
    assert!(disjoint_upload.guest_memory);
    assert!(contiguous_upload.guest_memory);
    assert!(
        std::ptr::eq(
            disjoint_upload.buffer.as_ref(),
            contiguous_upload.buffer.as_ref()
        ),
        "physically identical aliases must hit the same cached Metal resource"
    );
    assert_eq!(disjoint_upload.offset, contiguous_upload.offset);
}

#[test]
fn writable_physical_alias_uses_exact_staging_and_preserves_adjacent_bytes() {
    let mut address_space = GuestAddressSpace::new(5 * GUEST_PAGE_SIZE).unwrap();
    let base = address_space.base();
    let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).unwrap();
    let producer_alias = address_space
        .map_shared(
            GuestMapPlacement::Fixed(base),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    let observer_alias = address_space
        .map_shared(
            GuestMapPlacement::Fixed(base.checked_add(2 * GUEST_PAGE_SIZE).unwrap()),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    let binding_offset = 128_u64;
    let binding_byte_count = 16_usize;
    let before = producer_alias.get() + binding_offset - 1;
    let binding_address = producer_alias.get() + binding_offset;
    let after = binding_address + u64::try_from(binding_byte_count).unwrap();
    // SAFETY: both aliases remain mapped read/write for the duration of the test.
    unsafe {
        (before as *mut u8).write(0xa1);
        (after as *mut u8).write(0xb2);
        std::ptr::write_bytes(binding_address as *mut u8, 0x11, binding_byte_count);
    }

    let (_sender, receiver) = gpu_event_channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let queue = presenter
        .command_queue
        .as_ref()
        .expect("Metal queue")
        .clone();
    presenter.ensure_open_guest_batch(&queue);
    let binding = GlobalMemoryBinding {
        scalar_address: 0,
        base_address: binding_address,
        instruction_pcs: Default::default(),
        byte_count: binding_byte_count,
        data: Vec::new(),
        writable: true,
        write_back_to_guest: true,
    };
    let uploaded = presenter
        .upload_global_buffer(&device, &binding)
        .expect("writable alias uses staged storage");
    assert!(!uploaded.guest_memory);
    assert_eq!(uploaded.offset, 0);
    assert_eq!(
        uploaded.buffer.length(),
        u64::try_from(binding_byte_count).unwrap()
    );
    assert_eq!(
        uploaded.physical_identity,
        guest_shared_backing_identity(binding_address, u64::try_from(binding_byte_count).unwrap())
    );

    let evaluation = evaluation_with_global(binding.clone());
    let mut pending = Vec::new();
    collect_pending_global_writebacks(std::slice::from_ref(&uploaded), &evaluation, &mut pending)
        .expect("exact staged writeback is captured");
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].mapping_identity, uploaded.physical_identity);
    assert!(pending[0].write_watch.is_none());
    // SAFETY: the exact shared buffer is binding_byte_count bytes long and retained by `uploaded`.
    unsafe {
        std::ptr::write_bytes(
            uploaded.buffer.contents().cast::<u8>(),
            0x5a,
            binding_byte_count,
        );
    }
    write_back_pending_globals(&pending).expect("identity-checked exact writeback succeeds");

    let observer_binding = observer_alias.get() + binding_offset;
    // SAFETY: both aliases remain mapped and share the same physical backing.
    unsafe {
        assert_eq!((before as *const u8).read(), 0xa1);
        assert_eq!((after as *const u8).read(), 0xb2);
        assert_eq!(
            ((observer_binding - 1) as *const u8).read(),
            0xa1,
            "the byte before the exact writable range is preserved through the alias"
        );
        assert_eq!(
            ((observer_binding + u64::try_from(binding_byte_count).unwrap()) as *const u8).read(),
            0xb2,
            "the byte after the exact writable range is preserved through the alias"
        );
        for offset in 0..binding_byte_count {
            assert_eq!(
                ((observer_binding + u64::try_from(offset).unwrap()) as *const u8).read(),
                0x5a
            );
        }
    }
}

#[test]
fn queued_zero_copy_writer_stages_writable_alias_across_shifted_aliases_and_eviction() {
    let mut address_space = GuestAddressSpace::new(8 * GUEST_PAGE_SIZE).unwrap();
    let base = address_space.base();
    let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).unwrap();
    let producer_alias = address_space
        .map_shared(
            GuestMapPlacement::Fixed(base),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    let consumer_alias = address_space
        .map_shared(
            GuestMapPlacement::Fixed(base.checked_add(2 * GUEST_PAGE_SIZE).unwrap()),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    let third_alias = address_space
        .map_shared(
            GuestMapPlacement::Fixed(base.checked_add(4 * GUEST_PAGE_SIZE).unwrap()),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();

    let (_sender, receiver) = gpu_event_channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let producer_binding = GlobalMemoryBinding {
        scalar_address: 0,
        base_address: producer_alias.get() + 128,
        instruction_pcs: Default::default(),
        byte_count: 256,
        data: Vec::new(),
        writable: true,
        write_back_to_guest: false,
    };
    let producer = record_zero_copy_guest_writer(&mut presenter, &device, &producer_binding);

    // Eviction must not let the consumer recreate an aliasing Metal object. The queued producer
    // retains the exact resource and its mmap independently of the cache entry.
    presenter.guest_global_buffer_cache.clear();
    presenter.guest_global_buffer_cache_bytes = 0;
    let consumer_binding = GlobalMemoryBinding {
        scalar_address: 0,
        base_address: consumer_alias.get() + 192,
        instruction_pcs: Default::default(),
        byte_count: 64,
        data: Vec::new(),
        writable: true,
        write_back_to_guest: true,
    };
    let consumer = presenter
        .upload_global_buffer(&device, &consumer_binding)
        .expect("contained writable alias stages the queued producer");
    assert!(!consumer.guest_memory);
    assert!(!std::ptr::eq(
        producer.buffer.as_ref(),
        consumer.buffer.as_ref()
    ));
    assert_eq!(consumer.offset, 0);
    assert_eq!(consumer.buffer.length(), 64);

    let consumer_evaluation = evaluation_with_global(consumer_binding.clone());
    let mut pending_consumer = Vec::new();
    collect_pending_global_writebacks(
        std::slice::from_ref(&consumer),
        &consumer_evaluation,
        &mut pending_consumer,
    )
    .unwrap();
    append_unique_global_writebacks(
        &mut presenter
            .open_guest_batch
            .as_mut()
            .expect("producer batch stays open")
            .global_writebacks,
        pending_consumer,
    );

    let third_binding = GlobalMemoryBinding {
        scalar_address: 0,
        base_address: third_alias.get() + 208,
        instruction_pcs: Default::default(),
        byte_count: 8,
        data: Vec::new(),
        writable: false,
        write_back_to_guest: false,
    };
    let third = presenter
        .upload_global_buffer(&device, &third_binding)
        .expect("third read forwards the latest writable alias");
    assert!(!third.guest_memory);
    assert!(std::ptr::eq(
        consumer.buffer.as_ref(),
        third.buffer.as_ref()
    ));
    assert_eq!(third.offset, 16);
}

#[test]
fn partially_overlapping_zero_copy_writer_synchronizes_instead_of_aliasing_buffers() {
    let mut address_space = GuestAddressSpace::new(5 * GUEST_PAGE_SIZE).unwrap();
    let base = address_space.base();
    let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).unwrap();
    let producer_alias = address_space
        .map_shared(
            GuestMapPlacement::Fixed(base),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    let consumer_alias = address_space
        .map_shared(
            GuestMapPlacement::Fixed(base.checked_add(2 * GUEST_PAGE_SIZE).unwrap()),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    let (_sender, receiver) = gpu_event_channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let producer_binding = GlobalMemoryBinding {
        scalar_address: 0,
        base_address: producer_alias.get() + 128,
        instruction_pcs: Default::default(),
        byte_count: 256,
        data: Vec::new(),
        writable: true,
        write_back_to_guest: false,
    };
    let producer = record_zero_copy_guest_writer(&mut presenter, &device, &producer_binding);
    let partial_binding = GlobalMemoryBinding {
        scalar_address: 0,
        base_address: consumer_alias.get() + 64,
        instruction_pcs: Default::default(),
        byte_count: 128,
        data: Vec::new(),
        writable: false,
        write_back_to_guest: false,
    };
    let synchronized = presenter
        .upload_global_buffer(&device, &partial_binding)
        .expect("partial overlap drains before taking a fresh snapshot");
    assert!(!synchronized.guest_memory);
    assert!(!std::ptr::eq(
        producer.buffer.as_ref(),
        synchronized.buffer.as_ref()
    ));
}

#[test]
fn ambiguous_zero_copy_writer_resources_synchronize_before_read() {
    let mut address_space = GuestAddressSpace::new(2 * GUEST_PAGE_SIZE).unwrap();
    let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).unwrap();
    let alias = address_space
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    let (_sender, receiver) = gpu_event_channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let first_binding = GlobalMemoryBinding {
        scalar_address: 0,
        base_address: alias.get() + 128,
        instruction_pcs: Default::default(),
        byte_count: 256,
        data: Vec::new(),
        writable: true,
        write_back_to_guest: false,
    };
    let first = record_zero_copy_guest_writer(&mut presenter, &device, &first_binding);
    let first_identity = first
        .physical_identity
        .expect("zero-copy upload captures physical identity");
    let conflicting_buffer =
        device.new_buffer(GUEST_PAGE_SIZE, MTLResourceOptions::StorageModeShared);
    presenter
        .open_guest_batch
        .as_mut()
        .expect("producer batch stays open")
        .guest_write_ranges
        .push(super::super::PendingGuestWriteRange {
            address: alias.get() + 160,
            byte_count: 256,
            buffer: conflicting_buffer,
            offset: 160,
            physical_identity: remu_memory::GuestSharedMappingIdentity {
                backing_id: first_identity.backing_id,
                backing_offset: first_identity.backing_offset + 32,
            },
        });

    let read_binding = GlobalMemoryBinding {
        scalar_address: 0,
        base_address: alias.get() + 192,
        instruction_pcs: Default::default(),
        byte_count: 8,
        data: Vec::new(),
        writable: false,
        write_back_to_guest: false,
    };
    let synchronized = presenter
        .upload_global_buffer(&device, &read_binding)
        .expect("ambiguous resources drain before read");
    assert!(!synchronized.guest_memory);
    assert!(!std::ptr::eq(
        first.buffer.as_ref(),
        synchronized.buffer.as_ref()
    ));
}

#[test]
fn remapped_zero_copy_writer_address_synchronizes_before_reuse() {
    let mut address_space = GuestAddressSpace::new(3 * GUEST_PAGE_SIZE).unwrap();
    let base = address_space.base();
    let original_backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).unwrap();
    let producer_alias = address_space
        .map_shared(
            GuestMapPlacement::Fixed(base),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &original_backing,
            0,
        )
        .unwrap();
    let (_sender, receiver) = gpu_event_channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let producer_binding = GlobalMemoryBinding {
        scalar_address: 0,
        base_address: producer_alias.get() + 128,
        instruction_pcs: Default::default(),
        byte_count: 256,
        data: Vec::new(),
        writable: true,
        write_back_to_guest: false,
    };
    let producer = record_zero_copy_guest_writer(&mut presenter, &device, &producer_binding);

    address_space
        .unmap(producer_alias, GUEST_PAGE_SIZE)
        .unwrap();
    let replacement_backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).unwrap();
    address_space
        .map_shared(
            GuestMapPlacement::Fixed(producer_alias),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &replacement_backing,
            0,
        )
        .unwrap();
    let remapped_binding = GlobalMemoryBinding {
        scalar_address: 0,
        base_address: producer_alias.get() + 160,
        instruction_pcs: Default::default(),
        byte_count: 12,
        data: Vec::new(),
        writable: false,
        write_back_to_guest: false,
    };
    let synchronized = presenter
        .upload_global_buffer(&device, &remapped_binding)
        .expect("replacement mapping is read only after draining the old producer");
    assert!(!synchronized.guest_memory);
    assert!(!std::ptr::eq(
        producer.buffer.as_ref(),
        synchronized.buffer.as_ref()
    ));
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
        committed_at: None,
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
    let pending = [
        pin_guest_memory_write(GuestMemoryWriteOperation::Release(GpuMemoryRelease {
            address,
            value: GpuMemoryReleaseValue::Immediate32(7),
        }))
        .expect("pin release label mapping"),
    ];

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
        read_test_label(address),
        7,
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
    let completion = pin_guest_memory_write(GuestMemoryWriteOperation::Release(GpuMemoryRelease {
        address: completion_address,
        value: GpuMemoryReleaseValue::Immediate32(9),
    }))
    .expect("pin completion label mapping");
    for memory_writes in [vec![invalid], vec![completion]] {
        let command_buffer = queue.new_command_buffer().to_owned();
        command_buffer.commit();
        presenter
            .committed_guest_batches
            .push_back(GuestCommandBatch {
                command_buffer,
                work: Vec::new(),
                committed_at: None,
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
        read_test_label(completion_address),
        9,
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
            committed_at: None,
            global_upload_chunks: Vec::new(),
            global_writebacks: Vec::new(),
            guest_write_ranges: Vec::new(),
            memory_writes: vec![
                pin_guest_memory_write(GuestMemoryWriteOperation::Release(GpuMemoryRelease {
                    address: completion_address,
                    value: GpuMemoryReleaseValue::Immediate32(9),
                }))
                .expect("pin completion label mapping"),
            ],
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
        read_test_label(completion_address),
        0,
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
        committed_at: None,
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
fn stale_shared_global_writeback_does_not_poison_valid_or_later_ordered_work() {
    let mut address_space =
        GuestAddressSpace::new(4 * GUEST_PAGE_SIZE).expect("guest address space");
    let base = address_space.base();
    let stale_backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).expect("stale backing");
    let replacement_backing =
        GuestPhysicalBacking::new(GUEST_PAGE_SIZE).expect("replacement backing");
    let valid_backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).expect("valid backing");
    let stale_address = address_space
        .map_shared(
            GuestMapPlacement::Fixed(base),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &stale_backing,
            0,
        )
        .expect("stale mapping");
    let valid_address = address_space
        .map_shared(
            GuestMapPlacement::Fixed(base.checked_add(2 * GUEST_PAGE_SIZE).unwrap()),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &valid_backing,
            0,
        )
        .expect("valid mapping");
    let stale_identity =
        guest_shared_backing_identity(stale_address.get(), 4).expect("stale mapping identity");
    let valid_identity =
        guest_shared_backing_identity(valid_address.get(), 4).expect("valid mapping identity");
    address_space
        .unmap(stale_address, GUEST_PAGE_SIZE)
        .expect("unmap stale allocation");
    address_space
        .map_shared(
            GuestMapPlacement::Fixed(stale_address),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &replacement_backing,
            0,
        )
        .expect("replacement mapping");

    let (_sender, receiver) = gpu_event_channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
    let device = presenter.device.as_ref().expect("Metal device");
    let queue = presenter
        .command_queue
        .as_ref()
        .expect("Metal queue")
        .clone();
    let stale_buffer = device.new_buffer_with_data(
        [1_u8, 2, 3, 4].as_ptr().cast(),
        4,
        MTLResourceOptions::StorageModeShared,
    );
    let valid_buffer = device.new_buffer_with_data(
        [5_u8, 6, 7, 8].as_ptr().cast(),
        4,
        MTLResourceOptions::StorageModeShared,
    );
    let first_command_buffer = queue.new_command_buffer().to_owned();
    first_command_buffer.commit();
    presenter
        .committed_guest_batches
        .push_back(GuestCommandBatch {
            command_buffer: first_command_buffer,
            work: Vec::new(),
            committed_at: None,
            global_upload_chunks: Vec::new(),
            global_writebacks: vec![
                PendingGlobalWriteback {
                    buffer: stale_buffer,
                    offset: 0,
                    guest_address: stale_address.get(),
                    byte_count: 4,
                    mapping_identity: Some(stale_identity),
                    write_watch: None,
                    original_bytes: vec![0; 4],
                },
                PendingGlobalWriteback {
                    buffer: valid_buffer,
                    offset: 0,
                    guest_address: valid_address.get(),
                    byte_count: 4,
                    mapping_identity: Some(valid_identity),
                    write_watch: None,
                    original_bytes: vec![0; 4],
                },
            ],
            guest_write_ranges: Vec::new(),
            memory_writes: Vec::new(),
            render_pass: None,
            completion_sequences: Vec::new(),
        });

    let (_completion_space, completion_address) = mapped_test_label();
    let second_command_buffer = queue.new_command_buffer().to_owned();
    second_command_buffer.commit();
    presenter
        .committed_guest_batches
        .push_back(GuestCommandBatch {
            command_buffer: second_command_buffer,
            work: Vec::new(),
            committed_at: None,
            global_upload_chunks: Vec::new(),
            global_writebacks: Vec::new(),
            guest_write_ranges: Vec::new(),
            memory_writes: vec![
                pin_guest_memory_write(GuestMemoryWriteOperation::Release(GpuMemoryRelease {
                    address: completion_address,
                    value: GpuMemoryReleaseValue::Immediate32(9),
                }))
                .expect("pin later completion label"),
            ],
            render_pass: None,
            completion_sequences: Vec::new(),
        });

    presenter
        .wait_for_guest_command_batches()
        .expect("a stale writeback must not fail this or later ordered batches");

    let mut replacement = [0_u8; 4];
    address_space
        .read(stale_address, &mut replacement)
        .expect("read replacement mapping");
    assert_eq!(
        replacement, [0; 4],
        "the retired writeback must never touch replacement storage"
    );
    let mut valid = [0_u8; 4];
    address_space
        .read(valid_address, &mut valid)
        .expect("read valid mapping");
    assert_eq!(
        valid,
        [5, 6, 7, 8],
        "an independent valid writeback in the same batch must complete"
    );
    assert_eq!(
        read_test_label(completion_address),
        9,
        "later ordered work must execute after retiring the stale entry"
    );
}

#[test]
fn malformed_global_writeback_remains_fatal() {
    let device = Device::system_default().expect("Metal device");
    let buffer = device.new_buffer_with_data(
        [1_u8, 2, 3, 4].as_ptr().cast(),
        4,
        MTLResourceOptions::StorageModeShared,
    );
    let mut destination = [0_u8; 4];
    let guest_address =
        u64::try_from(destination.as_mut_ptr() as usize).expect("host pointer fits u64");

    let error = write_back_pending_globals(&[PendingGlobalWriteback {
        buffer,
        offset: 0,
        guest_address,
        byte_count: destination.len(),
        mapping_identity: None,
        write_watch: None,
        original_bytes: vec![0; destination.len() - 1],
    }])
    .err()
    .expect("a malformed snapshot must remain fatal");

    assert!(error.contains("snapshot length mismatch"));
    assert_eq!(destination, [0; 4]);
}

#[test]
fn anonymous_global_writeback_discards_replaced_guest_mapping() {
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

    let pending = [pending];
    let applied =
        write_back_pending_globals(&pending).expect("stale anonymous writeback is discarded");
    assert!(
        applied.is_empty(),
        "a discarded writeback must not be published as a producer"
    );
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
    assert_eq!(fill.value_at(fill_address + 4, false), Some(0xaabb_ccdd));
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
    let key = super::cold_graphics_pipeline_key();
    let mut capture_key = key.clone();
    capture_key.diagnostics.capture_vertex_position = true;
    capture_key.diagnostics.capture_pixel_output = true;
    assert_ne!(
        key, capture_key,
        "capture hooks change both Metal function interfaces and must not reuse a hook-free pipeline"
    );
    let mut mesh_key = key.clone();
    mesh_key.uses_mesh_stage = true;
    assert_ne!(
        key, mesh_key,
        "mesh functions must not alias ordinary vertex pipelines"
    );
    let mut procedural_key = key.clone();
    procedural_key.uses_procedural_rectangle_stage = true;
    assert_ne!(
        key, procedural_key,
        "synthesized rectangle functions must not alias ordinary triangle-class pipelines"
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
    let prefetched = ordered.prefetch_pipeline_work(16, 16, |_| true);
    assert_eq!(
        prefetched
            .iter()
            .filter_map(|candidate| match candidate {
                PipelineCompileCandidate::Graphics(draw) => Some(draw.sequence),
                PipelineCompileCandidate::Compute(_) => None,
            })
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
fn queue_local_compute_pipeline_gate_preserves_and_prefetches_dispatch() {
    let (sender, receiver) = gpu_event_channel();
    let mut ordered = OrderedGpuEventReceiver::new(receiver);
    let dispatch = ComputeDispatch {
        sequence: 41,
        source: ComputeDispatchSource::Direct,
        launch: ComputeLaunch {
            grid: ComputeGrid::Threadgroups([1, 1, 1]),
            group_count: [1, 1, 1],
            base_group: [0, 0, 0],
            threads_per_group: [8, 8, 1],
            wave_lane_count: 64,
            lds_size_bytes: 0,
            initiator: 1,
        },
        shader_address: 0x5000,
        shader_prefix: [0; 4],
        shader: None,
        shader_evaluation: None,
        shader_registers: Vec::new(),
    };
    let program = Arc::new(Program::new(dispatch.shader_address, Vec::new()));
    let key = ComputePipelineKey {
        program_identity: ProgramIdentity(program),
        lds_size_bytes: 0,
        wave_lane_count: 64,
        allocated_vector_register_count: 4,
        compute_system_registers: None,
        initial_scalar_count: 0,
        global_bindings: Vec::new(),
        image_bindings: Vec::new(),
    };
    sender
        .send(GpuEvent::QueueSubmission {
            queue_id: 0,
            events: vec![
                GpuEvent::ComputeDispatch(Box::new(dispatch.clone())),
                GpuEvent::SubmissionBoundary(42),
            ],
        })
        .expect("queue compute submission");

    let prefetched = ordered.prefetch_pipeline_work(16, 16, |_| true);
    assert_eq!(
        prefetched
            .iter()
            .filter_map(|candidate| match candidate {
                PipelineCompileCandidate::Graphics(_) => None,
                PipelineCompileCandidate::Compute(dispatch) => Some(dispatch.sequence),
            })
            .collect::<Vec<_>>(),
        [41]
    );

    let (source, event) = match ordered.try_recv(|_| false) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::ComputeDispatch(actual),
        } => {
            assert_eq!(actual.as_ref(), &dispatch);
            (source, GpuEvent::ComputeDispatch(actual))
        }
        other => panic!("expected cold compute dispatch, got {other:?}"),
    };
    ordered.block_event(
        source,
        event,
        QueueBlockReason::ComputePipeline(key.clone()),
    );
    assert!(matches!(
        ordered.try_recv(|_| false),
        OrderedEventPoll::Blocked
    ));

    match ordered.try_recv(
        |reason| matches!(reason, QueueBlockReason::ComputePipeline(actual) if actual == &key),
    ) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::ComputeDispatch(actual),
        } => {
            assert_eq!(actual.as_ref(), &dispatch);
            ordered.finish_event(source);
        }
        other => panic!("expected the unblocked compute dispatch exactly once, got {other:?}"),
    }
    let source = match ordered.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::SubmissionBoundary(42),
        } => source,
        other => panic!("expected compute boundary, got {other:?}"),
    };
    ordered.finish_event(source);
    assert!(matches!(
        ordered.try_recv(|_| true),
        OrderedEventPoll::Empty
    ));
}

#[test]
fn pipeline_prefetch_filters_before_clone_and_shares_lookahead_across_queues() {
    let (sender, receiver) = gpu_event_channel();
    let mut first = super::indexed_draw(0, 0, 3);
    first.sequence = 1;
    let mut second = first.clone();
    second.sequence = 2;
    let mut third = first.clone();
    third.sequence = 3;
    let dispatch = ComputeDispatch {
        sequence: 9,
        source: ComputeDispatchSource::Direct,
        launch: ComputeLaunch {
            grid: ComputeGrid::Threadgroups([1, 1, 1]),
            group_count: [1, 1, 1],
            base_group: [0, 0, 0],
            threads_per_group: [8, 8, 1],
            wave_lane_count: 64,
            lds_size_bytes: 0,
            initiator: 1,
        },
        shader_address: 0x9000,
        shader_prefix: [0; 4],
        shader: None,
        shader_evaluation: None,
        shader_registers: Vec::new(),
    };
    sender
        .send(GpuEvent::QueueSubmission {
            queue_id: 0,
            events: vec![
                GpuEvent::IndexedDraw(Box::new(first)),
                GpuEvent::IndexedDraw(Box::new(second)),
                GpuEvent::IndexedDraw(Box::new(third)),
            ],
        })
        .expect("queue graphics lookahead");
    sender
        .send(GpuEvent::QueueSubmission {
            queue_id: 33,
            events: vec![GpuEvent::ComputeDispatch(Box::new(dispatch))],
        })
        .expect("queue compute lookahead");
    let mut ordered = OrderedGpuEventReceiver::new(receiver);

    let prefetched = ordered.prefetch_pipeline_work(
        16,
        2,
        |event| !matches!(event, GpuEvent::IndexedDraw(draw) if draw.sequence == 1),
    );

    assert_eq!(prefetched.len(), 2);
    assert!(matches!(
        &prefetched[0],
        PipelineCompileCandidate::Graphics(draw) if draw.sequence == 2
    ));
    assert!(matches!(
        &prefetched[1],
        PipelineCompileCandidate::Compute(dispatch) if dispatch.sequence == 9
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
fn unpolled_wait_front_does_not_hide_a_third_queue_dependency() {
    let (sender, receiver) = std::sync::mpsc::channel();
    let graphics_wait = GpuMemoryWait {
        address: 0x4c00,
        reference: 1,
        mask: u64::from(u32::MAX),
        compare_function: 3,
        is_64_bit: false,
    };
    let compute_wait = GpuMemoryWait {
        address: 0x4d00,
        ..graphics_wait
    };
    sender
        .send(GpuEvent::QueueSubmission {
            queue_id: 0,
            events: vec![GpuEvent::MemoryWait(graphics_wait)],
        })
        .expect("queue graphics wait");
    sender
        .send(GpuEvent::QueueSubmission {
            queue_id: 33,
            events: vec![GpuEvent::MemoryWait(compute_wait)],
        })
        .expect("queue compute wait");
    for _ in
        0..(remu_gpu::GPU_PENDING_EVENT_CAPACITY / remu_gpu::GPU_EVENT_SUBMISSION_CHUNK_SIZE - 1)
    {
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
            .expect("queue graphics tail");
    }
    sender
        .send(GpuEvent::QueueSubmission {
            queue_id: 34,
            events: vec![GpuEvent::MemoryRelease(GpuMemoryRelease {
                address: graphics_wait.address,
                value: GpuMemoryReleaseValue::Immediate32(1),
            })],
        })
        .expect("queue producer behind two wait fronts");

    let mut ordered = OrderedGpuEventReceiver::new(receiver);
    let source = match ordered.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::MemoryWait(actual),
        } => {
            assert_eq!(actual, graphics_wait);
            source
        }
        other => panic!("expected graphics wait, got {other:?}"),
    };
    ordered.block_event(
        source,
        GpuEvent::MemoryWait(graphics_wait),
        QueueBlockReason::GuestMemoryWait(graphics_wait),
    );
    assert!(
        !ordered.submissions.contains_key(&34),
        "the producer starts behind the ordinary decoded-event cap"
    );
    assert!(
        !ordered.blocked_queues.contains_key(&33),
        "the second wait has not yet been polled"
    );

    ordered.retry_guest_memory_waits();

    assert!(
        ordered.submissions.contains_key(&34),
        "an unpolled wait front must not suppress bounded dependency ingress"
    );
    assert!(
        ordered.pending_event_count()
            <= remu_gpu::GPU_PENDING_EVENT_CAPACITY
                + remu_gpu::GPU_EVENT_QUEUE_CAPACITY
                    * remu_gpu::GPU_EVENT_SUBMISSION_CHUNK_SIZE
                    * 2,
        "dependency discovery must retain the finite pending-event bound"
    );
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
    // The wait has no ingested producer, so the presenter keeps polling in case a CPU store
    // satisfies it. That must not let the 64 blocked barrier submissions behind the wait run.
    assert!(ordered.has_runnable_events());
    assert!(
        matches!(ordered.try_recv(|_| true), OrderedEventPoll::Blocked),
        "blocked queue tails must never become runnable ahead of their wait"
    );
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
    // Both waits already have an ingested producer, so no CPU store is expected to break the
    // deadlock and the presenter may sleep on its idle deadline instead of spinning.
    assert!(!ordered.has_runnable_events());
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
    assert!(matches!(
        ordered.try_recv(|_| true),
        OrderedEventPoll::Blocked
    ));
    // Nothing is runnable, but no ingested packet can write the label either, so only a CPU
    // store can release this queue. Keep polling rather than sleeping on an AppKit deadline.
    assert!(ordered.has_runnable_events());

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
