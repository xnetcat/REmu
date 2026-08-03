// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::super::{MetalPresenter, UploadedGlobalBuffer, compile_metal_compute_library};
use metal::{ComputePipelineState, Device, MTLSize};

const RECTANGLE_INDIRECT_ARGUMENT_SOURCE: &str = r"
#include <metal_stdlib>
using namespace metal;

struct RemuDrawPrimitivesIndirectArguments {
    uint vertex_count;
    uint instance_count;
    uint vertex_start;
    uint base_instance;
};

kernel void remu_expand_rectangle_indirect(
    device const RemuDrawPrimitivesIndirectArguments* source [[buffer(0)]],
    device RemuDrawPrimitivesIndirectArguments* destination [[buffer(1)]])
{
    RemuDrawPrimitivesIndirectArguments arguments = source[0];
    // AMD rectangle lists consume three guest vertices. REmu's Metal rectangle path emits the
    // generated fourth corner as a triangle strip, so only the native vertex count changes.
    // Preserve zero and all unexpected counts instead of manufacturing work from stale bytes.
    if (arguments.vertex_count == 3u) {
        arguments.vertex_count = 4u;
    }
    destination[0] = arguments;
}
";

fn compile_rectangle_indirect_pipeline(device: &Device) -> Result<ComputePipelineState, String> {
    let library = compile_metal_compute_library(device, RECTANGLE_INDIRECT_ARGUMENT_SOURCE)?;
    let function = library
        .get_function("remu_expand_rectangle_indirect", None)
        .map_err(|error| format!("Metal rectangle indirect function: {error}"))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| format!("Metal rectangle indirect pipeline: {error}"))
}

impl MetalPresenter {
    /// Converts one live AMD rectangle-list indirect record into Metal's native draw ABI.
    ///
    /// The source can be a pending GPU writeback from an earlier encoder. Encoding this kernel in
    /// the same guest command buffer keeps that producer ordered without reading its count through
    /// stale CPU backing. The destination lives in the batch arena until Metal has consumed it.
    pub(super) fn convert_rectangle_indirect_arguments(
        &mut self,
        device: &Device,
        source: &UploadedGlobalBuffer,
    ) -> Result<UploadedGlobalBuffer, String> {
        if self.rectangle_indirect_pipeline.is_none() {
            self.rectangle_indirect_pipeline = Some(compile_rectangle_indirect_pipeline(device)?);
        }
        let pipeline = self
            .rectangle_indirect_pipeline
            .as_ref()
            .expect("rectangle indirect pipeline was initialized")
            .clone();
        let (destination, destination_offset, ended_render_pass) = {
            let batch = self
                .active_open_guest_batch_mut()
                .ok_or("rectangle indirect conversion has no open guest batch")?;
            let (destination, destination_offset) = batch.upload_global_bytes(device, &[0; 16])?;
            batch.end_compute_pass();
            let ended_render_pass = batch.end_render_pass();
            let encoder = batch.command_buffer.new_compute_command_encoder();
            encoder.set_label("REMu rectangle indirect argument conversion");
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&source.buffer), source.offset);
            encoder.set_buffer(1, Some(&destination), destination_offset);
            encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            encoder.end_encoding();
            (destination, destination_offset, ended_render_pass)
        };
        if ended_render_pass {
            self.statistics.render_pass_ends_for_compute = self
                .statistics
                .render_pass_ends_for_compute
                .saturating_add(1);
        }
        Ok(UploadedGlobalBuffer {
            buffer: destination,
            offset: destination_offset,
            guest_memory: false,
            physical_identity: None,
            original_version: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{RECTANGLE_INDIRECT_ARGUMENT_SOURCE, compile_rectangle_indirect_pipeline};
    use metal::{Device, MTLResourceOptions, MTLSize};

    #[test]
    fn rectangle_indirect_conversion_runs_after_a_gpu_producer_and_preserves_the_abi() {
        let device = Device::system_default().expect("Metal device");
        let queue = device.new_command_queue();
        let setup_source = r"
#include <metal_stdlib>
using namespace metal;
kernel void write_arguments(device uint4* output [[buffer(0)]]) {
    output[0] = uint4(3u, 7u, 11u, 13u);
    output[1] = uint4(0u, 17u, 19u, 23u);
    output[2] = uint4(6u, 29u, 31u, 37u);
}
";
        let setup_library = device
            .new_library_with_source(setup_source, &metal::CompileOptions::new())
            .expect("producer MSL compiles");
        let setup_function = setup_library
            .get_function("write_arguments", None)
            .expect("producer function exists");
        let setup_pipeline = device
            .new_compute_pipeline_state_with_function(&setup_function)
            .expect("producer pipeline compiles");
        let conversion_pipeline =
            compile_rectangle_indirect_pipeline(&device).expect("conversion pipeline compiles");
        let source = device.new_buffer(48, MTLResourceOptions::StorageModeShared);
        let destination = device.new_buffer(48, MTLResourceOptions::StorageModeShared);
        let command = queue.new_command_buffer().to_owned();

        let producer = command.new_compute_command_encoder();
        producer.set_compute_pipeline_state(&setup_pipeline);
        producer.set_buffer(0, Some(&source), 0);
        producer.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        producer.end_encoding();

        for offset in [0, 16, 32] {
            let conversion = command.new_compute_command_encoder();
            conversion.set_compute_pipeline_state(&conversion_pipeline);
            conversion.set_buffer(0, Some(&source), offset);
            conversion.set_buffer(1, Some(&destination), offset);
            conversion.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            conversion.end_encoding();
        }
        command.commit();
        command.wait_until_completed();

        // The shared destination is 48 bytes and the completed kernels wrote three contiguous
        // uint4 records.
        let converted_bytes = super::super::super::MetalBytes::new(&destination);
        let converted = converted_bytes
            .typed_slice::<u32>(0, 12)
            .expect("destination exposes 12 u32 records");
        assert_eq!(converted, &[4, 7, 11, 13, 0, 17, 19, 23, 6, 29, 31, 37]);
        assert!(RECTANGLE_INDIRECT_ARGUMENT_SOURCE.contains("vertex_count == 3u"));
    }
}
