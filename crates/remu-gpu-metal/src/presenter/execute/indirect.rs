// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::super::{
    MetalPresenter, NGG_MESHLET_WORDS_PER_GROUP, NggSubgroupLimits, UploadedGlobalBuffer,
    compile_metal_compute_library,
};
use metal::{Buffer, ComputePipelineState, Device, MTLSize};
use std::ffi::c_void;

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

const NGG_INDIRECT_ARGUMENT_SOURCE: &str = r"
#include <metal_stdlib>
using namespace metal;

inline uint remu_read_index(device const uchar* source, uint element, uint index_size)
{
    if (index_size == 0u) {
        return uint(((device const ushort*)source)[element]);
    }
    return ((device const uint*)source)[element];
}

kernel void remu_build_indirect_ngg(
    device const uint* arguments [[buffer(0)]],
    device const uchar* index_source [[buffer(1)]],
    device uint* meshlets [[buffer(2)]],
    device uint* mesh_draw [[buffer(3)]],
    device uint* mesh_dispatch [[buffer(4)]],
    constant uint* config [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    const bool indexed = config[0] != 0u;
    const uint index_size = config[1];
    const uint topology = config[2];
    const uint maximum_input_count = config[3];
    const uint primitives_per_group = config[4];
    const uint maximum_group_count = config[5];
    const uint first = arguments[2];
    const uint available = indexed
        ? (first < maximum_input_count ? maximum_input_count - first : 0u)
        : maximum_input_count;
    const uint input_count = min(arguments[0], available);
    const uint primitive_count = topology == 4u
        ? input_count / 3u
        : (topology == 6u ? (input_count > 2u ? input_count - 2u : 0u) : input_count);
    const uint group_count = min(
        (primitive_count + primitives_per_group - 1u) / primitives_per_group,
        maximum_group_count);

    if (tid == 0u) {
        mesh_dispatch[0] = group_count;
        mesh_dispatch[1] = arguments[1];
        mesh_dispatch[2] = 1u;
        mesh_draw[0] = 0u;
        mesh_draw[1] = 0u;
        mesh_draw[2] = indexed ? arguments[3] : 0u;
        mesh_draw[3] = indexed ? arguments[4] : arguments[3];
    }

    const uint group = tid / 514u;
    const uint word = tid - group * 514u;
    if (group >= maximum_group_count) {
        return;
    }
    const uint first_primitive = group * primitives_per_group;
    const uint group_primitives = first_primitive < primitive_count
        ? min(primitives_per_group, primitive_count - first_primitive)
        : 0u;
    const bool points = topology == 1u;
    const uint group_vertices = points ? group_primitives : group_primitives * 3u;
    uint value = 0u;
    if (word == 0u) {
        value = (group_vertices & 0xffu) | ((group_primitives & 0xffu) << 8u);
    } else if (word == 1u) {
        value = first_primitive;
    } else if (word < 258u) {
        const uint local_vertex = word - 2u;
        if (local_vertex < group_vertices) {
            const uint primitive_in_group = points ? local_vertex : local_vertex / 3u;
            uint component = points ? 0u : local_vertex - primitive_in_group * 3u;
            const uint primitive = first_primitive + primitive_in_group;
            if (topology == 6u && (primitive & 1u) != 0u && component < 2u) {
                component = 1u - component;
            }
            const uint source_element = topology == 4u
                ? primitive * 3u + component
                : (topology == 6u ? primitive + component : primitive);
            value = indexed
                ? remu_read_index(index_source, first + source_element, index_size)
                : first + source_element;
        }
    } else {
        const uint primitive_in_group = word - 258u;
        if (primitive_in_group < group_primitives) {
            const uint local = points ? primitive_in_group : primitive_in_group * 3u;
            value = points
                ? local | (local << 10u) | (local << 20u)
                : local | ((local + 1u) << 10u) | ((local + 2u) << 20u);
        } else {
            value = 0x80000000u;
        }
    }
    meshlets[group * 514u + word] = value;
}
";

#[derive(Clone)]
pub(super) struct PreparedIndirectNgg {
    pub(super) meshlets: Buffer,
    pub(super) meshlets_offset: u64,
    pub(super) mesh_draw: Buffer,
    pub(super) mesh_draw_offset: u64,
    pub(super) mesh_dispatch: Buffer,
    pub(super) mesh_dispatch_offset: u64,
}

fn compile_rectangle_indirect_pipeline(device: &Device) -> Result<ComputePipelineState, String> {
    let library = compile_metal_compute_library(device, RECTANGLE_INDIRECT_ARGUMENT_SOURCE)?;
    let function = library
        .get_function("remu_expand_rectangle_indirect", None)
        .map_err(|error| format!("Metal rectangle indirect function: {error}"))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| format!("Metal rectangle indirect pipeline: {error}"))
}

fn compile_ngg_indirect_pipeline(device: &Device) -> Result<ComputePipelineState, String> {
    let library = compile_metal_compute_library(device, NGG_INDIRECT_ARGUMENT_SOURCE)?;
    let function = library
        .get_function("remu_build_indirect_ngg", None)
        .map_err(|error| format!("Metal indirect NGG function: {error}"))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| format!("Metal indirect NGG pipeline: {error}"))
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

    /// Builds the meshlet records and native indirect mesh-dispatch arguments after the guest's
    /// ordered GPU producer, without observing stale CPU copies of either draw or index data.
    pub(super) fn prepare_indirect_ngg(
        &mut self,
        device: &Device,
        arguments: &UploadedGlobalBuffer,
        index_source: Option<&UploadedGlobalBuffer>,
        indexed: bool,
        index_size: u32,
        topology: u32,
        maximum_input_count: u32,
        limits: NggSubgroupLimits,
    ) -> Result<PreparedIndirectNgg, String> {
        if self.ngg_indirect_pipeline.is_none() {
            self.ngg_indirect_pipeline = Some(compile_ngg_indirect_pipeline(device)?);
        }
        let triangle_topology = matches!(topology, 4 | 6);
        if !matches!(topology, 1 | 4 | 6) {
            return Err(format!(
                "indirect NGG topology {topology:#x} is unsupported"
            ));
        }
        if indexed && !matches!(index_size, 0 | 1) {
            return Err(format!(
                "indirect NGG index size {index_size} is unsupported"
            ));
        }
        let primitives_per_group = if triangle_topology {
            limits
                .gs_primitives_per_subgroup
                .min(limits.es_vertices_per_subgroup / 3)
        } else {
            limits
                .gs_primitives_per_subgroup
                .min(limits.es_vertices_per_subgroup)
        };
        if primitives_per_group == 0 {
            return Err("indirect NGG subgroup cannot hold one input primitive".to_owned());
        }
        let maximum_primitive_count = match topology {
            4 => maximum_input_count / 3,
            6 => maximum_input_count.saturating_sub(2),
            1 => maximum_input_count,
            _ => unreachable!(),
        };
        let maximum_group_count = maximum_primitive_count
            .div_ceil(primitives_per_group)
            .max(1);
        let meshlet_byte_count = usize::try_from(maximum_group_count)
            .ok()
            .and_then(|groups| groups.checked_mul(NGG_MESHLET_WORDS_PER_GROUP))
            .and_then(|words| words.checked_mul(std::mem::size_of::<u32>()))
            .ok_or("indirect NGG meshlet allocation size overflows")?;
        let pipeline = self
            .ngg_indirect_pipeline
            .as_ref()
            .expect("indirect NGG pipeline was initialized")
            .clone();
        let config = [
            u32::from(indexed),
            index_size,
            topology,
            maximum_input_count,
            primitives_per_group,
            maximum_group_count,
        ];
        let (
            meshlets,
            meshlets_offset,
            mesh_draw,
            mesh_draw_offset,
            mesh_dispatch,
            mesh_dispatch_offset,
            ended_render_pass,
        ) = {
            let batch = self
                .active_open_guest_batch_mut()
                .ok_or("indirect NGG preparation has no open guest batch")?;
            let (meshlets, meshlets_offset) =
                batch.allocate_global_bytes(device, meshlet_byte_count)?;
            let (mesh_draw, mesh_draw_offset) = batch.allocate_global_bytes(device, 16)?;
            let (mesh_dispatch, mesh_dispatch_offset) = batch.allocate_global_bytes(device, 12)?;
            batch.end_compute_pass();
            let ended_render_pass = batch.end_render_pass();
            let encoder = batch.command_buffer.new_compute_command_encoder();
            encoder.set_label("REMu indirect NGG meshlet preparation");
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&arguments.buffer), arguments.offset);
            let index_source = index_source.unwrap_or(arguments);
            encoder.set_buffer(1, Some(&index_source.buffer), index_source.offset);
            encoder.set_buffer(2, Some(&meshlets), meshlets_offset);
            encoder.set_buffer(3, Some(&mesh_draw), mesh_draw_offset);
            encoder.set_buffer(4, Some(&mesh_dispatch), mesh_dispatch_offset);
            encoder.set_bytes(
                5,
                u64::try_from(std::mem::size_of_val(&config))
                    .expect("indirect NGG config size fits u64"),
                config.as_ptr().cast::<c_void>(),
            );
            let threads = u64::from(maximum_group_count)
                .checked_mul(
                    u64::try_from(NGG_MESHLET_WORDS_PER_GROUP)
                        .expect("meshlet record size fits u64"),
                )
                .ok_or("indirect NGG preparation grid overflows")?;
            encoder.dispatch_threads(MTLSize::new(threads, 1, 1), MTLSize::new(256, 1, 1));
            encoder.end_encoding();
            (
                meshlets,
                meshlets_offset,
                mesh_draw,
                mesh_draw_offset,
                mesh_dispatch,
                mesh_dispatch_offset,
                ended_render_pass,
            )
        };
        if ended_render_pass {
            self.statistics.render_pass_ends_for_compute = self
                .statistics
                .render_pass_ends_for_compute
                .saturating_add(1);
        }
        Ok(PreparedIndirectNgg {
            meshlets,
            meshlets_offset,
            mesh_draw,
            mesh_draw_offset,
            mesh_dispatch,
            mesh_dispatch_offset,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        NGG_INDIRECT_ARGUMENT_SOURCE, RECTANGLE_INDIRECT_ARGUMENT_SOURCE,
        compile_ngg_indirect_pipeline, compile_rectangle_indirect_pipeline,
    };
    use metal::{Device, MTLResourceOptions, MTLSize};
    use std::ffi::c_void;

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

    #[test]
    fn indexed_indirect_ngg_builds_live_meshlets_and_dispatch_arguments_on_gpu() {
        let device = Device::system_default().expect("Metal device");
        let queue = device.new_command_queue();
        let pipeline = compile_ngg_indirect_pipeline(&device).expect("NGG pipeline compiles");
        let arguments = device.new_buffer(20, MTLResourceOptions::StorageModeShared);
        let indices = device.new_buffer(12, MTLResourceOptions::StorageModeShared);
        let meshlets = device.new_buffer(514 * 4, MTLResourceOptions::StorageModeShared);
        let mesh_draw = device.new_buffer(16, MTLResourceOptions::StorageModeShared);
        let dispatch = device.new_buffer(12, MTLResourceOptions::StorageModeShared);
        let argument_words = [6_u32, 5, 0, (-3_i32).cast_unsigned(), 13];
        let index_words = [7_u16, 8, 9, 9, 8, 10];
        let argument_bytes =
            unsafe { std::slice::from_raw_parts(argument_words.as_ptr().cast::<u8>(), 20) };
        let index_bytes =
            unsafe { std::slice::from_raw_parts(index_words.as_ptr().cast::<u8>(), 12) };
        assert!(super::super::super::MetalBytes::new(&arguments).write(0, argument_bytes));
        assert!(super::super::super::MetalBytes::new(&indices).write(0, index_bytes));
        let config = [1_u32, 0, 4, 6, 2, 1];
        let command = queue.new_command_buffer().to_owned();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&arguments), 0);
        encoder.set_buffer(1, Some(&indices), 0);
        encoder.set_buffer(2, Some(&meshlets), 0);
        encoder.set_buffer(3, Some(&mesh_draw), 0);
        encoder.set_buffer(4, Some(&dispatch), 0);
        encoder.set_bytes(
            5,
            u64::try_from(std::mem::size_of_val(&config)).expect("config size"),
            config.as_ptr().cast::<c_void>(),
        );
        encoder.dispatch_threads(MTLSize::new(514, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();

        let meshlet_words = super::super::super::MetalBytes::new(&meshlets);
        let meshlet_words = meshlet_words
            .typed_slice::<u32>(0, 514)
            .expect("meshlet output");
        assert_eq!(meshlet_words[0], 6 | (2 << 8));
        assert_eq!(meshlet_words[1], 0);
        assert_eq!(&meshlet_words[2..8], &[7, 8, 9, 9, 8, 10]);
        assert_eq!(meshlet_words[258], 1 << 10 | 2 << 20);
        assert_eq!(meshlet_words[259], 3 | 4 << 10 | 5 << 20);
        assert_eq!(meshlet_words[260], 0x8000_0000);
        let draw_words = super::super::super::MetalBytes::new(&mesh_draw);
        assert_eq!(
            draw_words.typed_slice::<u32>(0, 4).expect("mesh draw"),
            &[0, 0, (-3_i32).cast_unsigned(), 13]
        );
        let dispatch_words = super::super::super::MetalBytes::new(&dispatch);
        assert_eq!(
            dispatch_words
                .typed_slice::<u32>(0, 3)
                .expect("mesh dispatch"),
            &[1, 5, 1]
        );
        assert!(NGG_INDIRECT_ARGUMENT_SOURCE.contains("mesh_dispatch[0] = group_count"));
    }
}
