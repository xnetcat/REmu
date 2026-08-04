// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Ordered guest shader execution on a native Direct3D 12 queue.

use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use remu_gpu::{ComputeDispatch, ComputeDispatchSource, ComputeGrid, IndexedDraw};
use remu_memory::{
    CpuWriteWatch, GuestRangeLease, GuestSharedMappingIdentity, GuestWriteSite,
    LeasePublishOutcome, guest_shared_backing_identity,
};
use remu_shader::{Program, ShaderEvaluation};
use remu_shader_metal::{HlslEvaluationKey, HlslShader, MetalCompileOptions};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::{
    D3D_PRIMITIVE_TOPOLOGY_LINELIST, D3D_PRIMITIVE_TOPOLOGY_LINELIST_ADJ,
    D3D_PRIMITIVE_TOPOLOGY_LINESTRIP, D3D_PRIMITIVE_TOPOLOGY_LINESTRIP_ADJ,
    D3D_PRIMITIVE_TOPOLOGY_POINTLIST, D3D_PRIMITIVE_TOPOLOGY_TRIANGLEFAN,
    D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST_ADJ,
    D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP, D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP_ADJ, ID3DBlob,
};
use windows::Win32::Graphics::Direct3D12::{
    D3D_ROOT_SIGNATURE_VERSION_1, D3D12_BLEND_DESC, D3D12_BUFFER_SRV, D3D12_BUFFER_SRV_FLAG_NONE,
    D3D12_BUFFER_SRV_FLAG_RAW, D3D12_BUFFER_UAV, D3D12_BUFFER_UAV_FLAG_RAW,
    D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC, D3D12_COMMAND_QUEUE_FLAG_NONE,
    D3D12_COMPARISON_FUNC_ALWAYS, D3D12_COMPUTE_PIPELINE_STATE_DESC,
    D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF, D3D12_CULL_MODE_NONE,
    D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, D3D12_DEPTH_STENCIL_DESC,
    D3D12_DEPTH_STENCIL_VIEW_DESC, D3D12_DEPTH_STENCILOP_DESC, D3D12_DEPTH_WRITE_MASK_ALL,
    D3D12_DEPTH_WRITE_MASK_ZERO, D3D12_DESCRIPTOR_HEAP_DESC, D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
    D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
    D3D12_DESCRIPTOR_HEAP_TYPE_RTV, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
    D3D12_DSV_DIMENSION_TEXTURE2D, D3D12_DSV_DIMENSION_TEXTURE2DMS, D3D12_DSV_FLAG_NONE,
    D3D12_FEATURE_DATA_FORMAT_SUPPORT, D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS,
    D3D12_FEATURE_FORMAT_SUPPORT, D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, D3D12_FENCE_FLAG_NONE,
    D3D12_FILL_MODE_SOLID, D3D12_FORMAT_SUPPORT1_TEXTURE2D, D3D12_FORMAT_SUPPORT1_TEXTURE3D,
    D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW, D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE,
    D3D12_GRAPHICS_PIPELINE_STATE_DESC, D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES,
    D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_TYPE_UPLOAD,
    D3D12_INDEX_BUFFER_STRIP_CUT_VALUE, D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF,
    D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED, D3D12_INDEX_BUFFER_VIEW, D3D12_LOGIC_OP_NOOP,
    D3D12_MESSAGE, D3D12_MIP_LOD_BIAS_MAX, D3D12_MIP_LOD_BIAS_MIN,
    D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE, D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE, D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT,
    D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, D3D12_RASTERIZER_DESC,
    D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT, D3D12_RENDER_TARGET_BLEND_DESC, D3D12_RESOLVE_MODE_MAX,
    D3D12_RESOLVE_MODE_MIN, D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0,
    D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_BARRIER_FLAG_NONE,
    D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_BUFFER,
    D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_DIMENSION_TEXTURE3D,
    D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE,
    D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET,
    D3D12_RESOURCE_STATE_RESOLVE_DEST, D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_TRANSITION_BARRIER, D3D12_ROOT_CONSTANTS,
    D3D12_ROOT_PARAMETER, D3D12_ROOT_PARAMETER_0, D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
    D3D12_ROOT_SIGNATURE_DESC, D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED,
    D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED, D3D12_SAMPLER_DESC,
    D3D12_SHADER_BYTECODE, D3D12_SHADER_RESOURCE_VIEW_DESC, D3D12_SHADER_RESOURCE_VIEW_DESC_0,
    D3D12_SHADER_VISIBILITY_ALL, D3D12_SRV_DIMENSION_BUFFER, D3D12_SRV_DIMENSION_TEXTURE2D,
    D3D12_SRV_DIMENSION_TEXTURE2DARRAY, D3D12_SRV_DIMENSION_TEXTURE2DMS,
    D3D12_SRV_DIMENSION_TEXTURE3D, D3D12_STENCIL_OP_DECR, D3D12_STENCIL_OP_DECR_SAT,
    D3D12_STENCIL_OP_INCR, D3D12_STENCIL_OP_INCR_SAT, D3D12_STENCIL_OP_INVERT,
    D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_REPLACE, D3D12_STENCIL_OP_ZERO,
    D3D12_SUBRESOURCE_FOOTPRINT, D3D12_TEX2D_ARRAY_SRV, D3D12_TEX2D_ARRAY_UAV, D3D12_TEX2D_SRV,
    D3D12_TEX2D_UAV, D3D12_TEX2DMS_SRV, D3D12_TEX3D_SRV, D3D12_TEX3D_UAV,
    D3D12_TEXTURE_COPY_LOCATION, D3D12_TEXTURE_COPY_LOCATION_0,
    D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
    D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_UAV_DIMENSION_BUFFER,
    D3D12_UAV_DIMENSION_TEXTURE2D, D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
    D3D12_UAV_DIMENSION_TEXTURE3D, D3D12_UNORDERED_ACCESS_VIEW_DESC,
    D3D12_UNORDERED_ACCESS_VIEW_DESC_0, D3D12_VIEWPORT, D3D12SerializeRootSignature,
    ID3D12CommandAllocator, ID3D12CommandList, ID3D12CommandQueue, ID3D12DescriptorHeap,
    ID3D12Device, ID3D12Fence, ID3D12GraphicsCommandList, ID3D12GraphicsCommandList1,
    ID3D12InfoQueue, ID3D12PipelineState, ID3D12Resource, ID3D12RootSignature,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, DXGI_FORMAT_R8G8B8A8_UNORM,
    DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R32_FLOAT,
    DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS, DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R32_UINT,
    DXGI_FORMAT_R32G8X24_TYPELESS, DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_X32_TYPELESS_G8X24_UINT,
    DXGI_SAMPLE_DESC,
};
use windows::core::{BOOL, Interface};

use crate::backend::backend_error;
#[cfg(test)]
use crate::backend::create_default_device;
use crate::context::{read_readback, wait_for_fence, write_upload, write_upload_at};
use crate::diagnostics::{device_removed_report, name_object};
use crate::guest_shader::{compile_guest_shader, translate_guest_shader};
use crate::process_memory::read_process_memory;
#[cfg(test)]
use crate::texture::write_texture_mip_chain;
use crate::texture::{
    DecodedTexture, decode_dxgi_format, decode_texture, encode_texture_linear_over,
    read_texture_linear, read_texture_mip_chain_with_raw, write_texture_linear,
};

const MAXIMUM_GLOBAL_BUFFER_BYTES: usize = 512 * 1024 * 1024;
const MAXIMUM_SHADER_CACHE_ENTRIES: usize = 256;
const MAXIMUM_TRANSLATED_SHADER_CACHE_ENTRIES: usize = 256;
const MAXIMUM_GRAPHICS_PIPELINE_CACHE_ENTRIES: usize = 512;
const MAXIMUM_COMPUTE_PIPELINE_CACHE_ENTRIES: usize = 256;
const MAXIMUM_PERSISTENT_RENDER_TARGETS: usize = 8;
const MAXIMUM_PENDING_DRAWS: usize = 128;
const MAXIMUM_RESIDENT_SAMPLED_TEXTURES: usize = 128;
const MAXIMUM_ORDERED_GLOBAL_VERSION_BYTES: usize = 8 * 1024 * 1024;
const MAXIMUM_ORDERED_GLOBAL_VERSIONS: usize = 128;
const MAXIMUM_ORDERED_GLOBAL_RETAINED_BYTES: usize = 64 * 1024 * 1024;
const D3D12_MAX_DISPATCH_GROUPS_PER_DIMENSION: u32 = 65_535;
const TEXTURE_WRITEBACK_CHUNK_BYTES: usize = 1024 * 1024;
const SHARED_DESCRIPTOR_CAPACITY: usize = 65_536;
const SHARED_SAMPLER_CAPACITY: usize = 2_048;
const SHARED_UPLOAD_CAPACITY: usize = 64 * 1024 * 1024;

// D3D12 pipeline-state streams are alternating type/value records, each aligned like a pointer.
// This is the Rust equivalent of Microsoft's `CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT` helper.
#[repr(C, align(8))]
struct PipelineStateSubobject<T> {
    kind: windows::Win32::Graphics::Direct3D12::D3D12_PIPELINE_STATE_SUBOBJECT_TYPE,
    value: T,
}

impl<T> PipelineStateSubobject<T> {
    const fn new(
        kind: windows::Win32::Graphics::Direct3D12::D3D12_PIPELINE_STATE_SUBOBJECT_TYPE,
        value: T,
    ) -> Self {
        Self { kind, value }
    }
}

#[repr(C)]
struct IndependentStencilPipelineStream {
    root_signature: PipelineStateSubobject<*mut c_void>,
    primitive_topology:
        PipelineStateSubobject<windows::Win32::Graphics::Direct3D12::D3D12_PRIMITIVE_TOPOLOGY_TYPE>,
    index_buffer_strip_cut: PipelineStateSubobject<D3D12_INDEX_BUFFER_STRIP_CUT_VALUE>,
    vertex_shader: PipelineStateSubobject<D3D12_SHADER_BYTECODE>,
    pixel_shader: PipelineStateSubobject<D3D12_SHADER_BYTECODE>,
    blend: PipelineStateSubobject<D3D12_BLEND_DESC>,
    depth_stencil:
        PipelineStateSubobject<windows::Win32::Graphics::Direct3D12::D3D12_DEPTH_STENCIL_DESC2>,
    depth_stencil_format:
        PipelineStateSubobject<windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT>,
    rasterizer: PipelineStateSubobject<D3D12_RASTERIZER_DESC>,
    render_target_formats:
        PipelineStateSubobject<windows::Win32::Graphics::Direct3D12::D3D12_RT_FORMAT_ARRAY>,
    sample_description: PipelineStateSubobject<DXGI_SAMPLE_DESC>,
    sample_mask: PipelineStateSubobject<u32>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ShaderCacheKey {
    source: String,
    entry_point: String,
    target_profile: String,
}

#[derive(Clone)]
struct CompiledGuestShader {
    key: Arc<ShaderCacheKey>,
    dxil: Arc<[u8]>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct TranslatedShaderCacheKey {
    program_identity: usize,
    evaluation: HlslEvaluationKey,
    options: MetalCompileOptions,
    threads_per_group: [u32; 3],
}

#[derive(Clone)]
struct TranslatedGuestShader {
    // Retaining the program allocation makes its pointer identity unique for the entire lifetime
    // of the entry. A freed Arc allocation therefore cannot be reused as a false hit.
    _program: Arc<Program>,
    hlsl: HlslShader,
    compiled: CompiledGuestShader,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct GraphicsPipelineKey {
    vertex: Arc<ShaderCacheKey>,
    pixel: Option<Arc<ShaderCacheKey>>,
    topology: i32,
    index_buffer_strip_cut: i32,
    sample_count: u32,
    raster_control: u32,
    render_targets: Vec<(i32, u32, u32)>,
    blend_controls: Vec<u32>,
    depth: Option<(i32, u32, u32, u32, u32)>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct MsaaInitializationPipelineKey {
    format: i32,
    output_kind: remu_shader::PixelOutputKind,
    sample_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct DepthMsaaInitializationPipelineKey {
    format: i32,
    sample_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct IntegerMsaaResolvePipelineKey {
    format: i32,
    output_kind: remu_shader::PixelOutputKind,
}

pub(super) struct Dx12GuestExecutor {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    next_fence: u64,
    root: ID3D12RootSignature,
    independent_stencil_ref_mask: bool,
    triangle_fan_supported: bool,
    translated_shader_cache: HashMap<TranslatedShaderCacheKey, TranslatedGuestShader>,
    shader_cache: HashMap<Arc<ShaderCacheKey>, Arc<[u8]>>,
    compute_pipeline_cache: HashMap<Arc<ShaderCacheKey>, ID3D12PipelineState>,
    graphics_pipeline_cache: HashMap<GraphicsPipelineKey, ID3D12PipelineState>,
    msaa_initialization_pipeline_cache: HashMap<MsaaInitializationPipelineKey, ID3D12PipelineState>,
    depth_msaa_initialization_pipeline_cache:
        HashMap<DepthMsaaInitializationPipelineKey, ID3D12PipelineState>,
    integer_msaa_resolve_pipeline_cache:
        HashMap<IntegerMsaaResolvePipelineKey, ID3D12PipelineState>,
    render_target_cache: HashMap<u64, RenderTargetResource>,
    sampled_texture_cache: HashMap<DecodedTexture, CachedSampledTexture>,
    ordered_global_versions: OrderedGlobalVersions,
    descriptor_arena: DescriptorArena,
    upload_arena: UploadArena,
    command_pool: Vec<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>,
    pending_draw_resources: Vec<PendingDrawResources>,
}

pub(super) struct Dx12PresentTarget {
    pub(super) resource: ID3D12Resource,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
}

struct PendingDrawResources {
    _allocator: ID3D12CommandAllocator,
    _commands: ID3D12GraphicsCommandList,
    _descriptors: ID3D12DescriptorHeap,
    _samplers: Option<ID3D12DescriptorHeap>,
    _vertex_resources: ComputeResources,
    _pixel_resources: Option<ComputeResources>,
    _geometry: PreparedGeometry,
    _pipeline: ID3D12PipelineState,
}

struct DescriptorArena {
    descriptors: ID3D12DescriptorHeap,
    samplers: ID3D12DescriptorHeap,
    descriptor_cursor: usize,
    sampler_cursor: usize,
}

impl DescriptorArena {
    fn new(device: &ID3D12Device) -> Result<Self, String> {
        let descriptors = create_shader_visible_heap(
            device,
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            SHARED_DESCRIPTOR_CAPACITY,
            "shared graphics descriptors",
        )?;
        let samplers = create_shader_visible_heap(
            device,
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            SHARED_SAMPLER_CAPACITY,
            "shared graphics samplers",
        )?;
        Ok(Self {
            descriptors,
            samplers,
            descriptor_cursor: 0,
            sampler_cursor: 0,
        })
    }

    fn allocate(
        &mut self,
        descriptor_count: usize,
        sampler_count: usize,
    ) -> Option<(usize, usize)> {
        let descriptor_end = self.descriptor_cursor.checked_add(descriptor_count)?;
        let sampler_end = self.sampler_cursor.checked_add(sampler_count)?;
        if descriptor_end > SHARED_DESCRIPTOR_CAPACITY || sampler_end > SHARED_SAMPLER_CAPACITY {
            return None;
        }
        let allocation = (self.descriptor_cursor, self.sampler_cursor);
        self.descriptor_cursor = descriptor_end;
        self.sampler_cursor = sampler_end;
        Some(allocation)
    }

    fn reset(&mut self) {
        self.descriptor_cursor = 0;
        self.sampler_cursor = 0;
    }
}

struct UploadArena {
    resource: ID3D12Resource,
    cursor: usize,
}

impl UploadArena {
    fn new(device: &ID3D12Device) -> Result<Self, String> {
        Ok(Self {
            resource: create_buffer(
                device,
                SHARED_UPLOAD_CAPACITY,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_FLAG_NONE,
            )?,
            cursor: 0,
        })
    }

    fn allocate(
        &mut self,
        bytes: &[u8],
        alignment: usize,
    ) -> Result<Option<(ID3D12Resource, usize)>, String> {
        let offset = align_up(self.cursor, alignment)?;
        let end = offset
            .checked_add(bytes.len().max(4))
            .ok_or("shared upload allocation overflows")?;
        if end > SHARED_UPLOAD_CAPACITY {
            return Ok(None);
        }
        write_upload_at(&self.resource, offset, bytes).map_err(|error| error.to_string())?;
        self.cursor = end;
        Ok(Some((self.resource.clone(), offset)))
    }

    fn reset(&mut self) {
        self.cursor = 0;
    }
}

struct GlobalResource {
    gpu: ID3D12Resource,
    gpu_offset: usize,
    upload: Option<ID3D12Resource>,
    upload_offset: usize,
    readback: Option<ID3D12Resource>,
    byte_count: usize,
    writable: bool,
    original_bytes: Vec<u8>,
    writeback_guard: Option<GlobalWritebackGuard>,
}

struct OrderedGlobalVersion {
    address: u64,
    identity: Option<GuestSharedMappingIdentity>,
    original_bytes: Vec<u8>,
    current_bytes: Vec<u8>,
}

#[derive(Default)]
struct OrderedGlobalVersions {
    entries: VecDeque<OrderedGlobalVersion>,
    retained_bytes: usize,
}

impl OrderedGlobalVersions {
    fn forwarded_slice(&self, address: u64, packet_bytes: &[u8]) -> Option<&[u8]> {
        let byte_count = u64::try_from(packet_bytes.len()).ok()?;
        let end = address.checked_add(byte_count)?;
        let live_identity = guest_shared_backing_identity(address, byte_count);
        self.entries.iter().rev().find_map(|version| {
            let version_bytes = u64::try_from(version.current_bytes.len()).ok()?;
            let version_end = version.address.checked_add(version_bytes)?;
            if address < version.address || end > version_end {
                return None;
            }
            let offset = usize::try_from(address - version.address).ok()?;
            if let Some(identity) = version.identity {
                let expected = GuestSharedMappingIdentity {
                    backing_id: identity.backing_id,
                    backing_offset: identity.backing_offset.checked_add(offset as u64)?,
                };
                if live_identity != Some(expected) {
                    return None;
                }
            }
            let range = offset..offset.checked_add(packet_bytes.len())?;
            let original = version.original_bytes.get(range.clone())?;
            let current = version.current_bytes.get(range)?;
            (packet_bytes == original || packet_bytes == current).then_some(current)
        })
    }

    fn record(
        &mut self,
        address: u64,
        identity: Option<GuestSharedMappingIdentity>,
        original_bytes: &[u8],
        current_bytes: &[u8],
    ) {
        if original_bytes.len() != current_bytes.len()
            || current_bytes.is_empty()
            || current_bytes.len() > MAXIMUM_ORDERED_GLOBAL_VERSION_BYTES
        {
            return;
        }
        let Some(end) = address.checked_add(current_bytes.len() as u64) else {
            return;
        };
        let new_backing_range = identity.and_then(|identity| {
            identity
                .backing_offset
                .checked_add(current_bytes.len() as u64)
                .map(|backing_end| (identity, backing_end))
        });
        let mut retained_bytes = self.retained_bytes;
        self.entries.retain(|entry| {
            let entry_end = entry
                .address
                .saturating_add(entry.current_bytes.len() as u64);
            let virtual_overlap = address < entry_end && entry.address < end;
            let physical_overlap = new_backing_range.is_some_and(|(new_identity, new_end)| {
                entry.identity.is_some_and(|old_identity| {
                    old_identity.backing_id == new_identity.backing_id
                        && new_identity.backing_offset
                            < old_identity
                                .backing_offset
                                .saturating_add(entry.current_bytes.len() as u64)
                        && old_identity.backing_offset < new_end
                })
            });
            let keep = !virtual_overlap && !physical_overlap;
            if !keep {
                retained_bytes = retained_bytes.saturating_sub(
                    entry
                        .original_bytes
                        .len()
                        .saturating_add(entry.current_bytes.len()),
                );
            }
            keep
        });
        self.retained_bytes = retained_bytes;
        let entry_bytes = original_bytes.len().saturating_add(current_bytes.len());
        while self.entries.len() >= MAXIMUM_ORDERED_GLOBAL_VERSIONS
            || self.retained_bytes.saturating_add(entry_bytes)
                > MAXIMUM_ORDERED_GLOBAL_RETAINED_BYTES
        {
            let Some(evicted) = self.entries.pop_front() else {
                break;
            };
            self.retained_bytes = self.retained_bytes.saturating_sub(
                evicted
                    .original_bytes
                    .len()
                    .saturating_add(evicted.current_bytes.len()),
            );
        }
        self.entries.push_back(OrderedGlobalVersion {
            address,
            identity,
            original_bytes: original_bytes.to_vec(),
            current_bytes: current_bytes.to_vec(),
        });
        self.retained_bytes = self.retained_bytes.saturating_add(entry_bytes);
    }

    fn invalidate(&mut self, address: u64, byte_count: u64) {
        if byte_count == 0 {
            return;
        }
        let end = address.saturating_add(byte_count);
        let backing_range =
            guest_shared_backing_identity(address, byte_count).and_then(|identity| {
                identity
                    .backing_offset
                    .checked_add(byte_count)
                    .map(|backing_end| (identity, backing_end))
            });
        let mut retained_bytes = self.retained_bytes;
        self.entries.retain(|entry| {
            let entry_end = entry
                .address
                .saturating_add(entry.current_bytes.len() as u64);
            let virtual_overlap = address < entry_end && entry.address < end;
            let physical_overlap = backing_range.is_some_and(|(new_identity, new_end)| {
                entry.identity.is_some_and(|old_identity| {
                    old_identity.backing_id == new_identity.backing_id
                        && new_identity.backing_offset
                            < old_identity
                                .backing_offset
                                .saturating_add(entry.current_bytes.len() as u64)
                        && old_identity.backing_offset < new_end
                })
            });
            let keep = !virtual_overlap && !physical_overlap;
            if !keep {
                retained_bytes = retained_bytes.saturating_sub(
                    entry
                        .original_bytes
                        .len()
                        .saturating_add(entry.current_bytes.len()),
                );
            }
            keep
        });
        self.retained_bytes = retained_bytes;
    }
}

#[cfg_attr(test, allow(dead_code))]
enum GlobalWritebackGuard {
    Lease(GuestRangeLease),
    Anonymous(CpuWriteWatch),
    #[cfg(test)]
    Direct,
}

#[derive(Clone, Copy)]
struct WritebackJournalSites {
    live: GuestWriteSite,
    identity: GuestWriteSite,
    direct: GuestWriteSite,
}

impl WritebackJournalSites {
    const GLOBAL: Self = Self {
        live: GuestWriteSite::GlobalWriteback,
        identity: GuestWriteSite::GlobalWritebackIdentity,
        direct: GuestWriteSite::GlobalWritebackDirect,
    };

    const TEXTURE: Self = Self {
        live: GuestWriteSite::TextureWritebackRange,
        identity: GuestWriteSite::TextureWritebackRange,
        direct: GuestWriteSite::TextureWritebackRange,
    };
}

struct ComputeResources {
    binding_table: ID3D12Resource,
    binding_table_offset: usize,
    binding_table_word_count: usize,
    initial_scalars: ID3D12Resource,
    initial_scalar_offset: usize,
    initial_scalar_word_count: usize,
    globals: Vec<GlobalResource>,
    samplers: Vec<D3D12_SAMPLER_DESC>,
    textures: Vec<TextureResource>,
    binding_words: Vec<u32>,
    global_count: usize,
    texture_count: usize,
    sampler_count: usize,
}

#[derive(Clone)]
struct TextureResource {
    descriptor: DecodedTexture,
    gpu: ID3D12Resource,
    upload: ID3D12Resource,
    readback: Option<ID3D12Resource>,
    row_pitch: u32,
    slice_pitch: usize,
    copy_footprints: Vec<TextureCopyFootprint>,
    packed_byte_count: usize,
    source: TextureResourceSource,
    writeback: Option<Arc<TextureWritebackState>>,
}

struct TextureWritebackState {
    baseline: Vec<u8>,
    guard: TextureWritebackGuard,
}

enum TextureWritebackGuard {
    Lease(GuestRangeLease),
    #[cfg(not(test))]
    Anonymous(CpuWriteWatch),
    #[cfg(test)]
    Direct,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TextureResourceSource {
    GuestUpload,
    ResidentRenderTarget,
    ResidentSampledTexture,
}

struct CachedSampledTexture {
    resource: TextureResource,
    write_watch: remu_memory::CpuWriteWatch,
}

#[derive(Clone, Copy)]
struct TextureCopyFootprint {
    offset: u64,
    subresource: u32,
    format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    width: u32,
    height: u32,
    depth: u32,
    row_pitch: u32,
}

#[derive(Clone, Copy)]
struct DepthCopyPlane {
    footprint: TextureCopyFootprint,
    row_count: u32,
    row_size: usize,
}

#[derive(Clone, Copy)]
struct DecodedRenderTarget {
    guest_slot: u32,
    descriptor: DecodedTexture,
    output_kind: remu_shader::PixelOutputKind,
    write_mask: u32,
    sample_count: u32,
}

#[derive(Clone, Copy)]
struct DecodedRasterState {
    cull_mode: windows::Win32::Graphics::Direct3D12::D3D12_CULL_MODE,
    front_counter_clockwise: bool,
    fill_mode: windows::Win32::Graphics::Direct3D12::D3D12_FILL_MODE,
    discard_triangles: bool,
}

struct RenderTargetResource {
    target: DecodedRenderTarget,
    gpu: ID3D12Resource,
    resolve: Option<ID3D12Resource>,
    upload: ID3D12Resource,
    readback: ID3D12Resource,
    rtv_heap: ID3D12DescriptorHeap,
    initialization_srv_heap: Option<ID3D12DescriptorHeap>,
    integer_resolve_rtv_heap: Option<ID3D12DescriptorHeap>,
    integer_resolve_srv_heap: Option<ID3D12DescriptorHeap>,
    row_pitch: u32,
    slice_pitch: usize,
    write_watch: remu_memory::CpuWriteWatch,
    presentable_linear: Option<Vec<u8>>,
    upload_required: bool,
    pending_writeback: bool,
}

#[derive(Clone, Copy)]
struct DecodedDepthTarget {
    descriptor: DecodedTexture,
    compare: windows::Win32::Graphics::Direct3D12::D3D12_COMPARISON_FUNC,
    test_enabled: bool,
    write_enabled: bool,
    clear_enabled: bool,
    clear_depth: f32,
    sample_count: u32,
    stencil: Option<DecodedStencilTarget>,
}

#[derive(Clone, Copy)]
struct DecodedStencilFace {
    compare: windows::Win32::Graphics::Direct3D12::D3D12_COMPARISON_FUNC,
    fail: windows::Win32::Graphics::Direct3D12::D3D12_STENCIL_OP,
    depth_fail: windows::Win32::Graphics::Direct3D12::D3D12_STENCIL_OP,
    pass: windows::Win32::Graphics::Direct3D12::D3D12_STENCIL_OP,
}

#[derive(Clone, Copy)]
struct DecodedStencilTarget {
    descriptor: DecodedTexture,
    front: DecodedStencilFace,
    back: DecodedStencilFace,
    front_reference: u8,
    back_reference: u8,
    front_read_mask: u8,
    back_read_mask: u8,
    front_write_mask: u8,
    back_write_mask: u8,
    test_enabled: bool,
    clear_enabled: bool,
    clear_stencil: u8,
}

struct DepthTargetResource {
    target: DecodedDepthTarget,
    gpu: ID3D12Resource,
    resolve: Option<ID3D12Resource>,
    upload: ID3D12Resource,
    readback: ID3D12Resource,
    dsv_heap: ID3D12DescriptorHeap,
    initialization_srv_heap: Option<ID3D12DescriptorHeap>,
    copy_planes: Vec<DepthCopyPlane>,
    packed_byte_count: usize,
}

struct PreparedGeometry {
    topology: windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY,
    topology_type: windows::Win32::Graphics::Direct3D12::D3D12_PRIMITIVE_TOPOLOGY_TYPE,
    index: Option<(ID3D12Resource, D3D12_INDEX_BUFFER_VIEW)>,
    element_count: u32,
    start_vertex: u32,
    base_vertex: i32,
    strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct IndirectDrawArguments {
    element_count: u32,
    instance_count: u32,
    element_offset: u32,
    base_vertex: i32,
    first_instance: u32,
}

impl Dx12GuestExecutor {
    #[cfg(test)]
    pub(super) fn new() -> Result<Self, String> {
        let native = create_default_device().map_err(|error| error.to_string())?;
        Self::new_with_device(
            native.device,
            native.capabilities.independent_front_back_stencil_ref_mask,
            native.capabilities.triangle_fan,
        )
    }

    pub(super) fn new_with_device(
        device: ID3D12Device,
        independent_stencil_ref_mask: bool,
        triangle_fan_supported: bool,
    ) -> Result<Self, String> {
        let description = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
            ..D3D12_COMMAND_QUEUE_DESC::default()
        };
        // SAFETY: the complete descriptor requests an ordinary direct queue.
        let queue =
            unsafe { device.CreateCommandQueue(&raw const description) }.map_err(|error| {
                backend_error("ID3D12Device::CreateCommandQueue(guest)", &error).to_string()
            })?;
        // SAFETY: the fence belongs to this device and starts at zero.
        let fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.map_err(|error| {
            backend_error("ID3D12Device::CreateFence(guest)", &error).to_string()
        })?;
        let root = create_guest_root_signature(&device)?;
        name_object(&queue, "REmu guest execution queue");
        name_object(&fence, "REmu guest execution fence");
        name_object(&root, "REmu guest root signature");
        let descriptor_arena = DescriptorArena::new(&device)?;
        let upload_arena = UploadArena::new(&device)?;
        Ok(Self {
            device,
            queue,
            fence,
            next_fence: 1,
            root,
            independent_stencil_ref_mask,
            triangle_fan_supported,
            translated_shader_cache: HashMap::new(),
            shader_cache: HashMap::new(),
            compute_pipeline_cache: HashMap::new(),
            graphics_pipeline_cache: HashMap::new(),
            msaa_initialization_pipeline_cache: HashMap::new(),
            depth_msaa_initialization_pipeline_cache: HashMap::new(),
            integer_msaa_resolve_pipeline_cache: HashMap::new(),
            render_target_cache: HashMap::new(),
            sampled_texture_cache: HashMap::new(),
            ordered_global_versions: OrderedGlobalVersions::default(),
            descriptor_arena,
            upload_arena,
            command_pool: Vec::new(),
            pending_draw_resources: Vec::new(),
        })
    }

    /// Returns a DRED report only after Windows has removed the shared device.
    ///
    /// Guest translation errors are recoverable and common while support is still incomplete, so
    /// the presenter must distinguish those from a terminal native-device failure before emitting
    /// the comparatively expensive DRED traversal.
    pub(super) fn device_removed_report(&self) -> Option<String> {
        let report = device_removed_report(&self.device);
        (report != "device_state=available").then_some(report)
    }

    pub(super) fn native_present_target(
        &mut self,
        address: u64,
        width: u32,
        height: u32,
    ) -> Result<Option<Dx12PresentTarget>, String> {
        self.complete_pending_gpu_work()?;
        let Some(target) = self.render_target_cache.get(&address) else {
            return Ok(None);
        };
        let descriptor = target.target.descriptor;
        if !target.pending_writeback
            || target.target.sample_count != 1
            || descriptor.width != width
            || descriptor.height != height
            || descriptor.bytes_per_element != 4
            || descriptor.block_extent != [1, 1]
            || !matches!(
                descriptor.format,
                DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM
            )
        {
            return Ok(None);
        }
        Ok(Some(Dx12PresentTarget {
            resource: target.gpu.clone(),
            width,
            height,
            format: descriptor.format,
        }))
    }

    pub(super) fn complete_pending_gpu_work(&mut self) -> Result<(), String> {
        if self.pending_draw_resources.is_empty() {
            Ok(())
        } else {
            self.wait()
        }
    }

    pub(super) fn execute_compute(&mut self, dispatch: &ComputeDispatch) -> Result<(), String> {
        let resolved_dispatch =
            resolve_indirect_compute_ordered(dispatch, &self.ordered_global_versions)?;
        let dispatch = resolved_dispatch.as_ref().unwrap_or(dispatch);
        if dispatch.launch.is_empty() {
            return Ok(());
        }
        let snapshot = dispatch
            .shader
            .as_ref()
            .ok_or("compute dispatch has no decoded shader snapshot")?;
        let evaluation = dispatch
            .shader_evaluation
            .as_ref()
            .ok_or("compute dispatch has no evaluated shader resources")?;
        if std::env::var_os("REMU_TRACE_D3D12_BINDINGS").is_some_and(|value| value == "1") {
            let globals = evaluation
                .global_memory_bindings
                .iter()
                .map(|binding| {
                    let descriptor = (0..4)
                        .map(|index| {
                            let scalar = binding.scalar_address.saturating_add(index);
                            register_value(&dispatch.shader_registers, 0x240 + scalar).or_else(
                                || {
                                    evaluation
                                        .initial_scalar_registers
                                        .get(scalar as usize)
                                        .copied()
                                },
                            )
                        })
                        .collect::<Option<Vec<_>>>()
                        .map(|words| preview_words(&words_as_bytes(&words)))
                        .unwrap_or_else(|| "unavailable".to_owned());
                    let head = (binding.byte_count <= 64)
                        .then(|| {
                            snapshot_ordered_global(binding, &self.ordered_global_versions).ok()
                        })
                        .flatten()
                        .map(|bytes| preview_words(&bytes))
                        .unwrap_or_default();
                    format!(
                        "{:#x}/{}:{}{}:s{}=[{}]{}",
                        binding.base_address,
                        binding.byte_count,
                        if binding.writable { "w" } else { "r" },
                        if binding.write_back_to_guest { "b" } else { "" },
                        binding.scalar_address,
                        descriptor,
                        if head.is_empty() {
                            String::new()
                        } else {
                            format!(":[{head}]")
                        },
                    )
                })
                .collect::<Vec<_>>();
            let images = evaluation
                .image_bindings
                .iter()
                .map(|binding| {
                    let sampler = if binding.sampler_descriptor.len() >= 4 {
                        decode_sampler(binding)
                            .map(|sampler| format!("{sampler:?}"))
                            .unwrap_or_else(|error| format!("decode-error({error})"))
                    } else {
                        "none".to_owned()
                    };
                    decode_texture(binding)
                        .map(|texture| format!("{texture:?} sampler={sampler}"))
                        .unwrap_or_else(|error| format!("decode-error({error})"))
                })
                .collect::<Vec<_>>();
            eprintln!(
                "[REMu][D3D12-BINDINGS] compute={} shader={:#x} grid={:?} local={:?} globals={globals:?} images={images:?}",
                dispatch.sequence,
                dispatch.shader_address,
                dispatch.launch.grid,
                dispatch.launch.threads_per_group,
            );
        }
        let total_threads = dispatch.launch.threads_per_group_total();
        if total_threads == 0 || total_threads > 1_024 {
            return Err(format!(
                "compute threadgroup {:?} has {total_threads} threads; D3D12 permits 1..=1024",
                dispatch.launch.threads_per_group
            ));
        }
        // Dispatch returns void. Passing a count beyond the native limit removes the entire
        // D3D12 device, so reject stale or partially produced indirect arguments before compiling
        // shaders or creating resource views.
        let group_count = dispatch_group_count(dispatch)?;
        let mut options = MetalCompileOptions::compute_wave(
            dispatch.launch.lds_size_bytes,
            dispatch.launch.wave_lane_count,
        );
        options.allocated_vector_register_count = allocated_compute_vgprs(dispatch);
        let (_, compiled, _) = self.translate_shader_cached(
            &snapshot.program,
            evaluation,
            &options,
            dispatch.launch.threads_per_group,
            &format!("remu-cs-{:x}.hlsl", dispatch.shader_address),
        )?;
        let pipeline = self.compute_pipeline_cached(&compiled)?;
        ensure_device_available(&self.device, "compute pipeline creation")?;
        let mut resources = create_stage_resources(
            &self.device,
            &mut self.upload_arena,
            evaluation,
            Some((&dispatch.shader_registers, 0x240, 0)),
            &self.render_target_cache,
            &mut self.sampled_texture_cache,
            &[],
            &self.ordered_global_versions,
        )?;
        ensure_device_available(&self.device, "compute resource creation")?;
        let (descriptors, samplers, root_descriptors) = combine_stage_heaps(
            &self.device,
            &mut self.descriptor_arena,
            &mut [&mut resources],
        )?;
        ensure_device_available(&self.device, "compute descriptor creation")?;

        // SAFETY: allocator and command list types match this executor's direct queue.
        let allocator: ID3D12CommandAllocator = unsafe {
            self.device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
        }
        .map_err(|error| {
            backend_error("ID3D12Device::CreateCommandAllocator(guest)", &error).to_string()
        })?;
        let commands: ID3D12GraphicsCommandList = unsafe {
            self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None::<&ID3D12PipelineState>,
            )
        }
        .map_err(|error| {
            backend_error("ID3D12Device::CreateCommandList(guest)", &error).to_string()
        })?;
        name_object(
            &commands,
            &format!(
                "REmu compute shader={:#x} fence={}",
                dispatch.shader_address, self.next_fence
            ),
        );

        record_stage_uploads(&commands, &resources);
        let mut root_constants = dispatch_root_constants(dispatch);
        root_constants[0] = root_descriptors[0];
        // SAFETY: descriptor zero is the binding table, all resources remain alive, and the root
        // signature/PSO were built from this exact translated shader interface.
        unsafe {
            let mut heaps = vec![Some(descriptors)];
            if let Some(samplers) = samplers {
                heaps.push(Some(samplers));
            }
            commands.SetDescriptorHeaps(&heaps);
            commands.SetComputeRootSignature(&self.root);
            commands.SetPipelineState(&pipeline);
            commands.SetComputeRoot32BitConstants(
                2,
                root_constants.len() as u32,
                root_constants.as_ptr().cast::<c_void>(),
                0,
            );
            commands.Dispatch(group_count[0], group_count[1], group_count[2]);
        }
        for resource in &resources.globals {
            if let Some(readback) = &resource.readback {
                record_transition(
                    &commands,
                    &resource.gpu,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                );
                // SAFETY: the transitioned source and readback buffer cover byte_count.
                unsafe {
                    commands.CopyBufferRegion(
                        readback,
                        0,
                        &resource.gpu,
                        0,
                        u64::try_from(resource.byte_count.max(4))
                            .expect("bounded buffer size fits u64"),
                    );
                }
            }
        }
        record_texture_readbacks(&commands, &resources.textures);
        // SAFETY: recording is complete and the open list has not previously been submitted.
        unsafe { commands.Close() }.map_err(|error| {
            backend_error("ID3D12GraphicsCommandList::Close(guest)", &error).to_string()
        })?;
        let list: ID3D12CommandList = commands.cast().map_err(|error| {
            backend_error("ID3D12GraphicsCommandList::cast(guest)", &error).to_string()
        })?;
        // SAFETY: every referenced resource remains owned through the following fence wait.
        unsafe { self.queue.ExecuteCommandLists(&[Some(list)]) };
        self.wait()?;
        write_back_globals(
            evaluation,
            &resources.globals,
            &mut self.ordered_global_versions,
        )?;
        write_back_textures(&resources.textures)?;
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn execute_draw(&mut self, draw: &IndexedDraw) -> Result<(), String> {
        self.execute_draw_with_policy(draw, false)
    }

    pub(super) fn execute_draw_deferred(&mut self, draw: &IndexedDraw) -> Result<(), String> {
        self.execute_draw_with_policy(draw, true)
    }

    fn execute_draw_with_policy(
        &mut self,
        draw: &IndexedDraw,
        defer_render_target_writeback: bool,
    ) -> Result<(), String> {
        let timing_started = Instant::now();
        let trace_timing =
            std::env::var_os("REMU_TRACE_D3D12_TIMING").is_some_and(|value| value == "1");
        if self.pending_draw_resources.len() >= MAXIMUM_PENDING_DRAWS {
            self.complete_pending_gpu_work()?;
        }
        if !indirect_draw_count_allows(draw)? {
            return Ok(());
        }
        let resolved_draw = draw
            .indirect_arguments_address
            .map(|address| resolve_indirect_draw(draw, address))
            .transpose()?;
        let first_instance = resolved_draw
            .as_ref()
            .map_or(0, |(_, arguments)| arguments.first_instance);
        let draw = resolved_draw.as_ref().map_or(draw, |(draw, _)| draw);
        if draw.index_count == 0 || draw.instance_count == 0 {
            return Ok(());
        }
        let export = draw
            .export_shader
            .as_ref()
            .ok_or("draw has no decoded export shader")?;
        let export_evaluation = draw
            .export_shader_evaluation
            .as_ref()
            .ok_or("draw has no evaluated export resources")?;
        let decoded_targets = decode_render_targets(draw)?;
        let decoded_depth = decode_depth_target(draw)?;
        if decoded_targets.is_empty() && decoded_depth.is_none() {
            // Attachment-less synchronization draws have no observable raster output.
            return Ok(());
        }
        let sample_count = render_sample_count(&decoded_targets, decoded_depth)?;
        let primary_extent = decoded_targets
            .first()
            .map(|target| (target.descriptor.width, target.descriptor.height))
            .or_else(|| {
                decoded_depth.map(|depth| (depth.descriptor.width, depth.descriptor.height))
            })
            .expect("a color or depth target supplies the render extent");
        let pixel_stage = match (&draw.pixel_shader, &draw.pixel_shader_evaluation) {
            (Some(pixel), Some(evaluation)) => {
                let mut options = MetalCompileOptions::pixel(
                    decoded_targets
                        .iter()
                        .enumerate()
                        .map(|(host_location, target)| remu_shader::PixelOutputBinding {
                            guest_slot: target.guest_slot,
                            host_location: host_location as u32,
                            kind: target.output_kind,
                        })
                        .collect(),
                    register_value(&draw.context_registers, 0x1b3).unwrap_or_default(),
                    register_value(&draw.context_registers, 0x1b4).unwrap_or_default(),
                );
                options.pixel_input_controls = std::array::from_fn(|index| {
                    remu_shader_metal::PixelInputControl::from_register(
                        register_value(&draw.context_registers, 0x191 + index as u32)
                            .unwrap_or(index as u32),
                    )
                });
                options.pixel_depth_export = decoded_depth.is_some()
                    && register_value(&draw.context_registers, 0x203)
                        .is_some_and(|control| control & 1 != 0);
                options.allocated_vector_register_count =
                    graphics_allocated_vgprs(&draw.shader_registers, 0x00c);
                let (hlsl, compiled, cache_hit) = self.translate_shader_cached(
                    &pixel.program,
                    evaluation,
                    &options,
                    [1, 1, 1],
                    &format!("remu-ps-{:x}.hlsl", draw.pixel_shader_address),
                )?;
                Some((hlsl, evaluation.as_ref(), compiled, cache_hit))
            }
            (None, None) => None,
            _ => return Err("draw has an incomplete pixel shader/evaluation pair".to_owned()),
        };
        let mut vertex_options = MetalCompileOptions::vertex(
            pixel_stage
                .as_ref()
                .map_or(0, |(shader, _, _, _)| shader.interpolated_attribute_count),
        );
        vertex_options.required_vertex_output_locations = pixel_stage
            .as_ref()
            .map(|(shader, _, _, _)| shader.interpolated_attribute_locations.clone());
        vertex_options.vertex_output_control =
            register_value(&draw.context_registers, 0x207).unwrap_or_default();
        let export_user_base = [0x08c, 0x0cc, 0x04c]
            .into_iter()
            .find(|base| register_value(&draw.shader_registers, base - 1).is_some())
            .unwrap_or(0x0cc);
        vertex_options.allocated_vector_register_count =
            graphics_allocated_vgprs(&draw.shader_registers, export_user_base);
        let (_, vertex_compiled, vertex_translation_cache_hit) = self.translate_shader_cached(
            &export.program,
            export_evaluation,
            &vertex_options,
            [1, 1, 1],
            &format!("remu-vs-{:x}.hlsl", draw.export_shader_address),
        )?;
        let pixel_compiled = pixel_stage.as_ref().map(|(_, _, compiled, _)| compiled);
        let shader_translation_cache_hits = usize::from(vertex_translation_cache_hit)
            + pixel_stage
                .as_ref()
                .map_or(0, |(_, _, _, cache_hit)| usize::from(*cache_hit));
        let translation_ready = timing_started.elapsed();
        let geometry = prepare_geometry(
            &self.device,
            &mut self.upload_arena,
            draw,
            self.triangle_fan_supported,
        )?;
        let geometry_ready = timing_started.elapsed();
        let pipeline = self.graphics_pipeline_cached(
            &vertex_compiled,
            pixel_compiled,
            draw,
            &decoded_targets,
            decoded_depth,
            geometry.topology_type,
            geometry.strip_cut,
            sample_count,
        )?;
        let pipeline_ready = timing_started.elapsed();
        let output_addresses = decoded_targets
            .iter()
            .map(|target| target.descriptor.address)
            .collect::<Vec<_>>();
        let mut vertex_resources = create_stage_resources(
            &self.device,
            &mut self.upload_arena,
            export_evaluation,
            Some((&draw.shader_registers, export_user_base, 8)),
            &self.render_target_cache,
            &mut self.sampled_texture_cache,
            &output_addresses,
            &self.ordered_global_versions,
        )?;
        let vertex_resources_ready = timing_started.elapsed();
        let mut pixel_resources = pixel_stage
            .as_ref()
            .map(|(_, evaluation, _, _)| {
                create_stage_resources(
                    &self.device,
                    &mut self.upload_arena,
                    evaluation,
                    Some((&draw.shader_registers, 0x00c, 0)),
                    &self.render_target_cache,
                    &mut self.sampled_texture_cache,
                    &output_addresses,
                    &self.ordered_global_versions,
                )
            })
            .transpose()?;
        let pixel_resources_ready = timing_started.elapsed();
        let (descriptors, samplers, root_descriptors) = {
            let mut stages = vec![&mut vertex_resources];
            if let Some(resources) = &mut pixel_resources {
                stages.push(resources);
            }
            combine_stage_heaps(&self.device, &mut self.descriptor_arena, &mut stages)?
        };
        let heaps_ready = timing_started.elapsed();
        let mut targets = Vec::with_capacity(decoded_targets.len());
        for target in decoded_targets.iter().copied() {
            let cached = (target.sample_count == 1 && target.descriptor.address != 0)
                .then(|| self.render_target_cache.remove(&target.descriptor.address))
                .flatten();
            let resource = match cached {
                Some(mut resource)
                    if render_target_resource_compatible(&resource, target)
                        && !resource.write_watch.is_retired() =>
                {
                    resource.target = target;
                    let cpu_dirty = resource.write_watch.consume_dirty();
                    if resource.pending_writeback && cpu_dirty {
                        // The guest has taken ownership of this allocation after the native draw
                        // completed. Discard the now-obsolete resident image and recreate it from
                        // the CPU-authored bytes in COPY_DEST state; trying to merge two complete
                        // color surfaces would preserve neither ordering nor contents.
                        create_render_target_resource(&self.device, target)?
                    } else {
                        resource.upload_required = cpu_dirty;
                        if resource.upload_required {
                            refresh_render_target_upload(&resource)?;
                            resource.write_watch.rearm().map_err(|error| {
                                format!("could not rearm render-target watch: {error}")
                            })?;
                        }
                        resource
                    }
                }
                _ => create_render_target_resource(&self.device, target)?,
            };
            targets.push(resource);
        }
        let depth = decoded_depth
            .map(|target| create_depth_target_resource(&self.device, target))
            .transpose()?;
        let targets_defer_eligible = targets
            .iter()
            .all(|target| target.target.sample_count == 1 && target.target.descriptor.address != 0);
        let vertex_requires_writeback = stage_resources_require_writeback(&vertex_resources);
        let pixel_requires_writeback = pixel_resources
            .as_ref()
            .is_some_and(|resources| stage_resources_require_writeback(resources));
        let defer_gpu_completion = defer_render_target_writeback
            && depth.is_none()
            && targets_defer_eligible
            && !vertex_requires_writeback
            && !pixel_requires_writeback;
        let resources_ready = timing_started.elapsed();
        // SAFETY: pooled command contexts are only recycled after their fence has completed.
        let (allocator, commands) = self.acquire_command_context()?;
        name_object(
            &commands,
            &format!(
                "REmu draw sequence={} es={:#x} ps={:#x} fence={}",
                draw.sequence,
                draw.export_shader_address,
                draw.pixel_shader_address,
                self.next_fence
            ),
        );
        record_stage_uploads(&commands, &vertex_resources);
        if let Some(resources) = &pixel_resources {
            record_stage_uploads(&commands, resources);
        }
        for target in &targets {
            record_render_target_upload(&commands, target);
        }
        if let Some(depth) = &depth {
            record_depth_target_upload(&commands, depth);
        }
        for target in &targets {
            if target.target.sample_count > 1 {
                let initialization = self.msaa_initialization_pipeline_cached(target.target)?;
                record_msaa_render_target_initialization(
                    &commands,
                    &self.root,
                    &initialization,
                    target,
                )?;
            }
        }
        if let Some(depth) = &depth
            && depth.target.sample_count > 1
        {
            let initialization = self.depth_msaa_initialization_pipeline_cached(depth.target)?;
            record_depth_msaa_initialization(&commands, &self.root, &initialization, depth)?;
        }
        let scissor = decode_scissor(draw, primary_extent.0, primary_extent.1);
        let viewport = decode_viewport(draw, scissor);
        let blend_factor = [0x105, 0x106, 0x107, 0x108].map(|offset| {
            register_value(&draw.context_registers, offset).map_or(0.0, f32::from_bits)
        });
        let discard_triangles = decode_raster_state(draw)?.discard_triangles
            && geometry.topology_type == D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        let rtvs = targets
            .iter()
            .map(|target| unsafe { target.rtv_heap.GetCPUDescriptorHandleForHeapStart() })
            .collect::<Vec<_>>();
        let dsv = depth
            .as_ref()
            .map(|target| unsafe { target.dsv_heap.GetCPUDescriptorHandleForHeapStart() });
        let mut heaps = vec![Some(descriptors.clone())];
        if let Some(samplers) = &samplers {
            heaps.push(Some(samplers.clone()));
        }
        let vertex_constants = [root_descriptors[0], 0, 0, 0, 0, 0, 0];
        let pixel_constants = pixel_resources
            .as_ref()
            .map(|_| [root_descriptors[1], 0, 0, 0, 0, 0, 0]);
        // SAFETY: root constants select each stage's combined-heap table, and all fixed-function
        // state matches the PSO and live target.
        unsafe {
            commands.SetDescriptorHeaps(&heaps);
            commands.SetGraphicsRootSignature(&self.root);
            commands.SetPipelineState(&pipeline);
            commands.SetGraphicsRoot32BitConstants(0, 7, vertex_constants.as_ptr().cast(), 0);
            if let Some(pixel_constants) = &pixel_constants {
                commands.SetGraphicsRoot32BitConstants(1, 7, pixel_constants.as_ptr().cast(), 0);
            }
            commands.RSSetViewports(std::slice::from_ref(&viewport));
            commands.RSSetScissorRects(std::slice::from_ref(&scissor));
            commands.OMSetBlendFactor(Some(&blend_factor));
            commands.OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            );
            if let (Some(depth), Some(dsv)) = (&depth, dsv) {
                let mut clear_flags = windows::Win32::Graphics::Direct3D12::D3D12_CLEAR_FLAGS(0);
                if depth.target.clear_enabled {
                    clear_flags |= windows::Win32::Graphics::Direct3D12::D3D12_CLEAR_FLAG_DEPTH;
                }
                if depth
                    .target
                    .stencil
                    .is_some_and(|stencil| stencil.clear_enabled)
                {
                    clear_flags |= windows::Win32::Graphics::Direct3D12::D3D12_CLEAR_FLAG_STENCIL;
                }
                if clear_flags.0 != 0 {
                    commands.ClearDepthStencilView(
                        dsv,
                        clear_flags,
                        depth.target.clear_depth,
                        depth
                            .target
                            .stencil
                            .map_or(0, |stencil| stencil.clear_stencil),
                        None,
                    );
                }
                if let Some(stencil) = depth.target.stencil {
                    if stencil.front_reference == stencil.back_reference {
                        commands.OMSetStencilRef(u32::from(stencil.front_reference));
                    } else {
                        let commands8: windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList8 =
                            commands.cast().map_err(|error| {
                                backend_error(
                                    "ID3D12GraphicsCommandList::cast(ID3D12GraphicsCommandList8)",
                                    &error,
                                )
                                .to_string()
                            })?;
                        commands8.OMSetFrontAndBackStencilRef(
                            u32::from(stencil.front_reference),
                            u32::from(stencil.back_reference),
                        );
                    }
                }
            }
            commands.IASetPrimitiveTopology(geometry.topology);
            if discard_triangles || geometry.element_count == 0 {
                // GFX10 permits both front- and back-face culling, which discards every triangle.
            } else if let Some((_, view)) = &geometry.index {
                commands.IASetIndexBuffer(Some(view));
                commands.DrawIndexedInstanced(
                    geometry.element_count,
                    draw.instance_count,
                    0,
                    geometry.base_vertex,
                    first_instance,
                );
            } else {
                commands.DrawInstanced(
                    geometry.element_count,
                    draw.instance_count,
                    geometry.start_vertex,
                    first_instance,
                );
            }
        }
        record_stage_readbacks(&commands, &vertex_resources);
        if let Some(resources) = &pixel_resources {
            record_stage_readbacks(&commands, resources);
        }
        for target in &targets {
            if defer_render_target_writeback
                && target.target.sample_count == 1
                && target.target.descriptor.address != 0
            {
                continue;
            }
            let integer_resolve = (target.target.sample_count > 1
                && target.target.output_kind != remu_shader::PixelOutputKind::Float)
                .then(|| self.integer_msaa_resolve_pipeline_cached(target.target))
                .transpose()?;
            record_render_target_readback(
                &commands,
                &self.root,
                target,
                integer_resolve.as_ref(),
                false,
            )?;
        }
        if let Some(depth) = &depth {
            record_depth_target_readback(&commands, depth)?;
        }
        // SAFETY: command recording is complete.
        unsafe { commands.Close() }.map_err(|error| {
            backend_error("ID3D12GraphicsCommandList::Close(draw)", &error).to_string()
        })?;
        let commands_ready = timing_started.elapsed();
        let list: ID3D12CommandList = commands.cast().map_err(|error| {
            backend_error("ID3D12GraphicsCommandList::cast(draw)", &error).to_string()
        })?;
        // SAFETY: all command resources live through the following wait.
        unsafe { self.queue.ExecuteCommandLists(&[Some(list)]) };
        let gpu_complete = if defer_gpu_completion {
            commands_ready
        } else {
            self.wait()?;
            let complete = timing_started.elapsed();
            write_back_stage(
                export_evaluation,
                &vertex_resources,
                &mut self.ordered_global_versions,
            )?;
            if let (Some((_, evaluation, _, _)), Some(resources)) = (&pixel_stage, &pixel_resources)
            {
                write_back_stage(evaluation, resources, &mut self.ordered_global_versions)?;
            }
            complete
        };
        for target in &mut targets {
            if defer_render_target_writeback
                && target.target.sample_count == 1
                && target.target.descriptor.address != 0
            {
                target.pending_writeback = true;
                target.presentable_linear = None;
            } else {
                write_back_render_target(target)?;
            }
        }
        if let Some(depth) = &depth {
            write_back_depth_target(depth)?;
        }
        for target in targets {
            if target.target.sample_count != 1 || target.target.descriptor.address == 0 {
                continue;
            }
            if self.render_target_cache.len() >= MAXIMUM_PERSISTENT_RENDER_TARGETS
                && !self
                    .render_target_cache
                    .contains_key(&target.target.descriptor.address)
            {
                self.flush_all_render_targets()?;
                self.render_target_cache.clear();
            }
            self.render_target_cache
                .insert(target.target.descriptor.address, target);
        }
        if defer_gpu_completion {
            self.pending_draw_resources.push(PendingDrawResources {
                _allocator: allocator,
                _commands: commands,
                _descriptors: descriptors,
                _samplers: samplers,
                _vertex_resources: vertex_resources,
                _pixel_resources: pixel_resources,
                _geometry: geometry,
                _pipeline: pipeline,
            });
        } else {
            self.command_pool.push((allocator, commands));
        }
        if trace_timing {
            let complete = timing_started.elapsed();
            eprintln!(
                "[REMu][D3D12-TIMING] draw_target={:#x} extent={}x{} samples={} batched={} depth={} targets_defer={} vertex_writeback={} pixel_writeback={} shader_translation_hits={} translation_us={} geometry_us={} pso_us={} vertex_resources_us={} pixel_resources_us={} heaps_us={} targets_us={} pipeline_us={} resources_us={} record_us={} gpu_us={} writeback_us={} total_us={}",
                decoded_targets.first().map_or_else(
                    || decoded_depth.map_or(0, |target| target.descriptor.address),
                    |target| target.descriptor.address
                ),
                primary_extent.0,
                primary_extent.1,
                sample_count,
                u8::from(defer_gpu_completion),
                u8::from(depth.is_some()),
                u8::from(targets_defer_eligible),
                u8::from(vertex_requires_writeback),
                u8::from(pixel_requires_writeback),
                shader_translation_cache_hits,
                translation_ready.as_micros(),
                geometry_ready.saturating_sub(translation_ready).as_micros(),
                pipeline_ready.saturating_sub(geometry_ready).as_micros(),
                vertex_resources_ready
                    .saturating_sub(pipeline_ready)
                    .as_micros(),
                pixel_resources_ready
                    .saturating_sub(vertex_resources_ready)
                    .as_micros(),
                heaps_ready
                    .saturating_sub(pixel_resources_ready)
                    .as_micros(),
                resources_ready.saturating_sub(heaps_ready).as_micros(),
                pipeline_ready.as_micros(),
                resources_ready.saturating_sub(pipeline_ready).as_micros(),
                commands_ready.saturating_sub(resources_ready).as_micros(),
                gpu_complete.saturating_sub(commands_ready).as_micros(),
                complete.saturating_sub(gpu_complete).as_micros(),
                complete.as_micros(),
            );
        }
        Ok(())
    }

    pub(super) fn take_presentable_bgra(
        &mut self,
        address: u64,
        width: u32,
        height: u32,
    ) -> Option<Vec<u8>> {
        let target = self.render_target_cache.get_mut(&address)?;
        let descriptor = target.target.descriptor;
        if descriptor.width != width
            || descriptor.height != height
            || !matches!(
                descriptor.format,
                DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM
            )
            || descriptor.bytes_per_element != 4
            || descriptor.block_extent != [1, 1]
        {
            return None;
        }
        let linear = target.presentable_linear.take()?;
        let expected = usize::try_from(width)
            .ok()?
            .checked_mul(usize::try_from(height).ok()?)?
            .checked_mul(4)?;
        (linear.len() == expected)
            .then(|| presentable_bgra(descriptor.format, linear))
            .flatten()
    }

    pub(super) fn prepare_presentable_bgra(
        &mut self,
        address: u64,
        width: u32,
        height: u32,
    ) -> Result<Option<Vec<u8>>, String> {
        let Some((descriptor, pending_writeback)) = self
            .render_target_cache
            .get(&address)
            .map(|target| (target.target.descriptor, target.pending_writeback))
        else {
            return Ok(None);
        };
        if descriptor.width != width
            || descriptor.height != height
            || !matches!(
                descriptor.format,
                DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM
            )
            || descriptor.bytes_per_element != 4
            || descriptor.block_extent != [1, 1]
        {
            if std::env::var_os("REMU_TRACE_D3D12_TIMING").is_some_and(|value| value == "1") {
                eprintln!(
                    "[REMu][D3D12-TIMING] present_fallback={address:#x} requested={}x{} target={descriptor:?}",
                    width, height
                );
            }
            // An unrecognized display layout still needs ordinary guest publication so the
            // generic scanout detiler can consume it.
            self.flush_render_target(address)?;
            return Ok(self.take_presentable_bgra(address, width, height));
        }
        if !pending_writeback {
            return Ok(self.take_presentable_bgra(address, width, height));
        }

        let timing_started = Instant::now();
        let mut target = self
            .render_target_cache
            .remove(&address)
            .expect("target inspected immediately before removal");
        let mut gpu_complete = None;
        let result = (|| {
            let allocator: ID3D12CommandAllocator = unsafe {
                self.device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            }
            .map_err(|error| {
                backend_error(
                    "ID3D12Device::CreateCommandAllocator(target present)",
                    &error,
                )
                .to_string()
            })?;
            let commands: ID3D12GraphicsCommandList = unsafe {
                self.device.CreateCommandList(
                    0,
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    &allocator,
                    None::<&ID3D12PipelineState>,
                )
            }
            .map_err(|error| {
                backend_error("ID3D12Device::CreateCommandList(target present)", &error).to_string()
            })?;
            name_object(
                &commands,
                &format!("REmu target present address={address:#x}"),
            );
            record_render_target_readback(&commands, &self.root, &target, None, true)?;
            unsafe { commands.Close() }.map_err(|error| {
                backend_error("ID3D12GraphicsCommandList::Close(target present)", &error)
                    .to_string()
            })?;
            let list: ID3D12CommandList = commands.cast().map_err(|error| {
                backend_error("ID3D12GraphicsCommandList::cast(target present)", &error).to_string()
            })?;
            unsafe { self.queue.ExecuteCommandLists(&[Some(list)]) };
            self.wait()?;
            gpu_complete = Some(timing_started.elapsed());
            read_render_target_linear(&target).and_then(|linear| {
                presentable_bgra(descriptor.format, linear).ok_or_else(|| {
                    format!("unsupported presentable format {:?}", descriptor.format)
                })
            })
        })();
        // Presentation observes the native image but deliberately leaves the GPU copy
        // authoritative. A later release/submission boundary or overlapping consumer performs
        // the guest-memory publication exactly once.
        target.presentable_linear = None;
        self.render_target_cache.insert(address, target);
        if std::env::var_os("REMU_TRACE_D3D12_TIMING").is_some_and(|value| value == "1") {
            let complete = timing_started.elapsed();
            let gpu_complete = gpu_complete.unwrap_or(complete);
            eprintln!(
                "[REMu][D3D12-TIMING] present_target={address:#x} gpu_us={} readback_us={} total_us={}",
                gpu_complete.as_micros(),
                complete.saturating_sub(gpu_complete).as_micros(),
                complete.as_micros(),
            );
        }
        result.map(Some)
    }

    pub(super) fn flush_all_render_targets(&mut self) -> Result<(), String> {
        let pending = self
            .render_target_cache
            .iter()
            .filter_map(|(&address, target)| target.pending_writeback.then_some(address))
            .collect::<Vec<_>>();
        for address in pending {
            self.flush_render_target(address)?;
        }
        Ok(())
    }

    pub(super) fn pending_render_target_count(&self) -> usize {
        self.render_target_cache
            .values()
            .filter(|target| target.pending_writeback)
            .count()
    }

    /// Applies the cache-invalidation semantics of `ACQUIRE_MEM` without publishing GPU-owned
    /// render targets back to guest RAM.
    ///
    /// An acquire makes earlier CPU writes visible to later GPU resource reads; it does not make
    /// earlier GPU color writes CPU-visible. Stage resources are rebuilt from guest memory for
    /// every dispatch/draw, while persistent render targets consult their CPU write watch when
    /// they are next bound. Queue ordering and the per-command fence already order prior GPU
    /// work. Publication remains the responsibility of release/wait/submission/flip boundaries
    /// or an overlapping GPU consumer.
    pub(super) fn synchronize_memory_acquire(&mut self, acquire: remu_gpu::GpuMemoryAcquire) {
        self.ordered_global_versions
            .invalidate(acquire.base_address, acquire.size_bytes);
    }

    pub(super) fn invalidate_ordered_global_range(&mut self, address: u64, byte_count: u64) {
        self.ordered_global_versions.invalidate(address, byte_count);
    }

    pub(super) fn flush_for_draw_inputs(&mut self, draw: &IndexedDraw) -> Result<(), String> {
        let output_addresses = decode_render_targets(draw)?
            .into_iter()
            .map(|target| target.descriptor.address)
            .collect::<Vec<_>>();
        let pending = self
            .render_target_cache
            .iter()
            .filter_map(|(&address, target)| {
                (target.pending_writeback
                    && draw_requires_render_target_publication(draw, target, &output_addresses))
                .then_some(address)
            })
            .collect::<Vec<_>>();
        for address in pending {
            self.flush_render_target(address)?;
        }
        Ok(())
    }

    pub(super) fn flush_for_compute_inputs(
        &mut self,
        dispatch: &ComputeDispatch,
    ) -> Result<(), String> {
        let pending = self
            .render_target_cache
            .iter()
            .filter_map(|(&address, target)| {
                (target.pending_writeback
                    && compute_requires_render_target_publication(dispatch, target))
                .then_some(address)
            })
            .collect::<Vec<_>>();
        for address in pending {
            self.flush_render_target(address)?;
        }
        Ok(())
    }

    fn flush_render_target(&mut self, address: u64) -> Result<(), String> {
        let timing_started = Instant::now();
        let Some(mut target) = self.render_target_cache.remove(&address) else {
            return Ok(());
        };
        if !target.pending_writeback {
            self.render_target_cache.insert(address, target);
            return Ok(());
        }
        let mut gpu_complete = None;
        let result = (|| {
            let allocator: ID3D12CommandAllocator = unsafe {
                self.device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            }
            .map_err(|error| {
                backend_error("ID3D12Device::CreateCommandAllocator(target flush)", &error)
                    .to_string()
            })?;
            let commands: ID3D12GraphicsCommandList = unsafe {
                self.device.CreateCommandList(
                    0,
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    &allocator,
                    None::<&ID3D12PipelineState>,
                )
            }
            .map_err(|error| {
                backend_error("ID3D12Device::CreateCommandList(target flush)", &error).to_string()
            })?;
            name_object(
                &commands,
                &format!("REmu target flush address={address:#x}"),
            );
            record_render_target_readback(&commands, &self.root, &target, None, false)?;
            unsafe { commands.Close() }.map_err(|error| {
                backend_error("ID3D12GraphicsCommandList::Close(target flush)", &error).to_string()
            })?;
            let list: ID3D12CommandList = commands.cast().map_err(|error| {
                backend_error("ID3D12GraphicsCommandList::cast(target flush)", &error).to_string()
            })?;
            unsafe { self.queue.ExecuteCommandLists(&[Some(list)]) };
            self.wait()?;
            gpu_complete = Some(timing_started.elapsed());
            write_back_render_target(&mut target)
        })();
        self.render_target_cache.insert(address, target);
        if std::env::var_os("REMU_TRACE_D3D12_TIMING").is_some_and(|value| value == "1") {
            let complete = timing_started.elapsed();
            let gpu_complete = gpu_complete.unwrap_or(complete);
            eprintln!(
                "[REMu][D3D12-TIMING] flush_target={address:#x} gpu_us={} writeback_us={} total_us={}",
                gpu_complete.as_micros(),
                complete.saturating_sub(gpu_complete).as_micros(),
                complete.as_micros(),
            );
        }
        result
    }

    fn acquire_command_context(
        &mut self,
    ) -> Result<(ID3D12CommandAllocator, ID3D12GraphicsCommandList), String> {
        if let Some((allocator, commands)) = self.command_pool.pop() {
            // SAFETY: wait() only places contexts in the pool after their queue fence completes.
            unsafe { allocator.Reset() }.map_err(|error| {
                backend_error("ID3D12CommandAllocator::Reset(draw)", &error).to_string()
            })?;
            // SAFETY: the list is closed, its allocator has been reset, and both are direct type.
            unsafe { commands.Reset(&allocator, None::<&ID3D12PipelineState>) }.map_err(
                |error| backend_error("ID3D12GraphicsCommandList::Reset(draw)", &error).to_string(),
            )?;
            return Ok((allocator, commands));
        }

        // SAFETY: direct allocator/list types match the executor queue.
        let allocator: ID3D12CommandAllocator = unsafe {
            self.device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
        }
        .map_err(|error| {
            backend_error("ID3D12Device::CreateCommandAllocator(draw)", &error).to_string()
        })?;
        let commands: ID3D12GraphicsCommandList = unsafe {
            self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None::<&ID3D12PipelineState>,
            )
        }
        .map_err(|error| {
            backend_error("ID3D12Device::CreateCommandList(draw)", &error).to_string()
        })?;
        Ok((allocator, commands))
    }

    fn wait(&mut self) -> Result<(), String> {
        let value = self.next_fence;
        self.next_fence = self.next_fence.saturating_add(1);
        // SAFETY: this queue and fence are live and the value monotonically increases.
        unsafe { self.queue.Signal(&self.fence, value) }.map_err(|error| {
            backend_error("ID3D12CommandQueue::Signal(guest)", &error).to_string()
        })?;
        wait_for_fence(&self.fence, value).map_err(|error| error.to_string())?;
        // A removed device can still leave a fence at the requested value when the failure is
        // reported asynchronously after that command completed. Query the device immediately so
        // the offending submission fails here instead of being misattributed to the next native
        // object creation.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            return Err(format!(
                "D3D12 device removed after guest fence {value}: {}",
                device_removed_report(&self.device)
            ));
        }
        for resources in std::mem::take(&mut self.pending_draw_resources) {
            self.command_pool
                .push((resources._allocator, resources._commands));
        }
        self.descriptor_arena.reset();
        self.upload_arena.reset();
        Ok(())
    }

    fn translate_shader_cached(
        &mut self,
        program: &Arc<Program>,
        evaluation: &Arc<ShaderEvaluation>,
        options: &MetalCompileOptions,
        threads_per_group: [u32; 3],
        source_name: &str,
    ) -> Result<(HlslShader, CompiledGuestShader, bool), String> {
        let key = TranslatedShaderCacheKey {
            program_identity: Arc::as_ptr(program) as usize,
            evaluation: HlslEvaluationKey::new(evaluation),
            options: options.clone(),
            threads_per_group,
        };
        if let Some(cached) = self.translated_shader_cache.get(&key) {
            let mut hlsl = cached.hlsl.clone();
            // Vertex addresses, strides, and packet-time snapshots are execution metadata rather
            // than shader code. Always pair reused DXIL with the current draw's geometry inputs.
            hlsl.vertex_inputs.clone_from(&evaluation.vertex_inputs);
            return Ok((hlsl, cached.compiled.clone(), true));
        }

        let hlsl = translate_guest_shader(program, evaluation, options, threads_per_group)
            .map_err(|error| format!("{source_name} translation: {error}"))?;
        if let Err(error) = dump_translated_shader(program, &hlsl, source_name) {
            eprintln!("[REMu][D3D12] could not dump {source_name}: {error}");
        }
        let compiled = self.compile_shader_cached(&hlsl, source_name)?;
        if self.translated_shader_cache.len() >= MAXIMUM_TRANSLATED_SHADER_CACHE_ENTRIES {
            // Bounded whole-cache eviction limits generated source/DXIL retention. Submitted draws
            // independently retain compiled shader and pipeline resources until their fence.
            self.translated_shader_cache.clear();
        }
        self.translated_shader_cache.insert(
            key,
            TranslatedGuestShader {
                _program: program.clone(),
                hlsl: hlsl.clone(),
                compiled: compiled.clone(),
            },
        );
        Ok((hlsl, compiled, false))
    }

    fn compile_shader_cached(
        &mut self,
        shader: &HlslShader,
        source_name: &str,
    ) -> Result<CompiledGuestShader, String> {
        let key = ShaderCacheKey {
            source: shader.source.clone(),
            entry_point: shader.entry_point.clone(),
            target_profile: shader.target_profile.clone(),
        };
        if let Some((cached_key, dxil)) = self.shader_cache.get_key_value(&key) {
            return Ok(CompiledGuestShader {
                key: cached_key.clone(),
                dxil: dxil.clone(),
            });
        }
        if self.shader_cache.len() >= MAXIMUM_SHADER_CACHE_ENTRIES {
            // Submitted draws retain their own compiled/pipeline objects until the queue fence;
            // clearing these lookup tables therefore only evicts future reuse. Whole-cache
            // eviction bounds dynamically generated shader source without adding an LRU.
            self.translated_shader_cache.clear();
            self.shader_cache.clear();
            self.compute_pipeline_cache.clear();
            self.graphics_pipeline_cache.clear();
        }
        let dxil: Arc<[u8]> = compile_guest_shader(shader, source_name)?.into();
        let key = Arc::new(key);
        self.shader_cache.insert(key.clone(), dxil.clone());
        Ok(CompiledGuestShader { key, dxil })
    }

    fn compute_pipeline_cached(
        &mut self,
        shader: &CompiledGuestShader,
    ) -> Result<ID3D12PipelineState, String> {
        if let Some(pipeline) = self.compute_pipeline_cache.get(&shader.key) {
            return Ok(pipeline.clone());
        }
        if self.compute_pipeline_cache.len() >= MAXIMUM_COMPUTE_PIPELINE_CACHE_ENTRIES {
            self.compute_pipeline_cache.clear();
        }
        let pipeline = create_compute_pipeline(&self.device, &self.root, &shader.dxil)?;
        self.compute_pipeline_cache
            .insert(shader.key.clone(), pipeline.clone());
        Ok(pipeline)
    }

    fn graphics_pipeline_cached(
        &mut self,
        vertex: &CompiledGuestShader,
        pixel: Option<&CompiledGuestShader>,
        draw: &IndexedDraw,
        targets: &[DecodedRenderTarget],
        depth: Option<DecodedDepthTarget>,
        topology: windows::Win32::Graphics::Direct3D12::D3D12_PRIMITIVE_TOPOLOGY_TYPE,
        strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE,
        sample_count: u32,
    ) -> Result<ID3D12PipelineState, String> {
        let key = GraphicsPipelineKey {
            vertex: vertex.key.clone(),
            pixel: pixel.map(|shader| shader.key.clone()),
            topology: topology.0,
            index_buffer_strip_cut: strip_cut.0,
            sample_count,
            raster_control: register_value(&draw.context_registers, 0x205).unwrap_or_default(),
            render_targets: targets
                .iter()
                .map(|target| {
                    (
                        target.descriptor.format.0,
                        target.guest_slot,
                        target.write_mask,
                    )
                })
                .collect(),
            blend_controls: targets
                .iter()
                .map(|target| {
                    register_value(&draw.context_registers, 0x1e0 + target.guest_slot)
                        .unwrap_or_default()
                })
                .collect(),
            depth: depth.map(|depth| {
                (
                    depth_host_format(depth).0,
                    register_value(&draw.context_registers, 0x200).unwrap_or_default(),
                    register_value(&draw.context_registers, 0x10b).unwrap_or_default(),
                    register_value(&draw.context_registers, 0x10c).unwrap_or_default(),
                    register_value(&draw.context_registers, 0x10d).unwrap_or_default(),
                )
            }),
        };
        if let Some(pipeline) = self.graphics_pipeline_cache.get(&key) {
            return Ok(pipeline.clone());
        }
        if self.graphics_pipeline_cache.len() >= MAXIMUM_GRAPHICS_PIPELINE_CACHE_ENTRIES {
            self.graphics_pipeline_cache.clear();
        }
        let pipeline = create_graphics_pipeline(
            &self.device,
            &self.root,
            &vertex.dxil,
            pixel.map(|shader| shader.dxil.as_ref()),
            draw,
            targets,
            depth,
            topology,
            strip_cut,
            sample_count,
            self.independent_stencil_ref_mask,
        )
        .map_err(|error| {
            if let Err(dump_error) = dump_failed_graphics_pipeline(vertex, pixel) {
                eprintln!("[REMu][D3D12] failed-PSO diagnostic dump failed: {dump_error}");
            }
            format!(
                "{error}; topology={} strip_cut={} samples={} raster={:#010x} targets={:?} blend={:x?} depth={:?} vs={} ps={} vs_semantics={:?} ps_semantics={:?}",
                key.topology,
                key.index_buffer_strip_cut,
                key.sample_count,
                key.raster_control,
                key.render_targets,
                key.blend_controls,
                key.depth,
                key.vertex.target_profile,
                key.pixel
                    .as_ref()
                    .map_or("none", |shader| shader.target_profile.as_str()),
                shader_semantics(&key.vertex.source),
                key.pixel
                    .as_ref()
                    .map(|shader| shader_semantics(&shader.source))
                    .unwrap_or_default(),
            )
        })?;
        self.graphics_pipeline_cache.insert(key, pipeline.clone());
        Ok(pipeline)
    }

    fn msaa_initialization_pipeline_cached(
        &mut self,
        target: DecodedRenderTarget,
    ) -> Result<ID3D12PipelineState, String> {
        let key = MsaaInitializationPipelineKey {
            format: target.descriptor.format.0,
            output_kind: target.output_kind,
            sample_count: target.sample_count,
        };
        if let Some(pipeline) = self.msaa_initialization_pipeline_cache.get(&key) {
            return Ok(pipeline.clone());
        }
        if self.msaa_initialization_pipeline_cache.len() >= MAXIMUM_GRAPHICS_PIPELINE_CACHE_ENTRIES
        {
            self.msaa_initialization_pipeline_cache.clear();
        }
        let pipeline = create_msaa_initialization_pipeline(&self.device, &self.root, target)?;
        self.msaa_initialization_pipeline_cache
            .insert(key, pipeline.clone());
        Ok(pipeline)
    }

    fn depth_msaa_initialization_pipeline_cached(
        &mut self,
        target: DecodedDepthTarget,
    ) -> Result<ID3D12PipelineState, String> {
        let key = DepthMsaaInitializationPipelineKey {
            format: depth_host_format(target).0,
            sample_count: target.sample_count,
        };
        if let Some(pipeline) = self.depth_msaa_initialization_pipeline_cache.get(&key) {
            return Ok(pipeline.clone());
        }
        if self.depth_msaa_initialization_pipeline_cache.len()
            >= MAXIMUM_GRAPHICS_PIPELINE_CACHE_ENTRIES
        {
            self.depth_msaa_initialization_pipeline_cache.clear();
        }
        let pipeline = create_depth_msaa_initialization_pipeline(&self.device, &self.root, target)?;
        self.depth_msaa_initialization_pipeline_cache
            .insert(key, pipeline.clone());
        Ok(pipeline)
    }

    fn integer_msaa_resolve_pipeline_cached(
        &mut self,
        target: DecodedRenderTarget,
    ) -> Result<ID3D12PipelineState, String> {
        let key = IntegerMsaaResolvePipelineKey {
            format: target.descriptor.format.0,
            output_kind: target.output_kind,
        };
        if let Some(pipeline) = self.integer_msaa_resolve_pipeline_cache.get(&key) {
            return Ok(pipeline.clone());
        }
        if self.integer_msaa_resolve_pipeline_cache.len() >= MAXIMUM_GRAPHICS_PIPELINE_CACHE_ENTRIES
        {
            self.integer_msaa_resolve_pipeline_cache.clear();
        }
        let pipeline = create_integer_msaa_resolve_pipeline(&self.device, &self.root, target)?;
        self.integer_msaa_resolve_pipeline_cache
            .insert(key, pipeline.clone());
        Ok(pipeline)
    }
}

/// Persist guest shader IR and translated HLSL for targeted Windows bring-up investigations.
///
/// `REMU_DUMP_HLSL_DIR` enables the dump. `REMU_DUMP_HLSL_CS` is deliberately required for
/// compute shaders because games can compile hundreds of transient variants: use `all` or the
/// hexadecimal guest shader address from the normal AGC log. Vertex and pixel shaders are dumped
/// whenever the directory is enabled, matching the existing Metal diagnostics.
fn dump_translated_shader(
    program: &Program,
    shader: &HlslShader,
    source_name: &str,
) -> Result<(), String> {
    let Some(directory) = std::env::var_os("REMU_DUMP_HLSL_DIR") else {
        return Ok(());
    };
    if source_name.starts_with("remu-cs-") {
        let Some(filter) = std::env::var_os("REMU_DUMP_HLSL_CS") else {
            return Ok(());
        };
        let filter = filter.to_string_lossy();
        if !filter.eq_ignore_ascii_case("all") {
            let filter = filter.strip_prefix("0x").unwrap_or(&filter);
            let address = source_name
                .strip_prefix("remu-cs-")
                .and_then(|name| name.strip_suffix(".hlsl"));
            if address != Some(filter) {
                return Ok(());
            }
        }
    }

    let directory = Path::new(&directory);
    std::fs::create_dir_all(directory)
        .map_err(|error| format!("create {}: {error}", directory.display()))?;
    let hlsl_path = directory.join(source_name);
    std::fs::write(&hlsl_path, shader.source.as_bytes())
        .map_err(|error| format!("write {}: {error}", hlsl_path.display()))?;
    let stem = source_name.strip_suffix(".hlsl").unwrap_or(source_name);
    let ir_path = directory.join(format!("{stem}-ir.txt"));
    std::fs::write(&ir_path, format!("{:#?}\n", program.instructions))
        .map_err(|error| format!("write {}: {error}", ir_path.display()))
}

fn dump_failed_graphics_pipeline(
    vertex: &CompiledGuestShader,
    pixel: Option<&CompiledGuestShader>,
) -> Result<(), String> {
    static DUMPED: AtomicBool = AtomicBool::new(false);
    let Some(directory) = std::env::var_os("REMU_FAILED_PSO_DUMP") else {
        return Ok(());
    };
    if DUMPED.swap(true, Ordering::Relaxed) {
        return Ok(());
    }
    let directory = std::path::PathBuf::from(directory);
    std::fs::create_dir_all(&directory)
        .map_err(|error| format!("create {}: {error}", directory.display()))?;
    std::fs::write(directory.join("vertex.hlsl"), vertex.key.source.as_bytes())
        .map_err(|error| format!("write vertex HLSL: {error}"))?;
    std::fs::write(directory.join("vertex.dxil"), vertex.dxil.as_ref())
        .map_err(|error| format!("write vertex DXIL: {error}"))?;
    if let Some(pixel) = pixel {
        std::fs::write(directory.join("pixel.hlsl"), pixel.key.source.as_bytes())
            .map_err(|error| format!("write pixel HLSL: {error}"))?;
        std::fs::write(directory.join("pixel.dxil"), pixel.dxil.as_ref())
            .map_err(|error| format!("write pixel DXIL: {error}"))?;
    }
    Ok(())
}

fn allocated_compute_vgprs(dispatch: &ComputeDispatch) -> u32 {
    dispatch
        .shader_registers
        .iter()
        .find(|register| (register.offset & !0x7000_0000) == 0x212)
        .map_or(256, |register| (register.value & 0x3f).max(1) * 4)
}

fn graphics_allocated_vgprs(registers: &[remu_gpu::GpuRegister], user_base: u32) -> u32 {
    register_value(registers, user_base.saturating_sub(2))
        .map_or(256, |resource| ((resource & 0x3f) + 1) * 4)
}

fn record_stage_uploads(commands: &ID3D12GraphicsCommandList, resources: &ComputeResources) {
    for resource in &resources.globals {
        let Some(upload) = &resource.upload else {
            continue;
        };
        // SAFETY: both buffers cover the padded resource byte count.
        unsafe {
            commands.CopyBufferRegion(
                &resource.gpu,
                u64::try_from(resource.gpu_offset).expect("bounded GPU buffer offset fits u64"),
                upload,
                u64::try_from(resource.upload_offset)
                    .expect("bounded upload buffer offset fits u64"),
                resource.byte_count.max(4) as u64,
            )
        };
        record_transition(
            commands,
            &resource.gpu,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
    }
    record_texture_uploads(commands, &resources.textures);
}

fn stage_resources_require_writeback(resources: &ComputeResources) -> bool {
    resources
        .globals
        .iter()
        .any(|resource| resource.readback.is_some())
        || resources
            .textures
            .iter()
            .any(|resource| resource.descriptor.writable)
}

fn record_stage_readbacks(commands: &ID3D12GraphicsCommandList, resources: &ComputeResources) {
    for resource in &resources.globals {
        if let Some(readback) = &resource.readback {
            record_transition(
                commands,
                &resource.gpu,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            // SAFETY: buffers cover the padded copy byte count.
            unsafe {
                commands.CopyBufferRegion(
                    readback,
                    0,
                    &resource.gpu,
                    0,
                    resource.byte_count.max(4) as u64,
                )
            };
        }
    }
    record_texture_readbacks(commands, &resources.textures);
}

fn write_back_stage(
    evaluation: &ShaderEvaluation,
    resources: &ComputeResources,
    ordered_versions: &mut OrderedGlobalVersions,
) -> Result<(), String> {
    write_back_globals(evaluation, &resources.globals, ordered_versions)?;
    write_back_textures(&resources.textures)
}

fn decode_native_primitive_topology(
    primitive: u32,
    triangle_fan_supported: bool,
) -> Result<
    (
        windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY,
        windows::Win32::Graphics::Direct3D12::D3D12_PRIMITIVE_TOPOLOGY_TYPE,
    ),
    String,
> {
    match primitive {
        1 => Ok((
            D3D_PRIMITIVE_TOPOLOGY_POINTLIST,
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT,
        )),
        2 => Ok((
            D3D_PRIMITIVE_TOPOLOGY_LINELIST,
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,
        )),
        3 => Ok((
            D3D_PRIMITIVE_TOPOLOGY_LINESTRIP,
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,
        )),
        4 => Ok((
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        )),
        5 if triangle_fan_supported => Ok((
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLEFAN,
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        )),
        5 => Err("triangle fan must be expanded for this D3D12 adapter".to_owned()),
        6 | 7 | 0x11 => Ok((
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        )),
        0x0a => Ok((
            D3D_PRIMITIVE_TOPOLOGY_LINELIST_ADJ,
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,
        )),
        0x0b => Ok((
            D3D_PRIMITIVE_TOPOLOGY_LINESTRIP_ADJ,
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,
        )),
        0x0c => Ok((
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST_ADJ,
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        )),
        0x0d => Ok((
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP_ADJ,
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        )),
        _ => Err(format!(
            "unsupported GFX10 primitive topology {primitive:#x}"
        )),
    }
}

const fn nonindexed_vertex_count(primitive: u32, count: u32) -> u32 {
    if matches!(primitive, 7 | 0x11) && count == 3 {
        4
    } else {
        count
    }
}

fn decode_scissor(draw: &IndexedDraw, target_width: u32, target_height: u32) -> RECT {
    let mut bounds = [0_i64, 0, i64::from(target_width), i64::from(target_height)];
    let window_scissor_tl = register_value(&draw.context_registers, 0x081);
    let enable_window_offset = window_scissor_tl.is_none_or(|value| value & 0x8000_0000 == 0);
    let (window_offset_x, window_offset_y) = if enable_window_offset {
        register_value(&draw.context_registers, 0x080).map_or((0, 0), |value| {
            (
                i64::from((value as u16) as i16),
                i64::from(((value >> 16) as u16) as i16),
            )
        })
    } else {
        (0, 0)
    };
    intersect_scissor_pair(draw, 0x00c, 0x00d, 0, 0, true, &mut bounds);
    intersect_scissor_pair(
        draw,
        0x081,
        0x082,
        window_offset_x,
        window_offset_y,
        false,
        &mut bounds,
    );
    intersect_scissor_pair(
        draw,
        0x090,
        0x091,
        window_offset_x,
        window_offset_y,
        false,
        &mut bounds,
    );
    if register_value(&draw.context_registers, 0x292).is_none_or(|mode| mode & 2 != 0) {
        intersect_scissor_pair(draw, 0x094, 0x095, 0, 0, false, &mut bounds);
    }
    bounds[0] = bounds[0].clamp(0, i64::from(target_width));
    bounds[1] = bounds[1].clamp(0, i64::from(target_height));
    bounds[2] = bounds[2].clamp(bounds[0], i64::from(target_width));
    bounds[3] = bounds[3].clamp(bounds[1], i64::from(target_height));
    RECT {
        left: bounds[0] as i32,
        top: bounds[1] as i32,
        right: bounds[2] as i32,
        bottom: bounds[3] as i32,
    }
}

#[allow(clippy::too_many_arguments)]
fn intersect_scissor_pair(
    draw: &IndexedDraw,
    top_left_register: u32,
    bottom_right_register: u32,
    offset_x: i64,
    offset_y: i64,
    ignore_all_zero: bool,
    bounds: &mut [i64; 4],
) {
    let (Some(top_left), Some(bottom_right)) = (
        register_value(&draw.context_registers, top_left_register),
        register_value(&draw.context_registers, bottom_right_register),
    ) else {
        return;
    };
    if ignore_all_zero && top_left == 0 && bottom_right == 0 {
        return;
    }
    let pair = [
        i64::from(top_left & 0x7fff) + offset_x,
        i64::from((top_left >> 16) & 0x7fff) + offset_y,
        i64::from(bottom_right & 0x7fff) + offset_x,
        i64::from((bottom_right >> 16) & 0x7fff) + offset_y,
    ];
    bounds[0] = bounds[0].max(pair[0]);
    bounds[1] = bounds[1].max(pair[1]);
    bounds[2] = bounds[2].min(pair[2]);
    bounds[3] = bounds[3].min(pair[3]);
}

fn decode_viewport(draw: &IndexedDraw, scissor: RECT) -> D3D12_VIEWPORT {
    let mut minimum_depth = 0.0;
    let mut maximum_depth = 1.0;
    if let (Some(minimum), Some(maximum)) = (
        register_value(&draw.context_registers, 0x0b4).map(f32::from_bits),
        register_value(&draw.context_registers, 0x0b5).map(f32::from_bits),
    ) && minimum.is_finite()
        && maximum.is_finite()
        && maximum > minimum
    {
        minimum_depth = minimum;
        maximum_depth = maximum;
    }
    let programmed = [0x10f, 0x110, 0x111, 0x112]
        .map(|offset| register_value(&draw.context_registers, offset).map(f32::from_bits));
    if let [Some(x_scale), Some(x_offset), Some(y_scale), Some(y_offset)] = programmed
        && [x_scale, x_offset, y_scale, y_offset]
            .into_iter()
            .all(f32::is_finite)
        && x_scale > 0.0
        && y_scale != 0.0
    {
        let mut top = y_offset - y_scale;
        let mut height = y_scale * 2.0;
        if height < 0.0 {
            top += height;
            height = -height;
        }
        return D3D12_VIEWPORT {
            TopLeftX: x_offset - x_scale,
            TopLeftY: top,
            Width: x_scale * 2.0,
            Height: height,
            MinDepth: minimum_depth,
            MaxDepth: maximum_depth,
        };
    }
    D3D12_VIEWPORT {
        TopLeftX: scissor.left as f32,
        TopLeftY: scissor.top as f32,
        Width: (scissor.right - scissor.left).max(0) as f32,
        Height: (scissor.bottom - scissor.top).max(0) as f32,
        MinDepth: minimum_depth,
        MaxDepth: maximum_depth,
    }
}

fn decode_raster_state(draw: &IndexedDraw) -> Result<DecodedRasterState, String> {
    let control = register_value(&draw.context_registers, 0x205).unwrap_or(1 << 2);
    let cull_front = control & 1 != 0;
    let cull_back = control & 2 != 0;
    let (cull_mode, discard_triangles) = match (cull_front, cull_back) {
        (false, false) => (D3D12_CULL_MODE_NONE, false),
        (true, false) => (
            windows::Win32::Graphics::Direct3D12::D3D12_CULL_MODE_FRONT,
            false,
        ),
        (false, true) => (
            windows::Win32::Graphics::Direct3D12::D3D12_CULL_MODE_BACK,
            false,
        ),
        (true, true) => (D3D12_CULL_MODE_NONE, true),
    };
    let polygon_mode = (control >> 3) & 3;
    let fill_mode = match polygon_mode {
        0 => D3D12_FILL_MODE_SOLID,
        1 => {
            let front = (control >> 5) & 7;
            let back = (control >> 8) & 7;
            let visible = if cull_front {
                back
            } else if cull_back || front == back {
                front
            } else {
                return Err(format!(
                    "different front/back polygon modes ({front}/{back}) cannot share a D3D12 PSO"
                ));
            };
            match visible {
                1 => windows::Win32::Graphics::Direct3D12::D3D12_FILL_MODE_WIREFRAME,
                2 => D3D12_FILL_MODE_SOLID,
                0 => {
                    return Err(
                        "point polygon mode is not supported by D3D12 raster state".to_owned()
                    );
                }
                value => return Err(format!("reserved GFX10 polygon primitive type {value}")),
            }
        }
        value => return Err(format!("reserved GFX10 polygon mode {value}")),
    };
    Ok(DecodedRasterState {
        cull_mode,
        front_counter_clockwise: control & (1 << 2) == 0,
        fill_mode,
        discard_triangles,
    })
}

fn decode_blend_attachment(
    control: u32,
    write_mask: u32,
) -> Result<D3D12_RENDER_TARGET_BLEND_DESC, String> {
    let enabled = control & (1 << 30) != 0;
    let source_rgb = decode_blend_factor(control & 0x1f)?;
    let operation_rgb = decode_blend_operation((control >> 5) & 7)?;
    let destination_rgb = decode_blend_factor((control >> 8) & 0x1f)?;
    let separate_alpha = control & (1 << 29) != 0;
    let (source_alpha, operation_alpha, destination_alpha) = if separate_alpha {
        (
            decode_blend_factor((control >> 16) & 0x1f)?,
            decode_blend_operation((control >> 21) & 7)?,
            decode_blend_factor((control >> 24) & 0x1f)?,
        )
    } else {
        (source_rgb, operation_rgb, destination_rgb)
    };
    Ok(D3D12_RENDER_TARGET_BLEND_DESC {
        BlendEnable: BOOL(i32::from(enabled)),
        LogicOpEnable: BOOL(0),
        SrcBlend: source_rgb,
        DestBlend: destination_rgb,
        BlendOp: operation_rgb,
        SrcBlendAlpha: source_alpha,
        DestBlendAlpha: destination_alpha,
        BlendOpAlpha: operation_alpha,
        LogicOp: D3D12_LOGIC_OP_NOOP,
        RenderTargetWriteMask: (write_mask & 0xf) as u8,
    })
}

fn decode_blend_factor(
    value: u32,
) -> Result<windows::Win32::Graphics::Direct3D12::D3D12_BLEND, String> {
    use windows::Win32::Graphics::Direct3D12::*;
    match value {
        0 => Ok(D3D12_BLEND_ZERO),
        1 => Ok(D3D12_BLEND_ONE),
        2 => Ok(D3D12_BLEND_SRC_COLOR),
        3 => Ok(D3D12_BLEND_INV_SRC_COLOR),
        4 => Ok(D3D12_BLEND_SRC_ALPHA),
        5 => Ok(D3D12_BLEND_INV_SRC_ALPHA),
        6 => Ok(D3D12_BLEND_DEST_ALPHA),
        7 => Ok(D3D12_BLEND_INV_DEST_ALPHA),
        8 => Ok(D3D12_BLEND_DEST_COLOR),
        9 => Ok(D3D12_BLEND_INV_DEST_COLOR),
        10 => Ok(D3D12_BLEND_SRC_ALPHA_SAT),
        13 | 19 => Ok(D3D12_BLEND_BLEND_FACTOR),
        14 | 20 => Ok(D3D12_BLEND_INV_BLEND_FACTOR),
        15 => Ok(D3D12_BLEND_SRC1_COLOR),
        16 => Ok(D3D12_BLEND_INV_SRC1_COLOR),
        17 => Ok(D3D12_BLEND_SRC1_ALPHA),
        18 => Ok(D3D12_BLEND_INV_SRC1_ALPHA),
        _ => Err(format!("unsupported GFX10 blend factor {value}")),
    }
}

fn decode_blend_operation(
    value: u32,
) -> Result<windows::Win32::Graphics::Direct3D12::D3D12_BLEND_OP, String> {
    use windows::Win32::Graphics::Direct3D12::*;
    match value {
        0 => Ok(D3D12_BLEND_OP_ADD),
        1 => Ok(D3D12_BLEND_OP_SUBTRACT),
        2 => Ok(D3D12_BLEND_OP_MIN),
        3 => Ok(D3D12_BLEND_OP_MAX),
        4 => Ok(D3D12_BLEND_OP_REV_SUBTRACT),
        _ => Err(format!("unsupported GFX10 blend operation {value}")),
    }
}

fn primitive_restart_index(draw: &IndexedDraw) -> Option<u32> {
    const VGT_MULTI_PRIM_IB_RESET_INDX: u32 = 0x103;
    const VGT_MULTI_PRIM_IB_RESET_EN: u32 = 0x2a5;

    if register_value(&draw.context_registers, VGT_MULTI_PRIM_IB_RESET_EN)
        .is_none_or(|value| value & 1 == 0)
    {
        return None;
    }
    let programmed =
        register_value(&draw.context_registers, VGT_MULTI_PRIM_IB_RESET_INDX).unwrap_or(u32::MAX);
    match draw.index_size {
        0 => Some(programmed & u32::from(u16::MAX)),
        1 => Some(programmed),
        2 => Some(programmed & u32::from(u8::MAX)),
        _ => None,
    }
}

fn read_indirect_draw_arguments(
    address: u64,
    indexed: bool,
) -> Result<IndirectDrawArguments, String> {
    let byte_count = if indexed { 20 } else { 16 };
    let mut bytes = [0_u8; 20];
    read_process_memory(address, &mut bytes[..byte_count])
        .map_err(|error| format!("indirect draw arguments at {address:#x}: {error}"))?;
    let word = |index: usize| {
        let offset = index * 4;
        u32::from_le_bytes(
            bytes[offset..offset + 4]
                .try_into()
                .expect("indirect argument dword"),
        )
    };
    Ok(IndirectDrawArguments {
        element_count: word(0),
        instance_count: word(1),
        element_offset: word(2),
        base_vertex: if indexed { word(3) as i32 } else { 0 },
        first_instance: word(if indexed { 4 } else { 3 }),
    })
}

fn resolve_indirect_draw(
    draw: &IndexedDraw,
    address: u64,
) -> Result<(IndexedDraw, IndirectDrawArguments), String> {
    let arguments = read_indirect_draw_arguments(address, draw.indexed)?;
    let mut resolved = draw.clone();
    resolved.index_count = arguments.element_count;
    resolved.instance_count = arguments.instance_count;
    resolved.index_offset = arguments.element_offset;
    resolved.base_vertex = arguments.base_vertex;
    // Packet-time snapshots are intentionally absent for indirect draws. Never retain one if a
    // synthetic caller supplied it: the live first/count pair names a different range.
    resolved.index_snapshot = None;
    Ok((resolved, arguments))
}

fn indirect_draw_count_allows(draw: &IndexedDraw) -> Result<bool, String> {
    let Some(address) = draw.indirect_count_address else {
        return Ok(true);
    };
    let mut bytes = [0_u8; 4];
    read_process_memory(address, &mut bytes)
        .map_err(|error| format!("indirect draw count at {address:#x}: {error}"))?;
    Ok(draw.indirect_draw_index < u32::from_le_bytes(bytes))
}

fn load_draw_indices(draw: &IndexedDraw) -> Result<Vec<u32>, String> {
    if let Some(snapshot) = &draw.index_snapshot {
        if snapshot.len() != draw.index_count as usize {
            return Err(format!(
                "index snapshot contains {} elements, expected {}",
                snapshot.len(),
                draw.index_count
            ));
        }
        return Ok(snapshot.to_vec());
    }
    if draw.index_buffer_address < 0x10_000 {
        return Err(format!(
            "invalid index buffer address {:#x}",
            draw.index_buffer_address
        ));
    }
    let count = draw.index_count as usize;
    let first = draw.index_offset as usize;
    let end = draw
        .index_offset
        .checked_add(draw.index_count)
        .ok_or("index range overflows")?;
    if draw.index_buffer_count != 0 && end > draw.index_buffer_count {
        return Err(format!(
            "index range {}..{} exceeds buffer count {}",
            draw.index_offset, end, draw.index_buffer_count
        ));
    }
    let element_bytes = match draw.index_size {
        0 => 2,
        1 => 4,
        2 => 1,
        value => return Err(format!("unsupported guest index size {value}")),
    };
    let byte_offset = first
        .checked_mul(element_bytes)
        .ok_or("index offset overflows")?;
    let byte_count = count
        .checked_mul(element_bytes)
        .ok_or("index byte count overflows")?;
    // SAFETY: the command processor evaluated this mapped range and retains it until draw
    // retirement; conversion normalizes all formats to D3D12 u32 indices.
    let bytes = unsafe {
        std::slice::from_raw_parts(
            (draw.index_buffer_address as usize + byte_offset) as *const u8,
            byte_count,
        )
    };
    Ok(bytes
        .chunks_exact(element_bytes)
        .map(|value| match element_bytes {
            1 => u32::from(value[0]),
            2 => u32::from(u16::from_le_bytes([value[0], value[1]])),
            4 => u32::from_le_bytes(value.try_into().expect("four-byte index")),
            _ => unreachable!(),
        })
        .collect())
}

fn logical_draw_indices(draw: &IndexedDraw) -> Result<Vec<u32>, String> {
    if draw.indexed {
        return load_draw_indices(draw);
    }
    (0..nonindexed_vertex_count(draw.primitive_type, draw.index_count))
        .map(|offset| {
            draw.index_offset
                .checked_add(offset)
                .ok_or_else(|| "non-indexed vertex range overflows".to_owned())
        })
        .collect()
}

fn for_each_index_segment(indices: &[u32], restart: Option<u32>, mut visit: impl FnMut(&[u32])) {
    let Some(restart) = restart else {
        visit(indices);
        return;
    };
    let mut first = 0;
    for (position, &index) in indices.iter().enumerate() {
        if index == restart {
            visit(&indices[first..position]);
            first = position + 1;
        }
    }
    visit(&indices[first..]);
}

fn expand_primitive_indices(
    primitive: u32,
    indices: &[u32],
    restart: Option<u32>,
) -> Result<Vec<u32>, String> {
    let capacity_multiplier = match primitive {
        5 | 0x15 => 3,
        0x12 => 2,
        0x13 | 0x14 => 2,
        _ => {
            return Err(format!(
                "primitive {primitive:#x} has no CPU index expansion"
            ));
        }
    };
    let mut output = Vec::new();
    output
        .try_reserve(indices.len().saturating_mul(capacity_multiplier))
        .map_err(|_| "expanded primitive index allocation failed")?;
    for_each_index_segment(indices, restart, |segment| match primitive {
        // Triangle fan and polygon: the first vertex is the common center.
        5 | 0x15 => {
            for triangle in segment[segment.len().min(1)..].windows(2) {
                output.extend_from_slice(&[segment[0], triangle[0], triangle[1]]);
            }
        }
        // Line loop: preserve every line-strip edge and close the final edge.
        0x12 if segment.len() >= 2 => {
            for line in segment.windows(2) {
                output.extend_from_slice(line);
            }
            output.extend_from_slice(&[segment[segment.len() - 1], segment[0]]);
        }
        // Quad list: 0-1-2-3 winds as two triangles around diagonal 0-2.
        0x13 => {
            for quad in segment.chunks_exact(4) {
                output.extend_from_slice(&[quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]]);
            }
        }
        // Quad strip quad order is 0-1-3-2; advance by two vertices per quad.
        0x14 => {
            for start in (0..segment.len().saturating_sub(3)).step_by(2) {
                let quad = &segment[start..start + 4];
                output.extend_from_slice(&[quad[0], quad[1], quad[3], quad[0], quad[3], quad[2]]);
            }
        }
        _ => {}
    });
    Ok(output)
}

fn regroup_restart_list_indices(
    primitive: u32,
    indices: &[u32],
    restart: u32,
) -> Result<Vec<u32>, String> {
    let primitive_width = match primitive {
        1 => 1,
        2 => 2,
        4 => 3,
        0x0a => 4,
        0x0c => 6,
        _ => {
            return Err(format!(
                "primitive restart is not representable for GFX10 topology {primitive:#x}"
            ));
        }
    };
    let mut output = Vec::with_capacity(indices.len());
    for_each_index_segment(indices, Some(restart), |segment| {
        for group in segment.chunks_exact(primitive_width) {
            output.extend_from_slice(group);
        }
    });
    Ok(output)
}

fn create_index_buffer(
    device: &ID3D12Device,
    upload_arena: &mut UploadArena,
    indices: &[u32],
) -> Result<Option<(ID3D12Resource, D3D12_INDEX_BUFFER_VIEW)>, String> {
    if indices.is_empty() {
        return Ok(None);
    }
    let bytes = words_as_bytes(indices);
    let (buffer, byte_offset) = match upload_arena.allocate(&bytes, 4)? {
        Some(allocation) => allocation,
        None => (create_upload_buffer(device, &bytes)?, 0),
    };
    let view = D3D12_INDEX_BUFFER_VIEW {
        BufferLocation: unsafe { buffer.GetGPUVirtualAddress() }
            .checked_add(u64::try_from(byte_offset).map_err(|_| "index offset exceeds u64")?)
            .ok_or("index GPU address overflows")?,
        SizeInBytes: u32::try_from(bytes.len()).map_err(|_| "index buffer exceeds u32")?,
        Format: DXGI_FORMAT_R32_UINT,
    };
    Ok(Some((buffer, view)))
}

fn prepare_geometry(
    device: &ID3D12Device,
    upload_arena: &mut UploadArena,
    draw: &IndexedDraw,
    triangle_fan_supported: bool,
) -> Result<PreparedGeometry, String> {
    let restart = draw
        .indexed
        .then(|| primitive_restart_index(draw))
        .flatten();
    let expand = matches!(draw.primitive_type, 0x12..=0x15)
        || (draw.primitive_type == 5 && (!triangle_fan_supported || restart.is_some()));
    if expand {
        let source = logical_draw_indices(draw)?;
        let indices = expand_primitive_indices(draw.primitive_type, &source, restart)?;
        let element_count = u32::try_from(indices.len())
            .map_err(|_| "expanded primitive index count exceeds D3D12")?;
        return Ok(PreparedGeometry {
            topology: if draw.primitive_type == 0x12 {
                D3D_PRIMITIVE_TOPOLOGY_LINELIST
            } else {
                D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST
            },
            topology_type: if draw.primitive_type == 0x12 {
                D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE
            } else {
                D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE
            },
            index: create_index_buffer(device, upload_arena, &indices)?,
            element_count,
            start_vertex: 0,
            base_vertex: if draw.indexed { draw.base_vertex } else { 0 },
            strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
        });
    }

    let (topology, topology_type) =
        decode_native_primitive_topology(draw.primitive_type, triangle_fan_supported)?;
    if !draw.indexed {
        return Ok(PreparedGeometry {
            topology,
            topology_type,
            index: None,
            element_count: nonindexed_vertex_count(draw.primitive_type, draw.index_count),
            start_vertex: draw.index_offset,
            base_vertex: 0,
            strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
        });
    }

    let mut indices = load_draw_indices(draw)?;
    let mut strip_cut = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;
    if let Some(restart) = restart {
        if matches!(draw.primitive_type, 3 | 6 | 0x0b | 0x0d) {
            for index in &mut indices {
                if *index == restart {
                    *index = u32::MAX;
                }
            }
            strip_cut = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF;
        } else {
            indices = regroup_restart_list_indices(draw.primitive_type, &indices, restart)?;
        }
    }
    let element_count = u32::try_from(indices.len()).map_err(|_| "index count exceeds D3D12")?;
    Ok(PreparedGeometry {
        topology,
        topology_type,
        index: create_index_buffer(device, upload_arena, &indices)?,
        element_count,
        start_vertex: 0,
        base_vertex: draw.base_vertex,
        strip_cut,
    })
}

fn create_graphics_pipeline(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    vertex: &[u8],
    pixel: Option<&[u8]>,
    draw: &IndexedDraw,
    targets: &[DecodedRenderTarget],
    depth: Option<DecodedDepthTarget>,
    topology: windows::Win32::Graphics::Direct3D12::D3D12_PRIMITIVE_TOPOLOGY_TYPE,
    strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE,
    sample_count: u32,
    independent_stencil_ref_mask_supported: bool,
) -> Result<ID3D12PipelineState, String> {
    if std::env::var_os("REMU_D3D12_DEBUG").is_some_and(|value| value == "1")
        && let Ok(info) = device.cast::<ID3D12InfoQueue>()
    {
        // SAFETY: the info queue belongs to this device; clearing only discards old diagnostics.
        unsafe { info.ClearStoredMessages() };
    }
    let mut blend = D3D12_BLEND_DESC::default();
    for (host_slot, target) in targets.iter().enumerate() {
        blend.RenderTarget[host_slot] = decode_blend_attachment(
            register_value(&draw.context_registers, 0x1e0 + target.guest_slot).unwrap_or_default(),
            target.write_mask,
        )?;
    }
    let raster = decode_raster_state(draw)?;
    let rasterizer = D3D12_RASTERIZER_DESC {
        FillMode: raster.fill_mode,
        CullMode: if raster.discard_triangles {
            D3D12_CULL_MODE_NONE
        } else {
            raster.cull_mode
        },
        FrontCounterClockwise: BOOL::from(raster.front_counter_clockwise),
        DepthClipEnable: BOOL(1),
        ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
        ..D3D12_RASTERIZER_DESC::default()
    };
    let default_stencil_face = windows::Win32::Graphics::Direct3D12::D3D12_DEPTH_STENCILOP_DESC {
        StencilFailOp: D3D12_STENCIL_OP_KEEP,
        StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
        StencilPassOp: D3D12_STENCIL_OP_KEEP,
        StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
    };
    let stencil_face = |face: DecodedStencilFace| {
        windows::Win32::Graphics::Direct3D12::D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: face.fail,
            StencilDepthFailOp: face.depth_fail,
            StencilPassOp: face.pass,
            StencilFunc: face.compare,
        }
    };
    let stencil_face1 = |face: DecodedStencilFace, read_mask, write_mask| {
        windows::Win32::Graphics::Direct3D12::D3D12_DEPTH_STENCILOP_DESC1 {
            StencilFailOp: face.fail,
            StencilDepthFailOp: face.depth_fail,
            StencilPassOp: face.pass,
            StencilFunc: face.compare,
            StencilReadMask: read_mask,
            StencilWriteMask: write_mask,
        }
    };
    let stencil = depth.and_then(|depth| depth.stencil);
    let depth_stencil = D3D12_DEPTH_STENCIL_DESC {
        DepthEnable: BOOL(i32::from(
            depth.is_some_and(|depth| depth.test_enabled || depth.write_enabled),
        )),
        DepthWriteMask: if depth.is_some_and(|depth| depth.write_enabled) {
            windows::Win32::Graphics::Direct3D12::D3D12_DEPTH_WRITE_MASK_ALL
        } else {
            D3D12_DEPTH_WRITE_MASK_ZERO
        },
        DepthFunc: depth.map_or(D3D12_COMPARISON_FUNC_ALWAYS, |depth| {
            if depth.test_enabled {
                depth.compare
            } else {
                D3D12_COMPARISON_FUNC_ALWAYS
            }
        }),
        StencilEnable: BOOL(i32::from(
            stencil.is_some_and(|stencil| stencil.test_enabled),
        )),
        StencilReadMask: stencil.map_or(0xff, |stencil| stencil.front_read_mask),
        StencilWriteMask: stencil.map_or(0xff, |stencil| stencil.front_write_mask),
        FrontFace: stencil.map_or(default_stencil_face, |stencil| stencil_face(stencil.front)),
        BackFace: stencil.map_or(default_stencil_face, |stencil| stencil_face(stencil.back)),
    };
    let mut description = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root.clone())),
        VS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: vertex.as_ptr().cast(),
            BytecodeLength: vertex.len(),
        },
        PS: pixel.map_or_else(D3D12_SHADER_BYTECODE::default, |pixel| {
            D3D12_SHADER_BYTECODE {
                pShaderBytecode: pixel.as_ptr().cast(),
                BytecodeLength: pixel.len(),
            }
        }),
        BlendState: blend,
        SampleMask: u32::MAX,
        RasterizerState: rasterizer,
        DepthStencilState: depth_stencil,
        PrimitiveTopologyType: topology,
        IBStripCutValue: strip_cut,
        NumRenderTargets: targets.len() as u32,
        DSVFormat: depth.map_or(DXGI_FORMAT_UNKNOWN, depth_host_format),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: sample_count,
            Quality: 0,
        },
        ..D3D12_GRAPHICS_PIPELINE_STATE_DESC::default()
    };
    for (host_slot, target) in targets.iter().enumerate() {
        description.RTVFormats[host_slot] = target.descriptor.format;
    }
    let independent_masks = stencil.is_some_and(|stencil| {
        stencil.front_read_mask != stencil.back_read_mask
            || stencil.front_write_mask != stencil.back_write_mask
    });
    let independent_references =
        stencil.is_some_and(|stencil| stencil.front_reference != stencil.back_reference);
    let result = if independent_masks || independent_references {
        if !independent_stencil_ref_mask_supported {
            Err(
                "the selected D3D12 adapter cannot represent distinct front/back stencil refs or masks"
                    .to_owned(),
            )
        } else if independent_masks {
            let stencil = stencil.expect("independent masks require stencil state");
            let advanced = windows::Win32::Graphics::Direct3D12::D3D12_DEPTH_STENCIL_DESC2 {
                DepthEnable: depth_stencil.DepthEnable,
                DepthWriteMask: depth_stencil.DepthWriteMask,
                DepthFunc: depth_stencil.DepthFunc,
                StencilEnable: depth_stencil.StencilEnable,
                FrontFace: stencil_face1(
                    stencil.front,
                    stencil.front_read_mask,
                    stencil.front_write_mask,
                ),
                BackFace: stencil_face1(
                    stencil.back,
                    stencil.back_read_mask,
                    stencil.back_write_mask,
                ),
                DepthBoundsTestEnable: BOOL(0),
            };
            create_independent_stencil_pipeline(device, root, &description, advanced)
        } else {
            // Distinct references are dynamic command-list state and retain the baseline PSO.
            // SAFETY: root signature and shader bytecode remain live for synchronous creation.
            unsafe { device.CreateGraphicsPipelineState(&raw const description) }
                .map_err(|error| graphics_pipeline_creation_error(device, &error))
        }
    } else {
        // SAFETY: root signature and shader bytecode remain live for synchronous PSO creation.
        unsafe { device.CreateGraphicsPipelineState(&raw const description) }
            .map_err(|error| graphics_pipeline_creation_error(device, &error))
    };
    // SAFETY: release the explicit descriptor clone after PSO creation returns.
    unsafe { ManuallyDrop::drop(&mut description.pRootSignature) };
    result
}

fn graphics_pipeline_creation_error(device: &ID3D12Device, error: &windows::core::Error) -> String {
    let mut output =
        backend_error("ID3D12Device::CreateGraphicsPipelineState(guest)", error).to_string();
    if !std::env::var_os("REMU_D3D12_DEBUG").is_some_and(|value| value == "1") {
        return output;
    }
    let Ok(info) = device.cast::<ID3D12InfoQueue>() else {
        return output;
    };
    // SAFETY: the query methods synchronously read messages owned by this device's info queue.
    let count = unsafe { info.GetNumStoredMessages() };
    for index in count.saturating_sub(8)..count {
        let mut byte_count = 0_usize;
        // SAFETY: a null message pointer requests the exact allocation size.
        if unsafe { info.GetMessage(index, None, &raw mut byte_count) }.is_err()
            || byte_count < std::mem::size_of::<D3D12_MESSAGE>()
        {
            continue;
        }
        let mut storage = vec![0_usize; byte_count.div_ceil(std::mem::size_of::<usize>())];
        let message = storage.as_mut_ptr().cast::<D3D12_MESSAGE>();
        // SAFETY: the pointer is usize-aligned and the backing allocation covers byte_count bytes.
        if unsafe { info.GetMessage(index, Some(message), &raw mut byte_count) }.is_err() {
            continue;
        }
        // SAFETY: a successfully returned message owns a description inside the same allocation.
        let description = unsafe {
            let message = &*message;
            std::slice::from_raw_parts(message.pDescription, message.DescriptionByteLength)
        };
        let description = description.strip_suffix(&[0]).unwrap_or(description);
        output.push_str("; debug=");
        output.push_str(&String::from_utf8_lossy(description));
    }
    output
}

fn create_independent_stencil_pipeline(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    legacy: &D3D12_GRAPHICS_PIPELINE_STATE_DESC,
    depth_stencil: windows::Win32::Graphics::Direct3D12::D3D12_DEPTH_STENCIL_DESC2,
) -> Result<ID3D12PipelineState, String> {
    use windows::Win32::Graphics::Direct3D12::*;

    let device2: ID3D12Device2 = device
        .cast()
        .map_err(|error| backend_error("ID3D12Device::cast(ID3D12Device2)", &error).to_string())?;
    let render_target_formats = D3D12_RT_FORMAT_ARRAY {
        RTFormats: legacy.RTVFormats,
        NumRenderTargets: legacy.NumRenderTargets,
    };
    let stream = IndependentStencilPipelineStream {
        root_signature: PipelineStateSubobject::new(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE,
            Interface::as_raw(root),
        ),
        primitive_topology: PipelineStateSubobject::new(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY,
            legacy.PrimitiveTopologyType,
        ),
        index_buffer_strip_cut: PipelineStateSubobject::new(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE,
            legacy.IBStripCutValue,
        ),
        vertex_shader: PipelineStateSubobject::new(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS,
            legacy.VS,
        ),
        pixel_shader: PipelineStateSubobject::new(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS,
            legacy.PS,
        ),
        blend: PipelineStateSubobject::new(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND,
            legacy.BlendState,
        ),
        depth_stencil: PipelineStateSubobject::new(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL2,
            depth_stencil,
        ),
        depth_stencil_format: PipelineStateSubobject::new(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT,
            legacy.DSVFormat,
        ),
        rasterizer: PipelineStateSubobject::new(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER,
            legacy.RasterizerState,
        ),
        render_target_formats: PipelineStateSubobject::new(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS,
            render_target_formats,
        ),
        sample_description: PipelineStateSubobject::new(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC,
            legacy.SampleDesc,
        ),
        sample_mask: PipelineStateSubobject::new(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK,
            legacy.SampleMask,
        ),
    };
    let description = D3D12_PIPELINE_STATE_STREAM_DESC {
        SizeInBytes: std::mem::size_of_val(&stream),
        pPipelineStateSubobjectStream: (&raw const stream).cast_mut().cast(),
    };
    // SAFETY: the stream follows D3D12's pointer-aligned type/value layout and every borrowed
    // shader/root-signature object remains live for this synchronous call.
    unsafe { device2.CreatePipelineState(&raw const description) }.map_err(|error| {
        backend_error("ID3D12Device2::CreatePipelineState(guest)", &error).to_string()
    })
}

fn register_value(registers: &[remu_gpu::GpuRegister], offset: u32) -> Option<u32> {
    registers
        .iter()
        .find(|register| (register.offset & !0x7000_0000) == offset)
        .map(|register| register.value)
}

fn shader_semantics(source: &str) -> String {
    source
        .lines()
        .filter(|line| line.contains(" : SV_") || line.contains(" : TEXCOORD"))
        .take(64)
        .map(str::trim)
        .collect::<Vec<_>>()
        .join(" | ")
}

fn dispatch_group_count(dispatch: &ComputeDispatch) -> Result<[u32; 3], String> {
    let groups = match dispatch.launch.grid {
        ComputeGrid::Threadgroups(groups) => groups,
        ComputeGrid::Threads(threads) => {
            let mut groups = [0_u32; 3];
            for axis in 0..3 {
                let size = u64::from(dispatch.launch.threads_per_group[axis]);
                groups[axis] = u32::try_from(threads[axis].div_ceil(size))
                    .map_err(|_| "compute thread grid exceeds the D3D12 dispatch range")?;
            }
            groups
        }
    };
    for (&count, axis) in groups.iter().zip(['x', 'y', 'z']) {
        if count > D3D12_MAX_DISPATCH_GROUPS_PER_DIMENSION {
            return Err(format!(
                "compute dispatch {axis}-dimension {count} exceeds D3D12's {D3D12_MAX_DISPATCH_GROUPS_PER_DIMENSION}-group limit"
            ));
        }
    }
    Ok(groups)
}

#[cfg(test)]
fn resolve_indirect_compute(dispatch: &ComputeDispatch) -> Result<Option<ComputeDispatch>, String> {
    resolve_indirect_compute_ordered(dispatch, &OrderedGlobalVersions::default())
}

fn resolve_indirect_compute_ordered(
    dispatch: &ComputeDispatch,
    ordered_versions: &OrderedGlobalVersions,
) -> Result<Option<ComputeDispatch>, String> {
    let ComputeDispatchSource::Indirect { dimensions_address } = dispatch.source else {
        return Ok(None);
    };
    let packet_dimensions = encoded_dispatch_dimensions(dispatch)?;
    let mut bytes = packet_dimensions.map(u32::to_le_bytes).concat();
    if let Some(forwarded) = ordered_versions.forwarded_slice(dimensions_address, &bytes) {
        bytes.copy_from_slice(forwarded);
        if std::env::var_os("REMU_TRACE_D3D12_BINDINGS").is_some_and(|value| value == "1") {
            eprintln!(
                "[REMu][D3D12-BINDINGS] forwarded ordered indirect compute address={dimensions_address:#x} packet=[{}] gpu=[{}]",
                preview_words(&packet_dimensions.map(u32::to_le_bytes).concat()),
                preview_words(&bytes),
            );
        }
    } else {
        read_process_memory(dimensions_address, &mut bytes).map_err(|error| {
            format!("indirect compute dimensions at {dimensions_address:#x}: {error}")
        })?;
    }
    let dimensions = std::array::from_fn(|axis| {
        let offset = axis * 4;
        u32::from_le_bytes(
            bytes[offset..offset + 4]
                .try_into()
                .expect("compute dimension dword"),
        )
    });
    let mut resolved = dispatch.clone();
    resolved.launch = remu_gpu::decode_compute_launch(
        dimensions,
        dispatch.launch.initiator,
        &dispatch.shader_registers,
    )
    .map_err(|error| format!("indirect compute dimensions at {dimensions_address:#x}: {error}"))?;
    Ok(Some(resolved))
}

fn encoded_dispatch_dimensions(dispatch: &ComputeDispatch) -> Result<[u32; 3], String> {
    const DISPATCH_USE_THREAD_DIMENSIONS: u32 = 1 << 5;
    let mut dimensions = [0_u32; 3];
    for (axis, dimension) in dimensions.iter_mut().enumerate() {
        let value = if dispatch.launch.initiator & DISPATCH_USE_THREAD_DIMENSIONS != 0 {
            let ComputeGrid::Threads(threads) = dispatch.launch.grid else {
                return Err("thread-dimension dispatch lost its exact thread grid".to_owned());
            };
            u64::from(dispatch.launch.base_group[axis])
                .checked_mul(u64::from(dispatch.launch.threads_per_group[axis]))
                .and_then(|start| start.checked_add(threads[axis]))
                .ok_or("indirect compute packet dimensions overflow")?
        } else {
            u64::from(dispatch.launch.base_group[axis])
                .checked_add(u64::from(dispatch.launch.group_count[axis]))
                .ok_or("indirect compute packet dimensions overflow")?
        };
        *dimension =
            u32::try_from(value).map_err(|_| "indirect compute packet dimension exceeds u32")?;
    }
    Ok(dimensions)
}

fn dispatch_root_constants(dispatch: &ComputeDispatch) -> [u32; 7] {
    let limit = match dispatch.launch.grid {
        ComputeGrid::Threadgroups(_) => [0; 3],
        ComputeGrid::Threads(threads) => {
            threads.map(|value| u32::try_from(value).unwrap_or(u32::MAX))
        }
    };
    [
        0,
        limit[0],
        limit[1],
        limit[2],
        dispatch.launch.base_group[0],
        dispatch.launch.base_group[1],
        dispatch.launch.base_group[2],
    ]
}

fn create_stage_resources(
    device: &ID3D12Device,
    upload_arena: &mut UploadArena,
    evaluation: &ShaderEvaluation,
    live_scalars: Option<(&[remu_gpu::GpuRegister], u32, usize)>,
    resident_targets: &HashMap<u64, RenderTargetResource>,
    sampled_texture_cache: &mut HashMap<DecodedTexture, CachedSampledTexture>,
    excluded_target_addresses: &[u64],
    ordered_versions: &OrderedGlobalVersions,
) -> Result<ComputeResources, String> {
    let mut scalar_words = evaluation.initial_scalar_registers.to_vec();
    if let Some((registers, user_base, scalar_base)) = live_scalars {
        let scalar_end = scalar_base.saturating_add(64).min(scalar_words.len());
        for scalar in scalar_base..scalar_end {
            let offset = u32::try_from(scalar - scalar_base)
                .map_err(|_| "live scalar-register offset exceeds u32")?;
            if let Some(value) = register_value(registers, user_base + offset) {
                scalar_words[scalar] = value;
            }
        }
    }
    for binding in &evaluation.global_memory_bindings {
        scalar_words.extend([
            binding.base_address as u32,
            (binding.base_address >> 32) as u32,
            u32::try_from(binding.byte_count)
                .map_err(|_| "global binding exceeds the shader's 32-bit byte range")?,
            u32::from(binding.writable),
        ]);
    }
    let mut global_snapshots = Vec::with_capacity(evaluation.global_memory_bindings.len());
    for binding in &evaluation.global_memory_bindings {
        if binding.byte_count > MAXIMUM_GLOBAL_BUFFER_BYTES {
            return Err(format!(
                "global binding at {:#x} is {} bytes, over the {}-byte D3D12 safety limit",
                binding.base_address, binding.byte_count, MAXIMUM_GLOBAL_BUFFER_BYTES
            ));
        }
        global_snapshots.push(snapshot_ordered_global(binding, ordered_versions)?);
    }

    let slots = evaluation.image_binding_slots();
    let mut texture_bindings = vec![None; slots.texture_count as usize];
    let mut sampler_bindings = vec![None; slots.sampler_count as usize];
    for (index, binding) in evaluation.image_bindings.iter().enumerate() {
        texture_bindings[slots.texture_slots[index] as usize].get_or_insert(binding);
        if let Some(slot) = slots.sampler_slots[index] {
            sampler_bindings[slot as usize].get_or_insert(binding);
        }
    }
    let textures = texture_bindings
        .into_iter()
        .map(|binding| {
            let binding = binding.ok_or("non-dense canonical texture slots")?;
            let descriptor = decode_texture(binding)?;
            if !excluded_target_addresses.contains(&descriptor.address)
                && let Some(target) = resident_targets
                    .get(&descriptor.address)
                    .filter(|target| render_target_can_alias_texture(target, descriptor))
            {
                if std::env::var_os("REMU_TRACE_D3D12_TIMING").is_some_and(|value| value == "1") {
                    eprintln!(
                        "[REMu][D3D12-TIMING] alias_target={:#x} extent={}x{}",
                        descriptor.address, descriptor.width, descriptor.height
                    );
                }
                Ok(create_resident_render_target_texture(target, descriptor))
            } else {
                create_or_reuse_sampled_texture(device, descriptor, sampled_texture_cache)
            }
        })
        .collect::<Result<Vec<_>, String>>()?;

    let mut binding_words = vec![1_u32];
    binding_words.extend((0..global_snapshots.len()).map(|index| {
        u32::try_from(index + 2).expect("descriptor count is bounded by host memory")
    }));
    binding_words.extend((0..textures.len()).map(|index| {
        u32::try_from(index + 2 + global_snapshots.len())
            .expect("descriptor count is bounded by host memory")
    }));
    binding_words.extend(
        (0..sampler_bindings.len())
            .map(|index| u32::try_from(index).expect("sampler descriptor count fits u32")),
    );
    let binding_table_bytes = words_as_bytes(&binding_words);
    let scalar_bytes = words_as_bytes(&scalar_words);
    let binding_table_offset = 0_usize;
    let initial_scalar_offset = align_up(binding_table_bytes.len().max(4), 4)?;
    let mut arena_size = align_up(
        initial_scalar_offset
            .checked_add(scalar_bytes.len().max(4))
            .ok_or("stage upload arena size overflows")?,
        D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT as usize,
    )?;
    let mut global_offsets = Vec::with_capacity(global_snapshots.len());
    for binding in &evaluation.global_memory_bindings {
        global_offsets.push(arena_size);
        arena_size = align_up(
            arena_size
                .checked_add(binding.byte_count.max(4))
                .ok_or("stage global upload range overflows")?,
            D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT as usize,
        )?;
    }
    let mut arena_bytes = vec![0_u8; arena_size.max(4)];
    arena_bytes[binding_table_offset..binding_table_offset + binding_table_bytes.len()]
        .copy_from_slice(&binding_table_bytes);
    arena_bytes[initial_scalar_offset..initial_scalar_offset + scalar_bytes.len()]
        .copy_from_slice(&scalar_bytes);
    for ((bytes, &offset), binding) in global_snapshots
        .iter()
        .zip(&global_offsets)
        .zip(&evaluation.global_memory_bindings)
    {
        arena_bytes[offset..offset + bytes.len()].copy_from_slice(bytes);
        debug_assert!(bytes.len() <= binding.byte_count.max(4));
    }
    let (stage_upload, arena_base) =
        match upload_arena.allocate(&arena_bytes, D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT as usize)? {
            Some(allocation) => allocation,
            None => (create_upload_buffer(device, &arena_bytes)?, 0),
        };
    let mut globals = Vec::with_capacity(evaluation.global_memory_bindings.len());
    for ((binding, &offset), original_bytes) in evaluation
        .global_memory_bindings
        .iter()
        .zip(&global_offsets)
        .zip(global_snapshots.into_iter())
    {
        let byte_count = binding.byte_count.max(4);
        let writable = binding.writable;
        let absolute_offset = arena_base
            .checked_add(offset)
            .ok_or("stage upload offset overflows")?;
        let writeback_guard = if binding.write_back_to_guest && binding.byte_count != 0 {
            if binding.base_address < 0x10_000 {
                return Err(format!(
                    "global writeback has invalid guest address {:#x}",
                    binding.base_address
                ));
            }
            let guest_byte_count = u64::try_from(binding.byte_count)
                .map_err(|_| "writable global buffer byte count does not fit u64")?;
            if let Some(lease) = GuestRangeLease::acquire(binding.base_address, guest_byte_count) {
                Some(GlobalWritebackGuard::Lease(lease))
            } else {
                #[cfg(test)]
                {
                    Some(GlobalWritebackGuard::Direct)
                }
                #[cfg(not(test))]
                {
                    // SAFETY: shader evaluation accepted this exact range as live readable and
                    // writable guest memory. Mapping replacement retires the watch before the
                    // address can be reused, and publication holds its registry lock.
                    let watch = unsafe {
                        CpuWriteWatch::new(binding.base_address, guest_byte_count)
                    }
                    .map_err(|error| {
                        format!(
                            "could not pin D3D12 writable global buffer address={:#x} bytes={}: {error}",
                            binding.base_address, binding.byte_count
                        )
                    })?;
                    Some(GlobalWritebackGuard::Anonymous(watch))
                }
            }
        } else {
            None
        };
        globals.push(GlobalResource {
            gpu: if writable {
                create_gpu_buffer(device, byte_count)?
            } else {
                // All read-only globals share one upload allocation and select disjoint raw-SRV
                // ranges. Their guest addresses remain per-binding scalar metadata.
                stage_upload.clone()
            },
            gpu_offset: if writable { 0 } else { absolute_offset },
            upload: writable.then(|| stage_upload.clone()),
            upload_offset: absolute_offset,
            // A writable null/transient binding may intentionally have no guest destination.
            // Only bindings whose evaluated contract requests guest publication need a readback
            // resource or a per-submission CPU visibility point.
            readback: binding
                .write_back_to_guest
                .then(|| create_readback_buffer(device, byte_count))
                .transpose()?,
            byte_count: binding.byte_count,
            writable,
            original_bytes: if writeback_guard.is_some() {
                original_bytes
            } else {
                Vec::new()
            },
            writeback_guard,
        });
    }
    let samplers = sampler_bindings
        .into_iter()
        .map(|binding| decode_sampler(binding.ok_or("non-dense canonical sampler slots")?))
        .collect::<Result<Vec<_>, String>>()?;
    Ok(ComputeResources {
        binding_table: stage_upload.clone(),
        binding_table_offset: arena_base
            .checked_add(binding_table_offset)
            .ok_or("binding-table upload offset overflows")?,
        binding_table_word_count: binding_words.len(),
        initial_scalars: stage_upload,
        initial_scalar_offset: arena_base
            .checked_add(initial_scalar_offset)
            .ok_or("initial-scalar upload offset overflows")?,
        initial_scalar_word_count: scalar_words.len(),
        globals,
        samplers,
        textures,
        binding_words,
        global_count: evaluation.global_memory_bindings.len(),
        texture_count: slots.texture_count as usize,
        sampler_count: slots.sampler_count as usize,
    })
}

fn create_shader_visible_heap(
    device: &ID3D12Device,
    heap_type: windows::Win32::Graphics::Direct3D12::D3D12_DESCRIPTOR_HEAP_TYPE,
    descriptor_count: usize,
    label: &str,
) -> Result<ID3D12DescriptorHeap, String> {
    let description = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: heap_type,
        NumDescriptors: u32::try_from(descriptor_count).map_err(|_| format!("too many {label}"))?,
        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
        NodeMask: 0,
    };
    // SAFETY: the caller supplies a supported shader-visible heap type and nonzero capacity.
    unsafe { device.CreateDescriptorHeap(&raw const description) }.map_err(|error| {
        format!(
            "{label}: {}",
            backend_error("ID3D12Device::CreateDescriptorHeap", &error)
        )
    })
}

fn combine_stage_heaps(
    device: &ID3D12Device,
    arena: &mut DescriptorArena,
    stages: &mut [&mut ComputeResources],
) -> Result<(ID3D12DescriptorHeap, Option<ID3D12DescriptorHeap>, Vec<u32>), String> {
    let descriptor_counts = stages
        .iter()
        .map(|stage| 2 + stage.global_count + stage.texture_count)
        .collect::<Vec<_>>();
    let sampler_counts = stages
        .iter()
        .map(|stage| stage.sampler_count)
        .collect::<Vec<_>>();
    let total_descriptors = descriptor_counts.iter().sum::<usize>();
    let total_samplers = sampler_counts.iter().sum::<usize>();
    let shared_allocation = arena.allocate(total_descriptors, total_samplers);
    let (descriptors, sampler_descriptors, descriptor_allocation_base, sampler_allocation_base) =
        if let Some((descriptor_base, sampler_base)) = shared_allocation {
            (
                arena.descriptors.clone(),
                (total_samplers != 0).then(|| arena.samplers.clone()),
                descriptor_base,
                sampler_base,
            )
        } else {
            (
                create_shader_visible_heap(
                    device,
                    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    total_descriptors,
                    "graphics descriptors",
                )?,
                (total_samplers != 0)
                    .then(|| {
                        create_shader_visible_heap(
                            device,
                            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                            total_samplers,
                            "graphics samplers",
                        )
                    })
                    .transpose()?,
                0,
                0,
            )
        };
    let descriptor_increment = usize::try_from(unsafe {
        device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
    })
    .expect("descriptor increment fits usize");
    let sampler_increment = usize::try_from(unsafe {
        device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER)
    })
    .expect("sampler increment fits usize");
    let descriptor_start = unsafe { descriptors.GetCPUDescriptorHandleForHeapStart() };
    let sampler_start = sampler_descriptors
        .as_ref()
        .map(|heap| unsafe { heap.GetCPUDescriptorHandleForHeapStart() });
    let mut descriptor_base = descriptor_allocation_base;
    let mut sampler_base = sampler_allocation_base;
    let mut root_descriptors = Vec::with_capacity(stages.len());
    for ((stage, &descriptor_count), &sampler_count) in stages
        .iter_mut()
        .zip(&descriptor_counts)
        .zip(&sampler_counts)
    {
        for word in &mut stage.binding_words[..1 + stage.global_count + stage.texture_count] {
            *word = word
                .checked_add(
                    u32::try_from(descriptor_base).map_err(|_| "descriptor base exceeds u32")?,
                )
                .ok_or("descriptor-table index overflows")?;
        }
        for word in &mut stage.binding_words[1 + stage.global_count + stage.texture_count..] {
            *word = word
                .checked_add(u32::try_from(sampler_base).map_err(|_| "sampler base exceeds u32")?)
                .ok_or("sampler-table index overflows")?;
        }
        write_upload_at(
            &stage.binding_table,
            stage.binding_table_offset,
            &words_as_bytes(&stage.binding_words),
        )
        .map_err(|error| error.to_string())?;
        let mut handle = descriptor_start;
        handle.ptr += descriptor_base * descriptor_increment;
        create_structured_srv(
            device,
            &stage.binding_table,
            stage.binding_table_offset / 4,
            stage.binding_table_word_count,
            handle,
        )?;
        handle.ptr += descriptor_increment;
        create_structured_srv(
            device,
            &stage.initial_scalars,
            stage.initial_scalar_offset / 4,
            stage.initial_scalar_word_count,
            handle,
        )?;
        for resource in &stage.globals {
            handle.ptr += descriptor_increment;
            if resource.writable {
                create_raw_uav(device, &resource.gpu, resource.byte_count.max(4), handle)?;
            } else {
                create_raw_srv(
                    device,
                    &resource.gpu,
                    resource.gpu_offset,
                    resource.byte_count.max(4),
                    handle,
                )?;
            }
        }
        for resource in &stage.textures {
            handle.ptr += descriptor_increment;
            create_texture_view(device, resource, handle)?;
        }
        if sampler_count != 0 {
            let mut handle = sampler_start.expect("nonzero sampler total has a heap");
            handle.ptr += sampler_base * sampler_increment;
            for (index, sampler) in stage.samplers.iter().enumerate() {
                // SAFETY: each handle selects one initialized slot in the shader-visible heap.
                unsafe { device.CreateSampler(sampler, handle) };
                if trace_d3d12_bindings() {
                    ensure_device_available(
                        device,
                        &format!("sampler descriptor {index} {sampler:?}"),
                    )?;
                }
                handle.ptr += sampler_increment;
            }
        }
        root_descriptors
            .push(u32::try_from(descriptor_base).map_err(|_| "descriptor base exceeds u32")?);
        descriptor_base += descriptor_count;
        sampler_base += sampler_count;
    }
    Ok((descriptors, sampler_descriptors, root_descriptors))
}

fn decode_render_targets(draw: &IndexedDraw) -> Result<Vec<DecodedRenderTarget>, String> {
    let pixel = match draw.pixel_shader.as_ref() {
        Some(pixel) => pixel,
        None => return Ok(Vec::new()),
    };
    let exports = pixel.program.pixel_color_export_masks();
    if let Some(scanout) = draw.scanout_target {
        let mask = (exports >> (scanout.export_slot * 4)) & scanout.write_mask & 0xf;
        if mask == 0 {
            return Err("scanout target has no matching pixel export".to_owned());
        }
        let tile_mode = match scanout.tiling_mode {
            0 => 27,
            1 => 0,
            mode => return Err(format!("unsupported scanout tiling mode {mode}")),
        };
        return Ok(vec![DecodedRenderTarget {
            guest_slot: scanout.export_slot,
            descriptor: DecodedTexture {
                address: scanout.address,
                width: scanout.width,
                height: scanout.height,
                pitch: scanout.pitch,
                tile_mode,
                format: DXGI_FORMAT_B8G8R8A8_UNORM,
                bytes_per_element: 4,
                block_extent: [1, 1],
                texture_type: 9,
                layer_count: 1,
                mip_level_count: 1,
                writable: true,
            },
            output_kind: remu_shader::PixelOutputKind::Float,
            write_mask: mask,
            sample_count: 1,
        }]);
    }
    let target_mask = register_value(&draw.context_registers, 0x08e).unwrap_or_default();
    let mut targets = Vec::new();
    for slot in 0..8_u32 {
        let write_mask = ((target_mask & exports) >> (slot * 4)) & 0xf;
        if write_mask == 0 {
            continue;
        }
        let primary = slot * 15;
        let required = |offset| {
            register_value(&draw.context_registers, offset)
                .ok_or_else(|| format!("color target {slot} is missing register {offset:#x}"))
        };
        let base = required(0x318 + primary)?;
        let base_ext = required(0x390 + slot)?;
        let info = required(0x31c + primary)?;
        let attrib = required(0x31d + primary)?;
        let attrib2 = required(0x3b0 + slot)?;
        let attrib3 = required(0x3b8 + slot)?;
        let address = (u64::from(base_ext & 0xff) << 40) | (u64::from(base) << 8);
        let width = ((attrib2 >> 14) & 0x3fff) + 1;
        let height = (attrib2 & 0x3fff) + 1;
        let sample_log2 = (attrib >> 12) & 7;
        let fragment_log2 = (attrib >> 15) & 3;
        if sample_log2 != fragment_log2 || sample_log2 > 3 {
            return Err(format!(
                "color target {slot} has unrepresentable sample/fragment counts ({sample_log2}/{fragment_log2})"
            ));
        }
        let sample_count = 1_u32 << sample_log2;
        if info & ((1 << 14) | (1 << 28)) != 0 {
            return Err("FMASK/DCC guest render targets require metadata decompression".to_owned());
        }
        let data_format = (info >> 2) & 0x1f;
        let number_format = (info >> 8) & 7;
        let component_swap = (info >> 11) & 3;
        let (mut format, bytes, block_extent) = decode_dxgi_format(data_format, number_format)?;
        if data_format == 10 && component_swap == 1 && number_format == 0 {
            format = DXGI_FORMAT_B8G8R8A8_UNORM;
        } else if component_swap != 0 {
            return Err(format!("unsupported color component swap {component_swap}"));
        }
        let tile_mode = (attrib3 >> 14) & 0x1f;
        if !matches!(tile_mode, 0 | 27) {
            return Err(format!("unsupported color-target tile mode {tile_mode}"));
        }
        let pitch = if tile_mode == 0 {
            let alignment = (256 / bytes.max(1)).max(1);
            (width + alignment - 1) & !(alignment - 1)
        } else {
            width
        };
        let output_kind = match number_format {
            4 => remu_shader::PixelOutputKind::Uint,
            5 => remu_shader::PixelOutputKind::Sint,
            _ => remu_shader::PixelOutputKind::Float,
        };
        targets.push(DecodedRenderTarget {
            guest_slot: slot,
            descriptor: DecodedTexture {
                address,
                width,
                height,
                pitch,
                tile_mode,
                format,
                bytes_per_element: bytes,
                block_extent,
                texture_type: 9,
                layer_count: 1,
                mip_level_count: 1,
                writable: true,
            },
            output_kind,
            write_mask,
            sample_count,
        });
    }
    Ok(targets)
}

fn decode_depth_target(draw: &IndexedDraw) -> Result<Option<DecodedDepthTarget>, String> {
    let control = register_value(&draw.context_registers, 0x200).unwrap_or_default();
    let render_control = register_value(&draw.context_registers, 0x000).unwrap_or_default();
    let test_enabled = control & (1 << 1) != 0;
    let write_enabled = control & (1 << 2) != 0;
    let clear_enabled = render_control & 1 != 0;
    let stencil_enabled = control & 1 != 0;
    let stencil_clear_enabled = render_control & (1 << 1) != 0;
    if !test_enabled
        && !write_enabled
        && !clear_enabled
        && !stencil_enabled
        && !stencil_clear_enabled
    {
        return Ok(None);
    }
    let required = |offset, name| {
        register_value(&draw.context_registers, offset)
            .ok_or_else(|| format!("active depth target is missing {name}"))
    };
    let z_info = required(0x010, "DB_Z_INFO")?;
    let depth_format = match z_info & 3 {
        0 if stencil_enabled || stencil_clear_enabled => None,
        0 => return Ok(None),
        1 => (
            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D16_UNORM,
            2,
        )
            .into(),
        3 => (
            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D32_FLOAT,
            4,
        )
            .into(),
        value => return Err(format!("unsupported GFX10 depth format {value}")),
    };
    let sample_count = 1_u32 << ((z_info >> 2) & 3);
    let tile_index = (z_info >> 20) & 7;
    if tile_index != 0 {
        return Err(format!(
            "unsupported GFX10 depth tile-mode index {tile_index}"
        ));
    }
    let size = required(0x007, "DB_DEPTH_SIZE_XY")?;
    let width = (size & 0x3fff) + 1;
    let height = ((size >> 16) & 0x3fff) + 1;
    let view = register_value(&draw.context_registers, 0x002).unwrap_or_default();
    let slice_start = view & 0x7ff;
    let slice_max = (view >> 13) & 0x7ff;
    let mip_level = (view >> 26) & 0xf;
    if slice_start > slice_max || mip_level != 0 {
        return Err(format!("unsupported GFX10 depth view {view:#010x}"));
    }
    let layer_count = slice_max - slice_start + 1;
    if layer_count != 1 {
        return Err(format!(
            "layered D3D12 depth targets ({layer_count} layers) are not implemented yet"
        ));
    }
    let depth_read_address =
        (u64::from(register_value(&draw.context_registers, 0x01a).unwrap_or_default() & 0xff)
            << 40)
            | (u64::from(register_value(&draw.context_registers, 0x012).unwrap_or_default()) << 8);
    let depth_write_address =
        (u64::from(register_value(&draw.context_registers, 0x01c).unwrap_or_default() & 0xff)
            << 40)
            | (u64::from(register_value(&draw.context_registers, 0x014).unwrap_or_default()) << 8);
    if depth_read_address != 0
        && depth_write_address != 0
        && depth_read_address != depth_write_address
    {
        return Err(format!(
            "separate depth read/write surfaces are unsupported ({depth_read_address:#x}/{depth_write_address:#x})"
        ));
    }
    let depth_base_address = if write_enabled && depth_write_address != 0 {
        depth_write_address
    } else if depth_read_address != 0 {
        depth_read_address
    } else {
        depth_write_address
    };
    if depth_format.is_some() && depth_base_address < 0x10_000 || width > 16_384 || height > 16_384
    {
        return Err(format!(
            "invalid depth target address/extent {depth_base_address:#x} {width}x{height}"
        ));
    }
    let (format, bytes_per_element) = depth_format.unwrap_or((
        windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D32_FLOAT,
        4,
    ));
    let descriptor = decode_depth_surface_descriptor(
        depth_base_address,
        width,
        height,
        slice_start,
        format,
        bytes_per_element,
        depth_format.is_some() && (write_enabled || clear_enabled),
    )?;
    let compare = decode_comparison((control >> 4) & 7);
    let clear_depth = register_value(&draw.context_registers, 0x00b)
        .map(f32::from_bits)
        .filter(|depth| depth.is_finite() && (0.0..=1.0).contains(depth))
        .unwrap_or(1.0);
    let stencil = (stencil_enabled || stencil_clear_enabled)
        .then(|| {
            decode_stencil_target(
                draw,
                width,
                height,
                slice_start,
                stencil_enabled,
                stencil_clear_enabled,
                control & (1 << 7) != 0,
            )
        })
        .transpose()?;
    Ok(Some(DecodedDepthTarget {
        descriptor,
        compare,
        test_enabled,
        write_enabled,
        clear_enabled,
        clear_depth,
        sample_count,
        stencil,
    }))
}

fn decode_comparison(value: u32) -> windows::Win32::Graphics::Direct3D12::D3D12_COMPARISON_FUNC {
    match value & 7 {
        0 => windows::Win32::Graphics::Direct3D12::D3D12_COMPARISON_FUNC_NEVER,
        1 => windows::Win32::Graphics::Direct3D12::D3D12_COMPARISON_FUNC_LESS,
        2 => windows::Win32::Graphics::Direct3D12::D3D12_COMPARISON_FUNC_EQUAL,
        3 => windows::Win32::Graphics::Direct3D12::D3D12_COMPARISON_FUNC_LESS_EQUAL,
        4 => windows::Win32::Graphics::Direct3D12::D3D12_COMPARISON_FUNC_GREATER,
        5 => windows::Win32::Graphics::Direct3D12::D3D12_COMPARISON_FUNC_NOT_EQUAL,
        6 => windows::Win32::Graphics::Direct3D12::D3D12_COMPARISON_FUNC_GREATER_EQUAL,
        _ => windows::Win32::Graphics::Direct3D12::D3D12_COMPARISON_FUNC_ALWAYS,
    }
}

fn decode_stencil_operation(
    value: u32,
) -> Result<windows::Win32::Graphics::Direct3D12::D3D12_STENCIL_OP, String> {
    match value & 0xf {
        0 => Ok(D3D12_STENCIL_OP_KEEP),
        1 => Ok(D3D12_STENCIL_OP_ZERO),
        3 | 4 => Ok(D3D12_STENCIL_OP_REPLACE),
        5 => Ok(D3D12_STENCIL_OP_INCR_SAT),
        6 => Ok(D3D12_STENCIL_OP_DECR_SAT),
        7 => Ok(D3D12_STENCIL_OP_INVERT),
        8 => Ok(D3D12_STENCIL_OP_INCR),
        9 => Ok(D3D12_STENCIL_OP_DECR),
        value => Err(format!(
            "GFX10 stencil operation {value} requires logical/op-value emulation"
        )),
    }
}

fn decode_depth_surface_descriptor(
    base_address: u64,
    width: u32,
    height: u32,
    slice_start: u32,
    format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    bytes_per_element: u32,
    writable: bool,
) -> Result<DecodedTexture, String> {
    let block_elements = 65_536_usize / bytes_per_element as usize;
    let block_width = 1_usize << (block_elements.trailing_zeros() as usize).div_ceil(2);
    let pitch = (width as usize).div_ceil(block_width) * block_width;
    let layer_bytes = pitch
        .div_ceil(block_width)
        .checked_mul((height as usize).div_ceil(block_elements / block_width))
        .and_then(|blocks| blocks.checked_mul(65_536))
        .ok_or("depth/stencil target layer size overflows")?;
    let address = if base_address == 0 {
        0
    } else {
        base_address
            .checked_add(
                u64::try_from(layer_bytes)
                    .map_err(|_| "depth/stencil layer size exceeds u64")?
                    .checked_mul(u64::from(slice_start))
                    .ok_or("depth/stencil slice offset overflows")?,
            )
            .ok_or("depth/stencil slice address overflows")?
    };
    Ok(DecodedTexture {
        address,
        width,
        height,
        pitch: pitch as u32,
        tile_mode: 24,
        format,
        bytes_per_element,
        block_extent: [1, 1],
        texture_type: 9,
        layer_count: 1,
        mip_level_count: 1,
        writable,
    })
}

fn decode_stencil_target(
    draw: &IndexedDraw,
    width: u32,
    height: u32,
    slice_start: u32,
    test_enabled: bool,
    clear_enabled: bool,
    backface_enabled: bool,
) -> Result<DecodedStencilTarget, String> {
    let required = |offset, name| {
        register_value(&draw.context_registers, offset)
            .ok_or_else(|| format!("active stencil target is missing {name}"))
    };
    let info = required(0x011, "DB_STENCIL_INFO")?;
    if info & 1 == 0 {
        return Err(format!(
            "active stencil target has invalid format in {info:#010x}"
        ));
    }
    let tile_mode = (info >> 4) & 0x1f;
    if tile_mode != 24 {
        return Err(format!("unsupported GFX10 stencil SW_MODE {tile_mode}"));
    }
    let read_address =
        (u64::from(register_value(&draw.context_registers, 0x01b).unwrap_or_default() & 0xff)
            << 40)
            | (u64::from(register_value(&draw.context_registers, 0x013).unwrap_or_default()) << 8);
    let write_address =
        (u64::from(register_value(&draw.context_registers, 0x01d).unwrap_or_default() & 0xff)
            << 40)
            | (u64::from(register_value(&draw.context_registers, 0x015).unwrap_or_default()) << 8);
    if read_address != 0 && write_address != 0 && read_address != write_address {
        return Err(format!(
            "separate stencil read/write surfaces are unsupported ({read_address:#x}/{write_address:#x})"
        ));
    }
    let base_address = if write_address != 0 {
        write_address
    } else {
        read_address
    };
    if base_address < 0x10_000 {
        return Err(format!("invalid stencil target address {base_address:#x}"));
    }
    let operation = register_value(&draw.context_registers, 0x10b).unwrap_or_default();
    let front_mask = register_value(&draw.context_registers, 0x10c).unwrap_or(0x00ff_ff00);
    let back_mask = if backface_enabled {
        register_value(&draw.context_registers, 0x10d).unwrap_or(front_mask)
    } else {
        front_mask
    };
    let front_reference = front_mask as u8;
    let back_reference = back_mask as u8;
    let front_read_mask = (front_mask >> 8) as u8;
    let back_read_mask = (back_mask >> 8) as u8;
    let front_write_mask = (front_mask >> 16) as u8;
    let back_write_mask = (back_mask >> 16) as u8;
    let face = |compare, shift| -> Result<DecodedStencilFace, String> {
        Ok(DecodedStencilFace {
            compare: decode_comparison(compare),
            fail: decode_stencil_operation(operation >> shift)?,
            pass: decode_stencil_operation(operation >> (shift + 4))?,
            depth_fail: decode_stencil_operation(operation >> (shift + 8))?,
        })
    };
    let front = face(
        register_value(&draw.context_registers, 0x200).unwrap_or_default() >> 8,
        0,
    )?;
    let back = if backface_enabled {
        face(
            register_value(&draw.context_registers, 0x200).unwrap_or_default() >> 20,
            12,
        )?
    } else {
        front
    };
    let write_enabled = front_write_mask != 0
        && [
            front.fail,
            front.pass,
            front.depth_fail,
            back.fail,
            back.pass,
            back.depth_fail,
        ]
        .into_iter()
        .any(|value| value != D3D12_STENCIL_OP_KEEP);
    Ok(DecodedStencilTarget {
        descriptor: decode_depth_surface_descriptor(
            base_address,
            width,
            height,
            slice_start,
            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8_UINT,
            1,
            write_enabled || clear_enabled,
        )?,
        front,
        back,
        front_reference,
        back_reference,
        front_read_mask,
        back_read_mask,
        front_write_mask,
        back_write_mask,
        test_enabled,
        clear_enabled,
        clear_stencil: register_value(&draw.context_registers, 0x00a).unwrap_or_default() as u8,
    })
}

fn depth_host_format(
    depth: DecodedDepthTarget,
) -> windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT {
    if depth.stencil.is_some() {
        DXGI_FORMAT_D32_FLOAT_S8X24_UINT
    } else {
        depth.descriptor.format
    }
}

fn depth_typeless_format(
    depth: DecodedDepthTarget,
) -> Result<windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT, String> {
    if depth.stencil.is_some() {
        return Ok(DXGI_FORMAT_R32G8X24_TYPELESS);
    }
    match depth.descriptor.format {
        windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D16_UNORM => {
            Ok(DXGI_FORMAT_R16_TYPELESS)
        }
        windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D32_FLOAT => {
            Ok(DXGI_FORMAT_R32_TYPELESS)
        }
        format => Err(format!(
            "depth format {format:?} has no implemented typeless MSAA resource"
        )),
    }
}

fn depth_srv_format(
    depth: DecodedDepthTarget,
) -> Result<windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT, String> {
    if depth.stencil.is_some() {
        return Ok(DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS);
    }
    match depth.descriptor.format {
        windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D16_UNORM => Ok(DXGI_FORMAT_R16_UNORM),
        windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D32_FLOAT => Ok(DXGI_FORMAT_R32_FLOAT),
        format => Err(format!(
            "depth format {format:?} has no implemented MSAA shader view"
        )),
    }
}

fn stencil_srv_format(
    depth: DecodedDepthTarget,
) -> Result<windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT, String> {
    depth
        .stencil
        .map(|_| DXGI_FORMAT_X32_TYPELESS_G8X24_UINT)
        .ok_or("depth target has no stencil plane".to_owned())
}

fn render_sample_count(
    targets: &[DecodedRenderTarget],
    depth: Option<DecodedDepthTarget>,
) -> Result<u32, String> {
    let sample_count = targets.first().map_or_else(
        || depth.map_or(1, |target| target.sample_count),
        |target| target.sample_count,
    );
    if let Some(target) = targets
        .iter()
        .find(|target| target.sample_count != sample_count)
    {
        return Err(format!(
            "color target {} uses {} samples while the render pass uses {sample_count}",
            target.guest_slot, target.sample_count
        ));
    }
    if let Some(depth) = depth
        && depth.sample_count != sample_count
    {
        return Err(format!(
            "depth target uses {} samples while the render pass uses {sample_count}",
            depth.sample_count
        ));
    }
    Ok(sample_count)
}

fn create_render_target_resource(
    device: &ID3D12Device,
    target: DecodedRenderTarget,
) -> Result<RenderTargetResource, String> {
    let descriptor = target.descriptor;
    if descriptor.block_extent != [1, 1] {
        return Err("compressed resources cannot be color attachments".to_owned());
    }
    let integer_msaa =
        target.sample_count > 1 && target.output_kind != remu_shader::PixelOutputKind::Float;
    if target.sample_count > 1 {
        let mut quality = D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS {
            Format: descriptor.format,
            SampleCount: target.sample_count,
            Flags: D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE,
            NumQualityLevels: 0,
        };
        // SAFETY: `quality` is the exact structure for this feature query.
        unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
                (&raw mut quality).cast(),
                u32::try_from(std::mem::size_of_val(&quality))
                    .expect("multisample query size fits u32"),
            )
        }
        .map_err(|error| {
            backend_error("ID3D12Device::CheckFeatureSupport(MSAA)", &error).to_string()
        })?;
        if quality.NumQualityLevels == 0 {
            return Err(format!(
                "D3D12 adapter does not support {}x MSAA for {:?}",
                target.sample_count, descriptor.format
            ));
        }
    }
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..D3D12_HEAP_PROPERTIES::default()
    };
    let description = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: u64::from(descriptor.width),
        Height: descriptor.height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: descriptor.format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: target.sample_count,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        ..D3D12_RESOURCE_DESC::default()
    };
    let mut gpu = None;
    // SAFETY: validated descriptor and output slot are live for synchronous creation.
    unsafe {
        device.CreateCommittedResource(
            &raw const heap,
            D3D12_HEAP_FLAG_NONE,
            &raw const description,
            if target.sample_count > 1 {
                D3D12_RESOURCE_STATE_RENDER_TARGET
            } else {
                D3D12_RESOURCE_STATE_COPY_DEST
            },
            None,
            &raw mut gpu,
        )
    }
    .map_err(|error| {
        backend_error(
            "ID3D12Device::CreateCommittedResource(render target)",
            &error,
        )
        .to_string()
    })?;
    let gpu = gpu.ok_or("D3D12 returned no render target")?;
    name_object(
        &gpu,
        &format!(
            "REmu color target address={:#x} {}x{} samples={}",
            descriptor.address, descriptor.width, descriptor.height, target.sample_count
        ),
    );
    let resolve = if target.sample_count > 1 {
        let resolve_description = D3D12_RESOURCE_DESC {
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: if integer_msaa {
                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET
            } else {
                windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_FLAG_NONE
            },
            ..description
        };
        let mut resolve = None;
        // SAFETY: the resolve texture matches the MSAA attachment except for sample count/flags.
        unsafe {
            device.CreateCommittedResource(
                &raw const heap,
                D3D12_HEAP_FLAG_NONE,
                &raw const resolve_description,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &raw mut resolve,
            )
        }
        .map_err(|error| {
            backend_error(
                "ID3D12Device::CreateCommittedResource(MSAA resolve)",
                &error,
            )
            .to_string()
        })?;
        let resolve = resolve.ok_or("D3D12 returned no MSAA resolve target")?;
        name_object(
            &resolve,
            &format!("REmu color resolve address={:#x}", descriptor.address),
        );
        Some(resolve)
    } else {
        None
    };
    let linear = read_texture_linear(descriptor)?;
    let row_bytes = descriptor.width as usize * descriptor.bytes_per_element as usize;
    let row_pitch = align_up(row_bytes, 256)?;
    let slice_pitch = row_pitch
        .checked_mul(descriptor.height as usize)
        .ok_or("render-target footprint overflows")?;
    let mut packed = vec![0_u8; slice_pitch];
    for row in 0..descriptor.height as usize {
        packed[row * row_pitch..row * row_pitch + row_bytes]
            .copy_from_slice(&linear[row * row_bytes..(row + 1) * row_bytes]);
    }
    let rtv_description = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
        NumDescriptors: 1,
        Flags: windows::Win32::Graphics::Direct3D12::D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
        NodeMask: 0,
    };
    // SAFETY: one-descriptor RTV heap is valid and owns its CPU handle.
    let rtv_heap: ID3D12DescriptorHeap =
        unsafe { device.CreateDescriptorHeap(&raw const rtv_description) }.map_err(|error| {
            backend_error("ID3D12Device::CreateDescriptorHeap(RTV)", &error).to_string()
        })?;
    let rtv = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };
    // SAFETY: the resource permits render-target access and its native format supplies the view.
    unsafe { device.CreateRenderTargetView(&gpu, None, rtv) };
    let initialization_srv_heap = if let Some(resolve) = &resolve {
        let description = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: 1,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        };
        // SAFETY: one shader-visible descriptor is retained through initialization recording.
        let heap: ID3D12DescriptorHeap =
            unsafe { device.CreateDescriptorHeap(&raw const description) }.map_err(|error| {
                backend_error("ID3D12Device::CreateDescriptorHeap(MSAA SRV)", &error).to_string()
            })?;
        let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: descriptor.format,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 1,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 0.0,
                },
            },
        };
        let handle = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
        // SAFETY: the single-sample resolve texture and SRV description match exactly.
        unsafe { device.CreateShaderResourceView(resolve, Some(&raw const view), handle) };
        Some(heap)
    } else {
        None
    };
    let (integer_resolve_rtv_heap, integer_resolve_srv_heap) = if integer_msaa {
        let resolve = resolve
            .as_ref()
            .ok_or("integer MSAA target has no single-sample resolve resource")?;
        let rtv_heap: ID3D12DescriptorHeap =
            unsafe { device.CreateDescriptorHeap(&raw const rtv_description) }.map_err(
                |error| {
                    backend_error(
                        "ID3D12Device::CreateDescriptorHeap(integer MSAA RTV)",
                        &error,
                    )
                    .to_string()
                },
            )?;
        let rtv = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };
        // SAFETY: the single-sample resolve resource permits render-target access.
        unsafe { device.CreateRenderTargetView(resolve, None, rtv) };

        let srv_description = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: 1,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        };
        let srv_heap: ID3D12DescriptorHeap =
            unsafe { device.CreateDescriptorHeap(&raw const srv_description) }.map_err(
                |error| {
                    backend_error(
                        "ID3D12Device::CreateDescriptorHeap(integer MSAA SRV)",
                        &error,
                    )
                    .to_string()
                },
            )?;
        let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: descriptor.format,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DMS,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DMS: D3D12_TEX2DMS_SRV::default(),
            },
        };
        let handle = unsafe { srv_heap.GetCPUDescriptorHandleForHeapStart() };
        // SAFETY: the multisample resource and typed MSAA SRV description match.
        unsafe { device.CreateShaderResourceView(&gpu, Some(&raw const view), handle) };
        (Some(rtv_heap), Some(srv_heap))
    } else {
        (None, None)
    };
    let raw_byte_count = descriptor.raw_byte_count()?;
    // SAFETY: decoded render targets remain mapped and writable through ordered event retirement;
    // the watch is owned by the persistent target and is dropped before that cache entry retires.
    let write_watch = unsafe {
        remu_memory::CpuWriteWatch::new(
            descriptor.address,
            u64::try_from(raw_byte_count).map_err(|_| "render target size does not fit u64")?,
        )
    }
    .map_err(|error| format!("could not watch render target guest memory: {error}"))?;
    Ok(RenderTargetResource {
        target,
        gpu,
        resolve,
        upload: create_upload_buffer(device, &packed)?,
        readback: create_readback_buffer(device, slice_pitch)?,
        rtv_heap,
        initialization_srv_heap,
        integer_resolve_rtv_heap,
        integer_resolve_srv_heap,
        row_pitch: row_pitch as u32,
        slice_pitch,
        write_watch,
        presentable_linear: None,
        upload_required: true,
        pending_writeback: false,
    })
}

fn render_target_resource_compatible(
    resource: &RenderTargetResource,
    target: DecodedRenderTarget,
) -> bool {
    resource.target.sample_count == 1
        && target.sample_count == 1
        && resource.target.descriptor == target.descriptor
}

fn guest_ranges_overlap(
    left_address: u64,
    left_byte_count: u64,
    right_address: u64,
    right_byte_count: u64,
) -> bool {
    if left_byte_count == 0 || right_byte_count == 0 {
        return false;
    }
    let left_end = left_address.saturating_add(left_byte_count);
    let right_end = right_address.saturating_add(right_byte_count);
    left_address < right_end && right_address < left_end
}

fn render_target_can_alias_texture(target: &RenderTargetResource, texture: DecodedTexture) -> bool {
    let render = target.target.descriptor;
    target.pending_writeback
        && target.target.sample_count == 1
        && !texture.writable
        && texture.address == render.address
        && texture.width == render.width
        && texture.height == render.height
        && texture.pitch == render.pitch
        && texture.tile_mode == render.tile_mode
        && texture.format == render.format
        && texture.bytes_per_element == render.bytes_per_element
        && texture.block_extent == render.block_extent
        && texture.texture_type == render.texture_type
        && texture.layer_count == render.layer_count
        && texture.mip_level_count == render.mip_level_count
}

fn evaluation_requires_render_target_publication(
    evaluation: &ShaderEvaluation,
    target: &RenderTargetResource,
    allow_image_alias: bool,
) -> bool {
    let descriptor = target.target.descriptor;
    let Ok(target_byte_count) = descriptor.raw_byte_count() else {
        return true;
    };
    let target_byte_count = u64::try_from(target_byte_count).unwrap_or(u64::MAX);
    if evaluation.global_memory_bindings.iter().any(|binding| {
        guest_ranges_overlap(
            binding.base_address,
            u64::try_from(binding.byte_count).unwrap_or(u64::MAX),
            descriptor.address,
            target_byte_count,
        )
    }) {
        trace_render_target_dependency("global-buffer", descriptor, None);
        return true;
    }
    if evaluation.scalar_memory_reads.iter().any(|&(address, _)| {
        guest_ranges_overlap(address, 4, descriptor.address, target_byte_count)
    }) {
        trace_render_target_dependency("scalar-read", descriptor, None);
        return true;
    }
    if evaluation.vertex_inputs.iter().any(|binding| {
        guest_ranges_overlap(
            binding.base_address,
            u64::try_from(binding.data.len().max(1)).unwrap_or(u64::MAX),
            descriptor.address,
            target_byte_count,
        )
    }) {
        trace_render_target_dependency("vertex-input", descriptor, None);
        return true;
    }
    evaluation.image_bindings.iter().any(|binding| {
        let Ok(texture) = decode_texture(binding) else {
            // An unresolved image could name the pending target. Flushing is the fail-closed path.
            trace_render_target_dependency("undecodable-image", descriptor, None);
            return true;
        };
        let Ok(byte_count) = texture.raw_byte_count() else {
            trace_render_target_dependency("invalid-image-size", descriptor, Some(texture));
            return true;
        };
        let overlaps = guest_ranges_overlap(
            texture.address,
            u64::try_from(byte_count).unwrap_or(u64::MAX),
            descriptor.address,
            target_byte_count,
        );
        let requires_publication =
            overlaps && (!allow_image_alias || !render_target_can_alias_texture(target, texture));
        if requires_publication {
            trace_render_target_dependency(
                if allow_image_alias {
                    "incompatible-image"
                } else {
                    "read-write-feedback"
                },
                descriptor,
                Some(texture),
            );
        }
        requires_publication
    })
}

fn trace_render_target_dependency(
    kind: &str,
    target: DecodedTexture,
    texture: Option<DecodedTexture>,
) {
    if std::env::var_os("REMU_TRACE_D3D12_TIMING").is_some_and(|value| value == "1") {
        eprintln!(
            "[REMu][D3D12-TIMING] target_dependency={kind} target={target:?} texture={texture:?}"
        );
    }
}

fn draw_requires_render_target_publication(
    draw: &IndexedDraw,
    target: &RenderTargetResource,
    output_addresses: &[u64],
) -> bool {
    let descriptor = target.target.descriptor;
    let Ok(target_byte_count) = descriptor.raw_byte_count() else {
        return true;
    };
    let target_byte_count = u64::try_from(target_byte_count).unwrap_or(u64::MAX);
    if let Some(address) = draw.indirect_arguments_address
        && guest_ranges_overlap(address, 20, descriptor.address, target_byte_count)
    {
        trace_render_target_dependency("indirect-arguments", descriptor, None);
        return true;
    }
    if let Some(address) = draw.indirect_count_address
        && guest_ranges_overlap(address, 4, descriptor.address, target_byte_count)
    {
        trace_render_target_dependency("indirect-count", descriptor, None);
        return true;
    }
    if draw.index_snapshot.is_none()
        && guest_ranges_overlap(
            draw.index_buffer_address,
            u64::from(draw.index_count).saturating_mul(u64::from(draw.index_size.max(1))),
            descriptor.address,
            target_byte_count,
        )
    {
        trace_render_target_dependency("index-buffer", descriptor, None);
        return true;
    }
    let allow_image_alias = !output_addresses.contains(&descriptor.address);
    draw.export_shader_evaluation
        .as_deref()
        .is_some_and(|evaluation| {
            evaluation_requires_render_target_publication(evaluation, target, allow_image_alias)
        })
        || draw
            .pixel_shader_evaluation
            .as_deref()
            .is_some_and(|evaluation| {
                evaluation_requires_render_target_publication(evaluation, target, allow_image_alias)
            })
}

fn compute_requires_render_target_publication(
    dispatch: &ComputeDispatch,
    target: &RenderTargetResource,
) -> bool {
    let descriptor = target.target.descriptor;
    let Ok(target_byte_count) = descriptor.raw_byte_count() else {
        return true;
    };
    let target_byte_count = u64::try_from(target_byte_count).unwrap_or(u64::MAX);
    if let ComputeDispatchSource::Indirect { dimensions_address } = dispatch.source
        && guest_ranges_overlap(
            dimensions_address,
            12,
            descriptor.address,
            target_byte_count,
        )
    {
        trace_render_target_dependency("indirect-dispatch", descriptor, None);
        return true;
    }
    dispatch
        .shader_evaluation
        .as_deref()
        .is_some_and(|evaluation| {
            evaluation_requires_render_target_publication(evaluation, target, true)
        })
}

fn create_resident_render_target_texture(
    target: &RenderTargetResource,
    descriptor: DecodedTexture,
) -> TextureResource {
    TextureResource {
        descriptor,
        gpu: target.gpu.clone(),
        upload: target.upload.clone(),
        readback: None,
        row_pitch: target.row_pitch,
        slice_pitch: target.slice_pitch,
        copy_footprints: Vec::new(),
        packed_byte_count: target.slice_pitch,
        source: TextureResourceSource::ResidentRenderTarget,
        writeback: None,
    }
}

fn refresh_render_target_upload(target: &RenderTargetResource) -> Result<(), String> {
    let descriptor = target.target.descriptor;
    let linear = read_texture_linear(descriptor)?;
    let row_bytes = descriptor.width as usize * descriptor.bytes_per_element as usize;
    let mut packed = vec![0_u8; target.slice_pitch];
    for row in 0..descriptor.height as usize {
        packed[row * target.row_pitch as usize..row * target.row_pitch as usize + row_bytes]
            .copy_from_slice(&linear[row * row_bytes..(row + 1) * row_bytes]);
    }
    write_upload(&target.upload, &packed).map_err(|error| error.to_string())
}

fn record_render_target_upload(
    commands: &ID3D12GraphicsCommandList,
    target: &RenderTargetResource,
) {
    if target.pending_writeback {
        debug_assert!(!target.upload_required);
        return;
    }
    let upload_target = target.resolve.as_ref().unwrap_or(&target.gpu);
    let proxy = TextureResource {
        descriptor: target.target.descriptor,
        gpu: upload_target.clone(),
        upload: target.upload.clone(),
        readback: None,
        row_pitch: target.row_pitch,
        slice_pitch: target.slice_pitch,
        copy_footprints: Vec::new(),
        packed_byte_count: target.slice_pitch,
        source: TextureResourceSource::GuestUpload,
        writeback: None,
    };
    if target.upload_required {
        record_texture_copy(
            commands,
            &target.upload,
            upload_target,
            0,
            0,
            &proxy,
            1,
            false,
        );
    }
    record_transition(
        commands,
        upload_target,
        D3D12_RESOURCE_STATE_COPY_DEST,
        if target.resolve.is_some() {
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
        } else {
            D3D12_RESOURCE_STATE_RENDER_TARGET
        },
    );
}

fn record_render_target_readback(
    commands: &ID3D12GraphicsCommandList,
    root: &ID3D12RootSignature,
    target: &RenderTargetResource,
    integer_resolve_pipeline: Option<&ID3D12PipelineState>,
    retain_for_rendering: bool,
) -> Result<(), String> {
    let copy_source = if let Some(resolve) = &target.resolve {
        if target.target.output_kind == remu_shader::PixelOutputKind::Float {
            record_transition(
                commands,
                &target.gpu,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
            );
            // SAFETY: source/destination have matching typed format and extent, with N samples to
            // one for a floating-point or normalized format.
            unsafe {
                commands.ResolveSubresource(
                    resolve,
                    0,
                    &target.gpu,
                    0,
                    target.target.descriptor.format,
                )
            };
            record_transition(
                commands,
                resolve,
                D3D12_RESOURCE_STATE_RESOLVE_DEST,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
        } else {
            record_integer_msaa_resolve(
                commands,
                root,
                integer_resolve_pipeline.ok_or("integer MSAA resolve has no pipeline")?,
                target,
            )?;
        }
        resolve
    } else {
        record_transition(
            commands,
            &target.gpu,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        &target.gpu
    };
    let proxy = TextureResource {
        descriptor: target.target.descriptor,
        gpu: copy_source.clone(),
        upload: target.upload.clone(),
        readback: None,
        row_pitch: target.row_pitch,
        slice_pitch: target.slice_pitch,
        copy_footprints: Vec::new(),
        packed_byte_count: target.slice_pitch,
        source: TextureResourceSource::GuestUpload,
        writeback: None,
    };
    record_texture_copy(
        commands,
        &target.readback,
        copy_source,
        0,
        0,
        &proxy,
        1,
        true,
    );
    if target.resolve.is_none() {
        // A presentation-only readback preserves the native target as the authoritative copy.
        // Guest-memory publication returns it to COPY_DEST so a future CPU-dirtied version can
        // be uploaded before the next draw.
        record_transition(
            commands,
            &target.gpu,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            if retain_for_rendering {
                D3D12_RESOURCE_STATE_RENDER_TARGET
            } else {
                D3D12_RESOURCE_STATE_COPY_DEST
            },
        );
    }
    Ok(())
}

fn read_render_target_linear(target: &RenderTargetResource) -> Result<Vec<u8>, String> {
    let packed =
        read_readback(&target.readback, target.slice_pitch).map_err(|error| error.to_string())?;
    let descriptor = target.target.descriptor;
    let row_bytes = descriptor.width as usize * descriptor.bytes_per_element as usize;
    let mut linear = vec![0_u8; row_bytes * descriptor.height as usize];
    for row in 0..descriptor.height as usize {
        linear[row * row_bytes..(row + 1) * row_bytes].copy_from_slice(
            &packed[row * target.row_pitch as usize..row * target.row_pitch as usize + row_bytes],
        );
    }
    Ok(linear)
}

fn presentable_bgra(
    format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    mut linear: Vec<u8>,
) -> Option<Vec<u8>> {
    match format {
        DXGI_FORMAT_B8G8R8A8_UNORM => Some(linear),
        DXGI_FORMAT_R8G8B8A8_UNORM => {
            for pixel in linear.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
            Some(linear)
        }
        _ => None,
    }
}

fn write_back_render_target(target: &mut RenderTargetResource) -> Result<(), String> {
    let linear = read_render_target_linear(target)?;
    let descriptor = target.target.descriptor;
    let write_result = target
        .write_watch
        .with_live_mapping(|| write_texture_linear(descriptor, &linear))
        .map_err(|error| format!("could not open render-target watch for writeback: {error}"))?;
    let rearm_result = target
        .write_watch
        .rearm()
        .map_err(|error| format!("could not rearm render-target watch: {error}"));
    write_result?;
    rearm_result?;
    target.presentable_linear = Some(linear);
    target.upload_required = false;
    target.pending_writeback = false;
    Ok(())
}

const MSAA_INITIALIZATION_VERTEX_HLSL: &str = r#"
struct RemuMsaaVertex {
    float4 position : SV_Position;
};

RemuMsaaVertex remu_msaa_init_vs(uint vertex_id : SV_VertexID) {
    const float2 positions[3] = {
        float2(-1.0,  1.0),
        float2( 3.0,  1.0),
        float2(-1.0, -3.0)
    };
    RemuMsaaVertex output;
    output.position = float4(positions[vertex_id], 0.0, 1.0);
    return output;
}
"#;

fn msaa_initialization_pixel_hlsl(kind: remu_shader::PixelOutputKind) -> String {
    let scalar = match kind {
        remu_shader::PixelOutputKind::Float => "float4",
        remu_shader::PixelOutputKind::Uint => "uint4",
        remu_shader::PixelOutputKind::Sint => "int4",
    };
    format!(
        r#"
struct RemuMsaaVertex {{
    float4 position : SV_Position;
}};

{scalar} remu_msaa_init_ps(RemuMsaaVertex input) : SV_Target0 {{
    Texture2D<{scalar}> source = ResourceDescriptorHeap[0];
    return source.Load(int3(int2(input.position.xy), 0));
}}
"#
    )
}

fn create_msaa_initialization_pipeline(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    target: DecodedRenderTarget,
) -> Result<ID3D12PipelineState, String> {
    let vertex = crate::dxc::compile_hlsl(
        "remu-msaa-init-vs.hlsl",
        MSAA_INITIALIZATION_VERTEX_HLSL,
        "remu_msaa_init_vs",
        "vs_6_7",
    )?;
    let pixel_source = msaa_initialization_pixel_hlsl(target.output_kind);
    let pixel = crate::dxc::compile_hlsl(
        "remu-msaa-init-ps.hlsl",
        &pixel_source,
        "remu_msaa_init_ps",
        "ps_6_7",
    )?;
    let mut blend = D3D12_BLEND_DESC::default();
    blend.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
        RenderTargetWriteMask: 0x0f,
        ..D3D12_RENDER_TARGET_BLEND_DESC::default()
    };
    let rasterizer = D3D12_RASTERIZER_DESC {
        FillMode: D3D12_FILL_MODE_SOLID,
        CullMode: D3D12_CULL_MODE_NONE,
        DepthClipEnable: BOOL(1),
        ..D3D12_RASTERIZER_DESC::default()
    };
    let mut description = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root.clone())),
        VS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: vertex.as_ptr().cast(),
            BytecodeLength: vertex.len(),
        },
        PS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: pixel.as_ptr().cast(),
            BytecodeLength: pixel.len(),
        },
        BlendState: blend,
        SampleMask: u32::MAX,
        RasterizerState: rasterizer,
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: target.sample_count,
            Quality: 0,
        },
        ..D3D12_GRAPHICS_PIPELINE_STATE_DESC::default()
    };
    description.RTVFormats[0] = target.descriptor.format;
    // SAFETY: root signature and compiled bytecode remain live for synchronous PSO creation.
    let result =
        unsafe { device.CreateGraphicsPipelineState(&raw const description) }.map_err(|error| {
            backend_error(
                "ID3D12Device::CreateGraphicsPipelineState(MSAA initialization)",
                &error,
            )
            .to_string()
        });
    // SAFETY: release the explicit descriptor clone after PSO creation returns.
    unsafe { ManuallyDrop::drop(&mut description.pRootSignature) };
    result
}

fn record_msaa_render_target_initialization(
    commands: &ID3D12GraphicsCommandList,
    root: &ID3D12RootSignature,
    pipeline: &ID3D12PipelineState,
    target: &RenderTargetResource,
) -> Result<(), String> {
    let resolve = target
        .resolve
        .as_ref()
        .ok_or("MSAA initialization has no resolve texture")?;
    let descriptors = target
        .initialization_srv_heap
        .as_ref()
        .ok_or("MSAA initialization has no source descriptor")?;
    let rtv = unsafe { target.rtv_heap.GetCPUDescriptorHandleForHeapStart() };
    let viewport = D3D12_VIEWPORT {
        Width: target.target.descriptor.width as f32,
        Height: target.target.descriptor.height as f32,
        MinDepth: 0.0,
        MaxDepth: 1.0,
        ..D3D12_VIEWPORT::default()
    };
    let scissor = RECT {
        left: 0,
        top: 0,
        right: i32::try_from(target.target.descriptor.width)
            .map_err(|_| "MSAA target width exceeds i32")?,
        bottom: i32::try_from(target.target.descriptor.height)
            .map_err(|_| "MSAA target height exceeds i32")?,
    };
    // SAFETY: the source SRV, MSAA RTV, root signature, and pipeline remain live through submit.
    unsafe {
        commands.SetDescriptorHeaps(&[Some(descriptors.clone())]);
        commands.SetGraphicsRootSignature(root);
        commands.SetPipelineState(pipeline);
        commands.RSSetViewports(std::slice::from_ref(&viewport));
        commands.RSSetScissorRects(std::slice::from_ref(&scissor));
        commands.OMSetRenderTargets(1, Some(&raw const rtv), false, None);
        commands.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        commands.DrawInstanced(3, 1, 0, 0);
    }
    record_transition(
        commands,
        resolve,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_RESOLVE_DEST,
    );
    Ok(())
}

fn integer_msaa_resolve_pixel_hlsl(kind: remu_shader::PixelOutputKind) -> Result<String, String> {
    let scalar = match kind {
        remu_shader::PixelOutputKind::Uint => "uint4",
        remu_shader::PixelOutputKind::Sint => "int4",
        remu_shader::PixelOutputKind::Float => {
            return Err("floating-point MSAA targets use native D3D12 resolve".to_owned());
        }
    };
    Ok(format!(
        r#"
struct RemuMsaaVertex {{
    float4 position : SV_Position;
}};

{scalar} remu_integer_msaa_resolve_ps(RemuMsaaVertex input) : SV_Target0 {{
    Texture2DMS<{scalar}> source = ResourceDescriptorHeap[0];
    return source.Load(int2(input.position.xy), 0);
}}
"#
    ))
}

fn create_integer_msaa_resolve_pipeline(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    target: DecodedRenderTarget,
) -> Result<ID3D12PipelineState, String> {
    let vertex = crate::dxc::compile_hlsl(
        "remu-integer-msaa-resolve-vs.hlsl",
        MSAA_INITIALIZATION_VERTEX_HLSL,
        "remu_msaa_init_vs",
        "vs_6_7",
    )?;
    let pixel_source = integer_msaa_resolve_pixel_hlsl(target.output_kind)?;
    let pixel = crate::dxc::compile_hlsl(
        "remu-integer-msaa-resolve-ps.hlsl",
        &pixel_source,
        "remu_integer_msaa_resolve_ps",
        "ps_6_7",
    )?;
    let mut blend = D3D12_BLEND_DESC::default();
    blend.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
        RenderTargetWriteMask: 0x0f,
        ..D3D12_RENDER_TARGET_BLEND_DESC::default()
    };
    let rasterizer = D3D12_RASTERIZER_DESC {
        FillMode: D3D12_FILL_MODE_SOLID,
        CullMode: D3D12_CULL_MODE_NONE,
        DepthClipEnable: BOOL(1),
        ..D3D12_RASTERIZER_DESC::default()
    };
    let mut description = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root.clone())),
        VS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: vertex.as_ptr().cast(),
            BytecodeLength: vertex.len(),
        },
        PS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: pixel.as_ptr().cast(),
            BytecodeLength: pixel.len(),
        },
        BlendState: blend,
        SampleMask: u32::MAX,
        RasterizerState: rasterizer,
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..D3D12_GRAPHICS_PIPELINE_STATE_DESC::default()
    };
    description.RTVFormats[0] = target.descriptor.format;
    // SAFETY: root signature and compiled bytecode remain live for synchronous PSO creation.
    let result =
        unsafe { device.CreateGraphicsPipelineState(&raw const description) }.map_err(|error| {
            backend_error(
                "ID3D12Device::CreateGraphicsPipelineState(integer MSAA resolve)",
                &error,
            )
            .to_string()
        });
    // SAFETY: release the explicit descriptor clone after PSO creation returns.
    unsafe { ManuallyDrop::drop(&mut description.pRootSignature) };
    result
}

fn record_integer_msaa_resolve(
    commands: &ID3D12GraphicsCommandList,
    root: &ID3D12RootSignature,
    pipeline: &ID3D12PipelineState,
    target: &RenderTargetResource,
) -> Result<(), String> {
    let resolve = target
        .resolve
        .as_ref()
        .ok_or("integer MSAA resolve has no destination")?;
    let descriptors = target
        .integer_resolve_srv_heap
        .as_ref()
        .ok_or("integer MSAA resolve has no source descriptor")?;
    let rtv_heap = target
        .integer_resolve_rtv_heap
        .as_ref()
        .ok_or("integer MSAA resolve has no destination RTV")?;
    let rtv = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };
    let viewport = D3D12_VIEWPORT {
        Width: target.target.descriptor.width as f32,
        Height: target.target.descriptor.height as f32,
        MinDepth: 0.0,
        MaxDepth: 1.0,
        ..D3D12_VIEWPORT::default()
    };
    let scissor = RECT {
        left: 0,
        top: 0,
        right: i32::try_from(target.target.descriptor.width)
            .map_err(|_| "integer MSAA target width exceeds i32")?,
        bottom: i32::try_from(target.target.descriptor.height)
            .map_err(|_| "integer MSAA target height exceeds i32")?,
    };
    record_transition(
        commands,
        &target.gpu,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
    );
    record_transition(
        commands,
        resolve,
        D3D12_RESOURCE_STATE_RESOLVE_DEST,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
    );
    // SAFETY: source descriptor, destination RTV, root, and pipeline remain live through submit.
    unsafe {
        commands.SetDescriptorHeaps(&[Some(descriptors.clone())]);
        commands.SetGraphicsRootSignature(root);
        commands.SetPipelineState(pipeline);
        commands.RSSetViewports(std::slice::from_ref(&viewport));
        commands.RSSetScissorRects(std::slice::from_ref(&scissor));
        commands.OMSetRenderTargets(1, Some(&raw const rtv), false, None);
        commands.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        commands.DrawInstanced(3, 1, 0, 0);
    }
    record_transition(
        commands,
        resolve,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );
    Ok(())
}

const DEPTH_MSAA_INITIALIZATION_PIXEL_HLSL: &str = r#"
struct RemuMsaaVertex {
    float4 position : SV_Position;
};

float remu_depth_msaa_init_ps(RemuMsaaVertex input) : SV_Depth {
    Texture2D<float> source = ResourceDescriptorHeap[0];
    return source.Load(int3(int2(input.position.xy), 0));
}
"#;

const DEPTH_STENCIL_MSAA_INITIALIZATION_PIXEL_HLSL: &str = r#"
struct RemuMsaaVertex {
    float4 position : SV_Position;
};

cbuffer RemuDepthStencilInitialization : register(b0, space1) {
    uint remu_stencil_value;
};

float remu_depth_msaa_init_ps(RemuMsaaVertex input) : SV_Depth {
    Texture2D<float> depth_source = ResourceDescriptorHeap[0];
    Texture2D<uint> stencil_source = ResourceDescriptorHeap[1];
    int3 coordinate = int3(int2(input.position.xy), 0);
    if (stencil_source.Load(coordinate) != remu_stencil_value) {
        discard;
    }
    return depth_source.Load(coordinate);
}
"#;

fn create_depth_msaa_initialization_pipeline(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    target: DecodedDepthTarget,
) -> Result<ID3D12PipelineState, String> {
    let vertex = crate::dxc::compile_hlsl(
        "remu-depth-msaa-init-vs.hlsl",
        MSAA_INITIALIZATION_VERTEX_HLSL,
        "remu_msaa_init_vs",
        "vs_6_7",
    )?;
    let pixel = crate::dxc::compile_hlsl(
        "remu-depth-msaa-init-ps.hlsl",
        if target.stencil.is_some() {
            DEPTH_STENCIL_MSAA_INITIALIZATION_PIXEL_HLSL
        } else {
            DEPTH_MSAA_INITIALIZATION_PIXEL_HLSL
        },
        "remu_depth_msaa_init_ps",
        "ps_6_7",
    )?;
    let replace_stencil = D3D12_DEPTH_STENCILOP_DESC {
        StencilFailOp: D3D12_STENCIL_OP_KEEP,
        StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
        StencilPassOp: D3D12_STENCIL_OP_REPLACE,
        StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
    };
    let depth_stencil = D3D12_DEPTH_STENCIL_DESC {
        DepthEnable: BOOL(1),
        DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
        DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        StencilEnable: BOOL(i32::from(target.stencil.is_some())),
        StencilReadMask: u8::MAX,
        StencilWriteMask: u8::MAX,
        FrontFace: replace_stencil,
        BackFace: replace_stencil,
        ..D3D12_DEPTH_STENCIL_DESC::default()
    };
    let rasterizer = D3D12_RASTERIZER_DESC {
        FillMode: D3D12_FILL_MODE_SOLID,
        CullMode: D3D12_CULL_MODE_NONE,
        DepthClipEnable: BOOL(1),
        ..D3D12_RASTERIZER_DESC::default()
    };
    let mut description = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root.clone())),
        VS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: vertex.as_ptr().cast(),
            BytecodeLength: vertex.len(),
        },
        PS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: pixel.as_ptr().cast(),
            BytecodeLength: pixel.len(),
        },
        SampleMask: u32::MAX,
        RasterizerState: rasterizer,
        DepthStencilState: depth_stencil,
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        DSVFormat: depth_host_format(target),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: target.sample_count,
            Quality: 0,
        },
        ..D3D12_GRAPHICS_PIPELINE_STATE_DESC::default()
    };
    // SAFETY: root signature and compiled bytecode remain live for synchronous PSO creation.
    let result =
        unsafe { device.CreateGraphicsPipelineState(&raw const description) }.map_err(|error| {
            backend_error(
                "ID3D12Device::CreateGraphicsPipelineState(depth MSAA initialization)",
                &error,
            )
            .to_string()
        });
    // SAFETY: release the explicit descriptor clone after PSO creation returns.
    unsafe { ManuallyDrop::drop(&mut description.pRootSignature) };
    result
}

fn record_depth_msaa_initialization(
    commands: &ID3D12GraphicsCommandList,
    root: &ID3D12RootSignature,
    pipeline: &ID3D12PipelineState,
    target: &DepthTargetResource,
) -> Result<(), String> {
    let resolve = target
        .resolve
        .as_ref()
        .ok_or("depth MSAA initialization has no resolve texture")?;
    let descriptors = target
        .initialization_srv_heap
        .as_ref()
        .ok_or("depth MSAA initialization has no source descriptor")?;
    let dsv = unsafe { target.dsv_heap.GetCPUDescriptorHandleForHeapStart() };
    let viewport = D3D12_VIEWPORT {
        Width: target.target.descriptor.width as f32,
        Height: target.target.descriptor.height as f32,
        MinDepth: 0.0,
        MaxDepth: 1.0,
        ..D3D12_VIEWPORT::default()
    };
    let scissor = RECT {
        left: 0,
        top: 0,
        right: i32::try_from(target.target.descriptor.width)
            .map_err(|_| "depth MSAA target width exceeds i32")?,
        bottom: i32::try_from(target.target.descriptor.height)
            .map_err(|_| "depth MSAA target height exceeds i32")?,
    };
    // SAFETY: the source SRV, MSAA DSV, root signature, and pipeline live through submission.
    unsafe {
        commands.SetDescriptorHeaps(&[Some(descriptors.clone())]);
        commands.SetGraphicsRootSignature(root);
        commands.SetPipelineState(pipeline);
        commands.RSSetViewports(std::slice::from_ref(&viewport));
        commands.RSSetScissorRects(std::slice::from_ref(&scissor));
        commands.OMSetRenderTargets(0, None, false, Some(&raw const dsv));
        commands.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        if target.target.stencil.is_some() {
            for stencil in 0..=u8::MAX {
                commands.SetGraphicsRoot32BitConstant(1, u32::from(stencil), 0);
                commands.OMSetStencilRef(u32::from(stencil));
                commands.DrawInstanced(3, 1, 0, 0);
            }
        } else {
            commands.DrawInstanced(3, 1, 0, 0);
        }
    }
    record_transition(
        commands,
        resolve,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_RESOLVE_DEST,
    );
    if !depth_target_writable(target.target) {
        record_transition(
            commands,
            &target.gpu,
            windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_DEPTH_WRITE,
            windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_DEPTH_READ,
        );
    }
    Ok(())
}

fn create_depth_target_resource(
    device: &ID3D12Device,
    target: DecodedDepthTarget,
) -> Result<DepthTargetResource, String> {
    let descriptor = target.descriptor;
    let host_format = depth_host_format(target);
    if target.sample_count > 1 {
        let mut quality = D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS {
            Format: host_format,
            SampleCount: target.sample_count,
            Flags: D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE,
            NumQualityLevels: 0,
        };
        // SAFETY: `quality` is the exact structure for this feature query.
        unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
                (&raw mut quality).cast(),
                u32::try_from(std::mem::size_of_val(&quality))
                    .expect("multisample query size fits u32"),
            )
        }
        .map_err(|error| {
            backend_error("ID3D12Device::CheckFeatureSupport(depth MSAA)", &error).to_string()
        })?;
        if quality.NumQualityLevels == 0 {
            return Err(format!(
                "D3D12 adapter does not support {}x MSAA for depth format {:?}",
                target.sample_count, host_format
            ));
        }
    }
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..D3D12_HEAP_PROPERTIES::default()
    };
    let description = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: u64::from(descriptor.width),
        Height: descriptor.height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: if target.sample_count > 1 {
            depth_typeless_format(target)?
        } else {
            host_format
        },
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: target.sample_count,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        ..D3D12_RESOURCE_DESC::default()
    };
    let mut gpu = None;
    // SAFETY: the validated depth descriptor supplies a native depth format and owned output.
    unsafe {
        device.CreateCommittedResource(
            &raw const heap,
            D3D12_HEAP_FLAG_NONE,
            &raw const description,
            if target.sample_count > 1 {
                windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_DEPTH_WRITE
            } else {
                D3D12_RESOURCE_STATE_COPY_DEST
            },
            None,
            &raw mut gpu,
        )
    }
    .map_err(|error| {
        backend_error(
            "ID3D12Device::CreateCommittedResource(depth target)",
            &error,
        )
        .to_string()
    })?;
    let gpu = gpu.ok_or("D3D12 returned no depth target")?;
    name_object(
        &gpu,
        &format!(
            "REmu depth target address={:#x} {}x{} samples={}",
            descriptor.address, descriptor.width, descriptor.height, target.sample_count
        ),
    );

    let resolve_description = D3D12_RESOURCE_DESC {
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..description
    };
    let resolve = if target.sample_count > 1 {
        let mut resolve = None;
        // SAFETY: the single-sample resource matches the typeless depth attachment and carries
        // both depth-stencil and resolve-destination capabilities.
        unsafe {
            device.CreateCommittedResource(
                &raw const heap,
                D3D12_HEAP_FLAG_NONE,
                &raw const resolve_description,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &raw mut resolve,
            )
        }
        .map_err(|error| {
            backend_error(
                "ID3D12Device::CreateCommittedResource(depth MSAA resolve)",
                &error,
            )
            .to_string()
        })?;
        let resolve = resolve.ok_or("D3D12 returned no depth MSAA resolve target")?;
        name_object(
            &resolve,
            &format!("REmu depth resolve address={:#x}", descriptor.address),
        );
        Some(resolve)
    } else {
        None
    };

    let plane_count = if target.stencil.is_some() { 2 } else { 1 };
    let mut layouts = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); plane_count];
    let mut row_counts = vec![0_u32; plane_count];
    let mut row_sizes = vec![0_u64; plane_count];
    let mut total_bytes = 0_u64;
    // SAFETY: all output arrays contain one entry per requested plane and the resource
    // description is identical to the copyable single-sample texture.
    unsafe {
        device.GetCopyableFootprints(
            &raw const resolve_description,
            0,
            plane_count as u32,
            0,
            Some(layouts.as_mut_ptr()),
            Some(row_counts.as_mut_ptr()),
            Some(row_sizes.as_mut_ptr()),
            Some(&raw mut total_bytes),
        )
    };
    if total_bytes == u64::MAX {
        return Err("D3D12 rejected the depth/stencil copy footprint".to_owned());
    }
    let packed_byte_count = usize::try_from(total_bytes)
        .map_err(|_| "depth/stencil copy allocation exceeds host address space")?;
    let copy_planes = layouts
        .into_iter()
        .zip(row_counts)
        .zip(row_sizes)
        .enumerate()
        .map(|(subresource, ((layout, row_count), row_size))| {
            let row_size = usize::try_from(row_size)
                .map_err(|_| "depth/stencil copy row exceeds host address space")?;
            let footprint = TextureCopyFootprint {
                offset: layout.Offset,
                subresource: subresource as u32,
                format: layout.Footprint.Format,
                width: layout.Footprint.Width,
                height: layout.Footprint.Height,
                depth: layout.Footprint.Depth,
                row_pitch: layout.Footprint.RowPitch,
            };
            let final_row = usize::try_from(row_count.saturating_sub(1))
                .ok()
                .and_then(|row| row.checked_mul(footprint.row_pitch as usize))
                .and_then(|offset| offset.checked_add(row_size))
                .and_then(|bytes| bytes.checked_add(footprint.offset as usize))
                .ok_or("depth/stencil copy footprint overflows")?;
            if final_row > packed_byte_count {
                return Err("D3D12 depth/stencil footprint exceeds its allocation".to_owned());
            }
            Ok(DepthCopyPlane {
                footprint,
                row_count,
                row_size,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    let depth_linear = read_texture_linear(descriptor)?;
    let stencil_linear = target
        .stencil
        .map(|stencil| read_texture_linear(stencil.descriptor))
        .transpose()?;
    let mut packed = vec![0_u8; packed_byte_count];
    let depth_plane = copy_planes
        .first()
        .ok_or("D3D12 returned no depth copy plane")?;
    let guest_depth_row_bytes = descriptor.width as usize * descriptor.bytes_per_element as usize;
    let host_depth_row_bytes = if target.stencil.is_some() {
        descriptor.width as usize * 4
    } else {
        guest_depth_row_bytes
    };
    validate_depth_copy_plane(
        depth_plane,
        descriptor.height,
        host_depth_row_bytes,
        "depth",
    )?;
    for row in 0..descriptor.height as usize {
        let destination =
            depth_plane.footprint.offset as usize + row * depth_plane.footprint.row_pitch as usize;
        let source = row * guest_depth_row_bytes;
        if target.stencil.is_some() {
            for column in 0..descriptor.width as usize {
                let guest = source + column * descriptor.bytes_per_element as usize;
                let depth = if descriptor.address == 0 {
                    1.0
                } else if descriptor.bytes_per_element == 2 {
                    f32::from(u16::from_le_bytes(
                        depth_linear[guest..guest + 2]
                            .try_into()
                            .expect("D16 sample"),
                    )) / f32::from(u16::MAX)
                } else {
                    f32::from_le_bytes(
                        depth_linear[guest..guest + 4]
                            .try_into()
                            .expect("D32 sample"),
                    )
                };
                let host = destination + column * 4;
                packed[host..host + 4].copy_from_slice(&depth.to_le_bytes());
            }
        } else {
            packed[destination..destination + host_depth_row_bytes]
                .copy_from_slice(&depth_linear[source..source + guest_depth_row_bytes]);
        }
    }
    if let (Some(stencil), Some(stencil_plane)) = (stencil_linear.as_ref(), copy_planes.get(1)) {
        let row_bytes = descriptor.width as usize;
        validate_depth_copy_plane(stencil_plane, descriptor.height, row_bytes, "stencil")?;
        for row in 0..descriptor.height as usize {
            let source = row * row_bytes;
            let destination = stencil_plane.footprint.offset as usize
                + row * stencil_plane.footprint.row_pitch as usize;
            packed[destination..destination + row_bytes]
                .copy_from_slice(&stencil[source..source + row_bytes]);
        }
    }
    let dsv_description = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: windows::Win32::Graphics::Direct3D12::D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
        NumDescriptors: 1,
        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
        NodeMask: 0,
    };
    // SAFETY: a one-descriptor CPU DSV heap is valid.
    let dsv_heap: ID3D12DescriptorHeap =
        unsafe { device.CreateDescriptorHeap(&raw const dsv_description) }.map_err(|error| {
            backend_error("ID3D12Device::CreateDescriptorHeap(DSV)", &error).to_string()
        })?;
    let dsv = unsafe { dsv_heap.GetCPUDescriptorHandleForHeapStart() };
    let dsv_view = D3D12_DEPTH_STENCIL_VIEW_DESC {
        Format: host_format,
        ViewDimension: if target.sample_count > 1 {
            D3D12_DSV_DIMENSION_TEXTURE2DMS
        } else {
            D3D12_DSV_DIMENSION_TEXTURE2D
        },
        Flags: D3D12_DSV_FLAG_NONE,
        ..D3D12_DEPTH_STENCIL_VIEW_DESC::default()
    };
    // SAFETY: the explicit view gives the typed interpretation of the matching depth resource.
    unsafe { device.CreateDepthStencilView(&gpu, Some(&raw const dsv_view), dsv) };
    let initialization_srv_heap = if let Some(resolve) = &resolve {
        let heap_description = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: if target.stencil.is_some() { 2 } else { 1 },
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        };
        // SAFETY: the single shader-visible descriptor remains owned through command execution.
        let descriptors: ID3D12DescriptorHeap =
            unsafe { device.CreateDescriptorHeap(&raw const heap_description) }.map_err(
                |error| {
                    backend_error("ID3D12Device::CreateDescriptorHeap(depth MSAA SRV)", &error)
                        .to_string()
                },
            )?;
        let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: depth_srv_format(target)?,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 1,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 0.0,
                },
            },
        };
        let mut handle = unsafe { descriptors.GetCPUDescriptorHandleForHeapStart() };
        // SAFETY: the typeless resolve resource admits this depth-plane SRV interpretation.
        unsafe { device.CreateShaderResourceView(resolve, Some(&raw const view), handle) };
        if target.stencil.is_some() {
            handle.ptr += usize::try_from(unsafe {
                device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
            })
            .expect("descriptor increment fits usize");
            let stencil_view = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: stencil_srv_format(target)?,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV {
                        MostDetailedMip: 0,
                        MipLevels: 1,
                        PlaneSlice: 1,
                        ResourceMinLODClamp: 0.0,
                    },
                },
            };
            // SAFETY: descriptor slot one exposes the stencil plane of the combined resource.
            unsafe {
                device.CreateShaderResourceView(resolve, Some(&raw const stencil_view), handle)
            };
        }
        Some(descriptors)
    } else {
        None
    };
    Ok(DepthTargetResource {
        target,
        gpu,
        resolve,
        upload: create_upload_buffer(device, &packed)?,
        readback: create_readback_buffer(device, packed_byte_count)?,
        dsv_heap,
        initialization_srv_heap,
        copy_planes,
        packed_byte_count,
    })
}

fn validate_depth_copy_plane(
    plane: &DepthCopyPlane,
    height: u32,
    required_row_bytes: usize,
    name: &str,
) -> Result<(), String> {
    if plane.row_count < height || plane.row_size < required_row_bytes {
        return Err(format!(
            "D3D12 {name} plane is too small: {} rows of {} bytes for {height}x{required_row_bytes}",
            plane.row_count, plane.row_size
        ));
    }
    Ok(())
}

fn record_depth_target_upload(commands: &ID3D12GraphicsCommandList, target: &DepthTargetResource) {
    let upload_target = target.resolve.as_ref().unwrap_or(&target.gpu);
    for plane in &target.copy_planes {
        record_texture_copy_footprint(
            commands,
            &target.upload,
            upload_target,
            plane.footprint,
            false,
        );
    }
    let state = if target.resolve.is_some() {
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
    } else if depth_target_writable(target.target) {
        windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_DEPTH_WRITE
    } else {
        windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_DEPTH_READ
    };
    record_transition(
        commands,
        upload_target,
        D3D12_RESOURCE_STATE_COPY_DEST,
        state,
    );
}

fn depth_target_writable(target: DecodedDepthTarget) -> bool {
    target.descriptor.writable
        || target
            .stencil
            .is_some_and(|stencil| stencil.descriptor.writable)
}

fn record_depth_target_readback(
    commands: &ID3D12GraphicsCommandList,
    target: &DepthTargetResource,
) -> Result<(), String> {
    if !target.target.descriptor.writable
        && !target
            .target
            .stencil
            .is_some_and(|stencil| stencil.descriptor.writable)
    {
        return Ok(());
    }
    let source_state = if depth_target_writable(target.target) {
        windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_DEPTH_WRITE
    } else {
        windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_DEPTH_READ
    };
    record_transition(
        commands,
        &target.gpu,
        source_state,
        if target.resolve.is_some() {
            D3D12_RESOURCE_STATE_RESOLVE_SOURCE
        } else {
            D3D12_RESOURCE_STATE_COPY_SOURCE
        },
    );
    let copy_source = if let Some(resolve) = &target.resolve {
        let commands1: ID3D12GraphicsCommandList1 = commands.cast().map_err(|error| {
            backend_error("ID3D12GraphicsCommandList::cast(depth resolve)", &error).to_string()
        })?;
        let mode = if matches!(
            target.target.compare,
            windows::Win32::Graphics::Direct3D12::D3D12_COMPARISON_FUNC_GREATER
                | windows::Win32::Graphics::Direct3D12::D3D12_COMPARISON_FUNC_GREATER_EQUAL
        ) {
            D3D12_RESOLVE_MODE_MAX
        } else {
            D3D12_RESOLVE_MODE_MIN
        };
        // SAFETY: the typeless resources have matching extents; the typed format selects their
        // depth plane, and the destination is single-sampled in RESOLVE_DEST state.
        unsafe {
            commands1.ResolveSubresourceRegion(
                resolve,
                0,
                0,
                0,
                &target.gpu,
                0,
                None,
                depth_srv_format(target.target)?,
                mode,
            )
        };
        if target.target.stencil.is_some() {
            // SAFETY: subresource one selects the stencil plane on both combined resources. MIN
            // is a defined integer stencil resolve and preserves identical per-sample values.
            unsafe {
                commands1.ResolveSubresourceRegion(
                    resolve,
                    1,
                    0,
                    0,
                    &target.gpu,
                    1,
                    None,
                    stencil_srv_format(target.target)?,
                    D3D12_RESOLVE_MODE_MIN,
                )
            };
        }
        record_transition(
            commands,
            resolve,
            D3D12_RESOURCE_STATE_RESOLVE_DEST,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        resolve
    } else {
        &target.gpu
    };
    for plane in &target.copy_planes {
        record_texture_copy_footprint(
            commands,
            &target.readback,
            copy_source,
            plane.footprint,
            true,
        );
    }
    Ok(())
}

fn write_back_depth_target(target: &DepthTargetResource) -> Result<(), String> {
    let depth_writable = target.target.descriptor.writable;
    let stencil_writable = target
        .target
        .stencil
        .is_some_and(|stencil| stencil.descriptor.writable);
    if !depth_writable && !stencil_writable {
        return Ok(());
    }
    let packed = read_readback(&target.readback, target.packed_byte_count)
        .map_err(|error| error.to_string())?;
    let descriptor = target.target.descriptor;
    let depth_plane = target
        .copy_planes
        .first()
        .ok_or("D3D12 returned no depth copy plane")?;
    if target.target.stencil.is_none() {
        let row_bytes = descriptor.width as usize * descriptor.bytes_per_element as usize;
        let mut linear = vec![0_u8; row_bytes * descriptor.height as usize];
        for row in 0..descriptor.height as usize {
            let source = depth_plane.footprint.offset as usize
                + row * depth_plane.footprint.row_pitch as usize;
            linear[row * row_bytes..(row + 1) * row_bytes]
                .copy_from_slice(&packed[source..source + row_bytes]);
        }
        return write_texture_linear(descriptor, &linear);
    }
    let pixel_count = descriptor.width as usize * descriptor.height as usize;
    if depth_writable && descriptor.address != 0 {
        let mut depth_linear = vec![0_u8; pixel_count * descriptor.bytes_per_element as usize];
        for row in 0..descriptor.height as usize {
            let source_row = depth_plane.footprint.offset as usize
                + row * depth_plane.footprint.row_pitch as usize;
            for column in 0..descriptor.width as usize {
                let pixel = row * descriptor.width as usize + column;
                let source = source_row + column * 4;
                let depth = f32::from_le_bytes(
                    packed[source..source + 4]
                        .try_into()
                        .expect("planar depth sample"),
                );
                if descriptor.bytes_per_element == 2 {
                    let value = (depth.clamp(0.0, 1.0) * f32::from(u16::MAX)).round() as u16;
                    depth_linear[pixel * 2..pixel * 2 + 2].copy_from_slice(&value.to_le_bytes());
                } else {
                    depth_linear[pixel * 4..pixel * 4 + 4].copy_from_slice(&depth.to_le_bytes());
                }
            }
        }
        write_texture_linear(descriptor, &depth_linear)?;
    }
    if stencil_writable {
        let stencil = target
            .target
            .stencil
            .expect("stencil writeback has descriptor");
        let stencil_plane = target
            .copy_planes
            .get(1)
            .ok_or("D3D12 returned no stencil copy plane")?;
        let row_bytes = descriptor.width as usize;
        let mut linear = vec![0_u8; pixel_count];
        for row in 0..descriptor.height as usize {
            let source = stencil_plane.footprint.offset as usize
                + row * stencil_plane.footprint.row_pitch as usize;
            linear[row * row_bytes..(row + 1) * row_bytes]
                .copy_from_slice(&packed[source..source + row_bytes]);
        }
        write_texture_linear(stencil.descriptor, &linear)?;
    }
    Ok(())
}

fn create_texture_resource(
    device: &ID3D12Device,
    descriptor: DecodedTexture,
) -> Result<TextureResource, String> {
    if descriptor.writable && descriptor.block_extent != [1, 1] {
        return Err("compressed images cannot be bound for D3D12 UAV writes".to_owned());
    }
    if descriptor.writable && descriptor.mip_level_count != 1 {
        return Err("mipmapped D3D12 UAV writeback is not implemented yet".to_owned());
    }
    if descriptor.mip_level_count > 1
        && (descriptor.texture_type != 9 || descriptor.layer_count != 1)
    {
        return Err(format!(
            "mipmapped D3D12 resources require one 2D layer, got type={} layers={}",
            descriptor.texture_type, descriptor.layer_count
        ));
    }
    let dimension = if descriptor.texture_type == 10 {
        D3D12_RESOURCE_DIMENSION_TEXTURE3D
    } else {
        D3D12_RESOURCE_DIMENSION_TEXTURE2D
    };
    validate_texture_format_support(device, descriptor, dimension)?;
    let depth_or_array =
        u16::try_from(descriptor.layer_count).map_err(|_| "texture layer count exceeds D3D12")?;
    let flags = if descriptor.writable {
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
    } else {
        windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_FLAG_NONE
    };
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..D3D12_HEAP_PROPERTIES::default()
    };
    let resource_description = D3D12_RESOURCE_DESC {
        Dimension: dimension,
        Width: u64::from(descriptor.width),
        Height: descriptor.height,
        DepthOrArraySize: depth_or_array,
        MipLevels: u16::try_from(descriptor.mip_level_count)
            .map_err(|_| "texture mip count exceeds D3D12")?,
        Format: descriptor.format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: flags,
        ..D3D12_RESOURCE_DESC::default()
    };
    let mut gpu = None;
    // SAFETY: the texture descriptor is validated and the output slot receives an owned handle.
    unsafe {
        device.CreateCommittedResource(
            &raw const heap,
            D3D12_HEAP_FLAG_NONE,
            &raw const resource_description,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &raw mut gpu,
        )
    }
    .map_err(|error| {
        backend_error("ID3D12Device::CreateCommittedResource(texture)", &error).to_string()
    })?;
    let gpu = gpu.ok_or("D3D12 returned no guest texture")?;
    name_object(
        &gpu,
        &format!(
            "REmu texture address={:#x} {}x{} mips={} layers={}",
            descriptor.address,
            descriptor.width,
            descriptor.height,
            descriptor.mip_level_count,
            descriptor.layer_count
        ),
    );

    // Keep the exact raw bytes used to seed a writable resource. Async completion may race guest
    // allocator reuse, so publication must compare against this upload-time ownership baseline.
    let (levels, raw_baseline) = read_texture_mip_chain_with_raw(descriptor)?;
    let mut copy_footprints = Vec::new();
    let mut packed = Vec::<u8>::new();
    let mut row_pitch = 0_usize;
    let mut slice_pitch = 0_usize;
    if descriptor.mip_level_count > 1 {
        for (mip, linear) in levels.iter().enumerate() {
            let shift = u32::try_from(mip).expect("mip count is bounded");
            let width = descriptor
                .width
                .checked_shr(shift)
                .unwrap_or_default()
                .max(1);
            let height = descriptor
                .height
                .checked_shr(shift)
                .unwrap_or_default()
                .max(1);
            let width_elements = width.div_ceil(descriptor.block_extent[0]) as usize;
            let height_elements = height.div_ceil(descriptor.block_extent[1]) as usize;
            let row_bytes = width_elements
                .checked_mul(descriptor.bytes_per_element as usize)
                .ok_or("mip row byte count overflows")?;
            let mip_row_pitch = align_up(row_bytes, 256)?;
            let offset = align_up(packed.len(), 512)?;
            let byte_count = mip_row_pitch
                .checked_mul(height_elements)
                .ok_or("mip upload footprint overflows")?;
            packed.resize(offset + byte_count, 0);
            for row in 0..height_elements {
                packed[offset + row * mip_row_pitch..offset + row * mip_row_pitch + row_bytes]
                    .copy_from_slice(&linear[row * row_bytes..(row + 1) * row_bytes]);
            }
            copy_footprints.push(TextureCopyFootprint {
                offset: offset as u64,
                subresource: mip as u32,
                format: descriptor.format,
                width,
                height,
                depth: 1,
                row_pitch: mip_row_pitch as u32,
            });
            if mip == 0 {
                row_pitch = mip_row_pitch;
                slice_pitch = byte_count;
            }
        }
    } else {
        let linear = &levels[0];
        let [width_elements, height_elements] = descriptor.element_extent();
        let row_bytes = width_elements
            .checked_mul(descriptor.bytes_per_element as usize)
            .ok_or("texture row byte count overflows")?;
        row_pitch = align_up(row_bytes, 256)?;
        let packed_slice = row_pitch
            .checked_mul(height_elements)
            .ok_or("texture upload slice size overflows")?;
        slice_pitch = if dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
            packed_slice
        } else {
            align_up(packed_slice, 512)?
        };
        let total = slice_pitch
            .checked_mul(descriptor.layer_count as usize)
            .ok_or("texture upload allocation overflows")?;
        packed.resize(total.max(4), 0);
        let linear_slice = row_bytes
            .checked_mul(height_elements)
            .ok_or("texture linear slice size overflows")?;
        for layer in 0..descriptor.layer_count as usize {
            for row in 0..height_elements {
                let source = layer * linear_slice + row * row_bytes;
                let destination = layer * slice_pitch + row * row_pitch;
                packed[destination..destination + row_bytes]
                    .copy_from_slice(&linear[source..source + row_bytes]);
            }
        }
        if dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
            copy_footprints.push(TextureCopyFootprint {
                offset: 0,
                subresource: 0,
                format: descriptor.format,
                width: descriptor.width,
                height: descriptor.height,
                depth: descriptor.layer_count,
                row_pitch: row_pitch as u32,
            });
        } else {
            copy_footprints.extend(
                (0..descriptor.layer_count).map(|layer| TextureCopyFootprint {
                    offset: (layer as usize * slice_pitch) as u64,
                    subresource: layer,
                    format: descriptor.format,
                    width: descriptor.width,
                    height: descriptor.height,
                    depth: 1,
                    row_pitch: row_pitch as u32,
                }),
            );
        }
    }
    let packed_byte_count = packed.len().max(4);
    let writeback = if descriptor.writable && descriptor.address != 0 {
        let guest_byte_count = u64::try_from(raw_baseline.len())
            .map_err(|_| "writable texture byte count does not fit u64")?;
        let guard =
            if let Some(lease) = GuestRangeLease::acquire(descriptor.address, guest_byte_count) {
                TextureWritebackGuard::Lease(lease)
            } else {
                #[cfg(test)]
                {
                    TextureWritebackGuard::Direct
                }
                #[cfg(not(test))]
                {
                    // SAFETY: descriptor evaluation accepted this exact range as live readable and
                    // writable guest memory. Mapping replacement retires the watch before address
                    // reuse, and publication holds its registry lock while comparing and copying.
                    let watch = unsafe { CpuWriteWatch::new(descriptor.address, guest_byte_count) }
                    .map_err(|error| {
                        format!(
                            "could not pin D3D12 writable texture address={:#x} bytes={}: {error}",
                            descriptor.address,
                            raw_baseline.len()
                        )
                    })?;
                    TextureWritebackGuard::Anonymous(watch)
                }
            };
        Some(Arc::new(TextureWritebackState {
            baseline: raw_baseline,
            guard,
        }))
    } else {
        None
    };
    Ok(TextureResource {
        descriptor,
        gpu,
        upload: create_upload_buffer(device, &packed)?,
        readback: descriptor
            .writable
            .then(|| create_readback_buffer(device, packed_byte_count))
            .transpose()?,
        row_pitch: u32::try_from(row_pitch).map_err(|_| "texture row pitch exceeds u32")?,
        slice_pitch,
        copy_footprints,
        packed_byte_count,
        source: TextureResourceSource::GuestUpload,
        writeback,
    })
}

fn validate_texture_format_support(
    device: &ID3D12Device,
    descriptor: DecodedTexture,
    dimension: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DIMENSION,
) -> Result<(), String> {
    let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        Format: descriptor.format,
        ..D3D12_FEATURE_DATA_FORMAT_SUPPORT::default()
    };
    let size = u32::try_from(std::mem::size_of::<D3D12_FEATURE_DATA_FORMAT_SUPPORT>())
        .expect("D3D12 format-support structure fits u32");
    // SAFETY: `support` is the exact writable structure required by FORMAT_SUPPORT.
    unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_FORMAT_SUPPORT,
            (&raw mut support).cast::<c_void>(),
            size,
        )
    }
    .map_err(|error| {
        backend_error("ID3D12Device::CheckFeatureSupport(texture format)", &error).to_string()
    })?;

    let required_dimension = if dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        D3D12_FORMAT_SUPPORT1_TEXTURE3D
    } else {
        D3D12_FORMAT_SUPPORT1_TEXTURE2D
    };
    if !support.Support1.contains(required_dimension) {
        return Err(format!(
            "D3D12 format {:?} does not support the requested {:?} guest texture",
            descriptor.format, dimension
        ));
    }
    if descriptor.writable
        && (!support
            .Support1
            .contains(D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW)
            || !support
                .Support2
                .contains(D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE))
    {
        return Err(format!(
            "D3D12 format {:?} lacks typed UAV store support (support1={:#x} support2={:#x})",
            descriptor.format, support.Support1.0, support.Support2.0
        ));
    }
    Ok(())
}

fn create_or_reuse_sampled_texture(
    device: &ID3D12Device,
    descriptor: DecodedTexture,
    cache: &mut HashMap<DecodedTexture, CachedSampledTexture>,
) -> Result<TextureResource, String> {
    if !descriptor.writable {
        let reusable = cache.get_mut(&descriptor).is_some_and(|cached| {
            !cached.write_watch.is_retired() && !cached.write_watch.consume_dirty()
        });
        if reusable {
            if std::env::var_os("REMU_TRACE_D3D12_TIMING").is_some_and(|value| value == "1") {
                eprintln!(
                    "[REMu][D3D12-TIMING] sampled_texture_hit={:#x} extent={}x{}",
                    descriptor.address, descriptor.width, descriptor.height
                );
            }
            return Ok(cache
                .get(&descriptor)
                .expect("reusable cache entry remains present")
                .resource
                .clone());
        }
        cache.remove(&descriptor);
    }

    let resource = create_texture_resource(device, descriptor)?;
    if descriptor.writable || descriptor.address == 0 {
        return Ok(resource);
    }
    let raw_byte_count = descriptor.raw_byte_count()?;
    let byte_count =
        u64::try_from(raw_byte_count).map_err(|_| "sampled texture byte count does not fit u64")?;
    // SAFETY: evaluated guest image ranges remain mapped while their draw events are live. The
    // cache owns the watch and drops it before replacing or evicting the native texture.
    if let Ok(write_watch) =
        unsafe { remu_memory::CpuWriteWatch::new(descriptor.address, byte_count) }
    {
        if cache.len() >= MAXIMUM_RESIDENT_SAMPLED_TEXTURES {
            cache.clear();
        }
        let mut resident = resource.clone();
        resident.source = TextureResourceSource::ResidentSampledTexture;
        cache.insert(
            descriptor,
            CachedSampledTexture {
                resource: resident,
                write_watch,
            },
        );
    }
    Ok(resource)
}

fn create_texture_view(
    device: &ID3D12Device,
    resource: &TextureResource,
    handle: windows::Win32::Graphics::Direct3D12::D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Result<(), String> {
    let descriptor = resource.descriptor;
    if descriptor.writable {
        let (dimension, anonymous) = if descriptor.texture_type == 10 {
            (
                D3D12_UAV_DIMENSION_TEXTURE3D,
                D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture3D: D3D12_TEX3D_UAV {
                        MipSlice: 0,
                        FirstWSlice: 0,
                        WSize: descriptor.layer_count,
                    },
                },
            )
        } else if matches!(descriptor.texture_type, 11 | 13) {
            (
                D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
                D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_UAV {
                        MipSlice: 0,
                        FirstArraySlice: 0,
                        ArraySize: descriptor.layer_count,
                        PlaneSlice: 0,
                    },
                },
            )
        } else {
            (
                D3D12_UAV_DIMENSION_TEXTURE2D,
                D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_UAV {
                        MipSlice: 0,
                        PlaneSlice: 0,
                    },
                },
            )
        };
        let view = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: descriptor.format,
            ViewDimension: dimension,
            Anonymous: anonymous,
        };
        // SAFETY: the resource allows UAV access and the view matches its native dimension.
        unsafe {
            device.CreateUnorderedAccessView(
                &resource.gpu,
                None::<&ID3D12Resource>,
                Some(&raw const view),
                handle,
            )
        };
    } else {
        let (dimension, anonymous) = if descriptor.texture_type == 10 {
            (
                D3D12_SRV_DIMENSION_TEXTURE3D,
                D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture3D: D3D12_TEX3D_SRV {
                        MostDetailedMip: 0,
                        MipLevels: descriptor.mip_level_count,
                        ResourceMinLODClamp: 0.0,
                    },
                },
            )
        } else if matches!(descriptor.texture_type, 11 | 13) {
            (
                D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
                D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                        MostDetailedMip: 0,
                        MipLevels: descriptor.mip_level_count,
                        FirstArraySlice: 0,
                        ArraySize: descriptor.layer_count,
                        PlaneSlice: 0,
                        ResourceMinLODClamp: 0.0,
                    },
                },
            )
        } else {
            (
                D3D12_SRV_DIMENSION_TEXTURE2D,
                D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV {
                        MostDetailedMip: 0,
                        MipLevels: descriptor.mip_level_count,
                        PlaneSlice: 0,
                        ResourceMinLODClamp: 0.0,
                    },
                },
            )
        };
        let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: descriptor.format,
            ViewDimension: dimension,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: anonymous,
        };
        // SAFETY: the SRV dimension and format match the immutable resource.
        unsafe { device.CreateShaderResourceView(&resource.gpu, Some(&raw const view), handle) };
    }
    if trace_d3d12_bindings() {
        ensure_device_available(
            device,
            &format!("texture descriptor {:?}", resource.descriptor),
        )?;
    }
    Ok(())
}

fn trace_d3d12_bindings() -> bool {
    std::env::var_os("REMU_TRACE_D3D12_BINDINGS").is_some_and(|value| value == "1")
}

fn ensure_device_available(device: &ID3D12Device, context: &str) -> Result<(), String> {
    if unsafe { device.GetDeviceRemovedReason() }.is_ok() {
        Ok(())
    } else {
        Err(format!(
            "{context} removed the D3D12 device: {}",
            device_removed_report(device)
        ))
    }
}

fn decode_sampler(binding: &remu_shader::ImageBinding) -> Result<D3D12_SAMPLER_DESC, String> {
    let words = &binding.sampler_descriptor;
    if words.len() < 4 {
        return Err(format!(
            "sampler at PC {:#x} has fewer than four dwords",
            binding.pc
        ));
    }
    let address = |value| match value {
        0 => windows::Win32::Graphics::Direct3D12::D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        1 => windows::Win32::Graphics::Direct3D12::D3D12_TEXTURE_ADDRESS_MODE_MIRROR,
        2 => windows::Win32::Graphics::Direct3D12::D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        3 | 5 | 7 => windows::Win32::Graphics::Direct3D12::D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE,
        _ => windows::Win32::Graphics::Direct3D12::D3D12_TEXTURE_ADDRESS_MODE_BORDER,
    };
    let compare = remu_shader::image_is_comparison(binding.opcode);
    let linear = matches!((words[2] >> 20) & 3, 1 | 3)
        || matches!((words[2] >> 22) & 3, 1 | 3)
        || matches!((words[2] >> 26) & 3, 2 | 3);
    let filter = match (compare, linear) {
        (false, false) => windows::Win32::Graphics::Direct3D12::D3D12_FILTER_MIN_MAG_MIP_POINT,
        (false, true) => windows::Win32::Graphics::Direct3D12::D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        (true, false) => {
            windows::Win32::Graphics::Direct3D12::D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT
        }
        (true, true) => {
            windows::Win32::Graphics::Direct3D12::D3D12_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR
        }
    };
    let comparison = match (words[0] >> 12) & 7 {
        0 => windows::Win32::Graphics::Direct3D12::D3D12_COMPARISON_FUNC_NEVER,
        1 => windows::Win32::Graphics::Direct3D12::D3D12_COMPARISON_FUNC_LESS,
        2 => windows::Win32::Graphics::Direct3D12::D3D12_COMPARISON_FUNC_EQUAL,
        3 => windows::Win32::Graphics::Direct3D12::D3D12_COMPARISON_FUNC_LESS_EQUAL,
        4 => windows::Win32::Graphics::Direct3D12::D3D12_COMPARISON_FUNC_GREATER,
        5 => windows::Win32::Graphics::Direct3D12::D3D12_COMPARISON_FUNC_NOT_EQUAL,
        6 => windows::Win32::Graphics::Direct3D12::D3D12_COMPARISON_FUNC_GREATER_EQUAL,
        _ => windows::Win32::Graphics::Direct3D12::D3D12_COMPARISON_FUNC_ALWAYS,
    };
    let bias_raw = (words[2] & 0x3fff) as i32;
    let guest_bias = ((bias_raw ^ 0x2000) - 0x2000) as f32 / 256.0;
    let bias = guest_bias.clamp(D3D12_MIP_LOD_BIAS_MIN, D3D12_MIP_LOD_BIAS_MAX);
    let min_lod = (words[1] & 0xfff) as f32 / 256.0;
    let guest_max_lod = ((words[1] >> 12) & 0xfff) as f32 / 256.0;
    // GCN exposes independent U4.8 bounds and an S5.8 bias. D3D12 accepts a narrower
    // bias and requires MaxLOD >= MinLOD, so preserve the guest's lower bound while
    // collapsing an inverted interval to the nearest representable D3D12 interval.
    let max_lod = guest_max_lod.max(min_lod);
    if trace_d3d12_bindings() && (bias != guest_bias || max_lod != guest_max_lod) {
        eprintln!(
            "d3d12: normalized sampler at PC {:#x}: bias {guest_bias} -> {bias}, LOD [{min_lod}, {guest_max_lod}] -> [{min_lod}, {max_lod}]",
            binding.pc
        );
    }
    Ok(D3D12_SAMPLER_DESC {
        Filter: filter,
        AddressU: address(words[0] & 7),
        AddressV: address((words[0] >> 3) & 7),
        AddressW: address((words[0] >> 6) & 7),
        MipLODBias: bias,
        MaxAnisotropy: 1_u32 << ((words[0] >> 9) & 7).min(4),
        ComparisonFunc: comparison,
        BorderColor: [0.0; 4],
        MinLOD: min_lod,
        MaxLOD: max_lod,
    })
}

fn record_texture_uploads(commands: &ID3D12GraphicsCommandList, textures: &[TextureResource]) {
    for resource in textures {
        let descriptor = resource.descriptor;
        match resource.source {
            TextureResourceSource::GuestUpload => {
                for footprint in &resource.copy_footprints {
                    record_texture_copy_footprint(
                        commands,
                        &resource.upload,
                        &resource.gpu,
                        *footprint,
                        false,
                    );
                }
                let state = if descriptor.writable {
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                } else {
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                        | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
                };
                record_transition(
                    commands,
                    &resource.gpu,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    state,
                );
            }
            TextureResourceSource::ResidentRenderTarget => record_transition(
                commands,
                &resource.gpu,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                    | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            ),
            TextureResourceSource::ResidentSampledTexture => {}
        }
    }
}

fn record_texture_readbacks(commands: &ID3D12GraphicsCommandList, textures: &[TextureResource]) {
    for resource in textures
        .iter()
        .filter(|resource| resource.source == TextureResourceSource::ResidentRenderTarget)
    {
        record_transition(
            commands,
            &resource.gpu,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
    }
    for resource in textures.iter().filter(|resource| {
        resource.source == TextureResourceSource::GuestUpload && resource.descriptor.writable
    }) {
        let readback = resource
            .readback
            .as_ref()
            .expect("writable texture has readback");
        record_transition(
            commands,
            &resource.gpu,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        for footprint in &resource.copy_footprints {
            record_texture_copy_footprint(commands, readback, &resource.gpu, *footprint, true);
        }
    }
}

fn record_texture_copy_footprint(
    commands: &ID3D12GraphicsCommandList,
    buffer: &ID3D12Resource,
    texture: &ID3D12Resource,
    copy: TextureCopyFootprint,
    texture_to_buffer: bool,
) {
    let mut footprint = D3D12_TEXTURE_COPY_LOCATION {
        pResource: ManuallyDrop::new(Some(buffer.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: copy.offset,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: copy.format,
                    Width: copy.width,
                    Height: copy.height,
                    Depth: copy.depth,
                    RowPitch: copy.row_pitch,
                },
            },
        },
    };
    let mut subresource_location = D3D12_TEXTURE_COPY_LOCATION {
        pResource: ManuallyDrop::new(Some(texture.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: copy.subresource,
        },
    };
    // SAFETY: both copy locations name complete live subresources and D3D12-aligned footprints.
    unsafe {
        if texture_to_buffer {
            commands.CopyTextureRegion(
                &raw const footprint,
                0,
                0,
                0,
                &raw const subresource_location,
                None,
            );
        } else {
            commands.CopyTextureRegion(
                &raw const subresource_location,
                0,
                0,
                0,
                &raw const footprint,
                None,
            );
        }
        ManuallyDrop::drop(&mut footprint.pResource);
        ManuallyDrop::drop(&mut subresource_location.pResource);
    }
}

#[allow(clippy::too_many_arguments)]
fn record_texture_copy(
    commands: &ID3D12GraphicsCommandList,
    buffer: &ID3D12Resource,
    texture: &ID3D12Resource,
    offset: u64,
    subresource: u32,
    resource: &TextureResource,
    depth: u32,
    texture_to_buffer: bool,
) {
    let mut footprint = D3D12_TEXTURE_COPY_LOCATION {
        pResource: ManuallyDrop::new(Some(buffer.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: offset,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: resource.descriptor.format,
                    Width: resource.descriptor.width,
                    Height: resource.descriptor.height,
                    Depth: depth,
                    RowPitch: resource.row_pitch,
                },
            },
        },
    };
    let mut subresource_location = D3D12_TEXTURE_COPY_LOCATION {
        pResource: ManuallyDrop::new(Some(texture.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: subresource,
        },
    };
    // SAFETY: both locations describe live resources and the footprint is fully initialized.
    unsafe {
        if texture_to_buffer {
            commands.CopyTextureRegion(
                &raw const footprint,
                0,
                0,
                0,
                &raw const subresource_location,
                None,
            );
        } else {
            commands.CopyTextureRegion(
                &raw const subresource_location,
                0,
                0,
                0,
                &raw const footprint,
                None,
            );
        }
        ManuallyDrop::drop(&mut footprint.pResource);
        ManuallyDrop::drop(&mut subresource_location.pResource);
    }
}

fn write_back_textures(textures: &[TextureResource]) -> Result<(), String> {
    for resource in textures
        .iter()
        .filter(|resource| resource.descriptor.writable)
    {
        let readback = resource
            .readback
            .as_ref()
            .expect("writable texture has readback");
        let packed = read_readback(readback, resource.packed_byte_count)
            .map_err(|error| error.to_string())?;
        let [width, height] = resource.descriptor.element_extent();
        let row_bytes = width * resource.descriptor.bytes_per_element as usize;
        let linear_slice = row_bytes * height;
        let mut linear = vec![0_u8; linear_slice * resource.descriptor.layer_count as usize];
        for layer in 0..resource.descriptor.layer_count as usize {
            for row in 0..height {
                let source = layer * resource.slice_pitch + row * resource.row_pitch as usize;
                let destination = layer * linear_slice + row * row_bytes;
                linear[destination..destination + row_bytes]
                    .copy_from_slice(&packed[source..source + row_bytes]);
            }
        }
        publish_texture_linear(resource, &linear)?;
    }
    Ok(())
}

fn publish_texture_linear(resource: &TextureResource, linear: &[u8]) -> Result<(), String> {
    if resource.descriptor.address == 0 {
        return Ok(());
    }
    let writeback = resource
        .writeback
        .as_ref()
        .ok_or("D3D12 writable texture has no publication guard")?;
    let candidate =
        encode_texture_linear_over(resource.descriptor, linear, writeback.baseline.as_slice())?;
    if std::env::var_os("REMU_TRACE_D3D12_WRITEBACKS").is_some_and(|value| value == "1") {
        eprintln!(
            "[REMu][D3D12-WRITEBACK] guarded-texture address={:#x} raw_bytes={:#x} width={} height={} pitch={} tile={} format={:?} bytes_per_element={} layers={}",
            resource.descriptor.address,
            candidate.len(),
            resource.descriptor.width,
            resource.descriptor.height,
            resource.descriptor.pitch,
            resource.descriptor.tile_mode,
            resource.descriptor.format,
            resource.descriptor.bytes_per_element,
            resource.descriptor.layer_count,
        );
    }
    match &writeback.guard {
        TextureWritebackGuard::Lease(lease) => write_changed_ranges_leased(
            resource.descriptor.address,
            &candidate,
            &writeback.baseline,
            lease,
            WritebackJournalSites::TEXTURE,
            "texture",
            TEXTURE_WRITEBACK_CHUNK_BYTES,
        ),
        #[cfg(not(test))]
        TextureWritebackGuard::Anonymous(watch) => match watch.with_live_mapping(|| {
            write_changed_ranges_direct(
                resource.descriptor.address,
                &candidate,
                &writeback.baseline,
                WritebackJournalSites::TEXTURE.direct,
                "texture",
                TEXTURE_WRITEBACK_CHUNK_BYTES,
            )
        }) {
            Ok(result) => result,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                trace_discarded_writeback(
                    resource.descriptor.address,
                    writeback.baseline.len(),
                    "anonymous texture mapping retired",
                );
                Ok(())
            }
            Err(error) => Err(format!(
                "D3D12 writable texture mapping could not be pinned for completion address={:#x} bytes={}: {error}",
                resource.descriptor.address,
                writeback.baseline.len()
            )),
        },
        #[cfg(test)]
        TextureWritebackGuard::Direct => write_changed_ranges_direct(
            resource.descriptor.address,
            &candidate,
            &writeback.baseline,
            WritebackJournalSites::TEXTURE.direct,
            "texture",
            TEXTURE_WRITEBACK_CHUNK_BYTES,
        ),
    }
}

fn align_up(value: usize, alignment: usize) -> Result<usize, String> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| "alignment calculation overflows".to_owned())
}

fn snapshot_global(binding: &remu_shader::GlobalMemoryBinding) -> Result<Vec<u8>, String> {
    if binding.byte_count == 0 {
        return Ok(Vec::new());
    }
    if binding.data.len() >= binding.byte_count {
        return Ok(binding.data[..binding.byte_count].to_vec());
    }
    if binding.base_address < 0x10_000 {
        return Err(format!(
            "global binding has invalid guest address {:#x}",
            binding.base_address
        ));
    }
    // SAFETY: shader evaluation only publishes mapped guest ranges. Queue ordering prevents the
    // guest producer from reusing this range until the synchronous dispatch retires.
    Ok(unsafe {
        std::slice::from_raw_parts(binding.base_address as *const u8, binding.byte_count).to_vec()
    })
}

fn snapshot_ordered_global(
    binding: &remu_shader::GlobalMemoryBinding,
    ordered_versions: &OrderedGlobalVersions,
) -> Result<Vec<u8>, String> {
    let packet_bytes = snapshot_global(binding)?;
    let Some(forwarded) =
        ordered_versions.forwarded_slice(binding.base_address, packet_bytes.as_slice())
    else {
        return Ok(packet_bytes);
    };
    if std::env::var_os("REMU_TRACE_D3D12_BINDINGS").is_some_and(|value| value == "1") {
        eprintln!(
            "[REMu][D3D12-BINDINGS] forwarded ordered global address={:#x} bytes={} packet=[{}] gpu=[{}]",
            binding.base_address,
            binding.byte_count,
            preview_words(&packet_bytes),
            preview_words(forwarded),
        );
    }
    Ok(forwarded.to_vec())
}

fn snapshot_live_global(binding: &remu_shader::GlobalMemoryBinding) -> Result<Vec<u8>, String> {
    if binding.byte_count == 0 {
        return Ok(Vec::new());
    }
    if binding.base_address < 0x10_000 {
        return binding
            .data
            .get(..binding.byte_count)
            .map(<[u8]>::to_vec)
            .ok_or_else(|| {
                format!(
                    "global binding has invalid guest address {:#x}",
                    binding.base_address
                )
            });
    }
    let mut bytes = vec![0_u8; binding.byte_count];
    read_process_memory(binding.base_address, &mut bytes)?;
    Ok(bytes)
}

fn write_back_globals(
    evaluation: &ShaderEvaluation,
    resources: &[GlobalResource],
    ordered_versions: &mut OrderedGlobalVersions,
) -> Result<(), String> {
    if evaluation.global_memory_bindings.len() != resources.len() {
        return Err("D3D12/global buffer binding count mismatch".to_owned());
    }
    for (binding, resource) in evaluation.global_memory_bindings.iter().zip(resources) {
        if !binding.write_back_to_guest || binding.byte_count == 0 {
            continue;
        }
        let readback = resource
            .readback
            .as_ref()
            .ok_or("writable global binding has no readback resource")?;
        let bytes = read_readback(readback, resource.byte_count.max(4))
            .map_err(|error| error.to_string())?;
        if binding.base_address < 0x10_000 {
            return Err(format!(
                "global writeback has invalid guest address {:#x}",
                binding.base_address
            ));
        }
        if resource.original_bytes.len() != binding.byte_count {
            return Err(format!(
                "D3D12 writable global baseline length mismatch address={:#x} bytes={} baseline={}",
                binding.base_address,
                binding.byte_count,
                resource.original_bytes.len()
            ));
        }
        let current_bytes = bytes
            .get(..binding.byte_count)
            .ok_or("D3D12 writable global readback is shorter than its guest range")?;
        if binding.byte_count <= 64
            && std::env::var_os("REMU_TRACE_D3D12_BINDINGS").is_some_and(|value| value == "1")
        {
            let guest_bytes = snapshot_live_global(binding)?;
            eprintln!(
                "[REMu][D3D12-WRITEBACK] address={:#x} baseline=[{}] gpu=[{}] guest=[{}]",
                binding.base_address,
                preview_words(&resource.original_bytes),
                preview_words(current_bytes),
                preview_words(&guest_bytes),
            );
        }
        let guard = resource
            .writeback_guard
            .as_ref()
            .ok_or("D3D12 writable global binding has no publication guard")?;
        match guard {
            GlobalWritebackGuard::Lease(lease) => {
                write_changed_ranges_leased(
                    binding.base_address,
                    current_bytes,
                    &resource.original_bytes,
                    lease,
                    WritebackJournalSites::GLOBAL,
                    "global",
                    0,
                )?;
            }
            GlobalWritebackGuard::Anonymous(watch) => {
                match watch.with_live_mapping(|| {
                    write_changed_ranges_direct(
                        binding.base_address,
                        current_bytes,
                        &resource.original_bytes,
                        WritebackJournalSites::GLOBAL.direct,
                        "global",
                        0,
                    )
                }) {
                    Ok(result) => result?,
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                        trace_discarded_writeback(
                            binding.base_address,
                            binding.byte_count,
                            "anonymous mapping retired",
                        );
                    }
                    Err(error) => {
                        return Err(format!(
                            "D3D12 writable global mapping could not be pinned for completion address={:#x} bytes={}: {error}",
                            binding.base_address, binding.byte_count
                        ));
                    }
                }
            }
            #[cfg(test)]
            GlobalWritebackGuard::Direct => write_changed_ranges_direct(
                binding.base_address,
                current_bytes,
                &resource.original_bytes,
                WritebackJournalSites::GLOBAL.direct,
                "global",
                0,
            )?,
        }
        let identity = match guard {
            GlobalWritebackGuard::Lease(lease) => Some(lease.identity()),
            GlobalWritebackGuard::Anonymous(_) => guest_shared_backing_identity(
                binding.base_address,
                u64::try_from(binding.byte_count).unwrap_or(u64::MAX),
            ),
            #[cfg(test)]
            GlobalWritebackGuard::Direct => None,
        };
        ordered_versions.record(
            binding.base_address,
            identity,
            &resource.original_bytes,
            current_bytes,
        );
        if binding.byte_count <= 64
            && std::env::var_os("REMU_TRACE_D3D12_BINDINGS").is_some_and(|value| value == "1")
        {
            let guest_bytes = snapshot_live_global(binding)?;
            eprintln!(
                "[REMu][D3D12-WRITEBACK] address={:#x} published=[{}]",
                binding.base_address,
                preview_words(&guest_bytes),
            );
        }
    }
    Ok(())
}

fn preview_words(bytes: &[u8]) -> String {
    bytes
        .chunks(4)
        .take(16)
        .map(|chunk| {
            let mut word = [0_u8; 4];
            word[..chunk.len()].copy_from_slice(chunk);
            format!("{:08x}", u32::from_le_bytes(word))
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn write_changed_ranges_leased(
    guest_address: u64,
    current_bytes: &[u8],
    original_bytes: &[u8],
    lease: &GuestRangeLease,
    sites: WritebackJournalSites,
    resource_kind: &str,
    chunk_bytes: usize,
) -> Result<(), String> {
    visit_changed_byte_ranges(current_bytes, original_bytes, chunk_bytes, |start, end| {
        let run_offset = u64::try_from(start).map_err(|_| {
            format!("D3D12 writable {resource_kind} changed offset does not fit u64")
        })?;
        let address = guest_address
            .checked_add(run_offset)
            .ok_or_else(|| format!("D3D12 writable {resource_kind} changed address overflows"))?;
        let source = &current_bytes[start..end];
        let baseline = &original_bytes[start..end];
        let outcome = lease.publish_run_if_baseline(run_offset, baseline, source);
        if current_bytes.len() <= 64
            && std::env::var_os("REMU_TRACE_D3D12_BINDINGS").is_some_and(|value| value == "1")
        {
            eprintln!(
                "[REMu][D3D12-WRITEBACK] address={address:#x} bytes={} guard=lease outcome={outcome:?} live_va_matches={}",
                source.len(),
                lease.live_va_matches(),
            );
        }
        match outcome {
            LeasePublishOutcome::AppliedAtLiveVa => {
                journal_guest_write(sites.live, address, source)
            }
            LeasePublishOutcome::AppliedByIdentity => {
                journal_guest_write(sites.identity, address, source)
            }
            LeasePublishOutcome::GuestModified => {
                trace_kept_guest_bytes(address, source.len(), "leased")
            }
            LeasePublishOutcome::Revoked => {
                trace_discarded_writeback(address, source.len(), "lease revoked");
                return Ok(false);
            }
            LeasePublishOutcome::WriteFailed => {
                return Err(format!(
                    "D3D12 writable {resource_kind} backing write failed address={address:#x} bytes={} identity={:?}",
                    source.len(),
                    lease.identity()
                ));
            }
        }
        Ok(true)
    })
}

fn write_changed_ranges_direct(
    guest_address: u64,
    current_bytes: &[u8],
    original_bytes: &[u8],
    site: GuestWriteSite,
    resource_kind: &str,
    chunk_bytes: usize,
) -> Result<(), String> {
    let destination = usize::try_from(guest_address)
        .map_err(|_| format!("D3D12 writable {resource_kind} address does not fit usize"))?;
    visit_changed_byte_ranges(current_bytes, original_bytes, chunk_bytes, |start, end| {
        // SAFETY: Production holds the CpuWriteWatch registry lock across this comparison and
        // any following copy. Unit tests retain their destination allocation for the whole call.
        let guest_now = unsafe {
            std::slice::from_raw_parts((destination as *const u8).add(start), end - start)
        };
        if guest_now != &original_bytes[start..end] {
            trace_kept_guest_bytes(
                guest_address.checked_add(start as u64).ok_or_else(|| {
                    format!("D3D12 writable {resource_kind} changed address overflows")
                })?,
                end - start,
                "anonymous",
            );
            return Ok(true);
        }
        let address = guest_address
            .checked_add(start as u64)
            .ok_or_else(|| format!("D3D12 writable {resource_kind} changed address overflows"))?;
        journal_guest_write(site, address, &current_bytes[start..end]);
        // SAFETY: the same mapping lock keeps the destination live and prevents replacement.
        unsafe {
            std::ptr::copy_nonoverlapping(
                current_bytes[start..end].as_ptr(),
                (destination as *mut u8).add(start),
                end - start,
            );
        }
        Ok(true)
    })
}

fn visit_changed_byte_ranges(
    current_bytes: &[u8],
    original_bytes: &[u8],
    chunk_bytes: usize,
    mut visit: impl FnMut(usize, usize) -> Result<bool, String>,
) -> Result<(), String> {
    if current_bytes.len() != original_bytes.len() {
        return Err(format!(
            "D3D12 writable global comparison length mismatch current={} baseline={}",
            current_bytes.len(),
            original_bytes.len()
        ));
    }
    if chunk_bytes != 0 {
        for start in (0..current_bytes.len()).step_by(chunk_bytes) {
            let end = start.saturating_add(chunk_bytes).min(current_bytes.len());
            if current_bytes[start..end] != original_bytes[start..end] && !visit(start, end)? {
                break;
            }
        }
        return Ok(());
    }
    let mut cursor = 0;
    while cursor < current_bytes.len() {
        while cursor < current_bytes.len() && current_bytes[cursor] == original_bytes[cursor] {
            cursor += 1;
        }
        let start = cursor;
        while cursor < current_bytes.len() && current_bytes[cursor] != original_bytes[cursor] {
            cursor += 1;
        }
        if start != cursor {
            if !visit(start, cursor)? {
                break;
            }
        }
    }
    Ok(())
}

fn journal_guest_write(site: GuestWriteSite, address: u64, bytes: &[u8]) {
    remu_memory::record_guest_write(
        site,
        address,
        bytes.len() as u64,
        bytes,
        remu_gpu::gpu_completed_submission_sequence(),
    );
}

fn trace_kept_guest_bytes(address: u64, byte_count: usize, guard: &str) {
    static KEPT: AtomicUsize = AtomicUsize::new(0);
    let sequence = KEPT.fetch_add(1, Ordering::Relaxed) + 1;
    if sequence <= 16 || sequence.is_power_of_two() {
        eprintln!(
            "[REMu][D3D12] kept newer guest bytes over GPU writeback sequence={sequence} address={address:#x} bytes={byte_count} guard={guard}"
        );
    }
}

fn trace_discarded_writeback(address: u64, byte_count: usize, reason: &str) {
    static DISCARDED: AtomicUsize = AtomicUsize::new(0);
    let sequence = DISCARDED.fetch_add(1, Ordering::Relaxed) + 1;
    if sequence <= 16 || sequence.is_power_of_two() {
        eprintln!(
            "[REMu][D3D12] discarded stale GPU writeback sequence={sequence} address={address:#x} bytes={byte_count} reason={reason}"
        );
    }
}

fn create_upload_buffer(device: &ID3D12Device, bytes: &[u8]) -> Result<ID3D12Resource, String> {
    let resource = create_buffer(
        device,
        bytes.len().max(4),
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_FLAG_NONE,
    )?;
    let mut padded = Vec::new();
    let bytes = if bytes.is_empty() {
        padded.resize(4, 0);
        padded.as_slice()
    } else {
        bytes
    };
    write_upload(&resource, bytes).map_err(|error| error.to_string())?;
    Ok(resource)
}

fn create_gpu_buffer(device: &ID3D12Device, byte_count: usize) -> Result<ID3D12Resource, String> {
    create_buffer(
        device,
        byte_count,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    )
}

fn create_readback_buffer(
    device: &ID3D12Device,
    byte_count: usize,
) -> Result<ID3D12Resource, String> {
    create_buffer(
        device,
        byte_count,
        D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_STATE_COPY_DEST,
        windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_FLAG_NONE,
    )
}

fn create_buffer(
    device: &ID3D12Device,
    byte_count: usize,
    heap_type: windows::Win32::Graphics::Direct3D12::D3D12_HEAP_TYPE,
    state: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES,
    flags: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_FLAGS,
) -> Result<ID3D12Resource, String> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: heap_type,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..D3D12_HEAP_PROPERTIES::default()
    };
    let description = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: u64::try_from(byte_count).map_err(|_| "D3D12 buffer size does not fit u64")?,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..D3D12_RESOURCE_DESC::default()
    };
    let mut resource = None;
    // SAFETY: descriptors are complete and the output slot receives an owned COM reference.
    unsafe {
        device.CreateCommittedResource(
            &raw const heap,
            D3D12_HEAP_FLAG_NONE,
            &raw const description,
            state,
            None,
            &raw mut resource,
        )
    }
    .map_err(|error| {
        backend_error("ID3D12Device::CreateCommittedResource(guest)", &error).to_string()
    })?;
    resource.ok_or_else(|| "D3D12 returned no guest buffer".to_owned())
}

fn create_structured_srv(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    first_word: usize,
    word_count: usize,
    handle: windows::Win32::Graphics::Direct3D12::D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Result<(), String> {
    let description = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_UNKNOWN,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                FirstElement: u64::try_from(first_word)
                    .map_err(|_| "structured SRV offset is too large")?,
                NumElements: u32::try_from(word_count.max(1))
                    .map_err(|_| "structured SRV is too large")?,
                StructureByteStride: 4,
                Flags: D3D12_BUFFER_SRV_FLAG_NONE,
            },
        },
    };
    // SAFETY: the upload resource contains word_count tightly packed u32 values.
    unsafe { device.CreateShaderResourceView(resource, Some(&raw const description), handle) };
    Ok(())
}

fn create_raw_uav(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    byte_count: usize,
    handle: windows::Win32::Graphics::Direct3D12::D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Result<(), String> {
    let description = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_UAV {
                FirstElement: 0,
                NumElements: u32::try_from(byte_count.div_ceil(4))
                    .map_err(|_| "raw UAV is too large")?,
                StructureByteStride: 0,
                CounterOffsetInBytes: 0,
                Flags: D3D12_BUFFER_UAV_FLAG_RAW,
            },
        },
    };
    // SAFETY: the default buffer allows UAV access and the descriptor covers its dwords.
    unsafe {
        device.CreateUnorderedAccessView(
            resource,
            None::<&ID3D12Resource>,
            Some(&raw const description),
            handle,
        )
    };
    Ok(())
}

fn create_raw_srv(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    byte_offset: usize,
    byte_count: usize,
    handle: windows::Win32::Graphics::Direct3D12::D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Result<(), String> {
    if !byte_offset.is_multiple_of(D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT as usize) {
        return Err(format!(
            "raw SRV offset is not {D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT}-byte aligned"
        ));
    }
    let description = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                FirstElement: u64::try_from(byte_offset / 4)
                    .map_err(|_| "raw SRV offset is too large")?,
                NumElements: u32::try_from(byte_count.div_ceil(4))
                    .map_err(|_| "raw SRV is too large")?,
                StructureByteStride: 0,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
            },
        },
    };
    // SAFETY: the upload buffer remains in GENERIC_READ and covers the raw dword view.
    unsafe { device.CreateShaderResourceView(resource, Some(&raw const description), handle) };
    Ok(())
}

pub(super) fn create_guest_root_signature(
    device: &ID3D12Device,
) -> Result<ID3D12RootSignature, String> {
    let spaces = [0_u32, 1, 2];
    let visibility = [
        windows::Win32::Graphics::Direct3D12::D3D12_SHADER_VISIBILITY_VERTEX,
        windows::Win32::Graphics::Direct3D12::D3D12_SHADER_VISIBILITY_PIXEL,
        D3D12_SHADER_VISIBILITY_ALL,
    ];
    let parameters = std::array::from_fn::<_, 3, _>(|index| D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Constants: D3D12_ROOT_CONSTANTS {
                ShaderRegister: 0,
                RegisterSpace: spaces[index],
                Num32BitValues: 7,
            },
        },
        ShaderVisibility: visibility[index],
    });
    let flags = D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
        | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED
        | windows::Win32::Graphics::Direct3D12::D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    let description = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: parameters.len() as u32,
        pParameters: parameters.as_ptr(),
        NumStaticSamplers: 0,
        pStaticSamplers: std::ptr::null(),
        Flags: flags,
    };
    let mut blob = None;
    let mut diagnostics = None;
    // SAFETY: all pointers remain valid for the synchronous serializer call.
    unsafe {
        D3D12SerializeRootSignature(
            &raw const description,
            D3D_ROOT_SIGNATURE_VERSION_1,
            &raw mut blob,
            Some(&raw mut diagnostics),
        )
    }
    .map_err(|error| backend_error("D3D12SerializeRootSignature(guest)", &error).to_string())?;
    let blob: ID3DBlob = blob.ok_or("root-signature serializer returned no blob")?;
    let bytes = blob_bytes(&blob);
    // SAFETY: serializer produced a complete version-one root signature.
    unsafe { device.CreateRootSignature(0, bytes) }.map_err(|error| {
        backend_error("ID3D12Device::CreateRootSignature(guest)", &error).to_string()
    })
}

fn create_compute_pipeline(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    dxil: &[u8],
) -> Result<ID3D12PipelineState, String> {
    let mut description = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root.clone())),
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: dxil.as_ptr().cast::<c_void>(),
            BytecodeLength: dxil.len(),
        },
        ..D3D12_COMPUTE_PIPELINE_STATE_DESC::default()
    };
    // SAFETY: root signature and DXIL bytes remain live for synchronous PSO creation.
    let result =
        unsafe { device.CreateComputePipelineState(&raw const description) }.map_err(|error| {
            backend_error("ID3D12Device::CreateComputePipelineState(guest)", &error).to_string()
        });
    // SAFETY: release the explicit root-signature clone held in the C union field.
    unsafe { ManuallyDrop::drop(&mut description.pRootSignature) };
    result
}

fn record_transition(
    commands: &ID3D12GraphicsCommandList,
    resource: &ID3D12Resource,
    before: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES,
    after: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES,
) {
    let mut barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: ManuallyDrop::new(Some(resource.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    // SAFETY: Transition is the active union member and the recorded states match resource use.
    unsafe {
        commands.ResourceBarrier(std::slice::from_ref(&barrier));
        ManuallyDrop::drop(&mut (*barrier.Anonymous.Transition).pResource);
    }
}

fn words_as_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn blob_bytes(blob: &ID3DBlob) -> &[u8] {
    // SAFETY: ID3DBlob owns immutable serialized bytes for its complete lifetime.
    unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
    }
}

// These resources are intentionally named fields: they keep descriptor-referenced allocations
// alive until queue completion even when Rust otherwise sees no later direct reads.
impl Drop for ComputeResources {
    fn drop(&mut self) {
        let _ = (&self.binding_table, &self.initial_scalars);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use remu_gpu::{
        ComputeDispatch, ComputeDispatchSource, ComputeGrid, ComputeLaunch, GpuMemoryAcquire,
        GpuRegister, GpuScanoutTarget, IndexedDraw, ShaderSnapshot,
    };
    use remu_shader::{
        BufferMemoryControl, ExportControl, GlobalMemoryBinding, GlobalMemoryControl, ImageBinding,
        ImageControl, Instruction, InstructionControl, Opcode, Operand, OperandKind, Program,
        ShaderEncoding, ShaderEvaluation,
    };

    use super::{
        D3D12_MIP_LOD_BIAS_MAX, Dx12GuestExecutor, OrderedGlobalVersions, WritebackJournalSites,
        decode_sampler, dispatch_group_count, expand_primitive_indices, indirect_draw_count_allows,
        presentable_bgra, primitive_restart_index, regroup_restart_list_indices,
        resolve_indirect_compute, resolve_indirect_compute_ordered, resolve_indirect_draw,
        snapshot_ordered_global, write_changed_ranges_direct,
    };

    fn dispatch_with_grid(grid: ComputeGrid) -> ComputeDispatch {
        ComputeDispatch {
            sequence: 1,
            source: ComputeDispatchSource::Direct,
            launch: ComputeLaunch {
                grid,
                group_count: [1; 3],
                base_group: [0; 3],
                threads_per_group: [1; 3],
                wave_lane_count: 32,
                lds_size_bytes: 0,
                initiator: 1,
            },
            shader_address: 0,
            shader_prefix: [0; 4],
            shader: None,
            shader_evaluation: None,
            shader_registers: Vec::new(),
        }
    }

    #[test]
    fn normalizes_guest_sampler_to_the_d3d12_lod_domain() {
        let min_lod_raw = 0x18d;
        let max_lod_raw = 0x040;
        let binding = ImageBinding {
            pc: 0x1234,
            opcode: Opcode::Known("ImageSample"),
            control: ImageControl {
                dmask: 1,
                vector_address: 0,
                address_registers: vec![0, 1],
                vector_data: 4,
                scalar_resource: 0,
                scalar_sampler: 0,
                dimension: 1,
                is_array: false,
                glc: false,
                slc: false,
                a16: false,
                d16: false,
                unorm: false,
            },
            resource_descriptor: Vec::new().into(),
            sampler_descriptor: vec![0, min_lod_raw | (max_lod_raw << 12), 0x1ba8, 0].into(),
            mip_level: None,
        };

        let sampler = decode_sampler(&binding).expect("guest sampler should be transcodable");
        assert_eq!(sampler.MipLODBias, D3D12_MIP_LOD_BIAS_MAX);
        assert_eq!(sampler.MinLOD, min_lod_raw as f32 / 256.0);
        assert_eq!(sampler.MaxLOD, sampler.MinLOD);
    }

    #[test]
    fn rejects_dispatch_dimensions_that_would_remove_a_d3d12_device() {
        assert_eq!(
            dispatch_group_count(&dispatch_with_grid(ComputeGrid::Threadgroups([
                65_535, 2, 1
            ])))
            .expect("maximum native dispatch is valid"),
            [65_535, 2, 1]
        );
        let error = dispatch_group_count(&dispatch_with_grid(ComputeGrid::Threadgroups([
            65_536, 2, 1,
        ])))
        .expect_err("oversized native dispatch must be rejected");
        assert!(error.contains("x-dimension 65536"), "{error}");
    }

    #[test]
    fn converts_rgba_scanout_bytes_to_presenter_bgra() {
        assert_eq!(
            presentable_bgra(
                windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
                vec![1, 2, 3, 4, 10, 20, 30, 40],
            ),
            Some(vec![3, 2, 1, 4, 30, 20, 10, 40])
        );
    }

    #[test]
    fn writable_global_publication_preserves_newer_guest_bytes() {
        let original = [0_u8; 12];
        let current = [1_u8; 12];
        let mut guest = original;
        guest[4..8].fill(9);

        write_changed_ranges_direct(
            guest.as_mut_ptr() as u64,
            &current,
            &original,
            WritebackJournalSites::GLOBAL.direct,
            "global",
            0,
        )
        .expect("publish baseline-guarded D3D12 global bytes");

        // A single contiguous GPU change is conservatively withheld when any byte in the run
        // changed on the CPU. This prevents a stale full-buffer snapshot from corrupting the
        // guest-owned update.
        assert_eq!(guest, [0, 0, 0, 0, 9, 9, 9, 9, 0, 0, 0, 0]);
    }

    #[test]
    fn linear_texture_writeback_preserves_guest_pitch_padding() {
        const WIDTH: u32 = 2;
        const HEIGHT: u32 = 2;
        const PITCH: u32 = 4;
        const BYTES_PER_PIXEL: usize = 4;
        let mut raw = vec![0xa5_u8; (PITCH * HEIGHT) as usize * BYTES_PER_PIXEL];
        let descriptor = super::DecodedTexture {
            address: raw.as_mut_ptr() as u64,
            width: WIDTH,
            height: HEIGHT,
            pitch: PITCH,
            tile_mode: 0,
            format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
            bytes_per_element: BYTES_PER_PIXEL as u32,
            block_extent: [1, 1],
            texture_type: 9,
            layer_count: 1,
            mip_level_count: 1,
            writable: true,
        };
        let linear = vec![0x11_u8; (WIDTH * HEIGHT) as usize * BYTES_PER_PIXEL];

        super::write_texture_linear(descriptor, &linear)
            .expect("publish a pitched linear guest texture");

        let row_bytes = WIDTH as usize * BYTES_PER_PIXEL;
        let pitch_bytes = PITCH as usize * BYTES_PER_PIXEL;
        for row in 0..HEIGHT as usize {
            assert_eq!(
                &raw[row * pitch_bytes..row * pitch_bytes + row_bytes],
                vec![0x11; row_bytes]
            );
            assert_eq!(
                &raw[row * pitch_bytes + row_bytes..(row + 1) * pitch_bytes],
                vec![0xa5; pitch_bytes - row_bytes]
            );
        }
    }

    #[test]
    fn async_texture_publication_preserves_reused_guest_bytes() {
        const WIDTH: u32 = 2;
        const HEIGHT: u32 = 2;
        const PITCH: u32 = 4;
        const BYTES_PER_PIXEL: usize = 4;
        let baseline = vec![0_u8; (PITCH * HEIGHT) as usize * BYTES_PER_PIXEL];
        let mut guest = baseline.clone();
        let descriptor = super::DecodedTexture {
            address: guest.as_mut_ptr() as u64,
            width: WIDTH,
            height: HEIGHT,
            pitch: PITCH,
            tile_mode: 0,
            format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
            bytes_per_element: BYTES_PER_PIXEL as u32,
            block_extent: [1, 1],
            texture_type: 9,
            layer_count: 1,
            mip_level_count: 1,
            writable: true,
        };
        let linear = vec![0x11_u8; (WIDTH * HEIGHT) as usize * BYTES_PER_PIXEL];
        let candidate = super::encode_texture_linear_over(descriptor, &linear, &baseline)
            .expect("encode a texture against its upload-time baseline");

        // The guest reused part of the first texel run and one byte of pitch padding while the
        // GPU submission was in flight. The first run is withheld, the independent second row is
        // published, and padding is never considered GPU-owned.
        guest[4..8].fill(0x99);
        guest[10] = 0xa5;
        write_changed_ranges_direct(
            descriptor.address,
            &candidate,
            &baseline,
            WritebackJournalSites::TEXTURE.direct,
            "texture",
            16,
        )
        .expect("publish baseline-guarded D3D12 texture bytes");

        assert_eq!(&guest[0..4], &[0; 4]);
        assert_eq!(&guest[4..8], &[0x99; 4]);
        assert_eq!(guest[10], 0xa5);
        assert_eq!(&guest[16..24], &[0x11; 8]);
    }

    fn literal(value: u32) -> Operand {
        Operand {
            kind: OperandKind::LiteralConstant,
            value,
        }
    }

    fn empty_evaluation_for_graphics() -> Arc<ShaderEvaluation> {
        Arc::new(ShaderEvaluation {
            initial_scalar_registers: Vec::new().into(),
            scalar_registers: Vec::new().into(),
            scalar_memory_reads: Vec::new(),
            image_bindings: Vec::new(),
            global_memory_bindings: Vec::new(),
            compute_system_registers: None,
            vertex_inputs: Vec::new(),
        })
    }

    #[test]
    fn expands_legacy_gfx10_primitives_without_crossing_restart_boundaries() {
        let restart = 99;
        assert_eq!(
            expand_primitive_indices(5, &[10, 11, 12, 13, restart, 20, 21, 22], Some(restart))
                .expect("expand triangle fans"),
            [10, 11, 12, 10, 12, 13, 20, 21, 22]
        );
        assert_eq!(
            expand_primitive_indices(0x12, &[0, 1, 2, restart, 8, 9], Some(restart))
                .expect("expand line loops"),
            [0, 1, 1, 2, 2, 0, 8, 9, 9, 8]
        );
        assert_eq!(
            expand_primitive_indices(0x13, &[0, 1, 2, 3, 4, 5, 6, 7], None)
                .expect("expand quad list"),
            [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]
        );
        assert_eq!(
            expand_primitive_indices(0x14, &[0, 1, 2, 3, 4, 5], None).expect("expand quad strip"),
            [0, 1, 3, 0, 3, 2, 2, 3, 5, 2, 5, 4]
        );
        assert_eq!(
            expand_primitive_indices(0x15, &[4, 5, 6, 7], None).expect("expand polygon"),
            [4, 5, 6, 4, 6, 7]
        );
    }

    #[test]
    fn restart_enabled_list_drops_incomplete_primitives_at_each_boundary() {
        assert_eq!(
            regroup_restart_list_indices(4, &[0, 1, 2, 3, 4, 99, 5, 6, 7], 99)
                .expect("regroup triangle lists"),
            [0, 1, 2, 5, 6, 7]
        );
    }

    #[test]
    fn restart_register_uses_the_programmed_guest_index_width() {
        let mut draw = IndexedDraw {
            sequence: 0,
            indexed: true,
            indirect_arguments_address: None,
            indirect_count_address: None,
            indirect_draw_index: 0,
            instance_count: 1,
            index_buffer_address: 0,
            index_resource_address: 0,
            index_buffer_count: 0,
            index_size: 0,
            index_offset: 0,
            base_vertex: 0,
            index_count: 0,
            index_snapshot: None,
            draw_flags: 0,
            primitive_type: 6,
            export_shader_address: 0,
            pixel_shader_address: 0,
            export_shader_prefix: [0; 4],
            pixel_shader_prefix: [0; 4],
            export_shader: None,
            pixel_shader: None,
            export_shader_evaluation: None,
            pixel_shader_evaluation: None,
            color_target_address: 0,
            scanout_target: None,
            context_registers: vec![
                remu_gpu::GpuRegister {
                    offset: 0x103,
                    value: 0x1234_abcd,
                },
                remu_gpu::GpuRegister {
                    offset: 0x2a5,
                    value: 1,
                },
            ],
            shader_registers: Vec::new(),
            user_config_registers: Vec::new(),
        };
        assert_eq!(primitive_restart_index(&draw), Some(0xabcd));
        draw.index_size = 1;
        assert_eq!(primitive_restart_index(&draw), Some(0x1234_abcd));
        draw.index_size = 2;
        assert_eq!(primitive_restart_index(&draw), Some(0xcd));

        let indexed_arguments = [6_u32, 7, 11, (-3_i32) as u32, 13];
        let (resolved, arguments) = resolve_indirect_draw(&draw, indexed_arguments.as_ptr() as u64)
            .expect("resolve live indexed arguments");
        assert_eq!(resolved.index_count, 6);
        assert_eq!(resolved.instance_count, 7);
        assert_eq!(resolved.index_offset, 11);
        assert_eq!(resolved.base_vertex, -3);
        assert_eq!(arguments.first_instance, 13);

        let mut live_count = 1_u32;
        draw.indirect_count_address = Some((&raw mut live_count).addr() as u64);
        draw.indirect_draw_index = 0;
        assert!(indirect_draw_count_allows(&draw).expect("read live multi-draw count"));
        draw.indirect_draw_index = 1;
        assert!(!indirect_draw_count_allows(&draw).expect("skip draw outside live count"));
        live_count = 2;
        std::hint::black_box(live_count);
        assert!(indirect_draw_count_allows(&draw).expect("observe updated multi-draw count"));

        draw.indexed = false;
        let nonindexed_arguments = [17_u32, 19, 23, 29];
        let (resolved, arguments) =
            resolve_indirect_draw(&draw, nonindexed_arguments.as_ptr() as u64)
                .expect("resolve live non-indexed arguments");
        assert_eq!(resolved.index_count, 17);
        assert_eq!(resolved.instance_count, 19);
        assert_eq!(resolved.index_offset, 23);
        assert_eq!(resolved.base_vertex, 0);
        assert_eq!(arguments.first_instance, 29);
    }

    #[test]
    fn executes_translated_guest_compute_and_writes_back_guest_memory() {
        let mut guest = vec![0_u32; 64];
        let base = guest.as_mut_ptr() as u64;
        let program = Arc::new(Program::new(
            0x9000,
            vec![
                Instruction {
                    pc: 0,
                    encoding: ShaderEncoding::Vop2,
                    opcode: Opcode::Known("VLshlrevB32"),
                    words: vec![0],
                    sources: vec![literal(2), Operand::vector(0)],
                    destinations: vec![Operand::vector(2)],
                    control: None,
                },
                Instruction {
                    pc: 4,
                    encoding: ShaderEncoding::Vop2,
                    opcode: Opcode::Known("VAddI32"),
                    words: vec![0],
                    sources: vec![Operand::vector(0), literal(0x5678_0000)],
                    destinations: vec![Operand::vector(1)],
                    control: None,
                },
                Instruction {
                    pc: 8,
                    encoding: ShaderEncoding::Flat,
                    opcode: Opcode::Known("GlobalStoreDword"),
                    words: vec![0, 0],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: Some(InstructionControl::GlobalMemory(GlobalMemoryControl {
                        dword_count: 1,
                        vector_address: 2,
                        vector_data: 1,
                        vector_destination: 0,
                        scalar_address: Some(0),
                        offset_bytes: 0,
                        dlc: false,
                        lds: false,
                        glc: false,
                        slc: false,
                    })),
                },
                Instruction {
                    pc: 16,
                    encoding: ShaderEncoding::Sopp,
                    opcode: Opcode::END_PROGRAM,
                    words: vec![0xbf81_0000],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: None,
                },
            ],
        ));
        let evaluation = Arc::new(ShaderEvaluation {
            initial_scalar_registers: [base as u32, (base >> 32) as u32].into_iter().collect(),
            scalar_registers: Vec::new().into(),
            scalar_memory_reads: Vec::new(),
            image_bindings: Vec::new(),
            global_memory_bindings: vec![GlobalMemoryBinding {
                scalar_address: 0,
                base_address: base,
                instruction_pcs: [8].into_iter().collect(),
                byte_count: guest.len() * 4,
                data: Vec::new(),
                writable: true,
                write_back_to_guest: true,
            }],
            compute_system_registers: None,
            vertex_inputs: Vec::new(),
        });
        let dispatch = ComputeDispatch {
            sequence: 1,
            source: ComputeDispatchSource::Direct,
            launch: ComputeLaunch {
                grid: ComputeGrid::Threadgroups([1, 1, 1]),
                group_count: [1, 1, 1],
                base_group: [0; 3],
                threads_per_group: [64, 1, 1],
                wave_lane_count: 32,
                lds_size_bytes: 0,
                initiator: 1,
            },
            shader_address: 0x9000,
            shader_prefix: [0; 4],
            shader: Some(ShaderSnapshot {
                header_address: 0x9000,
                program,
                supplemental_programs: Arc::from([]),
                metadata: None,
            }),
            shader_evaluation: Some(evaluation),
            shader_registers: Vec::new(),
        };
        Dx12GuestExecutor::new()
            .expect("create D3D12 guest executor")
            .execute_compute(&dispatch)
            .expect("execute guest compute dispatch");
        for (index, value) in guest.into_iter().enumerate() {
            assert_eq!(value, 0x5678_0000 + index as u32);
        }
    }

    #[test]
    fn executes_formatted_indirect_argument_expansion() {
        let input = [0_u32, 0, 0, 0x2428, 0];
        let mut output = vec![0_u32; 16_383];
        let input_base = input.as_ptr() as u64;
        let output_base = output.as_mut_ptr() as u64;
        let buffer = |pc, opcode, vector_address, vector_data, scalar_resource| Instruction {
            pc,
            encoding: ShaderEncoding::Mubuf,
            opcode: Opcode::Known(opcode),
            words: vec![0, 0],
            sources: vec![
                Operand::vector(vector_address),
                Operand::scalar(scalar_resource),
                Operand::source(128, None),
            ],
            destinations: Vec::new(),
            control: Some(InstructionControl::BufferMemory(BufferMemoryControl {
                dword_count: 1,
                vector_address,
                vector_data,
                scalar_resource,
                instruction_format: None,
                offset_bytes: 0,
                index_enabled: true,
                offset_enabled: false,
                dlc: false,
                lds: false,
                glc: false,
                slc: false,
                tfe: false,
            })),
        };
        let program = Arc::new(Program::new(
            0x9800,
            vec![
                buffer(0, "BufferLoadDword", 0, 1, 0),
                Instruction {
                    pc: 8,
                    encoding: ShaderEncoding::Vop2,
                    opcode: Opcode::Known("VMulU32U24"),
                    words: vec![0],
                    sources: vec![literal(3), Operand::vector(0)],
                    destinations: vec![Operand::vector(2)],
                    control: None,
                },
                Instruction {
                    pc: 12,
                    encoding: ShaderEncoding::Vop2,
                    opcode: Opcode::Known("VAddI32"),
                    words: vec![0],
                    sources: vec![Operand::vector(2), literal(1)],
                    destinations: vec![Operand::vector(3)],
                    control: None,
                },
                Instruction {
                    pc: 16,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VMovB32"),
                    words: vec![0],
                    sources: vec![literal(1)],
                    destinations: vec![Operand::vector(4)],
                    control: None,
                },
                Instruction {
                    pc: 20,
                    encoding: ShaderEncoding::Vop2,
                    opcode: Opcode::Known("VAddI32"),
                    words: vec![0],
                    sources: vec![Operand::vector(2), literal(2)],
                    destinations: vec![Operand::vector(0)],
                    control: None,
                },
                buffer(24, "BufferStoreFormatX", 2, 1, 4),
                buffer(32, "BufferStoreFormatX", 3, 4, 4),
                buffer(40, "BufferStoreFormatX", 0, 4, 4),
                Instruction {
                    pc: 48,
                    encoding: ShaderEncoding::Sopp,
                    opcode: Opcode::END_PROGRAM,
                    words: vec![0xbf81_0000],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: None,
                },
            ],
        ));
        let mut initial = vec![0_u32; 8];
        initial[0] = input_base as u32;
        initial[1] = ((input_base >> 32) as u32 & 0xffff) | (4 << 16);
        initial[2] = input.len() as u32;
        initial[4] = output_base as u32;
        initial[5] = ((output_base >> 32) as u32 & 0xffff) | (4 << 16);
        initial[6] = output.len() as u32;
        // BUF_DATA_FORMAT_32 + BUF_NUM_FORMAT_UINT, with component X selecting VGPR data 0.
        initial[7] = 4 | (20 << 12);
        let evaluation = Arc::new(ShaderEvaluation {
            initial_scalar_registers: initial.into_iter().collect(),
            scalar_registers: Vec::new().into(),
            scalar_memory_reads: Vec::new(),
            image_bindings: Vec::new(),
            global_memory_bindings: vec![
                GlobalMemoryBinding {
                    scalar_address: 0,
                    base_address: input_base,
                    instruction_pcs: [0].into_iter().collect(),
                    byte_count: input.len() * 4,
                    data: Vec::new(),
                    writable: false,
                    write_back_to_guest: false,
                },
                GlobalMemoryBinding {
                    scalar_address: 4,
                    base_address: output_base,
                    instruction_pcs: [24, 32, 40].into_iter().collect(),
                    byte_count: output.len() * 4,
                    data: Vec::new(),
                    writable: true,
                    write_back_to_guest: true,
                },
            ],
            compute_system_registers: None,
            vertex_inputs: Vec::new(),
        });
        let dispatch = ComputeDispatch {
            sequence: 2,
            source: ComputeDispatchSource::Direct,
            launch: ComputeLaunch {
                grid: ComputeGrid::Threadgroups([1, 1, 1]),
                group_count: [1, 1, 1],
                base_group: [0; 3],
                threads_per_group: [5, 1, 1],
                wave_lane_count: 32,
                lds_size_bytes: 0,
                initiator: 1,
            },
            shader_address: 0x9800,
            shader_prefix: [0; 4],
            shader: Some(ShaderSnapshot {
                header_address: 0x9800,
                program,
                supplemental_programs: Arc::from([]),
                metadata: None,
            }),
            shader_evaluation: Some(evaluation),
            shader_registers: Vec::new(),
        };
        Dx12GuestExecutor::new()
            .expect("create D3D12 guest executor")
            .execute_compute(&dispatch)
            .expect("execute formatted argument expansion");
        assert_eq!(
            &output[..15],
            &[0, 1, 1, 0, 1, 1, 0, 1, 1, 0x2428, 1, 1, 0, 1, 1]
        );
    }

    #[test]
    fn indirect_compute_uses_live_ordered_dimensions() {
        let dimensions = [7_u32, 5, 3];
        let dispatch = ComputeDispatch {
            sequence: 1,
            source: ComputeDispatchSource::Indirect {
                dimensions_address: dimensions.as_ptr() as u64,
            },
            launch: ComputeLaunch {
                grid: ComputeGrid::Threadgroups([1, 1, 1]),
                group_count: [1, 1, 1],
                base_group: [0; 3],
                threads_per_group: [8, 4, 2],
                wave_lane_count: 32,
                lds_size_bytes: 0,
                initiator: 1 | (1 << 2) | (1 << 15),
            },
            shader_address: 0,
            shader_prefix: [0; 4],
            shader: None,
            shader_evaluation: None,
            shader_registers: vec![
                remu_gpu::GpuRegister {
                    offset: 0x207,
                    value: 8,
                },
                remu_gpu::GpuRegister {
                    offset: 0x208,
                    value: 4,
                },
                remu_gpu::GpuRegister {
                    offset: 0x209,
                    value: 2,
                },
            ],
        };
        let resolved = resolve_indirect_compute(&dispatch)
            .expect("read live dimensions")
            .expect("indirect dispatch resolves");
        assert_eq!(resolved.launch.grid, ComputeGrid::Threadgroups(dimensions));
        assert_eq!(resolved.launch.group_count, dimensions);
        assert_eq!(resolved.launch.threads_per_group, [8, 4, 2]);
        assert_eq!(resolved.launch.wave_lane_count, 32);
    }

    #[test]
    fn ordered_global_versions_forward_gpu_output_over_predating_packet_bytes() {
        let mut live = [0_u32, 0x1800_3fff, 0, 0x1800_3fff, 0];
        let address = live.as_mut_ptr() as u64;
        let packet = vec![0_u8; live.len() * 4];
        let produced_words = [0_u32, 0, 0, 0x2428, 0];
        let produced = super::words_as_bytes(&produced_words);
        let binding = GlobalMemoryBinding {
            scalar_address: 0,
            base_address: address,
            instruction_pcs: Default::default(),
            byte_count: packet.len(),
            data: packet.clone(),
            writable: false,
            write_back_to_guest: false,
        };
        let mut versions = OrderedGlobalVersions::default();
        versions.record(address, None, &packet, &produced);

        assert_eq!(
            snapshot_ordered_global(&binding, &versions).expect("forward ordered producer"),
            produced
        );

        versions.invalidate(address + 4, 4);
        assert_eq!(
            snapshot_ordered_global(&binding, &versions).expect("observe invalidated packet bytes"),
            packet
        );

        versions.record(address, None, &packet, &produced);

        let mut future = binding;
        future.data = super::words_as_bytes(&live);
        assert_eq!(
            snapshot_ordered_global(&future, &versions).expect("keep newer packet bytes"),
            future.data
        );
    }

    #[test]
    fn indirect_compute_forwards_ordered_gpu_arguments() {
        let mut live = [0x1800_3fff_u32, 1, 1];
        let address = live.as_mut_ptr() as u64;
        let packet = super::words_as_bytes(&live);
        let produced = super::words_as_bytes(&[9_u32, 1, 1]);
        let mut versions = OrderedGlobalVersions::default();
        versions.record(address, None, &packet, &produced);
        let dispatch = ComputeDispatch {
            sequence: 1,
            source: ComputeDispatchSource::Indirect {
                dimensions_address: address,
            },
            launch: ComputeLaunch {
                grid: ComputeGrid::Threadgroups(live),
                group_count: live,
                base_group: [0; 3],
                threads_per_group: [8, 4, 2],
                wave_lane_count: 32,
                lds_size_bytes: 0,
                initiator: 1 | (1 << 2) | (1 << 15),
            },
            shader_address: 0,
            shader_prefix: [0; 4],
            shader: None,
            shader_evaluation: None,
            shader_registers: vec![
                GpuRegister {
                    offset: 0x207,
                    value: 8,
                },
                GpuRegister {
                    offset: 0x208,
                    value: 4,
                },
                GpuRegister {
                    offset: 0x209,
                    value: 2,
                },
            ],
        };

        let resolved = resolve_indirect_compute_ordered(&dispatch, &versions)
            .expect("resolve ordered dimensions")
            .expect("indirect dispatch resolves");
        assert_eq!(resolved.launch.group_count, [9, 1, 1]);
    }

    #[test]
    fn executes_guest_storage_image_store_and_retiles_guest_memory() {
        const WIDTH: usize = 64;
        let mut allocation = vec![0_u8; WIDTH * 4 + 255];
        let allocation_base = allocation.as_mut_ptr() as usize;
        let aligned = (allocation_base + 255) & !255;
        let address = aligned as u64;
        let unified = (0..0x200)
            .find(|value| remu_shader::decode_unified_format(*value) == Some((10, 4)))
            .expect("RGBA8_UINT unified format exists");
        let width_minus_one = WIDTH as u32 - 1;
        let mut descriptor = vec![0_u32; 8];
        descriptor[0] = (address >> 8) as u32;
        descriptor[1] =
            (((address >> 40) as u32) & 0xff) | (unified << 20) | ((width_minus_one & 3) << 30);
        descriptor[2] = (width_minus_one >> 2) | (0 << 14);
        descriptor[3] = 0x0fac | (9 << 28);
        descriptor[4] = WIDTH as u32 - 1;
        let control = ImageControl {
            dmask: 1,
            vector_address: 0,
            address_registers: vec![0, 1],
            vector_data: 4,
            scalar_resource: 0,
            scalar_sampler: 0,
            dimension: 1,
            is_array: false,
            glc: false,
            slc: false,
            a16: false,
            d16: false,
            unorm: false,
        };
        let program = Arc::new(Program::new(
            0xa000,
            vec![
                Instruction {
                    pc: 0,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VMovB32"),
                    words: vec![0],
                    sources: vec![literal(0)],
                    destinations: vec![Operand::vector(1)],
                    control: None,
                },
                Instruction {
                    pc: 4,
                    encoding: ShaderEncoding::Vop2,
                    opcode: Opcode::Known("VAddI32"),
                    words: vec![0],
                    sources: vec![Operand::vector(0), literal(100)],
                    destinations: vec![Operand::vector(4)],
                    control: None,
                },
                Instruction {
                    pc: 8,
                    encoding: ShaderEncoding::Mimg,
                    opcode: Opcode::Known("ImageStore"),
                    words: vec![0, 0],
                    sources: vec![Operand::vector(0), Operand::vector(1), Operand::vector(4)],
                    destinations: Vec::new(),
                    control: Some(InstructionControl::Image(control.clone())),
                },
                Instruction {
                    pc: 16,
                    encoding: ShaderEncoding::Sopp,
                    opcode: Opcode::END_PROGRAM,
                    words: vec![0xbf81_0000],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: None,
                },
            ],
        ));
        let evaluation = Arc::new(ShaderEvaluation {
            initial_scalar_registers: descriptor.iter().copied().collect(),
            scalar_registers: Vec::new().into(),
            scalar_memory_reads: Vec::new(),
            image_bindings: vec![ImageBinding {
                pc: 8,
                opcode: Opcode::Known("ImageStore"),
                control,
                resource_descriptor: descriptor.into(),
                sampler_descriptor: Vec::new().into(),
                mip_level: None,
            }],
            global_memory_bindings: Vec::new(),
            compute_system_registers: None,
            vertex_inputs: Vec::new(),
        });
        let dispatch = ComputeDispatch {
            sequence: 2,
            source: ComputeDispatchSource::Direct,
            launch: ComputeLaunch {
                grid: ComputeGrid::Threadgroups([1, 1, 1]),
                group_count: [1, 1, 1],
                base_group: [0; 3],
                threads_per_group: [64, 1, 1],
                wave_lane_count: 32,
                lds_size_bytes: 0,
                initiator: 1,
            },
            shader_address: 0xa000,
            shader_prefix: [0; 4],
            shader: Some(ShaderSnapshot {
                header_address: 0xa000,
                program,
                supplemental_programs: Arc::from([]),
                metadata: None,
            }),
            shader_evaluation: Some(evaluation),
            shader_registers: Vec::new(),
        };
        Dx12GuestExecutor::new()
            .expect("create D3D12 guest executor")
            .execute_compute(&dispatch)
            .expect("execute guest storage-image dispatch");
        // SAFETY: aligned points into `allocation` and the tested row is WIDTH*4 bytes.
        let pixels = unsafe { std::slice::from_raw_parts(aligned as *const u8, WIDTH * 4) };
        for (index, pixel) in pixels.chunks_exact(4).enumerate() {
            assert_eq!(pixel, &[(100 + index) as u8, 0, 0, 0]);
        }
    }

    #[test]
    fn samples_guest_texture_through_a_native_sampler() {
        const WIDTH: usize = 64;
        let mut texture_allocation = vec![0_u8; 64 * 1024 + 255];
        let texture_address = ((texture_allocation.as_mut_ptr() as usize + 255) & !255) as u64;
        let unified = (0..0x200)
            .find(|value| remu_shader::decode_unified_format(*value) == Some((10, 0)))
            .expect("RGBA8_UNORM unified format exists");
        let width_minus_one = WIDTH as u32 - 1;
        let mut descriptor = vec![0_u32; 8];
        descriptor[0] = (texture_address >> 8) as u32;
        descriptor[1] = (((texture_address >> 40) as u32) & 0xff)
            | (unified << 20)
            | ((width_minus_one & 3) << 30);
        descriptor[2] = width_minus_one >> 2;
        descriptor[3] = 0x0fac | (1 << 16) | (27 << 20) | (9 << 28);
        descriptor[4] = WIDTH as u32 - 1;
        let texture_descriptor = super::DecodedTexture {
            address: texture_address,
            width: WIDTH as u32,
            height: 1,
            pitch: WIDTH as u32,
            tile_mode: 27,
            format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
            bytes_per_element: 4,
            block_extent: [1, 1],
            texture_type: 9,
            layer_count: 1,
            mip_level_count: 2,
            writable: false,
        };
        super::write_texture_mip_chain(
            texture_descriptor,
            &[
                vec![200, 0, 0, 255].repeat(WIDTH),
                vec![100, 0, 0, 255].repeat(WIDTH / 2),
            ],
        )
        .expect("encode two-level guest mip chain");

        let mut output = vec![0_u32; WIDTH];
        let output_base = output.as_mut_ptr() as u64;
        let control = ImageControl {
            dmask: 1,
            vector_address: 0,
            address_registers: vec![0, 1, 2],
            vector_data: 4,
            scalar_resource: 0,
            scalar_sampler: 8,
            dimension: 1,
            is_array: false,
            glc: false,
            slc: false,
            a16: false,
            d16: false,
            unorm: false,
        };
        let program = Arc::new(Program::new(
            0xb000,
            vec![
                Instruction {
                    pc: 0,
                    encoding: ShaderEncoding::Vop2,
                    opcode: Opcode::Known("VLshlrevB32"),
                    words: vec![0],
                    sources: vec![literal(2), Operand::vector(0)],
                    destinations: vec![Operand::vector(3)],
                    control: None,
                },
                Instruction {
                    pc: 4,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VCvtF32U32"),
                    words: vec![0],
                    sources: vec![Operand::vector(0)],
                    destinations: vec![Operand::vector(2)],
                    control: None,
                },
                Instruction {
                    pc: 8,
                    encoding: ShaderEncoding::Vop2,
                    opcode: Opcode::Known("VAddF32"),
                    words: vec![0],
                    sources: vec![Operand::vector(2), literal(0.5_f32.to_bits())],
                    destinations: vec![Operand::vector(2)],
                    control: None,
                },
                Instruction {
                    pc: 12,
                    encoding: ShaderEncoding::Vop2,
                    opcode: Opcode::Known("VMulF32"),
                    words: vec![0],
                    sources: vec![
                        Operand::vector(2),
                        literal((1.0_f32 / WIDTH as f32).to_bits()),
                    ],
                    destinations: vec![Operand::vector(0)],
                    control: None,
                },
                Instruction {
                    pc: 16,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VMovB32"),
                    words: vec![0],
                    sources: vec![literal(0.5_f32.to_bits())],
                    destinations: vec![Operand::vector(1)],
                    control: None,
                },
                Instruction {
                    pc: 20,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VMovB32"),
                    words: vec![0],
                    sources: vec![literal(1.0_f32.to_bits())],
                    destinations: vec![Operand::vector(2)],
                    control: None,
                },
                Instruction {
                    pc: 24,
                    encoding: ShaderEncoding::Mimg,
                    opcode: Opcode::Known("ImageSampleL"),
                    words: vec![0, 0],
                    sources: vec![
                        Operand::vector(0),
                        Operand::vector(1),
                        Operand::vector(2),
                        Operand::scalar(0),
                    ],
                    destinations: vec![Operand::vector(4)],
                    control: Some(InstructionControl::Image(control.clone())),
                },
                Instruction {
                    pc: 32,
                    encoding: ShaderEncoding::Flat,
                    opcode: Opcode::Known("GlobalStoreDword"),
                    words: vec![0, 0],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: Some(InstructionControl::GlobalMemory(GlobalMemoryControl {
                        dword_count: 1,
                        vector_address: 3,
                        vector_data: 4,
                        vector_destination: 0,
                        scalar_address: Some(12),
                        offset_bytes: 0,
                        dlc: false,
                        lds: false,
                        glc: false,
                        slc: false,
                    })),
                },
                Instruction {
                    pc: 40,
                    encoding: ShaderEncoding::Sopp,
                    opcode: Opcode::END_PROGRAM,
                    words: vec![0xbf81_0000],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: None,
                },
            ],
        ));
        let mut initial = vec![0_u32; 14];
        initial[..8].copy_from_slice(&descriptor);
        initial[12] = output_base as u32;
        initial[13] = (output_base >> 32) as u32;
        let evaluation = Arc::new(ShaderEvaluation {
            initial_scalar_registers: initial.into_iter().collect(),
            scalar_registers: Vec::new().into(),
            scalar_memory_reads: Vec::new(),
            image_bindings: vec![ImageBinding {
                pc: 24,
                opcode: Opcode::Known("ImageSampleL"),
                control,
                resource_descriptor: descriptor.into(),
                sampler_descriptor: vec![0, 0x00ff_f000, 0, 0].into(),
                mip_level: None,
            }],
            global_memory_bindings: vec![GlobalMemoryBinding {
                scalar_address: 12,
                base_address: output_base,
                instruction_pcs: [32].into_iter().collect(),
                byte_count: WIDTH * 4,
                data: Vec::new(),
                writable: true,
                write_back_to_guest: true,
            }],
            compute_system_registers: None,
            vertex_inputs: Vec::new(),
        });
        let dispatch = ComputeDispatch {
            sequence: 3,
            source: ComputeDispatchSource::Direct,
            launch: ComputeLaunch {
                grid: ComputeGrid::Threadgroups([1, 1, 1]),
                group_count: [1, 1, 1],
                base_group: [0; 3],
                threads_per_group: [64, 1, 1],
                wave_lane_count: 32,
                lds_size_bytes: 0,
                initiator: 1,
            },
            shader_address: 0xb000,
            shader_prefix: [0; 4],
            shader: Some(ShaderSnapshot {
                header_address: 0xb000,
                program,
                supplemental_programs: Arc::from([]),
                metadata: None,
            }),
            shader_evaluation: Some(evaluation),
            shader_registers: Vec::new(),
        };
        Dx12GuestExecutor::new()
            .expect("create D3D12 guest executor")
            .execute_compute(&dispatch)
            .expect("execute guest sampled-image dispatch");
        let expected = 100.0_f32 / 255.0;
        for (index, value) in output.into_iter().enumerate() {
            assert!(
                (f32::from_bits(value) - expected).abs() < 0.001,
                "sample {index} returned {:#010x} ({}) instead of {expected}",
                value,
                f32::from_bits(value),
            );
        }
    }

    #[test]
    fn executes_guest_vertex_and_pixel_shaders_into_guest_scanout_memory() {
        const WIDTH: u32 = 64;
        const HEIGHT: u32 = 64;
        let mut allocation = vec![0_u8; (WIDTH * HEIGHT * 4) as usize + 255];
        let scanout_address = ((allocation.as_mut_ptr() as usize + 255) & !255_usize) as u64;

        let vertex = Arc::new(Program::new(
            0xc000,
            vec![
                Instruction {
                    pc: 0,
                    encoding: ShaderEncoding::Vop2,
                    opcode: Opcode::Known("VLshlrevB32"),
                    words: vec![0],
                    sources: vec![literal(1), Operand::vector(5)],
                    destinations: vec![Operand::vector(0)],
                    control: None,
                },
                Instruction {
                    pc: 4,
                    encoding: ShaderEncoding::Vop2,
                    opcode: Opcode::Known("VAndB32"),
                    words: vec![0],
                    sources: vec![Operand::vector(0), literal(2)],
                    destinations: vec![Operand::vector(0)],
                    control: None,
                },
                Instruction {
                    pc: 8,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VCvtF32U32"),
                    words: vec![0],
                    sources: vec![Operand::vector(0)],
                    destinations: vec![Operand::vector(0)],
                    control: None,
                },
                Instruction {
                    pc: 12,
                    encoding: ShaderEncoding::Vop2,
                    opcode: Opcode::Known("VMulF32"),
                    words: vec![0],
                    sources: vec![Operand::vector(0), literal(2.0_f32.to_bits())],
                    destinations: vec![Operand::vector(0)],
                    control: None,
                },
                Instruction {
                    pc: 16,
                    encoding: ShaderEncoding::Vop2,
                    opcode: Opcode::Known("VAddF32"),
                    words: vec![0],
                    sources: vec![Operand::vector(0), literal((-1.0_f32).to_bits())],
                    destinations: vec![Operand::vector(0)],
                    control: None,
                },
                Instruction {
                    pc: 20,
                    encoding: ShaderEncoding::Vop2,
                    opcode: Opcode::Known("VAndB32"),
                    words: vec![0],
                    sources: vec![Operand::vector(5), literal(2)],
                    destinations: vec![Operand::vector(1)],
                    control: None,
                },
                Instruction {
                    pc: 24,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VCvtF32U32"),
                    words: vec![0],
                    sources: vec![Operand::vector(1)],
                    destinations: vec![Operand::vector(1)],
                    control: None,
                },
                Instruction {
                    pc: 28,
                    encoding: ShaderEncoding::Vop2,
                    opcode: Opcode::Known("VMulF32"),
                    words: vec![0],
                    sources: vec![Operand::vector(1), literal((-2.0_f32).to_bits())],
                    destinations: vec![Operand::vector(1)],
                    control: None,
                },
                Instruction {
                    pc: 32,
                    encoding: ShaderEncoding::Vop2,
                    opcode: Opcode::Known("VAddF32"),
                    words: vec![0],
                    sources: vec![Operand::vector(1), literal(1.0_f32.to_bits())],
                    destinations: vec![Operand::vector(1)],
                    control: None,
                },
                Instruction {
                    pc: 36,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VMovB32"),
                    words: vec![0],
                    sources: vec![literal(0)],
                    destinations: vec![Operand::vector(2)],
                    control: None,
                },
                Instruction {
                    pc: 40,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VMovB32"),
                    words: vec![0],
                    sources: vec![literal(1.0_f32.to_bits())],
                    destinations: vec![Operand::vector(3)],
                    control: None,
                },
                Instruction {
                    pc: 44,
                    encoding: ShaderEncoding::Exp,
                    opcode: Opcode::EXPORT,
                    words: vec![0, 0],
                    sources: (0..4).map(Operand::vector).collect(),
                    destinations: Vec::new(),
                    control: Some(InstructionControl::Export(ExportControl {
                        target: 12,
                        enable_mask: 0xf,
                        compressed: false,
                        done: true,
                        valid_mask: true,
                    })),
                },
                Instruction {
                    pc: 52,
                    encoding: ShaderEncoding::Sopp,
                    opcode: Opcode::END_PROGRAM,
                    words: vec![0xbf81_0000],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: None,
                },
            ],
        ));
        let pixel = Arc::new(Program::new(
            0xd000,
            vec![
                Instruction {
                    pc: 0,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VMovB32"),
                    words: vec![0],
                    sources: vec![literal(1.0_f32.to_bits())],
                    destinations: vec![Operand::vector(0)],
                    control: None,
                },
                Instruction {
                    pc: 4,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VMovB32"),
                    words: vec![0],
                    sources: vec![literal(0)],
                    destinations: vec![Operand::vector(1)],
                    control: None,
                },
                Instruction {
                    pc: 8,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VMovB32"),
                    words: vec![0],
                    sources: vec![literal(0)],
                    destinations: vec![Operand::vector(2)],
                    control: None,
                },
                Instruction {
                    pc: 12,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VMovB32"),
                    words: vec![0],
                    sources: vec![literal(1.0_f32.to_bits())],
                    destinations: vec![Operand::vector(3)],
                    control: None,
                },
                Instruction {
                    pc: 16,
                    encoding: ShaderEncoding::Exp,
                    opcode: Opcode::EXPORT,
                    words: vec![0, 0],
                    sources: (0..4).map(Operand::vector).collect(),
                    destinations: Vec::new(),
                    control: Some(InstructionControl::Export(ExportControl {
                        target: 0,
                        enable_mask: 0xf,
                        compressed: false,
                        done: true,
                        valid_mask: true,
                    })),
                },
                Instruction {
                    pc: 24,
                    encoding: ShaderEncoding::Sopp,
                    opcode: Opcode::END_PROGRAM,
                    words: vec![0xbf81_0000],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: None,
                },
            ],
        ));
        let evaluation = empty_evaluation_for_graphics();
        let mut draw = IndexedDraw {
            sequence: 4,
            indexed: false,
            indirect_arguments_address: None,
            indirect_count_address: None,
            indirect_draw_index: 0,
            instance_count: 1,
            index_buffer_address: 0,
            index_resource_address: 0,
            index_buffer_count: 0,
            index_size: 1,
            index_offset: 0,
            base_vertex: 0,
            index_count: 3,
            index_snapshot: None,
            draw_flags: 0,
            primitive_type: 4,
            export_shader_address: 0xc000,
            pixel_shader_address: 0xd000,
            export_shader_prefix: [0; 4],
            pixel_shader_prefix: [0; 4],
            export_shader: Some(ShaderSnapshot {
                header_address: 0xc000,
                program: vertex,
                supplemental_programs: Arc::from([]),
                metadata: None,
            }),
            pixel_shader: Some(ShaderSnapshot {
                header_address: 0xd000,
                program: pixel,
                supplemental_programs: Arc::from([]),
                metadata: None,
            }),
            export_shader_evaluation: Some(evaluation.clone()),
            pixel_shader_evaluation: Some(evaluation),
            color_target_address: scanout_address,
            scanout_target: Some(GpuScanoutTarget {
                address: scanout_address,
                width: WIDTH,
                height: HEIGHT,
                pitch: WIDTH,
                pixel_format: 0,
                tiling_mode: 1,
                export_slot: 0,
                write_mask: 0xf,
                color_surface: None,
            }),
            context_registers: Vec::new(),
            shader_registers: Vec::new(),
            user_config_registers: Vec::new(),
        };

        let mut executor = Dx12GuestExecutor::new().expect("create D3D12 guest executor");
        if executor.triangle_fan_supported {
            draw.primitive_type = 5;
        }
        executor
            .execute_draw(&draw)
            .expect("execute guest graphics draw");
        draw.context_registers.push(GpuRegister {
            offset: 0x1e0,
            value: 0x6400_0400,
        });
        executor
            .execute_draw(&draw)
            .expect("execute destination-source-alpha blend graphics draw");
        draw.context_registers.clear();
        executor
            .execute_draw(&draw)
            .expect("restore unblended graphics draw after blend PSO validation");

        let assert_red_surface = |kind: &str| {
            // SAFETY: the aligned scanout covers WIDTH*HEIGHT tightly packed BGRA pixels.
            let pixels = unsafe {
                std::slice::from_raw_parts(
                    scanout_address as *const u8,
                    (WIDTH * HEIGHT * 4) as usize,
                )
            };
            let red_pixels = pixels
                .chunks_exact(4)
                .filter(|pixel| pixel[0] < 4 && pixel[1] < 4 && pixel[2] > 250 && pixel[3] > 250)
                .count();
            assert!(
                red_pixels > 3_500,
                "only {red_pixels} pixels were rendered red through the {kind} target"
            );
        };
        assert_red_surface("single-sample scanout");
        let presentable = executor
            .take_presentable_bgra(scanout_address, WIDTH, HEIGHT)
            .expect("single-sample target retains its linear presentation result");
        assert!(
            presentable
                .chunks_exact(4)
                .filter(|pixel| pixel[0] < 4 && pixel[1] < 4 && pixel[2] > 250 && pixel[3] > 250)
                .count()
                > 3_500
        );
        executor
            .execute_draw(&draw)
            .expect("reuse coherent persistent single-sample target");
        assert_red_surface("persistent single-sample scanout");
        allocation.fill(0);
        executor
            .execute_draw(&draw)
            .expect("refresh CPU-dirtied persistent single-sample target");
        assert_red_surface("CPU-invalidated persistent single-sample scanout");
        allocation.fill(0);
        executor
            .execute_draw_deferred(&draw)
            .expect("defer coherent target publication");
        executor.synchronize_memory_acquire(GpuMemoryAcquire {
            engine: 0,
            cb_db_control: 0,
            base_address: 0,
            size_bytes: 0,
            poll_interval: 10,
            gcr_control: 1 << 14,
        });
        assert_eq!(
            executor.pending_render_target_count(),
            1,
            "ACQUIRE_MEM must not publish a GPU-owned color target"
        );
        // The draw is complete on the D3D12 queue, but its guest bytes remain unpublished until
        // an ordered CPU-visibility point. Flip reads the native target without changing that
        // ownership.
        let unpublished_red = unsafe {
            std::slice::from_raw_parts(scanout_address as *const u8, (WIDTH * HEIGHT * 4) as usize)
        }
        .chunks_exact(4)
        .filter(|pixel| pixel[2] > 250)
        .count();
        assert_eq!(unpublished_red, 0);
        let presentable = executor
            .prepare_presentable_bgra(scanout_address, WIDTH, HEIGHT)
            .expect("flush deferred target")
            .expect("flushed target supplies a frame");
        assert!(
            presentable
                .chunks_exact(4)
                .filter(|pixel| pixel[0] < 4 && pixel[1] < 4 && pixel[2] > 250 && pixel[3] > 250)
                .count()
                > 3_500
        );
        assert_eq!(executor.pending_render_target_count(), 1);
        let unpublished_red_after_present = unsafe {
            std::slice::from_raw_parts(scanout_address as *const u8, (WIDTH * HEIGHT * 4) as usize)
        }
        .chunks_exact(4)
        .filter(|pixel| pixel[2] > 250)
        .count();
        assert_eq!(unpublished_red_after_present, 0);
        executor
            .flush_all_render_targets()
            .expect("publish deferred target at submission boundary");
        assert_eq!(executor.pending_render_target_count(), 0);
        assert_red_surface("deferred single-sample scanout");

        // Exercise the full MSAA path on the physical adapter: upload the guest surface to a
        // single-sample texture, broadcast it into a 4x attachment, render, resolve, and copy the
        // resolved bytes back into guest memory.
        allocation.fill(0);
        let register = |offset, value| remu_gpu::GpuRegister { offset, value };
        draw.scanout_target = None;
        draw.primitive_type = 4;
        draw.context_registers = vec![
            register(0x08e, 0x0000_000f),
            register(0x318, (scanout_address >> 8) as u32),
            register(0x390, ((scanout_address >> 40) & 0xff) as u32),
            register(0x31c, (10 << 2) | (1 << 11)),
            register(0x31d, (2 << 12) | (2 << 15)),
            register(0x3b0, ((WIDTH - 1) << 14) | (HEIGHT - 1)),
            register(0x3b8, 0),
        ];
        executor
            .execute_draw(&draw)
            .expect("execute 4x MSAA guest graphics draw");
        assert_red_surface("4x MSAA color");

        allocation.fill(0);
        draw.context_registers = vec![
            register(0x08e, 0x0000_000f),
            register(0x318, (scanout_address >> 8) as u32),
            register(0x390, ((scanout_address >> 40) & 0xff) as u32),
            register(0x31c, (10 << 2) | (4 << 8)),
            register(0x31d, (2 << 12) | (2 << 15)),
            register(0x3b0, ((WIDTH - 1) << 14) | (HEIGHT - 1)),
            register(0x3b8, 0),
        ];
        executor
            .execute_draw(&draw)
            .expect("execute 4x uint MSAA guest graphics draw");
        // SAFETY: the aligned target covers WIDTH*HEIGHT tightly packed RGBA8_UINT pixels.
        let integer_pixels = unsafe {
            std::slice::from_raw_parts(scanout_address as *const u8, (WIDTH * HEIGHT * 4) as usize)
        };
        let integer_red_pixels = integer_pixels
            .chunks_exact(4)
            .filter(|pixel| pixel[0] > 250 && pixel[1] < 4 && pixel[2] < 4 && pixel[3] > 250)
            .count();
        assert!(
            integer_red_pixels > 3_500,
            "only {integer_red_pixels} uint pixels survived the sample-zero MSAA resolve"
        );
    }

    #[test]
    fn executes_depth_stencil_compare_replace_without_a_pixel_shader_and_retiles_planes() {
        const WIDTH: u32 = 64;
        const HEIGHT: u32 = 64;
        const RAW_BYTES: usize = 64 * 1024;
        let mut allocation = vec![0_u8; RAW_BYTES + 255];
        let depth_address = ((allocation.as_mut_ptr() as usize + 255) & !255_usize) as u64;
        let mut stencil_allocation = vec![0_u8; RAW_BYTES + 255];
        let stencil_address =
            ((stencil_allocation.as_mut_ptr() as usize + 255) & !255_usize) as u64;
        let vertex = Arc::new(Program::new(
            0xe000,
            vec![
                Instruction {
                    pc: 0,
                    encoding: ShaderEncoding::Vop2,
                    opcode: Opcode::Known("VLshlrevB32"),
                    words: vec![0],
                    sources: vec![literal(1), Operand::vector(5)],
                    destinations: vec![Operand::vector(0)],
                    control: None,
                },
                Instruction {
                    pc: 4,
                    encoding: ShaderEncoding::Vop2,
                    opcode: Opcode::Known("VAndB32"),
                    words: vec![0],
                    sources: vec![Operand::vector(0), literal(2)],
                    destinations: vec![Operand::vector(0)],
                    control: None,
                },
                Instruction {
                    pc: 8,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VCvtF32U32"),
                    words: vec![0],
                    sources: vec![Operand::vector(0)],
                    destinations: vec![Operand::vector(0)],
                    control: None,
                },
                Instruction {
                    pc: 12,
                    encoding: ShaderEncoding::Vop2,
                    opcode: Opcode::Known("VMulF32"),
                    words: vec![0],
                    sources: vec![Operand::vector(0), literal(2.0_f32.to_bits())],
                    destinations: vec![Operand::vector(0)],
                    control: None,
                },
                Instruction {
                    pc: 16,
                    encoding: ShaderEncoding::Vop2,
                    opcode: Opcode::Known("VAddF32"),
                    words: vec![0],
                    sources: vec![Operand::vector(0), literal((-1.0_f32).to_bits())],
                    destinations: vec![Operand::vector(0)],
                    control: None,
                },
                Instruction {
                    pc: 20,
                    encoding: ShaderEncoding::Vop2,
                    opcode: Opcode::Known("VAndB32"),
                    words: vec![0],
                    sources: vec![Operand::vector(5), literal(2)],
                    destinations: vec![Operand::vector(1)],
                    control: None,
                },
                Instruction {
                    pc: 24,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VCvtF32U32"),
                    words: vec![0],
                    sources: vec![Operand::vector(1)],
                    destinations: vec![Operand::vector(1)],
                    control: None,
                },
                Instruction {
                    pc: 28,
                    encoding: ShaderEncoding::Vop2,
                    opcode: Opcode::Known("VMulF32"),
                    words: vec![0],
                    sources: vec![Operand::vector(1), literal((-2.0_f32).to_bits())],
                    destinations: vec![Operand::vector(1)],
                    control: None,
                },
                Instruction {
                    pc: 32,
                    encoding: ShaderEncoding::Vop2,
                    opcode: Opcode::Known("VAddF32"),
                    words: vec![0],
                    sources: vec![Operand::vector(1), literal(1.0_f32.to_bits())],
                    destinations: vec![Operand::vector(1)],
                    control: None,
                },
                Instruction {
                    pc: 36,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VMovB32"),
                    words: vec![0],
                    sources: vec![literal(0.5_f32.to_bits())],
                    destinations: vec![Operand::vector(2)],
                    control: None,
                },
                Instruction {
                    pc: 40,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VMovB32"),
                    words: vec![0],
                    sources: vec![literal(1.0_f32.to_bits())],
                    destinations: vec![Operand::vector(3)],
                    control: None,
                },
                Instruction {
                    pc: 44,
                    encoding: ShaderEncoding::Exp,
                    opcode: Opcode::EXPORT,
                    words: vec![0, 0],
                    sources: (0..4).map(Operand::vector).collect(),
                    destinations: Vec::new(),
                    control: Some(InstructionControl::Export(ExportControl {
                        target: 12,
                        enable_mask: 0xf,
                        compressed: false,
                        done: true,
                        valid_mask: true,
                    })),
                },
                Instruction {
                    pc: 52,
                    encoding: ShaderEncoding::Sopp,
                    opcode: Opcode::END_PROGRAM,
                    words: vec![0xbf81_0000],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: None,
                },
            ],
        ));
        let register = |offset, value| remu_gpu::GpuRegister { offset, value };
        let mut draw = IndexedDraw {
            sequence: 5,
            indexed: false,
            indirect_arguments_address: None,
            indirect_count_address: None,
            indirect_draw_index: 0,
            instance_count: 1,
            index_buffer_address: 0,
            index_resource_address: 0,
            index_buffer_count: 0,
            index_size: 1,
            index_offset: 0,
            base_vertex: 0,
            index_count: 3,
            index_snapshot: None,
            draw_flags: 0,
            primitive_type: 4,
            export_shader_address: 0xe000,
            pixel_shader_address: 0,
            export_shader_prefix: [0; 4],
            pixel_shader_prefix: [0; 4],
            export_shader: Some(ShaderSnapshot {
                header_address: 0xe000,
                program: vertex,
                supplemental_programs: Arc::from([]),
                metadata: None,
            }),
            pixel_shader: None,
            export_shader_evaluation: Some(empty_evaluation_for_graphics()),
            pixel_shader_evaluation: None,
            color_target_address: 0,
            scanout_target: None,
            context_registers: vec![
                register(0x000, 3),
                register(0x007, (WIDTH - 1) | ((HEIGHT - 1) << 16)),
                register(0x00a, 0x5a),
                register(0x00b, 0.25_f32.to_bits()),
                register(0x010, 3),
                register(0x011, 1 | (24 << 4)),
                register(0x012, (depth_address >> 8) as u32),
                register(0x013, (stencil_address >> 8) as u32),
                register(0x014, (depth_address >> 8) as u32),
                register(0x015, (stencil_address >> 8) as u32),
                register(0x01a, ((depth_address >> 40) & 0xff) as u32),
                register(0x01b, ((stencil_address >> 40) & 0xff) as u32),
                register(0x01c, ((depth_address >> 40) & 0xff) as u32),
                register(0x01d, ((stencil_address >> 40) & 0xff) as u32),
                register(0x10b, (3 << 4) | (3 << 16)),
                register(0x10c, 0x00ff_ffa5),
                register(0x10d, 0x000f_0f3c),
                register(0x200, 1 | (1 << 7) | (7 << 8) | (7 << 20)),
            ],
            shader_registers: Vec::new(),
            user_config_registers: Vec::new(),
        };

        Dx12GuestExecutor::new()
            .expect("create D3D12 guest executor")
            .execute_draw(&draw)
            .expect("execute depth/stencil-only guest draw");

        let linear = super::read_texture_linear(super::DecodedTexture {
            address: depth_address,
            width: WIDTH,
            height: HEIGHT,
            pitch: WIDTH,
            tile_mode: 24,
            format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D32_FLOAT,
            bytes_per_element: 4,
            block_extent: [1, 1],
            texture_type: 9,
            layer_count: 1,
            mip_level_count: 1,
            writable: true,
        })
        .expect("detile depth clear");
        assert!(linear.chunks_exact(4).all(|bytes| {
            (f32::from_le_bytes(bytes.try_into().expect("depth dword")) - 0.25).abs() < f32::EPSILON
        }));
        let stencil = super::read_texture_linear(super::DecodedTexture {
            address: stencil_address,
            width: WIDTH,
            height: HEIGHT,
            pitch: 256,
            tile_mode: 24,
            format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8_UINT,
            bytes_per_element: 1,
            block_extent: [1, 1],
            texture_type: 9,
            layer_count: 1,
            mip_level_count: 1,
            writable: true,
        })
        .expect("detile stencil clear");
        let replaced = stencil.iter().filter(|&&value| value != 0x5a).count();
        assert!(
            replaced > 3_500,
            "only {replaced} stencil samples passed and were replaced"
        );
        assert!(
            stencil
                .into_iter()
                .all(|value| matches!(value, 0x5a | 0xa5 | 0x5c))
        );

        allocation.fill(0);
        draw.sequence = 6;
        draw.context_registers = vec![
            register(0x000, 1),
            register(0x007, (WIDTH - 1) | ((HEIGHT - 1) << 16)),
            register(0x00b, 0.25_f32.to_bits()),
            register(0x010, 3 | (2 << 2)),
            register(0x012, (depth_address >> 8) as u32),
            register(0x014, (depth_address >> 8) as u32),
            register(0x01a, ((depth_address >> 40) & 0xff) as u32),
            register(0x01c, ((depth_address >> 40) & 0xff) as u32),
            register(0x200, (1 << 1) | (1 << 2) | (7 << 4)),
        ];
        Dx12GuestExecutor::new()
            .expect("create D3D12 guest executor")
            .execute_draw(&draw)
            .expect("execute 4x MSAA depth-only guest draw");

        let msaa_depth = super::read_texture_linear(super::DecodedTexture {
            address: depth_address,
            width: WIDTH,
            height: HEIGHT,
            pitch: WIDTH,
            tile_mode: 24,
            format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D32_FLOAT,
            bytes_per_element: 4,
            block_extent: [1, 1],
            texture_type: 9,
            layer_count: 1,
            mip_level_count: 1,
            writable: true,
        })
        .expect("detile resolved 4x MSAA depth");
        let drawn_samples = msaa_depth
            .chunks_exact(4)
            .filter(|bytes| {
                (f32::from_le_bytes((*bytes).try_into().expect("depth dword")) - 0.5).abs()
                    < f32::EPSILON
            })
            .count();
        assert!(
            drawn_samples > 3_500,
            "only {drawn_samples} resolved MSAA depth samples were written"
        );

        allocation.fill(0);
        stencil_allocation.fill(0);
        draw.sequence = 7;
        draw.context_registers = vec![
            register(0x000, 3),
            register(0x007, (WIDTH - 1) | ((HEIGHT - 1) << 16)),
            register(0x00a, 0x5a),
            register(0x00b, 0.25_f32.to_bits()),
            register(0x010, 3 | (2 << 2)),
            register(0x011, 1 | (24 << 4)),
            register(0x012, (depth_address >> 8) as u32),
            register(0x013, (stencil_address >> 8) as u32),
            register(0x014, (depth_address >> 8) as u32),
            register(0x015, (stencil_address >> 8) as u32),
            register(0x01a, ((depth_address >> 40) & 0xff) as u32),
            register(0x01b, ((stencil_address >> 40) & 0xff) as u32),
            register(0x01c, ((depth_address >> 40) & 0xff) as u32),
            register(0x01d, ((stencil_address >> 40) & 0xff) as u32),
            register(0x10b, (3 << 4) | (3 << 16)),
            register(0x10c, 0x00ff_ffa5),
            register(0x10d, 0x000f_0f3c),
            register(0x200, 1 | (1 << 7) | (7 << 8) | (7 << 20)),
        ];
        Dx12GuestExecutor::new()
            .expect("create D3D12 guest executor")
            .execute_draw(&draw)
            .expect("execute 4x MSAA combined depth/stencil guest draw");

        let resolved_stencil = super::read_texture_linear(super::DecodedTexture {
            address: stencil_address,
            width: WIDTH,
            height: HEIGHT,
            pitch: 256,
            tile_mode: 24,
            format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8_UINT,
            bytes_per_element: 1,
            block_extent: [1, 1],
            texture_type: 9,
            layer_count: 1,
            mip_level_count: 1,
            writable: true,
        })
        .expect("detile resolved 4x MSAA stencil");
        let replaced = resolved_stencil
            .iter()
            .filter(|&&value| value != 0x5a)
            .count();
        assert!(
            replaced > 3_500,
            "only {replaced} resolved MSAA stencil samples were replaced"
        );
    }
}
