// SPDX-License-Identifier: GPL-2.0-or-later

//! Headless Shader Model 6 graphics and compute validation.

use std::ffi::c_void;
use std::mem::ManuallyDrop;

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::{D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::{
    D3D_ROOT_SIGNATURE_VERSION_1, D3D12_BLEND_DESC, D3D12_BLEND_ONE, D3D12_BLEND_OP_ADD,
    D3D12_BLEND_ZERO, D3D12_BUFFER_SRV, D3D12_BUFFER_SRV_FLAG_NONE, D3D12_BUFFER_UAV,
    D3D12_BUFFER_UAV_FLAG_RAW, D3D12_COLOR_WRITE_ENABLE_ALL, D3D12_COMMAND_LIST_TYPE_DIRECT,
    D3D12_COMMAND_QUEUE_DESC, D3D12_COMMAND_QUEUE_FLAG_NONE, D3D12_COMPARISON_FUNC_ALWAYS,
    D3D12_COMPUTE_PIPELINE_STATE_DESC, D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
    D3D12_CULL_MODE_NONE, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, D3D12_DEPTH_STENCIL_DESC,
    D3D12_DEPTH_WRITE_MASK_ZERO, D3D12_DESCRIPTOR_HEAP_DESC, D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
    D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
    D3D12_DESCRIPTOR_HEAP_TYPE_RTV, D3D12_FENCE_FLAG_NONE, D3D12_FILL_MODE_SOLID,
    D3D12_GRAPHICS_PIPELINE_STATE_DESC, D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES,
    D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_TYPE_READBACK, D3D12_LOGIC_OP_NOOP,
    D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, D3D12_RASTERIZER_DESC, D3D12_RENDER_TARGET_BLEND_DESC,
    D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
    D3D12_RESOURCE_BARRIER_FLAG_NONE, D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_DESC,
    D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_DIMENSION_TEXTURE2D,
    D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE,
    D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_RENDER_TARGET,
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_TRANSITION_BARRIER,
    D3D12_ROOT_DESCRIPTOR, D3D12_ROOT_PARAMETER, D3D12_ROOT_PARAMETER_0,
    D3D12_ROOT_PARAMETER_TYPE_UAV, D3D12_ROOT_SIGNATURE_DESC,
    D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT, D3D12_ROOT_SIGNATURE_FLAG_NONE,
    D3D12_SHADER_BYTECODE, D3D12_SHADER_RESOURCE_VIEW_DESC, D3D12_SHADER_RESOURCE_VIEW_DESC_0,
    D3D12_SHADER_VISIBILITY_ALL, D3D12_SRV_DIMENSION_BUFFER, D3D12_STENCIL_OP_KEEP,
    D3D12_SUBRESOURCE_FOOTPRINT, D3D12_TEXTURE_COPY_LOCATION, D3D12_TEXTURE_COPY_LOCATION_0,
    D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
    D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_UAV_DIMENSION_BUFFER,
    D3D12_UNORDERED_ACCESS_VIEW_DESC, D3D12_UNORDERED_ACCESS_VIEW_DESC_0, D3D12_VIEWPORT,
    D3D12SerializeRootSignature, ID3D12CommandAllocator, ID3D12CommandList, ID3D12CommandQueue,
    ID3D12DescriptorHeap, ID3D12Device, ID3D12Fence, ID3D12GraphicsCommandList,
    ID3D12PipelineState, ID3D12Resource, ID3D12RootSignature,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC,
};
use windows::core::{BOOL, Interface};

use crate::backend::{backend_error, create_default_device};
use crate::context::{create_buffer, read_readback, wait_for_fence, write_upload};
use crate::dxc::compile_hlsl;
use crate::executor::create_guest_root_signature;
use crate::guest_shader::{build_guest_compute_self_test, compile_guest_shader};
use crate::{Dx12PipelineSelfTestReport, Dx12ProbeError};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;
const ROW_PITCH: u32 = 256;
const COMPUTE_VALUES: usize = 64;

const SELF_TEST_HLSL: &str = r#"
struct VertexOutput {
    float4 position : SV_Position;
    float3 color : COLOR0;
};

VertexOutput vs_main(uint vertex_id : SV_VertexID) {
    const float2 positions[3] = {
        float2(-0.8, -0.8),
        float2( 0.0,  0.8),
        float2( 0.8, -0.8)
    };
    const float3 colors[3] = {
        float3(1.0, 0.0, 0.0),
        float3(0.0, 1.0, 0.0),
        float3(0.0, 0.0, 1.0)
    };
    VertexOutput output;
    output.position = float4(positions[vertex_id], 0.0, 1.0);
    output.color = colors[vertex_id];
    return output;
}

float4 ps_main(VertexOutput input) : SV_Target0 {
    return float4(input.color, 1.0);
}

RWByteAddressBuffer compute_output : register(u0);

[numthreads(64, 1, 1)]
void cs_main(uint3 thread_id : SV_DispatchThreadID) {
    compute_output.Store(thread_id.x * 4,
        thread_id.x * 2654435761u + 0x01234567u);
}
"#;

pub(super) fn run_pipeline_self_test() -> Result<Dx12PipelineSelfTestReport, Dx12ProbeError> {
    let vertex = compile_hlsl(
        "remu_dx12_self_test.hlsl",
        SELF_TEST_HLSL,
        "vs_main",
        "vs_6_0",
    )
    .map_err(Dx12ProbeError::ShaderCompiler)?;
    let pixel = compile_hlsl(
        "remu_dx12_self_test.hlsl",
        SELF_TEST_HLSL,
        "ps_main",
        "ps_6_0",
    )
    .map_err(Dx12ProbeError::ShaderCompiler)?;
    let compute = compile_hlsl(
        "remu_dx12_self_test.hlsl",
        SELF_TEST_HLSL,
        "cs_main",
        "cs_6_0",
    )
    .map_err(Dx12ProbeError::ShaderCompiler)?;
    let guest_test = build_guest_compute_self_test().map_err(|error| {
        Dx12ProbeError::ShaderCompiler(format!("guest shader translation failed: {error}"))
    })?;
    let guest_compute = compile_guest_shader(&guest_test.shader, "remu_guest_self_test.hlsl")
        .map_err(Dx12ProbeError::ShaderCompiler)?;

    let native = create_default_device()?;
    let device = native.device;
    let queue_desc = D3D12_COMMAND_QUEUE_DESC {
        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
        Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
        ..D3D12_COMMAND_QUEUE_DESC::default()
    };
    // SAFETY: descriptors are initialized and each returned COM object owns its reference.
    let queue: ID3D12CommandQueue = unsafe { device.CreateCommandQueue(&raw const queue_desc) }
        .map_err(|error| {
            backend_error("ID3D12Device::CreateCommandQueue(pipeline test)", &error)
        })?;
    // SAFETY: allocator and command list use the same direct-list type.
    let allocator: ID3D12CommandAllocator =
        unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }.map_err(
            |error| {
                backend_error(
                    "ID3D12Device::CreateCommandAllocator(pipeline test)",
                    &error,
                )
            },
        )?;
    // SAFETY: allocator is live and the list begins without an initial PSO.
    let commands: ID3D12GraphicsCommandList = unsafe {
        device.CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            &allocator,
            None::<&ID3D12PipelineState>,
        )
    }
    .map_err(|error| backend_error("ID3D12Device::CreateCommandList(pipeline test)", &error))?;

    let graphics_root = create_root_signature(&device, &[], true)?;
    let compute_parameter = D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR {
                ShaderRegister: 0,
                RegisterSpace: 0,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    };
    let compute_root =
        create_root_signature(&device, std::slice::from_ref(&compute_parameter), false)?;
    let guest_root =
        create_guest_root_signature(&device).map_err(Dx12ProbeError::ShaderCompiler)?;
    let graphics_pipeline = create_graphics_pipeline(&device, &graphics_root, &vertex, &pixel)?;
    let compute_pipeline = create_compute_pipeline(&device, &compute_root, &compute)?;
    let guest_pipeline = create_compute_pipeline(&device, &guest_root, &guest_compute)?;

    let render_target = create_render_target(&device)?;
    let rtv_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
        NumDescriptors: 1,
        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
        NodeMask: 0,
    };
    // SAFETY: descriptor is valid and the returned heap owns its storage.
    let rtv_heap: ID3D12DescriptorHeap =
        unsafe { device.CreateDescriptorHeap(&raw const rtv_heap_desc) }
            .map_err(|error| backend_error("ID3D12Device::CreateDescriptorHeap(RTV)", &error))?;
    // SAFETY: a one-element RTV heap has a valid start handle.
    let rtv = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };
    // SAFETY: resource has ALLOW_RENDER_TARGET and its format is inferable from the resource.
    unsafe { device.CreateRenderTargetView(&render_target, None, rtv) };

    let texture_readback = create_buffer(
        &device,
        (ROW_PITCH * HEIGHT) as usize,
        D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_STATE_COPY_DEST,
    )?;
    let compute_buffer = create_uav_buffer(&device, COMPUTE_VALUES * 4)?;
    let compute_readback = create_buffer(
        &device,
        COMPUTE_VALUES * 4,
        D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_STATE_COPY_DEST,
    )?;
    let guest_resources =
        create_guest_compute_resources(&device, &guest_test.initial_scalars, COMPUTE_VALUES * 4)?;

    record_graphics(
        &commands,
        &graphics_root,
        &graphics_pipeline,
        &render_target,
        rtv,
        &texture_readback,
    );
    record_compute(
        &commands,
        &compute_root,
        &compute_pipeline,
        &compute_buffer,
        &compute_readback,
    );
    record_guest_compute(
        &commands,
        &guest_root,
        &guest_pipeline,
        &guest_resources.descriptors,
        &guest_resources.output,
        &guest_resources.readback,
    );

    // SAFETY: command recording is complete and the list is open.
    unsafe { commands.Close() }.map_err(|error| {
        backend_error("ID3D12GraphicsCommandList::Close(pipeline test)", &error)
    })?;
    let list: ID3D12CommandList = commands
        .cast()
        .map_err(|error| backend_error("ID3D12GraphicsCommandList::cast(pipeline test)", &error))?;
    // SAFETY: the list is closed and every referenced object remains live through fence wait.
    unsafe { queue.ExecuteCommandLists(&[Some(list)]) };
    // SAFETY: the fence belongs to this device and starts before the signal value.
    let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
        .map_err(|error| backend_error("ID3D12Device::CreateFence(pipeline test)", &error))?;
    // SAFETY: queue and fence are live.
    unsafe { queue.Signal(&fence, 1) }
        .map_err(|error| backend_error("ID3D12CommandQueue::Signal(pipeline test)", &error))?;
    wait_for_fence(&fence, 1)?;

    let pixels = read_readback(&texture_readback, (ROW_PITCH * HEIGHT) as usize)?;
    let colored_pixels = count_colored_pixels(&pixels);
    if colored_pixels < 800 {
        return Err(Dx12ProbeError::PipelineSelfTestMismatch(
            "triangle render target did not contain the expected colored area",
        ));
    }
    let compute_bytes = read_readback(&compute_readback, COMPUTE_VALUES * 4)?;
    let computed_values = validate_compute(&compute_bytes)?;
    let guest_bytes = read_readback(&guest_resources.readback, COMPUTE_VALUES * 4)?;
    let guest_computed_values = validate_guest_compute(&guest_bytes)?;

    Ok(Dx12PipelineSelfTestReport {
        adapter: native.capabilities.name,
        shader_model: "6.0".to_owned(),
        colored_pixels,
        computed_values,
        guest_computed_values,
    })
}

fn create_root_signature(
    device: &ID3D12Device,
    parameters: &[D3D12_ROOT_PARAMETER],
    input_assembler: bool,
) -> Result<ID3D12RootSignature, Dx12ProbeError> {
    let flags = if input_assembler {
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
    } else {
        D3D12_ROOT_SIGNATURE_FLAG_NONE
    };
    create_root_signature_with_flags(device, parameters, flags)
}

fn create_root_signature_with_flags(
    device: &ID3D12Device,
    parameters: &[D3D12_ROOT_PARAMETER],
    flags: windows::Win32::Graphics::Direct3D12::D3D12_ROOT_SIGNATURE_FLAGS,
) -> Result<ID3D12RootSignature, Dx12ProbeError> {
    let description = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: u32::try_from(parameters.len()).expect("root parameter count fits u32"),
        pParameters: parameters.as_ptr(),
        NumStaticSamplers: 0,
        pStaticSamplers: std::ptr::null(),
        Flags: flags,
    };
    let mut blob = None;
    let mut diagnostics = None;
    // SAFETY: description and output slots remain live for the synchronous serializer call.
    unsafe {
        D3D12SerializeRootSignature(
            &raw const description,
            D3D_ROOT_SIGNATURE_VERSION_1,
            &raw mut blob,
            Some(&raw mut diagnostics),
        )
    }
    .map_err(|error| backend_error("D3D12SerializeRootSignature", &error))?;
    let blob: ID3DBlob = blob.ok_or(Dx12ProbeError::MissingCreatedResource)?;
    let bytes = blob_bytes(&blob);
    // SAFETY: serialized bytes describe a version-1 root signature.
    unsafe { device.CreateRootSignature(0, bytes) }
        .map_err(|error| backend_error("ID3D12Device::CreateRootSignature", &error))
}

fn create_graphics_pipeline(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    vertex: &[u8],
    pixel: &[u8],
) -> Result<ID3D12PipelineState, Dx12ProbeError> {
    let mut blend = D3D12_BLEND_DESC::default();
    blend.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
        BlendEnable: BOOL(0),
        LogicOpEnable: BOOL(0),
        SrcBlend: D3D12_BLEND_ONE,
        DestBlend: D3D12_BLEND_ZERO,
        BlendOp: D3D12_BLEND_OP_ADD,
        SrcBlendAlpha: D3D12_BLEND_ONE,
        DestBlendAlpha: D3D12_BLEND_ZERO,
        BlendOpAlpha: D3D12_BLEND_OP_ADD,
        LogicOp: D3D12_LOGIC_OP_NOOP,
        RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
    };
    let rasterizer = D3D12_RASTERIZER_DESC {
        FillMode: D3D12_FILL_MODE_SOLID,
        CullMode: D3D12_CULL_MODE_NONE,
        DepthClipEnable: BOOL(1),
        ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
        ..D3D12_RASTERIZER_DESC::default()
    };
    let depth_stencil = D3D12_DEPTH_STENCIL_DESC {
        DepthEnable: BOOL(0),
        DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
        DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        StencilEnable: BOOL(0),
        StencilReadMask: 0xff,
        StencilWriteMask: 0xff,
        FrontFace: windows::Win32::Graphics::Direct3D12::D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        },
        BackFace: windows::Win32::Graphics::Direct3D12::D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        },
    };
    let mut description = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root.clone())),
        VS: shader_bytecode(vertex),
        PS: shader_bytecode(pixel),
        BlendState: blend,
        SampleMask: u32::MAX,
        RasterizerState: rasterizer,
        DepthStencilState: depth_stencil,
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..D3D12_GRAPHICS_PIPELINE_STATE_DESC::default()
    };
    description.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    // SAFETY: shader bytecode, root signature, and descriptor fields remain live for creation.
    let result = unsafe { device.CreateGraphicsPipelineState(&raw const description) }
        .map_err(|error| backend_error("ID3D12Device::CreateGraphicsPipelineState", &error));
    // SAFETY: this explicit clone was retained only while the PSO descriptor was in use.
    unsafe { ManuallyDrop::drop(&mut description.pRootSignature) };
    result
}

fn create_compute_pipeline(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    compute: &[u8],
) -> Result<ID3D12PipelineState, Dx12ProbeError> {
    let mut description = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root.clone())),
        CS: shader_bytecode(compute),
        ..D3D12_COMPUTE_PIPELINE_STATE_DESC::default()
    };
    // SAFETY: shader bytecode and root signature remain live for synchronous PSO creation.
    let result = unsafe { device.CreateComputePipelineState(&raw const description) }
        .map_err(|error| backend_error("ID3D12Device::CreateComputePipelineState", &error));
    // SAFETY: this explicit clone was retained only while the PSO descriptor was in use.
    unsafe { ManuallyDrop::drop(&mut description.pRootSignature) };
    result
}

fn create_render_target(device: &ID3D12Device) -> Result<ID3D12Resource, Dx12ProbeError> {
    create_resource(
        device,
        D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(WIDTH),
            Height: HEIGHT,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..D3D12_RESOURCE_DESC::default()
        },
        D3D12_RESOURCE_STATE_RENDER_TARGET,
    )
}

fn create_uav_buffer(
    device: &ID3D12Device,
    byte_count: usize,
) -> Result<ID3D12Resource, Dx12ProbeError> {
    create_resource(
        device,
        D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: u64::try_from(byte_count).expect("compute test buffer size fits u64"),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..D3D12_RESOURCE_DESC::default()
        },
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    )
}

struct GuestComputeResources {
    descriptors: ID3D12DescriptorHeap,
    output: ID3D12Resource,
    readback: ID3D12Resource,
    _binding_table: ID3D12Resource,
    _initial_scalars: ID3D12Resource,
}

fn create_guest_compute_resources(
    device: &ID3D12Device,
    initial_scalars: &[u32],
    output_bytes: usize,
) -> Result<GuestComputeResources, Dx12ProbeError> {
    let binding_words = [1_u32, 2_u32];
    let binding_bytes = words_as_bytes(&binding_words);
    let scalar_bytes = words_as_bytes(initial_scalars);
    let binding_table = create_buffer(
        device,
        binding_bytes.len(),
        windows::Win32::Graphics::Direct3D12::D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;
    let initial_scalars_resource = create_buffer(
        device,
        scalar_bytes.len(),
        windows::Win32::Graphics::Direct3D12::D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;
    write_upload(&binding_table, &binding_bytes)?;
    write_upload(&initial_scalars_resource, &scalar_bytes)?;
    let output = create_uav_buffer(device, output_bytes)?;
    let readback = create_buffer(
        device,
        output_bytes,
        D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_STATE_COPY_DEST,
    )?;

    let heap_description = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        NumDescriptors: 3,
        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
        NodeMask: 0,
    };
    // SAFETY: the descriptor is complete and the returned heap owns three descriptor slots.
    let descriptors: ID3D12DescriptorHeap =
        unsafe { device.CreateDescriptorHeap(&raw const heap_description) }.map_err(|error| {
            backend_error(
                "ID3D12Device::CreateDescriptorHeap(guest shader test)",
                &error,
            )
        })?;
    // SAFETY: a three-element heap has a valid CPU start handle.
    let mut handle = unsafe { descriptors.GetCPUDescriptorHandleForHeapStart() };
    let increment = usize::try_from(unsafe {
        device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
    })
    .expect("descriptor increment fits usize");
    create_structured_u32_srv(
        device,
        &binding_table,
        u32::try_from(binding_words.len()).expect("binding word count fits u32"),
        handle,
    );
    handle.ptr += increment;
    create_structured_u32_srv(
        device,
        &initial_scalars_resource,
        u32::try_from(initial_scalars.len()).expect("scalar word count fits u32"),
        handle,
    );
    handle.ptr += increment;
    let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_UAV {
                FirstElement: 0,
                NumElements: u32::try_from(output_bytes / 4)
                    .expect("guest output dword count fits u32"),
                StructureByteStride: 0,
                CounterOffsetInBytes: 0,
                Flags: D3D12_BUFFER_UAV_FLAG_RAW,
            },
        },
    };
    // SAFETY: output allows UAV access and the raw descriptor covers its complete dword range.
    unsafe {
        device.CreateUnorderedAccessView(
            &output,
            None::<&ID3D12Resource>,
            Some(&raw const uav),
            handle,
        )
    };
    Ok(GuestComputeResources {
        descriptors,
        output,
        readback,
        _binding_table: binding_table,
        _initial_scalars: initial_scalars_resource,
    })
}

fn create_structured_u32_srv(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    word_count: u32,
    handle: windows::Win32::Graphics::Direct3D12::D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let description = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_UNKNOWN,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                FirstElement: 0,
                NumElements: word_count,
                StructureByteStride: 4,
                Flags: D3D12_BUFFER_SRV_FLAG_NONE,
            },
        },
    };
    // SAFETY: the upload buffer contains `word_count` tightly packed u32 values.
    unsafe { device.CreateShaderResourceView(resource, Some(&raw const description), handle) };
}

fn words_as_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn create_resource(
    device: &ID3D12Device,
    description: D3D12_RESOURCE_DESC,
    state: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES,
) -> Result<ID3D12Resource, Dx12ProbeError> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..D3D12_HEAP_PROPERTIES::default()
    };
    let mut resource = None;
    // SAFETY: heap/resource descriptors and output storage are valid.
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
        backend_error(
            "ID3D12Device::CreateCommittedResource(pipeline test)",
            &error,
        )
    })?;
    resource.ok_or(Dx12ProbeError::MissingCreatedResource)
}

fn record_graphics(
    commands: &ID3D12GraphicsCommandList,
    root: &ID3D12RootSignature,
    pipeline: &ID3D12PipelineState,
    render_target: &ID3D12Resource,
    rtv: windows::Win32::Graphics::Direct3D12::D3D12_CPU_DESCRIPTOR_HANDLE,
    readback: &ID3D12Resource,
) {
    let viewport = D3D12_VIEWPORT {
        Width: WIDTH as f32,
        Height: HEIGHT as f32,
        MinDepth: 0.0,
        MaxDepth: 1.0,
        ..D3D12_VIEWPORT::default()
    };
    let scissor = RECT {
        left: 0,
        top: 0,
        right: WIDTH as i32,
        bottom: HEIGHT as i32,
    };
    // SAFETY: all state objects and descriptors are live; the root signature matches the PSO.
    unsafe {
        commands.SetGraphicsRootSignature(root);
        commands.SetPipelineState(pipeline);
        commands.RSSetViewports(std::slice::from_ref(&viewport));
        commands.RSSetScissorRects(std::slice::from_ref(&scissor));
        commands.OMSetRenderTargets(1, Some(&raw const rtv), false, None);
        commands.ClearRenderTargetView(rtv, &[0.0, 0.0, 0.0, 1.0], None);
        commands.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        commands.DrawInstanced(3, 1, 0, 0);
    }
    record_transition(
        commands,
        render_target,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );
    let mut source = D3D12_TEXTURE_COPY_LOCATION {
        pResource: ManuallyDrop::new(Some(render_target.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: 0,
        },
    };
    let mut destination = D3D12_TEXTURE_COPY_LOCATION {
        pResource: ManuallyDrop::new(Some(readback.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint:
                windows::Win32::Graphics::Direct3D12::D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        Width: WIDTH,
                        Height: HEIGHT,
                        Depth: 1,
                        RowPitch: ROW_PITCH,
                    },
                },
        },
    };
    // SAFETY: source is COPY_SOURCE and destination is a large-enough readback footprint.
    unsafe {
        commands.CopyTextureRegion(&raw const destination, 0, 0, 0, &raw const source, None);
        ManuallyDrop::drop(&mut source.pResource);
        ManuallyDrop::drop(&mut destination.pResource);
    }
}

fn record_compute(
    commands: &ID3D12GraphicsCommandList,
    root: &ID3D12RootSignature,
    pipeline: &ID3D12PipelineState,
    output: &ID3D12Resource,
    readback: &ID3D12Resource,
) {
    // SAFETY: root signature matches the PSO and parameter zero is a root UAV descriptor.
    unsafe {
        commands.SetComputeRootSignature(root);
        commands.SetPipelineState(pipeline);
        commands.SetComputeRootUnorderedAccessView(0, output.GetGPUVirtualAddress());
        commands.Dispatch(1, 1, 1);
    }
    record_transition(
        commands,
        output,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );
    // SAFETY: transitioned source and readback destination both cover the copied byte range.
    unsafe { commands.CopyBufferRegion(readback, 0, output, 0, (COMPUTE_VALUES * 4) as u64) };
}

fn record_guest_compute(
    commands: &ID3D12GraphicsCommandList,
    root: &ID3D12RootSignature,
    pipeline: &ID3D12PipelineState,
    descriptors: &ID3D12DescriptorHeap,
    output: &ID3D12Resource,
    readback: &ID3D12Resource,
) {
    // SAFETY: the shader-visible heap contains the binding table, initial SGPR SRV and raw UAV;
    // root constant zero selects descriptor zero and the root signature matches embedded DXIL.
    unsafe {
        commands.SetDescriptorHeaps(&[Some(descriptors.clone())]);
        commands.SetComputeRootSignature(root);
        commands.SetPipelineState(pipeline);
        let root_constants = [0_u32; 7];
        commands.SetComputeRoot32BitConstants(
            2,
            root_constants.len() as u32,
            root_constants.as_ptr().cast::<c_void>(),
            0,
        );
        commands.Dispatch(1, 1, 1);
    }
    record_transition(
        commands,
        output,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );
    // SAFETY: the transitioned output and readback resource both cover the complete test range.
    unsafe { commands.CopyBufferRegion(readback, 0, output, 0, (COMPUTE_VALUES * 4) as u64) };
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
    // SAFETY: Transition is the active union member and states match recorded resource use.
    unsafe {
        commands.ResourceBarrier(std::slice::from_ref(&barrier));
        ManuallyDrop::drop(&mut (*barrier.Anonymous.Transition).pResource);
    }
}

fn shader_bytecode(bytes: &[u8]) -> D3D12_SHADER_BYTECODE {
    D3D12_SHADER_BYTECODE {
        pShaderBytecode: bytes.as_ptr().cast::<c_void>(),
        BytecodeLength: bytes.len(),
    }
}

fn blob_bytes(blob: &ID3DBlob) -> &[u8] {
    // SAFETY: ID3DBlob owns an immutable allocation for its lifetime.
    unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
    }
}

fn count_colored_pixels(bytes: &[u8]) -> usize {
    bytes
        .chunks_exact(ROW_PITCH as usize)
        .take(HEIGHT as usize)
        .flat_map(|row| row[..(WIDTH * 4) as usize].chunks_exact(4))
        .filter(|pixel| pixel[0] > 8 || pixel[1] > 8 || pixel[2] > 8)
        .count()
}

fn validate_compute(bytes: &[u8]) -> Result<usize, Dx12ProbeError> {
    let mut validated = 0;
    for (index, value) in bytes.chunks_exact(4).enumerate() {
        let actual = u32::from_le_bytes(value.try_into().expect("chunk is four bytes"));
        let expected = (index as u32)
            .wrapping_mul(2_654_435_761)
            .wrapping_add(0x0123_4567);
        if actual != expected {
            return Err(Dx12ProbeError::PipelineSelfTestMismatch(
                "compute UAV contents did not match the dispatched shader",
            ));
        }
        validated += 1;
    }
    Ok(validated)
}

fn validate_guest_compute(bytes: &[u8]) -> Result<usize, Dx12ProbeError> {
    let mut validated = 0;
    for (index, value) in bytes.chunks_exact(4).enumerate() {
        let actual = u32::from_le_bytes(value.try_into().expect("chunk is four bytes"));
        let expected = 0x1234_0000_u32.wrapping_add(index as u32);
        if actual != expected {
            return Err(Dx12ProbeError::PipelineSelfTestMismatch(
                "translated guest compute UAV contents did not match the Gen5 program",
            ));
        }
        validated += 1;
    }
    Ok(validated)
}

#[cfg(test)]
mod tests {
    use super::{COMPUTE_VALUES, ROW_PITCH, count_colored_pixels, validate_compute};

    #[test]
    fn colored_pixel_count_ignores_alpha_and_padding() {
        let mut data = vec![0_u8; ROW_PITCH as usize];
        data[3] = 255;
        data[4] = 9;
        data[ROW_PITCH as usize - 1] = 255;
        assert_eq!(count_colored_pixels(&data), 1);
    }

    #[test]
    fn compute_validation_matches_shader_formula() {
        let bytes = (0..COMPUTE_VALUES)
            .flat_map(|index| {
                (index as u32)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(0x0123_4567)
                    .to_le_bytes()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            validate_compute(&bytes).expect("known values"),
            COMPUTE_VALUES
        );
    }
}
