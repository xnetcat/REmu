// SPDX-License-Identifier: GPL-2.0-or-later

use std::ffi::c_void;
use std::mem::size_of;

use windows::Win32::Graphics::Direct3D::{
    D3D_FEATURE_LEVEL, D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_12_0,
    D3D_FEATURE_LEVEL_12_1, D3D_FEATURE_LEVEL_12_2,
};
use windows::Win32::Graphics::Direct3D12::{
    D3D_SHADER_MODEL, D3D_SHADER_MODEL_5_1, D3D_SHADER_MODEL_6_0, D3D_SHADER_MODEL_6_1,
    D3D_SHADER_MODEL_6_2, D3D_SHADER_MODEL_6_3, D3D_SHADER_MODEL_6_4, D3D_SHADER_MODEL_6_5,
    D3D_SHADER_MODEL_6_6, D3D_SHADER_MODEL_6_7, D3D_SHADER_MODEL_6_8, D3D_SHADER_MODEL_6_9,
    D3D12_FEATURE, D3D12_FEATURE_ARCHITECTURE1, D3D12_FEATURE_D3D12_OPTIONS,
    D3D12_FEATURE_D3D12_OPTIONS1, D3D12_FEATURE_D3D12_OPTIONS7, D3D12_FEATURE_D3D12_OPTIONS12,
    D3D12_FEATURE_D3D12_OPTIONS14, D3D12_FEATURE_D3D12_OPTIONS15, D3D12_FEATURE_D3D12_OPTIONS16,
    D3D12_FEATURE_DATA_ARCHITECTURE1, D3D12_FEATURE_DATA_D3D12_OPTIONS,
    D3D12_FEATURE_DATA_D3D12_OPTIONS1, D3D12_FEATURE_DATA_D3D12_OPTIONS7,
    D3D12_FEATURE_DATA_D3D12_OPTIONS12, D3D12_FEATURE_DATA_D3D12_OPTIONS14,
    D3D12_FEATURE_DATA_D3D12_OPTIONS15, D3D12_FEATURE_DATA_D3D12_OPTIONS16,
    D3D12_FEATURE_DATA_FEATURE_LEVELS, D3D12_FEATURE_DATA_SHADER_MODEL,
    D3D12_FEATURE_FEATURE_LEVELS, D3D12_FEATURE_SHADER_MODEL, D3D12CreateDevice, ID3D12Device,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_ADAPTER_FLAG3_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS,
    DXGI_ERROR_NOT_FOUND, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, DXGI_MEMORY_SEGMENT_GROUP,
    DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,
    DXGI_QUERY_VIDEO_MEMORY_INFO, IDXGIAdapter4, IDXGIFactory6,
};

use crate::diagnostics::{configure_before_device_creation, device_removed_report};
use crate::{Dx12Capabilities, Dx12MemoryBudget, Dx12ProbeError};

pub(super) struct NativeDevice {
    pub(super) device: ID3D12Device,
    pub(super) capabilities: Dx12Capabilities,
}

pub(super) fn probe_adapters() -> Result<Vec<Dx12Capabilities>, Dx12ProbeError> {
    Ok(enumerate_native_devices()?
        .into_iter()
        .map(|native| native.capabilities)
        .collect())
}

pub(super) fn create_default_device() -> Result<NativeDevice, Dx12ProbeError> {
    enumerate_native_devices()?
        .into_iter()
        .find(|native| native.capabilities.is_supported())
        .ok_or(Dx12ProbeError::NoCompatibleAdapter)
}

fn enumerate_native_devices() -> Result<Vec<NativeDevice>, Dx12ProbeError> {
    configure_before_device_creation();
    // SAFETY: Creating a DXGI factory has no pointer preconditions and returns an owned COM handle.
    let factory: IDXGIFactory6 = unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }
        .map_err(|error| backend_error("CreateDXGIFactory2", &error))?;
    let mut devices = Vec::new();
    for index in 0..64 {
        // SAFETY: `index` is bounded and the generic result owns the returned COM reference.
        let adapter: IDXGIAdapter4 = match unsafe {
            factory.EnumAdapterByGpuPreference(index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE)
        } {
            Ok(adapter) => adapter,
            Err(error) if error.code() == DXGI_ERROR_NOT_FOUND => break,
            Err(error) => return Err(backend_error("EnumAdapterByGpuPreference", &error)),
        };
        // SAFETY: `adapter` is a live DXGI adapter and `GetDesc3` returns its description by value.
        let description = unsafe { adapter.GetDesc3() }
            .map_err(|error| backend_error("IDXGIAdapter4::GetDesc3", &error))?;
        if description.Flags.contains(DXGI_ADAPTER_FLAG3_SOFTWARE) {
            continue;
        }

        let mut device = None;
        // SAFETY: The adapter is live and the output slot is valid. Requiring FL 12_0 here filters
        // hardware before any backend resources are constructed.
        if unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &raw mut device) }.is_err()
        {
            continue;
        }
        let device = device.expect("successful D3D12CreateDevice returns a device");
        devices.push(NativeDevice {
            capabilities: read_capabilities(&device, &adapter, &description),
            device,
        });
    }
    if devices.is_empty() {
        Err(Dx12ProbeError::NoHardwareAdapter)
    } else {
        Ok(devices)
    }
}

fn read_capabilities(
    device: &ID3D12Device,
    adapter: &IDXGIAdapter4,
    description: &windows::Win32::Graphics::Dxgi::DXGI_ADAPTER_DESC3,
) -> Dx12Capabilities {
    let options =
        query_feature::<D3D12_FEATURE_DATA_D3D12_OPTIONS>(device, D3D12_FEATURE_D3D12_OPTIONS)
            .unwrap_or_default();
    let wave =
        query_feature::<D3D12_FEATURE_DATA_D3D12_OPTIONS1>(device, D3D12_FEATURE_D3D12_OPTIONS1)
            .unwrap_or_default();
    let architecture =
        query_feature::<D3D12_FEATURE_DATA_ARCHITECTURE1>(device, D3D12_FEATURE_ARCHITECTURE1)
            .unwrap_or_default();
    let mesh =
        query_feature::<D3D12_FEATURE_DATA_D3D12_OPTIONS7>(device, D3D12_FEATURE_D3D12_OPTIONS7)
            .unwrap_or_default();
    let enhanced_barriers =
        query_feature::<D3D12_FEATURE_DATA_D3D12_OPTIONS12>(device, D3D12_FEATURE_D3D12_OPTIONS12)
            .is_some_and(|options| options.EnhancedBarriersSupported.as_bool());
    let options14 =
        query_feature::<D3D12_FEATURE_DATA_D3D12_OPTIONS14>(device, D3D12_FEATURE_D3D12_OPTIONS14)
            .unwrap_or_default();
    let options15 =
        query_feature::<D3D12_FEATURE_DATA_D3D12_OPTIONS15>(device, D3D12_FEATURE_D3D12_OPTIONS15)
            .unwrap_or_default();
    let options16 =
        query_feature::<D3D12_FEATURE_DATA_D3D12_OPTIONS16>(device, D3D12_FEATURE_D3D12_OPTIONS16)
            .unwrap_or_default();

    Dx12Capabilities {
        name: utf16_name(&description.Description),
        vendor_id: description.VendorId,
        device_id: description.DeviceId,
        adapter_luid: (u64::from(description.AdapterLuid.HighPart.cast_unsigned()) << 32)
            | u64::from(description.AdapterLuid.LowPart),
        dedicated_video_memory: description.DedicatedVideoMemory as u64,
        dedicated_system_memory: description.DedicatedSystemMemory as u64,
        shared_system_memory: description.SharedSystemMemory as u64,
        local_memory: query_memory_budget(adapter, DXGI_MEMORY_SEGMENT_GROUP_LOCAL),
        non_local_memory: query_memory_budget(adapter, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL),
        feature_level: query_feature_level(device).0.cast_unsigned(),
        shader_model: query_shader_model(device).0.cast_unsigned(),
        resource_binding_tier: options.ResourceBindingTier.0.cast_unsigned(),
        resource_heap_tier: options.ResourceHeapTier.0.cast_unsigned(),
        wave_ops: wave.WaveOps.as_bool(),
        int64_shader_ops: wave.Int64ShaderOps.as_bool(),
        wave_lane_count_min: wave.WaveLaneCountMin,
        wave_lane_count_max: wave.WaveLaneCountMax,
        total_lane_count: wave.TotalLaneCount,
        mesh_shader_tier: mesh.MeshShaderTier.0.cast_unsigned(),
        uma: architecture.UMA.as_bool(),
        cache_coherent_uma: architecture.CacheCoherentUMA.as_bool(),
        enhanced_barriers,
        writable_msaa_textures: options14.WriteableMSAATexturesSupported.as_bool(),
        independent_front_back_stencil_ref_mask: options14
            .IndependentFrontAndBackStencilRefMaskSupported
            .as_bool(),
        triangle_fan: options15.TriangleFanSupported.as_bool(),
        dynamic_index_buffer_strip_cut: options15.DynamicIndexBufferStripCutSupported.as_bool(),
        dynamic_depth_bias: options16.DynamicDepthBiasSupported.as_bool(),
    }
}

fn query_memory_budget(
    adapter: &IDXGIAdapter4,
    segment: DXGI_MEMORY_SEGMENT_GROUP,
) -> Dx12MemoryBudget {
    let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
    // SAFETY: node zero is the selected single-GPU adapter and `info` is the exact writable DXGI
    // structure for this query. Unsupported/remote segments retain a zero report.
    if unsafe { adapter.QueryVideoMemoryInfo(0, segment, &raw mut info) }.is_err() {
        return Dx12MemoryBudget::default();
    }
    Dx12MemoryBudget {
        budget: info.Budget,
        current_usage: info.CurrentUsage,
        available_for_reservation: info.AvailableForReservation,
        current_reservation: info.CurrentReservation,
    }
}

fn query_feature<T: Default>(device: &ID3D12Device, feature: D3D12_FEATURE) -> Option<T> {
    let mut value = T::default();
    let size = u32::try_from(size_of::<T>()).expect("D3D12 feature structures fit u32");
    // SAFETY: `value` is the exact structure associated with `feature` at each call site.
    unsafe { device.CheckFeatureSupport(feature, (&raw mut value).cast::<c_void>(), size) }.ok()?;
    Some(value)
}

fn query_feature_level(device: &ID3D12Device) -> D3D_FEATURE_LEVEL {
    let requested = [
        D3D_FEATURE_LEVEL_12_2,
        D3D_FEATURE_LEVEL_12_1,
        D3D_FEATURE_LEVEL_12_0,
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
    ];
    let mut data = D3D12_FEATURE_DATA_FEATURE_LEVELS {
        NumFeatureLevels: u32::try_from(requested.len()).expect("feature-level count fits u32"),
        pFeatureLevelsRequested: requested.as_ptr(),
        MaxSupportedFeatureLevel: D3D_FEATURE_LEVEL_11_0,
    };
    let data_size = u32::try_from(size_of::<D3D12_FEATURE_DATA_FEATURE_LEVELS>())
        .expect("D3D12 feature-level structure fits u32");
    // SAFETY: `data` is the exact feature-level structure and its requested-level slice remains
    // live for the duration of the synchronous query.
    if unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_FEATURE_LEVELS,
            (&raw mut data).cast::<c_void>(),
            data_size,
        )
    }
    .is_ok()
    {
        data.MaxSupportedFeatureLevel
    } else {
        D3D_FEATURE_LEVEL_12_0
    }
}

fn query_shader_model(device: &ID3D12Device) -> D3D_SHADER_MODEL {
    let candidates = [
        D3D_SHADER_MODEL_6_9,
        D3D_SHADER_MODEL_6_8,
        D3D_SHADER_MODEL_6_7,
        D3D_SHADER_MODEL_6_6,
        D3D_SHADER_MODEL_6_5,
        D3D_SHADER_MODEL_6_4,
        D3D_SHADER_MODEL_6_3,
        D3D_SHADER_MODEL_6_2,
        D3D_SHADER_MODEL_6_1,
        D3D_SHADER_MODEL_6_0,
        D3D_SHADER_MODEL_5_1,
    ];
    let data_size = u32::try_from(size_of::<D3D12_FEATURE_DATA_SHADER_MODEL>())
        .expect("D3D12 shader-model structure fits u32");
    for candidate in candidates {
        let mut data = D3D12_FEATURE_DATA_SHADER_MODEL {
            HighestShaderModel: candidate,
        };
        // SAFETY: `data` is the exact shader-model feature structure. Retrying older values is
        // required because older D3D12 runtimes reject enum values they do not know.
        if unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_SHADER_MODEL,
                (&raw mut data).cast::<c_void>(),
                data_size,
            )
        }
        .is_ok()
        {
            return data.HighestShaderModel;
        }
    }
    D3D_SHADER_MODEL_5_1
}

fn utf16_name(value: &[u16]) -> String {
    let end = value
        .iter()
        .position(|code_unit| *code_unit == 0)
        .unwrap_or(value.len());
    String::from_utf16_lossy(&value[..end])
}

pub(super) fn backend_error(
    operation: &'static str,
    error: &windows::core::Error,
) -> Dx12ProbeError {
    Dx12ProbeError::Backend {
        operation,
        hresult: error.code().0.cast_unsigned(),
        message: error.message(),
    }
}

pub(super) fn backend_error_with_device(
    operation: &'static str,
    error: &windows::core::Error,
    device: &ID3D12Device,
) -> Dx12ProbeError {
    let report = device_removed_report(device);
    let message = if report == "device_state=available" {
        error.message()
    } else {
        format!("{}; {report}", error.message())
    };
    Dx12ProbeError::Backend {
        operation,
        hresult: error.code().0.cast_unsigned(),
        message,
    }
}
