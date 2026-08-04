// SPDX-License-Identifier: GPL-2.0-or-later

//! Direct3D 12 host capability discovery and, incrementally, the Windows presenter backend.

use remu_gpu::{GpuAdapter, GpuProbe, GpuProbeError, GraphicsApi};
use thiserror::Error;

#[cfg(windows)]
mod backend;
#[cfg(windows)]
mod context;
#[cfg(windows)]
mod diagnostics;
#[cfg(windows)]
mod dxc;
#[cfg(windows)]
mod executor;
#[cfg(windows)]
mod guest_shader;
#[cfg(windows)]
mod pipeline;
#[cfg(windows)]
mod presenter;
#[cfg(windows)]
mod process_memory;
#[cfg(windows)]
mod texture;

#[cfg(windows)]
pub use presenter::{
    XInputControllerReport, probe_xinput_controllers, run_presenter, test_xinput_vibration,
};

/// Selects opt-in D3D12 diagnostics before the first adapter probe or device creation.
///
/// The SDK debug layer may require the optional Windows Graphics Tools feature. DRED is supplied
/// by the D3D12 runtime and has separate auto-breadcrumb and page-fault overhead.
#[cfg(windows)]
pub fn set_diagnostic_options(
    debug_layer: bool,
    gpu_validation: bool,
    dred: bool,
) -> Result<(), &'static str> {
    diagnostics::set_requested_options(diagnostics::DiagnosticOptions {
        debug_layer,
        gpu_validation,
        dred,
    })
}

#[cfg(not(windows))]
pub fn run_presenter(
    _receiver: std::sync::mpsc::Receiver<remu_gpu::GpuEvent>,
) -> Result<(), String> {
    Err("the Direct3D 12 presenter is available only on Windows".to_owned())
}

/// Minimum backend contract for translating the current Gen5/RDNA2 shader IR to HLSL/DXIL.
pub const BASELINE_REQUIREMENTS: &str =
    "feature level 12_0, shader model 6.7, wave operations, resource binding tier 3";

#[derive(Clone, Debug, Eq, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct Dx12Capabilities {
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub adapter_luid: u64,
    pub dedicated_video_memory: u64,
    pub dedicated_system_memory: u64,
    pub shared_system_memory: u64,
    pub local_memory: Dx12MemoryBudget,
    pub non_local_memory: Dx12MemoryBudget,
    pub feature_level: u32,
    pub shader_model: u32,
    pub resource_binding_tier: u32,
    pub resource_heap_tier: u32,
    pub wave_ops: bool,
    pub int64_shader_ops: bool,
    pub wave_lane_count_min: u32,
    pub wave_lane_count_max: u32,
    pub total_lane_count: u32,
    /// `D3D12_MESH_SHADER_TIER`; zero means unsupported.
    pub mesh_shader_tier: u32,
    pub uma: bool,
    pub cache_coherent_uma: bool,
    pub enhanced_barriers: bool,
    pub writable_msaa_textures: bool,
    pub independent_front_back_stencil_ref_mask: bool,
    pub triangle_fan: bool,
    pub dynamic_index_buffer_strip_cut: bool,
    pub dynamic_depth_bias: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Dx12MemoryBudget {
    pub budget: u64,
    pub current_usage: u64,
    pub available_for_reservation: u64,
    pub current_reservation: u64,
}

impl Dx12Capabilities {
    #[must_use]
    pub const fn is_supported(&self) -> bool {
        self.feature_level >= 0xc000
            && self.shader_model >= 0x67
            && self.wave_ops
            && self.resource_binding_tier >= 3
    }

    #[must_use]
    pub fn unsupported_reasons(&self) -> Vec<&'static str> {
        let mut reasons = Vec::new();
        if self.feature_level < 0xc000 {
            reasons.push("Direct3D feature level is below 12_0");
        }
        if self.shader_model < 0x67 {
            reasons.push("shader model is below 6.7");
        }
        if !self.wave_ops {
            reasons.push("HLSL wave operations are unavailable");
        }
        if self.resource_binding_tier < 3 {
            reasons.push("resource binding tier is below 3");
        }
        reasons
    }

    #[must_use]
    pub const fn feature_level_name(&self) -> &'static str {
        match self.feature_level {
            0xc200 => "12_2",
            0xc100 => "12_1",
            0xc000 => "12_0",
            0xb100 => "11_1",
            0xb000 => "11_0",
            _ => "unknown",
        }
    }

    #[must_use]
    pub const fn shader_model_name(&self) -> &'static str {
        match self.shader_model {
            0x69 => "6.9",
            0x68 => "6.8",
            0x67 => "6.7",
            0x66 => "6.6",
            0x65 => "6.5",
            0x64 => "6.4",
            0x63 => "6.3",
            0x62 => "6.2",
            0x61 => "6.1",
            0x60 => "6.0",
            0x51 => "5.1",
            _ => "unknown",
        }
    }
}

#[derive(Debug, Error)]
pub enum Dx12ProbeError {
    #[error("Direct3D 12 is available only on Windows")]
    UnsupportedHost,
    #[error("{operation} failed: {message} (HRESULT 0x{hresult:08x})")]
    Backend {
        operation: &'static str,
        hresult: u32,
        message: String,
    },
    #[error("no hardware Direct3D 12 adapter is available")]
    NoHardwareAdapter,
    #[error("no hardware Direct3D 12 adapter meets the REmu baseline: {BASELINE_REQUIREMENTS}")]
    NoCompatibleAdapter,
    #[error("D3D12 fence {value} timed out waiting for GPU queue completion")]
    FenceTimeout { value: u64 },
    #[error("D3D12 fence signaled before completion: expected {expected}, completed {completed}")]
    FenceIncomplete { expected: u64, completed: u64 },
    #[error("D3D12 fence wait returned unexpected status 0x{status:08x}")]
    FenceWaitFailed { status: u32 },
    #[error("D3D12 self-test readback did not match the uploaded bytes")]
    SelfTestMismatch,
    #[error("DirectX Shader Compiler failed: {0}")]
    ShaderCompiler(String),
    #[error("D3D12 pipeline self-test failed validation: {0}")]
    PipelineSelfTestMismatch(&'static str),
    #[error("D3D12 reported success without returning the requested resource")]
    MissingCreatedResource,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Dx12SelfTestReport {
    pub adapter: String,
    pub copied_bytes: usize,
    pub checksum: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Dx12PipelineSelfTestReport {
    pub adapter: String,
    pub shader_model: String,
    pub colored_pixels: usize,
    pub computed_values: usize,
    pub guest_computed_values: usize,
}

pub struct Dx12Probe;

/// Enumerates hardware adapters in DXGI high-performance preference order.
pub fn probe_adapters() -> Result<Vec<Dx12Capabilities>, Dx12ProbeError> {
    #[cfg(windows)]
    {
        backend::probe_adapters()
    }
    #[cfg(not(windows))]
    {
        Err(Dx12ProbeError::UnsupportedHost)
    }
}

/// Returns the first high-performance adapter meeting the backend baseline.
pub fn probe_default_capabilities() -> Result<Dx12Capabilities, Dx12ProbeError> {
    probe_adapters()?
        .into_iter()
        .find(Dx12Capabilities::is_supported)
        .ok_or(Dx12ProbeError::NoCompatibleAdapter)
}

/// Creates a real device, submits an upload-to-default-to-readback buffer copy, waits on a fence,
/// and validates every returned byte.
pub fn run_headless_self_test() -> Result<Dx12SelfTestReport, Dx12ProbeError> {
    #[cfg(windows)]
    {
        context::run_self_test()
    }
    #[cfg(not(windows))]
    {
        Err(Dx12ProbeError::UnsupportedHost)
    }
}

/// Compiles Shader Model 6 HLSL to DXIL and validates real graphics and compute pipelines with
/// render-target/UAV readback. This exercises substantially more of the backend than a copy test.
pub fn run_pipeline_self_test() -> Result<Dx12PipelineSelfTestReport, Dx12ProbeError> {
    #[cfg(windows)]
    {
        pipeline::run_pipeline_self_test()
    }
    #[cfg(not(windows))]
    {
        Err(Dx12ProbeError::UnsupportedHost)
    }
}

#[cfg(windows)]
impl GpuProbe for Dx12Probe {
    fn probe_default() -> Result<GpuAdapter, GpuProbeError> {
        let adapter = probe_default_capabilities()
            .map_err(|_| GpuProbeError::NoDevice(GraphicsApi::Direct3D12))?;
        Ok(GpuAdapter {
            api: GraphicsApi::Direct3D12,
            name: adapter.name,
        })
    }
}

#[cfg(not(windows))]
impl GpuProbe for Dx12Probe {
    fn probe_default() -> Result<GpuAdapter, GpuProbeError> {
        Err(GpuProbeError::Unsupported(GraphicsApi::Direct3D12))
    }
}

#[cfg(test)]
mod tests {
    use super::{Dx12Capabilities, Dx12MemoryBudget};

    fn baseline() -> Dx12Capabilities {
        Dx12Capabilities {
            name: "test".to_owned(),
            vendor_id: 0,
            device_id: 0,
            adapter_luid: 0,
            dedicated_video_memory: 0,
            dedicated_system_memory: 0,
            shared_system_memory: 0,
            local_memory: Dx12MemoryBudget::default(),
            non_local_memory: Dx12MemoryBudget::default(),
            feature_level: 0xc000,
            shader_model: 0x67,
            resource_binding_tier: 3,
            resource_heap_tier: 1,
            wave_ops: true,
            int64_shader_ops: false,
            wave_lane_count_min: 32,
            wave_lane_count_max: 32,
            total_lane_count: 1,
            mesh_shader_tier: 0,
            uma: false,
            cache_coherent_uma: false,
            enhanced_barriers: false,
            writable_msaa_textures: false,
            independent_front_back_stencil_ref_mask: false,
            triangle_fan: false,
            dynamic_index_buffer_strip_cut: false,
            dynamic_depth_bias: false,
        }
    }

    #[test]
    fn baseline_is_explicit_and_does_not_require_optional_barriers() {
        assert!(baseline().is_supported());
    }

    #[test]
    fn missing_wave_operations_is_reported() {
        let mut capabilities = baseline();
        capabilities.wave_ops = false;
        assert!(!capabilities.is_supported());
        assert_eq!(
            capabilities.unsupported_reasons(),
            ["HLSL wave operations are unavailable"]
        );
    }
}
