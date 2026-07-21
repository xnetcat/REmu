// Licensed under the Apache License, Version 2.0.

use remu_gpu::{GpuAdapter, GpuProbe, GpuProbeError, GraphicsApi};

pub struct MetalProbe;

#[cfg(target_os = "macos")]
impl GpuProbe for MetalProbe {
    fn probe_default() -> Result<GpuAdapter, GpuProbeError> {
        let device =
            metal::Device::system_default().ok_or(GpuProbeError::NoDevice(GraphicsApi::Metal))?;

        Ok(GpuAdapter {
            api: GraphicsApi::Metal,
            name: device.name().to_owned(),
        })
    }
}

#[cfg(not(target_os = "macos"))]
impl GpuProbe for MetalProbe {
    fn probe_default() -> Result<GpuAdapter, GpuProbeError> {
        Err(GpuProbeError::Unsupported(GraphicsApi::Metal))
    }
}
