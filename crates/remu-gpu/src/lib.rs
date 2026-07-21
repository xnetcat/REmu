// Licensed under the Apache License, Version 2.0.

use std::fmt;

use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphicsApi {
    Metal,
    Direct3D12,
    Vulkan,
}

impl fmt::Display for GraphicsApi {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Metal => formatter.write_str("Metal"),
            Self::Direct3D12 => formatter.write_str("Direct3D 12"),
            Self::Vulkan => formatter.write_str("Vulkan"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuAdapter {
    pub api: GraphicsApi,
    pub name: String,
}

#[derive(Debug, Error)]
pub enum GpuProbeError {
    #[error("{0} is unavailable on this host")]
    Unsupported(GraphicsApi),
    #[error("no {0} device is available")]
    NoDevice(GraphicsApi),
}

pub trait GpuProbe {
    fn probe_default() -> Result<GpuAdapter, GpuProbeError>;
}
