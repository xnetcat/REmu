// SPDX-License-Identifier: GPL-2.0-or-later

//! Gen5 compute dispatch packet and register decoding.

use std::sync::Arc;

use remu_shader::{ComputeSystemRegisters, ShaderEvaluation};
use thiserror::Error;

use crate::{GpuRegister, ShaderSnapshot};

pub const COMPUTE_START_X_REGISTER: u32 = 0x204;

pub const COMPUTE_START_Y_REGISTER: u32 = 0x205;

pub const COMPUTE_START_Z_REGISTER: u32 = 0x206;

pub const COMPUTE_NUM_THREAD_X_REGISTER: u32 = 0x207;

pub const COMPUTE_NUM_THREAD_Y_REGISTER: u32 = 0x208;

pub const COMPUTE_NUM_THREAD_Z_REGISTER: u32 = 0x209;

pub const COMPUTE_PGM_LO_REGISTER: u32 = 0x20c;

pub const COMPUTE_PGM_HI_REGISTER: u32 = 0x20d;

pub const COMPUTE_PGM_RSRC2_REGISTER: u32 = 0x213;

const PM4_TYPE3: u32 = 3;

const PM4_DISPATCH_DIRECT: u32 = 0x15;

const PM4_DISPATCH_INDIRECT: u32 = 0x16;

const DISPATCH_ENABLE: u32 = 1;

const DISPATCH_PARTIAL_THREADGROUP: u32 = 1 << 1;

const DISPATCH_FORCE_START_AT_ZERO: u32 = 1 << 2;

const DISPATCH_USE_THREAD_DIMENSIONS: u32 = 1 << 5;

const DISPATCH_WAVE32: u32 = 1 << 15;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ComputeGrid {
    /// A regular grid whose dimensions are workgroup counts.
    Threadgroups([u32; 3]),
    /// An exact thread grid. Metal must use `dispatchThreads` so the final workgroup is not padded.
    Threads([u64; 3]),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ComputeLaunch {
    pub grid: ComputeGrid,
    pub group_count: [u32; 3],
    pub base_group: [u32; 3],
    pub threads_per_group: [u32; 3],
    pub wave_lane_count: u32,
    pub lds_size_bytes: u32,
    pub initiator: u32,
}

impl ComputeLaunch {
    #[must_use]
    pub const fn threads_per_group_total(self) -> u64 {
        self.threads_per_group[0] as u64
            * self.threads_per_group[1] as u64
            * self.threads_per_group[2] as u64
    }

    /// A dispatch with a zero extent on any axis launches no workgroups.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.group_count[0] == 0 || self.group_count[1] == 0 || self.group_count[2] == 0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ComputeDispatchSource {
    Direct,
    Indirect { dimensions_address: u64 },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ComputeIndirectDimensions {
    BaseOffset(u32),
    Absolute(u64),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ComputeIndirectPacket {
    pub dimensions: ComputeIndirectDimensions,
    pub initiator: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeDispatch {
    pub sequence: u64,
    pub source: ComputeDispatchSource,
    pub launch: ComputeLaunch,
    pub shader_address: u64,
    pub shader_prefix: [u32; 4],
    pub shader: Option<ShaderSnapshot>,
    pub shader_evaluation: Option<Arc<ShaderEvaluation>>,
    pub shader_registers: Vec<GpuRegister>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
pub enum ComputeDispatchDecodeError {
    #[error("compute packet has {actual} dwords, expected {expected}")]
    PacketLength { expected: u32, actual: usize },
    #[error("compute packet declares unsupported length of {0} dwords")]
    DeclaredPacketLength(u32),
    #[error("PM4 header 0x{0:08X} is not DISPATCH_DIRECT")]
    NotDispatchDirect(u32),
    #[error("PM4 header 0x{0:08X} is not DISPATCH_INDIRECT")]
    NotDispatchIndirect(u32),
    #[error("dispatch initiator 0x{0:08X} does not enable compute")]
    Disabled(u32),
    #[error("compute {axis}-dimension end {end} is not after base {base}")]
    EndNotAfterBase { axis: char, end: u64, base: u64 },
    #[error("compute {axis}-dimension partial size {partial} exceeds full size {full}")]
    InvalidPartialSize { axis: char, partial: u32, full: u32 },
    #[error("compute {axis}-dimension thread count overflows")]
    ThreadCountOverflow { axis: char },
}

/// Decodes the complete geometry of a Gen5 direct or resolved-indirect compute dispatch.
///
/// Packet dimensions are end coordinates unless `USE_THREAD_DIMENSIONS` is set. When
/// `FORCE_START_AT_000` is clear, `COMPUTE_START_{X,Y,Z}` supplies the base coordinates. Partial
/// workgroups become an exact thread grid instead of being rounded up with extra invocations.
pub fn decode_compute_launch(
    dimensions: [u32; 3],
    initiator: u32,
    shader_registers: &[GpuRegister],
) -> Result<ComputeLaunch, ComputeDispatchDecodeError> {
    if initiator & DISPATCH_ENABLE == 0 {
        return Err(ComputeDispatchDecodeError::Disabled(initiator));
    }
    let register = |offset| {
        shader_registers
            .iter()
            .find(|register| register.offset == offset)
            .map_or(0, |register| register.value)
    };
    let thread_registers = [
        register(COMPUTE_NUM_THREAD_X_REGISTER),
        register(COMPUTE_NUM_THREAD_Y_REGISTER),
        register(COMPUTE_NUM_THREAD_Z_REGISTER),
    ];
    let threads_per_group = thread_registers.map(|value| (value & 0xffff).max(1));
    let base_group = if initiator & DISPATCH_FORCE_START_AT_ZERO != 0 {
        [0; 3]
    } else {
        [
            register(COMPUTE_START_X_REGISTER),
            register(COMPUTE_START_Y_REGISTER),
            register(COMPUTE_START_Z_REGISTER),
        ]
    };
    let axes = ['x', 'y', 'z'];
    let mut group_count = [0_u32; 3];
    let grid = if initiator & DISPATCH_USE_THREAD_DIMENSIONS != 0 {
        let mut threads = [0_u64; 3];
        for axis in 0..3 {
            let start_thread = u64::from(base_group[axis])
                .checked_mul(u64::from(threads_per_group[axis]))
                .ok_or(ComputeDispatchDecodeError::ThreadCountOverflow { axis: axes[axis] })?;
            let end_thread = u64::from(dimensions[axis]);
            if end_thread < start_thread {
                return Err(ComputeDispatchDecodeError::EndNotAfterBase {
                    axis: axes[axis],
                    end: end_thread,
                    base: start_thread,
                });
            }
            threads[axis] = end_thread - start_thread;
            let groups = threads[axis].div_ceil(u64::from(threads_per_group[axis]));
            group_count[axis] = u32::try_from(groups).map_err(|_| {
                ComputeDispatchDecodeError::ThreadCountOverflow { axis: axes[axis] }
            })?;
        }
        ComputeGrid::Threads(threads)
    } else {
        for axis in 0..3 {
            if dimensions[axis] < base_group[axis] {
                return Err(ComputeDispatchDecodeError::EndNotAfterBase {
                    axis: axes[axis],
                    end: u64::from(dimensions[axis]),
                    base: u64::from(base_group[axis]),
                });
            }
            group_count[axis] = dimensions[axis] - base_group[axis];
        }
        if initiator & DISPATCH_PARTIAL_THREADGROUP == 0 || group_count.contains(&0) {
            ComputeGrid::Threadgroups(group_count)
        } else {
            let mut threads = [0_u64; 3];
            for axis in 0..3 {
                let partial = thread_registers[axis] >> 16;
                let full = threads_per_group[axis];
                if partial == 0 || partial > full {
                    return Err(ComputeDispatchDecodeError::InvalidPartialSize {
                        axis: axes[axis],
                        partial,
                        full,
                    });
                }
                threads[axis] = u64::from(group_count[axis] - 1)
                    .checked_mul(u64::from(full))
                    .and_then(|prefix| prefix.checked_add(u64::from(partial)))
                    .ok_or(ComputeDispatchDecodeError::ThreadCountOverflow { axis: axes[axis] })?;
            }
            ComputeGrid::Threads(threads)
        }
    };
    let resource2 = register(COMPUTE_PGM_RSRC2_REGISTER);
    Ok(ComputeLaunch {
        grid,
        group_count,
        base_group,
        threads_per_group,
        wave_lane_count: if initiator & DISPATCH_WAVE32 != 0 {
            32
        } else {
            64
        },
        lds_size_bytes: ((resource2 >> 15) & 0x1ff) * 512,
        initiator,
    })
}

/// Decodes one complete type-3 `DISPATCH_DIRECT` packet and its active compute registers.
pub fn decode_compute_dispatch_direct(
    packet: &[u32],
    shader_registers: &[GpuRegister],
) -> Result<ComputeLaunch, ComputeDispatchDecodeError> {
    let Some(&header) = packet.first() else {
        return Err(ComputeDispatchDecodeError::PacketLength {
            expected: 5,
            actual: 0,
        });
    };
    let declared = ((header >> 16) & 0x3fff) + 2;
    if declared != 5 {
        return Err(ComputeDispatchDecodeError::DeclaredPacketLength(declared));
    }
    if packet.len() != 5 {
        return Err(ComputeDispatchDecodeError::PacketLength {
            expected: 5,
            actual: packet.len(),
        });
    }
    if header >> 30 != PM4_TYPE3 || (header >> 8) & 0xff != PM4_DISPATCH_DIRECT {
        return Err(ComputeDispatchDecodeError::NotDispatchDirect(header));
    }
    decode_compute_launch(
        [packet[1], packet[2], packet[3]],
        packet[4],
        shader_registers,
    )
}

/// Decodes either observed Gen5 `DISPATCH_INDIRECT` packet layout without dereferencing guest
/// memory. Three-dword packets carry an offset from the active indirect-argument base; four-dword
/// packets carry an absolute 64-bit dimensions address.
pub fn decode_compute_dispatch_indirect(
    packet: &[u32],
) -> Result<ComputeIndirectPacket, ComputeDispatchDecodeError> {
    let Some(&header) = packet.first() else {
        return Err(ComputeDispatchDecodeError::PacketLength {
            expected: 3,
            actual: 0,
        });
    };
    if header >> 30 != PM4_TYPE3 || (header >> 8) & 0xff != PM4_DISPATCH_INDIRECT {
        return Err(ComputeDispatchDecodeError::NotDispatchIndirect(header));
    }
    let declared = ((header >> 16) & 0x3fff) + 2;
    if !matches!(declared, 3 | 4) {
        return Err(ComputeDispatchDecodeError::DeclaredPacketLength(declared));
    }
    if packet.len() != usize::try_from(declared).unwrap_or(usize::MAX) {
        return Err(ComputeDispatchDecodeError::PacketLength {
            expected: declared,
            actual: packet.len(),
        });
    }
    if declared == 3 {
        Ok(ComputeIndirectPacket {
            dimensions: ComputeIndirectDimensions::BaseOffset(packet[1]),
            initiator: packet[2],
        })
    } else {
        Ok(ComputeIndirectPacket {
            dimensions: ComputeIndirectDimensions::Absolute(
                u64::from(packet[1]) | (u64::from(packet[2]) << 32),
            ),
            initiator: packet[3],
        })
    }
}

#[must_use]
pub fn compute_program_address(shader_registers: &[GpuRegister]) -> Option<u64> {
    let low = shader_registers
        .iter()
        .find(|register| register.offset == COMPUTE_PGM_LO_REGISTER)?
        .value;
    let high = shader_registers
        .iter()
        .find(|register| register.offset == COMPUTE_PGM_HI_REGISTER)?
        .value;
    // GFX10 COMPUTE_PGM_HI carries only address bits 47:40. Reserved high bits are not part of
    // the virtual address (Kyty's command processor applies the same 8-bit mask).
    Some((u64::from(high & 0xff) << 40) | (u64::from(low) << 8))
}

#[must_use]
pub fn decode_compute_system_registers(shader_registers: &[GpuRegister]) -> ComputeSystemRegisters {
    let resource2 = shader_registers
        .iter()
        .find(|register| register.offset == COMPUTE_PGM_RSRC2_REGISTER)
        .map_or(0, |register| register.value);
    let mut next_register = (resource2 >> 1) & 0x1f;
    let mut consume_enabled = |mask| {
        if resource2 & mask == 0 {
            return None;
        }
        let register = next_register;
        next_register += 1;
        Some(register)
    };
    ComputeSystemRegisters {
        work_group_x: consume_enabled(1 << 7),
        work_group_y: consume_enabled(1 << 8),
        work_group_z: consume_enabled(1 << 9),
        thread_group_size: consume_enabled(1 << 10),
    }
}
