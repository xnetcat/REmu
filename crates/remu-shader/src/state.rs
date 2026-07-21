// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::sync::Arc;

use crate::{ComputeSystemRegisters, Program, ShaderMetadata, ShaderState};

const PS_USER_DATA_REGISTER: u32 = 0x00c;
const VS_USER_DATA_REGISTER: u32 = 0x04c;
const GS_USER_DATA_REGISTER: u32 = 0x08c;
const ES_USER_DATA_REGISTER: u32 = 0x0cc;
const COMPUTE_USER_DATA_REGISTER: u32 = 0x240;
const COMPUTE_RSRC2_REGISTER: u32 = 0x213;
const MAXIMUM_HARDWARE_USER_SGPRS: u32 = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ShaderStateError {
    #[error("invalid shader user-data base register 0x{0:X}")]
    InvalidUserDataBase(u32),
    #[error("missing shader resource-2 register 0x{register:X}")]
    MissingResource2 { register: u32 },
    #[error("shader declares invalid hardware user-SGPR count {count}")]
    InvalidUserSgprCount { count: u32 },
}

/// Builds the per-draw scalar input state from a backend-neutral register lookup. The closure is
/// intentionally abstract so command processors can use sparse maps without coupling this crate
/// to a particular packet or backend type.
pub fn build_shader_state(
    program: Arc<Program>,
    metadata: Option<ShaderMetadata>,
    user_data_base_register: u32,
    user_data_scalar_register_base: u32,
    compute_system_registers: Option<ComputeSystemRegisters>,
    mut register_value: impl FnMut(u32) -> Option<u32>,
) -> Result<ShaderState, ShaderStateError> {
    if !matches!(
        user_data_base_register,
        PS_USER_DATA_REGISTER
            | VS_USER_DATA_REGISTER
            | GS_USER_DATA_REGISTER
            | ES_USER_DATA_REGISTER
            | COMPUTE_USER_DATA_REGISTER
    ) {
        return Err(ShaderStateError::InvalidUserDataBase(
            user_data_base_register,
        ));
    }
    let resource2_register = if user_data_base_register == COMPUTE_USER_DATA_REGISTER {
        COMPUTE_RSRC2_REGISTER
    } else {
        user_data_base_register - 1
    };
    let resource2 =
        register_value(resource2_register).ok_or(ShaderStateError::MissingResource2 {
            register: resource2_register,
        })?;
    let mut count = (resource2 >> 1) & 0x1f;
    if matches!(
        user_data_base_register,
        PS_USER_DATA_REGISTER | VS_USER_DATA_REGISTER | GS_USER_DATA_REGISTER
    ) && resource2 & (1 << 27) != 0
    {
        count |= 0x20;
    }
    if count > MAXIMUM_HARDWARE_USER_SGPRS {
        return Err(ShaderStateError::InvalidUserSgprCount { count });
    }
    let user_data = (0..count)
        .map(|index| register_value(user_data_base_register + index).unwrap_or_default())
        .collect();
    Ok(ShaderState {
        program,
        user_data,
        metadata,
        compute_system_registers,
        user_data_scalar_register_base,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

    #[test]
    fn decodes_stage_specific_user_sgpr_count_bits() {
        let program = Arc::new(Program::new(0x1000, Vec::<Instruction>::new()));
        let state = build_shader_state(program, None, PS_USER_DATA_REGISTER, 0, None, |register| {
            match register {
                0x00b => Some((3 << 1) | (1 << 27)),
                _ => Some(register),
            }
        })
        .expect("pixel user data");
        assert_eq!(state.user_data.len(), 35);
        assert_eq!(state.user_data[0], PS_USER_DATA_REGISTER);
        assert_eq!(state.user_data[34], PS_USER_DATA_REGISTER + 34);
    }

    #[test]
    fn es_does_not_interpret_the_stage_specific_msb() {
        let program = Arc::new(Program::new(0x1000, Vec::<Instruction>::new()));
        let state = build_shader_state(program, None, ES_USER_DATA_REGISTER, 0, None, |register| {
            (register == 0x0cb).then_some((7 << 1) | (1 << 27))
        })
        .expect("export user data");
        assert_eq!(state.user_data.len(), 7);
    }
}
