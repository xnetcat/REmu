use std::io::Write;
use std::path::Path;
#[cfg(target_arch = "x86_64")]
use std::path::PathBuf;

use crate::{
    ProbeDispatchState, SaveDataMount, SaveDataTransaction, low_i32, read_guest_u32,
    read_guest_u64, sce_kernel_error, write_guest_u32, write_guest_u64,
};

#[cfg(target_arch = "x86_64")]
pub(crate) fn save_data_root(game_root: &Path) -> PathBuf {
    let base = std::env::var_os("REMU_SAVEDATA_DIR")
        .map_or_else(|| std::env::temp_dir().join("remu-savedata"), PathBuf::from);
    let title = game_root
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_suffix("-app0").or(Some(name)))
        .unwrap_or("unknown-title");
    base.join(sanitize_save_segment(title))
}

pub(crate) fn sanitize_save_segment(value: &str) -> String {
    let value = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if value.trim_matches('_').is_empty() {
        "default".to_owned()
    } else {
        value
    }
}

const SAVE_DATA_ERROR_PARAMETER: u32 = 0x809f_0000;
const SAVE_DATA_ERROR_NOT_INITIALIZED: u32 = 0x809f_0001;
const SAVE_DATA_ERROR_OUT_OF_MEMORY: u32 = 0x809f_0002;
const SAVE_DATA_ERROR_BUSY: u32 = 0x809f_0003;
const SAVE_DATA_ERROR_NOT_MOUNTED: u32 = 0x809f_0004;
pub(crate) const SAVE_DATA_ERROR_EXISTS: u32 = 0x809f_0007;
pub(crate) const SAVE_DATA_ERROR_NOT_FOUND: u32 = 0x809f_0008;
const SAVE_DATA_ERROR_INTERNAL: u32 = 0x809f_000b;
const SAVE_DATA_ERROR_MOUNT_FULL: u32 = 0x809f_000c;
const SAVE_DATA_ERROR_BAD_MOUNTED: u32 = 0x809f_000d;
const SAVE_DATA_DIR_NAME_SIZE: usize = 32;
pub(crate) const SAVE_DATA_PARAM_SIZE: usize = 0x530;
const SAVE_DATA_SEARCH_INFO_SIZE: usize = 0x30;
const SAVE_DATA_BLOCK_SIZE: u64 = 32 * 1024;
const SAVE_DATA_MIN_BLOCKS: u64 = 96;
const SAVE_DATA_MAX_ICON_SIZE: u64 = 16 * 1024 * 1024;
const SAVE_DATA_MOUNT_MODE_READ_ONLY: u32 = 1 << 0;
const SAVE_DATA_MOUNT_MODE_READ_WRITE: u32 = 1 << 1;
const SAVE_DATA_MOUNT_MODE_CREATE: u32 = 1 << 2;
const SAVE_DATA_MOUNT_MODE_CREATE_IF_MISSING: u32 = 1 << 5;

#[derive(Debug)]
pub(crate) struct SaveDataSearchEntry {
    name: String,
    param: [u8; SAVE_DATA_PARAM_SIZE],
    blocks: u64,
    free_blocks: u64,
    modified_seconds: i64,
    user_param: u32,
}

pub(crate) fn save_data_initialize(state: &mut ProbeDispatchState) -> u64 {
    match std::fs::create_dir_all(&state.save_root) {
        Ok(()) => {
            state.save_data_initialized = true;
            eprintln!(
                "[REMu][SAVEDATA] initialized root={:?}",
                state.save_root.display()
            );
            0
        }
        Err(error) => {
            eprintln!("[REMu][SAVEDATA] initialize failed: {error}");
            sce_kernel_error(SAVE_DATA_ERROR_INTERNAL)
        }
    }
}

pub(crate) fn save_data_dir_name_search(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if !state.save_data_initialized {
        return sce_kernel_error(SAVE_DATA_ERROR_NOT_INITIALIZED);
    }
    let (condition, result) = (arguments[0], arguments[1]);
    if condition == 0 || result == 0 {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }

    let user_id = low_i32(u64::from(read_guest_u32(condition)));
    let title_id_address = read_guest_u64(condition + 0x08);
    let pattern_address = read_guest_u64(condition + 0x10);
    let sort_key = read_guest_u32(condition + 0x18);
    let sort_order = read_guest_u32(condition + 0x1c);
    if user_id < 0 || !matches!(sort_key, 0 | 1 | 2 | 3 | 5) || sort_order > 1 {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }
    let pattern = if pattern_address == 0 {
        String::new()
    } else {
        let Some(pattern) = read_guest_fixed_ascii(pattern_address, SAVE_DATA_DIR_NAME_SIZE) else {
            return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
        };
        pattern
    };
    let title_root = if title_id_address == 0 {
        state.save_root.clone()
    } else {
        let Some(title_id) = read_guest_fixed_ascii(title_id_address, 10) else {
            return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
        };
        if title_id.is_empty() {
            return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
        }
        state
            .save_root
            .parent()
            .unwrap_or(&state.save_root)
            .join(sanitize_save_segment(&title_id))
    };

    let mut entries = match enumerate_save_data_entries(&title_root, &pattern) {
        Ok(entries) => entries,
        Err(error) => {
            eprintln!(
                "[REMu][SAVEDATA] search failed root={:?}: {error}",
                title_root.display()
            );
            return sce_kernel_error(SAVE_DATA_ERROR_INTERNAL);
        }
    };
    sort_save_data_entries(&mut entries, sort_key, sort_order);

    let dir_names_address = read_guest_u64(result + 0x08);
    let dir_names_capacity = read_guest_u32(result + 0x10) as usize;
    let params_address = read_guest_u64(result + 0x18);
    let infos_address = read_guest_u64(result + 0x20);
    let set_count = dir_names_capacity.min(entries.len());
    if set_count != 0 && dir_names_address == 0 {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }
    write_guest_u32(result, u32::try_from(entries.len()).unwrap_or(u32::MAX));
    write_guest_u32(
        result + 0x14,
        u32::try_from(set_count).expect("set count is bounded by a u32 capacity"),
    );
    for (index, entry) in entries.iter().take(set_count).enumerate() {
        let name_address = dir_names_address
            + u64::try_from(index * SAVE_DATA_DIR_NAME_SIZE).expect("save result offset fits u64");
        write_guest_fixed_ascii(name_address, SAVE_DATA_DIR_NAME_SIZE, &entry.name);
        if params_address != 0 {
            let address = params_address
                + u64::try_from(index * SAVE_DATA_PARAM_SIZE).expect("param offset fits u64");
            write_guest_bytes(address, &entry.param);
        }
        if infos_address != 0 {
            let address = infos_address
                + u64::try_from(index * SAVE_DATA_SEARCH_INFO_SIZE)
                    .expect("search info offset fits u64");
            // SAFETY: the ABI supplies one writable 0x30-byte info record per requested result.
            unsafe { std::ptr::write_bytes(address as *mut u8, 0, SAVE_DATA_SEARCH_INFO_SIZE) };
            write_guest_u64(address, entry.blocks);
            write_guest_u64(address + 0x08, entry.free_blocks);
        }
    }
    eprintln!(
        "[REMu][SAVEDATA] search user={user_id} pattern={pattern:?} sort={sort_key}/{sort_order} hits={} set={set_count}",
        entries.len()
    );
    0
}

pub(crate) fn save_data_create_transaction_resource(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    if !state.save_data_initialized {
        return sce_kernel_error(SAVE_DATA_ERROR_NOT_INITIALIZED);
    }
    if arguments[0] == 0 {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }
    let id = state.next_save_transaction_resource.max(1);
    let Some(next) = id.checked_add(1) else {
        return sce_kernel_error(SAVE_DATA_ERROR_OUT_OF_MEMORY);
    };
    state.next_save_transaction_resource = next;
    state.save_transaction_resources.insert(id);
    eprintln!(
        "[REMu][SAVEDATA] transaction resource={id} memory_size={:#x}",
        arguments[0]
    );
    u64::from(id)
}

pub(crate) fn save_data_mount3(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    const MOUNT_RESULT_SIZE: usize = 0x40;
    if !state.save_data_initialized {
        return sce_kernel_error(SAVE_DATA_ERROR_NOT_INITIALIZED);
    }
    let (mount, result) = (arguments[0], arguments[1]);
    if mount == 0 || result == 0 {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }
    let user_id = low_i32(u64::from(read_guest_u32(mount)));
    let dir_name_address = read_guest_u64(mount + 0x08);
    let blocks = read_guest_u64(mount + 0x10);
    let system_blocks = read_guest_u64(mount + 0x18);
    let mount_mode = read_guest_u32(mount + 0x20);
    let mode = read_guest_u32(mount + 0x24);
    let resource = read_guest_u32(mount + 0x28);
    let Some(dir_name) = read_guest_fixed_ascii(dir_name_address, SAVE_DATA_DIR_NAME_SIZE) else {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    };
    if user_id < 0 || dir_name.is_empty() {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }
    if resource != 0 && !state.save_transaction_resources.contains(&resource) {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }

    let slot = state.save_root.join(sanitize_save_segment(&dir_name));
    let existed = slot.is_dir();
    let create = mount_mode & SAVE_DATA_MOUNT_MODE_CREATE != 0;
    let create_if_missing = mount_mode & SAVE_DATA_MOUNT_MODE_CREATE_IF_MISSING != 0;
    if !existed && !create && !create_if_missing {
        return sce_kernel_error(SAVE_DATA_ERROR_NOT_FOUND);
    }
    if existed && create {
        return sce_kernel_error(SAVE_DATA_ERROR_EXISTS);
    }
    if state
        .save_mounts
        .values()
        .any(|mounted| mounted.path == slot)
    {
        return sce_kernel_error(SAVE_DATA_ERROR_BUSY);
    }
    if !existed && let Err(error) = create_save_data_slot(&slot, &dir_name, blocks) {
        eprintln!(
            "[REMu][SAVEDATA] mount creation failed path={:?}: {error}",
            slot.display()
        );
        return sce_kernel_error(SAVE_DATA_ERROR_INTERNAL);
    }

    let Some(mount_index) =
        (0_u8..16).find(|index| !state.save_mounts.contains_key(&format!("/savedata{index}")))
    else {
        return sce_kernel_error(SAVE_DATA_ERROR_MOUNT_FULL);
    };
    let mount_point = format!("/savedata{mount_index}");

    // SAFETY: sceSaveDataMount3 supplies a writable 0x40-byte result structure.
    unsafe {
        std::ptr::write_bytes(result as *mut u8, 0, MOUNT_RESULT_SIZE);
    }
    write_guest_fixed_ascii(result, 16, &mount_point);
    write_guest_u32(result + 0x1c, u32::from(create_if_missing && !existed));
    state.save_mounts.insert(
        mount_point.clone(),
        SaveDataMount {
            path: slot.clone(),
            committed_path: slot.clone(),
            dir_name: dir_name.clone(),
            user_id,
            resource,
            read_only: mount_mode & SAVE_DATA_MOUNT_MODE_READ_ONLY != 0
                && mount_mode & SAVE_DATA_MOUNT_MODE_READ_WRITE == 0,
        },
    );
    eprintln!(
        "[REMu][SAVEDATA] mount3 user={user_id} dir={dir_name:?} blocks={blocks:#x} system_blocks={system_blocks:#x} mount_mode={mount_mode:#x} resource={resource} mode={mode:#x} mount={mount_point:?} host={:?}",
        slot.display()
    );
    0
}

pub(crate) fn save_data_umount2(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if !state.save_data_initialized {
        return sce_kernel_error(SAVE_DATA_ERROR_NOT_INITIALIZED);
    }
    // Gen5's Umount2 places the mount-point pointer in RSI; retain RDI as a compatibility fallback
    // for the older ABI that used the same export name.
    let mount_point_address = if arguments[1] != 0 {
        arguments[1]
    } else {
        arguments[0]
    };
    let Some(mount_point) = read_guest_fixed_ascii(mount_point_address, 16) else {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    };
    let Some(mounted) = state.save_mounts.get(&mount_point) else {
        return sce_kernel_error(SAVE_DATA_ERROR_NOT_FOUND);
    };
    if mounted.resource != 0 && state.prepared_save_resources.contains(&mounted.resource) {
        return sce_kernel_error(SAVE_DATA_ERROR_BUSY);
    }
    state.save_mounts.remove(&mount_point);
    eprintln!("[REMu][SAVEDATA] unmounted {mount_point:?}");
    0
}

pub(crate) fn save_data_prepare(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if !state.save_data_initialized {
        return sce_kernel_error(SAVE_DATA_ERROR_NOT_INITIALIZED);
    }
    let (mount_point_address, prepare_address) = (arguments[0], arguments[1]);
    if mount_point_address == 0 || prepare_address == 0 {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }
    let Some(mount_point) = read_guest_fixed_ascii(mount_point_address, 16) else {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    };
    let resource = read_guest_u32(prepare_address);
    if resource == 0 || !state.save_transaction_resources.contains(&resource) {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }
    let Some(mounted) = state.save_mounts.get(&mount_point) else {
        return sce_kernel_error(SAVE_DATA_ERROR_NOT_MOUNTED);
    };
    if mounted.read_only {
        return sce_kernel_error(SAVE_DATA_ERROR_BAD_MOUNTED);
    }
    if mounted.resource != resource {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }
    if state.prepared_save_resources.contains(&resource) {
        return sce_kernel_error(SAVE_DATA_ERROR_BUSY);
    }

    let committed_path = mounted.committed_path.clone();
    let staging_path = state.save_root.join(format!(
        "sce_transaction_{}_{}",
        std::process::id(),
        resource
    ));
    if staging_path.exists() {
        return sce_kernel_error(SAVE_DATA_ERROR_BUSY);
    }
    if let Err(error) = copy_save_data_tree(&committed_path, &staging_path) {
        eprintln!(
            "[REMu][SAVEDATA] prepare failed resource={resource} staging={:?}: {error}",
            staging_path.display()
        );
        return sce_kernel_error(SAVE_DATA_ERROR_INTERNAL);
    }
    let mounted = state
        .save_mounts
        .get_mut(&mount_point)
        .expect("validated save mount remains live under dispatcher lock");
    mounted.path.clone_from(&staging_path);
    state.prepared_save_resources.insert(resource);
    state.save_transactions.insert(
        resource,
        SaveDataTransaction {
            mount_point: mount_point.clone(),
            committed_path,
            staging_path,
        },
    );
    eprintln!("[REMu][SAVEDATA] prepared mount={mount_point:?} resource={resource}");
    0
}

pub(crate) fn save_data_set_param(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if !state.save_data_initialized {
        return sce_kernel_error(SAVE_DATA_ERROR_NOT_INITIALIZED);
    }
    let (mount_point_address, param_type, param_address, param_size) =
        (arguments[0], arguments[1], arguments[2], arguments[3]);
    if mount_point_address == 0 || param_address == 0 || param_type > 4 {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }
    let Some(mount_point) = read_guest_fixed_ascii(mount_point_address, 16) else {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    };
    let Some(mounted) = state.save_mounts.get(&mount_point) else {
        return sce_kernel_error(SAVE_DATA_ERROR_NOT_MOUNTED);
    };
    if mounted.read_only {
        return sce_kernel_error(SAVE_DATA_ERROR_BAD_MOUNTED);
    }
    let mut param = match read_save_data_param(&mounted.path, &mounted.dir_name) {
        Ok(param) => param,
        Err(error) => {
            eprintln!("[REMu][SAVEDATA] read param failed: {error}");
            return sce_kernel_error(SAVE_DATA_ERROR_INTERNAL);
        }
    };

    match param_type {
        0 => {
            if param_size != SAVE_DATA_PARAM_SIZE as u64 {
                return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
            }
            read_guest_bytes(param_address, &mut param);
        }
        1 => {
            if !replace_save_data_string_field(&mut param[0x00..0x80], param_address, param_size) {
                return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
            }
        }
        2 => {
            if !replace_save_data_string_field(&mut param[0x80..0x100], param_address, param_size) {
                return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
            }
        }
        3 => {
            if !replace_save_data_string_field(&mut param[0x100..0x500], param_address, param_size)
            {
                return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
            }
        }
        4 => {
            if param_size != 4 {
                return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
            }
            param[0x500..0x504].copy_from_slice(&read_guest_u32(param_address).to_le_bytes());
        }
        _ => unreachable!("validated save-data parameter type"),
    }
    param[0x508..0x510].copy_from_slice(&current_unix_seconds().to_le_bytes());
    if let Err(error) = write_save_data_param(&mounted.path, &param) {
        eprintln!("[REMu][SAVEDATA] write param failed: {error}");
        return sce_kernel_error(SAVE_DATA_ERROR_INTERNAL);
    }
    eprintln!(
        "[REMu][SAVEDATA] set_param mount={mount_point:?} type={param_type} size={param_size:#x}"
    );
    0
}

pub(crate) fn save_data_save_icon(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if !state.save_data_initialized {
        return sce_kernel_error(SAVE_DATA_ERROR_NOT_INITIALIZED);
    }
    let (mount_point_address, icon_address) = (arguments[0], arguments[1]);
    if mount_point_address == 0 || icon_address == 0 {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }
    let Some(mount_point) = read_guest_fixed_ascii(mount_point_address, 16) else {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    };
    let Some(mounted) = state.save_mounts.get(&mount_point) else {
        return sce_kernel_error(SAVE_DATA_ERROR_NOT_MOUNTED);
    };
    if mounted.read_only {
        return sce_kernel_error(SAVE_DATA_ERROR_BAD_MOUNTED);
    }
    let buffer_address = read_guest_u64(icon_address);
    let buffer_size = read_guest_u64(icon_address + 0x08);
    let data_size = read_guest_u64(icon_address + 0x10);
    let length = buffer_size.min(data_size);
    if buffer_address == 0 || length > SAVE_DATA_MAX_ICON_SIZE {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }
    let Ok(length) = usize::try_from(length) else {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    };
    let mut icon = vec![0_u8; length];
    read_guest_bytes(buffer_address, &mut icon);
    let icon_path = mounted.path.join("sce_sys").join("icon0.png");
    if let Err(error) = write_host_file_durable(&icon_path, &icon) {
        eprintln!("[REMu][SAVEDATA] write icon failed: {error}");
        return sce_kernel_error(SAVE_DATA_ERROR_INTERNAL);
    }
    eprintln!("[REMu][SAVEDATA] save_icon mount={mount_point:?} bytes={length:#x}");
    0
}

pub(crate) fn save_data_commit(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if !state.save_data_initialized {
        return sce_kernel_error(SAVE_DATA_ERROR_NOT_INITIALIZED);
    }
    let commit_address = arguments[0];
    if commit_address == 0 {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }
    let resource = read_guest_u32(commit_address);
    if !state.save_transaction_resources.contains(&resource) {
        return sce_kernel_error(SAVE_DATA_ERROR_PARAMETER);
    }
    let Some(transaction) = state.save_transactions.get(&resource).cloned() else {
        return sce_kernel_error(SAVE_DATA_ERROR_BUSY);
    };
    if let Err(error) = commit_save_data_transaction(&transaction) {
        eprintln!("[REMu][SAVEDATA] commit failed resource={resource}: {error}");
        return sce_kernel_error(SAVE_DATA_ERROR_INTERNAL);
    }
    let Some(mounted) = state.save_mounts.get_mut(&transaction.mount_point) else {
        return sce_kernel_error(SAVE_DATA_ERROR_NOT_MOUNTED);
    };
    mounted.path.clone_from(&transaction.committed_path);
    state.save_transactions.remove(&resource);
    state.prepared_save_resources.remove(&resource);
    eprintln!("[REMu][SAVEDATA] committed resource={resource}");
    0
}

fn create_save_data_slot(slot: &Path, dir_name: &str, blocks: u64) -> std::io::Result<()> {
    std::fs::create_dir_all(slot)?;
    let param = default_save_data_param(dir_name, current_unix_seconds());
    write_save_data_param(slot, &param)?;
    write_host_file_durable(
        &slot.join("sce_sys").join("blocks.bin"),
        &blocks.to_le_bytes(),
    )
}

pub(crate) fn enumerate_save_data_entries(
    root: &Path,
    pattern: &str,
) -> std::io::Result<Vec<SaveDataSearchEntry>> {
    if !root.is_dir() {
        return Ok(Vec::new());
    }
    let mut entries = Vec::new();
    for item in std::fs::read_dir(root)? {
        let item = item?;
        if !item.file_type()?.is_dir() {
            continue;
        }
        let Some(name) = item.file_name().to_str().map(ToOwned::to_owned) else {
            continue;
        };
        if name.to_ascii_lowercase().starts_with("sce_")
            || name.eq_ignore_ascii_case("temp0")
            || (!pattern.is_empty() && !save_data_pattern_matches(&name, pattern))
        {
            continue;
        }
        let path = item.path();
        let (size, modified_seconds) = save_data_directory_stats(&path)?;
        let used_blocks = size.div_ceil(SAVE_DATA_BLOCK_SIZE);
        let stored_blocks = read_save_data_blocks(&path)?.unwrap_or(SAVE_DATA_MIN_BLOCKS);
        let blocks = stored_blocks.max(used_blocks).max(1);
        let mut param = read_save_data_param(&path, &name)?;
        param[0x508..0x510].copy_from_slice(&modified_seconds.to_le_bytes());
        let user_param = u32::from_le_bytes(
            param[0x500..0x504]
                .try_into()
                .expect("fixed save user-param field"),
        );
        entries.push(SaveDataSearchEntry {
            name,
            param,
            blocks,
            free_blocks: blocks.saturating_sub(used_blocks),
            modified_seconds,
            user_param,
        });
    }
    Ok(entries)
}

fn sort_save_data_entries(entries: &mut [SaveDataSearchEntry], sort_key: u32, sort_order: u32) {
    entries.sort_by(|left, right| {
        let order = match sort_key {
            0 => left
                .name
                .to_ascii_lowercase()
                .cmp(&right.name.to_ascii_lowercase()),
            1 => left.user_param.cmp(&right.user_param),
            2 => left.blocks.cmp(&right.blocks),
            3 => left.modified_seconds.cmp(&right.modified_seconds),
            5 => left.free_blocks.cmp(&right.free_blocks),
            _ => unreachable!("validated save-data sort key"),
        };
        order.then_with(|| {
            left.name
                .to_ascii_lowercase()
                .cmp(&right.name.to_ascii_lowercase())
        })
    });
    if sort_order == 1 {
        entries.reverse();
    }
}

pub(crate) fn save_data_pattern_matches(value: &str, pattern: &str) -> bool {
    let value = value
        .chars()
        .map(|character| character.to_ascii_lowercase())
        .collect::<Vec<_>>();
    let pattern = pattern
        .chars()
        .map(|character| character.to_ascii_lowercase())
        .collect::<Vec<_>>();
    let mut previous = vec![false; value.len() + 1];
    previous[0] = true;
    for token in pattern {
        let mut current = vec![false; value.len() + 1];
        if token == '%' {
            current[0] = previous[0];
            for index in 1..=value.len() {
                current[index] = previous[index] || current[index - 1];
            }
        } else {
            for index in 1..=value.len() {
                current[index] = previous[index - 1] && (token == '_' || token == value[index - 1]);
            }
        }
        previous = current;
    }
    previous[value.len()]
}

fn default_save_data_param(dir_name: &str, modified_seconds: i64) -> [u8; SAVE_DATA_PARAM_SIZE] {
    let mut param = [0_u8; SAVE_DATA_PARAM_SIZE];
    write_ascii_slice(&mut param[0x00..0x80], dir_name);
    param[0x508..0x510].copy_from_slice(&modified_seconds.to_le_bytes());
    param
}

fn read_save_data_param(
    slot: &Path,
    dir_name: &str,
) -> std::io::Result<[u8; SAVE_DATA_PARAM_SIZE]> {
    let path = slot.join("sce_sys").join("param.bin");
    if !path.exists() {
        let modified = std::fs::metadata(slot)
            .and_then(|metadata| metadata.modified())
            .map_or_else(|_| current_unix_seconds(), system_time_unix_seconds);
        return Ok(default_save_data_param(dir_name, modified));
    }
    let bytes = std::fs::read(path)?;
    bytes.try_into().map_err(|bytes: Vec<u8>| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "save parameter has {} bytes, expected {SAVE_DATA_PARAM_SIZE}",
                bytes.len()
            ),
        )
    })
}

fn write_save_data_param(slot: &Path, param: &[u8; SAVE_DATA_PARAM_SIZE]) -> std::io::Result<()> {
    write_host_file_durable(&slot.join("sce_sys").join("param.bin"), param)
}

fn read_save_data_blocks(slot: &Path) -> std::io::Result<Option<u64>> {
    let path = slot.join("sce_sys").join("blocks.bin");
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(path)?;
    let bytes: [u8; 8] = bytes.try_into().map_err(|bytes: Vec<u8>| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("save block metadata has {} bytes, expected 8", bytes.len()),
        )
    })?;
    Ok(Some(u64::from_le_bytes(bytes)))
}

fn replace_save_data_string_field(field: &mut [u8], address: u64, size: u64) -> bool {
    let Ok(size) = usize::try_from(size) else {
        return false;
    };
    if size == 0 || size > field.len() {
        return false;
    }
    let mut source = vec![0_u8; size];
    read_guest_bytes(address, &mut source);
    field.fill(0);
    let source_length = source.iter().position(|byte| *byte == 0).unwrap_or(size);
    let copy_length = source_length.min(field.len().saturating_sub(1));
    field[..copy_length].copy_from_slice(&source[..copy_length]);
    true
}

fn write_ascii_slice(destination: &mut [u8], value: &str) {
    destination.fill(0);
    let copy_length = value.len().min(destination.len().saturating_sub(1));
    for (output, input) in destination[..copy_length].iter_mut().zip(value.as_bytes()) {
        *output = if input.is_ascii() { *input } else { b'?' };
    }
}

fn read_guest_fixed_ascii(address: u64, length: usize) -> Option<String> {
    if address == 0 {
        return None;
    }
    // SAFETY: fixed-width SaveData ABI strings point to readable guest records of `length` bytes.
    let bytes = unsafe { std::slice::from_raw_parts(address as *const u8, length) };
    let end = bytes.iter().position(|byte| *byte == 0).unwrap_or(length);
    if !bytes[..end].iter().all(u8::is_ascii) {
        return None;
    }
    String::from_utf8(bytes[..end].to_vec()).ok()
}

fn write_guest_fixed_ascii(address: u64, length: usize, value: &str) {
    // SAFETY: the SaveData ABI supplies a writable fixed-width result record.
    let destination = unsafe { std::slice::from_raw_parts_mut(address as *mut u8, length) };
    write_ascii_slice(destination, value);
}

pub(crate) fn read_guest_bytes(address: u64, destination: &mut [u8]) {
    // SAFETY: the guest ABI supplies a readable range matching the declared buffer size.
    unsafe {
        std::ptr::copy_nonoverlapping(
            address as *const u8,
            destination.as_mut_ptr(),
            destination.len(),
        );
    }
}

pub(crate) fn write_guest_bytes(address: u64, source: &[u8]) {
    // SAFETY: the guest ABI supplies a writable range matching the declared result capacity.
    unsafe {
        std::ptr::copy_nonoverlapping(source.as_ptr(), address as *mut u8, source.len());
    }
}

fn save_data_directory_stats(root: &Path) -> std::io::Result<(u64, i64)> {
    let root_metadata = std::fs::metadata(root)?;
    let mut total = 0_u64;
    let mut latest = root_metadata
        .modified()
        .map_or_else(|_| current_unix_seconds(), system_time_unix_seconds);
    let mut pending = vec![root.to_owned()];
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(directory)? {
            let entry = entry?;
            let metadata = entry.metadata()?;
            latest = latest.max(
                metadata
                    .modified()
                    .map_or_else(|_| latest, system_time_unix_seconds),
            );
            if metadata.is_dir() {
                pending.push(entry.path());
            } else if metadata.is_file() {
                total = total.saturating_add(metadata.len());
            }
        }
    }
    Ok((total, latest))
}

pub(crate) fn current_unix_seconds() -> i64 {
    system_time_unix_seconds(std::time::SystemTime::now())
}

fn system_time_unix_seconds(time: std::time::SystemTime) -> i64 {
    time.duration_since(std::time::UNIX_EPOCH).map_or_else(
        |error| -i64::try_from(error.duration().as_secs()).unwrap_or(i64::MAX),
        |duration| i64::try_from(duration.as_secs()).unwrap_or(i64::MAX),
    )
}

fn write_host_file_durable(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

fn copy_save_data_tree(source: &Path, destination: &Path) -> std::io::Result<()> {
    std::fs::create_dir(destination)?;
    let mut pending = vec![(source.to_owned(), destination.to_owned())];
    while let Some((source_directory, destination_directory)) = pending.pop() {
        for entry in std::fs::read_dir(source_directory)? {
            let entry = entry?;
            let source_path = entry.path();
            let destination_path = destination_directory.join(entry.file_name());
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                std::fs::create_dir(&destination_path)?;
                pending.push((source_path, destination_path));
            } else if file_type.is_file() {
                std::fs::copy(source_path, destination_path)?;
            }
        }
    }
    Ok(())
}

fn sync_save_data_tree(root: &Path) -> std::io::Result<()> {
    let mut directories = vec![root.to_owned()];
    let mut index = 0;
    while index < directories.len() {
        for entry in std::fs::read_dir(&directories[index])? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                directories.push(entry.path());
            } else if file_type.is_file() {
                std::fs::File::open(entry.path())?.sync_all()?;
            }
        }
        index += 1;
    }
    for directory in directories.into_iter().rev() {
        std::fs::File::open(directory)?.sync_all()?;
    }
    Ok(())
}

fn commit_save_data_transaction(transaction: &SaveDataTransaction) -> std::io::Result<()> {
    sync_save_data_tree(&transaction.staging_path)?;
    let file_name = transaction
        .committed_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("save");
    let backup_path = transaction.committed_path.with_file_name(format!(
        "sce_commit_backup_{}_{}_{}",
        std::process::id(),
        file_name,
        transaction
            .staging_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("transaction")
    ));
    std::fs::rename(&transaction.committed_path, &backup_path)?;
    if let Err(error) = std::fs::rename(&transaction.staging_path, &transaction.committed_path) {
        let _ = std::fs::rename(&backup_path, &transaction.committed_path);
        return Err(error);
    }
    if let Some(parent) = transaction.committed_path.parent() {
        std::fs::File::open(parent)?.sync_all()?;
    }
    std::fs::remove_dir_all(backup_path)
}
