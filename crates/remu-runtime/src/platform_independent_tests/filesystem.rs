// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::path::Path;

use std::sync::Mutex;

use super::{
    AMPR_INVALID_FILE_ID, AprResolveOutputs, DIRENT_BLOCK_SIZE, DIRENT_HEADER_SIZE, KERNEL_FILES,
    KernelDirectory, KernelFileState, ProbeDispatchState, SAVE_DATA_ERROR_BUSY,
    SAVE_DATA_ERROR_EXISTS, SAVE_DATA_ERROR_NOT_FOUND, SAVE_DATA_ERROR_RESOURCE_BUSY,
    SAVE_DATA_ERROR_RESOURCE_FULL, SAVE_DATA_ERROR_RESOURCE_INVALID, SAVE_DATA_PARAM_SIZE,
    SCE_KERNEL_ERROR_NOT_FOUND, SaveDataMount, TemporaryTestDirectory, ampr_resolve_file_paths,
    ampr_resolve_paths_core, discover_runtime_module_paths, guest_kernel_fstat,
    guest_kernel_ftruncate, guest_kernel_lseek, guest_kernel_read, guest_posix_file_result,
    join_guest_path, kernel_stat_payload, read_guest_u32, resolve_guest_path,
    resolve_probe_guest_path, sanitize_save_segment, save_data_commit,
    save_data_create_transaction_resource, save_data_delete, save_data_delete_transaction_resource,
    save_data_dir_name_search, save_data_mount3, save_data_pattern_matches, save_data_prepare,
    save_data_save_icon, save_data_set_param, save_data_terminate, save_data_umount2,
    sce_kernel_error, set_current_guest_thread_pointer, write_guest_u32, write_guest_u64,
};

#[test]
fn posix_file_errors_return_minus_one_and_store_guest_errno() {
    let mut tls = [0_u64; 16];
    let thread_pointer = u64::try_from(tls.as_mut_ptr().addr()).unwrap();
    set_current_guest_thread_pointer(thread_pointer);
    assert_eq!(guest_posix_file_result(7), 7);
    assert_eq!(read_guest_u32(thread_pointer + 0x40), 0);
    assert_eq!(
        guest_posix_file_result(crate::sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)),
        u64::MAX
    );
    assert_eq!(
        read_guest_u32(thread_pointer + 0x40),
        libc::ENOENT.cast_unsigned()
    );
    assert_eq!(
        guest_posix_file_result(crate::sce_kernel_error(
            crate::SCE_KERNEL_ERROR_INVALID_ARGUMENT
        )),
        u64::MAX
    );
    assert_eq!(
        read_guest_u32(thread_pointer + 0x40),
        libc::EINVAL.cast_unsigned()
    );
    set_current_guest_thread_pointer(0);
}

#[test]
fn resolves_duplicate_console_path_separators() {
    assert_eq!(
        resolve_guest_path(Path::new("/games/title"), "/app0//shaders/presenter.vert"),
        Some(Path::new("/games/title/shaders/presenter.vert").to_owned())
    );
}

#[test]
fn resolves_persistent_download_mount_without_allowing_escape() {
    let temporary = TemporaryTestDirectory::new("download0-mount");
    let state = ProbeDispatchState {
        save_root: temporary.path.clone(),
        ..ProbeDispatchState::default()
    };
    let root = temporary.path.join("download0");
    std::fs::create_dir_all(&root).unwrap();

    assert_eq!(
        resolve_probe_guest_path(&state, "/download0"),
        Some((root.clone(), true))
    );
    assert_eq!(
        resolve_probe_guest_path(&state, "/download0/shpc/saved/ps5/cache.bin"),
        Some((root.join("shpc/saved/ps5/cache.bin"), true))
    );
    assert_eq!(
        resolve_probe_guest_path(&state, "/download0/../../outside"),
        None
    );
}

#[test]
fn immutable_app_path_cache_reuses_borrowed_keys_without_growing_on_replacement() {
    let mut cache = crate::ImmutableAppPathCache::default();
    let root = Path::new("/games/title");
    cache.insert(
        root,
        "Content/Paks/game.pak",
        root.join("content/paks/game.pak"),
    );

    assert_eq!(
        cache.get(root, "Content/Paks/game.pak"),
        Some(root.join("content/paks/game.pak"))
    );
    assert_eq!(cache.entries, 1);

    cache.insert(
        root,
        "Content/Paks/game.pak",
        root.join("CONTENT/PAKS/game.pak"),
    );
    assert_eq!(
        cache.get(root, "Content/Paks/game.pak"),
        Some(root.join("CONTENT/PAKS/game.pak"))
    );
    assert_eq!(cache.entries, 1);
}

#[test]
fn resolves_guest_paths_that_differ_only_by_case() {
    let root = std::env::temp_dir().join(format!("remu-case-{}", std::process::id()));
    std::fs::create_dir_all(root.join("prx")).unwrap();
    std::fs::write(root.join("prx/akroomverb.prx"), b"module").unwrap();
    // The resolved path must name a file that exists. On a case-insensitive host the request
    // already resolves; on a case-sensitive one only the fallback finds the stored name.
    let resolved = resolve_guest_path(&root, "/app0/prx/AkRoomVerb.prx").unwrap();
    assert!(resolved.is_file(), "unresolved: {}", resolved.display());
    // A genuine miss still reports the name the guest asked for.
    assert_eq!(
        resolve_guest_path(&root, "/app0/prx/Missing.prx"),
        Some(root.join("prx/Missing.prx"))
    );
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn discovers_runtime_modules_in_media_and_app0_prx_layouts() {
    let root = std::env::temp_dir().join(format!("remu-runtime-modules-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    let decrypted_plugins = root.join("decrypted/Media/Plugins/Runtime");
    let media_plugins = root.join("Media/Plugins");
    let app_prx = root.join("prx");
    std::fs::create_dir_all(&decrypted_plugins).unwrap();
    std::fs::create_dir_all(&media_plugins).unwrap();
    std::fs::create_dir_all(&app_prx).unwrap();
    std::fs::write(decrypted_plugins.join("overlay_plugin.sprx"), b"module").unwrap();
    std::fs::write(media_plugins.join("unity_plugin.prx"), b"module").unwrap();
    std::fs::write(app_prx.join("akroomverb.prx"), b"module").unwrap();
    std::fs::write(app_prx.join("akroomverb.prx.esbak"), b"backup").unwrap();

    let modules = discover_runtime_module_paths(&root).unwrap();
    assert_eq!(modules.len(), 3);
    assert!(
        modules
            .iter()
            .any(|path| path.ends_with("overlay_plugin.sprx"))
    );
    assert!(
        modules
            .iter()
            .any(|path| path.ends_with("unity_plugin.prx")),
        "flattened dumps expose plugins directly under Media/Plugins"
    );
    assert!(modules.iter().any(|path| path.ends_with("akroomverb.prx")));

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn resolves_parent_components_inside_the_game_root() {
    assert_eq!(
        resolve_guest_path(
            Path::new("/games/title"),
            "/app0/engine/binaries/ps5/../../../shpc/config/binaryconfig.ini",
        ),
        Some(Path::new("/games/title/shpc/config/binaryconfig.ini").to_owned())
    );
    assert_eq!(
        resolve_guest_path(Path::new("/games/title"), "/app0/shpc/./content/paks"),
        Some(Path::new("/games/title/shpc/content/paks").to_owned())
    );
}

#[test]
fn rejects_game_root_parent_traversal() {
    assert_eq!(
        resolve_guest_path(Path::new("/games/title"), "/app0/../outside.bin"),
        None
    );
}

#[test]
fn builds_the_gen5_kernel_stat_layout() {
    let payload = kernel_stat_payload(1_025, false, b"/app0/data.bin");

    assert_eq!(
        u16::from_le_bytes(payload[8..10].try_into().unwrap()),
        0x81ff
    );
    assert_eq!(u16::from_le_bytes(payload[10..12].try_into().unwrap()), 1);
    assert_eq!(
        i64::from_le_bytes(payload[72..80].try_into().unwrap()),
        1_025
    );
    assert_eq!(i64::from_le_bytes(payload[80..88].try_into().unwrap()), 3);
    assert_eq!(u32::from_le_bytes(payload[88..92].try_into().unwrap()), 512);
}

#[test]
fn sanitizes_save_slot_path_segments() {
    assert_eq!(sanitize_save_segment("SAVEDATA00"), "SAVEDATA00");
    assert_eq!(sanitize_save_segment("../outside"), "___outside");
    assert_eq!(sanitize_save_segment("///"), "default");
}

#[test]
fn matches_save_data_wildcards_case_insensitively() {
    assert!(save_data_pattern_matches("SAVEDATA00", "save%"));
    assert!(save_data_pattern_matches("slot-A", "SLOT__"));
    assert!(save_data_pattern_matches("alpha-beta", "%ha_b%"));
    assert!(!save_data_pattern_matches("slot", "slot_"));
    assert!(!save_data_pattern_matches("other", "save%"));
}

#[test]
fn temporary_app_content_mount_is_not_a_save_data_slot() {
    let temporary = TemporaryTestDirectory::new("savedata-temp-mount");
    let state = ProbeDispatchState {
        save_root: temporary.path.clone(),
        ..ProbeDispatchState::default()
    };
    let mut mount_point = [0_u8; 16];

    assert_eq!(
        crate::app_content_temporary_data_mount(
            [0, mount_point.as_mut_ptr().addr() as u64, 0, 0, 0, 0],
            &state,
        ),
        0
    );
    assert_eq!(&mount_point[..7], b"/temp0\0");
    assert!(temporary.path.join("temp0").is_dir());
    // `/download0` shares the flat save root, so a search must not report it as a save slot.
    std::fs::create_dir_all(temporary.path.join("download0")).unwrap();
    assert_eq!(
        crate::enumerate_save_data_entries(&temporary.path, "")
            .unwrap()
            .len(),
        0
    );
}

#[test]
fn bounds_and_reuses_transaction_resources_without_deleting_live_ones() {
    let temporary = TemporaryTestDirectory::new("savedata-resource-lifecycle");
    let mut state = ProbeDispatchState {
        save_root: temporary.path.clone(),
        save_data_initialized: true,
        next_save_transaction_resource: 1,
        ..ProbeDispatchState::default()
    };

    assert_eq!(
        save_data_create_transaction_resource([0xb_ffff, 0, 0, 0, 0, 0], &mut state),
        sce_kernel_error(0x809f_0000)
    );
    for expected in 1..=16 {
        assert_eq!(
            save_data_create_transaction_resource([0xc_0000, 0, 0, 0, 0, 0], &mut state),
            expected
        );
    }
    assert_eq!(
        save_data_create_transaction_resource([0xc_0000, 0, 0, 0, 0, 0], &mut state),
        sce_kernel_error(SAVE_DATA_ERROR_RESOURCE_FULL)
    );

    state.save_mounts.insert(
        "/savedata0".to_owned(),
        SaveDataMount {
            path: state.save_root.join("slot"),
            committed_path: state.save_root.join("slot"),
            dir_name: "slot".to_owned(),
            user_id: 0x1000_0000,
            resource: 1,
            read_only: false,
        },
    );
    assert_eq!(
        save_data_delete_transaction_resource([1, 0, 0, 0, 0, 0], &mut state),
        sce_kernel_error(SAVE_DATA_ERROR_RESOURCE_BUSY)
    );
    state.save_mounts.clear();
    assert_eq!(
        save_data_delete_transaction_resource([1, 0, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(
        save_data_delete_transaction_resource([1, 0, 0, 0, 0, 0], &mut state),
        sce_kernel_error(SAVE_DATA_ERROR_RESOURCE_INVALID)
    );
    assert_eq!(
        save_data_create_transaction_resource([0xc_0000, 0, 0, 0, 0, 0], &mut state),
        1,
        "the bounded allocator reuses a released resource ID"
    );
}

#[test]
fn deletes_only_unmounted_save_data_slots() {
    let temporary = TemporaryTestDirectory::new("savedata-delete");
    let mut state = ProbeDispatchState {
        save_root: temporary.path.clone(),
        save_data_initialized: true,
        ..ProbeDispatchState::default()
    };
    let slot = state.save_root.join("SAVEDATA00");
    std::fs::create_dir_all(&slot).unwrap();
    std::fs::write(slot.join("save.bin"), b"data").unwrap();

    let mut dir_name = [0_u8; 32];
    dir_name[..10].copy_from_slice(b"SAVEDATA00");
    let mut delete = [0_u8; 0x40];
    let delete_address = delete.as_mut_ptr().addr() as u64;
    write_guest_u32(delete_address, 0x1000_0000);
    write_guest_u64(delete_address + 0x10, dir_name.as_ptr().addr() as u64);

    state.save_mounts.insert(
        "/savedata0".to_owned(),
        SaveDataMount {
            path: slot.clone(),
            committed_path: slot.clone(),
            dir_name: "SAVEDATA00".to_owned(),
            user_id: 0x1000_0000,
            resource: u32::MAX,
            read_only: true,
        },
    );
    assert_eq!(
        save_data_delete([delete_address, 0, 0, 0, 0, 0], &mut state),
        sce_kernel_error(SAVE_DATA_ERROR_BUSY)
    );
    state.save_mounts.clear();
    assert_eq!(
        save_data_delete([delete_address, 0, 0, 0, 0, 0], &mut state),
        0
    );
    assert!(!slot.exists());
    assert_eq!(
        save_data_delete([delete_address, 0, 0, 0, 0, 0], &mut state),
        sce_kernel_error(SAVE_DATA_ERROR_NOT_FOUND)
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn persists_and_searches_transactional_save_data() {
    let temporary = TemporaryTestDirectory::new("savedata-transaction");
    let mut state = ProbeDispatchState {
        save_root: temporary.path.clone(),
        save_data_initialized: true,
        next_save_transaction_resource: 1,
        ..ProbeDispatchState::default()
    };
    std::fs::create_dir_all(&state.save_root).unwrap();

    let resource = save_data_create_transaction_resource([0xc_0000, 0, 0, 0, 0, 0], &mut state);
    assert_eq!(resource, 1);
    let resource = u32::try_from(resource).unwrap();
    let mut dir_name = [0_u8; 32];
    dir_name[..10].copy_from_slice(b"SAVEDATA00");
    let mut mount = [0_u8; 0x30];
    let mut mount_result = [0_u8; 0x40];
    let mount_address = mount.as_mut_ptr().addr() as u64;
    write_guest_u32(mount_address, 0x1000_0000);
    write_guest_u64(mount_address + 0x08, dir_name.as_ptr().addr() as u64);
    write_guest_u64(mount_address + 0x10, 56);
    write_guest_u32(mount_address + 0x20, 1 << 5);
    write_guest_u32(mount_address + 0x28, resource);
    let result_address = mount_result.as_mut_ptr().addr() as u64;
    assert_eq!(
        save_data_mount3([mount_address, result_address, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(&mount_result[..11], b"/savedata0\0");
    assert_eq!(
        u32::from_le_bytes(mount_result[0x1c..0x20].try_into().unwrap()),
        1
    );
    assert_eq!(
        resolve_probe_guest_path(&state, "/savedata0/save.bin")
            .unwrap()
            .1,
        false,
        "a transaction mount is not writable before Prepare"
    );

    let mut prepare = [0_u8; 0x28];
    prepare[..4].copy_from_slice(&resource.to_le_bytes());
    assert_eq!(
        save_data_prepare(
            [result_address, prepare.as_ptr().addr() as u64, 0, 0, 0, 0,],
            &mut state,
        ),
        0
    );
    assert!(
        resolve_probe_guest_path(&state, "/savedata0/save.bin")
            .unwrap()
            .1,
        "Prepare exposes the staging tree for writes"
    );

    let mut param = [0_u8; SAVE_DATA_PARAM_SIZE];
    param[..14].copy_from_slice(b"Save Metadata\0");
    param[0x500..0x504].copy_from_slice(&42_u32.to_le_bytes());
    assert_eq!(
        save_data_set_param(
            [
                result_address,
                0,
                param.as_ptr().addr() as u64,
                SAVE_DATA_PARAM_SIZE as u64,
                0,
                0,
            ],
            &mut state,
        ),
        0
    );
    let icon_bytes = b"\x89PNG\r\n\x1a\ntransaction-icon";
    let icon = [
        icon_bytes.as_ptr().addr() as u64,
        icon_bytes.len() as u64,
        icon_bytes.len() as u64,
    ];
    assert_eq!(
        save_data_save_icon(
            [result_address, icon.as_ptr().addr() as u64, 0, 0, 0, 0],
            &state,
        ),
        0
    );

    let committed_param = state.save_root.join("SAVEDATA00/sce_sys/param.bin");
    assert_ne!(
        &std::fs::read(&committed_param).unwrap()[..14],
        b"Save Metadata\0"
    );
    let mut commit = [0_u8; 0x28];
    commit[..4].copy_from_slice(&resource.to_le_bytes());
    assert_eq!(
        save_data_commit([commit.as_ptr().addr() as u64, 0, 0, 0, 0, 0], &mut state,),
        0
    );
    assert_eq!(
        resolve_probe_guest_path(&state, "/savedata0/save.bin")
            .unwrap()
            .1,
        false,
        "Commit closes the writable transaction window"
    );
    assert_eq!(
        &std::fs::read(&committed_param).unwrap()[..14],
        b"Save Metadata\0"
    );
    assert_eq!(
        std::fs::read(state.save_root.join("SAVEDATA00/sce_sys/icon0.png")).unwrap(),
        icon_bytes
    );

    let mut search_condition = [0_u8; 0x40];
    let mut search_result = [0_u8; 0x30];
    let pattern = b"save%\0";
    let mut found_names = [0_u8; 64];
    let mut found_params = vec![0_u8; SAVE_DATA_PARAM_SIZE * 2];
    let mut found_infos = [0_u8; 0x60];
    let condition_address = search_condition.as_mut_ptr().addr() as u64;
    let search_result_address = search_result.as_mut_ptr().addr() as u64;
    write_guest_u32(condition_address, 0x1000_0000);
    write_guest_u64(condition_address + 0x10, pattern.as_ptr().addr() as u64);
    write_guest_u32(condition_address + 0x18, 1);
    write_guest_u64(
        search_result_address + 0x08,
        found_names.as_mut_ptr().addr() as u64,
    );
    write_guest_u32(search_result_address + 0x10, 2);
    write_guest_u64(
        search_result_address + 0x18,
        found_params.as_mut_ptr().addr() as u64,
    );
    write_guest_u64(
        search_result_address + 0x20,
        found_infos.as_mut_ptr().addr() as u64,
    );
    assert_eq!(
        save_data_dir_name_search(
            [condition_address, search_result_address, 0, 0, 0, 0],
            &state,
        ),
        0
    );
    assert_eq!(
        u32::from_le_bytes(search_result[..4].try_into().unwrap()),
        1
    );
    assert_eq!(
        u32::from_le_bytes(search_result[0x14..0x18].try_into().unwrap()),
        1
    );
    assert_eq!(&found_names[..11], b"SAVEDATA00\0");
    assert_eq!(&found_params[..14], b"Save Metadata\0");
    assert_eq!(u64::from_le_bytes(found_infos[..8].try_into().unwrap()), 56);

    assert_eq!(
        save_data_umount2([0, result_address, 0, 0, 0, 0], &mut state),
        0
    );
    mount_result.fill(0);
    assert_eq!(
        save_data_mount3([mount_address, result_address, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(
        u32::from_le_bytes(mount_result[0x1c..0x20].try_into().unwrap()),
        0
    );
    assert_eq!(
        save_data_prepare(
            [result_address, prepare.as_ptr().addr() as u64, 0, 0, 0, 0,],
            &mut state,
        ),
        0
    );
    let committed_by_umount = state.save_root.join("SAVEDATA00/committed-by-umount.bin");
    let staging_file = resolve_probe_guest_path(&state, "/savedata0/committed-by-umount.bin")
        .unwrap()
        .0;
    std::fs::write(staging_file, b"committed").unwrap();
    assert_eq!(
        save_data_umount2([1, result_address, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(std::fs::read(committed_by_umount).unwrap(), b"committed");
    assert!(!state.save_mounts.contains_key("/savedata0"));
    write_guest_u32(mount_address + 0x20, 1 << 2);
    assert_eq!(
        save_data_mount3([mount_address, result_address, 0, 0, 0, 0], &mut state),
        sce_kernel_error(SAVE_DATA_ERROR_EXISTS)
    );
    dir_name.fill(0);
    dir_name[..7].copy_from_slice(b"MISSING");
    write_guest_u32(mount_address + 0x20, 1 << 1);
    assert_eq!(
        save_data_mount3([mount_address, result_address, 0, 0, 0, 0], &mut state),
        sce_kernel_error(SAVE_DATA_ERROR_NOT_FOUND)
    );
    assert_eq!(save_data_terminate(&mut state), 0);
    assert!(!state.save_data_initialized);
    assert!(state.save_transaction_resources.is_empty());
}

#[test]
fn immutable_app_directory_index_resolves_case_once_for_sibling_paths() {
    let temporary = TemporaryTestDirectory::new("app-directory-index");
    let content = temporary.path.join("Content");
    std::fs::create_dir(&content).unwrap();
    std::fs::write(content.join("First.PAK"), []).unwrap();
    std::fs::write(content.join("Second.UCAS"), []).unwrap();

    let cache = std::sync::RwLock::new(crate::ImmutableAppPathCache::default());
    assert_eq!(
        crate::resolve_immutable_app_path(&temporary.path, "content/first.pak", &cache),
        Some(content.join("First.PAK"))
    );
    assert_eq!(
        crate::resolve_immutable_app_path(&temporary.path, "CONTENT/second.ucas", &cache),
        Some(content.join("Second.UCAS"))
    );

    let cache = cache
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert!(cache.directories.contains_key(&temporary.path));
    assert!(cache.directories.contains_key(&content));
    assert_eq!(cache.directories.len(), 2);
}

#[test]
fn immutable_app_directory_index_proves_absent_subtrees_without_descending() {
    let temporary = TemporaryTestDirectory::new("app-directory-absence");
    std::fs::create_dir(temporary.path.join("Content")).unwrap();
    let cache = std::sync::RwLock::new(crate::ImmutableAppPathCache::default());

    let first = crate::resolve_immutable_app_path_with_presence(
        &temporary.path,
        "Missing/Config/first.ini",
        &cache,
    )
    .unwrap();
    let second = crate::resolve_immutable_app_path_with_presence(
        &temporary.path,
        "Missing/Config/second.ini",
        &cache,
    )
    .unwrap();

    assert!(first.known_absent);
    assert!(second.known_absent);
    assert_eq!(first.path, temporary.path.join("Missing/Config/first.ini"));
    assert_eq!(
        second.path,
        temporary.path.join("Missing/Config/second.ini")
    );
    let cache = cache
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert_eq!(cache.directories.len(), 1);
    assert!(cache.directories.contains_key(&temporary.path));
}

#[test]
fn immutable_app_directory_read_failures_do_not_claim_absence() {
    let temporary = TemporaryTestDirectory::new("app-directory-unreadable");
    let missing_root = temporary.path.join("not-mounted");
    let cache = std::sync::RwLock::new(crate::ImmutableAppPathCache::default());

    let resolved = crate::resolve_immutable_app_path_with_presence(
        &missing_root,
        "Content/config.ini",
        &cache,
    )
    .unwrap();

    assert!(!resolved.known_absent);
    assert_eq!(resolved.path, missing_root.join("Content/config.ini"));
    assert!(
        cache
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .directories
            .is_empty()
    );
}

#[test]
fn implements_seek_read_stat_and_truncate_on_one_guest_descriptor() {
    let temporary = TemporaryTestDirectory::new("file-abi");
    let path = temporary.path.join("payload.bin");
    std::fs::write(&path, b"abcdef").unwrap();
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    let mut files = KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *files = KernelFileState::default();
    files.files.insert(3, file);
    drop(files);

    assert_eq!(guest_kernel_lseek([3, 2, 0, 0, 0, 0]), 2);
    let mut bytes = [0_u8; 2];
    assert_eq!(
        guest_kernel_read([3, bytes.as_mut_ptr().addr() as u64, 2, 0, 0, 0]),
        2
    );
    assert_eq!(&bytes, b"cd");

    let mut stat = [0_u8; crate::SCE_KERNEL_STAT_SIZE];
    assert_eq!(
        guest_kernel_fstat([3, stat.as_mut_ptr().addr() as u64, 0, 0, 0, 0]),
        0
    );
    assert_eq!(i64::from_le_bytes(stat[72..80].try_into().unwrap()), 6);
    assert_ne!(u32::from_le_bytes(stat[4..8].try_into().unwrap()), 0);
    assert_ne!(u16::from_le_bytes(stat[8..10].try_into().unwrap()), 0);
    assert_ne!(u32::from_le_bytes(stat[88..92].try_into().unwrap()), 0);

    assert_eq!(guest_kernel_ftruncate([3, 4, 0, 0, 0, 0]), 0);
    assert_eq!(
        KERNEL_FILES
            .get()
            .unwrap()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .files
            .get(&3)
            .unwrap()
            .metadata()
            .unwrap()
            .len(),
        4
    );
}

/// The per-entry form attempts every path and reports how many resolved, instead of stopping
/// at the first rejection the way the error-index form does.
#[cfg(target_os = "macos")]
#[test]
fn per_entry_apr_resolution_continues_past_a_rejected_path() {
    let mut state = ProbeDispatchState::default();
    let missing = b"/app0/missing-one.bin\0";
    let also_missing = b"/app0/missing-two.bin\0";
    let list = [
        missing.as_ptr().addr() as u64,
        also_missing.as_ptr().addr() as u64,
    ];
    let mut ids = [0x1234_5678_u32; 2];
    let mut results = [0_u32; 2];
    // Every entry is attempted, so the reported count is the number that resolved: none.
    assert_eq!(
        ampr_resolve_paths_core(
            list.as_ptr().addr() as u64,
            2,
            None,
            AprResolveOutputs {
                ids: ids.as_mut_ptr().addr() as u64,
                results: results.as_mut_ptr().addr() as u64,
                ..AprResolveOutputs::default()
            },
            &mut state,
        ),
        0
    );
    assert_eq!(ids, [AMPR_INVALID_FILE_ID; 2]);
    assert!(results.iter().all(|result| *result != 0));
}

#[test]
fn resolver_prefix_keeps_one_separator() {
    assert_eq!(join_guest_path("/app0", "config.ini"), "/app0/config.ini");
    assert_eq!(join_guest_path("/app0/", "/config.ini"), "/app0/config.ini");
    assert_eq!(join_guest_path("", "/config.ini"), "/config.ini");
}

/// A rejected path must leave an identifier the guest recognizes as invalid. Leaving the
/// caller's buffer untouched lets uninitialized stack data pass as a usable file id, and the
/// guest then issues a read for a file it never resolved.
#[cfg(target_os = "macos")]
#[test]
fn unresolved_apr_paths_report_an_invalid_identifier() {
    let mut state = ProbeDispatchState::default();
    let path = b"/app0/does-not-exist.res\0";
    let path_list = [path.as_ptr().addr() as u64];
    // Pre-fill with values that would look like plausible identifiers if left untouched.
    let mut ids = [0x006e_0069_u32; 1];
    let mut sizes = [0xdead_beef_u64; 1];
    let mut error_index = 0_u32;
    assert_ne!(
        ampr_resolve_file_paths(
            [
                path_list.as_ptr().addr() as u64,
                1,
                ids.as_mut_ptr().addr() as u64,
                sizes.as_mut_ptr().addr() as u64,
                (&raw mut error_index).addr() as u64,
                0,
            ],
            &mut state,
        ),
        0
    );
    assert_eq!(ids[0], AMPR_INVALID_FILE_ID);
    assert_eq!(sizes[0], 0);
    assert_eq!(error_index, 0);
}

/// The error-index form is fail-fast: entries before the rejected path are committed, the
/// rejected entry is invalidated, and entries after it retain the caller's values.
#[cfg(target_os = "macos")]
#[test]
fn apr_resolution_stops_at_the_rejected_entry_without_touching_the_tail() {
    let root = TemporaryTestDirectory::new("apr-fail-fast");
    std::fs::write(root.path.join("asset.bin"), b"asset").unwrap();
    let mut state = ProbeDispatchState {
        game_root: root.path.clone(),
        ..ProbeDispatchState::default()
    };
    let present = b"/app0/asset.bin\0";
    let missing = b"/app0/missing.bin\0";
    let list = [
        present.as_ptr().addr() as u64,
        missing.as_ptr().addr() as u64,
        present.as_ptr().addr() as u64,
    ];
    let mut ids = [0x1111_1111_u32, 0x2222_2222, 0x3333_3333];
    let mut sizes = [0x1111_u64, 0x2222, 0x3333];
    let mut error_index = u32::MAX;

    assert_eq!(
        ampr_resolve_paths_core(
            list.as_ptr().addr() as u64,
            3,
            None,
            AprResolveOutputs {
                ids: ids.as_mut_ptr().addr() as u64,
                sizes: sizes.as_mut_ptr().addr() as u64,
                error_index: (&raw mut error_index).addr() as u64,
                ..AprResolveOutputs::default()
            },
            &mut state,
        ),
        sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
    );
    assert_ne!(ids[0], AMPR_INVALID_FILE_ID);
    assert_eq!(sizes[0], 5);
    assert_eq!(ids[1], AMPR_INVALID_FILE_ID);
    assert_eq!(sizes[1], 0);
    assert_eq!(error_index, 1);
    assert_eq!(ids[2], 0x3333_3333);
    assert_eq!(sizes[2], 0x3333);
}

#[test]
fn directory_records_pack_console_dirents_without_crossing_blocks() {
    let entries: Vec<(String, bool)> = (0..40)
        .map(|index| (format!("pakchunk{index}-ps5.pak"), index % 7 == 0))
        .collect();
    let mut next_file_number = 1;
    let directory = KernelDirectory::new(&entries, &mut next_file_number);
    assert_eq!(directory.records.len() % DIRENT_BLOCK_SIZE, 0);

    // Walk the records exactly as the guest does and recover every name in order.
    let mut names = Vec::new();
    let mut offset = 0;
    while offset < directory.records.len() {
        let record = &directory.records[offset..];
        let reclen = usize::from(u16::from_le_bytes([record[4], record[5]]));
        let namlen = usize::from(record[7]);
        if namlen == 0 {
            offset += DIRENT_BLOCK_SIZE - offset % DIRENT_BLOCK_SIZE;
            continue;
        }
        assert!(reclen >= DIRENT_HEADER_SIZE + namlen + 1);
        // A record must lie entirely inside one 512-byte block.
        assert_eq!(
            offset / DIRENT_BLOCK_SIZE,
            (offset + DIRENT_HEADER_SIZE + namlen) / DIRENT_BLOCK_SIZE,
            "record at {offset} crossed a block boundary"
        );
        names.push(String::from_utf8_lossy(&record[8..8 + namlen]).into_owned());
        assert_eq!(record[6], if names.len() % 7 == 1 { 4 } else { 8 });
        offset += reclen;
    }
    assert_eq!(
        names,
        entries
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>()
    );
    assert_eq!(next_file_number, 41);
}
