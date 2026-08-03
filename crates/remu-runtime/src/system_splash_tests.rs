use super::{ProbeDispatchState, SplashState, load_system_splash, system_service_hide_splash};

#[test]
fn installed_splash_loads_case_insensitively_as_rgba8() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("test clock follows Unix epoch")
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "remu-system-splash-{}-{unique}",
        std::process::id()
    ));
    let system = root.join("SCE_SYS");
    std::fs::create_dir_all(&system).expect("create test system directory");
    let path = system.join("PIC0.PNG");
    let file = std::fs::File::create(&path).expect("create test splash");
    let mut encoder = png::Encoder::new(file, 2, 1);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().expect("write PNG header");
    writer
        .write_image_data(&[1, 2, 3, 4, 5, 6])
        .expect("write PNG pixels");
    drop(writer);

    let (actual_path, image) = load_system_splash(&root).expect("load installed splash");
    assert_eq!(
        actual_path.canonicalize().expect("canonical actual splash"),
        path.canonicalize().expect("canonical expected splash")
    );
    assert_eq!((image.width(), image.height()), (2, 1));
    assert_eq!(image.rgba8(), &[1, 2, 3, 255, 4, 5, 6, 255]);
    std::fs::remove_dir_all(&root).expect("remove test system directory");
}

#[test]
fn hide_splash_publishes_one_idempotent_presenter_event() {
    remu_gpu::clear_thread_local_event_sink();
    let (sender, receiver) = remu_gpu::gpu_event_channel();
    remu_gpu::install_thread_local_event_sink(sender);
    let mut state = ProbeDispatchState::default();

    assert_eq!(system_service_hide_splash(&mut state), 0);
    assert_eq!(state.services.splash, SplashState::Hidden);
    assert_eq!(
        receiver.recv().expect("hide event"),
        remu_gpu::GpuEvent::HideHostSplash
    );
    assert_eq!(system_service_hide_splash(&mut state), 0);
    assert_eq!(
        receiver.try_recv(),
        Err(std::sync::mpsc::TryRecvError::Empty)
    );
    remu_gpu::clear_thread_local_event_sink();
}
