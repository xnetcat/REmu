// SPDX-License-Identifier: GPL-2.0-or-later

//! Win32 window, input, ordered guest-event scheduling, and DXGI presentation.

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::sync::OnceLock;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::{Duration, Instant};

use remu_gpu::{
    GpuEvent, GpuFlipCompletion, GpuMemoryCopy, GpuMemoryRelease, GpuMemoryReleaseValue,
    GpuMemoryWait, GpuMemoryWrite, HostDialogButtons, HostDialogRequest, HostDialogResultCode,
    HostPadState, HostPadVibration, HostToastRequest, PAD_BUTTON_CIRCLE, PAD_BUTTON_CROSS,
    PAD_BUTTON_DOWN, PAD_BUTTON_L1, PAD_BUTTON_L2, PAD_BUTTON_L3, PAD_BUTTON_LEFT,
    PAD_BUTTON_OPTIONS, PAD_BUTTON_R1, PAD_BUTTON_R2, PAD_BUTTON_R3, PAD_BUTTON_RIGHT,
    PAD_BUTTON_SQUARE, PAD_BUTTON_TRIANGLE, PAD_BUTTON_UP,
};
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC, D3D12_COMMAND_QUEUE_FLAG_NONE,
    D3D12_FENCE_FLAG_NONE, D3D12_PLACED_SUBRESOURCE_FOOTPRINT, D3D12_RESOURCE_BARRIER,
    D3D12_RESOURCE_BARRIER_0, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
    D3D12_RESOURCE_BARRIER_FLAG_NONE, D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
    D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE,
    D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_PRESENT,
    D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_TRANSITION_BARRIER,
    D3D12_SUBRESOURCE_FOOTPRINT, D3D12_TEXTURE_COPY_LOCATION, D3D12_TEXTURE_COPY_LOCATION_0,
    D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
    ID3D12CommandAllocator, ID3D12CommandList, ID3D12CommandQueue, ID3D12Device, ID3D12Fence,
    ID3D12GraphicsCommandList, ID3D12PipelineState, ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_ALPHA_MODE_UNSPECIFIED, DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM,
    DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, DXGI_MWA_NO_ALT_ENTER, DXGI_PRESENT,
    DXGI_SCALING_STRETCH, DXGI_SWAP_CHAIN_DESC1, DXGI_SWAP_CHAIN_FLAG,
    DXGI_SWAP_EFFECT_FLIP_DISCARD, DXGI_USAGE_RENDER_TARGET_OUTPUT, IDXGIFactory4, IDXGIOutput,
    IDXGISwapChain3,
};
use windows::Win32::UI::Input::XboxController::{
    XINPUT_GAMEPAD, XINPUT_GAMEPAD_A, XINPUT_GAMEPAD_B, XINPUT_GAMEPAD_DPAD_DOWN,
    XINPUT_GAMEPAD_DPAD_LEFT, XINPUT_GAMEPAD_DPAD_RIGHT, XINPUT_GAMEPAD_DPAD_UP,
    XINPUT_GAMEPAD_LEFT_SHOULDER, XINPUT_GAMEPAD_LEFT_THUMB, XINPUT_GAMEPAD_RIGHT_SHOULDER,
    XINPUT_GAMEPAD_RIGHT_THUMB, XINPUT_GAMEPAD_START, XINPUT_GAMEPAD_X, XINPUT_GAMEPAD_Y,
    XINPUT_STATE, XINPUT_VIBRATION, XInputGetState, XInputSetState,
};
use windows::core::Interface;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::{Fullscreen, Window, WindowId};

use crate::backend::{backend_error, backend_error_with_device, create_default_device};
use crate::context::{create_buffer, wait_for_fence};
use crate::diagnostics::name_object;
use crate::executor::{Dx12GuestExecutor, Dx12PresentTarget};
use crate::process_memory::{read_process_memory, write_process_memory};

const EVENT_POLL_INTERVAL: Duration = Duration::from_millis(1);
const EVENTS_PER_TICK: usize = 1_024;
const MAXIMUM_FRAME_BYTES: usize = 256 * 1024 * 1024;
const SWAP_CHAIN_BUFFER_COUNT: u32 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct XInputControllerReport {
    pub user_index: u32,
    pub packet_number: u32,
    pub buttons: u16,
    pub left_trigger: u8,
    pub right_trigger: u8,
    pub left_x: i16,
    pub left_y: i16,
    pub right_x: i16,
    pub right_y: i16,
}

pub fn probe_xinput_controllers() -> Vec<XInputControllerReport> {
    let mut controllers = Vec::new();
    for user_index in 0..4 {
        let mut state = XINPUT_STATE::default();
        // SAFETY: `state` is a complete writable XINPUT_STATE for the synchronous query.
        if unsafe { XInputGetState(user_index, &raw mut state) } != 0 {
            continue;
        }
        controllers.push(XInputControllerReport {
            user_index,
            packet_number: state.dwPacketNumber,
            buttons: state.Gamepad.wButtons.0,
            left_trigger: state.Gamepad.bLeftTrigger,
            right_trigger: state.Gamepad.bRightTrigger,
            left_x: state.Gamepad.sThumbLX,
            left_y: state.Gamepad.sThumbLY,
            right_x: state.Gamepad.sThumbRX,
            right_y: state.Gamepad.sThumbRY,
        });
    }
    controllers
}

pub fn test_xinput_vibration(user_index: u32) -> Result<(), String> {
    if !probe_xinput_controllers()
        .iter()
        .any(|controller| controller.user_index == user_index)
    {
        return Err(format!("XInput user {user_index} is not connected"));
    }
    let vibration = XINPUT_VIBRATION {
        wLeftMotorSpeed: 0x4000,
        wRightMotorSpeed: 0x2000,
    };
    // SAFETY: `vibration` is a complete immutable XInput vibration structure.
    let result = unsafe { XInputSetState(user_index, &raw const vibration) };
    if result != 0 {
        return Err(format!(
            "XInputSetState(user {user_index}) failed with code {result}"
        ));
    }
    struct StopVibration(u32);
    impl Drop for StopVibration {
        fn drop(&mut self) {
            let stopped = XINPUT_VIBRATION::default();
            // SAFETY: this best-effort guard borrows a complete zeroed vibration structure.
            let _ = unsafe { XInputSetState(self.0, &raw const stopped) };
        }
    }
    let _stop = StopVibration(user_index);
    std::thread::sleep(Duration::from_millis(250));
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct DisplayConfiguration {
    width: u32,
    height: u32,
    pitch: u32,
    pixel_format: u64,
    tiling_mode: u32,
}

#[derive(Clone, Debug)]
struct HostFrame {
    width: u32,
    height: u32,
    bgra8: Vec<u8>,
}

enum EventOutcome {
    Processed,
    Blocked(GpuEvent),
    Shutdown,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct PresentationState {
    suspended: bool,
    occluded: bool,
    zero_sized: bool,
}

impl PresentationState {
    const fn is_paused(self) -> bool {
        self.suspended || self.occluded || self.zero_sized
    }
}

struct Dx12Renderer {
    adapter_name: String,
    device: windows::Win32::Graphics::Direct3D12::ID3D12Device,
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    commands: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    next_fence: u64,
    swap_chain: IDXGISwapChain3,
    buffer_size: (u32, u32),
    buffer_format: DXGI_FORMAT,
}

impl Dx12Renderer {
    fn new(window: &Window, device: ID3D12Device, adapter_name: String) -> Result<Self, String> {
        let queue_description = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
            ..D3D12_COMMAND_QUEUE_DESC::default()
        };
        // SAFETY: descriptors are initialized and returned COM objects own their references.
        let queue: ID3D12CommandQueue =
            unsafe { device.CreateCommandQueue(&raw const queue_description) }.map_err(
                |error| backend_error("ID3D12Device::CreateCommandQueue", &error).to_string(),
            )?;
        name_object(&queue, "REMu presentation queue");
        // SAFETY: allocator and list types agree.
        let allocator = unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
            .map_err(|error| {
                backend_error("ID3D12Device::CreateCommandAllocator", &error).to_string()
            })?;
        // SAFETY: no pipeline state is required for copy commands.
        let commands: ID3D12GraphicsCommandList = unsafe {
            device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None::<&ID3D12PipelineState>,
            )
        }
        .map_err(|error| backend_error("ID3D12Device::CreateCommandList", &error).to_string())?;
        name_object(&commands, "REMu presentation command list");
        // Begin from a closed list; every frame resets it after waiting for the previous fence.
        unsafe { commands.Close() }.map_err(|error| {
            backend_error("ID3D12GraphicsCommandList::Close", &error).to_string()
        })?;
        // SAFETY: the fence begins below the first signal value.
        let fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
            .map_err(|error| backend_error("ID3D12Device::CreateFence", &error).to_string())?;
        name_object(&fence, "REMu presentation fence");

        let handle = window
            .window_handle()
            .map_err(|error| format!("could not get the Win32 window handle: {error}"))?;
        let RawWindowHandle::Win32(handle) = handle.as_raw() else {
            return Err("winit did not create a Win32 window".to_owned());
        };
        let hwnd = HWND(handle.hwnd.get() as *mut c_void);
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);
        // SAFETY: factory creation has no pointer preconditions.
        let factory: IDXGIFactory4 = unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }
            .map_err(|error| backend_error("CreateDXGIFactory2", &error).to_string())?;
        let description = swap_chain_description(width, height);
        // SAFETY: queue and HWND are live and the descriptor is initialized.
        let swap_chain = unsafe {
            factory.CreateSwapChainForHwnd(
                &queue,
                hwnd,
                &raw const description,
                None,
                None::<&IDXGIOutput>,
            )
        }
        .map_err(|error| {
            backend_error("IDXGIFactory2::CreateSwapChainForHwnd", &error).to_string()
        })?
        .cast::<IDXGISwapChain3>()
        .map_err(|error| backend_error("IDXGISwapChain1::cast", &error).to_string())?;
        // SAFETY: this associates only the live presenter window with its factory.
        unsafe { factory.MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER) }.map_err(|error| {
            backend_error("IDXGIFactory::MakeWindowAssociation", &error).to_string()
        })?;

        Ok(Self {
            adapter_name,
            device,
            queue,
            allocator,
            commands,
            fence,
            next_fence: 1,
            swap_chain,
            buffer_size: (width, height),
            buffer_format: DXGI_FORMAT_B8G8R8A8_UNORM,
        })
    }

    fn present(&mut self, frame: &HostFrame) -> Result<(), String> {
        validate_frame(frame)?;
        self.ensure_swap_chain(frame.width, frame.height, DXGI_FORMAT_B8G8R8A8_UNORM)?;

        let row_bytes = frame
            .width
            .checked_mul(4)
            .ok_or("frame row byte count overflows")?;
        let row_pitch = row_bytes
            .checked_add(255)
            .map(|value| value & !255)
            .ok_or("aligned frame row pitch overflows")?;
        let upload_bytes = u64::from(row_pitch)
            .checked_mul(u64::from(frame.height))
            .and_then(|value| usize::try_from(value).ok())
            .ok_or("frame upload byte count overflows")?;
        let upload = create_buffer(
            &self.device,
            upload_bytes,
            windows::Win32::Graphics::Direct3D12::D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )
        .map_err(|error| error.to_string())?;
        write_frame_upload(&upload, frame, row_pitch)?;

        let back_buffer_index = unsafe { self.swap_chain.GetCurrentBackBufferIndex() };
        // SAFETY: the current index is owned by this swap chain and returned as an owned COM ref.
        let back_buffer: ID3D12Resource =
            unsafe { self.swap_chain.GetBuffer(back_buffer_index) }
                .map_err(|error| backend_error("IDXGISwapChain::GetBuffer", &error).to_string())?;
        // SAFETY: the previous submission is complete before this allocator is reset.
        unsafe { self.allocator.Reset() }
            .map_err(|error| backend_error("ID3D12CommandAllocator::Reset", &error).to_string())?;
        // SAFETY: allocator is reset and no pipeline is required.
        unsafe {
            self.commands
                .Reset(&self.allocator, None::<&ID3D12PipelineState>)
        }
        .map_err(|error| backend_error("ID3D12GraphicsCommandList::Reset", &error).to_string())?;

        record_transition(
            &self.commands,
            &back_buffer,
            D3D12_RESOURCE_STATE_PRESENT,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        let mut source = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(upload.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                        Width: frame.width,
                        Height: frame.height,
                        Depth: 1,
                        RowPitch: row_pitch,
                    },
                },
            },
        };
        let mut destination = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(back_buffer.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        // SAFETY: both resources are live, match the declared footprint, and are in copy states.
        unsafe {
            self.commands.CopyTextureRegion(
                &raw const destination,
                0,
                0,
                0,
                &raw const source,
                None,
            );
            ManuallyDrop::drop(&mut source.pResource);
            ManuallyDrop::drop(&mut destination.pResource);
        }
        record_transition(
            &self.commands,
            &back_buffer,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_PRESENT,
        );
        self.submit_and_present()
    }

    fn present_native(&mut self, target: &Dx12PresentTarget) -> Result<(), String> {
        let started = Instant::now();
        if !matches!(
            target.format,
            DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM
        ) {
            return Err(format!(
                "unsupported native swap-chain format {:?}",
                target.format
            ));
        }
        self.ensure_swap_chain(target.width, target.height, target.format)?;
        let back_buffer_index = unsafe { self.swap_chain.GetCurrentBackBufferIndex() };
        let back_buffer: ID3D12Resource =
            unsafe { self.swap_chain.GetBuffer(back_buffer_index) }
                .map_err(|error| backend_error("IDXGISwapChain::GetBuffer", &error).to_string())?;
        unsafe { self.allocator.Reset() }
            .map_err(|error| backend_error("ID3D12CommandAllocator::Reset", &error).to_string())?;
        unsafe {
            self.commands
                .Reset(&self.allocator, None::<&ID3D12PipelineState>)
        }
        .map_err(|error| backend_error("ID3D12GraphicsCommandList::Reset", &error).to_string())?;
        record_transition(
            &self.commands,
            &target.resource,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        record_transition(
            &self.commands,
            &back_buffer,
            D3D12_RESOURCE_STATE_PRESENT,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        // SAFETY: both textures have the same extent, format and sample count and are in copy
        // states on queues belonging to their shared D3D12 device.
        unsafe { self.commands.CopyResource(&back_buffer, &target.resource) };
        record_transition(
            &self.commands,
            &back_buffer,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_PRESENT,
        );
        record_transition(
            &self.commands,
            &target.resource,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        self.submit_and_present()?;
        if std::env::var_os("REMU_TRACE_D3D12_TIMING").is_some_and(|value| value == "1") {
            eprintln!(
                "[REMu][D3D12-TIMING] direct_present={:#x} total_us={}",
                target.resource.as_raw() as usize,
                started.elapsed().as_micros()
            );
        }
        Ok(())
    }

    fn ensure_swap_chain(
        &mut self,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> Result<(), String> {
        if self.buffer_size != (width, height) || self.buffer_format != format {
            self.wait_idle()?;
            // SAFETY: no back-buffer reference is retained across this call and the queue is idle.
            unsafe {
                self.swap_chain.ResizeBuffers(
                    SWAP_CHAIN_BUFFER_COUNT,
                    width,
                    height,
                    format,
                    DXGI_SWAP_CHAIN_FLAG(0),
                )
            }
            .map_err(|error| backend_error("IDXGISwapChain::ResizeBuffers", &error).to_string())?;
            self.buffer_size = (width, height);
            self.buffer_format = format;
        }
        Ok(())
    }

    fn submit_and_present(&mut self) -> Result<(), String> {
        // SAFETY: all commands are recorded and the list is open.
        unsafe { self.commands.Close() }.map_err(|error| {
            backend_error("ID3D12GraphicsCommandList::Close", &error).to_string()
        })?;
        let list: ID3D12CommandList = self.commands.cast().map_err(|error| {
            backend_error("ID3D12GraphicsCommandList::cast", &error).to_string()
        })?;
        // SAFETY: the list is closed and resources stay alive until the fence below completes.
        unsafe { self.queue.ExecuteCommandLists(&[Some(list)]) };
        self.signal_and_wait()?;
        // SAFETY: the back buffer has returned to PRESENT and all copy work is complete.
        unsafe { self.swap_chain.Present(1, DXGI_PRESENT(0)) }
            .ok()
            .map_err(|error| {
                backend_error_with_device("IDXGISwapChain::Present", &error, &self.device)
                    .to_string()
            })?;
        Ok(())
    }

    fn signal_and_wait(&mut self) -> Result<(), String> {
        let value = self.next_fence;
        self.next_fence = self.next_fence.saturating_add(1);
        // SAFETY: queue and fence are live and values increase monotonically.
        unsafe { self.queue.Signal(&self.fence, value) }
            .map_err(|error| backend_error("ID3D12CommandQueue::Signal", &error).to_string())?;
        wait_for_fence(&self.fence, value).map_err(|error| error.to_string())
    }

    fn wait_idle(&mut self) -> Result<(), String> {
        self.signal_and_wait()
    }
}

struct Dx12Presenter {
    receiver: Receiver<GpuEvent>,
    window: Option<Window>,
    renderer: Option<Dx12Renderer>,
    guest_executor: Option<Dx12GuestExecutor>,
    startup_error: Option<String>,
    display: Option<DisplayConfiguration>,
    last_flip_address: Option<u64>,
    guest_frame: Option<HostFrame>,
    splash: Option<HostFrame>,
    dialog: Option<HostDialogRequest>,
    toast: Option<(HostToastRequest, Instant)>,
    pressed_keys: HashSet<KeyCode>,
    modifiers: ModifiersState,
    focused: bool,
    controller: Option<XINPUT_GAMEPAD>,
    controller_user: Option<u32>,
    controller_buttons: u16,
    applied_vibration: Option<(u32, HostPadVibration)>,
    host_events: VecDeque<GpuEvent>,
    guest_queues: BTreeMap<u64, VecDeque<GpuEvent>>,
    queue_turns: VecDeque<u64>,
    presentation_state: PresentationState,
    shutdown: bool,
}

impl Dx12Presenter {
    fn new(receiver: Receiver<GpuEvent>) -> Self {
        Self {
            receiver,
            window: None,
            renderer: None,
            guest_executor: None,
            startup_error: None,
            display: None,
            last_flip_address: None,
            guest_frame: None,
            splash: None,
            dialog: None,
            toast: None,
            pressed_keys: HashSet::new(),
            modifiers: ModifiersState::default(),
            focused: true,
            controller: None,
            controller_user: None,
            controller_buttons: 0,
            applied_vibration: None,
            host_events: VecDeque::new(),
            guest_queues: BTreeMap::new(),
            queue_turns: VecDeque::new(),
            presentation_state: PresentationState::default(),
            shutdown: false,
        }
    }

    fn initialize(&mut self, event_loop: &ActiveEventLoop) -> Result<(), String> {
        let attributes = Window::default_attributes()
            .with_title("REMu | Direct3D 12")
            .with_inner_size(LogicalSize::new(1280, 720))
            .with_active(true);
        let window = event_loop
            .create_window(attributes)
            .map_err(|error| format!("could not create the Windows window: {error}"))?;
        let native = create_default_device().map_err(|error| error.to_string())?;
        let guest_executor = Dx12GuestExecutor::new_with_device(
            native.device.clone(),
            native.capabilities.independent_front_back_stencil_ref_mask,
            native.capabilities.triangle_fan,
        )?;
        let renderer = Dx12Renderer::new(&window, native.device, native.capabilities.name.clone())?;
        eprintln!(
            "[REMu][D3D12] presenter started on {}",
            renderer.adapter_name
        );
        window.set_visible(true);
        window.focus_window();
        self.renderer = Some(renderer);
        self.guest_executor = Some(guest_executor);
        self.window = Some(window);
        remu_gpu::record_startup_milestone(remu_gpu::StartupMilestone::WindowReady);
        Ok(())
    }

    fn receive_events(&mut self) {
        for _ in 0..EVENTS_PER_TICK {
            match self.receiver.try_recv() {
                Ok(event) => {
                    remu_gpu::record_gpu_event_consumed();
                    self.enqueue_event(event);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.shutdown = true;
                    break;
                }
            }
        }
    }

    fn enqueue_event(&mut self, event: GpuEvent) {
        if let GpuEvent::QueueSubmission { queue_id, events } = event {
            let queue = self.guest_queues.entry(queue_id).or_default();
            let activate = queue.is_empty();
            queue.extend(events);
            if activate && !queue.is_empty() {
                self.queue_turns.push_back(queue_id);
            }
        } else {
            self.host_events.push_back(event);
        }
    }

    fn process_events(&mut self) {
        let mut processed = 0;
        while processed < EVENTS_PER_TICK {
            if let Some(event) = self.host_events.pop_front() {
                let outcome = self.process_event(event);
                remu_gpu::record_gpu_work_item_processed();
                processed += 1;
                if matches!(outcome, EventOutcome::Shutdown) {
                    self.shutdown = true;
                    return;
                }
                continue;
            }
            let turns = self.queue_turns.len();
            if turns == 0 {
                break;
            }
            let mut made_progress = false;
            for _ in 0..turns {
                if processed >= EVENTS_PER_TICK {
                    break;
                }
                let queue_id = self
                    .queue_turns
                    .pop_front()
                    .expect("turn count came from queue length");
                let event = self
                    .guest_queues
                    .get_mut(&queue_id)
                    .and_then(VecDeque::pop_front)
                    .expect("an active queue has a front event");
                match self.process_event(event) {
                    EventOutcome::Blocked(event) => {
                        self.guest_queues
                            .get_mut(&queue_id)
                            .expect("blocked queue remains registered")
                            .push_front(event);
                    }
                    EventOutcome::Processed => {
                        remu_gpu::record_gpu_work_item_processed();
                        processed += 1;
                        made_progress = true;
                    }
                    EventOutcome::Shutdown => {
                        remu_gpu::record_gpu_work_item_processed();
                        self.shutdown = true;
                        return;
                    }
                }
                if self
                    .guest_queues
                    .get(&queue_id)
                    .is_some_and(|queue| !queue.is_empty())
                {
                    self.queue_turns.push_back(queue_id);
                } else {
                    self.guest_queues.remove(&queue_id);
                }
            }
            if !made_progress {
                break;
            }
        }
    }

    fn process_event(&mut self, event: GpuEvent) -> EventOutcome {
        match event {
            GpuEvent::QueueSubmission { .. } => {
                eprintln!("[REMu][D3D12] ignored invalid nested queue submission");
            }
            GpuEvent::DisplayRegistered {
                width,
                height,
                pitch,
                pixel_format,
                tiling_mode,
                ..
            } => {
                self.display = Some(DisplayConfiguration {
                    width,
                    height,
                    pitch,
                    pixel_format,
                    tiling_mode,
                });
                eprintln!(
                    "[REMu][D3D12] display registered {width}x{height} pitch={pitch} format={pixel_format:#018x} tiling={tiling_mode}"
                );
            }
            GpuEvent::ShowHostSplash(image) => {
                self.splash = Some(host_image_frame(&image));
                self.render_current_frame();
            }
            GpuEvent::HideHostSplash => {
                self.splash = None;
                self.render_current_frame();
            }
            GpuEvent::ShowHostDialog(request) => {
                self.dialog = Some(request);
                self.render_current_frame();
            }
            GpuEvent::HideHostDialog { sequence } => {
                if self
                    .dialog
                    .as_ref()
                    .is_some_and(|dialog| dialog.sequence == sequence)
                {
                    self.dialog = None;
                    self.render_current_frame();
                }
            }
            GpuEvent::ShowHostToast(request) => {
                let expires =
                    Instant::now() + Duration::from_millis(u64::from(request.duration_ms));
                self.toast = Some((request, expires));
                self.render_current_frame();
            }
            GpuEvent::Flip {
                address,
                buffer_index,
                flip_argument,
                width,
                height,
                pitch,
                pixel_format,
                tiling_mode,
                ..
            } => {
                let display = DisplayConfiguration {
                    width,
                    height,
                    pitch,
                    pixel_format,
                    tiling_mode,
                };
                self.display = Some(display);
                self.last_flip_address = Some(address);
                let direct_target = if self.splash.is_none()
                    && self.dialog.is_none()
                    && self.toast.is_none()
                    && !self.presentation_state.is_paused()
                {
                    self.guest_executor.as_mut().and_then(|executor| {
                        match executor.native_present_target(address, width, height) {
                            Ok(target) => target,
                            Err(error) => {
                                eprintln!(
                                    "[REMu][D3D12] could not complete native flip target at {address:#x}: {error}"
                                );
                                None
                            }
                        }
                    })
                } else {
                    None
                };
                let presented_directly = direct_target.as_ref().is_some_and(|target| {
                    self.renderer.as_mut().is_some_and(|renderer| {
                        match renderer.present_native(target) {
                            Ok(()) => true,
                            Err(error) => {
                                eprintln!(
                                    "[REMu][D3D12] direct presentation at {address:#x} failed: {error}"
                                );
                                false
                            }
                        }
                    })
                });
                if !presented_directly {
                    let rendered = self.guest_executor.as_mut().and_then(|executor| {
                        match executor.prepare_presentable_bgra(address, width, height) {
                            Ok(frame) => frame.map(|bgra8| HostFrame {
                                width,
                                height,
                                bgra8,
                            }),
                            Err(error) => {
                                eprintln!(
                                    "[REMu][D3D12] could not flush flip target at {address:#x}: {error}"
                                );
                                None
                            }
                        }
                    });
                    match rendered.map_or_else(|| snapshot_guest_frame(address, display), Ok) {
                        Ok(frame) => {
                            self.guest_frame = Some(frame);
                            self.render_current_frame();
                        }
                        Err(error) => eprintln!(
                            "[REMu][D3D12] could not snapshot flip at {address:#x}: {error}"
                        ),
                    }
                }
                remu_gpu::record_gpu_flip_completed(GpuFlipCompletion {
                    address,
                    buffer_index,
                    flip_argument,
                });
            }
            GpuEvent::WaitFlipDone(_) => {}
            // Draw/dispatch execution is queue ordered and synchronously fenced today, so these
            // stage barriers are already satisfied without making color bytes CPU visible. A
            // guest-memory acquire/release or overlapping consumer below remains a publication
            // point.
            GpuEvent::CommandProcessorBarrier | GpuEvent::PipelineBarrier(_) => {}
            GpuEvent::MemoryAcquire(acquire) => {
                // ACQUIRE_MEM invalidates GPU-side views of CPU-written guest resources. It is
                // not a GPU-to-CPU publication point, so keep GPU-owned color targets resident;
                // the executor's write watches refresh an overlapping target at its next use.
                if let Some(executor) = self.guest_executor.as_mut() {
                    executor.synchronize_memory_acquire(acquire);
                }
            }
            GpuEvent::MemoryCopy(copy) => {
                self.flush_render_targets("memory copy");
                if let Err(error) = execute_memory_copy(&copy) {
                    eprintln!("[REMu][D3D12] DMA_DATA failed: {error}");
                }
            }
            GpuEvent::MemoryWrite(write) => {
                self.flush_render_targets("memory write");
                if let Err(error) = execute_memory_write(&write) {
                    eprintln!("[REMu][D3D12] WRITE_DATA failed: {error}");
                }
            }
            GpuEvent::MemoryRelease(release) => {
                self.flush_render_targets("memory release");
                if let Err(error) = execute_memory_release(release) {
                    eprintln!("[REMu][D3D12] RELEASE_MEM failed: {error}");
                }
            }
            GpuEvent::MemoryWait(wait) => {
                self.flush_render_targets("memory wait");
                if !memory_wait_is_satisfied(wait) {
                    return EventOutcome::Blocked(GpuEvent::MemoryWait(wait));
                }
            }
            GpuEvent::RegisteredMemoryWait { id, wait } => {
                self.flush_render_targets("registered memory wait");
                if remu_gpu::take_gpu_wait_latch(id).is_none() {
                    if !memory_wait_is_satisfied(wait) {
                        return EventOutcome::Blocked(GpuEvent::RegisteredMemoryWait { id, wait });
                    }
                    remu_gpu::retire_gpu_wait(id);
                }
            }
            GpuEvent::ComputeDispatch(dispatch) => {
                let result = self
                    .guest_executor
                    .as_mut()
                    .ok_or_else(|| "D3D12 guest executor is unavailable".to_owned())
                    .and_then(|executor| {
                        executor.flush_for_compute_inputs(&dispatch)?;
                        executor.execute_compute(&dispatch)
                    });
                if let Err(error) = result {
                    eprintln!(
                        "[REMu][D3D12] compute shader {:#x} execution failed: {error}",
                        dispatch.shader_address,
                    );
                }
            }
            GpuEvent::IndexedDraw(draw) => {
                let result = self
                    .guest_executor
                    .as_mut()
                    .ok_or_else(|| "D3D12 guest executor is unavailable".to_owned())
                    .and_then(|executor| {
                        executor.flush_for_draw_inputs(&draw)?;
                        executor.execute_draw_deferred(&draw)
                    });
                if let Err(error) = result {
                    eprintln!(
                        "[REMu][D3D12] draw {} ES={:#x} PS={:#x} execution failed: {error}",
                        draw.sequence, draw.export_shader_address, draw.pixel_shader_address,
                    );
                }
            }
            GpuEvent::SubmissionBoundary(sequence) => {
                // This marker completes the emulated queue submission, not a CPU request to map
                // every color allocation. Guest draw/dispatch calls are already synchronously
                // fenced, so completion is ordered while native targets remain authoritative.
                // Release/wait packets, overlapping consumers, CPU writes and cache eviction
                // publish only the allocations that actually cross back to guest memory.
                let completed = self
                    .guest_executor
                    .as_mut()
                    .is_none_or(|executor| match executor.complete_pending_gpu_work() {
                        Ok(()) => true,
                        Err(error) => {
                            eprintln!(
                                "[REMu][D3D12] submission {sequence} completion failed: {error}"
                            );
                            false
                        }
                    });
                if completed {
                    remu_gpu::record_gpu_submission_completed(sequence);
                }
            }
            GpuEvent::Shutdown => return EventOutcome::Shutdown,
        }
        EventOutcome::Processed
    }

    fn flush_render_targets(&mut self, reason: &str) -> bool {
        let Some(executor) = self.guest_executor.as_mut() else {
            return true;
        };
        let pending = executor.pending_render_target_count();
        if pending != 0
            && std::env::var_os("REMU_TRACE_D3D12_TIMING").is_some_and(|value| value == "1")
        {
            eprintln!("[REMu][D3D12-TIMING] flush_reason={reason:?} pending={pending}");
        }
        if let Err(error) = executor.flush_all_render_targets() {
            eprintln!("[REMu][D3D12] {reason} render-target flush failed: {error}");
            false
        } else {
            true
        }
    }

    fn render_current_frame(&mut self) {
        if self.presentation_state.is_paused() {
            return;
        }
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        let base = self
            .splash
            .as_ref()
            .or(self.guest_frame.as_ref())
            .cloned()
            .unwrap_or_else(black_frame);
        let toast = self.toast.as_ref().map(|(toast, _)| toast);
        let frame = compose_system_ui(base, self.dialog.as_ref(), toast);
        if let Err(error) = renderer.present(&frame) {
            eprintln!("[REMu][D3D12] presentation failed: {error}");
        }
    }

    fn capture_latest_guest_frame(&mut self) {
        let (Some(address), Some(display)) = (self.last_flip_address, self.display) else {
            return;
        };
        let rendered = self.guest_executor.as_mut().and_then(|executor| {
            match executor.prepare_presentable_bgra(address, display.width, display.height) {
                Ok(frame) => frame.map(|bgra8| HostFrame {
                    width: display.width,
                    height: display.height,
                    bgra8,
                }),
                Err(error) => {
                    eprintln!(
                        "[REMu][D3D12] could not preserve latest frame while presentation paused: {error}"
                    );
                    None
                }
            }
        });
        match rendered.map_or_else(|| snapshot_guest_frame(address, display), Ok) {
            Ok(frame) => self.guest_frame = Some(frame),
            Err(error) => eprintln!(
                "[REMu][D3D12] could not snapshot latest paused frame at {address:#x}: {error}"
            ),
        }
    }

    fn apply_presentation_state_change(&mut self, was_paused: bool) {
        let paused = self.presentation_state.is_paused();
        if paused && !was_paused {
            self.capture_latest_guest_frame();
        } else if !paused && was_paused {
            self.render_current_frame();
        }
    }

    fn toggle_fullscreen(&self) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        if window.fullscreen().is_some() {
            window.set_fullscreen(None);
            eprintln!("[REMu][D3D12] left borderless fullscreen");
        } else {
            window.set_fullscreen(Some(Fullscreen::Borderless(window.current_monitor())));
            eprintln!("[REMu][D3D12] entered borderless fullscreen");
        }
    }

    fn update_keyboard(&mut self, code: KeyCode, state: ElementState) {
        match state {
            ElementState::Pressed => {
                self.pressed_keys.insert(code);
            }
            ElementState::Released => {
                self.pressed_keys.remove(&code);
            }
        }
        self.publish_pad_state();
        if state == ElementState::Pressed {
            if matches!(code, KeyCode::Enter | KeyCode::NumpadEnter | KeyCode::KeyZ) {
                self.dismiss_dialog(true);
            } else if matches!(code, KeyCode::Escape | KeyCode::KeyX) {
                self.dismiss_dialog(false);
            }
        }
    }

    fn dismiss_dialog(&mut self, affirmative: bool) {
        let Some(dialog) = self.dialog.take() else {
            return;
        };
        let code = match (affirmative, dialog.buttons) {
            (true, HostDialogButtons::YesNo) => HostDialogResultCode::Yes,
            (true, _) => HostDialogResultCode::Ok,
            (false, HostDialogButtons::YesNo) => HostDialogResultCode::No,
            (false, _) => HostDialogResultCode::Cancel,
        };
        remu_gpu::complete_host_dialog(dialog.sequence, code);
        self.render_current_frame();
    }

    fn publish_pad_state(&self) {
        remu_gpu::set_host_pad_state(compose_pad_state(
            &self.pressed_keys,
            self.controller,
            self.input_active(),
        ));
    }

    fn input_active(&self) -> bool {
        self.focused && !self.presentation_state.suspended
    }

    fn poll_controller(&mut self) {
        let mut connected = None;
        for user in 0..4 {
            let mut state = XINPUT_STATE::default();
            // SAFETY: `state` is a complete writable XINPUT_STATE for the synchronous query.
            if unsafe { XInputGetState(user, &raw mut state) } == 0 {
                connected = Some((user, state.Gamepad));
                break;
            }
        }
        let next_user = connected.map(|(user, _)| user);
        if self.controller_user != next_user {
            match (self.controller_user, next_user) {
                (None, Some(user)) => eprintln!("[REMu][INPUT] XInput user {user} connected"),
                (Some(user), None) => eprintln!("[REMu][INPUT] XInput user {user} disconnected"),
                (Some(previous), Some(user)) => {
                    eprintln!("[REMu][INPUT] switched XInput user {previous} -> {user}")
                }
                (None, None) => {}
            }
        }
        let next_controller = connected.map(|(_, gamepad)| gamepad);
        let next_buttons = next_controller.map_or(0, |gamepad| gamepad.wButtons.0);
        let pressed = if self.input_active() {
            next_buttons & !self.controller_buttons
        } else {
            0
        };
        self.controller_user = next_user;
        self.controller = next_controller;
        self.controller_buttons = next_buttons;
        self.apply_controller_vibration();
        self.publish_pad_state();
        if pressed & XINPUT_GAMEPAD_A.0 != 0 {
            self.dismiss_dialog(true);
        } else if pressed & XINPUT_GAMEPAD_B.0 != 0 {
            self.dismiss_dialog(false);
        }
    }

    fn apply_controller_vibration(&mut self) {
        let requested = if self.input_active() {
            remu_gpu::host_pad_vibration()
        } else {
            HostPadVibration::default()
        };
        let next = self.controller_user.map(|user| (user, requested));
        if self.applied_vibration == next {
            return;
        }
        if let Some((previous_user, previous)) = self.applied_vibration
            && Some(previous_user) != self.controller_user
            && previous != HostPadVibration::default()
        {
            let stopped = XINPUT_VIBRATION::default();
            // SAFETY: `stopped` is a complete immutable XInput vibration structure.
            let _ = unsafe { XInputSetState(previous_user, &raw const stopped) };
        }
        if let Some((user, requested)) = next {
            let vibration = xinput_vibration(requested);
            // SAFETY: `vibration` is a complete immutable XInput vibration structure.
            let result = unsafe { XInputSetState(user, &raw const vibration) };
            if result != 0 {
                eprintln!("[REMu][INPUT] XInput user {user} vibration update failed code={result}");
            }
        }
        self.applied_vibration = next;
    }

    fn expire_toast(&mut self) {
        if self
            .toast
            .as_ref()
            .is_some_and(|(_, deadline)| Instant::now() >= *deadline)
        {
            self.toast = None;
            self.render_current_frame();
        }
    }

    fn has_pending_work(&self) -> bool {
        !self.host_events.is_empty()
            || !self.queue_turns.is_empty()
            || remu_gpu::gpu_events_in_flight() != 0
    }
}

impl Drop for Dx12Presenter {
    fn drop(&mut self) {
        if let Some(user) = self.controller_user {
            let stopped = XINPUT_VIBRATION::default();
            // SAFETY: stopping a known XInput slot borrows a complete vibration structure only
            // for this synchronous call. A disconnect is harmless and returns an error code.
            let _ = unsafe { XInputSetState(user, &raw const stopped) };
        }
    }
}

fn compose_pad_state(
    pressed_keys: &HashSet<KeyCode>,
    controller: Option<XINPUT_GAMEPAD>,
    input_active: bool,
) -> HostPadState {
    if !input_active {
        return HostPadState::default();
    }
    let down = |code| pressed_keys.contains(&code);
    let mut buttons = 0;
    for (pressed, button) in [
        (down(KeyCode::ArrowUp), PAD_BUTTON_UP),
        (down(KeyCode::ArrowRight), PAD_BUTTON_RIGHT),
        (down(KeyCode::ArrowDown), PAD_BUTTON_DOWN),
        (down(KeyCode::ArrowLeft), PAD_BUTTON_LEFT),
        (
            down(KeyCode::Enter) || down(KeyCode::NumpadEnter) || down(KeyCode::KeyZ),
            PAD_BUTTON_CROSS,
        ),
        (
            down(KeyCode::Escape) || down(KeyCode::KeyX),
            PAD_BUTTON_CIRCLE,
        ),
        (down(KeyCode::KeyC), PAD_BUTTON_SQUARE),
        (down(KeyCode::KeyV), PAD_BUTTON_TRIANGLE),
        (down(KeyCode::KeyQ), PAD_BUTTON_L1),
        (down(KeyCode::KeyE), PAD_BUTTON_R1),
        (down(KeyCode::KeyR), PAD_BUTTON_L2),
        (down(KeyCode::KeyF), PAD_BUTTON_R2),
        (
            down(KeyCode::Tab) || down(KeyCode::Backspace),
            PAD_BUTTON_OPTIONS,
        ),
    ] {
        if pressed {
            buttons |= button;
        }
    }
    let axis = |negative, positive| match (down(negative), down(positive)) {
        (true, false) => 0,
        (false, true) => 255,
        _ => 128,
    };
    let mut state = HostPadState {
        buttons,
        left_x: axis(KeyCode::KeyA, KeyCode::KeyD),
        left_y: axis(KeyCode::KeyW, KeyCode::KeyS),
        right_x: axis(KeyCode::KeyJ, KeyCode::KeyL),
        right_y: axis(KeyCode::KeyI, KeyCode::KeyK),
        l2: if down(KeyCode::KeyR) { 255 } else { 0 },
        r2: if down(KeyCode::KeyF) { 255 } else { 0 },
    };
    if let Some(controller) = controller {
        merge_xinput(&mut state, controller);
    }
    state
}

fn merge_xinput(state: &mut HostPadState, controller: XINPUT_GAMEPAD) {
    let pressed = |button| controller.wButtons.contains(button);
    for (is_pressed, button) in [
        (pressed(XINPUT_GAMEPAD_DPAD_UP), PAD_BUTTON_UP),
        (pressed(XINPUT_GAMEPAD_DPAD_RIGHT), PAD_BUTTON_RIGHT),
        (pressed(XINPUT_GAMEPAD_DPAD_DOWN), PAD_BUTTON_DOWN),
        (pressed(XINPUT_GAMEPAD_DPAD_LEFT), PAD_BUTTON_LEFT),
        (pressed(XINPUT_GAMEPAD_A), PAD_BUTTON_CROSS),
        (pressed(XINPUT_GAMEPAD_B), PAD_BUTTON_CIRCLE),
        (pressed(XINPUT_GAMEPAD_X), PAD_BUTTON_SQUARE),
        (pressed(XINPUT_GAMEPAD_Y), PAD_BUTTON_TRIANGLE),
        (pressed(XINPUT_GAMEPAD_LEFT_SHOULDER), PAD_BUTTON_L1),
        (pressed(XINPUT_GAMEPAD_RIGHT_SHOULDER), PAD_BUTTON_R1),
        (pressed(XINPUT_GAMEPAD_LEFT_THUMB), PAD_BUTTON_L3),
        (pressed(XINPUT_GAMEPAD_RIGHT_THUMB), PAD_BUTTON_R3),
        (pressed(XINPUT_GAMEPAD_START), PAD_BUTTON_OPTIONS),
    ] {
        if is_pressed {
            state.buttons |= button;
        }
    }
    if controller.bLeftTrigger > 30 {
        state.buttons |= PAD_BUTTON_L2;
    }
    if controller.bRightTrigger > 30 {
        state.buttons |= PAD_BUTTON_R2;
    }
    state.l2 = state.l2.max(controller.bLeftTrigger);
    state.r2 = state.r2.max(controller.bRightTrigger);

    merge_axis(&mut state.left_x, controller.sThumbLX, 7_849, false);
    merge_axis(&mut state.left_y, controller.sThumbLY, 7_849, true);
    merge_axis(&mut state.right_x, controller.sThumbRX, 8_689, false);
    merge_axis(&mut state.right_y, controller.sThumbRY, 8_689, true);
}

fn xinput_vibration(vibration: HostPadVibration) -> XINPUT_VIBRATION {
    // Expanding 8-bit guest intensity by 257 preserves both endpoints and every bit pattern.
    XINPUT_VIBRATION {
        wLeftMotorSpeed: u16::from(vibration.large_motor) * 257,
        wRightMotorSpeed: u16::from(vibration.small_motor) * 257,
    }
}

fn merge_axis(destination: &mut u8, value: i16, deadzone: i32, inverted: bool) {
    let value = i32::from(value);
    let magnitude = value.unsigned_abs() as i32;
    if magnitude <= deadzone {
        return;
    }
    let positive = (value > 0) != inverted;
    let input_limit = if value > 0 { 32_767 } else { 32_768 };
    let output_range = if positive { 127 } else { 128 };
    let scaled = (magnitude - deadzone) * output_range / (input_limit - deadzone);
    *destination = if positive {
        u8::try_from(128 + scaled).expect("positive XInput axis fits u8")
    } else {
        u8::try_from(128 - scaled).expect("negative XInput axis fits u8")
    };
}

impl ApplicationHandler for Dx12Presenter {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            let was_paused = self.presentation_state.is_paused();
            self.presentation_state.suspended = false;
            self.publish_pad_state();
            self.apply_controller_vibration();
            self.apply_presentation_state_change(was_paused);
            return;
        }
        if self.startup_error.is_some() {
            return;
        }
        if let Err(error) = self.initialize(event_loop) {
            eprintln!("[REMu][D3D12] {error}");
            self.startup_error = Some(error);
            event_loop.exit();
        }
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        let was_paused = self.presentation_state.is_paused();
        self.presentation_state.suspended = true;
        self.pressed_keys.clear();
        self.modifiers = ModifiersState::default();
        self.publish_pad_state();
        self.apply_controller_vibration();
        self.apply_presentation_state_change(was_paused);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                if let Some(dialog) = self.dialog.take() {
                    remu_gpu::cancel_host_dialog(dialog.sequence);
                }
                eprintln!("[REMu][D3D12] window close requested");
                event_loop.exit();
            }
            WindowEvent::Focused(focused) => {
                self.focused = focused;
                if !focused {
                    self.pressed_keys.clear();
                    self.modifiers = ModifiersState::default();
                }
                self.publish_pad_state();
                self.apply_controller_vibration();
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.modifiers = modifiers.state();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if is_fullscreen_shortcut(code, event.state, self.modifiers) {
                        self.toggle_fullscreen();
                        return;
                    }
                    self.update_keyboard(code, event.state);
                }
            }
            WindowEvent::Occluded(occluded) => {
                let was_paused = self.presentation_state.is_paused();
                self.presentation_state.occluded = occluded;
                self.apply_presentation_state_change(was_paused);
            }
            WindowEvent::Resized(size) => {
                let was_paused = self.presentation_state.is_paused();
                self.presentation_state.zero_sized = size.width == 0 || size.height == 0;
                self.apply_presentation_state_change(was_paused);
                if !self.presentation_state.is_paused()
                    && let Some(window) = self.window.as_ref()
                {
                    window.request_redraw();
                }
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                if !self.presentation_state.is_paused()
                    && let Some(window) = self.window.as_ref()
                {
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => self.render_current_frame(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.poll_controller();
        self.receive_events();
        self.process_events();
        self.expire_toast();
        if self.shutdown {
            event_loop.exit();
        } else if self.has_pending_work() {
            event_loop.set_control_flow(ControlFlow::Poll);
        } else {
            event_loop
                .set_control_flow(ControlFlow::WaitUntil(Instant::now() + EVENT_POLL_INTERVAL));
        }
    }
}

fn is_fullscreen_shortcut(code: KeyCode, state: ElementState, modifiers: ModifiersState) -> bool {
    state == ElementState::Pressed
        && (code == KeyCode::F11
            || (matches!(code, KeyCode::Enter | KeyCode::NumpadEnter) && modifiers.alt_key()))
}

pub fn run_presenter(receiver: Receiver<GpuEvent>) -> Result<(), String> {
    remu_gpu::record_startup_milestone(remu_gpu::StartupMilestone::PresenterEnter);
    let mut presenter = Dx12Presenter::new(receiver);
    remu_gpu::record_startup_milestone(remu_gpu::StartupMilestone::PresenterResourcesReady);
    let event_loop = EventLoop::builder()
        .build()
        .map_err(|error| format!("Windows event loop failed: {error}"))?;
    event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + EVENT_POLL_INTERVAL));
    event_loop
        .run_app(&mut presenter)
        .map_err(|error| format!("Windows event loop failed: {error}"))?;
    if let Some(error) = presenter.startup_error.take() {
        Err(error)
    } else {
        Ok(())
    }
}

fn swap_chain_description(width: u32, height: u32) -> DXGI_SWAP_CHAIN_DESC1 {
    DXGI_SWAP_CHAIN_DESC1 {
        Width: width,
        Height: height,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
        BufferCount: SWAP_CHAIN_BUFFER_COUNT,
        Scaling: DXGI_SCALING_STRETCH,
        SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
        AlphaMode: DXGI_ALPHA_MODE_UNSPECIFIED,
        ..DXGI_SWAP_CHAIN_DESC1::default()
    }
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
    // SAFETY: Transition is active and describes the caller's current resource state.
    unsafe {
        commands.ResourceBarrier(std::slice::from_ref(&barrier));
        ManuallyDrop::drop(&mut (*barrier.Anonymous.Transition).pResource);
    }
}

fn write_frame_upload(
    upload: &ID3D12Resource,
    frame: &HostFrame,
    row_pitch: u32,
) -> Result<(), String> {
    let mut mapped = std::ptr::null_mut::<c_void>();
    let no_read = windows::Win32::Graphics::Direct3D12::D3D12_RANGE { Begin: 0, End: 0 };
    // SAFETY: upload heaps are CPU-writable and the output slot is valid.
    unsafe { upload.Map(0, Some(&raw const no_read), Some(&raw mut mapped)) }
        .map_err(|error| backend_error("ID3D12Resource::Map(frame upload)", &error).to_string())?;
    let source_row = usize::try_from(frame.width)
        .ok()
        .and_then(|width| width.checked_mul(4))
        .ok_or("source frame row size overflows")?;
    let destination_row = usize::try_from(row_pitch).map_err(|_| "row pitch does not fit usize")?;
    for y in 0..usize::try_from(frame.height).map_err(|_| "frame height does not fit usize")? {
        // SAFETY: the upload allocation is `row_pitch * height` bytes and source frame validation
        // proves every tightly packed source row exists.
        unsafe {
            std::ptr::copy_nonoverlapping(
                frame.bgra8.as_ptr().add(y * source_row),
                mapped.cast::<u8>().add(y * destination_row),
                source_row,
            );
        }
    }
    let written = windows::Win32::Graphics::Direct3D12::D3D12_RANGE {
        Begin: 0,
        End: destination_row
            .checked_mul(usize::try_from(frame.height).map_err(|_| "height does not fit usize")?)
            .ok_or("written range overflows")?,
    };
    // SAFETY: subresource zero is mapped and `written` is within the allocation.
    unsafe { upload.Unmap(0, Some(&raw const written)) };
    Ok(())
}

fn validate_frame(frame: &HostFrame) -> Result<(), String> {
    let expected = usize::try_from(frame.width)
        .ok()
        .and_then(|width| width.checked_mul(usize::try_from(frame.height).ok()?))
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or("frame dimensions overflow")?;
    if frame.width == 0
        || frame.height == 0
        || expected > MAXIMUM_FRAME_BYTES
        || frame.bgra8.len() != expected
    {
        return Err(format!(
            "invalid {}x{} BGRA frame with {} bytes",
            frame.width,
            frame.height,
            frame.bgra8.len()
        ));
    }
    Ok(())
}

fn black_frame() -> HostFrame {
    HostFrame {
        width: remu_gpu::SYSTEM_UI_WIDTH,
        height: remu_gpu::SYSTEM_UI_HEIGHT,
        bgra8: vec![
            0;
            usize::try_from(remu_gpu::SYSTEM_UI_WIDTH * remu_gpu::SYSTEM_UI_HEIGHT * 4)
                .expect("system UI dimensions fit usize")
        ],
    }
}

fn host_image_frame(image: &remu_gpu::GpuHostImage) -> HostFrame {
    let mut bgra8 = image.rgba8().to_vec();
    for pixel in bgra8.chunks_exact_mut(4) {
        pixel.swap(0, 2);
    }
    HostFrame {
        width: image.width(),
        height: image.height(),
        bgra8,
    }
}

fn compose_system_ui(
    mut frame: HostFrame,
    dialog: Option<&HostDialogRequest>,
    toast: Option<&HostToastRequest>,
) -> HostFrame {
    if dialog.is_none() && toast.is_none() {
        return frame;
    }
    let overlay = remu_gpu::rasterize_system_ui(dialog, toast);
    let frame_width = usize::try_from(frame.width).expect("validated frame width fits usize");
    let frame_height = usize::try_from(frame.height).expect("validated frame height fits usize");
    let overlay_width = usize::try_from(overlay.width).expect("system UI width fits usize");
    let overlay_height = usize::try_from(overlay.height).expect("system UI height fits usize");
    for y in 0..frame_height {
        let source_y = y * overlay_height / frame_height;
        for x in 0..frame_width {
            let source_x = x * overlay_width / frame_width;
            let source = (source_y * overlay_width + source_x) * 4;
            let destination = (y * frame_width + x) * 4;
            let alpha = u32::from(overlay.rgba[source + 3]);
            if alpha == 0 {
                continue;
            }
            let inverse = 255 - alpha;
            for (destination_channel, source_channel) in [(0, 2), (1, 1), (2, 0)] {
                frame.bgra8[destination + destination_channel] = u8::try_from(
                    (u32::from(overlay.rgba[source + source_channel]) * alpha
                        + u32::from(frame.bgra8[destination + destination_channel]) * inverse
                        + 127)
                        / 255,
                )
                .expect("alpha blend remains in u8 range");
            }
            frame.bgra8[destination + 3] = 255;
        }
    }
    frame
}

fn snapshot_guest_frame(address: u64, display: DisplayConfiguration) -> Result<HostFrame, String> {
    if address < 0x10_000
        || display.width == 0
        || display.height == 0
        || display.width > 16_384
        || display.height > 16_384
        || display.pitch < display.width
    {
        return Err("invalid display address, extent, or pitch".to_owned());
    }
    let width = usize::try_from(display.width).map_err(|_| "display width does not fit usize")?;
    let height =
        usize::try_from(display.height).map_err(|_| "display height does not fit usize")?;
    let pitch = usize::try_from(display.pitch).map_err(|_| "display pitch does not fit usize")?;
    let linear_bytes = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .filter(|bytes| *bytes <= MAXIMUM_FRAME_BYTES)
        .ok_or("display frame byte count overflows or exceeds the limit")?;
    let (raw_bytes, tiled) = match display.tiling_mode {
        0 => {
            let blocks_wide = pitch.div_ceil(128);
            let blocks_high = height.div_ceil(128);
            (
                blocks_wide
                    .checked_mul(blocks_high)
                    .and_then(|blocks| blocks.checked_mul(65_536))
                    .ok_or("tiled display byte count overflows")?,
                true,
            )
        }
        1 => (
            pitch
                .checked_mul(height)
                .and_then(|pixels| pixels.checked_mul(4))
                .ok_or("linear display byte count overflows")?,
            false,
        ),
        mode => return Err(format!("unsupported VideoOut tiling mode {mode}")),
    };
    if raw_bytes > MAXIMUM_FRAME_BYTES {
        return Err("raw display allocation exceeds the snapshot limit".to_owned());
    }
    let mut raw = vec![0_u8; raw_bytes];
    read_process_memory(address, &mut raw)?;
    let bgra8 = if tiled {
        detile_mode27_bgra(&raw, width, height, pitch)?
    } else {
        let mut output = vec![0_u8; linear_bytes];
        for y in 0..height {
            let source = y * pitch * 4;
            let destination = y * width * 4;
            output[destination..destination + width * 4]
                .copy_from_slice(&raw[source..source + width * 4]);
        }
        output
    };
    let _ = display.pixel_format;
    Ok(HostFrame {
        width: display.width,
        height: display.height,
        bgra8,
    })
}

fn detile_mode27_bgra(
    raw: &[u8],
    width: usize,
    height: usize,
    pitch: usize,
) -> Result<Vec<u8>, String> {
    crate::texture::detile_mode27_bgra(raw, width, height, pitch)
}

#[cfg(test)]
fn mode27_offsets(block_x: usize, block_y: usize) -> &'static [u16] {
    static OFFSETS: OnceLock<[Box<[u16]>; 4]> = OnceLock::new();
    let tables = OFFSETS.get_or_init(|| {
        std::array::from_fn(|parity| {
            let parity_x = parity & 1;
            let parity_y = parity >> 1;
            let mut values = vec![0_u16; 128 * 128];
            for y in 0..128 {
                for x in 0..128 {
                    let global_x = x | (parity_x * 128);
                    let global_y = y | (parity_y * 128);
                    values[y * 128 + x] = u16::try_from(mode27_bgra_offset(global_x, global_y))
                        .expect("mode-27 byte offset fits u16");
                }
            }
            values.into_boxed_slice()
        })
    });
    &tables[(block_y & 1) * 2 + (block_x & 1)]
}

#[cfg(test)]
const fn mode27_bgra_offset(x: usize, y: usize) -> usize {
    ((y << 4) & 0x0070)
        ^ ((y << 5) & 0x0f00)
        ^ ((y << 9) & 0x1000)
        ^ ((y << 8) & 0x4000)
        ^ ((x << 2) & 0x000c)
        ^ ((x << 5) & 0x0380)
        ^ ((x << 4) & 0x0400)
        ^ ((x << 6) & 0x0800)
        ^ ((x << 9) & 0xa000)
}

fn execute_memory_copy(copy: &GpuMemoryCopy) -> Result<(), String> {
    if !matches!(copy.source_selection, 0 | 2 | 3) || !matches!(copy.destination_selection, 0 | 3) {
        return Err(format!(
            "unsupported selectors source={} destination={}",
            copy.source_selection, copy.destination_selection
        ));
    }
    let byte_count =
        usize::try_from(copy.byte_count).map_err(|_| "DMA byte count does not fit usize")?;
    if byte_count == 0 || byte_count > 512 * 1024 * 1024 {
        return Err(format!("invalid DMA byte count {byte_count:#x}"));
    }
    let bytes = if copy.source_selection == 2 {
        if !byte_count.is_multiple_of(4) || !copy.destination_address.is_multiple_of(4) {
            return Err("immediate DMA fill is not dword aligned".to_owned());
        }
        let value = (copy.source_address as u32).to_le_bytes();
        value
            .into_iter()
            .cycle()
            .take(byte_count)
            .collect::<Vec<_>>()
    } else if let Some(snapshot) = copy.source_snapshot.as_ref() {
        if snapshot.len() != byte_count {
            return Err("DMA source snapshot length mismatch".to_owned());
        }
        snapshot.to_vec()
    } else {
        let mut bytes = vec![0_u8; byte_count];
        read_process_memory(copy.source_address, &mut bytes)?;
        bytes
    };
    write_process_memory(copy.destination_address, &bytes)?;
    notify_wait_producers(copy.destination_address, copy.byte_count);
    Ok(())
}

fn execute_memory_write(write: &GpuMemoryWrite) -> Result<(), String> {
    if write.address == 0 || !write.address.is_multiple_of(4) {
        return Err("WRITE_DATA destination is null or unaligned".to_owned());
    }
    if write.increment_address {
        let mut bytes = Vec::with_capacity(write.values.len().saturating_mul(4));
        for value in &write.values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        write_process_memory(write.address, &bytes)?;
        notify_wait_producers(
            write.address,
            u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        );
    } else {
        for value in &write.values {
            write_process_memory(write.address, &value.to_le_bytes())?;
        }
        if !write.values.is_empty() {
            notify_wait_producers(write.address, 4);
        }
    }
    Ok(())
}

fn execute_memory_release(release: GpuMemoryRelease) -> Result<(), String> {
    if release.address == 0 {
        return Err("release destination is null".to_owned());
    }
    match release.value {
        GpuMemoryReleaseValue::Immediate32(value) => {
            if !release.address.is_multiple_of(4) {
                return Err("32-bit release destination is unaligned".to_owned());
            }
            write_process_memory(release.address, &value.to_le_bytes())?;
            remu_gpu::record_gpu_wait_producer(release.address, false, u64::from(value));
        }
        GpuMemoryReleaseValue::Immediate64(value) => {
            if !release.address.is_multiple_of(8) {
                return Err("64-bit release destination is unaligned".to_owned());
            }
            write_process_memory(release.address, &value.to_le_bytes())?;
            remu_gpu::record_gpu_wait_producer(release.address, true, value);
        }
        GpuMemoryReleaseValue::Timestamp => {
            if !release.address.is_multiple_of(8) {
                return Err("timestamp release destination is unaligned".to_owned());
            }
            let timestamp = gpu_timestamp();
            write_process_memory(release.address, &timestamp.to_le_bytes())?;
            remu_gpu::record_gpu_wait_producer(release.address, true, timestamp);
        }
    }
    Ok(())
}

fn gpu_timestamp() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    u64::try_from(START.get_or_init(Instant::now).elapsed().as_nanos())
        .unwrap_or(u64::MAX)
        .max(1)
}

fn memory_wait_is_satisfied(wait: GpuMemoryWait) -> bool {
    let mut bytes = [0_u8; 8];
    let width = if wait.is_64_bit { 8 } else { 4 };
    if read_process_memory(wait.address, &mut bytes[..width]).is_err() {
        return false;
    }
    let value = if wait.is_64_bit {
        u64::from_le_bytes(bytes)
    } else {
        u64::from(u32::from_le_bytes(
            bytes[..4].try_into().expect("four-byte slice"),
        ))
    };
    wait.is_satisfied_by(value)
}

fn notify_wait_producers(address: u64, byte_count: u64) {
    for (_, wait) in remu_gpu::pending_gpu_waits_overlapping(address, byte_count) {
        let mut bytes = [0_u8; 8];
        let width = if wait.is_64_bit { 8 } else { 4 };
        if read_process_memory(wait.address, &mut bytes[..width]).is_err() {
            continue;
        }
        let value = if wait.is_64_bit {
            u64::from_le_bytes(bytes)
        } else {
            u64::from(u32::from_le_bytes(
                bytes[..4].try_into().expect("four-byte slice"),
            ))
        };
        remu_gpu::record_gpu_wait_producer(wait.address, wait.is_64_bit, value);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{
        HostFrame, PresentationState, compose_pad_state, compose_system_ui, detile_mode27_bgra,
        is_fullscreen_shortcut, merge_xinput, mode27_offsets, xinput_vibration,
    };
    use remu_gpu::{
        HostDialogButtons, HostDialogKind, HostDialogRequest, HostPadState, HostPadVibration,
        PAD_BUTTON_CROSS, PAD_BUTTON_L2, PAD_BUTTON_OPTIONS, PAD_BUTTON_R3, PAD_BUTTON_RIGHT,
        PAD_BUTTON_SQUARE,
    };
    use windows::Win32::UI::Input::XboxController::{
        XINPUT_GAMEPAD, XINPUT_GAMEPAD_A, XINPUT_GAMEPAD_DPAD_RIGHT,
        XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE, XINPUT_GAMEPAD_RIGHT_THUMB, XINPUT_GAMEPAD_START,
        XINPUT_GAMEPAD_X,
    };
    use winit::event::ElementState;
    use winit::keyboard::{KeyCode, ModifiersState};

    #[test]
    fn presentation_pauses_for_every_windows_lifecycle_constraint() {
        let mut state = PresentationState::default();
        assert!(!state.is_paused());
        state.occluded = true;
        assert!(state.is_paused());
        state.occluded = false;
        state.zero_sized = true;
        assert!(state.is_paused());
        state.zero_sized = false;
        state.suspended = true;
        assert!(state.is_paused());
        state.suspended = false;
        assert!(!state.is_paused());
    }

    #[test]
    fn fullscreen_shortcuts_require_a_press_and_alt_for_enter() {
        assert!(is_fullscreen_shortcut(
            KeyCode::F11,
            ElementState::Pressed,
            ModifiersState::default()
        ));
        assert!(is_fullscreen_shortcut(
            KeyCode::Enter,
            ElementState::Pressed,
            ModifiersState::ALT
        ));
        assert!(!is_fullscreen_shortcut(
            KeyCode::Enter,
            ElementState::Pressed,
            ModifiersState::default()
        ));
        assert!(!is_fullscreen_shortcut(
            KeyCode::F11,
            ElementState::Released,
            ModifiersState::default()
        ));
    }

    #[test]
    fn inactive_window_neutralizes_keyboard_and_controller_input() {
        let mut keys = HashSet::new();
        keys.insert(KeyCode::KeyD);
        keys.insert(KeyCode::Enter);
        let mut controller = XINPUT_GAMEPAD::default();
        controller.wButtons = XINPUT_GAMEPAD_A | XINPUT_GAMEPAD_DPAD_RIGHT;
        controller.bRightTrigger = 255;
        controller.sThumbLX = i16::MAX;

        assert_eq!(
            compose_pad_state(&keys, Some(controller), false),
            HostPadState::default()
        );
        let active = compose_pad_state(&keys, Some(controller), true);
        assert_ne!(active, HostPadState::default());
        assert_ne!(active.buttons & PAD_BUTTON_CROSS, 0);
        assert_ne!(active.buttons & PAD_BUTTON_RIGHT, 0);
        assert_eq!(active.left_x, 255);
        assert_eq!(active.r2, 255);
    }

    #[test]
    fn guest_vibration_expands_exactly_to_xinput_motor_ranges() {
        let stopped = xinput_vibration(HostPadVibration::default());
        assert_eq!(stopped.wLeftMotorSpeed, 0);
        assert_eq!(stopped.wRightMotorSpeed, 0);

        let vibration = xinput_vibration(HostPadVibration {
            large_motor: u8::MAX,
            small_motor: 0x80,
        });
        assert_eq!(vibration.wLeftMotorSpeed, u16::MAX);
        assert_eq!(vibration.wRightMotorSpeed, 0x8080);
    }

    #[test]
    fn mode27_scanout_round_trips_multiple_blocks_and_pitch_padding() {
        let width = 173_usize;
        let height = 139_usize;
        let pitch = 256_usize;
        let blocks_per_row = pitch.div_ceil(128);
        let mut raw = vec![0_u8; blocks_per_row * height.div_ceil(128) * 65_536];
        let mut expected = vec![0_u8; width * height * 4];
        for y in 0..height {
            for x in 0..width {
                let pixel = [x as u8, y as u8, (x ^ y) as u8, 255];
                let destination = (y * width + x) * 4;
                expected[destination..destination + 4].copy_from_slice(&pixel);
                let block_x = x / 128;
                let block_y = y / 128;
                let block = (block_y * blocks_per_row + block_x) * 65_536;
                let offset =
                    usize::from(mode27_offsets(block_x, block_y)[(y & 127) * 128 + (x & 127)]);
                raw[block + offset..block + offset + 4].copy_from_slice(&pixel);
            }
        }
        assert_eq!(
            detile_mode27_bgra(&raw, width, height, pitch).unwrap(),
            expected
        );
    }

    #[test]
    fn system_ui_alpha_blends_rgba_over_bgra() {
        let frame = HostFrame {
            width: 64,
            height: 64,
            bgra8: vec![0; 64 * 64 * 4],
        };
        let dialog = HostDialogRequest {
            kind: HostDialogKind::Message,
            title: "Title".to_owned(),
            message: "Message".to_owned(),
            buttons: HostDialogButtons::Ok,
            sequence: 1,
        };
        let composed = compose_system_ui(frame, Some(&dialog), None);
        assert!(composed.bgra8.iter().any(|byte| *byte != 0));
    }

    #[test]
    fn xinput_maps_buttons_triggers_and_full_stick_ranges() {
        let mut state = HostPadState::default();
        merge_xinput(
            &mut state,
            XINPUT_GAMEPAD {
                wButtons: XINPUT_GAMEPAD_A
                    | XINPUT_GAMEPAD_X
                    | XINPUT_GAMEPAD_START
                    | XINPUT_GAMEPAD_RIGHT_THUMB,
                bLeftTrigger: 200,
                bRightTrigger: 20,
                sThumbLX: i16::MIN,
                sThumbLY: i16::MAX,
                sThumbRX: i16::MAX,
                sThumbRY: i16::MIN,
            },
        );
        assert_eq!(state.left_x, 0);
        assert_eq!(state.left_y, 0);
        assert_eq!(state.right_x, 255);
        assert_eq!(state.right_y, 255);
        assert_eq!(state.l2, 200);
        assert_eq!(state.r2, 20);
        assert_eq!(
            state.buttons,
            PAD_BUTTON_CROSS
                | PAD_BUTTON_SQUARE
                | PAD_BUTTON_OPTIONS
                | PAD_BUTTON_R3
                | PAD_BUTTON_L2
        );
    }

    #[test]
    fn xinput_deadzone_preserves_keyboard_axes() {
        let mut state = HostPadState {
            left_x: 255,
            left_y: 0,
            ..HostPadState::default()
        };
        merge_xinput(
            &mut state,
            XINPUT_GAMEPAD {
                sThumbLX: XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE.0 as i16,
                sThumbLY: -(XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE.0 as i16),
                ..XINPUT_GAMEPAD::default()
            },
        );
        assert_eq!((state.left_x, state.left_y), (255, 0));
    }
}
