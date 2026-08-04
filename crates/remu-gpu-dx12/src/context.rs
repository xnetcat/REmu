// SPDX-License-Identifier: GPL-2.0-or-later

//! Native D3D12 queue and resource validation.

use std::ffi::c_void;
use std::mem::ManuallyDrop;

use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0, WAIT_TIMEOUT};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC, D3D12_COMMAND_QUEUE_FLAG_NONE,
    D3D12_FENCE_FLAG_NONE, D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_DEFAULT,
    D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_BARRIER,
    D3D12_RESOURCE_BARRIER_0, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
    D3D12_RESOURCE_BARRIER_FLAG_NONE, D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_DESC,
    D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
    D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_GENERIC_READ,
    D3D12_RESOURCE_TRANSITION_BARRIER, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ID3D12CommandAllocator,
    ID3D12CommandList, ID3D12CommandQueue, ID3D12Device, ID3D12Fence, ID3D12GraphicsCommandList,
    ID3D12PipelineState, ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::Interface;

use crate::backend::{backend_error, backend_error_with_device, create_default_device};
use crate::diagnostics::name_object;
use crate::{Dx12ProbeError, Dx12SelfTestReport};

const COPY_BYTES: usize = 64 * 1024;
const FENCE_TIMEOUT_MS: u32 = 5_000;

struct OwnedEvent(HANDLE);

impl Drop for OwnedEvent {
    fn drop(&mut self) {
        // SAFETY: this type is constructed only from a successful `CreateEventW` call and owns
        // the returned handle exactly once.
        let _ = unsafe { CloseHandle(self.0) };
    }
}

pub(super) fn run_self_test() -> Result<Dx12SelfTestReport, Dx12ProbeError> {
    let native = create_default_device()?;
    let device = native.device;
    let queue_description = D3D12_COMMAND_QUEUE_DESC {
        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
        Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
        ..D3D12_COMMAND_QUEUE_DESC::default()
    };
    // SAFETY: every descriptor is fully initialized and each returned COM interface owns its
    // reference. No command list is submitted until all resources below remain live.
    let queue: ID3D12CommandQueue =
        unsafe { device.CreateCommandQueue(&raw const queue_description) }
            .map_err(|error| backend_error("ID3D12Device::CreateCommandQueue", &error))?;
    name_object(&queue, "REMu self-test queue");
    // SAFETY: the direct allocator matches the direct command list type.
    let allocator: ID3D12CommandAllocator =
        unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
            .map_err(|error| backend_error("ID3D12Device::CreateCommandAllocator", &error))?;
    // SAFETY: `allocator` is live and no initial pipeline state is needed for copy commands.
    let commands: ID3D12GraphicsCommandList = unsafe {
        device.CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            &allocator,
            None::<&ID3D12PipelineState>,
        )
    }
    .map_err(|error| backend_error("ID3D12Device::CreateCommandList", &error))?;
    name_object(&commands, "REMu self-test command list");

    let upload = create_buffer(
        &device,
        COPY_BYTES,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;
    let gpu = create_buffer(
        &device,
        COPY_BYTES,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COPY_DEST,
    )?;
    let readback = create_buffer(
        &device,
        COPY_BYTES,
        D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_STATE_COPY_DEST,
    )?;

    let expected = (0..COPY_BYTES)
        .map(|index| (index.wrapping_mul(37).wrapping_add(11) & 0xff) as u8)
        .collect::<Vec<_>>();
    write_upload(&upload, &expected)?;

    // SAFETY: source and destination buffers are live, large enough, and in the required states.
    let copy_bytes = u64::try_from(COPY_BYTES).expect("self-test buffer size fits u64");
    unsafe { commands.CopyBufferRegion(&gpu, 0, &upload, 0, copy_bytes) };
    let mut barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: ManuallyDrop::new(Some(gpu.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: D3D12_RESOURCE_STATE_COPY_DEST,
                StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
            }),
        },
    };
    // SAFETY: the union contains its Transition member and the resource is in COPY_DEST.
    unsafe { commands.ResourceBarrier(std::slice::from_ref(&barrier)) };
    // SAFETY: the barrier above makes the source copy-readable; the readback buffer is COPY_DEST.
    unsafe { commands.CopyBufferRegion(&readback, 0, &gpu, 0, copy_bytes) };
    // SAFETY: the transition member is active. Release the cloned COM reference retained only for
    // the synchronous ResourceBarrier recording call.
    unsafe {
        ManuallyDrop::drop(&mut (*barrier.Anonymous.Transition).pResource);
    }
    // SAFETY: all commands are recorded and the list has not previously been closed.
    unsafe { commands.Close() }
        .map_err(|error| backend_error("ID3D12GraphicsCommandList::Close", &error))?;
    let command_list: ID3D12CommandList = commands
        .cast()
        .map_err(|error| backend_error("ID3D12GraphicsCommandList::cast", &error))?;
    // SAFETY: the command list is closed and all referenced objects outlive queue completion.
    unsafe { queue.ExecuteCommandLists(&[Some(command_list)]) };
    // SAFETY: the fence belongs to this device and begins below the signaled value.
    let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
        .map_err(|error| backend_error("ID3D12Device::CreateFence", &error))?;
    name_object(&fence, "REMu self-test fence");
    // SAFETY: the queue and fence are live and the value advances monotonically.
    unsafe { queue.Signal(&fence, 1) }
        .map_err(|error| backend_error("ID3D12CommandQueue::Signal", &error))?;
    wait_for_fence(&fence, 1)?;

    let actual = read_readback(&readback, COPY_BYTES)?;
    if actual != expected {
        return Err(Dx12ProbeError::SelfTestMismatch);
    }
    Ok(Dx12SelfTestReport {
        adapter: native.capabilities.name,
        copied_bytes: COPY_BYTES,
        checksum: actual.iter().fold(0_u64, |checksum, byte| {
            checksum.wrapping_add(u64::from(*byte))
        }),
    })
}

pub(super) fn create_buffer(
    device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
    byte_count: usize,
    heap_type: windows::Win32::Graphics::Direct3D12::D3D12_HEAP_TYPE,
    initial_state: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES,
) -> Result<ID3D12Resource, Dx12ProbeError> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: heap_type,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..D3D12_HEAP_PROPERTIES::default()
    };
    let description = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: u64::try_from(byte_count).expect("D3D12 buffer size fits u64"),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_NONE,
        ..D3D12_RESOURCE_DESC::default()
    };
    let mut resource = None;
    // SAFETY: heap and resource descriptors are complete and the output slot is writable.
    unsafe {
        device.CreateCommittedResource(
            &raw const heap,
            D3D12_HEAP_FLAG_NONE,
            &raw const description,
            initial_state,
            None,
            &raw mut resource,
        )
    }
    .map_err(|error| backend_error("ID3D12Device::CreateCommittedResource", &error))?;
    resource.ok_or(Dx12ProbeError::MissingCreatedResource)
}

pub(super) fn write_upload(resource: &ID3D12Resource, bytes: &[u8]) -> Result<(), Dx12ProbeError> {
    write_upload_at(resource, 0, bytes)
}

pub(super) fn write_upload_at(
    resource: &ID3D12Resource,
    offset: usize,
    bytes: &[u8],
) -> Result<(), Dx12ProbeError> {
    let mut mapped = std::ptr::null_mut::<c_void>();
    let no_read = windows::Win32::Graphics::Direct3D12::D3D12_RANGE { Begin: 0, End: 0 };
    // SAFETY: upload resources are CPU-writable and `mapped` receives the persistent mapping.
    unsafe { resource.Map(0, Some(&raw const no_read), Some(&raw mut mapped)) }
        .map_err(|error| backend_error("ID3D12Resource::Map(upload)", &error))?;
    // SAFETY: the caller allocated the resource for `offset + bytes.len()` bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.cast::<u8>().add(offset), bytes.len())
    };
    let written = windows::Win32::Graphics::Direct3D12::D3D12_RANGE {
        Begin: offset,
        End: offset + bytes.len(),
    };
    // SAFETY: subresource zero is mapped and `written` names the initialized range.
    unsafe { resource.Unmap(0, Some(&raw const written)) };
    Ok(())
}

pub(super) fn read_readback(
    resource: &ID3D12Resource,
    byte_count: usize,
) -> Result<Vec<u8>, Dx12ProbeError> {
    let mut mapped = std::ptr::null_mut::<c_void>();
    let read = windows::Win32::Graphics::Direct3D12::D3D12_RANGE {
        Begin: 0,
        End: byte_count,
    };
    // SAFETY: queue completion makes the readback allocation CPU-readable.
    unsafe { resource.Map(0, Some(&raw const read), Some(&raw mut mapped)) }
        .map_err(|error| backend_error("ID3D12Resource::Map(readback)", &error))?;
    // SAFETY: the mapping contains the requested byte count for the life of this copy.
    let bytes = unsafe { std::slice::from_raw_parts(mapped.cast::<u8>(), byte_count) }.to_vec();
    let no_write = windows::Win32::Graphics::Direct3D12::D3D12_RANGE { Begin: 0, End: 0 };
    // SAFETY: subresource zero is mapped and the CPU wrote no bytes.
    unsafe { resource.Unmap(0, Some(&raw const no_write)) };
    Ok(bytes)
}

pub(super) fn wait_for_fence(fence: &ID3D12Fence, value: u64) -> Result<(), Dx12ProbeError> {
    wait_for_fence_timeout(fence, value, FENCE_TIMEOUT_MS)
}

fn wait_for_fence_timeout(
    fence: &ID3D12Fence,
    value: u64,
    timeout_ms: u32,
) -> Result<(), Dx12ProbeError> {
    // SAFETY: reading the completed value of a live fence is side-effect free.
    if unsafe { fence.GetCompletedValue() } >= value {
        return Ok(());
    }

    // Microsoft documents SetEventOnCompletion plus WaitForSingleObject as the CPU-side D3D12
    // fence wait. It blocks the calling thread in the kernel instead of repeatedly rescheduling a
    // busy poll, while retaining a bounded timeout for wedged or removed devices.
    // SAFETY: null security attributes and an unnamed auto-reset event have no borrowed lifetime.
    let event = OwnedEvent(
        unsafe { CreateEventW(None, false, false, windows::core::PCWSTR::null()) }
            .map_err(|error| backend_error("CreateEventW(D3D12 fence)", &error))?,
    );
    // SAFETY: the fence and event are live; `event` remains open through the wait below.
    unsafe { fence.SetEventOnCompletion(value, event.0) }
        .map_err(|error| backend_error("ID3D12Fence::SetEventOnCompletion", &error))?;
    // SAFETY: `event` is a live waitable event handle.
    let status = unsafe { WaitForSingleObject(event.0, timeout_ms) };
    if status == WAIT_OBJECT_0 {
        // Device removal raises outstanding fence events with UINT64_MAX. Surface the actual
        // removal HRESULT instead of reporting false queue completion.
        let completed = unsafe { fence.GetCompletedValue() };
        if completed == u64::MAX
            && let Some(error) = device_removed_error(fence)
        {
            return Err(error);
        }
        if completed >= value {
            Ok(())
        } else {
            Err(Dx12ProbeError::FenceIncomplete {
                expected: value,
                completed,
            })
        }
    } else if status == WAIT_TIMEOUT {
        if let Some(error) = device_removed_error(fence) {
            Err(error)
        } else {
            Err(Dx12ProbeError::FenceTimeout { value })
        }
    } else {
        Err(Dx12ProbeError::FenceWaitFailed { status: status.0 })
    }
}

fn device_removed_error(fence: &ID3D12Fence) -> Option<Dx12ProbeError> {
    let mut device = None::<ID3D12Device>;
    // SAFETY: the output slot receives an owned interface for the live fence's parent device.
    unsafe { fence.GetDevice(&raw mut device) }.ok()?;
    let device = device?;
    // SAFETY: querying a live device's removal reason has no pointer preconditions.
    let error = unsafe { device.GetDeviceRemovedReason() }.err()?;
    Some(backend_error_with_device(
        "ID3D12Device::GetDeviceRemovedReason",
        &error,
        &device,
    ))
}

#[cfg(test)]
mod tests {
    use super::{D3D12_FENCE_FLAG_NONE, wait_for_fence_timeout};
    use crate::Dx12ProbeError;
    use crate::backend::create_default_device;
    use std::time::Duration;
    use windows::Win32::Graphics::Direct3D12::ID3D12Fence;

    #[test]
    fn event_wait_observes_async_completion_and_timeout() {
        let device = create_default_device()
            .expect("the physical D3D12 test suite requires a compatible adapter")
            .device;
        // SAFETY: the device is live and the fence starts at zero.
        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
            .expect("test fence should be created");
        let signal_fence = fence.clone();
        let signal = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(10));
            // SAFETY: the cloned fence remains live on this thread.
            unsafe { signal_fence.Signal(1) }.expect("CPU fence signal should succeed");
        });

        wait_for_fence_timeout(&fence, 1, 1_000)
            .expect("event wait should observe the asynchronous signal");
        signal.join().expect("signal thread should finish");
        assert!(matches!(
            wait_for_fence_timeout(&fence, 2, 1),
            Err(Dx12ProbeError::FenceTimeout { value: 2 })
        ));
    }
}
