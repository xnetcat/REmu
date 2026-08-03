// SPDX-License-Identifier: GPL-2.0-or-later

//! Opt-in D3D12 validation and bounded device-removal diagnostics.

use std::ffi::OsStr;
use std::sync::OnceLock;

use windows::Win32::Graphics::Direct3D12::{
    D3D12_AUTO_BREADCRUMB_NODE1, D3D12_AUTO_BREADCRUMB_OP, D3D12_DRED_ALLOCATION_NODE1,
    D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT1, D3D12_DRED_ENABLEMENT_FORCED_ON,
    D3D12_DRED_PAGE_FAULT_OUTPUT1, D3D12_DRED_PAGE_FAULT_OUTPUT2, D3D12GetDebugInterface,
    ID3D12Debug, ID3D12Debug1, ID3D12Device, ID3D12DeviceRemovedExtendedData1,
    ID3D12DeviceRemovedExtendedData2, ID3D12DeviceRemovedExtendedDataSettings,
    ID3D12DeviceRemovedExtendedDataSettings1, ID3D12Object,
};
use windows::core::{Interface, PCWSTR};

const MAX_DRED_NODES: usize = 256;
const MAX_DRED_ALLOCATIONS: usize = 64;
const MAX_DRED_CONTEXTS: usize = 32;
const MAX_DRED_STRING_CODE_UNITS: usize = 4_096;
const DRED_HISTORY_CAPACITY: usize = 65_536;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct DiagnosticOptions {
    pub debug_layer: bool,
    pub gpu_validation: bool,
    pub dred: bool,
}

static REQUESTED_OPTIONS: OnceLock<DiagnosticOptions> = OnceLock::new();

pub(super) fn set_requested_options(options: DiagnosticOptions) -> Result<(), &'static str> {
    REQUESTED_OPTIONS
        .set(options)
        .map_err(|_| "D3D12 diagnostics were already configured")
}

pub(super) fn configure_before_device_creation() {
    let requested = REQUESTED_OPTIONS.get().copied().unwrap_or_default();
    let debug_layer = requested.debug_layer || diagnostic_flag("REMU_D3D12_DEBUG");
    let gpu_validation = requested.gpu_validation || diagnostic_flag("REMU_D3D12_GPU_VALIDATION");
    if debug_layer || gpu_validation {
        configure_debug_layer(gpu_validation);
    }
    if requested.dred || diagnostic_flag("REMU_D3D12_DRED") {
        configure_dred();
    }
}

fn configure_debug_layer(gpu_validation: bool) {
    let mut debug = None::<ID3D12Debug1>;
    // SAFETY: the output slot receives an owned process-global diagnostics interface. Microsoft
    // requires this configuration to happen before D3D12 device creation.
    match unsafe { D3D12GetDebugInterface(&raw mut debug) } {
        Ok(()) => {
            let debug = debug.expect("successful debug query returns an interface");
            // SAFETY: these process-global switches are configured before the caller creates a
            // device. GPU validation is deliberately opt-in because it is substantially slower.
            unsafe {
                debug.EnableDebugLayer();
                debug.SetEnableGPUBasedValidation(gpu_validation);
            }
            eprintln!("[REMu][D3D12] debug layer enabled gpu_validation={gpu_validation}");
        }
        Err(error) => {
            // Some older SDK-layer installations expose only ID3D12Debug. Retain ordinary debug
            // validation even when the newer GPU-validation interface is unavailable.
            let mut fallback = None::<ID3D12Debug>;
            // SAFETY: this is the same pre-device process-global query with an older interface.
            match unsafe { D3D12GetDebugInterface(&raw mut fallback) } {
                Ok(()) => {
                    // SAFETY: the fallback interface is live and no device has been created yet.
                    unsafe {
                        fallback
                            .expect("successful debug query returns an interface")
                            .EnableDebugLayer()
                    };
                    eprintln!(
                        "[REMu][D3D12] debug layer enabled; GPU validation unavailable: {error}"
                    );
                }
                Err(fallback_error) => {
                    eprintln!("[REMu][D3D12] debug layer unavailable: {fallback_error}")
                }
            }
        }
    }
}

fn configure_dred() {
    let mut settings = None::<ID3D12DeviceRemovedExtendedDataSettings1>;
    // SAFETY: DRED settings are process-global and this is called before device creation.
    match unsafe { D3D12GetDebugInterface(&raw mut settings) } {
        Ok(()) => {
            let settings = settings.expect("successful DRED query returns an interface");
            // SAFETY: all three settings must be forced on before the D3D12 device is created.
            unsafe {
                settings.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                settings.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                settings.SetBreadcrumbContextEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
            eprintln!(
                "[REMu][D3D12] DRED enabled auto_breadcrumbs=true page_faults=true contexts=true"
            );
        }
        Err(error) => {
            let mut fallback = None::<ID3D12DeviceRemovedExtendedDataSettings>;
            // SAFETY: this requests the DRED 1.0/1.1 configuration interface before device
            // creation when breadcrumb contexts are not supported by the installed runtime.
            match unsafe { D3D12GetDebugInterface(&raw mut fallback) } {
                Ok(()) => {
                    let settings = fallback.expect("successful DRED query returns an interface");
                    // SAFETY: the fallback settings interface is live for these synchronous calls.
                    unsafe {
                        settings.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                        settings.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                    }
                    eprintln!(
                        "[REMu][D3D12] DRED enabled auto_breadcrumbs=true page_faults=true contexts=false ({error})"
                    );
                }
                Err(fallback_error) => {
                    eprintln!("[REMu][D3D12] DRED unavailable: {fallback_error}");
                }
            }
        }
    }
}

pub(super) fn name_object<T: Interface>(object: &T, name: &str) {
    let Ok(object) = object.cast::<ID3D12Object>() else {
        return;
    };
    let wide = name.encode_utf16().chain(Some(0)).collect::<Vec<_>>();
    // SAFETY: `wide` is terminated and remains live for the synchronous SetName call.
    let _ = unsafe { object.SetName(PCWSTR(wide.as_ptr())) };
}

pub(super) fn device_removed_report(device: &ID3D12Device) -> String {
    let reason = match unsafe { device.GetDeviceRemovedReason() } {
        Ok(()) => return "device_state=available".to_owned(),
        Err(error) => format!(
            "reason=0x{:08x} ({})",
            error.code().0.cast_unsigned(),
            error.message()
        ),
    };

    let Ok(dred1) = device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
        return format!("{reason}; dred=interface-unavailable");
    };
    // SAFETY: the device is removed and owns the returned diagnostic nodes for the duration of
    // this immediate bounded traversal.
    let breadcrumbs = unsafe { dred1.GetAutoBreadcrumbsOutput1() }
        .map(|output| summarize_breadcrumbs(&output))
        .unwrap_or_else(|error| format!("unavailable({error})"));

    let (device_state, page_fault) = match device.cast::<ID3D12DeviceRemovedExtendedData2>() {
        Ok(dred2) => {
            let state = unsafe { dred2.GetDeviceState() }.0;
            let mut output = D3D12_DRED_PAGE_FAULT_OUTPUT2::default();
            // SAFETY: `output` is the exact writable structure required by this DRED interface.
            let page_fault = unsafe { dred2.GetPageFaultAllocationOutput2(&raw mut output) }
                .map(|()| summarize_page_fault2(&output))
                .unwrap_or_else(|error| format!("unavailable({error})"));
            (state.to_string(), page_fault)
        }
        Err(_) => {
            // SAFETY: the removed device owns the returned diagnostic nodes during this traversal.
            let page_fault = unsafe { dred1.GetPageFaultAllocationOutput1() }
                .map(|output| summarize_page_fault1(&output))
                .unwrap_or_else(|error| format!("unavailable({error})"));
            ("legacy".to_owned(), page_fault)
        }
    };
    format!(
        "{reason}; dred_state={device_state}; breadcrumbs={breadcrumbs}; page_fault={page_fault}"
    )
}

fn summarize_breadcrumbs(output: &D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT1) -> String {
    let mut summaries = Vec::new();
    let mut node_pointer = output.pHeadAutoBreadcrumbNode;
    for _ in 0..MAX_DRED_NODES {
        if node_pointer.is_null() {
            break;
        }
        // SAFETY: DRED owns a live linked list until the device is released. The traversal is
        // bounded so a corrupt/cyclic diagnostic list cannot loop forever.
        let node = unsafe { &*node_pointer };
        summaries.push(summarize_breadcrumb_node(node));
        node_pointer = node.pNext;
    }
    if !node_pointer.is_null() {
        summaries.push("truncated".to_owned());
    }
    if summaries.is_empty() {
        "none".to_owned()
    } else {
        format!("[{}]", summaries.join(", "))
    }
}

fn summarize_breadcrumb_node(node: &D3D12_AUTO_BREADCRUMB_NODE1) -> String {
    let queue = dred_name(node.pCommandQueueDebugNameW, node.pCommandQueueDebugNameA)
        .unwrap_or_else(|| "unnamed-queue".to_owned());
    let list = dred_name(node.pCommandListDebugNameW, node.pCommandListDebugNameA)
        .unwrap_or_else(|| "unnamed-list".to_owned());
    let completed = if node.pLastBreadcrumbValue.is_null() {
        None
    } else {
        // SAFETY: the DRED node owns one live breadcrumb counter.
        Some(unsafe { *node.pLastBreadcrumbValue })
    };
    let completed_text = completed.map_or_else(|| "unknown".to_owned(), |value| value.to_string());
    let mut detail = format!(
        "queue={queue:?} list={list:?} completed={completed_text}/{}",
        node.BreadcrumbCount
    );
    if let Some(completed) = completed {
        if completed > 0
            && let Some(operation) = breadcrumb_operation(node, completed - 1)
        {
            detail.push_str(&format!(" last={}", breadcrumb_op_name(operation)));
        }
        if completed < node.BreadcrumbCount
            && let Some(operation) = breadcrumb_operation(node, completed)
        {
            detail.push_str(&format!(" next={}", breadcrumb_op_name(operation)));
        }
    }
    let contexts = breadcrumb_contexts(node);
    if !contexts.is_empty() {
        detail.push_str(&format!(" contexts=[{}]", contexts.join(",")));
    }
    detail
}

fn breadcrumb_operation(
    node: &D3D12_AUTO_BREADCRUMB_NODE1,
    operation_index: u32,
) -> Option<D3D12_AUTO_BREADCRUMB_OP> {
    if node.pCommandHistory.is_null() || node.BreadcrumbCount == 0 {
        return None;
    }
    let retained = usize::try_from(node.BreadcrumbCount)
        .unwrap_or(usize::MAX)
        .min(DRED_HISTORY_CAPACITY);
    let index = usize::try_from(operation_index).ok()? % retained;
    // SAFETY: DRED retains at most 64K history entries and `index` is reduced to that capacity.
    Some(unsafe { *node.pCommandHistory.add(index) })
}

fn breadcrumb_contexts(node: &D3D12_AUTO_BREADCRUMB_NODE1) -> Vec<String> {
    if node.pBreadcrumbContexts.is_null() {
        return Vec::new();
    }
    let count = usize::try_from(node.BreadcrumbContextsCount)
        .unwrap_or(usize::MAX)
        .min(MAX_DRED_CONTEXTS);
    (0..count)
        .filter_map(|index| {
            // SAFETY: DRED reports `BreadcrumbContextsCount` entries; the traversal is capped.
            let context = unsafe { &*node.pBreadcrumbContexts.add(index) };
            wide_string(context.pContextString)
                .map(|text| format!("{}:{text:?}", context.BreadcrumbIndex))
        })
        .collect()
}

fn summarize_page_fault1(output: &D3D12_DRED_PAGE_FAULT_OUTPUT1) -> String {
    summarize_page_fault(
        output.PageFaultVA,
        None,
        output.pHeadExistingAllocationNode,
        output.pHeadRecentFreedAllocationNode,
    )
}

fn summarize_page_fault2(output: &D3D12_DRED_PAGE_FAULT_OUTPUT2) -> String {
    summarize_page_fault(
        output.PageFaultVA,
        Some(output.PageFaultFlags.0),
        output.pHeadExistingAllocationNode,
        output.pHeadRecentFreedAllocationNode,
    )
}

fn summarize_page_fault(
    address: u64,
    flags: Option<i32>,
    existing: *const D3D12_DRED_ALLOCATION_NODE1,
    recently_freed: *const D3D12_DRED_ALLOCATION_NODE1,
) -> String {
    let existing = summarize_allocations(existing);
    let recently_freed = summarize_allocations(recently_freed);
    let flags = flags.map_or_else(String::new, |flags| format!(" flags=0x{flags:x}"));
    format!(
        "va=0x{address:x}{flags} existing=[{}] recently_freed=[{}]",
        existing.join(","),
        recently_freed.join(",")
    )
}

fn summarize_allocations(mut node_pointer: *const D3D12_DRED_ALLOCATION_NODE1) -> Vec<String> {
    let mut allocations = Vec::new();
    for _ in 0..MAX_DRED_ALLOCATIONS {
        if node_pointer.is_null() {
            break;
        }
        // SAFETY: the removed device owns the DRED allocation list during this bounded traversal.
        let node = unsafe { &*node_pointer };
        let name =
            dred_name(node.ObjectNameW, node.ObjectNameA).unwrap_or_else(|| "unnamed".to_owned());
        allocations.push(format!("{}:{name:?}", node.AllocationType.0));
        node_pointer = node.pNext;
    }
    if !node_pointer.is_null() {
        allocations.push("truncated".to_owned());
    }
    allocations
}

fn dred_name(wide: PCWSTR, ansi: *const u8) -> Option<String> {
    wide_string(wide).or_else(|| ansi_string(ansi))
}

fn wide_string(value: PCWSTR) -> Option<String> {
    if value.is_null() {
        return None;
    }
    let mut length = 0;
    while length < MAX_DRED_STRING_CODE_UNITS {
        // SAFETY: DRED supplies a null-terminated diagnostic name. The scan is explicitly bounded.
        if unsafe { *value.0.add(length) } == 0 {
            break;
        }
        length += 1;
    }
    // SAFETY: the preceding bounded scan proved all selected code units readable and non-null.
    Some(String::from_utf16_lossy(unsafe {
        std::slice::from_raw_parts(value.0, length)
    }))
}

fn ansi_string(value: *const u8) -> Option<String> {
    if value.is_null() {
        return None;
    }
    let mut length = 0;
    while length < MAX_DRED_STRING_CODE_UNITS {
        // SAFETY: DRED supplies a null-terminated diagnostic name. The scan is explicitly bounded.
        if unsafe { *value.add(length) } == 0 {
            break;
        }
        length += 1;
    }
    // SAFETY: the preceding bounded scan proved all selected bytes readable and non-null.
    Some(String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(value, length) }).into_owned())
}

fn breadcrumb_op_name(operation: D3D12_AUTO_BREADCRUMB_OP) -> &'static str {
    match operation.0 {
        0 => "SetMarker",
        1 => "BeginEvent",
        2 => "EndEvent",
        3 => "DrawInstanced",
        4 => "DrawIndexedInstanced",
        5 => "ExecuteIndirect",
        6 => "Dispatch",
        7 => "CopyBufferRegion",
        8 => "CopyTextureRegion",
        9 => "CopyResource",
        10 => "CopyTiles",
        11 => "ResolveSubresource",
        12 => "ClearRenderTargetView",
        13 => "ClearUnorderedAccessView",
        14 => "ClearDepthStencilView",
        15 => "ResourceBarrier",
        16 => "ExecuteBundle",
        17 => "Present",
        18 => "ResolveQueryData",
        19 => "BeginSubmission",
        20 => "EndSubmission",
        23 => "AtomicCopyBufferUint",
        24 => "AtomicCopyBufferUint64",
        25 => "ResolveSubresourceRegion",
        26 => "WriteBufferImmediate",
        31 => "BuildRaytracingAccelerationStructure",
        33 => "CopyRaytracingAccelerationStructure",
        34 => "DispatchRays",
        39 => "SetPipelineState1",
        42 => "DispatchMesh",
        45 => "Barrier",
        46 => "BeginCommandList",
        47 => "DispatchGraph",
        48 => "SetProgram",
        _ => "Other",
    }
}

fn diagnostic_flag(name: &str) -> bool {
    parse_diagnostic_flag(std::env::var_os(name).as_deref())
}

fn parse_diagnostic_flag(value: Option<&OsStr>) -> bool {
    value.is_some_and(|value| {
        let value = value.to_string_lossy();
        value == "1"
            || value.eq_ignore_ascii_case("true")
            || value.eq_ignore_ascii_case("yes")
            || value.eq_ignore_ascii_case("on")
    })
}

#[cfg(test)]
mod tests {
    use std::ffi::{CString, OsStr};

    use windows::Win32::Graphics::Direct3D12::{
        D3D12_AUTO_BREADCRUMB_NODE1, D3D12_AUTO_BREADCRUMB_OP_COPYBUFFERREGION,
        D3D12_AUTO_BREADCRUMB_OP_DRAWINDEXEDINSTANCED, D3D12_DRED_ALLOCATION_NODE1,
        D3D12_DRED_ALLOCATION_TYPE_RESOURCE, D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT1,
        D3D12_DRED_PAGE_FAULT_OUTPUT1,
    };

    use super::{parse_diagnostic_flag, summarize_breadcrumbs, summarize_page_fault1};

    #[test]
    fn diagnostic_flags_accept_explicit_truthy_values_only() {
        for value in ["1", "true", "TRUE", "yes", "on"] {
            assert!(parse_diagnostic_flag(Some(OsStr::new(value))));
        }
        for value in ["", "0", "false", "disabled", "10"] {
            assert!(!parse_diagnostic_flag(Some(OsStr::new(value))));
        }
        assert!(!parse_diagnostic_flag(None));
    }

    #[test]
    fn bounded_dred_summary_names_progress_and_page_fault_allocations() {
        let queue = CString::new("REmu guest queue").expect("literal has no null");
        let list = CString::new("submission 42").expect("literal has no null");
        let history = [
            D3D12_AUTO_BREADCRUMB_OP_COPYBUFFERREGION,
            D3D12_AUTO_BREADCRUMB_OP_DRAWINDEXEDINSTANCED,
        ];
        let completed = 1_u32;
        let mut node = D3D12_AUTO_BREADCRUMB_NODE1::default();
        node.pCommandQueueDebugNameA = queue.as_ptr().cast();
        node.pCommandListDebugNameA = list.as_ptr().cast();
        node.BreadcrumbCount = history.len() as u32;
        node.pLastBreadcrumbValue = &raw const completed;
        node.pCommandHistory = history.as_ptr();
        let breadcrumbs = summarize_breadcrumbs(&D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT1 {
            pHeadAutoBreadcrumbNode: &raw const node,
        });
        assert!(breadcrumbs.contains("REmu guest queue"), "{breadcrumbs}");
        assert!(breadcrumbs.contains("submission 42"), "{breadcrumbs}");
        assert!(
            breadcrumbs.contains("last=CopyBufferRegion"),
            "{breadcrumbs}"
        );
        assert!(
            breadcrumbs.contains("next=DrawIndexedInstanced"),
            "{breadcrumbs}"
        );

        let resource = CString::new("target 0x1234").expect("literal has no null");
        let mut allocation = D3D12_DRED_ALLOCATION_NODE1::default();
        allocation.ObjectNameA = resource.as_ptr().cast();
        allocation.AllocationType = D3D12_DRED_ALLOCATION_TYPE_RESOURCE;
        let page_fault = summarize_page_fault1(&D3D12_DRED_PAGE_FAULT_OUTPUT1 {
            PageFaultVA: 0x1234,
            pHeadExistingAllocationNode: &raw const allocation,
            pHeadRecentFreedAllocationNode: std::ptr::null(),
        });
        assert!(page_fault.contains("va=0x1234"));
        assert!(page_fault.contains("target 0x1234"));
    }
}
