// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! The audio-job-manager (AJM) engine.
//!
//! On hardware the Audio Coprocessor (ACP) runs codec jobs a client batches through
//! `libSceAjm`: an instance is a voice, a batch is a list of jobs against instances, and the
//! batch executes asynchronously until completed or killed (ajm.h:16-42). REmu has no
//! coprocessor, so this module is that engine: a host worker pool parses the batch the
//! runtime's own job builders wrote into the caller's arena, runs every job through a codec
//! registered by `SceAjmCodecType`, writes each job's sideband result (sideband.h:410-423),
//! and retires the batch. `sceAjmBatchStart` returns a live identifier immediately
//! (ajm.h:1228-1273); `sceAjmBatchWait` blocks until the workers retire it, honouring
//! `SCE_AJM_WAIT_INFINITE` and finite millisecond timeouts (ajm.h:1275-1317, ajm.h:152).
//!
//! A batch that references a codec this host does not implement fails whole and loudly at
//! start: `SCE_AJM_ERROR_MALFORMED_BATCH` with `SCE_AJM_ERROR_CODEC_NOT_SUPPORTED`
//! (error.h:79) in the caller's batch-error report — never a fabricated success. An audio
//! engine that verifies its own pipeline (Sony's SCREAM does) must see the truth; a stall it
//! can assert on is acceptable, a lie is not. Per-codec-type first-hit telemetry ranks which
//! decoder to implement next.
//!
//! [`dispatch`] is consulted by the runtime's pre-lock import dispatcher, so a blocking wait
//! never parks while holding the process-wide dispatch lock. Every `sceAjm*` NID is served
//! here and nowhere else.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use crate::{
    assert_wait_is_off_the_dispatch_lock, guest_memset, guest_output_pointer, low_i32, low_u32,
    process_guest_range_has_read_access, sce_kernel_error, write_guest_bytes, write_guest_u32,
};

// Error numbers, ajm/error.h:37-229.
pub(crate) const SCE_AJM_ERROR_INVALID_CONTEXT: u32 = 0x8093_0002;
pub(crate) const SCE_AJM_ERROR_INVALID_INSTANCE: u32 = 0x8093_0003;
pub(crate) const SCE_AJM_ERROR_INVALID_BATCH: u32 = 0x8093_0004;
pub(crate) const SCE_AJM_ERROR_INVALID_PARAMETER: u32 = 0x8093_0005;
pub(crate) const SCE_AJM_ERROR_OUT_OF_RESOURCES: u32 = 0x8093_0007;
pub(crate) const SCE_AJM_ERROR_CODEC_NOT_SUPPORTED: u32 = 0x8093_0008;
pub(crate) const SCE_AJM_ERROR_CODEC_ALREADY_REGISTERED: u32 = 0x8093_0009;
pub(crate) const SCE_AJM_ERROR_CODEC_NOT_REGISTERED: u32 = 0x8093_000a;
pub(crate) const SCE_AJM_ERROR_BUSY: u32 = 0x8093_000d;
pub(crate) const SCE_AJM_ERROR_BAD_PRIORITY: u32 = 0x8093_000e;
pub(crate) const SCE_AJM_ERROR_IN_PROGRESS: u32 = 0x8093_000f;
pub(crate) const SCE_AJM_ERROR_MALFORMED_BATCH: u32 = 0x8093_0011;
pub(crate) const SCE_AJM_ERROR_JOB_CREATION: u32 = 0x8093_0012;
pub(crate) const SCE_AJM_ERROR_INVALID_OPCODE: u32 = 0x8093_0013;
pub(crate) const SCE_AJM_ERROR_INVALID_ADDRESS: u32 = 0x8093_0016;
pub(crate) const SCE_AJM_ERROR_CANCELLED: u32 = 0x8093_0017;

// Codec result flags returned inside `SceAjmSidebandResult.iResult`, ajm/error.h:243-303.
const SCE_AJM_RESULT_NOT_INITIALIZED: i32 = 0x0000_0001;
const SCE_AJM_RESULT_INVALID_PARAMETER: i32 = 0x0000_0004;
const SCE_AJM_RESULT_SIDEBAND_TRUNCATED: i32 = 0x0000_0100;
const SCE_AJM_RESULT_FATAL: i32 = 0x8000_0000_u32 as i32;

/// Waits forever (ajm.h:152).
const SCE_AJM_WAIT_INFINITE: u32 = u32::MAX;
/// Batch priority bounds (ajm.h:160-176).
const SCE_AJM_PRIORITY_GAME_LOWEST: i32 = 10;
const SCE_AJM_PRIORITY_GAME_HIGHEST: i32 = 90;
/// Instance identifier statistics jobs run against (ajm.h:1368).
const SCE_AJM_INSTANCE_STATISTICS: u32 = 0x8_0000;

// Job flag layout for CONTROL/RUN, sideband.h:917-939: reserved0 bits 0-2, codec flags bits
// 3-10, command flags bits 11-15, sideband flags bits 44-47.
const AJM_FLAG_RUN_GET_CODEC_INFO: u64 = 1 << 11; // sideband.h:120
const AJM_FLAG_RUN_MULTIPLE_FRAMES: u64 = 1 << 12; // sideband.h:129
const AJM_FLAG_CONTROL_RESET: u64 = 1 << 13; // sideband.h:138
const AJM_FLAG_CONTROL_INITIALIZE: u64 = 1 << 14; // sideband.h:146
const AJM_FLAG_SIDEBAND_GAPLESS_DECODE: u64 = 1 << 45; // sideband.h:184
const AJM_FLAG_SIDEBAND_FORMAT: u64 = 1 << 46; // sideband.h:193
const AJM_FLAG_SIDEBAND_STREAM: u64 = 1 << 47; // sideband.h:202

// Codec types, defined per codec header (ajm.h:130-135).
const AJM_CODEC_DEC_MP3: u32 = 0; // mp3_decoder.h:30
const AJM_CODEC_DEC_AT9: u32 = 1; // at9_decoder.h:31
const AJM_CODEC_DEC_M4AAC: u32 = 2; // m4aac_decoder.h:25
const AJM_CODEC_DEC_CELP8: u32 = 3; // celp8_decoder.h:23
const AJM_CODEC_ENC_CELP8: u32 = 4; // celp8_encoder.h:23
const AJM_CODEC_DEC_CELP: u32 = 12; // celp_decoder.h:23
const AJM_CODEC_ENC_CELP: u32 = 13; // celp_encoder.h:23
const AJM_CODEC_DEC_HEVAG: u32 = 22; // hevag_decoder.h:21
pub(crate) const AJM_CODEC_DEC_LPCM: u32 = 23; // lpcm_decoder.h:30
const AJM_CODEC_DEC_OPUS: u32 = 24; // opus_decoder.h:20

/// Job sizes the guest sized its arena with (ajm.h:184, ajm.h:192, ajm.h:208, ajm.h:304 with
/// `SceAjmGetStatisticsParameters` = 16 bytes, sideband.h:856-859).
const AJM_JOB_CONTROL_SIZE: u64 = 48;
const AJM_JOB_RUN_SIZE: u64 = 64;
const AJM_JOB_GET_STATISTICS_SIZE: u64 = 16 + 8 + AJM_JOB_RUN_SIZE;
const fn ajm_job_run_split_size(buffers: u64) -> u64 {
    16 * buffers + 32
}

/// Sideband result prefix every job writes (`SceAjmSidebandResult`, sideband.h:419-423).
const AJM_SIDEBAND_RESULT_SIZE: u64 = 8;
/// `SceAjmGetStatisticsResult`: result + engine + memory (sideband.h:870-875).
const AJM_STATISTICS_RESULT_SIZE: u64 = 48;

/// Smallest arena a caller may hand `sceAjmBatchInitialize` — one control job.
const AJM_BATCH_MIN_SIZE: u64 = AJM_JOB_CONTROL_SIZE;
const AJM_BATCH_MAX_SIZE: u64 = 0x10_000;
/// Live batches one host engine will track before reporting exhaustion.
const AJM_BATCH_LIMIT: usize = 1024;

/// Bytes one page of memory published to the coprocessor covers ("Size of the region in 2-MB
/// pages", ajm.h:552, ajm.h:566).
pub(crate) const AJM_MEMORY_PAGE_SIZE: u64 = 2 * 1024 * 1024;
/// Pages one context may publish. No console ceiling is published; the bound keeps a runaway
/// caller from growing the table without limit and reports honest exhaustion.
pub(crate) const AJM_MEMORY_PAGE_LIMIT: u64 = 512;

/// Instance memory the engine models, from the documented maximum expansion
/// (`SCE_AJM_MAX_EXTENDED_INSTANCE_MEMORY`, ajm.h:75).
const AJM_INSTANCE_MEMORY_BUDGET: u32 = 32 * 1024 * 1024;
/// Nominal instance footprint this host engine charges against that budget.
const AJM_INSTANCE_NOMINAL_SIZE: u32 = 0x1_0000;
/// Buffer memory the engine models for in-flight batches.
const AJM_BUFFER_MEMORY_BUDGET: u32 = 16 * 1024 * 1024;

/// Job records this engine's builders write into the guest arena. The arena layout is
/// declared internal by the SDK (ajm.h:344-349), so the encoding is REmu's own; only the
/// record SIZES are ABI, because guests size their arenas with the `SCE_AJM_JOB_*_SIZE`
/// macros at compile time.
const AJM_JOB_MAGIC: u32 = 0x414a_4d00; // "AJM\0" | opcode
const AJM_OPCODE_CONTROL: u32 = 1;
const AJM_OPCODE_RUN: u32 = 2;
const AJM_OPCODE_RUN_SPLIT: u32 = 3;
const AJM_OPCODE_STATISTICS: u32 = 4;

/// Batch buffer descriptor (`SceAjmBatchInfo`, ajm.h:351-358). Five qwords; the arena itself
/// belongs to the caller.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct SceAjmBatchInfo {
    pub(crate) buffer: u64,
    pub(crate) offset: u64,
    pub(crate) size: u64,
    pub(crate) last_good_job: u64,
    pub(crate) last_good_job_return_address: u64,
}

const _: () = {
    assert!(std::mem::size_of::<SceAjmBatchInfo>() == 0x28);
    assert!(std::mem::offset_of!(SceAjmBatchInfo, buffer) == 0x00);
    assert!(std::mem::offset_of!(SceAjmBatchInfo, offset) == 0x08);
    assert!(std::mem::offset_of!(SceAjmBatchInfo, size) == 0x10);
    assert!(std::mem::offset_of!(SceAjmBatchInfo, last_good_job) == 0x18);
    assert!(std::mem::offset_of!(SceAjmBatchInfo, last_good_job_return_address) == 0x20);
};

/// Batch error description (`SceAjmBatchError`, ajm.h:374-380): `int iErrorCode`, `const
/// void* pJobAddress`, `unsigned int uiCommandOffset`, `const void* pJobOriginRa`. LP64
/// padding places the fields at 0/8/16/24 for 32 bytes total.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct SceAjmBatchError {
    error_code: i32,
    _pad0: u32,
    job_address: u64,
    command_offset: u32,
    _pad1: u32,
    job_origin_return_address: u64,
}

const _: () = {
    assert!(std::mem::size_of::<SceAjmBatchError>() == 0x20);
    assert!(std::mem::offset_of!(SceAjmBatchError, error_code) == 0x00);
    assert!(std::mem::offset_of!(SceAjmBatchError, job_address) == 0x08);
    assert!(std::mem::offset_of!(SceAjmBatchError, command_offset) == 0x10);
    assert!(std::mem::offset_of!(SceAjmBatchError, job_origin_return_address) == 0x18);
};

const AJM_BATCH_ERROR_SIZE: u64 = 0x20;

/// One buffer of a split job (`SceAjmBuffer`, ajm.h:334-340).
#[derive(Clone, Copy, Debug)]
struct AjmBuffer {
    address: u64,
    size: u64,
}

/// A typed job parsed from the batch arena.
#[derive(Clone, Debug)]
struct AjmJob {
    /// Guest address of this job's record, for the malformed-batch report.
    address: u64,
    instance: u32,
    kind: AjmJobKind,
}

#[derive(Clone, Debug)]
enum AjmJobKind {
    Control {
        flags: u64,
        sideband_input: AjmBuffer,
        sideband_output: AjmBuffer,
    },
    Run {
        flags: u64,
        inputs: Vec<AjmBuffer>,
        outputs: Vec<AjmBuffer>,
        sideband: AjmBuffer,
    },
    Statistics {
        interval: f32,
        result: u64,
    },
}

/// The failure `sceAjmBatchStart` reports through `pBatchError` (ajm.h:1247-1249).
struct BatchFailure {
    code: u32,
    job_address: u64,
    command_offset: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BatchStatus {
    Pending,
    Running,
    Retired,
    Failed,
    Cancelled,
}

struct Batch {
    context: u32,
    jobs: Vec<AjmJob>,
    status: BatchStatus,
    /// Filled when `status == Failed`; reported by `sceAjmBatchWait`.
    error: Option<SceAjmBatchError>,
    /// One thread may wait on a batch at a time (ajm.h:1297-1298).
    waiter_active: bool,
    /// Arena bytes the guest built this batch in, for the memory statistics report.
    arena_bytes: u32,
    input_bytes: u32,
    output_bytes: u32,
    /// Set when a worker begins executing, for the real elapsed-time statistics report.
    execution_started: Option<Instant>,
}

/// Per-instance decoder state.
struct InstanceState {
    codec: u32,
    /// Creation flags: maximum channels in bits 0-6, output format in bits 7-9, standard
    /// flags above (sideband.h:896-914).
    flags: u64,
    /// LPCM initialize parameters once a `sceAjmBatchJobInitialize` job ran.
    lpcm: Option<LpcmParameters>,
    /// Sample frames decoded since the last reset (`uiTotalDecodedSamples`, sideband.h:436).
    total_decoded_samples: u64,
}

impl InstanceState {
    /// Output sample encoding requested at creation (`SCE_AJM_INSTANCE_FLAG_FORMAT`,
    /// sideband.h:75-77, offsets sideband.h:907-908).
    fn output_encoding(&self) -> u32 {
        (self.flags >> 7) as u32 & 0x7
    }
}

/// `SceAjmDecLpcmInitializeParameters` (lpcm_decoder.h:66-71).
#[derive(Clone, Copy, Debug)]
struct LpcmParameters {
    encoding: u32,
    channels: u32,
    sample_rate: u32,
}

/// Bytes one sample of an encoding occupies (`SceAjmFormatEncoding`, sideband.h:518-523).
fn encoding_sample_bytes(encoding: u32) -> Option<u64> {
    match encoding {
        0 => Some(2),        // SCE_AJM_FORMAT_ENCODING_S16
        1 | 2 => Some(4),    // S32 / FLOAT
        _ => None,
    }
}

struct ContextState {
    registered_codecs: BTreeSet<u32>,
    next_instance: u32,
    instances: BTreeMap<u32, InstanceState>,
    /// Memory regions published to the coprocessor, start address -> pages.
    regions: BTreeMap<u64, u64>,
}

impl ContextState {
    fn new() -> Self {
        Self {
            registered_codecs: BTreeSet::new(),
            next_instance: 1,
            instances: BTreeMap::new(),
            regions: BTreeMap::new(),
        }
    }
}

#[derive(Default)]
struct EngineState {
    next_context: u32,
    contexts: BTreeMap<u32, ContextState>,
    next_batch: u32,
    batches: BTreeMap<u32, Batch>,
    queue: VecDeque<u32>,
    /// Recent batch execution spans, newest last, for the engine-usage statistics report.
    history: VecDeque<(Instant, Duration)>,
    /// Jobs seen per codec type; the first hit per type logs loudly so real titles rank
    /// which decoders to implement next.
    codec_job_counts: BTreeMap<u32, u64>,
}

/// The host audio-job engine. One lives for the process; tests construct their own.
pub(crate) struct AjmEngine {
    state: Mutex<EngineState>,
    /// Signalled when a batch is queued for the workers.
    work_queued: Condvar,
    /// Signalled when a batch reaches a terminal status.
    batch_retired: Condvar,
}

/// Worker threads the process engine runs. Batches from distinct contexts are independent;
/// two workers keep one slow batch from starving another context's audio thread.
const AJM_WORKER_COUNT: usize = 2;

static ENGINE: OnceLock<&'static AjmEngine> = OnceLock::new();

impl AjmEngine {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(EngineState::default()),
            work_queued: Condvar::new(),
            batch_retired: Condvar::new(),
        }
    }

    /// The process-wide engine, with its worker pool started.
    pub(crate) fn global() -> &'static Self {
        ENGINE.get_or_init(|| {
            let engine: &'static Self = Box::leak(Box::new(Self::new()));
            for index in 0..AJM_WORKER_COUNT {
                std::thread::Builder::new()
                    .name(format!("ajm-worker-{index}"))
                    .spawn(move || engine.worker_loop())
                    .expect("the AJM worker pool must start");
            }
            engine
        })
    }

    fn lock(&self) -> MutexGuard<'_, EngineState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Discards every context, instance, and batch for a fresh process, waking any waiter so
    /// it reports the batch gone rather than parking across the reset.
    #[cfg(any(test, target_arch = "x86_64"))]
    pub(crate) fn reset(&self) {
        let mut state = self.lock();
        *state = EngineState::default();
        drop(state);
        self.batch_retired.notify_all();
    }

    fn worker_loop(&self) -> ! {
        loop {
            let mut state = self.lock();
            let batch_id = loop {
                if let Some(id) = state.queue.pop_front() {
                    break id;
                }
                state = self
                    .work_queued
                    .wait(state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            };
            execute_batch(&mut state, batch_id);
            drop(state);
            self.batch_retired.notify_all();
        }
    }

    /// Executes the oldest queued batch on the calling thread, so tests without a worker
    /// pool drive retirement deterministically.
    #[cfg(test)]
    pub(crate) fn execute_next_queued_batch(&self) -> bool {
        let mut state = self.lock();
        let Some(batch_id) = state.queue.pop_front() else {
            return false;
        };
        execute_batch(&mut state, batch_id);
        drop(state);
        self.batch_retired.notify_all();
        true
    }

    /// `sceAjmInitialize(initializeFlag, pContext)` (ajm.h:497-523).
    ///
    /// The flag carries the instance-memory expansion in bits 0-6 ("multiple of 2 from 2 to
    /// 32, or 0", ajm.h:81, layout sideband.h:879-894) and the zero-copy modes in bits 32-33.
    /// Gen5 titles additionally tag the high dword with the library revision, which is
    /// tolerated; undefined low-dword bits are refused.
    fn initialize(&self, arguments: [u64; 6]) -> u64 {
        let flag = arguments[0];
        let output = arguments[1];
        let expansion = low_u32(flag) & 0x7f;
        let undefined_low_bits = low_u32(flag) & !0x7f;
        if output == 0
            || undefined_low_bits != 0
            || expansion > 32
            || !expansion.is_multiple_of(2)
        {
            return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
        }
        let mut state = self.lock();
        let context = state.next_context.max(1);
        state.next_context = context.saturating_add(1);
        state.contexts.insert(context, ContextState::new());
        drop(state);
        write_guest_u32(output, context);
        eprintln!("[REMu][AJM] initialized context={context} flag={flag:#x}");
        0
    }

    /// `sceAjmFinalize(uiContext)` (ajm.h:525-542). A zero context keeps its historical
    /// clear-everything meaning here: no live id is ever zero, and a title tearing down its
    /// audio stack wholesale is better served than refused.
    fn finalize(&self, arguments: [u64; 6]) -> u64 {
        let context = low_u32(arguments[0]);
        let mut state = self.lock();
        if context == 0 {
            state.contexts.clear();
        } else if state.contexts.remove(&context).is_none() {
            return sce_kernel_error(SCE_AJM_ERROR_INVALID_CONTEXT);
        }
        drop(state);
        eprintln!("[REMu][AJM] finalized context={context}");
        0
    }

    /// `sceAjmModuleRegister(uiContext, uiCodec, iReserved)` (ajm.h:601-633).
    ///
    /// Registration is the guest declaring intent; whether this HOST implements the codec is
    /// reported when a batch actually runs a job through it, mirroring hardware where every
    /// shipped codec registers successfully.
    fn module_register(&self, arguments: [u64; 6]) -> u64 {
        let context = low_u32(arguments[0]);
        let codec = low_u32(arguments[1]);
        if arguments[2] != 0 || !codec_type_is_defined(codec) {
            return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
        }
        let mut state = self.lock();
        let Some(context_state) = state.contexts.get_mut(&context) else {
            return sce_kernel_error(SCE_AJM_ERROR_INVALID_CONTEXT);
        };
        if !context_state.registered_codecs.insert(codec) {
            return sce_kernel_error(SCE_AJM_ERROR_CODEC_ALREADY_REGISTERED);
        }
        drop(state);
        eprintln!(
            "[REMu][AJM] registered codec={codec} ({}) context={context} host_implementation={}",
            codec_display_name(codec),
            if host_codec(codec).is_some() {
                "yes"
            } else {
                "NO - decode jobs for this codec will fail loudly"
            }
        );
        0
    }

    /// `sceAjmModuleUnregister(uiContext, uiCodec)` (ajm.h:635-661).
    fn module_unregister(&self, arguments: [u64; 6]) -> u64 {
        let context = low_u32(arguments[0]);
        let codec = low_u32(arguments[1]);
        let mut state = self.lock();
        let Some(context_state) = state.contexts.get_mut(&context) else {
            return sce_kernel_error(SCE_AJM_ERROR_INVALID_CONTEXT);
        };
        if !context_state.registered_codecs.remove(&codec) {
            return sce_kernel_error(SCE_AJM_ERROR_CODEC_NOT_REGISTERED);
        }
        drop(state);
        eprintln!("[REMu][AJM] unregistered codec={codec} context={context}");
        0
    }

    /// `sceAjmInstanceCreate(uiContext, uiCodec, uiFlags, pInstance)` (ajm.h:663-710).
    ///
    /// The codec type must be one the SDK defines and be registered in the context; the
    /// creation flags (maximum channels, output format, standard flags — sideband.h:896-914)
    /// are recorded on the instance for the decode jobs to honour.
    fn instance_create(&self, arguments: [u64; 6]) -> u64 {
        let context = low_u32(arguments[0]);
        let codec = low_u32(arguments[1]);
        let flags = arguments[2];
        let output = arguments[3];
        if output == 0 || !codec_type_is_defined(codec) {
            return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
        }
        let mut state = self.lock();
        let Some(context_state) = state.contexts.get_mut(&context) else {
            return sce_kernel_error(SCE_AJM_ERROR_INVALID_CONTEXT);
        };
        if !context_state.registered_codecs.contains(&codec) {
            return sce_kernel_error(SCE_AJM_ERROR_CODEC_NOT_REGISTERED);
        }
        let index = context_state.next_instance.max(1) & 0x3fff;
        context_state.next_instance = index.saturating_add(1);
        let instance = index | (codec << 14);
        context_state.instances.insert(
            instance,
            InstanceState {
                codec,
                flags,
                lpcm: None,
                total_decoded_samples: 0,
            },
        );
        drop(state);
        write_guest_u32(output, instance);
        eprintln!(
            "[REMu][AJM] created instance={instance:#x} codec={codec} context={context} flags={flags:#x}"
        );
        0
    }

    /// `sceAjmInstanceDestroy(uiContext, uiInstance)` (ajm.h:799-822).
    fn instance_destroy(&self, arguments: [u64; 6]) -> u64 {
        let context = low_u32(arguments[0]);
        let instance = low_u32(arguments[1]);
        let mut state = self.lock();
        let Some(context_state) = state.contexts.get_mut(&context) else {
            return sce_kernel_error(SCE_AJM_ERROR_INVALID_CONTEXT);
        };
        if context_state.instances.remove(&instance).is_none() {
            return sce_kernel_error(SCE_AJM_ERROR_INVALID_INSTANCE);
        }
        drop(state);
        eprintln!("[REMu][AJM] destroyed instance={instance:#x} context={context}");
        0
    }

    /// `sceAjmMemoryRegister(uiContext, pRegion, szNumPages)` (ajm.h:544-572). The region
    /// must be 2-MB aligned and sized in 2-MB pages (ajm.h:552, ajm.h:566); overlapping or
    /// unbounded publications are refused.
    fn memory_register(&self, arguments: [u64; 6]) -> u64 {
        let (context, region, pages) = (low_u32(arguments[0]), arguments[1], arguments[2]);
        let Some(length) = pages.checked_mul(AJM_MEMORY_PAGE_SIZE) else {
            return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
        };
        let mut state = self.lock();
        let Some(context_state) = state.contexts.get_mut(&context) else {
            return sce_kernel_error(SCE_AJM_ERROR_INVALID_CONTEXT);
        };
        if region == 0
            || pages == 0
            || !region.is_multiple_of(AJM_MEMORY_PAGE_SIZE)
            || region.checked_add(length).is_none()
        {
            return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
        }
        let mut published = 0_u64;
        for (&start, &count) in &context_state.regions {
            published = published.saturating_add(count);
            let end = start.saturating_add(count.saturating_mul(AJM_MEMORY_PAGE_SIZE));
            if region < end && start < region.saturating_add(length) {
                return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
            }
        }
        if published.saturating_add(pages) > AJM_MEMORY_PAGE_LIMIT {
            return sce_kernel_error(SCE_AJM_ERROR_OUT_OF_RESOURCES);
        }
        context_state.regions.insert(region, pages);
        0
    }

    /// `sceAjmMemoryUnregister(uiContext, pRegion)` (ajm.h:574-599).
    fn memory_unregister(&self, arguments: [u64; 6]) -> u64 {
        let (context, region) = (low_u32(arguments[0]), arguments[1]);
        let mut state = self.lock();
        let Some(context_state) = state.contexts.get_mut(&context) else {
            return sce_kernel_error(SCE_AJM_ERROR_INVALID_CONTEXT);
        };
        if region == 0 || context_state.regions.remove(&region).is_none() {
            return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
        }
        0
    }

    /// `sceAjmBatchStart(uiContext, pInfo, iPriority, pBatchError, pBatch)`
    /// (ajm.h:1228-1273). Parses the arena into typed jobs, validates every referenced
    /// instance and buffer, queues the batch for the workers, and returns its live
    /// identifier immediately. A malformed batch — including one that references a codec
    /// this host does not implement — fails whole, with the detailed code in the caller's
    /// batch-error object.
    fn batch_start(&self, arguments: [u64; 6]) -> u64 {
        let context = low_u32(arguments[0]);
        let info = arguments[1];
        let priority = low_i32(arguments[2]);
        let error = arguments[3];
        let output = arguments[4];
        if info == 0
            || output == 0
            || guest_output_pointer(output, std::mem::size_of::<u32>()).is_none()
            || guest_output_pointer(info, std::mem::size_of::<SceAjmBatchInfo>()).is_none()
        {
            return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
        }
        if error != 0 && guest_output_pointer(error, AJM_BATCH_ERROR_SIZE as usize).is_none() {
            return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
        }
        if !(SCE_AJM_PRIORITY_GAME_LOWEST..=SCE_AJM_PRIORITY_GAME_HIGHEST).contains(&priority) {
            return sce_kernel_error(SCE_AJM_ERROR_BAD_PRIORITY);
        }
        // SAFETY: the complete cursor object was validated immediately above.
        let batch_info = unsafe { std::ptr::read_unaligned(info as *const SceAjmBatchInfo) };

        let mut state = self.lock();
        if !state.contexts.contains_key(&context) {
            return sce_kernel_error(SCE_AJM_ERROR_INVALID_CONTEXT);
        }
        if state.batches.len() >= AJM_BATCH_LIMIT {
            return sce_kernel_error(SCE_AJM_ERROR_OUT_OF_RESOURCES);
        }
        let jobs = match parse_batch(&mut state, context, &batch_info) {
            Ok(jobs) => jobs,
            Err(failure) => {
                drop(state);
                eprintln!(
                    "[REMu][AJM] batch refused context={context} error={:#x} job={:#x} offset={:#x}",
                    failure.code, failure.job_address, failure.command_offset
                );
                if error != 0 {
                    let report = SceAjmBatchError {
                        error_code: failure.code.cast_signed(),
                        job_address: failure.job_address,
                        command_offset: failure.command_offset,
                        job_origin_return_address: 0,
                        ..SceAjmBatchError::default()
                    };
                    write_batch_error(error, &report);
                }
                return sce_kernel_error(SCE_AJM_ERROR_MALFORMED_BATCH);
            }
        };
        if error != 0 {
            write_batch_error(error, &SceAjmBatchError::default());
        }
        let input_bytes = jobs
            .iter()
            .map(|job| match &job.kind {
                AjmJobKind::Run { inputs, .. } => {
                    inputs.iter().map(|buffer| buffer.size).sum::<u64>()
                }
                _ => 0,
            })
            .sum::<u64>();
        let output_bytes = jobs
            .iter()
            .map(|job| match &job.kind {
                AjmJobKind::Run { outputs, .. } => {
                    outputs.iter().map(|buffer| buffer.size).sum::<u64>()
                }
                _ => 0,
            })
            .sum::<u64>();
        let batch_id = state.next_batch.wrapping_add(1).max(1);
        state.next_batch = batch_id;
        state.batches.insert(
            batch_id,
            Batch {
                context,
                jobs,
                status: BatchStatus::Pending,
                error: None,
                waiter_active: false,
                arena_bytes: u32::try_from(batch_info.offset).unwrap_or(u32::MAX),
                input_bytes: u32::try_from(input_bytes).unwrap_or(u32::MAX),
                output_bytes: u32::try_from(output_bytes).unwrap_or(u32::MAX),
                execution_started: None,
            },
        );
        state.queue.push_back(batch_id);
        drop(state);
        self.work_queued.notify_one();
        write_guest_u32(output, batch_id);
        0
    }

    /// `sceAjmBatchWait(uiContext, uiBatch, uiTimeout, pBatchError)` (ajm.h:1275-1317).
    ///
    /// A zero timeout polls; `SCE_AJM_WAIT_INFINITE` blocks until the workers retire the
    /// batch; anything else is a bound in milliseconds that expires with
    /// `SCE_AJM_ERROR_IN_PROGRESS`. One thread may wait per batch (ajm.h:1297-1298). The
    /// wait parks on the engine's own condition variable — never the process dispatch lock.
    fn batch_wait(&self, arguments: [u64; 6]) -> u64 {
        let context = low_u32(arguments[0]);
        let batch_id = low_u32(arguments[1]);
        let timeout = low_u32(arguments[2]);
        let error = arguments[3];
        if error != 0 && guest_output_pointer(error, AJM_BATCH_ERROR_SIZE as usize).is_none() {
            return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
        }
        if timeout != 0 {
            assert_wait_is_off_the_dispatch_lock("sceAjmBatchWait");
        }
        let deadline = (timeout != 0 && timeout != SCE_AJM_WAIT_INFINITE)
            .then(|| Instant::now() + Duration::from_millis(u64::from(timeout)));
        let mut state = self.lock();
        if !state.contexts.contains_key(&context) {
            return sce_kernel_error(SCE_AJM_ERROR_INVALID_CONTEXT);
        }
        match state.batches.get_mut(&batch_id) {
            None => return sce_kernel_error(SCE_AJM_ERROR_INVALID_BATCH),
            Some(batch) => {
                if batch.waiter_active {
                    return sce_kernel_error(SCE_AJM_ERROR_BUSY);
                }
                batch.waiter_active = true;
            }
        }
        loop {
            let Some(batch) = state.batches.get(&batch_id) else {
                // The engine was reset while waiting; the batch is gone.
                return sce_kernel_error(SCE_AJM_ERROR_INVALID_BATCH);
            };
            match batch.status {
                BatchStatus::Retired => {
                    if error != 0 {
                        write_batch_error(error, &SceAjmBatchError::default());
                    }
                    state.batches.remove(&batch_id);
                    return 0;
                }
                BatchStatus::Cancelled => {
                    if error != 0 {
                        write_batch_error(error, &SceAjmBatchError::default());
                    }
                    state.batches.remove(&batch_id);
                    return sce_kernel_error(SCE_AJM_ERROR_CANCELLED);
                }
                BatchStatus::Failed => {
                    if error != 0 {
                        let report = batch.error.unwrap_or_default();
                        write_batch_error(error, &report);
                    }
                    state.batches.remove(&batch_id);
                    return sce_kernel_error(SCE_AJM_ERROR_MALFORMED_BATCH);
                }
                BatchStatus::Pending | BatchStatus::Running => {}
            }
            if timeout == 0 {
                release_waiter(&mut state, batch_id);
                return sce_kernel_error(SCE_AJM_ERROR_IN_PROGRESS);
            }
            state = if let Some(deadline) = deadline {
                let now = Instant::now();
                if now >= deadline {
                    release_waiter(&mut state, batch_id);
                    return sce_kernel_error(SCE_AJM_ERROR_IN_PROGRESS);
                }
                self.batch_retired
                    .wait_timeout(state, deadline - now)
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .0
            } else {
                self.batch_retired
                    .wait(state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
            };
        }
    }

    /// `sceAjmBatchCancel(uiContext, uiBatch)` (ajm.h:1319-1344). A pending batch is marked
    /// cancelled and never executes; a batch the workers already picked up retires normally,
    /// which the contract allows — cancellation is a request, and the caller must still
    /// wait (ajm.h:1337-1338).
    fn batch_cancel(&self, arguments: [u64; 6]) -> u64 {
        let context = low_u32(arguments[0]);
        let batch_id = low_u32(arguments[1]);
        let mut state = self.lock();
        if !state.contexts.contains_key(&context) {
            return sce_kernel_error(SCE_AJM_ERROR_INVALID_CONTEXT);
        }
        let Some(batch) = state.batches.get_mut(&batch_id) else {
            return sce_kernel_error(SCE_AJM_ERROR_INVALID_BATCH);
        };
        if batch.status == BatchStatus::Pending {
            batch.status = BatchStatus::Cancelled;
            state.queue.retain(|&queued| queued != batch_id);
            drop(state);
            self.batch_retired.notify_all();
        }
        0
    }

    /// Jobs seen per codec type, for tests and diagnostics.
    #[cfg(test)]
    fn codec_job_count(&self, codec: u32) -> u64 {
        self.lock().codec_job_counts.get(&codec).copied().unwrap_or(0)
    }
}

/// Clears a waiter's exclusive claim after a poll or timeout leaves the batch live.
fn release_waiter(state: &mut EngineState, batch_id: u32) {
    if let Some(batch) = state.batches.get_mut(&batch_id) {
        batch.waiter_active = false;
    }
}

/// Writes a batch-error report through a pointer `batch_start`/`batch_wait` validated.
fn write_batch_error(address: u64, report: &SceAjmBatchError) {
    let mut bytes = [0_u8; AJM_BATCH_ERROR_SIZE as usize];
    bytes[0..4].copy_from_slice(&report.error_code.to_le_bytes());
    bytes[8..16].copy_from_slice(&report.job_address.to_le_bytes());
    bytes[16..20].copy_from_slice(&report.command_offset.to_le_bytes());
    bytes[24..32].copy_from_slice(&report.job_origin_return_address.to_le_bytes());
    write_guest_bytes(address, &bytes);
}

/// Reports whether a codec type is one the SDK defines (ajm.h:422-465 and the codec
/// headers). Anything else is an invalid parameter, not a missing implementation.
fn codec_type_is_defined(codec: u32) -> bool {
    matches!(
        codec,
        AJM_CODEC_DEC_MP3
            | AJM_CODEC_DEC_AT9
            | AJM_CODEC_DEC_M4AAC
            | AJM_CODEC_DEC_CELP8
            | AJM_CODEC_ENC_CELP8
            | AJM_CODEC_DEC_CELP
            | AJM_CODEC_ENC_CELP
            | AJM_CODEC_DEC_HEVAG
            | AJM_CODEC_DEC_LPCM
            | AJM_CODEC_DEC_OPUS
    )
}

fn codec_display_name(codec: u32) -> &'static str {
    match codec {
        AJM_CODEC_DEC_MP3 => "MP3 decoder",
        AJM_CODEC_DEC_AT9 => "ATRAC9 decoder",
        AJM_CODEC_DEC_M4AAC => "MPEG4-AAC decoder",
        AJM_CODEC_DEC_CELP8 => "CELP8 decoder",
        AJM_CODEC_ENC_CELP8 => "CELP8 encoder",
        AJM_CODEC_DEC_CELP => "CELP decoder",
        AJM_CODEC_ENC_CELP => "CELP encoder",
        AJM_CODEC_DEC_HEVAG => "HE-VAG decoder",
        AJM_CODEC_DEC_LPCM => "LPCM decoder",
        AJM_CODEC_DEC_OPUS => "Opus decoder",
        _ => "undefined codec",
    }
}

//----------------------------------------------------------------------------
// Codec registry
//----------------------------------------------------------------------------

/// One host codec implementation, keyed by `SceAjmCodecType`.
trait AjmCodec: Sync {
    /// Applies `sceAjmBatchJobInitialize` parameters to an instance, returning the codec
    /// result flags for the job's sideband (`SceAjmResult`, error.h:312-326).
    fn initialize(&self, instance: &mut InstanceState, parameters: &[u8]) -> i32;

    /// Runs a decode over `input`, producing at most `output_capacity` bytes.
    fn run(&self, instance: &mut InstanceState, input: &[u8], output_capacity: usize) -> RunOutput;
}

struct RunOutput {
    result: i32,
    produced: Vec<u8>,
    consumed: usize,
    /// Sample frames decoded, for the stream sideband and the multiple-frames count.
    frames: u64,
}

/// The host codec registered for a type, if this engine implements one.
fn host_codec(codec: u32) -> Option<&'static dyn AjmCodec> {
    match codec {
        AJM_CODEC_DEC_LPCM => Some(&LpcmDecoder),
        _ => None,
    }
}

/// Linear-PCM passthrough decoder (`ajm/lpcm_decoder.h`). The input already is PCM in the
/// initialize-declared encoding; a run converts it to the instance's requested output
/// encoding sample by sample and counts real frames. This is the smallest complete decoder,
/// proving the batch -> job -> codec -> sideband shape end to end.
struct LpcmDecoder;

impl AjmCodec for LpcmDecoder {
    fn initialize(&self, instance: &mut InstanceState, parameters: &[u8]) -> i32 {
        // `SceAjmDecLpcmInitializeParameters`: eInputEncoding, uiNumChannels, uiSampleRate
        // (lpcm_decoder.h:66-71).
        if parameters.len() < 12 {
            return SCE_AJM_RESULT_FATAL | SCE_AJM_RESULT_INVALID_PARAMETER;
        }
        let encoding = u32::from_le_bytes(parameters[0..4].try_into().expect("sized above"));
        let channels = u32::from_le_bytes(parameters[4..8].try_into().expect("sized above"));
        let sample_rate = u32::from_le_bytes(parameters[8..12].try_into().expect("sized above"));
        // Up to 64 channels (lpcm_decoder.h:40); encodings from SceAjmFormatEncoding
        // (sideband.h:518-523).
        if encoding_sample_bytes(encoding).is_none()
            || channels == 0
            || channels > 64
            || sample_rate == 0
        {
            return SCE_AJM_RESULT_FATAL | SCE_AJM_RESULT_INVALID_PARAMETER;
        }
        instance.lpcm = Some(LpcmParameters {
            encoding,
            channels,
            sample_rate,
        });
        instance.total_decoded_samples = 0;
        0
    }

    fn run(&self, instance: &mut InstanceState, input: &[u8], output_capacity: usize) -> RunOutput {
        let Some(parameters) = instance.lpcm else {
            // A run before `sceAjmBatchJobInitialize` has no frame geometry to decode with.
            return RunOutput {
                result: SCE_AJM_RESULT_FATAL | SCE_AJM_RESULT_NOT_INITIALIZED,
                produced: Vec::new(),
                consumed: 0,
                frames: 0,
            };
        };
        let input_sample = encoding_sample_bytes(parameters.encoding).expect("validated at init");
        let output_encoding = instance.output_encoding();
        let Some(output_sample) = encoding_sample_bytes(output_encoding) else {
            return RunOutput {
                result: SCE_AJM_RESULT_FATAL | SCE_AJM_RESULT_INVALID_PARAMETER,
                produced: Vec::new(),
                consumed: 0,
                frames: 0,
            };
        };
        let channels = u64::from(parameters.channels);
        let input_frame = input_sample * channels;
        let output_frame = output_sample * channels;
        let frames = (input.len() as u64 / input_frame).min(output_capacity as u64 / output_frame);
        let consumed = usize::try_from(frames * input_frame).expect("bounded by input length");
        let samples = usize::try_from(frames * channels).expect("bounded by input length");
        let mut produced = Vec::with_capacity(
            usize::try_from(frames * output_frame).expect("bounded by output capacity"),
        );
        for sample in 0..samples {
            let offset = sample * usize::try_from(input_sample).expect("sample sizes are small");
            // Normalize through f64 so every encoding pair converts without a special case.
            let value = match parameters.encoding {
                0 => f64::from(i16::from_le_bytes(
                    input[offset..offset + 2].try_into().expect("sized above"),
                )) / f64::from(i16::MAX),
                1 => f64::from(i32::from_le_bytes(
                    input[offset..offset + 4].try_into().expect("sized above"),
                )) / f64::from(i32::MAX),
                _ => f64::from(f32::from_le_bytes(
                    input[offset..offset + 4].try_into().expect("sized above"),
                )),
            };
            match output_encoding {
                0 => {
                    let scaled = (value * f64::from(i16::MAX)).round();
                    produced.extend_from_slice(
                        &(scaled.clamp(f64::from(i16::MIN), f64::from(i16::MAX)) as i16)
                            .to_le_bytes(),
                    );
                }
                1 => {
                    let scaled = (value * f64::from(i32::MAX)).round();
                    produced.extend_from_slice(
                        &(scaled.clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32)
                            .to_le_bytes(),
                    );
                }
                _ => produced.extend_from_slice(&(value as f32).to_le_bytes()),
            }
        }
        instance.total_decoded_samples += frames;
        RunOutput {
            result: 0,
            produced,
            consumed,
            frames,
        }
    }
}

//----------------------------------------------------------------------------
// Batch parsing
//----------------------------------------------------------------------------

/// Parses every job record the builders wrote into a batch arena, validating instances,
/// codecs, and buffer reachability, and recording per-codec demand telemetry.
fn parse_batch(
    state: &mut EngineState,
    context: u32,
    info: &SceAjmBatchInfo,
) -> Result<Vec<AjmJob>, BatchFailure> {
    let mut jobs = Vec::new();
    let mut cursor = 0_u64;
    while cursor < info.offset {
        let job_address = info.buffer.wrapping_add(cursor);
        let failure = |code: u32| BatchFailure {
            code,
            job_address,
            command_offset: u32::try_from(cursor).unwrap_or(u32::MAX),
        };
        if info.offset - cursor < AJM_JOB_CONTROL_SIZE
            || !process_guest_range_has_read_access(job_address, AJM_JOB_CONTROL_SIZE as usize)
        {
            return Err(failure(SCE_AJM_ERROR_INVALID_OPCODE));
        }
        let opcode = read_arena_u32(job_address);
        if opcode & 0xffff_ff00 != AJM_JOB_MAGIC {
            return Err(failure(SCE_AJM_ERROR_INVALID_OPCODE));
        }
        let instance = read_arena_u32(job_address + 4);
        let (job, record_size) = match opcode & 0xff {
            AJM_OPCODE_CONTROL => (
                AjmJob {
                    address: job_address,
                    instance,
                    kind: AjmJobKind::Control {
                        flags: read_arena_u64(job_address + 8),
                        sideband_input: AjmBuffer {
                            address: read_arena_u64(job_address + 16),
                            size: read_arena_u64(job_address + 24),
                        },
                        sideband_output: AjmBuffer {
                            address: read_arena_u64(job_address + 32),
                            size: read_arena_u64(job_address + 40),
                        },
                    },
                },
                AJM_JOB_CONTROL_SIZE,
            ),
            AJM_OPCODE_RUN => {
                if info.offset - cursor < AJM_JOB_RUN_SIZE
                    || !process_guest_range_has_read_access(job_address, AJM_JOB_RUN_SIZE as usize)
                {
                    return Err(failure(SCE_AJM_ERROR_INVALID_OPCODE));
                }
                (
                    AjmJob {
                        address: job_address,
                        instance,
                        kind: AjmJobKind::Run {
                            flags: read_arena_u64(job_address + 8),
                            inputs: vec![AjmBuffer {
                                address: read_arena_u64(job_address + 16),
                                size: read_arena_u64(job_address + 24),
                            }],
                            outputs: vec![AjmBuffer {
                                address: read_arena_u64(job_address + 32),
                                size: read_arena_u64(job_address + 40),
                            }],
                            sideband: AjmBuffer {
                                address: read_arena_u64(job_address + 48),
                                size: read_arena_u64(job_address + 56),
                            },
                        },
                    },
                    AJM_JOB_RUN_SIZE,
                )
            }
            AJM_OPCODE_RUN_SPLIT => {
                let input_count = u64::from(read_arena_u16(job_address + 28));
                let output_count = u64::from(read_arena_u16(job_address + 30));
                let record_size = ajm_job_run_split_size(input_count + output_count);
                if info.offset - cursor < record_size
                    || !process_guest_range_has_read_access(job_address, record_size as usize)
                {
                    return Err(failure(SCE_AJM_ERROR_INVALID_OPCODE));
                }
                let buffer_at = |index: u64| AjmBuffer {
                    address: read_arena_u64(job_address + 32 + index * 16),
                    size: read_arena_u64(job_address + 40 + index * 16),
                };
                (
                    AjmJob {
                        address: job_address,
                        instance,
                        kind: AjmJobKind::Run {
                            flags: read_arena_u64(job_address + 8),
                            inputs: (0..input_count).map(buffer_at).collect(),
                            outputs: (input_count..input_count + output_count)
                                .map(buffer_at)
                                .collect(),
                            sideband: AjmBuffer {
                                address: read_arena_u64(job_address + 16),
                                size: u64::from(read_arena_u32(job_address + 24)),
                            },
                        },
                    },
                    record_size,
                )
            }
            AJM_OPCODE_STATISTICS => {
                if info.offset - cursor < AJM_JOB_GET_STATISTICS_SIZE
                    || !process_guest_range_has_read_access(
                        job_address,
                        AJM_JOB_GET_STATISTICS_SIZE as usize,
                    )
                {
                    return Err(failure(SCE_AJM_ERROR_INVALID_OPCODE));
                }
                (
                    AjmJob {
                        address: job_address,
                        instance: SCE_AJM_INSTANCE_STATISTICS,
                        kind: AjmJobKind::Statistics {
                            interval: f32::from_le_bytes(
                                read_arena_u32(job_address + 16).to_le_bytes(),
                            ),
                            result: read_arena_u64(job_address + 8),
                        },
                    },
                    AJM_JOB_GET_STATISTICS_SIZE,
                )
            }
            _ => return Err(failure(SCE_AJM_ERROR_INVALID_OPCODE)),
        };
        // A job against an instance must name one that exists in this context, and its codec
        // must have a host implementation — a batch accepted without a decoder behind it
        // would be a fabricated success the audio engine is entitled to catch.
        if job.instance != SCE_AJM_INSTANCE_STATISTICS {
            let codec = state
                .contexts
                .get(&context)
                .and_then(|context_state| context_state.instances.get(&job.instance))
                .map(|instance_state| instance_state.codec);
            let Some(codec) = codec else {
                return Err(failure(SCE_AJM_ERROR_INVALID_INSTANCE));
            };
            let uses = state.codec_job_counts.entry(codec).or_insert(0);
            *uses += 1;
            if *uses == 1 {
                eprintln!(
                    "[REMu][AJM] first job for codec={codec} ({}) host_implementation={}",
                    codec_display_name(codec),
                    if host_codec(codec).is_some() {
                        "yes"
                    } else {
                        "NO - implement this codec next"
                    }
                );
            }
            if host_codec(codec).is_none() {
                return Err(failure(SCE_AJM_ERROR_CODEC_NOT_SUPPORTED));
            }
        }
        // Every referenced buffer must be reachable before the workers touch it.
        let reachable = |buffer: &AjmBuffer| {
            buffer.size == 0
                || (buffer.address != 0
                    && usize::try_from(buffer.size).is_ok_and(|size| {
                        process_guest_range_has_read_access(buffer.address, size)
                    }))
        };
        let buffers_ok = match &job.kind {
            AjmJobKind::Control {
                sideband_input,
                sideband_output,
                ..
            } => reachable(sideband_input) && reachable(sideband_output),
            AjmJobKind::Run {
                inputs,
                outputs,
                sideband,
                ..
            } => {
                inputs.iter().all(reachable)
                    && outputs.iter().all(reachable)
                    && sideband.size >= AJM_SIDEBAND_RESULT_SIZE
                    && reachable(sideband)
            }
            AjmJobKind::Statistics { result, .. } => {
                *result != 0
                    && process_guest_range_has_read_access(
                        *result,
                        AJM_STATISTICS_RESULT_SIZE as usize,
                    )
            }
        };
        if !buffers_ok {
            return Err(failure(SCE_AJM_ERROR_INVALID_ADDRESS));
        }
        jobs.push(job);
        cursor += record_size;
    }
    Ok(jobs)
}

// The arena was written by this module's own builders and validated readable; these reads
// mirror the builder's raw writes.
fn read_arena_u16(address: u64) -> u16 {
    // SAFETY: callers validated the record's readability.
    unsafe { std::ptr::read_unaligned(address as *const u16) }
}

fn read_arena_u32(address: u64) -> u32 {
    // SAFETY: callers validated the record's readability.
    unsafe { std::ptr::read_unaligned(address as *const u32) }
}

fn read_arena_u64(address: u64) -> u64 {
    // SAFETY: callers validated the record's readability.
    unsafe { std::ptr::read_unaligned(address as *const u64) }
}

//----------------------------------------------------------------------------
// Batch execution
//----------------------------------------------------------------------------

/// Runs one batch to a terminal status. Buffer reachability was proven at start; a range
/// that became unreachable since fails the batch with the invalid-address code rather than
/// faulting a worker.
fn execute_batch(state: &mut EngineState, batch_id: u32) {
    let started = Instant::now();
    let Some(batch) = state.batches.get_mut(&batch_id) else {
        return;
    };
    if batch.status != BatchStatus::Pending {
        return;
    }
    batch.status = BatchStatus::Running;
    batch.execution_started = Some(started);
    let context = batch.context;
    let jobs = batch.jobs.clone();
    let mut failure: Option<SceAjmBatchError> = None;
    for job in &jobs {
        if let Err(code) = execute_job(state, context, batch_id, job) {
            failure = Some(SceAjmBatchError {
                error_code: code.cast_signed(),
                job_address: job.address,
                ..SceAjmBatchError::default()
            });
            break;
        }
    }
    let elapsed = started.elapsed();
    state.history.push_back((Instant::now(), elapsed));
    while state.history.len() > 256 {
        state.history.pop_front();
    }
    if let Some(batch) = state.batches.get_mut(&batch_id) {
        batch.status = if failure.is_some() {
            BatchStatus::Failed
        } else {
            BatchStatus::Retired
        };
        batch.error = failure;
    }
}

/// Runs one job, writing its sideband result, or fails the batch with an `SCE_AJM_ERROR_*`
/// code when a referenced range cannot be used.
fn execute_job(
    state: &mut EngineState,
    context: u32,
    batch_id: u32,
    job: &AjmJob,
) -> Result<(), u32> {
    match &job.kind {
        AjmJobKind::Control {
            flags,
            sideband_input,
            sideband_output,
        } => {
            let instance_state = state
                .contexts
                .get_mut(&context)
                .and_then(|context_state| context_state.instances.get_mut(&job.instance))
                .ok_or(SCE_AJM_ERROR_INVALID_INSTANCE)?;
            let codec = host_codec(instance_state.codec).ok_or(SCE_AJM_ERROR_CODEC_NOT_SUPPORTED)?;
            let mut result = 0_i32;
            if flags & AJM_FLAG_CONTROL_INITIALIZE != 0 {
                let parameters = read_guest_buffer(sideband_input)?;
                result = codec.initialize(instance_state, &parameters);
            }
            if flags & AJM_FLAG_CONTROL_RESET != 0 {
                instance_state.total_decoded_samples = 0;
            }
            write_sideband(sideband_output, result, &[])
        }
        AjmJobKind::Run {
            flags,
            inputs,
            outputs,
            sideband,
        } => {
            let instance_state = state
                .contexts
                .get_mut(&context)
                .and_then(|context_state| context_state.instances.get_mut(&job.instance))
                .ok_or(SCE_AJM_ERROR_INVALID_INSTANCE)?;
            let codec = host_codec(instance_state.codec).ok_or(SCE_AJM_ERROR_CODEC_NOT_SUPPORTED)?;
            let mut input = Vec::new();
            for buffer in inputs {
                input.extend_from_slice(&read_guest_buffer(buffer)?);
            }
            let capacity = outputs.iter().map(|buffer| buffer.size).sum::<u64>();
            let output = codec.run(
                instance_state,
                &input,
                usize::try_from(capacity).map_err(|_| SCE_AJM_ERROR_INVALID_ADDRESS)?,
            );
            let total_decoded_samples = instance_state.total_decoded_samples;
            scatter_guest_buffers(outputs, &output.produced)?;
            let mut sections: Vec<Vec<u8>> = Vec::new();
            if flags & AJM_FLAG_SIDEBAND_STREAM != 0 {
                // `SceAjmSidebandStream` (sideband.h:432-437).
                let mut stream = Vec::with_capacity(16);
                stream.extend_from_slice(
                    &i32::try_from(output.consumed).unwrap_or(i32::MAX).to_le_bytes(),
                );
                stream.extend_from_slice(
                    &i32::try_from(output.produced.len())
                        .unwrap_or(i32::MAX)
                        .to_le_bytes(),
                );
                stream.extend_from_slice(&total_decoded_samples.to_le_bytes());
                sections.push(stream);
            }
            if flags & AJM_FLAG_SIDEBAND_FORMAT != 0 {
                sections.push(format_sideband(state, context, job.instance));
            }
            if flags & AJM_FLAG_SIDEBAND_GAPLESS_DECODE != 0 {
                // `SceAjmSidebandGaplessDecode` (sideband.h:551-556): no gapless bounds are
                // set, so the whole bitstream decodes.
                let mut gapless = Vec::with_capacity(8);
                gapless.extend_from_slice(&u32::MAX.to_le_bytes()); // SCE_AJM_SAMPLES_INFINITE, sideband.h:368
                gapless.extend_from_slice(&0_u32.to_le_bytes());
                sections.push(gapless);
            }
            if flags & AJM_FLAG_RUN_GET_CODEC_INFO != 0 {
                sections.push(codec_info_sideband(state, context, job.instance));
            }
            if flags & AJM_FLAG_RUN_MULTIPLE_FRAMES != 0 {
                // `SceAjmSidebandMFrame` (sideband.h:614-618).
                let mut mframe = Vec::with_capacity(8);
                mframe.extend_from_slice(
                    &u32::try_from(output.frames).unwrap_or(u32::MAX).to_le_bytes(),
                );
                mframe.extend_from_slice(&0_u32.to_le_bytes());
                sections.push(mframe);
            }
            let section_slices: Vec<&[u8]> =
                sections.iter().map(std::vec::Vec::as_slice).collect();
            write_sideband(sideband, output.result, &section_slices)
        }
        AjmJobKind::Statistics { interval, result } => {
            let report = statistics_report(state, batch_id, *interval);
            if !process_guest_range_has_read_access(*result, report.len()) {
                return Err(SCE_AJM_ERROR_INVALID_ADDRESS);
            }
            write_guest_bytes(*result, &report);
            Ok(())
        }
    }
}

/// `SceAjmSidebandFormat` for an instance (sideband.h:535-543).
fn format_sideband(state: &EngineState, context: u32, instance: u32) -> Vec<u8> {
    let instance_state = state
        .contexts
        .get(&context)
        .and_then(|context_state| context_state.instances.get(&instance));
    let (channels, sample_rate, encoding) = instance_state.map_or((0, 0, 0), |instance_state| {
        let parameters = instance_state.lpcm;
        (
            parameters.map_or(0, |parameters| parameters.channels),
            parameters.map_or(0, |parameters| parameters.sample_rate),
            instance_state.output_encoding(),
        )
    });
    // Speaker-position presets (sideband.h:311-360); configurations beyond stereo are
    // reported undefined rather than guessed.
    let channel_mask: u32 = match channels {
        1 => 0x4,
        2 => 0x3,
        _ => 0x0,
    };
    let bits_per_sample = encoding_sample_bytes(encoding).unwrap_or(2) * 8;
    let bitrate = u64::from(sample_rate) * u64::from(channels) * bits_per_sample;
    let mut format = Vec::with_capacity(24);
    format.extend_from_slice(&channels.to_le_bytes());
    format.extend_from_slice(&channel_mask.to_le_bytes());
    format.extend_from_slice(&sample_rate.to_le_bytes());
    format.extend_from_slice(&encoding.to_le_bytes());
    format.extend_from_slice(&u32::try_from(bitrate).unwrap_or(u32::MAX).to_le_bytes());
    format.extend_from_slice(&0_u32.to_le_bytes());
    format
}

/// `SceAjmSidebandDecLpcmCodecInfo` (lpcm_decoder.h:81-85): the input encoding and two
/// reserved words.
fn codec_info_sideband(state: &EngineState, context: u32, instance: u32) -> Vec<u8> {
    let encoding = state
        .contexts
        .get(&context)
        .and_then(|context_state| context_state.instances.get(&instance))
        .and_then(|instance_state| instance_state.lpcm)
        .map_or(0, |parameters| parameters.encoding);
    let mut info = Vec::with_capacity(12);
    info.extend_from_slice(&encoding.to_le_bytes());
    info.extend_from_slice(&[0_u8; 8]);
    info
}

/// Builds a `SceAjmGetStatisticsResult` (sideband.h:870-875) from what this engine actually
/// did: real elapsed batch time, real busy fractions from the execution history, and the
/// engine's own memory accounting. Nothing here is invented decode work.
fn statistics_report(state: &EngineState, batch_id: u32, interval: f32) -> Vec<u8> {
    let now = Instant::now();
    let batch = state.batches.get(&batch_id);
    let batch_elapsed = batch
        .and_then(|batch| batch.execution_started)
        .map_or(0.0_f32, |started| started.elapsed().as_secs_f32());
    let busy_within = |window: f32| -> f32 {
        if window <= 0.0 {
            // Zero requests all available history (ajm.h:1377): the busy fraction since the
            // oldest span the engine still remembers.
            let Some((oldest, _)) = state.history.front() else {
                return 0.0;
            };
            let span = now.saturating_duration_since(*oldest).as_secs_f32();
            let busy: f32 = state
                .history
                .iter()
                .map(|(_, spent)| spent.as_secs_f32())
                .sum();
            return if span > f32::EPSILON { busy / span } else { busy.min(1.0) };
        }
        let bound = Duration::from_secs_f32(window.min(3600.0));
        state
            .history
            .iter()
            .filter(|(finished, _)| now.saturating_duration_since(*finished) <= bound)
            .map(|(_, spent)| spent.as_secs_f32())
            .sum::<f32>()
            / window
    };
    let instances: u32 = state
        .contexts
        .values()
        .map(|context_state| {
            u32::try_from(context_state.instances.len()).unwrap_or(u32::MAX)
        })
        .sum();
    let in_flight: u32 = state
        .batches
        .values()
        .map(|batch| {
            batch
                .arena_bytes
                .saturating_add(batch.input_bytes)
                .saturating_add(batch.output_bytes)
        })
        .sum();
    let mut report = Vec::with_capacity(AJM_STATISTICS_RESULT_SIZE as usize);
    // SceAjmSidebandResult (sideband.h:419-423).
    report.extend_from_slice(&0_i32.to_le_bytes());
    report.extend_from_slice(&0_i32.to_le_bytes());
    // SceAjmSidebandStatisticsEngine (sideband.h:809-813).
    report.extend_from_slice(&batch_elapsed.to_le_bytes());
    report.extend_from_slice(&busy_within(interval).to_le_bytes());
    report.extend_from_slice(&(-1.0_f32).to_le_bytes());
    report.extend_from_slice(&(-1.0_f32).to_le_bytes());
    // SceAjmSidebandStatisticsMemory (sideband.h:837-845).
    let instance_used = instances.saturating_mul(AJM_INSTANCE_NOMINAL_SIZE);
    report.extend_from_slice(
        &AJM_INSTANCE_MEMORY_BUDGET
            .saturating_sub(instance_used)
            .to_le_bytes(),
    );
    report.extend_from_slice(&AJM_BUFFER_MEMORY_BUDGET.saturating_sub(in_flight).to_le_bytes());
    let (arena, input, output) = batch.map_or((0, 0, 0), |batch| {
        (batch.arena_bytes, batch.input_bytes, batch.output_bytes)
    });
    report.extend_from_slice(&arena.to_le_bytes());
    report.extend_from_slice(&input.to_le_bytes());
    report.extend_from_slice(&output.to_le_bytes());
    report.extend_from_slice(&0_u32.to_le_bytes());
    report
}

/// Copies one referenced guest buffer, re-validating reachability at use time.
fn read_guest_buffer(buffer: &AjmBuffer) -> Result<Vec<u8>, u32> {
    let length = usize::try_from(buffer.size).map_err(|_| SCE_AJM_ERROR_INVALID_ADDRESS)?;
    if length == 0 {
        return Ok(Vec::new());
    }
    if buffer.address == 0 || !process_guest_range_has_read_access(buffer.address, length) {
        return Err(SCE_AJM_ERROR_INVALID_ADDRESS);
    }
    let mut bytes = vec![0_u8; length];
    // SAFETY: readability of the complete range was validated immediately above.
    unsafe {
        std::ptr::copy_nonoverlapping(buffer.address as *const u8, bytes.as_mut_ptr(), length);
    }
    Ok(bytes)
}

/// Scatters produced bytes across a job's output buffers in order.
fn scatter_guest_buffers(outputs: &[AjmBuffer], bytes: &[u8]) -> Result<(), u32> {
    let mut written = 0_usize;
    for buffer in outputs {
        if written == bytes.len() {
            break;
        }
        let capacity = usize::try_from(buffer.size).map_err(|_| SCE_AJM_ERROR_INVALID_ADDRESS)?;
        let chunk = capacity.min(bytes.len() - written);
        if chunk == 0 {
            continue;
        }
        if buffer.address == 0 || !process_guest_range_has_read_access(buffer.address, chunk) {
            return Err(SCE_AJM_ERROR_INVALID_ADDRESS);
        }
        write_guest_bytes(buffer.address, &bytes[written..written + chunk]);
        written += chunk;
    }
    Ok(())
}

/// Writes a job's sideband output: the result first (sideband.h:416-417), then each
/// requested section in declaration order. Sections beyond the caller's capacity are
/// dropped and reported through `SCE_AJM_RESULT_SIDEBAND_TRUNCATED` (error.h:283).
fn write_sideband(sideband: &AjmBuffer, result: i32, sections: &[&[u8]]) -> Result<(), u32> {
    if sideband.address == 0 && sideband.size == 0 {
        return Ok(());
    }
    let capacity = usize::try_from(sideband.size).map_err(|_| SCE_AJM_ERROR_INVALID_ADDRESS)?;
    if capacity < AJM_SIDEBAND_RESULT_SIZE as usize
        || !process_guest_range_has_read_access(sideband.address, capacity)
    {
        return Err(SCE_AJM_ERROR_INVALID_ADDRESS);
    }
    let mut payload = Vec::with_capacity(capacity);
    payload.extend_from_slice(&result.to_le_bytes());
    payload.extend_from_slice(&0_i32.to_le_bytes());
    let mut truncated = false;
    for section in sections {
        if payload.len() + section.len() <= capacity {
            payload.extend_from_slice(section);
        } else {
            truncated = true;
        }
    }
    if truncated {
        payload[0..4].copy_from_slice(&(result | SCE_AJM_RESULT_SIDEBAND_TRUNCATED).to_le_bytes());
    }
    write_guest_bytes(sideband.address, &payload);
    Ok(())
}

//----------------------------------------------------------------------------
// Job builders
//----------------------------------------------------------------------------

/// `sceAjmBatchInitialize(pBuffer, szBuffer, pInfo)` (ajm.h:850-879). Only the five-qword
/// cursor is initialized; the arena belongs to the caller and job builders append into it.
fn batch_initialize(arguments: [u64; 6]) -> u64 {
    let batch = arguments[0];
    let size = arguments[1];
    let info = arguments[2];
    let Ok(batch_size) = usize::try_from(size) else {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    };
    if batch == 0
        || info == 0
        || !(AJM_BATCH_MIN_SIZE..=AJM_BATCH_MAX_SIZE).contains(&size)
        || !process_guest_range_has_read_access(batch, batch_size)
        || guest_output_pointer(info, std::mem::size_of::<SceAjmBatchInfo>()).is_none()
    {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    }
    let batch_info = SceAjmBatchInfo {
        buffer: batch,
        offset: 0,
        size,
        last_good_job: 0,
        last_good_job_return_address: 0,
    };
    // SAFETY: the complete 0x28-byte output object was validated above.
    unsafe {
        std::ptr::write_unaligned(info as *mut SceAjmBatchInfo, batch_info);
    }
    0
}

/// Reserves `job_size` bytes in a batch arena and returns the job's address, updating the
/// caller's cursor exactly as the console library does.
fn append_batch_job(info: u64, job_size: u64) -> Result<u64, u32> {
    if info == 0 || guest_output_pointer(info, std::mem::size_of::<SceAjmBatchInfo>()).is_none() {
        return Err(SCE_AJM_ERROR_INVALID_PARAMETER);
    }
    // SAFETY: the complete cursor object was validated immediately above.
    let mut batch_info = unsafe { std::ptr::read_unaligned(info as *const SceAjmBatchInfo) };
    let Some(end) = batch_info.offset.checked_add(job_size) else {
        return Err(SCE_AJM_ERROR_JOB_CREATION);
    };
    if batch_info.buffer == 0 || end > batch_info.size {
        return Err(SCE_AJM_ERROR_JOB_CREATION);
    }
    let job = batch_info.buffer.wrapping_add(batch_info.offset);
    guest_memset(job, 0, job_size);
    batch_info.offset = end;
    batch_info.last_good_job = job;
    batch_info.last_good_job_return_address = 0;
    // SAFETY: the same validated cursor object.
    unsafe {
        std::ptr::write_unaligned(info as *mut SceAjmBatchInfo, batch_info);
    }
    Ok(job)
}

fn write_arena_u16(address: u64, value: u16) {
    // SAFETY: `append_batch_job` reserved this record inside the validated arena.
    unsafe { std::ptr::write_unaligned(address as *mut u16, value) }
}

fn write_arena_u32(address: u64, value: u32) {
    // SAFETY: `append_batch_job` reserved this record inside the validated arena.
    unsafe { std::ptr::write_unaligned(address as *mut u32, value) }
}

fn write_arena_u64(address: u64, value: u64) {
    // SAFETY: `append_batch_job` reserved this record inside the validated arena.
    unsafe { std::ptr::write_unaligned(address as *mut u64, value) }
}

/// Appends one control record.
fn append_control_job(
    info: u64,
    instance: u32,
    flags: u64,
    sideband_input: AjmBuffer,
    sideband_output: AjmBuffer,
) -> u64 {
    let job = match append_batch_job(info, AJM_JOB_CONTROL_SIZE) {
        Ok(job) => job,
        Err(error) => return sce_kernel_error(error),
    };
    write_arena_u32(job, AJM_JOB_MAGIC | AJM_OPCODE_CONTROL);
    write_arena_u32(job + 4, instance);
    write_arena_u64(job + 8, flags);
    write_arena_u64(job + 16, sideband_input.address);
    write_arena_u64(job + 24, sideband_input.size);
    write_arena_u64(job + 32, sideband_output.address);
    write_arena_u64(job + 40, sideband_output.size);
    0
}

/// Appends one run record.
fn append_run_job(
    info: u64,
    instance: u32,
    flags: u64,
    input: AjmBuffer,
    output: AjmBuffer,
    sideband: AjmBuffer,
) -> u64 {
    let job = match append_batch_job(info, AJM_JOB_RUN_SIZE) {
        Ok(job) => job,
        Err(error) => return sce_kernel_error(error),
    };
    write_arena_u32(job, AJM_JOB_MAGIC | AJM_OPCODE_RUN);
    write_arena_u32(job + 4, instance);
    write_arena_u64(job + 8, flags);
    write_arena_u64(job + 16, input.address);
    write_arena_u64(job + 24, input.size);
    write_arena_u64(job + 32, output.address);
    write_arena_u64(job + 40, output.size);
    write_arena_u64(job + 48, sideband.address);
    write_arena_u64(job + 56, sideband.size);
    0
}

/// `sceAjmBatchJobInitialize(pInfo, uiInstance, pCodecParameters, szCodecParametersSize,
/// pResult)` (ajm.h:882-908). The parameter blob is referenced, not copied: the caller owns
/// it until the batch retires.
fn batch_job_initialize(arguments: [u64; 6]) -> u64 {
    let (info, instance) = (arguments[0], low_u32(arguments[1]));
    let parameters = AjmBuffer {
        address: arguments[2],
        size: arguments[3],
    };
    let result = AjmBuffer {
        address: arguments[4],
        size: AJM_SIDEBAND_RESULT_SIZE,
    };
    append_control_job(info, instance, AJM_FLAG_CONTROL_INITIALIZE, parameters, result)
}

/// `sceAjmBatchJobClearContext(pInfo, uiInstance, pResult)` (ajm.h:910-929).
fn batch_job_clear_context(arguments: [u64; 6]) -> u64 {
    let (info, instance) = (arguments[0], low_u32(arguments[1]));
    let result = AjmBuffer {
        address: arguments[2],
        size: AJM_SIDEBAND_RESULT_SIZE,
    };
    append_control_job(
        info,
        instance,
        AJM_FLAG_CONTROL_RESET,
        AjmBuffer { address: 0, size: 0 },
        result,
    )
}

/// `sceAjmBatchJobRun(pInfo, uiInstance, uiFlags, pDataInput, szDataInputSize, pDataOutput,
/// szDataOutputSize, pSidebandOutput, szSidebandOutputSize)` (ajm.h:1418-1445). The last
/// three parameters arrive on the stack.
fn batch_job_run(arguments: [u64; 6], stack_arguments: [u64; 6]) -> u64 {
    let sideband = AjmBuffer {
        address: stack_arguments[1],
        size: stack_arguments[2],
    };
    if sideband.size < AJM_SIDEBAND_RESULT_SIZE {
        // The sideband output must hold at least a SceAjmSidebandResult (ajm.h:1435-1437).
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    }
    append_run_job(
        arguments[0],
        low_u32(arguments[1]),
        arguments[2],
        AjmBuffer {
            address: arguments[3],
            size: arguments[4],
        },
        AjmBuffer {
            address: arguments[5],
            size: stack_arguments[0],
        },
        sideband,
    )
}

/// `sceAjmBatchJobDecode(pInfo, uiInstance, pBitstreamInput, szBitstreamInputSize,
/// pPcmOutput, szPcmOutputSize, pResult)` (ajm.h:931-957). Decodes as many frames as
/// possible; the result is a `SceAjmDecodeResult` — result, stream, and multiple-frame
/// sideband (sideband.h:661-666) — so the equivalent run flags are implied.
fn batch_job_decode(arguments: [u64; 6], stack_arguments: [u64; 6]) -> u64 {
    append_run_job(
        arguments[0],
        low_u32(arguments[1]),
        AJM_FLAG_RUN_MULTIPLE_FRAMES | AJM_FLAG_SIDEBAND_STREAM,
        AjmBuffer {
            address: arguments[2],
            size: arguments[3],
        },
        AjmBuffer {
            address: arguments[4],
            size: arguments[5],
        },
        AjmBuffer {
            address: stack_arguments[0],
            // SceAjmDecodeResult: result + stream + mframe (sideband.h:661-666).
            size: 32,
        },
    )
}

/// Appends one split-run record: a 32-byte header followed by the caller's input then output
/// `SceAjmBuffer` descriptors, 16 bytes each (`SCE_AJM_JOB_RUN_SPLIT_SIZE`, ajm.h:208).
fn append_run_split_job(
    info: u64,
    instance: u32,
    flags: u64,
    input_buffers: u64,
    input_count: u64,
    output_buffers: u64,
    output_count: u64,
    sideband: AjmBuffer,
) -> u64 {
    if input_count > u64::from(u16::MAX)
        || output_count > u64::from(u16::MAX)
        || sideband.size > u64::from(u32::MAX)
    {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    }
    let descriptors = |address: u64, count: u64| -> Option<u64> {
        if count == 0 {
            return Some(address);
        }
        let length = usize::try_from(count.checked_mul(16)?).ok()?;
        (address != 0 && process_guest_range_has_read_access(address, length)).then_some(address)
    };
    let (Some(inputs), Some(outputs)) = (
        descriptors(input_buffers, input_count),
        descriptors(output_buffers, output_count),
    ) else {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    };
    let record_size = ajm_job_run_split_size(input_count + output_count);
    let job = match append_batch_job(info, record_size) {
        Ok(job) => job,
        Err(error) => return sce_kernel_error(error),
    };
    write_arena_u32(job, AJM_JOB_MAGIC | AJM_OPCODE_RUN_SPLIT);
    write_arena_u32(job + 4, instance);
    write_arena_u64(job + 8, flags);
    write_arena_u64(job + 16, sideband.address);
    write_arena_u32(job + 24, u32::try_from(sideband.size).expect("bounded above"));
    write_arena_u16(job + 28, u16::try_from(input_count).expect("bounded above"));
    write_arena_u16(job + 30, u16::try_from(output_count).expect("bounded above"));
    for index in 0..input_count {
        let source = inputs + index * 16;
        write_arena_u64(job + 32 + index * 16, read_arena_u64(source));
        write_arena_u64(job + 40 + index * 16, read_arena_u64(source + 8));
    }
    for index in 0..output_count {
        let source = outputs + index * 16;
        let slot = input_count + index;
        write_arena_u64(job + 32 + slot * 16, read_arena_u64(source));
        write_arena_u64(job + 40 + slot * 16, read_arena_u64(source + 8));
    }
    0
}

/// `sceAjmBatchJobRunSplit(pInfo, uiInstance, uiFlags, pDataInputBuffers,
/// szNumDataInputBuffers, pDataOutputBuffers, szNumDataOutputBuffers, pSidebandOutput,
/// szSidebandOutputSize)` (ajm.h:1447-1480).
fn batch_job_run_split(arguments: [u64; 6], stack_arguments: [u64; 6]) -> u64 {
    let sideband = AjmBuffer {
        address: stack_arguments[1],
        size: stack_arguments[2],
    };
    if sideband.size < AJM_SIDEBAND_RESULT_SIZE {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    }
    append_run_split_job(
        arguments[0],
        low_u32(arguments[1]),
        arguments[2],
        arguments[3],
        arguments[4],
        arguments[5],
        stack_arguments[0],
        sideband,
    )
}

/// `sceAjmBatchJobDecodeSplit(pInfo, uiInstance, pDataInputBuffers, szNumDataInputBuffers,
/// pDataOutputBuffers, szNumDataOutputBuffers, pResult)` (ajm.h:987-1014).
fn batch_job_decode_split(arguments: [u64; 6], stack_arguments: [u64; 6]) -> u64 {
    append_run_split_job(
        arguments[0],
        low_u32(arguments[1]),
        AJM_FLAG_RUN_MULTIPLE_FRAMES | AJM_FLAG_SIDEBAND_STREAM,
        arguments[2],
        arguments[3],
        arguments[4],
        arguments[5],
        AjmBuffer {
            address: stack_arguments[0],
            size: 32,
        },
    )
}

/// `sceAjmBatchJobGetStatistics(pInfo, fInterval, pResult)` (ajm.h:1370-1388). The interval
/// travels in a floating-point register the stateless dispatcher does not capture; zero —
/// all available history (ajm.h:1377) — is recorded instead, which is the honest default
/// rather than a guessed window.
fn batch_job_get_statistics(arguments: [u64; 6]) -> u64 {
    let result = arguments[1];
    let job = match append_batch_job(arguments[0], AJM_JOB_GET_STATISTICS_SIZE) {
        Ok(job) => job,
        Err(error) => return sce_kernel_error(error),
    };
    if result != 0 {
        if guest_output_pointer(result, AJM_STATISTICS_RESULT_SIZE as usize).is_none() {
            return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
        }
        guest_memset(result, 0, AJM_STATISTICS_RESULT_SIZE);
    }
    write_arena_u32(job, AJM_JOB_MAGIC | AJM_OPCODE_STATISTICS);
    write_arena_u64(job + 8, result);
    write_arena_u32(job + 16, 0.0_f32.to_bits());
    0
}

//----------------------------------------------------------------------------
// Dispatch
//----------------------------------------------------------------------------

/// Handles a `sceAjm*` NID, or returns `None` if the engine owns no such NID. Consulted from
/// the pre-lock import tier: batch waits park on the engine's own condition variable, never
/// the process-wide dispatch lock.
pub(crate) fn dispatch(nid: &str, arguments: [u64; 6], stack_arguments: [u64; 6]) -> Option<u64> {
    Some(match nid {
        "dl+4eHSzUu4" => AjmEngine::global().initialize(arguments),
        "MHur6qCsUus" => AjmEngine::global().finalize(arguments),
        "Q3dyFuwGn64" => AjmEngine::global().module_register(arguments),
        "Wi7DtlLV+KI" => AjmEngine::global().module_unregister(arguments),
        "AxoDrINp4J8" => AjmEngine::global().instance_create(arguments),
        "RbLbuKv8zho" => AjmEngine::global().instance_destroy(arguments),
        "bkRHEYG6lEM" => AjmEngine::global().memory_register(arguments),
        "pIpGiaYkHkM" => AjmEngine::global().memory_unregister(arguments),
        "MmpF1XsQiHw" => batch_initialize(arguments),
        "ezM2OhNxzck" => batch_job_initialize(arguments),
        "uJ3m8INuikg" => batch_job_clear_context(arguments),
        "jVCWcthifr8" => batch_job_run(arguments, stack_arguments),
        "39WxhR-ePew" => batch_job_decode(arguments, stack_arguments),
        "Z9NVCesiP0Q" => batch_job_run_split(arguments, stack_arguments),
        "SJ3i0DXP8vg" => batch_job_decode_split(arguments, stack_arguments),
        "3cAg7xN995U" => batch_job_get_statistics(arguments),
        "5tOfnaClcqM" => AjmEngine::global().batch_start(arguments),
        "-qLsfDAywIY" => AjmEngine::global().batch_wait(arguments),
        "NVDXiUesSbA" => AjmEngine::global().batch_cancel(arguments),
        _ => return None,
    })
}

/// Resets the process engine for a fresh title, if one was ever started. Wired from
/// `configure_probe_dispatch`, which exists only on the x86-guest dispatch path.
#[cfg(any(test, target_arch = "x86_64"))]
pub(crate) fn reset_for_new_process() {
    if let Some(engine) = ENGINE.get() {
        engine.reset();
    }
}

#[cfg(test)]
mod tests;
