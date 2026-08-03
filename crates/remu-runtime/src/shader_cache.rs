// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! The translated-shader cache and the worker pool that fills it.
//!
//! Draw submission looks a shader up by its guest code bytes and user-data bindings, so the key
//! is hashed from raw memory rather than from a translated program. Misses are handed to worker
//! threads; the guest thread keeps a pending entry so two draws never translate the same shader
//! twice.

use std::collections::{HashMap, VecDeque};
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

use crate::ProcessGuestMemory;
#[cfg(target_os = "macos")]
use crate::cached_kernel_memory_readable_range_end;

pub(crate) const SHADER_EVALUATION_CACHE_CAPACITY: usize = 16 * 1024;

pub(crate) const SHADER_EVALUATION_PENDING_CAPACITY: usize = SHADER_EVALUATION_CACHE_CAPACITY * 4;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ShaderEvaluationCacheKey {
    pub(crate) program_address: u64,
    pub(crate) user_data: ShaderEvaluationUserDataKey,
    pub(crate) compute_system_registers: Option<remu_shader::ComputeSystemRegisters>,
    pub(crate) user_data_scalar_register_base: u32,
    pub(crate) options: remu_shader::EvaluationOptions,
}

#[derive(Clone, Debug)]
pub(crate) struct ShaderEvaluationUserDataKey {
    pub(crate) words: [u32; 64],
    pub(crate) len: u8,
}

impl ShaderEvaluationUserDataKey {
    pub(crate) fn as_slice(&self) -> &[u32] {
        &self.words[..usize::from(self.len)]
    }
}

impl PartialEq for ShaderEvaluationUserDataKey {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.as_slice() == other.as_slice()
    }
}

impl Eq for ShaderEvaluationUserDataKey {}

impl Hash for ShaderEvaluationUserDataKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len.hash(state);
        self.as_slice().hash(state);
    }
}

#[derive(Debug)]
pub(crate) struct ShaderEvaluationCacheHasher(u64);

impl Default for ShaderEvaluationCacheHasher {
    fn default() -> Self {
        Self(0x517c_c1b7_2722_0a95)
    }
}

impl ShaderEvaluationCacheHasher {
    #[inline]
    pub(crate) fn mix(&mut self, value: u64) {
        self.0 = (self.0.rotate_left(5) ^ value).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

impl Hasher for ShaderEvaluationCacheHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            let mut word = [0_u8; 8];
            word.copy_from_slice(chunk);
            self.mix(u64::from_ne_bytes(word));
        }
        let remainder = chunks.remainder();
        if !remainder.is_empty() {
            let mut word = [0_u8; 8];
            word[..remainder.len()].copy_from_slice(remainder);
            self.mix(u64::from_ne_bytes(word) ^ ((remainder.len() as u64) << 56));
        }
    }

    #[inline]
    fn write_u8(&mut self, value: u8) {
        self.mix(u64::from(value));
    }

    #[inline]
    fn write_u32(&mut self, value: u32) {
        self.mix(u64::from(value));
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.mix(value);
    }

    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.mix(value as u64);
    }
}

pub(crate) type ShaderEvaluationCacheBuildHasher = BuildHasherDefault<ShaderEvaluationCacheHasher>;

pub(crate) type ShaderEvaluationCacheValue = Result<Arc<remu_shader::ShaderEvaluation>, Arc<str>>;

pub(crate) fn retire_shader_evaluation(value: ShaderEvaluationCacheValue) {
    static RETIRER: OnceLock<std::sync::mpsc::Sender<ShaderEvaluationCacheValue>> = OnceLock::new();
    let sender = RETIRER.get_or_init(|| {
        let (sender, receiver) = std::sync::mpsc::channel::<ShaderEvaluationCacheValue>();
        if let Err(error) = std::thread::Builder::new()
            .name("REMu shader cache reclaimer".into())
            .spawn(move || {
                while let Ok(value) = receiver.recv() {
                    drop(value);
                }
            })
        {
            eprintln!("[REMu][AGC] could not start shader cache reclaimer: {error}");
        }
        sender
    });
    if let Err(error) = sender.send(value) {
        drop(error.0);
    }
}

#[derive(Debug)]
pub(crate) struct ShaderEvaluationCacheEntry {
    pub(crate) value: ShaderEvaluationCacheValue,
    pub(crate) admission_id: u64,
}

pub(crate) type ShaderEvaluationCacheEntries =
    HashMap<ShaderEvaluationCacheKey, ShaderEvaluationCacheEntry, ShaderEvaluationCacheBuildHasher>;

pub(crate) type ShaderEvaluationPendingEntries =
    HashMap<ShaderEvaluationCacheKey, u64, ShaderEvaluationCacheBuildHasher>;

#[derive(Debug, Default)]
pub(crate) struct ShaderEvaluationCache {
    pub(crate) entries: ShaderEvaluationCacheEntries,
    pub(crate) seen_once: ShaderEvaluationPendingEntries,
    pub(crate) pending_admission_order: VecDeque<(ShaderEvaluationCacheKey, u64)>,
    pub(crate) admission_order: VecDeque<(ShaderEvaluationCacheKey, u64)>,
    pub(crate) next_admission_id: u64,
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) failures: u64,
    pub(crate) invalidations: u64,
    pub(crate) evictions: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ShaderEvaluationCacheStats {
    pub(crate) entries: usize,
    pub(crate) pending_admissions: usize,
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) failures: u64,
    pub(crate) invalidations: u64,
    pub(crate) evictions: u64,
}

impl ShaderEvaluationCache {
    pub(crate) fn stats(&self) -> ShaderEvaluationCacheStats {
        ShaderEvaluationCacheStats {
            entries: self.entries.len(),
            pending_admissions: self.seen_once.len(),
            hits: self.hits,
            misses: self.misses,
            failures: self.failures,
            invalidations: self.invalidations,
            evictions: self.evictions,
        }
    }
}

impl ShaderEvaluationCacheStats {
    pub(crate) fn merge(self, other: Self) -> Self {
        Self {
            entries: self.entries.saturating_add(other.entries),
            pending_admissions: self
                .pending_admissions
                .saturating_add(other.pending_admissions),
            hits: self.hits.saturating_add(other.hits),
            misses: self.misses.saturating_add(other.misses),
            failures: self.failures.saturating_add(other.failures),
            invalidations: self.invalidations.saturating_add(other.invalidations),
            evictions: self.evictions.saturating_add(other.evictions),
        }
    }
}

pub(crate) struct ShaderEvaluationWorkerRequest {
    pub(crate) state: remu_shader::ShaderState,
    pub(crate) options: remu_shader::EvaluationOptions,
}

pub(crate) struct ShaderEvaluationWorkerResult {
    pub(crate) evaluation: Result<Arc<remu_shader::ShaderEvaluation>, String>,
    pub(crate) stats: ShaderEvaluationCacheStats,
}

#[derive(Debug)]
pub(crate) struct ShaderEvaluationWorker {
    pub(crate) requests: std::sync::mpsc::SyncSender<ShaderEvaluationWorkerRequest>,
    pub(crate) results: Mutex<std::sync::mpsc::Receiver<ShaderEvaluationWorkerResult>>,
    pub(crate) stats: ShaderEvaluationCacheStats,
}

impl ShaderEvaluationWorker {
    pub(crate) fn spawn(name: &str) -> Result<Self, String> {
        let (requests, request_receiver) =
            std::sync::mpsc::sync_channel::<ShaderEvaluationWorkerRequest>(1);
        let (result_sender, results) =
            std::sync::mpsc::sync_channel::<ShaderEvaluationWorkerResult>(1);
        std::thread::Builder::new()
            .name(name.to_owned())
            .spawn(move || {
                let mut cache = ShaderEvaluationCache::default();
                while let Ok(request) = request_receiver.recv() {
                    let evaluation = cache.evaluate(&request.state, request.options);
                    if result_sender
                        .send(ShaderEvaluationWorkerResult {
                            evaluation,
                            stats: cache.stats(),
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            })
            .map_err(|error| format!("could not start {name}: {error}"))?;
        Ok(Self {
            requests,
            results: Mutex::new(results),
            stats: ShaderEvaluationCacheStats::default(),
        })
    }

    pub(crate) fn submit(&self, request: ShaderEvaluationWorkerRequest) -> Result<(), String> {
        self.requests
            .send(request)
            .map_err(|_| "shader evaluation worker stopped".to_owned())
    }

    pub(crate) fn receive(&mut self) -> Result<Arc<remu_shader::ShaderEvaluation>, String> {
        let result = self
            .results
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .recv()
            .map_err(|_| "shader evaluation worker stopped".to_owned())?;
        self.stats = result.stats;
        result.evaluation
    }
}

#[derive(Debug)]
pub(crate) struct ShaderEvaluationWorkers {
    pub(crate) parallel: ShaderEvaluationWorker,
}

impl ShaderEvaluationWorkers {
    pub(crate) fn new() -> Result<Self, String> {
        Ok(Self {
            parallel: ShaderEvaluationWorker::spawn("REMu shader evaluator")?,
        })
    }

    pub(crate) fn stats(&self) -> ShaderEvaluationCacheStats {
        self.parallel.stats
    }
}

pub(crate) fn canonical_vertex_record_count(record_count: Option<u32>) -> Option<u32> {
    record_count.map(|count| count.max(1).checked_next_power_of_two().unwrap_or(u32::MAX))
}

pub(crate) fn shader_evaluation_user_data_key(
    state: &remu_shader::ShaderState,
) -> ShaderEvaluationUserDataKey {
    let scalar_base = state.user_data_scalar_register_base;
    let mut words = [0_u32; 64];
    let mut len = 0_usize;
    for user_data_index in state
        .program
        .input_scalar_registers()
        .iter()
        .copied()
        .filter_map(|register| register.checked_sub(scalar_base))
        .filter(|offset| *offset < 64)
    {
        let user_data_index = usize::try_from(user_data_index).unwrap_or(usize::MAX);
        words[len] = state
            .user_data
            .get(user_data_index)
            .copied()
            .unwrap_or_default();
        len += 1;
    }
    ShaderEvaluationUserDataKey {
        words,
        len: u8::try_from(len).expect("hardware user data has at most 64 scalar registers"),
    }
}

pub(crate) fn shader_evaluation_cache_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| {
        std::env::var_os("REMU_DISABLE_SHADER_EVALUATION_CACHE").is_some_and(|value| value == "1")
    })
}

impl ShaderEvaluationCache {
    pub(crate) fn evaluate(
        &mut self,
        state: &remu_shader::ShaderState,
        mut options: remu_shader::EvaluationOptions,
    ) -> Result<Arc<remu_shader::ShaderEvaluation>, String> {
        // Draws using the same resources commonly differ only by a few indices. Binding the next
        // power-of-two record range is still bounded by the guest descriptor, yet lets those draws
        // share one evaluated shader instead of continuously rotating this cache.
        options.required_vertex_record_count =
            canonical_vertex_record_count(options.required_vertex_record_count);
        if shader_evaluation_cache_disabled() {
            self.misses = self.misses.saturating_add(1);
            return remu_shader::evaluate_shader(&ProcessGuestMemory, state, options)
                .map(Arc::new)
                .map_err(|error| {
                    self.failures = self.failures.saturating_add(1);
                    format!("evaluation-error({error})")
                });
        }
        let key = ShaderEvaluationCacheKey {
            program_address: state.program.address,
            // Only scalar inputs actually referenced by the decoded program can affect control
            // flow, resource discovery, or the generated Metal shader. Unity updates unused
            // hardware user-data slots extremely often, so including them here caused false
            // misses and expensive whole-generation cache destruction.
            user_data: shader_evaluation_user_data_key(state),
            compute_system_registers: state.compute_system_registers,
            user_data_scalar_register_base: state.user_data_scalar_register_base,
            options,
        };
        if let Some(cached) = self.entries.get(&key).map(|entry| entry.value.clone()) {
            if cached
                .as_ref()
                .is_ok_and(|evaluation| shader_evaluation_dependencies_are_current(evaluation))
            {
                self.hits = self.hits.saturating_add(1);
                return cached.map_err(|error| error.to_string());
            }
            // The pointer-valued user data can be identical while the guest rewrites the image,
            // sampler, or buffer descriptor behind it. A stale evaluation is worse than a miss:
            // it binds unrelated textures and can turn a complete frame into partial geometry.
            if let Some(retired) = self.entries.remove(&key) {
                retire_shader_evaluation(retired.value);
            }
            self.invalidations = self.invalidations.saturating_add(1);
        }
        self.misses = self.misses.saturating_add(1);
        let evaluated = remu_shader::evaluate_shader(&ProcessGuestMemory, state, options)
            .map(Arc::new)
            .map_err(|error| Arc::<str>::from(format!("evaluation-error({error})")));
        if evaluated.is_err() {
            self.failures = self.failures.saturating_add(1);
        }
        // Most Unity resource states are used by one draw only. Admit only successful states that
        // have been observed twice. Once the cache is full, evict one oldest live admission for
        // each replacement.
        // Whole-generation replacement caused large object graphs to retire together, visibly
        // stalling guest submission and starving audio for several frames.
        // Evaluation failures often describe a transient null descriptor while Unity is
        // populating a command buffer. Never make such a failure persistent for the same pointer
        // state; only successful evaluations participate in the two-hit admission filter.
        let admit = if evaluated.is_ok() {
            if self.seen_once.remove(&key).is_some() {
                true
            } else {
                while self.pending_admission_order.len() >= SHADER_EVALUATION_PENDING_CAPACITY {
                    let Some((old_key, admission_id)) = self.pending_admission_order.pop_front()
                    else {
                        break;
                    };
                    let is_current = self
                        .seen_once
                        .get(&old_key)
                        .is_some_and(|current| *current == admission_id);
                    if is_current {
                        self.seen_once.remove(&old_key);
                    }
                }
                let admission_id = self.next_admission_id;
                self.next_admission_id = self.next_admission_id.wrapping_add(1);
                self.seen_once.insert(key.clone(), admission_id);
                self.pending_admission_order
                    .push_back((key.clone(), admission_id));
                false
            }
        } else {
            false
        };
        if admit {
            self.seen_once.remove(&key);
            while self.entries.len() >= SHADER_EVALUATION_CACHE_CAPACITY {
                let Some((old_key, admission_id)) = self.admission_order.pop_front() else {
                    break;
                };
                let is_current = self
                    .entries
                    .get(&old_key)
                    .is_some_and(|entry| entry.admission_id == admission_id);
                if is_current
                    && let Some(retired) = self.entries.remove(&old_key) {
                        retire_shader_evaluation(retired.value);
                        self.evictions = self.evictions.saturating_add(1);
                    }
            }
            let admission_id = self.next_admission_id;
            self.next_admission_id = self.next_admission_id.wrapping_add(1);
            self.entries.insert(
                key.clone(),
                ShaderEvaluationCacheEntry {
                    value: evaluated.clone(),
                    admission_id,
                },
            );
            self.admission_order.push_back((key, admission_id));
        }
        evaluated.map_err(|error| error.to_string())
    }
}

pub(crate) fn shader_evaluation_dependencies_are_current(
    evaluation: &remu_shader::ShaderEvaluation,
) -> bool {
    #[cfg(target_os = "macos")]
    {
        // Dependencies are address-sorted by the evaluator. Reuse the containing readable range
        // for every following word instead of performing a TLS generation check and binary search
        // for each four-byte scalar load on every cache hit.
        let mut readable_end = 0_u64;
        for &(address, expected) in &evaluation.scalar_memory_reads {
            let Some(word_end) = address.checked_add(4) else {
                return false;
            };
            if word_end > readable_end {
                let Some(mapped_end) = cached_kernel_memory_readable_range_end(address) else {
                    return false;
                };
                if word_end > mapped_end {
                    return false;
                }
                readable_end = mapped_end;
            }
            // SAFETY: guest mapping mutations are serialized by the process dispatch lock. The
            // cached range snapshot proves this identity-mapped word readable for this dispatch.
            let current = unsafe { (address as *const u32).read_unaligned() };
            if u32::from_le(current) != expected {
                return false;
            }
        }
        true
    }
    #[cfg(not(target_os = "macos"))]
    {
        evaluation
            .scalar_memory_reads
            .iter()
            .all(|&(address, expected)| {
                let mut bytes = [0_u8; 4];
                ProcessGuestMemory.read(address, &mut bytes)
                    && u32::from_le_bytes(bytes) == expected
            })
    }
}
