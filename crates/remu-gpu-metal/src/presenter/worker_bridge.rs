// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Bounded protocol between the AppKit shell and the Metal translation worker.
//!
//! This module deliberately contains no AppKit types. Its locks protect only small queues and
//! value slots; callers must drop the returned guard-owned values before touching Metal, guest
//! memory, or the window system.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

const APP_CONTROL_CAPACITY: usize = 32;
const APP_CONTROL_LIFECYCLE_CAPACITY: usize = APP_CONTROL_CAPACITY - 1;
pub(super) const FRAME_POOL_CAPACITY: usize = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum WorkerControl {
    SurfaceAvailable,
    SurfaceUnavailable,
    BeginShutdown,
}

#[derive(Debug, Eq, PartialEq)]
pub(super) struct ControlMailboxOverflow {
    pub(super) attempted: WorkerControl,
    pub(super) depth: usize,
}

#[derive(Default)]
pub(super) struct AppControlMailbox {
    state: Mutex<AppControlState>,
}

#[derive(Default)]
struct AppControlState {
    queue: VecDeque<WorkerControl>,
    shutdown_queued: bool,
}

impl AppControlMailbox {
    pub(super) fn send(&self, control: WorkerControl) -> Result<bool, ControlMailboxOverflow> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Shutdown is a terminal mailbox state. AppKit can report repeated lifecycle edges while
        // a window is closing; none may appear behind BeginShutdown and reopen the worker.
        if state.shutdown_queued {
            return Ok(false);
        }
        if control == WorkerControl::BeginShutdown {
            // Non-shutdown messages can consume at most 31 slots, so shutdown is non-droppable.
            state.shutdown_queued = true;
            state.queue.push_back(control);
            return Ok(true);
        }
        if state.queue.len() >= APP_CONTROL_LIFECYCLE_CAPACITY {
            return Err(ControlMailboxOverflow {
                attempted: control,
                depth: state.queue.len(),
            });
        }
        state.queue.push_back(control);
        Ok(true)
    }

    pub(super) fn drain(&self) -> Vec<WorkerControl> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // `shutdown_queued` deliberately remains set after draining. There is no normal reopen
        // transition: a new run constructs a new mailbox.
        state.queue.drain(..).collect()
    }

    pub(super) fn depth(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .queue
            .len()
    }
}

const RECEIPT_OFFERED: u8 = 0;
const RECEIPT_SUBMITTED: u8 = 1;
const RECEIPT_SUPERSEDED: u8 = 2;
const RECEIPT_SURFACE_LESS: u8 = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PresentationDisposition {
    Presentation,
    Superseded,
    SurfaceLess,
    Error,
}

/// Atomic join observed by AppKit callbacks and retired by the translation worker.
pub(super) struct PresentationReceipt {
    state: AtomicU8,
    command_completed: AtomicBool,
    drawable_presented: AtomicBool,
    presentation_error: AtomicBool,
    waker: Arc<WorkerWaker>,
}

impl PresentationReceipt {
    pub(super) fn offered(waker: Arc<WorkerWaker>) -> Self {
        Self {
            state: AtomicU8::new(RECEIPT_OFFERED),
            command_completed: AtomicBool::new(false),
            drawable_presented: AtomicBool::new(false),
            presentation_error: AtomicBool::new(false),
            waker,
        }
    }

    pub(super) fn try_accept(&self) -> bool {
        let accepted = self
            .state
            .compare_exchange(
                RECEIPT_OFFERED,
                RECEIPT_SUBMITTED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok();
        if accepted {
            self.waker.notify();
        }
        accepted
    }

    pub(super) fn try_supersede(&self) -> bool {
        let superseded = self
            .state
            .compare_exchange(
                RECEIPT_OFFERED,
                RECEIPT_SUPERSEDED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok();
        if superseded {
            self.waker.notify();
        }
        superseded
    }

    pub(super) fn try_mark_surface_less(&self) -> bool {
        let terminal = self
            .state
            .compare_exchange(
                RECEIPT_OFFERED,
                RECEIPT_SURFACE_LESS,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok();
        if terminal {
            self.waker.notify();
        }
        terminal
    }

    pub(super) fn mark_command_completed(&self, error: bool) {
        if error {
            self.presentation_error.store(true, Ordering::Release);
        }
        self.command_completed.store(true, Ordering::Release);
        self.waker.notify();
    }

    pub(super) fn mark_drawable_presented(&self) {
        self.drawable_presented.store(true, Ordering::Release);
        self.waker.notify();
    }

    pub(super) fn terminal_disposition(&self) -> Option<PresentationDisposition> {
        match self.state.load(Ordering::Acquire) {
            RECEIPT_SUPERSEDED => Some(PresentationDisposition::Superseded),
            RECEIPT_SURFACE_LESS => Some(PresentationDisposition::SurfaceLess),
            RECEIPT_SUBMITTED
                if self.command_completed.load(Ordering::Acquire)
                    && self.presentation_error.load(Ordering::Acquire) =>
            {
                Some(PresentationDisposition::Error)
            }
            RECEIPT_SUBMITTED
                if self.command_completed.load(Ordering::Acquire)
                    && self.drawable_presented.load(Ordering::Acquire) =>
            {
                Some(PresentationDisposition::Presentation)
            }
            RECEIPT_OFFERED | RECEIPT_SUBMITTED => None,
            state => panic!("invalid presentation receipt state {state}"),
        }
    }
}

pub(super) struct FrameOffer<T> {
    pub(super) generation: u64,
    pub(super) payload: T,
    pub(super) receipt: Arc<PresentationReceipt>,
}

pub(super) struct LatestFrameMailbox<T> {
    latest: Mutex<Option<FrameOffer<T>>>,
    next_generation: AtomicU64,
}

impl<T> Default for LatestFrameMailbox<T> {
    fn default() -> Self {
        Self {
            latest: Mutex::new(None),
            next_generation: AtomicU64::new(1),
        }
    }
}

impl<T> LatestFrameMailbox<T> {
    /// Publishes a new offer and returns an older, atomically superseded offer if one existed.
    pub(super) fn publish(
        &self,
        payload: T,
        receipt: Arc<PresentationReceipt>,
    ) -> (u64, Option<FrameOffer<T>>) {
        let generation = self.next_generation.fetch_add(1, Ordering::Relaxed);
        let offer = FrameOffer {
            generation,
            payload,
            receipt,
        };
        let mut latest = self
            .latest
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let replaced = latest.replace(offer).map(|old| {
            assert!(
                old.receipt.try_supersede(),
                "an offer retained in the latest-frame slot must still be offered"
            );
            old
        });
        (generation, replaced)
    }

    pub(super) fn generation(&self) -> Option<u64> {
        self.latest
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .map(|offer| offer.generation)
    }

    /// Accepts exactly the generation AppKit observed. A competing replacement wins by making
    /// the stale generation unavailable; a submitted frame can never subsequently be replaced.
    pub(super) fn try_accept(&self, generation: u64) -> Option<FrameOffer<T>> {
        let mut latest = self
            .latest
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if latest.as_ref().map(|offer| offer.generation) != Some(generation) {
            return None;
        }
        let offer = latest
            .take()
            .expect("matching latest-frame generation exists");
        assert!(
            offer.receipt.try_accept(),
            "an offer removed from the latest-frame slot must still be offered"
        );
        Some(offer)
    }

    pub(super) fn reject_latest(&self) -> Option<FrameOffer<T>> {
        let offer = self
            .latest
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()?;
        assert!(offer.receipt.try_supersede());
        Some(offer)
    }
}

pub(super) struct FramePool<T> {
    state: Arc<FramePoolState<T>>,
}

struct FramePoolState<T> {
    available: Mutex<Vec<T>>,
    allocated: AtomicUsize,
    in_flight: AtomicUsize,
    high_water: AtomicUsize,
    waker: Arc<WorkerWaker>,
}

impl<T> FramePool<T> {
    pub(super) fn new(waker: Arc<WorkerWaker>) -> Self {
        Self {
            state: Arc::new(FramePoolState {
                available: Mutex::new(Vec::new()),
                allocated: AtomicUsize::new(0),
                in_flight: AtomicUsize::new(0),
                high_water: AtomicUsize::new(0),
                waker,
            }),
        }
    }

    pub(super) fn acquire_with(&self, create: impl FnOnce() -> T) -> Option<FrameLease<T>> {
        let resource = self
            .state
            .available
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop()
            .or_else(|| {
                self.state
                    .allocated
                    .fetch_update(Ordering::AcqRel, Ordering::Acquire, |allocated| {
                        (allocated < FRAME_POOL_CAPACITY).then_some(allocated + 1)
                    })
                    .ok()
                    .map(|_| create())
            })?;
        let in_flight = self.state.in_flight.fetch_add(1, Ordering::AcqRel) + 1;
        self.state
            .high_water
            .fetch_max(in_flight, Ordering::Relaxed);
        Some(FrameLease {
            inner: Arc::new(FrameLeaseInner {
                resource: Mutex::new(Some(resource)),
                pool: Arc::clone(&self.state),
            }),
        })
    }

    pub(super) fn occupancy(&self) -> usize {
        self.state.in_flight.load(Ordering::Acquire)
    }

    pub(super) fn high_water(&self) -> usize {
        self.state.high_water.load(Ordering::Relaxed)
    }
}

pub(super) struct FrameLease<T> {
    inner: Arc<FrameLeaseInner<T>>,
}

impl<T> Clone for FrameLease<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T> FrameLease<T> {
    pub(super) fn with_resource<R>(&self, action: impl FnOnce(&T) -> R) -> R {
        let resource = self
            .inner
            .resource
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        action(resource.as_ref().expect("live frame lease has a resource"))
    }
}

struct FrameLeaseInner<T> {
    resource: Mutex<Option<T>>,
    pool: Arc<FramePoolState<T>>,
}

impl<T> Drop for FrameLeaseInner<T> {
    fn drop(&mut self) {
        let resource = self
            .resource
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            .expect("frame lease returns its resource exactly once");
        self.pool
            .available
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(resource);
        self.pool.in_flight.fetch_sub(1, Ordering::AcqRel);
        self.pool.waker.notify();
    }
}

#[derive(Default)]
struct WorkerWakeState {
    generation: Mutex<u64>,
    changed: Condvar,
}

#[derive(Default)]
pub(super) struct WorkerWaker {
    state: WorkerWakeState,
}

impl WorkerWaker {
    pub(super) fn generation(&self) -> u64 {
        *self
            .state
            .generation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(super) fn notify(&self) {
        let mut generation = self
            .state
            .generation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *generation = generation.wrapping_add(1);
        self.state.changed.notify_all();
    }

    pub(super) fn wait_after(&self, observed: u64, timeout: Duration) -> u64 {
        let generation = self
            .state
            .generation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if *generation != observed {
            return *generation;
        }
        let (generation, _) = self
            .state
            .changed
            .wait_timeout_while(generation, timeout, |generation| *generation == observed)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *generation
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct SplitTelemetrySnapshot {
    pub(super) frames_produced: u64,
    pub(super) frames_replaced: u64,
    pub(super) frames_accepted: u64,
    pub(super) frames_command_completed: u64,
    pub(super) frames_presented: u64,
    pub(super) frames_errored: u64,
}

#[derive(Default)]
pub(super) struct SplitTelemetry {
    frames_produced: AtomicU64,
    frames_replaced: AtomicU64,
    frames_accepted: AtomicU64,
    frames_command_completed: AtomicU64,
    frames_presented: AtomicU64,
    frames_errored: AtomicU64,
}

impl SplitTelemetry {
    pub(super) fn snapshot(&self) -> SplitTelemetrySnapshot {
        SplitTelemetrySnapshot {
            frames_produced: self.frames_produced.load(Ordering::Relaxed),
            frames_replaced: self.frames_replaced.load(Ordering::Relaxed),
            frames_accepted: self.frames_accepted.load(Ordering::Relaxed),
            frames_command_completed: self.frames_command_completed.load(Ordering::Relaxed),
            frames_presented: self.frames_presented.load(Ordering::Relaxed),
            frames_errored: self.frames_errored.load(Ordering::Relaxed),
        }
    }
}

#[derive(Default)]
struct AppResultState {
    title: Option<String>,
    fatal: Option<String>,
    redraw_needed: bool,
    overlay_dirty: bool,
    worker_stopped: bool,
}

#[derive(Debug, Default, Eq, PartialEq)]
pub(super) struct AppResults {
    pub(super) title: Option<String>,
    pub(super) fatal: Option<String>,
    pub(super) redraw_needed: bool,
    pub(super) overlay_dirty: bool,
    pub(super) worker_stopped: bool,
}

pub(super) struct WorkerBridge<T> {
    pub(super) controls: AppControlMailbox,
    pub(super) frames: LatestFrameMailbox<T>,
    pub(super) waker: Arc<WorkerWaker>,
    pub(super) telemetry: SplitTelemetry,
    results: Mutex<AppResultState>,
    wake_appkit: Arc<dyn Fn() + Send + Sync>,
}

impl<T> WorkerBridge<T> {
    pub(super) fn new(wake_appkit: impl Fn() + Send + Sync + 'static) -> Self {
        Self {
            controls: AppControlMailbox::default(),
            frames: LatestFrameMailbox::default(),
            waker: Arc::new(WorkerWaker::default()),
            telemetry: SplitTelemetry::default(),
            results: Mutex::new(AppResultState::default()),
            wake_appkit: Arc::new(wake_appkit),
        }
    }

    pub(super) fn send_control(
        &self,
        control: WorkerControl,
    ) -> Result<(), ControlMailboxOverflow> {
        if self.controls.send(control)? {
            self.waker.notify();
        }
        Ok(())
    }

    pub(super) fn publish_title(&self, title: String) {
        self.results
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .title = Some(title);
        (self.wake_appkit)();
    }

    pub(super) fn publish_redraw_needed(&self) {
        self.results
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .redraw_needed = true;
        (self.wake_appkit)();
    }

    pub(super) fn publish_overlay_dirty(&self) {
        self.results
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .overlay_dirty = true;
        (self.wake_appkit)();
    }

    pub(super) fn publish_fatal(&self, error: String) {
        let mut results = self
            .results
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Retain the first fault: a shutdown cascade must not overwrite its root cause.
        if results.fatal.is_none() {
            results.fatal = Some(error);
        }
        drop(results);
        (self.wake_appkit)();
    }

    pub(super) fn publish_stopped(&self) {
        self.results
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .worker_stopped = true;
        (self.wake_appkit)();
    }

    pub(super) fn take_results(&self) -> AppResults {
        let mut state = self
            .results
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        AppResults {
            title: state.title.take(),
            fatal: state.fatal.take(),
            redraw_needed: std::mem::take(&mut state.redraw_needed),
            overlay_dirty: std::mem::take(&mut state.overlay_dirty),
            worker_stopped: std::mem::take(&mut state.worker_stopped),
        }
    }
}

fn assert_send_sync<T: Send + Sync>() {}

#[allow(dead_code)]
fn assert_bridge_types_are_send_sync() {
    assert_send_sync::<PresentationReceipt>();
    assert_send_sync::<WorkerBridge<usize>>();
    assert_send_sync::<FrameOffer<usize>>();
    assert_send_sync::<FrameLease<usize>>();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::thread;

    #[test]
    fn control_mailbox_reserves_shutdown_capacity_and_reports_overflow() {
        let mailbox = AppControlMailbox::default();
        for _ in 0..APP_CONTROL_LIFECYCLE_CAPACITY {
            assert_eq!(mailbox.send(WorkerControl::SurfaceAvailable), Ok(true));
        }
        assert_eq!(
            mailbox.send(WorkerControl::SurfaceUnavailable),
            Err(ControlMailboxOverflow {
                attempted: WorkerControl::SurfaceUnavailable,
                depth: APP_CONTROL_LIFECYCLE_CAPACITY,
            })
        );
        assert_eq!(mailbox.send(WorkerControl::BeginShutdown), Ok(true));
        assert_eq!(mailbox.send(WorkerControl::BeginShutdown), Ok(false));
        assert_eq!(mailbox.send(WorkerControl::SurfaceAvailable), Ok(false));
        assert_eq!(mailbox.depth(), APP_CONTROL_CAPACITY);
        let drained = mailbox.drain();
        assert_eq!(drained.last(), Some(&WorkerControl::BeginShutdown));
        assert_eq!(mailbox.send(WorkerControl::BeginShutdown), Ok(false));
        assert_eq!(mailbox.send(WorkerControl::SurfaceUnavailable), Ok(false));
        assert_eq!(mailbox.depth(), 0);
    }

    #[test]
    fn shutdown_sent_first_is_the_only_control_ever_observed() {
        let mailbox = AppControlMailbox::default();
        assert_eq!(mailbox.send(WorkerControl::BeginShutdown), Ok(true));
        assert_eq!(mailbox.send(WorkerControl::SurfaceAvailable), Ok(false));
        assert_eq!(mailbox.send(WorkerControl::SurfaceUnavailable), Ok(false));
        assert_eq!(mailbox.drain(), vec![WorkerControl::BeginShutdown]);
        assert!(mailbox.drain().is_empty());
    }

    #[test]
    fn accepting_and_replacing_one_observed_offer_have_exactly_one_winner() {
        for _ in 0..128 {
            let waker = Arc::new(WorkerWaker::default());
            let mailbox = Arc::new(LatestFrameMailbox::default());
            let old_receipt = Arc::new(PresentationReceipt::offered(Arc::clone(&waker)));
            let (old_generation, replaced) = mailbox.publish(1_u32, Arc::clone(&old_receipt));
            assert!(replaced.is_none());
            let barrier = Arc::new(Barrier::new(3));

            let accepting_mailbox = Arc::clone(&mailbox);
            let accepting_barrier = Arc::clone(&barrier);
            let accepting = thread::spawn(move || {
                accepting_barrier.wait();
                accepting_mailbox.try_accept(old_generation).is_some()
            });

            let replacing_mailbox = Arc::clone(&mailbox);
            let replacing_waker = Arc::clone(&waker);
            let replacing_barrier = Arc::clone(&barrier);
            let replacing = thread::spawn(move || {
                replacing_barrier.wait();
                let receipt = Arc::new(PresentationReceipt::offered(replacing_waker));
                replacing_mailbox.publish(2_u32, receipt).1.is_some()
            });

            barrier.wait();
            let accepted = accepting.join().expect("accept thread");
            let replaced = replacing.join().expect("replace thread");
            assert_ne!(accepted, replaced);
            assert_eq!(
                old_receipt.terminal_disposition(),
                replaced.then_some(PresentationDisposition::Superseded)
            );
        }
    }

    #[test]
    fn receipt_waits_for_both_command_and_drawable_callbacks_in_either_order() {
        let waker = Arc::new(WorkerWaker::default());
        for command_first in [false, true] {
            let receipt = PresentationReceipt::offered(Arc::clone(&waker));
            assert!(receipt.try_accept());
            if command_first {
                receipt.mark_command_completed(false);
            } else {
                receipt.mark_drawable_presented();
            }
            assert_eq!(receipt.terminal_disposition(), None);
            if command_first {
                receipt.mark_drawable_presented();
            } else {
                receipt.mark_command_completed(false);
            }
            assert_eq!(
                receipt.terminal_disposition(),
                Some(PresentationDisposition::Presentation)
            );
        }
    }

    #[test]
    fn presentation_error_is_terminal_after_command_completion() {
        let receipt = PresentationReceipt::offered(Arc::new(WorkerWaker::default()));
        assert!(receipt.try_accept());
        // Even if a callback's error store becomes visible before its completion store, the
        // worker cannot observe an error disposition until command completion is visible.
        receipt.presentation_error.store(true, Ordering::Release);
        assert_eq!(receipt.terminal_disposition(), None);
        receipt.mark_command_completed(true);
        assert_eq!(
            receipt.terminal_disposition(),
            Some(PresentationDisposition::Error)
        );
    }

    #[test]
    fn frame_pool_caps_occupancy_and_wakes_when_a_lease_returns() {
        let waker = Arc::new(WorkerWaker::default());
        let pool = FramePool::new(Arc::clone(&waker));
        let leases = (0..FRAME_POOL_CAPACITY)
            .map(|value| pool.acquire_with(|| value).expect("pool capacity"))
            .collect::<Vec<_>>();
        assert!(pool.acquire_with(|| usize::MAX).is_none());
        assert_eq!(pool.occupancy(), FRAME_POOL_CAPACITY);
        assert_eq!(pool.high_water(), FRAME_POOL_CAPACITY);
        let observed = waker.generation();
        drop(leases.into_iter().next().expect("lease"));
        assert_ne!(waker.wait_after(observed, Duration::from_secs(1)), observed);
        let reused = pool.acquire_with(|| usize::MAX).expect("returned slot");
        reused.with_resource(|resource| assert!(*resource < FRAME_POOL_CAPACITY));
    }

    #[test]
    fn results_coalesce_and_first_fatal_is_non_droppable() {
        let wakes = Arc::new(AtomicUsize::new(0));
        let wake_count = Arc::clone(&wakes);
        let bridge = WorkerBridge::<usize>::new(move || {
            wake_count.fetch_add(1, Ordering::Relaxed);
        });
        bridge.publish_title("old".to_owned());
        bridge.publish_title("new".to_owned());
        bridge.publish_redraw_needed();
        bridge.publish_redraw_needed();
        bridge.publish_overlay_dirty();
        bridge.publish_fatal("root".to_owned());
        bridge.publish_fatal("cascade".to_owned());
        bridge.publish_stopped();
        assert_eq!(
            bridge.take_results(),
            AppResults {
                title: Some("new".to_owned()),
                fatal: Some("root".to_owned()),
                redraw_needed: true,
                overlay_dirty: true,
                worker_stopped: true,
            }
        );
        assert!(wakes.load(Ordering::Relaxed) >= 8);
        assert_eq!(bridge.take_results(), AppResults::default());
    }
}
