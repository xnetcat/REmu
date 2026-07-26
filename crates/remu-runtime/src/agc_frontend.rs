// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Owned, resumable PM4 command-stream scheduling.
//!
//! This module deliberately does not read guest memory. Submission and indirect-buffer words
//! must be copied into an [`OwnedPm4Buffer`] before they are handed to the frontend. That makes a
//! suspended command stream independent of later guest writes and lets the caller release its
//! guest-memory lock as soon as submission has returned.

use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::sync::Arc;

const PM4_PACKET_TYPE_2: u32 = 2;
const PM4_PACKET_TYPE_3: u32 = 3;
const MAX_INDIRECT_BUFFER_NESTING: usize = 32;

/// A hardware queue as observed by the AGC frontend.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum AgcQueueId {
    Graphics,
    AsyncCompute(u32),
}

/// An immutable copy of a guest PM4 command buffer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OwnedPm4Buffer {
    guest_address: u64,
    words: Arc<[u32]>,
}

impl OwnedPm4Buffer {
    /// Copies the command words. The returned buffer never borrows guest memory.
    pub(crate) fn copy_from_slice(guest_address: u64, words: &[u32]) -> Self {
        Self {
            guest_address,
            words: Arc::from(words),
        }
    }

    pub(crate) fn guest_address(&self) -> u64 {
        self.guest_address
    }

    pub(crate) fn words(&self) -> &[u32] {
        &self.words
    }

    fn dword_address(&self, dword: usize) -> Result<u64, Pm4DecodeError> {
        let byte_offset = u64::try_from(dword)
            .ok()
            .and_then(|dword| dword.checked_mul(4))
            .ok_or(Pm4DecodeError::AddressOverflow {
                guest_address: self.guest_address,
                dword,
            })?;
        self.guest_address
            .checked_add(byte_offset)
            .ok_or(Pm4DecodeError::AddressOverflow {
                guest_address: self.guest_address,
                dword,
            })
    }
}

/// A decoded packet whose backing words remain owned for the lifetime of the packet.
#[derive(Clone, Debug)]
pub(crate) struct Pm4Packet {
    buffer: OwnedPm4Buffer,
    start_dword: usize,
    dword_count: usize,
    address: u64,
    opcode: u32,
    register: u32,
    predicated: bool,
}

impl Pm4Packet {
    pub(crate) fn address(&self) -> u64 {
        self.address
    }

    pub(crate) fn dword_count(&self) -> usize {
        self.dword_count
    }

    pub(crate) fn opcode(&self) -> u32 {
        self.opcode
    }

    pub(crate) fn register(&self) -> u32 {
        self.register
    }

    pub(crate) fn predicated(&self) -> bool {
        self.predicated
    }

    pub(crate) fn words(&self) -> &[u32] {
        &self.buffer.words[self.start_dword..self.start_dword + self.dword_count]
    }
}

/// An error detected while incrementally decoding a PM4 stream.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum Pm4DecodeError {
    UnsupportedPacketType {
        address: u64,
        packet_type: u32,
        header: u32,
    },
    InvalidPacketLength {
        address: u64,
        length: usize,
        remaining: usize,
    },
    AddressOverflow {
        guest_address: u64,
        dword: usize,
    },
    SkipPastEnd {
        address: u64,
        skip_dwords: usize,
        remaining: usize,
    },
    IndirectBufferNesting {
        maximum: usize,
    },
}

impl fmt::Display for Pm4DecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPacketType {
                address,
                packet_type,
                header,
            } => write!(
                formatter,
                "unsupported PM4 packet type {packet_type} at {address:#x} (header {header:#x})"
            ),
            Self::InvalidPacketLength {
                address,
                length,
                remaining,
            } => write!(
                formatter,
                "PM4 packet at {address:#x} contains {length} dwords with only {remaining} remaining"
            ),
            Self::AddressOverflow {
                guest_address,
                dword,
            } => write!(
                formatter,
                "PM4 address overflow from {guest_address:#x} at dword {dword}"
            ),
            Self::SkipPastEnd {
                address,
                skip_dwords,
                remaining,
            } => write!(
                formatter,
                "PM4 conditional skip at {address:#x} skips {skip_dwords} dwords with only {remaining} remaining"
            ),
            Self::IndirectBufferNesting { maximum } => {
                write!(
                    formatter,
                    "PM4 indirect-buffer nesting exceeds {maximum} levels"
                )
            }
        }
    }
}

impl std::error::Error for Pm4DecodeError {}

/// Opaque identity of a suspended wait.
///
/// A token is tied to a queue, submission and unique wait sequence. A stale acknowledgment can
/// therefore never release a later wait which happens to occupy the same queue.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct WaitToken {
    queue: AgcQueueId,
    submission_id: u64,
    sequence: u64,
}

impl WaitToken {
    pub(crate) fn queue(self) -> AgcQueueId {
        self.queue
    }

    pub(crate) fn submission_id(self) -> u64 {
        self.submission_id
    }
}

/// The action chosen by the PM4 packet executor.
pub(crate) enum PacketAction<W> {
    /// Commit the current packet and continue.
    Advance,
    /// Commit the current packet and skip this many following dwords in the same buffer.
    SkipDwords(usize),
    /// Commit the current packet and enter an already-owned indirect command buffer.
    EnterIndirectBuffer(OwnedPm4Buffer),
    /// Suspend on the current packet. Nothing after it is evaluated until acknowledgment.
    Block(W),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SubmissionBoundary {
    pub(crate) queue: AgcQueueId,
    pub(crate) submission_id: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct PollReport {
    pub(crate) processed_packets: usize,
    pub(crate) boundaries: Vec<SubmissionBoundary>,
    pub(crate) blocked: Option<WaitToken>,
}

#[derive(Debug)]
pub(crate) enum FrontendPollError<E> {
    Decode(Pm4DecodeError),
    Execute(E),
}

impl<E: fmt::Display> fmt::Display for FrontendPollError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode(error) => error.fmt(formatter),
            Self::Execute(error) => error.fmt(formatter),
        }
    }
}

impl<E> std::error::Error for FrontendPollError<E> where E: std::error::Error + 'static {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WaitAcknowledgeError {
    QueueIdle,
    NotBlocked,
    StaleToken,
}

impl fmt::Display for WaitAcknowledgeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QueueIdle => formatter.write_str("the AGC queue is idle"),
            Self::NotBlocked => formatter.write_str("the AGC queue is not blocked"),
            Self::StaleToken => formatter.write_str("the AGC wait token is stale"),
        }
    }
}

impl std::error::Error for WaitAcknowledgeError {}

#[derive(Clone, Debug)]
struct BufferCursor {
    buffer: OwnedPm4Buffer,
    next_dword: usize,
}

#[derive(Debug)]
struct BlockedWait<W> {
    token: WaitToken,
    condition: W,
    packet_dwords: usize,
}

#[derive(Debug)]
struct SubmissionCursor<W> {
    id: u64,
    stack: Vec<BufferCursor>,
    blocked: Option<BlockedWait<W>>,
}

impl<W> SubmissionCursor<W> {
    fn new(id: u64, buffer: OwnedPm4Buffer) -> Self {
        Self {
            id,
            stack: vec![BufferCursor {
                buffer,
                next_dword: 0,
            }],
            blocked: None,
        }
    }

    fn normalize(&mut self) {
        while self
            .stack
            .last()
            .is_some_and(|cursor| cursor.next_dword == cursor.buffer.words.len())
        {
            self.stack.pop();
        }
    }

    fn current_packet(&mut self) -> Result<Option<Pm4Packet>, Pm4DecodeError> {
        self.normalize();
        loop {
            let Some(cursor) = self.stack.last_mut() else {
                return Ok(None);
            };
            let header = cursor.buffer.words[cursor.next_dword];
            let address = cursor.buffer.dword_address(cursor.next_dword)?;
            match header >> 30 {
                PM4_PACKET_TYPE_2 => {
                    cursor.next_dword += 1;
                    self.normalize();
                }
                PM4_PACKET_TYPE_3 => {
                    let dword_count = usize::try_from(((header >> 16) & 0x3fff) + 2)
                        .expect("a 14-bit PM4 packet length fits usize");
                    let remaining = cursor.buffer.words.len() - cursor.next_dword;
                    if dword_count > remaining {
                        return Err(Pm4DecodeError::InvalidPacketLength {
                            address,
                            length: dword_count,
                            remaining,
                        });
                    }
                    return Ok(Some(Pm4Packet {
                        buffer: cursor.buffer.clone(),
                        start_dword: cursor.next_dword,
                        dword_count,
                        address,
                        opcode: (header >> 8) & 0xff,
                        register: (header >> 2) & 0x3f,
                        predicated: header & 1 != 0,
                    }));
                }
                packet_type => {
                    return Err(Pm4DecodeError::UnsupportedPacketType {
                        address,
                        packet_type,
                        header,
                    });
                }
            }
        }
    }

    fn advance_current(&mut self, packet_dwords: usize) {
        let cursor = self
            .stack
            .last_mut()
            .expect("a decoded PM4 packet has a live cursor");
        cursor.next_dword += packet_dwords;
        self.normalize();
    }

    fn apply_action(
        &mut self,
        action: PacketAction<W>,
        packet: &Pm4Packet,
        token: WaitToken,
    ) -> Result<(), Pm4DecodeError> {
        match action {
            PacketAction::Advance => self.advance_current(packet.dword_count),
            PacketAction::SkipDwords(skip_dwords) => {
                let cursor = self
                    .stack
                    .last()
                    .expect("a decoded PM4 packet has a live cursor");
                let remaining_after_packet = cursor
                    .buffer
                    .words
                    .len()
                    .saturating_sub(cursor.next_dword + packet.dword_count);
                if skip_dwords > remaining_after_packet {
                    return Err(Pm4DecodeError::SkipPastEnd {
                        address: packet.address,
                        skip_dwords,
                        remaining: remaining_after_packet,
                    });
                }
                self.advance_current(packet.dword_count + skip_dwords);
            }
            PacketAction::EnterIndirectBuffer(buffer) => {
                if self.stack.len() > MAX_INDIRECT_BUFFER_NESTING {
                    return Err(Pm4DecodeError::IndirectBufferNesting {
                        maximum: MAX_INDIRECT_BUFFER_NESTING,
                    });
                }
                self.advance_current(packet.dword_count);
                self.stack.push(BufferCursor {
                    buffer,
                    next_dword: 0,
                });
                self.normalize();
            }
            PacketAction::Block(condition) => {
                self.blocked = Some(BlockedWait {
                    token,
                    condition,
                    packet_dwords: packet.dword_count,
                });
            }
        }
        Ok(())
    }
}

/// Queue-local PM4 scheduler.
///
/// Only the front submission of a queue is visible to [`Self::poll_queue`]. Different queues are
/// independent; a blocked graphics queue does not prevent an async-compute queue from polling.
#[derive(Debug)]
pub(crate) struct AgcFrontend<W> {
    queues: BTreeMap<AgcQueueId, VecDeque<SubmissionCursor<W>>>,
    next_wait_sequence: u64,
}

impl<W> Default for AgcFrontend<W> {
    fn default() -> Self {
        Self {
            queues: BTreeMap::new(),
            next_wait_sequence: 1,
        }
    }
}

impl<W> AgcFrontend<W> {
    /// Copies and appends a root command buffer to a queue.
    ///
    /// Returning from this function guarantees that no root command word is borrowed from the
    /// caller. The executor must provide the same guarantee for buffers returned by
    /// [`PacketAction::EnterIndirectBuffer`].
    pub(crate) fn submit(
        &mut self,
        queue: AgcQueueId,
        submission_id: u64,
        guest_address: u64,
        words: &[u32],
    ) {
        self.submit_owned(
            queue,
            submission_id,
            OwnedPm4Buffer::copy_from_slice(guest_address, words),
        );
    }

    pub(crate) fn submit_owned(
        &mut self,
        queue: AgcQueueId,
        submission_id: u64,
        buffer: OwnedPm4Buffer,
    ) {
        self.queues
            .entry(queue)
            .or_default()
            .push_back(SubmissionCursor::new(submission_id, buffer));
    }

    pub(crate) fn queue_depth(&self, queue: AgcQueueId) -> usize {
        self.queues.get(&queue).map_or(0, VecDeque::len)
    }

    pub(crate) fn blocked_wait(&self, queue: AgcQueueId) -> Option<(WaitToken, &W)> {
        let blocked = self.queues.get(&queue)?.front()?.blocked.as_ref()?;
        Some((blocked.token, &blocked.condition))
    }

    /// Advances a queue only after the caller explicitly acknowledges its current logical wait.
    pub(crate) fn acknowledge_wait(
        &mut self,
        token: WaitToken,
    ) -> Result<(), WaitAcknowledgeError> {
        self.acknowledge_wait_if(token, |_| true).map(|_| ())
    }

    /// Rechecks a logical wait and advances only when `is_satisfied` proves it is ordered.
    pub(crate) fn acknowledge_wait_if(
        &mut self,
        token: WaitToken,
        is_satisfied: impl FnOnce(&W) -> bool,
    ) -> Result<bool, WaitAcknowledgeError> {
        let queue = self
            .queues
            .get_mut(&token.queue)
            .ok_or(WaitAcknowledgeError::QueueIdle)?;
        let submission = queue.front_mut().ok_or(WaitAcknowledgeError::QueueIdle)?;
        let blocked = submission
            .blocked
            .as_ref()
            .ok_or(WaitAcknowledgeError::NotBlocked)?;
        if blocked.token != token || submission.id != token.submission_id {
            return Err(WaitAcknowledgeError::StaleToken);
        }
        if !is_satisfied(&blocked.condition) {
            return Ok(false);
        }
        let blocked = submission
            .blocked
            .take()
            .expect("the proven AGC wait remains present");
        submission.advance_current(blocked.packet_dwords);
        Ok(true)
    }

    /// Polls one hardware queue for up to `packet_budget` executable packets.
    ///
    /// A blocked packet consumes budget only on the poll which first observes it. Later polls
    /// return the same token without invoking the executor. Submission boundaries are reported
    /// only after the root cursor and every nested indirect-buffer cursor have completed.
    pub(crate) fn poll_queue<E>(
        &mut self,
        queue_id: AgcQueueId,
        packet_budget: usize,
        mut execute: impl FnMut(AgcQueueId, u64, &Pm4Packet) -> Result<PacketAction<W>, E>,
    ) -> Result<PollReport, FrontendPollError<E>> {
        let mut report = PollReport::default();
        while report.processed_packets < packet_budget {
            let Some(queue) = self.queues.get_mut(&queue_id) else {
                break;
            };
            let Some(submission) = queue.front_mut() else {
                break;
            };
            if let Some(blocked) = &submission.blocked {
                report.blocked = Some(blocked.token);
                break;
            }
            let Some(packet) = submission
                .current_packet()
                .map_err(FrontendPollError::Decode)?
            else {
                let submission_id = submission.id;
                queue.pop_front();
                report.boundaries.push(SubmissionBoundary {
                    queue: queue_id,
                    submission_id,
                });
                continue;
            };
            let submission_id = submission.id;
            let action =
                execute(queue_id, submission_id, &packet).map_err(FrontendPollError::Execute)?;
            report.processed_packets += 1;
            let token = WaitToken {
                queue: queue_id,
                submission_id,
                sequence: self.next_wait_sequence,
            };
            self.next_wait_sequence = self.next_wait_sequence.wrapping_add(1).max(1);
            submission
                .apply_action(action, &packet, token)
                .map_err(FrontendPollError::Decode)?;
            if submission.blocked.is_some() {
                report.blocked = Some(token);
                break;
            }
        }

        // Completing the last budgeted packet is enough to prove the actual end of a submission.
        // Retire empty cursors now, but do not execute a packet from the next FIFO entry.
        if let Some(queue) = self.queues.get_mut(&queue_id)
            && let Some(submission) = queue.front_mut()
            && submission.blocked.is_none()
        {
            submission.normalize();
            if submission.stack.is_empty() {
                let submission_id = submission.id;
                queue.pop_front();
                report.boundaries.push(SubmissionBoundary {
                    queue: queue_id,
                    submission_id,
                });
            }
        }
        if self.queues.get(&queue_id).is_some_and(VecDeque::is_empty) {
            self.queues.remove(&queue_id);
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::{
        AgcFrontend, AgcQueueId, OwnedPm4Buffer, PacketAction, Pm4Packet, SubmissionBoundary,
    };

    const IT_WAIT_REG_MEM: u32 = 0x3c;
    const IT_INDIRECT_BUFFER: u32 = 0x3f;
    const IT_MARKER: u32 = 0x10;

    fn packet(opcode: u32, register: u32, payload: &[u32]) -> Vec<u32> {
        let dword_count = payload.len() + 1;
        assert!(dword_count >= 2);
        let mut words = Vec::with_capacity(dword_count);
        words.push(
            (3 << 30)
                | (u32::try_from(dword_count - 2).expect("test packet length fits u32") << 16)
                | (opcode << 8)
                | (register << 2),
        );
        words.extend_from_slice(payload);
        words
    }

    fn append(stream: &mut Vec<u32>, packet: Vec<u32>) {
        stream.extend(packet);
    }

    #[test]
    fn post_wait_packet_is_not_evaluated_until_a_proven_acknowledgment() {
        let memory = AtomicU32::new(3);
        let mut stream = packet(IT_WAIT_REG_MEM, 0, &[0]);
        append(&mut stream, packet(IT_MARKER, 1, &[0]));
        let mut frontend = AgcFrontend::<u32>::default();
        frontend.submit(AgcQueueId::Graphics, 7, 0x1000, &stream);

        let mut observed = Vec::new();
        let first = frontend
            .poll_queue(
                AgcQueueId::Graphics,
                usize::MAX,
                |_, _, packet| -> Result<_, Infallible> {
                    if packet.opcode() == IT_WAIT_REG_MEM {
                        Ok(PacketAction::Block(17))
                    } else {
                        observed.push(memory.load(Ordering::SeqCst));
                        Ok(PacketAction::Advance)
                    }
                },
            )
            .unwrap();
        let token = first.blocked.expect("the wait must suspend the queue");
        assert!(observed.is_empty(), "the post-wait read ran speculatively");
        assert!(first.boundaries.is_empty());

        memory.store(16, Ordering::SeqCst);
        assert!(
            !frontend
                .acknowledge_wait_if(token, |expected| {
                    memory.load(Ordering::SeqCst) == *expected
                })
                .unwrap()
        );
        let still_blocked = frontend
            .poll_queue(
                AgcQueueId::Graphics,
                usize::MAX,
                |_, _, _| -> Result<PacketAction<u32>, Infallible> {
                    panic!("a failed ordered proof must not call the executor")
                },
            )
            .unwrap();
        assert_eq!(still_blocked.blocked, Some(token));
        assert!(observed.is_empty());

        memory.store(17, Ordering::SeqCst);
        assert!(
            frontend
                .acknowledge_wait_if(token, |expected| {
                    memory.load(Ordering::SeqCst) == *expected
                })
                .unwrap()
        );
        let resumed = frontend
            .poll_queue(
                AgcQueueId::Graphics,
                usize::MAX,
                |_, _, packet| -> Result<_, Infallible> {
                    assert_eq!(packet.register(), 1);
                    observed.push(memory.load(Ordering::SeqCst));
                    Ok(PacketAction::Advance)
                },
            )
            .unwrap();
        assert_eq!(observed, vec![17]);
        assert_eq!(
            resumed.boundaries,
            vec![SubmissionBoundary {
                queue: AgcQueueId::Graphics,
                submission_id: 7,
            }]
        );
    }

    #[test]
    fn blocked_submission_keeps_same_queue_fifo_but_not_other_queues() {
        let mut first_graphics = packet(IT_MARKER, 1, &[0]);
        append(&mut first_graphics, packet(IT_WAIT_REG_MEM, 0, &[0]));
        append(&mut first_graphics, packet(IT_MARKER, 2, &[0]));
        let second_graphics = packet(IT_MARKER, 3, &[0]);
        let compute = packet(IT_MARKER, 4, &[0]);

        let mut frontend = AgcFrontend::<()>::default();
        frontend.submit(AgcQueueId::Graphics, 10, 0x1000, &first_graphics);
        frontend.submit(AgcQueueId::Graphics, 11, 0x2000, &second_graphics);
        frontend.submit(AgcQueueId::AsyncCompute(2), 12, 0x3000, &compute);
        let mut order = Vec::new();

        let graphics = frontend
            .poll_queue(
                AgcQueueId::Graphics,
                usize::MAX,
                |_, _, packet| -> Result<_, Infallible> {
                    if packet.opcode() == IT_WAIT_REG_MEM {
                        Ok(PacketAction::Block(()))
                    } else {
                        order.push(packet.register());
                        Ok(PacketAction::Advance)
                    }
                },
            )
            .unwrap();
        let wait = graphics.blocked.expect("graphics must block");
        assert_eq!(order, vec![1]);
        assert_eq!(frontend.queue_depth(AgcQueueId::Graphics), 2);

        let compute_report = frontend
            .poll_queue(
                AgcQueueId::AsyncCompute(2),
                usize::MAX,
                |_, _, packet| -> Result<_, Infallible> {
                    order.push(packet.register());
                    Ok(PacketAction::Advance)
                },
            )
            .unwrap();
        assert_eq!(order, vec![1, 4]);
        assert_eq!(compute_report.boundaries.len(), 1);

        frontend.acknowledge_wait(wait).unwrap();
        let resumed = frontend
            .poll_queue(
                AgcQueueId::Graphics,
                usize::MAX,
                |_, _, packet| -> Result<_, Infallible> {
                    order.push(packet.register());
                    Ok(PacketAction::Advance)
                },
            )
            .unwrap();
        assert_eq!(order, vec![1, 4, 2, 3]);
        assert_eq!(
            resumed.boundaries,
            vec![
                SubmissionBoundary {
                    queue: AgcQueueId::Graphics,
                    submission_id: 10,
                },
                SubmissionBoundary {
                    queue: AgcQueueId::Graphics,
                    submission_id: 11,
                },
            ]
        );
    }

    #[test]
    fn nested_indirect_cursor_resumes_before_its_parent() {
        let mut child_words = packet(IT_MARKER, 1, &[0]);
        append(&mut child_words, packet(IT_WAIT_REG_MEM, 0, &[0]));
        append(&mut child_words, packet(IT_MARKER, 2, &[0]));
        let child = OwnedPm4Buffer::copy_from_slice(0x4000, &child_words);

        let mut parent = packet(IT_INDIRECT_BUFFER, 0, &[0, 0, 0]);
        append(&mut parent, packet(IT_MARKER, 3, &[0]));
        let mut frontend = AgcFrontend::<()>::default();
        frontend.submit(AgcQueueId::Graphics, 20, 0x1000, &parent);
        let mut order = Vec::new();

        let child_for_executor = child.clone();
        let first = frontend
            .poll_queue(
                AgcQueueId::Graphics,
                usize::MAX,
                |_, _, packet| -> Result<_, Infallible> {
                    if packet.opcode() == IT_INDIRECT_BUFFER {
                        return Ok(PacketAction::EnterIndirectBuffer(
                            child_for_executor.clone(),
                        ));
                    }
                    if packet.opcode() == IT_WAIT_REG_MEM {
                        return Ok(PacketAction::Block(()));
                    }
                    order.push((packet.address(), packet.register()));
                    Ok(PacketAction::Advance)
                },
            )
            .unwrap();
        assert_eq!(order, vec![(0x4000, 1)]);
        assert!(first.boundaries.is_empty());

        frontend
            .acknowledge_wait(first.blocked.expect("child wait must block"))
            .unwrap();
        let resumed = frontend
            .poll_queue(
                AgcQueueId::Graphics,
                usize::MAX,
                |_, _, packet| -> Result<_, Infallible> {
                    order.push((packet.address(), packet.register()));
                    Ok(PacketAction::Advance)
                },
            )
            .unwrap();
        assert_eq!(order, vec![(0x4000, 1), (0x4010, 2), (0x1010, 3)]);
        assert_eq!(resumed.boundaries.len(), 1);
        assert_eq!(resumed.boundaries[0].submission_id, 20);
    }

    #[test]
    fn submission_and_indirect_buffers_own_their_words() {
        let mut root = packet(IT_MARKER, 1, &[0]);
        let mut frontend = AgcFrontend::<()>::default();
        frontend.submit(AgcQueueId::Graphics, 30, 0x1000, &root);
        root[0] = 0;

        let mut observed = Vec::new();
        frontend
            .poll_queue(
                AgcQueueId::Graphics,
                usize::MAX,
                |_, _, packet: &Pm4Packet| -> Result<_, Infallible> {
                    observed.push(packet.register());
                    Ok(PacketAction::Advance)
                },
            )
            .unwrap();
        assert_eq!(observed, vec![1]);
    }
}
