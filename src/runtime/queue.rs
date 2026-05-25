//! Bounded queue and backpressure contracts for packing.

use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, Receiver, RecvError, RecvTimeoutError, Sender, TrySendError};

/// Snapshot of bounded queue pressure.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct QueuePressure {
    /// Current queued item count.
    pub len: usize,
    /// Queue capacity.
    pub capacity: usize,
    /// Time spent waiting on queue operations.
    pub wait_time: Duration,
}

impl QueuePressure {
    /// Returns queue fill ratio in range 0.0..=1.0.
    #[must_use]
    pub fn ratio(self) -> f64 {
        if self.capacity == 0 {
            return 0.0;
        }
        (self.len as f64 / self.capacity as f64).min(1.0)
    }

    /// Returns true when the queue is at capacity.
    #[must_use]
    pub fn is_full(self) -> bool {
        self.len >= self.capacity
    }
}

/// Bounded queue wrapper used by packing runtimes.
#[derive(Clone, Debug)]
pub struct PackingQueue<T> {
    sender: Sender<T>,
    receiver: Receiver<T>,
    capacity: usize,
}

impl<T> PackingQueue<T> {
    /// Creates a bounded packing queue.
    #[must_use]
    pub fn bounded(capacity: usize) -> Self {
        let (sender, receiver) = bounded(capacity);
        Self {
            sender,
            receiver,
            capacity,
        }
    }

    /// Attempts to send without blocking.
    pub fn try_send(&self, item: T) -> Result<(), TrySendError<T>> {
        self.sender.try_send(item)
    }

    /// Sends with timeout and returns queue wait duration.
    pub fn send_timeout(&self, item: T, timeout: Duration) -> Result<Duration, TrySendError<T>> {
        let started = Instant::now();
        match self.sender.send_timeout(item, timeout) {
            Ok(()) => Ok(started.elapsed()),
            Err(error) => Err(match error {
                crossbeam_channel::SendTimeoutError::Timeout(item) => TrySendError::Full(item),
                crossbeam_channel::SendTimeoutError::Disconnected(item) => {
                    TrySendError::Disconnected(item)
                }
            }),
        }
    }

    /// Receives one item, blocking until an item is available or all senders disconnect.
    pub fn recv(&self) -> Result<T, RecvError> {
        self.receiver.recv()
    }

    /// Receives one item with timeout.
    pub fn recv_timeout(&self, timeout: Duration) -> Result<T, RecvTimeoutError> {
        self.receiver.recv_timeout(timeout)
    }

    /// Returns current queue length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.receiver.len()
    }

    /// Returns true when the queue has no pending items.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.receiver.is_empty()
    }

    /// Returns configured capacity.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns current pressure snapshot.
    #[must_use]
    pub fn pressure(&self, wait_time: Duration) -> QueuePressure {
        QueuePressure {
            len: self.len(),
            capacity: self.capacity,
            wait_time,
        }
    }
}
