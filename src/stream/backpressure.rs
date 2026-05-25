//! Stream backpressure contracts.

/// Bounded queue capacity snapshot.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct QueueCapacity {
    /// Current queue length.
    pub len: usize,
    /// Maximum queue capacity.
    pub capacity: usize,
}

impl QueueCapacity {
    /// Creates a capacity snapshot.
    #[must_use]
    pub const fn new(len: usize, capacity: usize) -> Self {
        Self { len, capacity }
    }

    /// Returns true when the queue is full.
    #[must_use]
    pub const fn is_full(self) -> bool {
        self.capacity > 0 && self.len >= self.capacity
    }

    /// Returns queue fill ratio.
    #[must_use]
    pub fn ratio(self) -> f64 {
        if self.capacity == 0 {
            return 0.0;
        }
        (self.len as f64 / self.capacity as f64).min(1.0)
    }
}

/// Backpressure decision for stream flow control.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackpressureAction {
    /// Continue processing immediately.
    Continue,
    /// Yield to the caller/runtime before continuing.
    Yield,
    /// Stop stream processing.
    Stop,
}

/// Hook called when bounded stream queues report pressure.
pub trait BackpressureHook: Send + Sync {
    /// Evaluates a queue capacity snapshot.
    fn on_capacity(&self, capacity: QueueCapacity) -> BackpressureAction;
}
