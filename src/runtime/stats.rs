//! Runtime metrics for parallel packing.

use std::time::Duration;

/// Per-worker packing counters and timing.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct WorkerPackingStats {
    /// Worker id assigned by the runtime.
    pub worker_id: usize,
    /// Chunks completed by this worker.
    pub chunks: usize,
    /// Samples consumed by this worker.
    pub samples: usize,
    /// Real tokens packed by this worker.
    pub tokens: usize,
    /// Padding tokens produced by this worker.
    pub padding_tokens: usize,
    /// End-to-end chunk packing latency.
    pub packing_latency: Duration,
    /// Time spent actively packing.
    pub busy_time: Duration,
}

impl WorkerPackingStats {
    /// Returns tokens packed per second for this worker.
    #[must_use]
    pub fn tokens_per_second(&self) -> f64 {
        if self.packing_latency.is_zero() {
            return 0.0;
        }
        self.tokens as f64 / self.packing_latency.as_secs_f64()
    }

    /// Returns busy time divided by total packing latency.
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.packing_latency.is_zero() {
            return 0.0;
        }
        (self.busy_time.as_secs_f64() / self.packing_latency.as_secs_f64()).min(1.0)
    }
}

/// Aggregate counters and timing for a parallel packing run.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ParallelPackingStats {
    /// Input batch count.
    pub input_batches: usize,
    /// Packed output batch count.
    pub output_batches: usize,
    /// Samples consumed.
    pub samples: usize,
    /// Real tokens packed.
    pub tokens: usize,
    /// Padding tokens produced.
    pub padding_tokens: usize,
    /// Total packing latency.
    pub packing_latency: Duration,
    /// Time spent waiting on queues/backpressure.
    pub queue_wait: Duration,
    /// Worker-level metrics.
    pub worker_stats: Vec<WorkerPackingStats>,
}

impl ParallelPackingStats {
    /// Creates aggregate stats from worker stats.
    #[must_use]
    pub fn from_workers(
        input_batches: usize,
        output_batches: usize,
        packing_latency: Duration,
        queue_wait: Duration,
        worker_stats: Vec<WorkerPackingStats>,
    ) -> Self {
        let samples = worker_stats.iter().map(|stats| stats.samples).sum();
        let tokens = worker_stats.iter().map(|stats| stats.tokens).sum();
        let padding_tokens = worker_stats.iter().map(|stats| stats.padding_tokens).sum();

        Self {
            input_batches,
            output_batches,
            samples,
            tokens,
            padding_tokens,
            packing_latency,
            queue_wait,
            worker_stats,
        }
    }

    /// Returns packed tokens per second.
    #[must_use]
    pub fn tokens_per_second(&self) -> f64 {
        if self.packing_latency.is_zero() {
            return 0.0;
        }
        self.tokens as f64 / self.packing_latency.as_secs_f64()
    }

    /// Returns padding waste ratio in range 0.0..=1.0.
    #[must_use]
    pub fn padding_waste_ratio(&self) -> f64 {
        let total = self.tokens + self.padding_tokens;
        if total == 0 {
            return 0.0;
        }
        self.padding_tokens as f64 / total as f64
    }

    /// Returns average worker utilization in range 0.0..=1.0.
    #[must_use]
    pub fn worker_utilization(&self) -> f64 {
        if self.worker_stats.is_empty() {
            return 0.0;
        }
        let total = self
            .worker_stats
            .iter()
            .map(WorkerPackingStats::utilization)
            .sum::<f64>();
        total / self.worker_stats.len() as f64
    }
}
