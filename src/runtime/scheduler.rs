//! Deterministic chunk scheduling for parallel packing.

use crate::{PackerError, PackerResult};

/// Contiguous sample range assigned to one packing task.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackChunk {
    /// Stable chunk id in schedule order.
    pub chunk_id: usize,
    /// Inclusive sample start index.
    pub start: usize,
    /// Exclusive sample end index.
    pub end: usize,
}

impl PackChunk {
    /// Returns the number of samples in this chunk.
    #[must_use]
    pub const fn len(self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Returns true when this chunk has no samples.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.start >= self.end
    }
}

/// Planned packing work for one input batch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackSchedule {
    /// Stable batch id assigned by the caller/runtime.
    pub batch_id: u64,
    /// Contiguous chunks in deterministic order.
    pub chunks: Vec<PackChunk>,
    /// Maximum worker count for this schedule.
    pub workers: usize,
}

impl PackSchedule {
    /// Returns true when there is no work to execute.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
}

/// Schedules tokenized samples into deterministic contiguous chunks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackingScheduler {
    chunk_size: usize,
    workers: usize,
}

impl PackingScheduler {
    /// Creates a scheduler with bounded chunk size and worker count.
    pub fn new(chunk_size: usize, workers: usize) -> PackerResult<Self> {
        if chunk_size == 0 {
            return Err(PackerError::InvalidConfig(
                "packing runtime chunk_size must be greater than zero".to_owned(),
            ));
        }
        if workers == 0 {
            return Err(PackerError::InvalidConfig(
                "packing runtime workers must be greater than zero".to_owned(),
            ));
        }
        Ok(Self {
            chunk_size,
            workers,
        })
    }

    /// Builds a deterministic contiguous schedule for `sample_count` samples.
    #[must_use]
    pub fn plan(&self, batch_id: u64, sample_count: usize) -> PackSchedule {
        let chunks = (0..sample_count)
            .step_by(self.chunk_size)
            .enumerate()
            .map(|(chunk_id, start)| PackChunk {
                chunk_id,
                start,
                end: (start + self.chunk_size).min(sample_count),
            })
            .collect();

        PackSchedule {
            batch_id,
            chunks,
            workers: self.workers,
        }
    }

    /// Returns configured chunk size.
    #[must_use]
    pub const fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Returns configured worker count.
    #[must_use]
    pub const fn workers(&self) -> usize {
        self.workers
    }
}
