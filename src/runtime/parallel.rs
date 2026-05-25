//! Rayon-backed deterministic parallel packing runtime.

use std::time::{Duration, Instant};

use rayon::prelude::*;
use tunex_core::types::TokenizedSample;

use crate::{
    packer::{PackedOutput, TunexSequencePacker},
    PackerError, PackerResult,
};

use super::{
    scheduler::{PackSchedule, PackingScheduler},
    stats::ParallelPackingStats,
    worker::{PackChunkOutput, PackingWorker},
};

/// Ordered packed outputs plus aggregate runtime stats.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParallelPackingResult {
    /// Packed outputs in deterministic chunk order.
    pub outputs: Vec<PackedOutput>,
    /// Aggregate runtime metrics.
    pub stats: ParallelPackingStats,
}

impl ParallelPackingResult {
    /// Returns true when no output batches were produced.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.outputs.is_empty()
    }
}

/// Parallel runtime wrapper around a concrete sequence packer.
#[derive(Clone, Debug)]
pub struct ParallelPacker<P> {
    packer: P,
    scheduler: PackingScheduler,
}

impl<P> ParallelPacker<P>
where
    P: TunexSequencePacker + Send + Sync,
{
    /// Creates a parallel runtime from a concrete packer and scheduler.
    #[must_use]
    pub const fn new(packer: P, scheduler: PackingScheduler) -> Self {
        Self { packer, scheduler }
    }

    /// Returns the scheduler used by this runtime.
    #[must_use]
    pub const fn scheduler(&self) -> &PackingScheduler {
        &self.scheduler
    }

    /// Returns the inner packer.
    #[must_use]
    pub const fn packer(&self) -> &P {
        &self.packer
    }

    /// Packs samples in parallel and returns outputs ordered by schedule chunk id.
    pub fn pack_ordered(
        &self,
        batch_id: u64,
        samples: &[TokenizedSample],
    ) -> PackerResult<ParallelPackingResult> {
        let schedule = self.scheduler.plan(batch_id, samples.len());
        self.pack_schedule(samples, &schedule)
    }

    /// Executes a precomputed schedule.
    pub fn pack_schedule(
        &self,
        samples: &[TokenizedSample],
        schedule: &PackSchedule,
    ) -> PackerResult<ParallelPackingResult> {
        if schedule.is_empty() {
            return Ok(ParallelPackingResult {
                outputs: Vec::new(),
                stats: ParallelPackingStats::default(),
            });
        }

        let started = Instant::now();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(schedule.workers)
            .build()
            .map_err(|error| PackerError::Parallel(error.to_string()))?;

        let mut chunks = pool.install(|| {
            schedule
                .chunks
                .par_iter()
                .map(|chunk| {
                    let worker = PackingWorker::new(chunk.chunk_id % schedule.workers);
                    worker.pack_chunk(&self.packer, samples, *chunk)
                })
                .collect::<PackerResult<Vec<PackChunkOutput>>>()
        })?;

        chunks.par_sort_unstable_by_key(|chunk| chunk.chunk_id);

        let worker_stats = chunks
            .iter()
            .map(|chunk| chunk.stats.clone())
            .collect::<Vec<_>>();
        let outputs = chunks
            .into_iter()
            .map(|chunk| chunk.output)
            .collect::<Vec<_>>();
        let stats = ParallelPackingStats::from_workers(
            1,
            outputs.len(),
            started.elapsed(),
            Duration::ZERO,
            worker_stats,
        );

        Ok(ParallelPackingResult { outputs, stats })
    }
}
