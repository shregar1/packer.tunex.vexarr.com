//! Packing worker contract.

use std::time::Instant;

use tunex_core::types::TokenizedSample;

use crate::{
    packer::{PackedOutput, TunexSequencePacker},
    PackerResult,
};

use super::{scheduler::PackChunk, stats::WorkerPackingStats};

/// Output from one scheduled packing chunk.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackChunkOutput {
    /// Stable chunk id in schedule order.
    pub chunk_id: usize,
    /// Packed output for this chunk.
    pub output: PackedOutput,
    /// Worker-level metrics for this chunk.
    pub stats: WorkerPackingStats,
}

/// Stateless worker wrapper around a concrete packer implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackingWorker {
    worker_id: usize,
}

impl PackingWorker {
    /// Creates a worker with a stable id.
    #[must_use]
    pub const fn new(worker_id: usize) -> Self {
        Self { worker_id }
    }

    /// Returns the worker id.
    #[must_use]
    pub const fn worker_id(self) -> usize {
        self.worker_id
    }

    /// Packs one scheduled chunk and records basic metrics.
    pub fn pack_chunk<P>(
        self,
        packer: &P,
        samples: &[TokenizedSample],
        chunk: PackChunk,
    ) -> PackerResult<PackChunkOutput>
    where
        P: TunexSequencePacker + ?Sized,
    {
        let started = Instant::now();
        let output = packer.pack_output(&samples[chunk.start..chunk.end])?;
        let elapsed = started.elapsed();
        let stats = WorkerPackingStats {
            worker_id: self.worker_id,
            chunks: 1,
            samples: output.waste.packed_samples + output.waste.oversized_samples,
            tokens: output.waste.real_tokens,
            padding_tokens: output.waste.padding_tokens,
            packing_latency: elapsed,
            busy_time: elapsed,
        };

        Ok(PackChunkOutput {
            chunk_id: chunk.chunk_id,
            output,
            stats,
        })
    }
}
