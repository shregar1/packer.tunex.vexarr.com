//! Extended packed output.

use tunex_core::types::PackedBatch;

use crate::batch::{PackedBatchLayout, SequenceBoundary};
use crate::packer::PackingWaste;

/// Skipped or flagged sample metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SkippedSample {
    /// Input sample index.
    pub sample_index: usize,
    /// Actual sample token length.
    pub actual_len: usize,
    /// Maximum allowed length.
    pub max_len: usize,
}

/// Extended output from packer implementations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedOutput {
    /// Core packed batch contract.
    pub batch: PackedBatch,
    /// Extended layout data.
    pub layout: PackedBatchLayout,
    /// Skipped samples.
    pub skipped: Vec<SkippedSample>,
    /// Packing waste metrics.
    pub waste: PackingWaste,
}

impl PackedOutput {
    /// Creates extended output.
    #[must_use]
    pub const fn new(
        batch: PackedBatch,
        layout: PackedBatchLayout,
        skipped: Vec<SkippedSample>,
    ) -> Self {
        Self {
            batch,
            layout,
            skipped,
            waste: PackingWaste {
                packed_sequences: 1,
                packed_samples: 0,
                real_tokens: 0,
                padding_tokens: 0,
                oversized_samples: 0,
            },
        }
    }

    /// Attaches waste metrics.
    #[must_use]
    pub const fn with_waste(mut self, waste: PackingWaste) -> Self {
        self.waste = waste;
        self
    }

    /// Returns true when any samples were skipped.
    #[must_use]
    pub fn has_skipped_samples(&self) -> bool {
        !self.skipped.is_empty()
    }

    /// Returns sequence boundaries.
    #[must_use]
    pub fn boundaries(&self) -> &[SequenceBoundary] {
        &self.layout.boundaries
    }
}
