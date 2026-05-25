//! Packed stream output contracts.

use crate::{packer::PackedOutput, stream::PackerStreamStats};

/// Packed batch plus stream correlation metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedBatchEnvelope {
    /// Input tokenized batch id.
    pub input_batch_id: u64,
    /// Output packed batch id.
    pub output_batch_id: u64,
    /// Packed output in deterministic order.
    pub output: PackedOutput,
    /// True when emitted from a partial input/flush.
    pub partial: bool,
    /// Stream stats snapshot after this output.
    pub stats: PackerStreamStats,
}
