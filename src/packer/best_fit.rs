//! Best-fit sequence packing contract.

use tunex_core::types::{PackedBatch, TokenizedSample};

use crate::{
    config::PackerConfig,
    packer::{GreedyPacker, PackedOutput, TunexSequencePacker},
    PackerResult,
};

/// Best-fit packer placeholder.
///
/// v0.1 exposes the strategy as a stable type and delegates to the greedy
/// implementation until bin-packing optimization is introduced.
#[derive(Clone, Debug)]
pub struct BestFitPacker {
    inner: GreedyPacker,
}

impl BestFitPacker {
    /// Creates a best-fit packer.
    pub fn new(config: PackerConfig) -> PackerResult<Self> {
        Ok(Self {
            inner: GreedyPacker::new(config)?,
        })
    }
}

impl TunexSequencePacker for BestFitPacker {
    fn pack_samples(&self, samples: &[TokenizedSample]) -> PackerResult<PackedBatch> {
        self.inner.pack_samples(samples)
    }

    fn pack_output(&self, samples: &[TokenizedSample]) -> PackerResult<PackedOutput> {
        self.inner.pack_output(samples)
    }
}
