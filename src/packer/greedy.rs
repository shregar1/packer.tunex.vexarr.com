//! Greedy sequence packing.

use async_trait::async_trait;
use tunex_core::{
    traits::SequencePacker,
    types::{PackedBatch, TokenizedSample},
    TunexResult,
};

use crate::{
    config::PackerConfig, packer::LabelMaskPolicy, packer::PaddingPolicy, PackerError, PackerResult,
};

/// Convenience trait for synchronous packer calls.
pub trait TunexSequencePacker: Send + Sync {
    /// Packs tokenized samples into a batch.
    fn pack_samples(&self, samples: &[TokenizedSample]) -> PackerResult<PackedBatch>;
}

/// Greedy packing plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackingPlan {
    /// Sequence lengths in input order.
    pub lengths: Vec<usize>,
    /// Total input token count.
    pub total_tokens: usize,
}

impl PackingPlan {
    /// Builds a plan from tokenized samples.
    #[must_use]
    pub fn from_samples(samples: &[TokenizedSample]) -> Self {
        let lengths = samples
            .iter()
            .map(|sample| sample.token_ids.len())
            .collect::<Vec<_>>();
        let total_tokens = lengths.iter().sum();
        Self {
            lengths,
            total_tokens,
        }
    }
}

/// Greedy sequence packer.
#[derive(Clone, Debug)]
pub struct GreedyPacker {
    config: PackerConfig,
}

impl GreedyPacker {
    /// Creates a greedy packer.
    pub fn new(config: PackerConfig) -> PackerResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Returns packer config.
    #[must_use]
    pub const fn config(&self) -> &PackerConfig {
        &self.config
    }
}

impl Default for GreedyPacker {
    fn default() -> Self {
        Self::new(PackerConfig::default()).expect("default packer config is valid")
    }
}

impl TunexSequencePacker for GreedyPacker {
    fn pack_samples(&self, samples: &[TokenizedSample]) -> PackerResult<PackedBatch> {
        let mut token_ids = Vec::new();
        let mut sequence_offsets = Vec::with_capacity(samples.len());
        let mut labels = Vec::new();

        for sample in samples {
            if sample.token_ids.len() > self.config.max_seq_len {
                return Err(PackerError::SequenceTooLong {
                    actual: sample.token_ids.len(),
                    max: self.config.max_seq_len,
                });
            }

            let offset = u32::try_from(token_ids.len()).map_err(|_| PackerError::IndexOverflow)?;
            sequence_offsets.push(offset);
            token_ids.extend_from_slice(&sample.token_ids);
            labels.extend(LabelMaskPolicy::from_config(&self.config).labels_for(sample));
        }

        PaddingPolicy::from_config(&self.config).pad(&mut token_ids, &mut labels);
        Ok(PackedBatch::new(token_ids, sequence_offsets, labels))
    }
}

#[async_trait]
impl SequencePacker for GreedyPacker {
    async fn pack(&self, samples: &[TokenizedSample]) -> TunexResult<PackedBatch> {
        self.pack_samples(samples)
            .map_err(tunex_core::TunexError::from)
    }
}
