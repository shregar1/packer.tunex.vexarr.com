//! Greedy sequence packing.

use async_trait::async_trait;
use tunex_core::{
    traits::SequencePacker,
    types::{PackedBatch, TokenizedSample},
    TunexResult,
};

use crate::{
    batch::{PackedBatchLayout, SequenceBoundary},
    config::{PackerConfig, PackingStrategy},
    packer::{LabelMaskPolicy, PackedOutput, PaddingPolicy, SkippedSample},
    PackerError, PackerResult,
};

/// Convenience trait for synchronous packer calls.
pub trait TunexSequencePacker: Send + Sync {
    /// Packs tokenized samples into a batch.
    fn pack_samples(&self, samples: &[TokenizedSample]) -> PackerResult<PackedBatch>;

    /// Packs tokenized samples into an extended output.
    fn pack_output(&self, samples: &[TokenizedSample]) -> PackerResult<PackedOutput>;
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
        self.pack_output(samples).map(|output| output.batch)
    }

    fn pack_output(&self, samples: &[TokenizedSample]) -> PackerResult<PackedOutput> {
        let mut token_ids = Vec::new();
        let mut sequence_offsets = Vec::with_capacity(samples.len());
        let mut labels = Vec::new();
        let mut attention_mask = Vec::new();
        let mut position_ids = Vec::new();
        let mut boundaries = Vec::with_capacity(samples.len());
        let mut skipped = Vec::new();

        for (sample_index, sample) in samples.iter().enumerate() {
            if sample.token_ids.len() > self.config.max_seq_len {
                let skipped_sample = SkippedSample {
                    sample_index,
                    actual_len: sample.token_ids.len(),
                    max_len: self.config.max_seq_len,
                };

                if self.config.skip_long_samples {
                    skipped.push(skipped_sample);
                    continue;
                }

                return Err(PackerError::SequenceTooLong {
                    actual: skipped_sample.actual_len,
                    max: skipped_sample.max_len,
                });
            }

            let offset = u32::try_from(token_ids.len()).map_err(|_| PackerError::IndexOverflow)?;
            sequence_offsets.push(offset);
            boundaries.push(SequenceBoundary {
                offset,
                len: u32::try_from(sample.token_ids.len())
                    .map_err(|_| PackerError::IndexOverflow)?,
            });
            token_ids.extend_from_slice(&sample.token_ids);
            labels.extend(LabelMaskPolicy::configured_labels(&self.config, sample));
            attention_mask.extend(std::iter::repeat_n(1, sample.token_ids.len()));
            let position_start = if self.config.reset_position_ids_per_sample {
                0
            } else {
                token_ids.len() - sample.token_ids.len()
            };
            let positions = crate::masks::position_ids(
                sample.token_ids.len(),
                position_start,
                if self.config.reset_position_ids_per_sample {
                    crate::masks::PositionIdPolicy::ResetPerSample
                } else {
                    crate::masks::PositionIdPolicy::Continuous
                },
            );
            position_ids.extend(positions);

            if self.config.strategy == PackingStrategy::PadOnly {
                break;
            }
        }

        if sequence_offsets.is_empty() {
            return Err(PackerError::EmptyPack);
        }

        PaddingPolicy::from_config(&self.config).pad_all(
            &mut token_ids,
            &mut labels,
            &mut attention_mask,
            &mut position_ids,
        );
        let packed_samples = boundaries.len();
        let batch = PackedBatch::new(token_ids, sequence_offsets, labels);
        let layout = PackedBatchLayout {
            attention_mask,
            position_ids,
            boundaries,
        };
        let real_tokens = layout
            .attention_mask
            .iter()
            .filter(|mask| **mask == 1)
            .count();
        let waste = crate::packer::PackingWaste::new(
            1,
            packed_samples,
            real_tokens,
            self.config.max_seq_len,
            skipped.len(),
        );
        Ok(PackedOutput::new(batch, layout, skipped).with_waste(waste))
    }
}

#[async_trait]
impl SequencePacker for GreedyPacker {
    async fn pack(&self, samples: &[TokenizedSample]) -> TunexResult<PackedBatch> {
        self.pack_samples(samples)
            .map_err(tunex_core::TunexError::from)
    }
}
