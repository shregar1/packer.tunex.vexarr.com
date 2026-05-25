//! Best-fit sequence packing.

use tunex_core::types::{PackedBatch, TokenizedSample};

use crate::{
    batch::{PackedBatchLayout, SequenceBoundary},
    config::PackerConfig,
    packer::{
        best_fit_bins, sort_by_length_desc, LabelMaskPolicy, PackedOutput, PackingWaste,
        SkippedSample, SortableSample, TunexSequencePacker,
    },
    PackerError, PackerResult,
};

/// Best-fit decreasing sequence packer.
#[derive(Clone, Debug)]
pub struct BestFitPacker {
    config: PackerConfig,
}

impl BestFitPacker {
    /// Creates a best-fit packer.
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

impl TunexSequencePacker for BestFitPacker {
    fn pack_samples(&self, samples: &[TokenizedSample]) -> PackerResult<PackedBatch> {
        self.pack_output(samples).map(|output| output.batch)
    }

    fn pack_output(&self, samples: &[TokenizedSample]) -> PackerResult<PackedOutput> {
        let mut skipped = Vec::new();
        let sortable = self.valid_samples(samples, &mut skipped)?;
        let bins = best_fit_bins(sortable, self.config.max_seq_len);

        if bins.is_empty() {
            return Err(PackerError::EmptyPack);
        }

        let mut token_ids = Vec::with_capacity(bins.len() * self.config.max_seq_len);
        let mut sequence_offsets = Vec::new();
        let mut labels = Vec::with_capacity(bins.len() * self.config.max_seq_len);
        let mut attention_mask = Vec::with_capacity(bins.len() * self.config.max_seq_len);
        let mut position_ids = Vec::with_capacity(bins.len() * self.config.max_seq_len);
        let mut boundaries = Vec::new();
        for bin in &bins {
            let bin_start = token_ids.len();
            for sample in &bin.samples {
                let offset =
                    u32::try_from(token_ids.len()).map_err(|_| PackerError::IndexOverflow)?;
                let len = u32::try_from(sample.len).map_err(|_| PackerError::IndexOverflow)?;
                sequence_offsets.push(offset);
                boundaries.push(SequenceBoundary { offset, len });
                token_ids.extend_from_slice(&sample.sample.token_ids);
                labels.extend(LabelMaskPolicy::configured_labels(
                    &self.config,
                    sample.sample,
                ));
                attention_mask.extend(std::iter::repeat_n(1, sample.len));
                let position_start = if self.config.reset_position_ids_per_sample {
                    0
                } else {
                    token_ids.len() - bin_start - sample.len
                };
                let positions = crate::masks::position_ids(
                    sample.len,
                    position_start,
                    if self.config.reset_position_ids_per_sample {
                        crate::masks::PositionIdPolicy::ResetPerSample
                    } else {
                        crate::masks::PositionIdPolicy::Continuous
                    },
                );
                position_ids.extend(positions);
            }

            let pad = self.config.max_seq_len - (token_ids.len() - bin_start);
            token_ids.extend(std::iter::repeat_n(self.config.pad_token_id, pad));
            labels.extend(std::iter::repeat_n(self.config.ignore_index, pad));
            attention_mask.extend(std::iter::repeat_n(0, pad));
            position_ids.extend(std::iter::repeat_n(0, pad));
        }

        let real_tokens = bins.iter().map(|bin| bin.used).sum();
        let packed_samples = bins.iter().map(|bin| bin.samples.len()).sum();
        let waste = PackingWaste::new(
            bins.len(),
            packed_samples,
            real_tokens,
            self.config.max_seq_len,
            skipped.len(),
        );
        let batch = PackedBatch::new(token_ids, sequence_offsets, labels);
        let layout = PackedBatchLayout {
            attention_mask,
            position_ids,
            boundaries,
        };
        Ok(PackedOutput::new(batch, layout, skipped).with_waste(waste))
    }
}

impl BestFitPacker {
    fn valid_samples<'a>(
        &self,
        samples: &'a [TokenizedSample],
        skipped: &mut Vec<SkippedSample>,
    ) -> PackerResult<Vec<SortableSample<'a>>> {
        let sorted = if self.config.deterministic {
            sort_by_length_desc(samples)
        } else {
            samples
                .iter()
                .enumerate()
                .map(|(index, sample)| SortableSample {
                    index,
                    sample,
                    len: sample.token_ids.len(),
                })
                .collect()
        };

        let mut valid = Vec::with_capacity(sorted.len());
        for sample in sorted {
            if sample.len > self.config.max_seq_len {
                let skipped_sample = SkippedSample {
                    sample_index: sample.index,
                    actual_len: sample.len,
                    max_len: self.config.max_seq_len,
                };
                if self.config.drop_oversized || self.config.skip_long_samples {
                    skipped.push(skipped_sample);
                    continue;
                }
                return Err(PackerError::SequenceTooLong {
                    actual: skipped_sample.actual_len,
                    max: skipped_sample.max_len,
                });
            }
            valid.push(sample);
        }
        Ok(valid)
    }
}
