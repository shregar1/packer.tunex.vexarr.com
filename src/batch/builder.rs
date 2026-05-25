//! Reusable builder for GPU-ready packed batches.

use super::{align_capacity, Alignment, PackedBatch, SequenceBoundary};

/// Reusable packed batch builder.
#[derive(Clone, Debug)]
pub struct PackedBatchBuilder {
    alignment: Alignment,
    input_ids: Vec<u32>,
    labels: Vec<i32>,
    attention_mask: Vec<u8>,
    position_ids: Vec<u32>,
    sequence_offsets: Vec<u32>,
    sample_offsets: Vec<u32>,
}

impl PackedBatchBuilder {
    /// Creates a builder with aligned capacity.
    #[must_use]
    pub fn with_capacity(tokens: usize, sequences: usize, alignment: Alignment) -> Self {
        let u32_capacity = align_capacity(tokens, alignment, std::mem::size_of::<u32>());
        let i32_capacity = align_capacity(tokens, alignment, std::mem::size_of::<i32>());
        Self {
            alignment,
            input_ids: Vec::with_capacity(u32_capacity),
            labels: Vec::with_capacity(i32_capacity),
            attention_mask: Vec::with_capacity(align_capacity(tokens, alignment, 1)),
            position_ids: Vec::with_capacity(u32_capacity),
            sequence_offsets: Vec::with_capacity(sequences),
            sample_offsets: Vec::with_capacity(sequences),
        }
    }

    /// Clears buffers while retaining capacity.
    pub fn clear(&mut self) {
        self.input_ids.clear();
        self.labels.clear();
        self.attention_mask.clear();
        self.position_ids.clear();
        self.sequence_offsets.clear();
        self.sample_offsets.clear();
    }

    /// Pushes one packed sample.
    pub fn push_sample(
        &mut self,
        input_ids: &[u32],
        labels: &[i32],
        sample_offset: u32,
    ) -> crate::PackerResult<SequenceBoundary> {
        let offset =
            u32::try_from(self.input_ids.len()).map_err(|_| crate::PackerError::IndexOverflow)?;
        let len = u32::try_from(input_ids.len()).map_err(|_| crate::PackerError::IndexOverflow)?;
        self.sequence_offsets.push(offset);
        self.sample_offsets.push(sample_offset);
        self.input_ids.extend_from_slice(input_ids);
        self.labels.extend_from_slice(labels);
        self.attention_mask
            .extend(std::iter::repeat_n(1, input_ids.len()));
        self.position_ids.extend(
            (0..input_ids.len()).map(|position| u32::try_from(position).unwrap_or(u32::MAX)),
        );
        Ok(SequenceBoundary { offset, len })
    }

    /// Pads current buffers to a target length.
    pub fn pad_to(&mut self, target_len: usize, pad_token_id: u32, ignore_index: i32) {
        if self.input_ids.len() >= target_len {
            return;
        }
        let pad = target_len - self.input_ids.len();
        self.input_ids
            .extend(std::iter::repeat_n(pad_token_id, pad));
        self.labels.extend(std::iter::repeat_n(ignore_index, pad));
        self.attention_mask.extend(std::iter::repeat_n(0, pad));
        self.position_ids.extend(std::iter::repeat_n(0, pad));
    }

    /// Finishes the current batch and resets this builder for reuse.
    #[must_use]
    pub fn finish(&mut self) -> PackedBatch {
        let batch = PackedBatch::new(
            std::mem::take(&mut self.input_ids),
            std::mem::take(&mut self.labels),
            std::mem::take(&mut self.attention_mask),
            std::mem::take(&mut self.position_ids),
            std::mem::take(&mut self.sequence_offsets),
            std::mem::take(&mut self.sample_offsets),
        );
        *self = Self::with_capacity(
            batch.input_ids.capacity(),
            batch.sequence_offsets.capacity(),
            self.alignment,
        );
        batch
    }
}
