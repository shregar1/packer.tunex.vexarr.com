//! Zero-copy views over packed batches.

use super::PackedBatch;

/// Borrowed packed batch view.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackedBatchView<'a> {
    pub(crate) batch: &'a PackedBatch,
}

impl<'a> PackedBatchView<'a> {
    /// Returns input ids.
    #[must_use]
    pub fn input_ids(&self) -> &'a [u32] {
        &self.batch.input_ids
    }

    /// Returns labels.
    #[must_use]
    pub fn labels(&self) -> &'a [i32] {
        &self.batch.labels
    }

    /// Returns attention mask.
    #[must_use]
    pub fn attention_mask(&self) -> &'a [u8] {
        &self.batch.attention_mask
    }

    /// Returns position ids.
    #[must_use]
    pub fn position_ids(&self) -> &'a [u32] {
        &self.batch.position_ids
    }

    /// Slices token-aligned buffers without cloning.
    #[must_use]
    pub fn slice(&self, start: usize, end: usize) -> Option<PackedBatchSlice<'a>> {
        if start > end || end > self.batch.input_ids.len() {
            return None;
        }

        Some(PackedBatchSlice {
            input_ids: &self.batch.input_ids[start..end],
            labels: &self.batch.labels[start..end],
            attention_mask: &self.batch.attention_mask[start..end],
            position_ids: &self.batch.position_ids[start..end],
        })
    }
}

/// Borrowed token-aligned slice.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackedBatchSlice<'a> {
    /// Input ids.
    pub input_ids: &'a [u32],
    /// Labels.
    pub labels: &'a [i32],
    /// Attention mask.
    pub attention_mask: &'a [u8],
    /// Position ids.
    pub position_ids: &'a [u32],
}
