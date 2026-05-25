//! GPU-ready contiguous packed batch.

use tunex_core::types::PackedBatch as CorePackedBatch;

use super::PackedBatchView;

/// GPU transfer-ready packed batch.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PackedBatch {
    /// Contiguous token ids.
    pub input_ids: Vec<u32>,
    /// Contiguous labels aligned with input ids.
    pub labels: Vec<i32>,
    /// Attention mask aligned with input ids.
    pub attention_mask: Vec<u8>,
    /// Position ids aligned with input ids.
    pub position_ids: Vec<u32>,
    /// Packed sequence start offsets.
    pub sequence_offsets: Vec<u32>,
    /// Original sample start offsets.
    pub sample_offsets: Vec<u32>,
}

impl PackedBatch {
    /// Creates a packed batch from aligned buffers.
    #[must_use]
    pub const fn new(
        input_ids: Vec<u32>,
        labels: Vec<i32>,
        attention_mask: Vec<u8>,
        position_ids: Vec<u32>,
        sequence_offsets: Vec<u32>,
        sample_offsets: Vec<u32>,
    ) -> Self {
        Self {
            input_ids,
            labels,
            attention_mask,
            position_ids,
            sequence_offsets,
            sample_offsets,
        }
    }

    /// Builds a GPU-ready layout from the core TuneX packed batch.
    #[must_use]
    pub fn from_core(
        core: CorePackedBatch,
        attention_mask: Vec<u8>,
        position_ids: Vec<u32>,
        sample_offsets: Vec<u32>,
    ) -> Self {
        Self {
            input_ids: core.token_ids,
            labels: core.labels,
            attention_mask,
            position_ids,
            sequence_offsets: core.sequence_offsets,
            sample_offsets,
        }
    }

    /// Converts into the core TuneX packed batch contract.
    #[must_use]
    pub fn into_core(self) -> CorePackedBatch {
        CorePackedBatch::new(self.input_ids, self.sequence_offsets, self.labels)
    }

    /// Returns token count.
    #[must_use]
    pub fn token_count(&self) -> usize {
        self.input_ids.len()
    }

    /// Returns a zero-copy view.
    #[must_use]
    pub const fn view(&self) -> PackedBatchView<'_> {
        PackedBatchView { batch: self }
    }
}
