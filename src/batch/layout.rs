//! Packed batch layout metadata.

/// Per-token attention mask.
pub type AttentionMask = Vec<u8>;

/// Per-token position ids.
pub type PositionIds = Vec<u32>;

/// Sequence boundary inside a packed token buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SequenceBoundary {
    /// Start offset.
    pub offset: u32,
    /// Sequence length.
    pub len: u32,
}

/// Extended packed batch layout metadata for downstream packer modules.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PackedBatchLayout {
    /// Attention mask aligned with packed token ids.
    pub attention_mask: AttentionMask,
    /// Position ids aligned with packed token ids.
    pub position_ids: PositionIds,
    /// Sequence boundaries aligned with packed sequence offsets.
    pub boundaries: Vec<SequenceBoundary>,
}
