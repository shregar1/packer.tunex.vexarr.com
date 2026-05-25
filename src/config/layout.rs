//! Packed memory layout configuration.

/// Layout config subset.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LayoutConfig {
    /// Maximum packed sequence length.
    pub max_seq_len: usize,
    /// Target packed batch size.
    pub batch_size: usize,
    /// Pad final lengths to this multiple.
    pub pad_to_multiple_of: Option<usize>,
    /// Padding token id.
    pub pad_token_id: u32,
}
