//! Padding policy.

use crate::config::PackerConfig;

/// Padding behavior for packed batches.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PaddingPolicy {
    max_seq_len: usize,
    pad_token_id: u32,
    ignore_index: i32,
}

impl PaddingPolicy {
    /// Creates padding policy from config.
    #[must_use]
    pub const fn from_config(config: &PackerConfig) -> Self {
        Self {
            max_seq_len: config.max_seq_len,
            pad_token_id: config.pad_token_id,
            ignore_index: config.ignore_index,
        }
    }

    /// Pads token ids and labels to the configured maximum length.
    pub fn pad(self, token_ids: &mut Vec<u32>, labels: &mut Vec<i32>) {
        if token_ids.len() >= self.max_seq_len {
            return;
        }

        let pad = self.max_seq_len - token_ids.len();
        token_ids.extend(std::iter::repeat_n(self.pad_token_id, pad));
        labels.extend(std::iter::repeat_n(self.ignore_index, pad));
    }
}
