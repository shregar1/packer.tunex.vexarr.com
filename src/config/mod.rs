//! Sequence packer configuration.

/// Packing strategy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackingStrategy {
    /// Append sequences greedily until the pack is full.
    Greedy,
    /// Keep sample order and pad to max length without sequence interleaving.
    PadOnly,
    /// Place sequences into the best remaining slot.
    BestFit,
}

/// Sequence packer configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackerConfig {
    /// Maximum packed sequence length.
    pub max_seq_len: usize,
    /// Padding token id.
    pub pad_token_id: u32,
    /// Label ignore index.
    pub ignore_index: i32,
    /// Packing strategy.
    pub strategy: PackingStrategy,
    /// Whether labels should mirror input ids.
    pub labels_from_inputs: bool,
    /// Whether samples longer than max length should be skipped.
    pub skip_long_samples: bool,
}

impl PackerConfig {
    /// Validates config values.
    pub fn validate(&self) -> crate::PackerResult<()> {
        if self.max_seq_len == 0 {
            return Err(crate::PackerError::InvalidConfig(
                "max_seq_len must be greater than zero".to_owned(),
            ));
        }
        Ok(())
    }
}

impl Default for PackerConfig {
    fn default() -> Self {
        Self {
            max_seq_len: 4096,
            pad_token_id: 0,
            ignore_index: -100,
            strategy: PackingStrategy::Greedy,
            labels_from_inputs: true,
            skip_long_samples: true,
        }
    }
}
