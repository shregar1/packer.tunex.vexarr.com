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

/// Label generation mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LabelMode {
    /// Causal language modeling labels.
    CausalLm,
    /// Ignore every token in loss.
    IgnoreAll,
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
    /// Whether packing order must be deterministic.
    pub deterministic: bool,
    /// Whether oversized samples should be dropped instead of failing.
    pub drop_oversized: bool,
    /// Label generation mode.
    pub label_mode: LabelMode,
    /// Whether only assistant tokens should contribute to loss.
    pub assistant_only_loss: bool,
    /// Prompt tokens to ignore at the start of each sample.
    pub prompt_tokens: usize,
    /// Whether labels should be masked across sample boundaries.
    pub mask_sample_boundaries: bool,
    /// Whether position ids reset per sample.
    pub reset_position_ids_per_sample: bool,
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
            deterministic: true,
            drop_oversized: true,
            label_mode: LabelMode::CausalLm,
            assistant_only_loss: false,
            prompt_tokens: 0,
            mask_sample_boundaries: false,
            reset_position_ids_per_sample: false,
        }
    }
}
