//! Sequence packer configuration.

pub mod layout;
pub mod masks;
pub mod strategy;
pub mod validation;

pub use layout::LayoutConfig;
pub use masks::{LabelMode, MaskConfig};
pub use strategy::PackingStrategy;
pub use validation::{ConfigValidationError, ConfigValidationReport, ValidatePackerConfig};

/// Sequence packer configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackerConfig {
    /// Packing strategy.
    pub strategy: PackingStrategy,
    /// Maximum packed sequence length.
    pub max_seq_len: usize,
    /// Target packed batch size.
    pub batch_size: usize,
    /// Padding token id.
    pub pad_token_id: u32,
    /// Pad final lengths to a multiple.
    pub pad_to_multiple_of: Option<usize>,
    /// Whether oversized samples should be dropped instead of failing.
    pub drop_oversized: bool,
    /// Label generation mode.
    pub label_mode: LabelMode,
    /// Label ignore index.
    pub ignore_index: i32,
    /// Whether labels should mirror input ids.
    pub labels_from_inputs: bool,
    /// Whether samples longer than max length should be skipped.
    pub skip_long_samples: bool,
    /// Whether only assistant tokens should contribute to loss.
    pub assistant_only_loss: bool,
    /// Whether role boundaries are available for assistant-only masking.
    pub role_boundaries: bool,
    /// Prompt tokens to ignore at the start of each sample.
    pub prompt_tokens: usize,
    /// Whether labels should be masked across sample boundaries.
    pub mask_sample_boundaries: bool,
    /// Whether position ids reset per sample.
    pub reset_position_ids_per_sample: bool,
    /// Whether packing order must be deterministic.
    pub deterministic: bool,
}

impl PackerConfig {
    /// Returns layout-specific config.
    #[must_use]
    pub const fn layout(&self) -> LayoutConfig {
        LayoutConfig {
            max_seq_len: self.max_seq_len,
            batch_size: self.batch_size,
            pad_to_multiple_of: self.pad_to_multiple_of,
            pad_token_id: self.pad_token_id,
        }
    }

    /// Returns mask-specific config.
    #[must_use]
    pub const fn masks(&self) -> MaskConfig {
        MaskConfig {
            label_mode: self.label_mode,
            ignore_index: self.ignore_index,
            assistant_only_loss: self.assistant_only_loss,
            role_boundaries: self.role_boundaries,
            prompt_tokens: self.prompt_tokens,
            mask_sample_boundaries: self.mask_sample_boundaries,
            reset_position_ids_per_sample: self.reset_position_ids_per_sample,
        }
    }

    /// Validates config values.
    pub fn validate(&self) -> crate::PackerResult<()> {
        let report = self.validation_report();
        if report.is_valid() {
            return Ok(());
        }

        Err(crate::PackerError::InvalidConfig(report.to_string()))
    }
}

impl Default for PackerConfig {
    fn default() -> Self {
        Self {
            strategy: PackingStrategy::Greedy,
            max_seq_len: 4096,
            batch_size: 1,
            pad_token_id: 0,
            pad_to_multiple_of: None,
            drop_oversized: true,
            label_mode: LabelMode::CausalLm,
            ignore_index: -100,
            labels_from_inputs: true,
            skip_long_samples: true,
            assistant_only_loss: false,
            role_boundaries: false,
            prompt_tokens: 0,
            mask_sample_boundaries: false,
            reset_position_ids_per_sample: false,
            deterministic: true,
        }
    }
}
