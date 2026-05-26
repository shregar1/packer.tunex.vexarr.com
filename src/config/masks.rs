//! Mask and label configuration.

/// Label generation mode.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum LabelMode {
    /// Causal language modeling labels.
    CausalLm,
    /// Ignore every token in loss.
    IgnoreAll,
}

impl LabelMode {
    /// Parses a label mode name.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "causal_lm" | "causal-lm" => Some(Self::CausalLm),
            "ignore_all" | "ignore-all" => Some(Self::IgnoreAll),
            _ => None,
        }
    }
}

/// Mask and label config subset.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaskConfig {
    /// Label generation mode.
    pub label_mode: LabelMode,
    /// Label ignore index.
    pub ignore_index: i32,
    /// Whether only assistant tokens should contribute to loss.
    pub assistant_only_loss: bool,
    /// Whether role boundaries are available.
    pub role_boundaries: bool,
    /// Prompt tokens to ignore.
    pub prompt_tokens: usize,
    /// Whether labels should be masked at sample boundaries.
    pub mask_sample_boundaries: bool,
    /// Whether position ids reset per sample.
    pub reset_position_ids_per_sample: bool,
}
