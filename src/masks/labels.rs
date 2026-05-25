//! Label generation.

use crate::config::{LabelMode, PackerConfig};

/// Generates causal LM labels for input ids.
#[must_use]
pub fn causal_lm_labels(input_ids: &[u32], config: &PackerConfig) -> Vec<i32> {
    match config.label_mode {
        LabelMode::CausalLm => input_ids
            .iter()
            .map(|token| i32::try_from(*token).unwrap_or(config.ignore_index))
            .collect(),
        LabelMode::IgnoreAll => vec![config.ignore_index; input_ids.len()],
    }
}

/// Applies a loss mask to labels.
#[must_use]
pub fn labels_from_loss_mask(labels: &[i32], loss_mask: &[u8], ignore_index: i32) -> Vec<i32> {
    labels
        .iter()
        .zip(loss_mask.iter())
        .map(|(label, mask)| if *mask == 1 { *label } else { ignore_index })
        .collect()
}
