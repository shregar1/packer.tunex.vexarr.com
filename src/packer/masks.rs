//! Label masking policy.

use tunex_core::types::TokenizedSample;

use crate::{
    config::PackerConfig,
    masks::{
        assistant_only_loss_mask, causal_lm_labels, labels_from_loss_mask, prompt_loss_mask,
        sample_boundary_mask,
    },
};

/// Label mask behavior.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LabelMaskPolicy {
    labels_from_inputs: bool,
    ignore_index: i32,
}

impl LabelMaskPolicy {
    /// Creates label mask policy from config.
    #[must_use]
    pub const fn from_config(config: &PackerConfig) -> Self {
        Self {
            labels_from_inputs: config.labels_from_inputs,
            ignore_index: config.ignore_index,
        }
    }

    /// Builds labels for one tokenized sample.
    #[must_use]
    pub fn labels_for(self, sample: &TokenizedSample) -> Vec<i32> {
        if self.labels_from_inputs {
            return sample
                .token_ids
                .iter()
                .map(|token| i32::try_from(*token).unwrap_or(self.ignore_index))
                .collect();
        }

        vec![self.ignore_index; sample.token_ids.len()]
    }

    /// Builds labels using the full packer mask configuration.
    #[must_use]
    pub fn configured_labels(config: &PackerConfig, sample: &TokenizedSample) -> Vec<i32> {
        let base = causal_lm_labels(&sample.token_ids, config);
        let mut mask = if config.assistant_only_loss {
            assistant_only_loss_mask(
                sample.token_ids.len(),
                &[(config.prompt_tokens, sample.token_ids.len())],
            )
        } else {
            prompt_loss_mask(sample.token_ids.len(), config.prompt_tokens)
        };
        let boundary_mask =
            sample_boundary_mask(sample.token_ids.len(), config.mask_sample_boundaries);
        for (mask_value, boundary_value) in mask.iter_mut().zip(boundary_mask.iter()) {
            *mask_value &= *boundary_value;
        }
        labels_from_loss_mask(&base, &mask, config.ignore_index)
    }
}
