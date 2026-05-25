//! Label masking policy.

use tunex_core::types::TokenizedSample;

use crate::config::PackerConfig;

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
}
