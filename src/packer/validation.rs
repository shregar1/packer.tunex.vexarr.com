//! Sequence packing validation.

use tunex_core::types::TokenizedSample;

use crate::{config::PackerConfig, PackerError, PackerResult};

/// Validates one sample length.
pub fn validate_sample_len(
    sample: &TokenizedSample,
    sample_index: usize,
    config: &PackerConfig,
) -> PackerResult<()> {
    if sample.token_ids.len() <= config.max_seq_len || config.skip_long_samples {
        return Ok(());
    }

    Err(PackerError::SequenceTooLong {
        actual: sample.token_ids.len(),
        max: config.max_seq_len,
    })
    .map_err(|error| match error {
        PackerError::SequenceTooLong { actual, max } => PackerError::InvalidConfig(format!(
            "sample {sample_index} length {actual} exceeds max_seq_len {max}"
        )),
        other => other,
    })
}
