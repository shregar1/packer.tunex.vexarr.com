//! Tokenized stream input contracts.

use tunex_core::types::TokenizedSample;

/// Tokenized samples consumed from the tokenizer stage.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TokenizedBatchInput {
    /// Input batch id from the tokenizer pipeline.
    pub batch_id: u64,
    /// Tokenized samples in deterministic order.
    pub samples: Vec<TokenizedSample>,
    /// True when this is a final or otherwise partial batch.
    pub partial: bool,
}

impl TokenizedBatchInput {
    /// Creates a tokenized batch input.
    #[must_use]
    pub const fn new(batch_id: u64, samples: Vec<TokenizedSample>, partial: bool) -> Self {
        Self {
            batch_id,
            samples,
            partial,
        }
    }

    /// Returns sample count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Returns true when there are no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}
