//! Deterministic sample sorting for packing.

use tunex_core::types::TokenizedSample;

/// Sample reference with original index and length.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SortableSample<'a> {
    /// Original input index.
    pub index: usize,
    /// Tokenized sample.
    pub sample: &'a TokenizedSample,
    /// Sequence length.
    pub len: usize,
}

/// Sorts samples by descending length with index tie-break for determinism.
#[must_use]
pub fn sort_by_length_desc(samples: &[TokenizedSample]) -> Vec<SortableSample<'_>> {
    let mut sorted = samples
        .iter()
        .enumerate()
        .map(|(index, sample)| SortableSample {
            index,
            sample,
            len: sample.token_ids.len(),
        })
        .collect::<Vec<_>>();
    sorted.sort_by(|left, right| {
        right
            .len
            .cmp(&left.len)
            .then_with(|| left.index.cmp(&right.index))
    });
    sorted
}
