//! Loss mask generation.

/// Per-token loss mask. `1` contributes to loss, `0` is ignored.
pub type LossMask = Vec<u8>;

/// Builds a prompt-aware loss mask.
#[must_use]
pub fn prompt_loss_mask(len: usize, prompt_tokens: usize) -> LossMask {
    (0..len)
        .map(|index| u8::from(index >= prompt_tokens))
        .collect()
}

/// Builds an assistant-only loss mask from assistant token spans.
#[must_use]
pub fn assistant_only_loss_mask(len: usize, assistant_spans: &[(usize, usize)]) -> LossMask {
    let mut mask = vec![0; len];
    for (start, end) in assistant_spans {
        let end = (*end).min(len);
        for value in mask.iter_mut().take(end).skip(*start) {
            *value = 1;
        }
    }
    mask
}
