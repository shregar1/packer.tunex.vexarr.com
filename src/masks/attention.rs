//! Packed sequence attention masks.

/// Flattened packed attention mask.
pub type PackedAttentionMask = Vec<u8>;

/// Generates a token-level packed attention mask.
#[must_use]
pub fn packed_attention_mask(real_tokens: usize, total_tokens: usize) -> PackedAttentionMask {
    (0..total_tokens)
        .map(|index| u8::from(index < real_tokens))
        .collect()
}
