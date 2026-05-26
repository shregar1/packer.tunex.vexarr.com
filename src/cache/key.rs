//! Cache key construction for deterministic packing outputs.

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

use tunex_core::types::TokenizedSample;

use crate::config::{LabelMode, PackerConfig, PackingStrategy};

/// Cacheable packer artifact family.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CacheTarget {
    /// Complete packed batch output.
    PackedBatch,
    /// Attention mask buffer.
    AttentionMask,
    /// Position id buffer.
    PositionIds,
    /// Loss mask or labels.
    LossMask,
    /// Packing plan.
    PackingPlan,
    /// Sorted length groups for best-fit packing.
    SortedLengthGroups,
}

/// Stable fingerprint of one tokenized input batch.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct BatchFingerprint(pub u64);

impl BatchFingerprint {
    /// Builds a deterministic fingerprint from tokenized samples.
    #[must_use]
    pub fn from_samples(samples: &[TokenizedSample]) -> Self {
        let mut hasher = DefaultHasher::new();
        samples.len().hash(&mut hasher);
        for sample in samples {
            sample.id.hash(&mut hasher);
            sample.token_ids.hash(&mut hasher);
            sample.attention_mask.hash(&mut hasher);
        }
        Self(hasher.finish())
    }
}

/// Complete cache key for packer artifacts.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct PackingCacheKey {
    /// Target artifact.
    pub target: CacheTarget,
    /// Tokenizer fingerprint that produced the input tokens.
    pub tokenizer_fingerprint: String,
    /// Packing strategy.
    pub strategy: PackingStrategy,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Padding token id.
    pub pad_token_id: u32,
    /// Optional padding multiple.
    pub pad_to_multiple_of: Option<usize>,
    /// Loss generation mode.
    pub label_mode: LabelMode,
    /// Ignore index used by labels/loss masks.
    pub ignore_index: i32,
    /// Assistant-only loss flag.
    pub assistant_only_loss: bool,
    /// Input batch fingerprint.
    pub batch_fingerprint: BatchFingerprint,
}

impl PackingCacheKey {
    /// Creates a key from config and tokenized samples.
    #[must_use]
    pub fn from_samples(
        target: CacheTarget,
        tokenizer_fingerprint: impl Into<String>,
        config: &PackerConfig,
        samples: &[TokenizedSample],
    ) -> Self {
        Self {
            target,
            tokenizer_fingerprint: tokenizer_fingerprint.into(),
            strategy: config.strategy,
            max_seq_len: config.max_seq_len,
            pad_token_id: config.pad_token_id,
            pad_to_multiple_of: config.pad_to_multiple_of,
            label_mode: config.label_mode,
            ignore_index: config.ignore_index,
            assistant_only_loss: config.assistant_only_loss,
            batch_fingerprint: BatchFingerprint::from_samples(samples),
        }
    }
}
