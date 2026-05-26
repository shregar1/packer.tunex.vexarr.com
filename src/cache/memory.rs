//! In-memory packing cache.

use std::collections::{HashMap, VecDeque};

use crate::{
    cache::{PackingCacheKey, PackingCachePolicy, PackingCacheStats},
    packer::{PackedOutput, PackingPlan, SortableSample},
};

/// Cache entry for packer artifacts.
#[derive(Clone, Debug)]
pub enum PackingCacheEntry {
    /// Complete packed output.
    PackedBatch(Box<PackedOutput>),
    /// Attention mask buffer.
    AttentionMask(Vec<u8>),
    /// Position id buffer.
    PositionIds(Vec<u32>),
    /// Loss mask or labels buffer.
    LossMask(Vec<i32>),
    /// Greedy packing plan.
    PackingPlan(PackingPlan),
    /// Sorted sample indexes and lengths.
    SortedLengthGroups(Vec<(usize, usize)>),
}

impl PackingCacheEntry {
    /// Builds a sorted length group entry from sorted samples.
    #[must_use]
    pub fn sorted_length_groups(samples: &[SortableSample<'_>]) -> Self {
        Self::SortedLengthGroups(
            samples
                .iter()
                .map(|sample| (sample.index, sample.len))
                .collect(),
        )
    }
}

/// Simple bounded in-memory cache with LRU-style eviction.
#[derive(Clone, Debug)]
pub struct MemoryPackingCache {
    entries: HashMap<PackingCacheKey, PackingCacheEntry>,
    order: VecDeque<PackingCacheKey>,
    policy: PackingCachePolicy,
    stats: PackingCacheStats,
}

impl MemoryPackingCache {
    /// Creates a cache with the provided policy.
    #[must_use]
    pub fn new(policy: PackingCachePolicy) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            policy,
            stats: PackingCacheStats::default(),
        }
    }

    /// Returns cache stats.
    #[must_use]
    pub const fn stats(&self) -> PackingCacheStats {
        self.stats
    }

    /// Returns current entry count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true when the cache has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Looks up an entry.
    pub fn get(&mut self, key: &PackingCacheKey) -> Option<PackingCacheEntry> {
        if !self.policy.enabled() {
            self.stats.record_miss();
            return None;
        }

        let Some(entry) = self.entries.get(key).cloned() else {
            self.stats.record_miss();
            return None;
        };

        self.touch(key);
        self.stats.record_hit();
        Some(entry)
    }

    /// Stores an entry.
    pub fn insert(&mut self, key: PackingCacheKey, entry: PackingCacheEntry) {
        if !self.policy.enabled() {
            return;
        }

        if !self.entries.contains_key(&key) {
            self.order.push_back(key.clone());
        }
        self.entries.insert(key, entry);
        self.stats.record_store();
        self.evict_if_needed();
    }

    /// Invalidates one entry.
    pub fn invalidate(&mut self, key: &PackingCacheKey) -> Option<PackingCacheEntry> {
        let removed = self.entries.remove(key);
        if removed.is_some() {
            self.order.retain(|candidate| candidate != key);
            self.stats.record_invalidation();
        }
        removed
    }

    fn touch(&mut self, key: &PackingCacheKey) {
        self.order.retain(|candidate| candidate != key);
        self.order.push_back(key.clone());
    }

    fn evict_if_needed(&mut self) {
        let limit = self.policy.memory_limit();
        if limit == 0 {
            self.entries.clear();
            self.order.clear();
            return;
        }

        while self.entries.len() > limit {
            let Some(key) = self.order.pop_front() else {
                break;
            };
            if self.entries.remove(&key).is_some() {
                self.stats.record_eviction();
            }
        }
    }
}
