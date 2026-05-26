//! Packing cache metrics.

/// Cache hit/miss and eviction counters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PackingCacheStats {
    /// Lookup attempts.
    pub lookups: u64,
    /// Successful lookups.
    pub hits: u64,
    /// Failed lookups.
    pub misses: u64,
    /// Store operations.
    pub stores: u64,
    /// Evicted entries.
    pub evictions: u64,
    /// Invalidated entries.
    pub invalidations: u64,
}

impl PackingCacheStats {
    /// Records a cache hit.
    pub fn record_hit(&mut self) {
        self.lookups = self.lookups.saturating_add(1);
        self.hits = self.hits.saturating_add(1);
    }

    /// Records a cache miss.
    pub fn record_miss(&mut self) {
        self.lookups = self.lookups.saturating_add(1);
        self.misses = self.misses.saturating_add(1);
    }

    /// Records a store.
    pub fn record_store(&mut self) {
        self.stores = self.stores.saturating_add(1);
    }

    /// Records an eviction.
    pub fn record_eviction(&mut self) {
        self.evictions = self.evictions.saturating_add(1);
    }

    /// Records an invalidation.
    pub fn record_invalidation(&mut self) {
        self.invalidations = self.invalidations.saturating_add(1);
    }

    /// Returns hit ratio in range 0.0..=1.0.
    #[must_use]
    pub fn hit_ratio(self) -> f64 {
        if self.lookups == 0 {
            return 0.0;
        }
        self.hits as f64 / self.lookups as f64
    }
}
