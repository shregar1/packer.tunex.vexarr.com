//! Packing cache policy contracts.

/// Cache policy for packer artifacts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackingCachePolicy {
    /// Cache disabled.
    Disabled,
    /// Memory-only least-recently-used cache.
    Lru {
        /// Maximum entries retained.
        max_entries: usize,
    },
    /// Memory-only cache without disk persistence.
    MemoryOnly {
        /// Maximum entries retained.
        max_entries: usize,
    },
    /// Disk-backed cache contract.
    DiskBacked {
        /// Maximum entries retained in memory.
        max_memory_entries: usize,
    },
    /// Future adaptive policy placeholder.
    Adaptive,
}

impl Default for PackingCachePolicy {
    fn default() -> Self {
        Self::Lru { max_entries: 1024 }
    }
}

impl PackingCachePolicy {
    /// Returns true when cache lookups/stores are enabled.
    #[must_use]
    pub const fn enabled(self) -> bool {
        !matches!(self, Self::Disabled)
    }

    /// Returns configured in-memory entry limit.
    #[must_use]
    pub const fn memory_limit(self) -> usize {
        match self {
            Self::Disabled => 0,
            Self::Lru { max_entries } | Self::MemoryOnly { max_entries } => max_entries,
            Self::DiskBacked { max_memory_entries } => max_memory_entries,
            Self::Adaptive => 1024,
        }
    }
}
