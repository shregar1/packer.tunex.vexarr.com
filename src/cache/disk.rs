//! Disk-backed packing cache contract.

use std::path::{Path, PathBuf};

use crate::cache::PackingCachePolicy;

/// Disk cache descriptor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiskPackingCache {
    root: PathBuf,
    policy: PackingCachePolicy,
}

impl DiskPackingCache {
    /// Creates a disk cache descriptor.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>, policy: PackingCachePolicy) -> Self {
        Self {
            root: root.into(),
            policy,
        }
    }

    /// Returns cache root path.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns cache policy.
    #[must_use]
    pub const fn policy(&self) -> PackingCachePolicy {
        self.policy
    }

    /// Returns true when disk persistence is enabled by policy.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        matches!(self.policy, PackingCachePolicy::DiskBacked { .. })
    }
}
