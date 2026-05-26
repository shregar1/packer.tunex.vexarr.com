//! Distributed packed batch ownership.

use super::{GlobalBatchId, PackerRank, ShardId};

/// Ownership metadata for a packed batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchOwnership {
    /// Source data shard.
    pub shard_id: ShardId,
    /// Packer rank that owns packing for this batch.
    pub packer_rank: PackerRank,
    /// Trainer rank expected to consume this batch.
    pub trainer_rank: PackerRank,
    /// Global deterministic batch id.
    pub global_batch_id: GlobalBatchId,
}

/// Deterministic assignment for one local batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DistributedPackingAssignment {
    /// Training epoch.
    pub epoch: u64,
    /// Local batch id within the owning shard/rank.
    pub local_batch_id: u64,
    /// Ownership metadata.
    pub ownership: BatchOwnership,
}

impl DistributedPackingAssignment {
    /// Creates a deterministic assignment.
    #[must_use]
    pub const fn new(
        shard_id: ShardId,
        packer_rank: PackerRank,
        trainer_rank: PackerRank,
        epoch: u64,
        local_batch_id: u64,
    ) -> Self {
        let global_batch_id = GlobalBatchId::new(shard_id, packer_rank, epoch, local_batch_id);
        Self {
            epoch,
            local_batch_id,
            ownership: BatchOwnership {
                shard_id,
                packer_rank,
                trainer_rank,
                global_batch_id,
            },
        }
    }
}
