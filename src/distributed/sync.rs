//! Distributed packing synchronization contracts.

use super::{DistributedPackingState, GlobalBatchId, PackerRank, ShardId};

/// Replay checkpoint for deterministic distributed packing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackingReplayCheckpoint {
    /// State snapshot.
    pub state: DistributedPackingState,
    /// Global batch id to replay from.
    pub replay_from: Option<GlobalBatchId>,
}

/// Synchronization hook for distributed packer ranks.
pub trait DistributedPackingSync: Send + Sync {
    /// Called when a rank starts an epoch.
    fn on_epoch_start(&self, epoch: u64, shard_id: ShardId, rank: PackerRank);

    /// Called when a global batch id is emitted.
    fn on_batch_emitted(&self, global_batch_id: GlobalBatchId);

    /// Returns the latest replay checkpoint if available.
    fn replay_checkpoint(&self) -> Option<PackingReplayCheckpoint>;
}
