//! Distributed packing replay state.

use super::{GlobalBatchId, PackerRank, ShardId};

/// Resume cursor for distributed packing.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DistributedPackingCursor {
    /// Current epoch.
    pub epoch: u64,
    /// Source shard.
    pub shard_id: ShardId,
    /// Owning packer rank.
    pub rank: PackerRank,
    /// Next local batch id to produce.
    pub next_local_batch_id: u64,
    /// Last emitted global batch id.
    pub last_global_batch_id: Option<GlobalBatchId>,
}

/// Distributed packing state snapshot.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DistributedPackingState {
    /// Cursor for deterministic replay.
    pub cursor: DistributedPackingCursor,
    /// Deterministic tokenizer fingerprint.
    pub tokenizer_fingerprint: String,
    /// Deterministic packer config fingerprint.
    pub packer_fingerprint: String,
}

impl DistributedPackingState {
    /// Advances state after emitting a batch.
    pub fn record_emitted(&mut self, global_batch_id: GlobalBatchId) {
        self.cursor.last_global_batch_id = Some(global_batch_id);
        self.cursor.next_local_batch_id = self.cursor.next_local_batch_id.saturating_add(1);
    }
}
