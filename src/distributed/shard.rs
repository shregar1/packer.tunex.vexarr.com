//! Distributed shard and rank identifiers.

/// Dataset/tokenizer shard identifier.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ShardId(pub u32);

/// Packer rank identifier.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PackerRank(pub u32);

/// Globally stable packed batch id.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct GlobalBatchId(pub u64);

impl GlobalBatchId {
    /// Builds a deterministic global id from shard, rank, epoch, and local batch id.
    #[must_use]
    pub const fn new(shard: ShardId, rank: PackerRank, epoch: u64, local_batch_id: u64) -> Self {
        let shard_part = (shard.0 as u64) << 48;
        let rank_part = (rank.0 as u64) << 32;
        let epoch_part = (epoch & 0xffff) << 16;
        Self(shard_part | rank_part | epoch_part | (local_batch_id & 0xffff))
    }
}
