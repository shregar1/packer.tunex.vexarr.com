//! Rank-aware packing routing.

use super::{DistributedPackingAssignment, PackerRank, ShardId};

/// Route from tokenizer shard/rank to packer/trainer rank.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackingRoute {
    /// Source shard.
    pub shard_id: ShardId,
    /// Target packer rank.
    pub packer_rank: PackerRank,
    /// Target trainer rank.
    pub trainer_rank: PackerRank,
}

/// Deterministic route planner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackingRouter {
    ranks: u32,
}

impl PackingRouter {
    /// Creates a router for a positive number of ranks.
    #[must_use]
    pub const fn new(ranks: u32) -> Self {
        Self { ranks }
    }

    /// Returns configured rank count.
    #[must_use]
    pub const fn ranks(self) -> u32 {
        self.ranks
    }

    /// Returns the deterministic route for a shard.
    #[must_use]
    pub const fn route(self, shard_id: ShardId) -> PackingRoute {
        let rank_count = if self.ranks == 0 { 1 } else { self.ranks };
        let rank = PackerRank(shard_id.0 % rank_count);
        PackingRoute {
            shard_id,
            packer_rank: rank,
            trainer_rank: rank,
        }
    }

    /// Assigns one local batch using the deterministic route.
    #[must_use]
    pub const fn assign(
        self,
        shard_id: ShardId,
        epoch: u64,
        local_batch_id: u64,
    ) -> DistributedPackingAssignment {
        let route = self.route(shard_id);
        DistributedPackingAssignment::new(
            shard_id,
            route.packer_rank,
            route.trainer_rank,
            epoch,
            local_batch_id,
        )
    }
}
