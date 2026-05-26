use tunex_packer::{DistributedPackingState, GlobalBatchId, PackerRank, PackingRouter, ShardId};

#[test]
fn router_assigns_shards_to_stable_ranks() {
    let router = PackingRouter::new(4);

    let route = router.route(ShardId(6));
    let assignment = router.assign(ShardId(6), 3, 9);

    assert_eq!(route.packer_rank, PackerRank(2));
    assert_eq!(route.trainer_rank, PackerRank(2));
    assert_eq!(assignment.ownership.shard_id, ShardId(6));
    assert_eq!(assignment.ownership.packer_rank, PackerRank(2));
    assert_eq!(
        assignment.ownership.global_batch_id,
        GlobalBatchId::new(ShardId(6), PackerRank(2), 3, 9)
    );
}

#[test]
fn global_batch_ids_are_deterministic_and_rank_safe() {
    let first = GlobalBatchId::new(ShardId(1), PackerRank(0), 2, 3);
    let same = GlobalBatchId::new(ShardId(1), PackerRank(0), 2, 3);
    let different_rank = GlobalBatchId::new(ShardId(1), PackerRank(1), 2, 3);

    assert_eq!(first, same);
    assert_ne!(first, different_rank);
}

#[test]
fn distributed_state_tracks_resume_cursor() {
    let mut state = DistributedPackingState {
        tokenizer_fingerprint: "tok-a".to_owned(),
        packer_fingerprint: "pack-a".to_owned(),
        ..DistributedPackingState::default()
    };
    let global = GlobalBatchId::new(ShardId(2), PackerRank(1), 4, 5);

    state.record_emitted(global);

    assert_eq!(state.cursor.last_global_batch_id, Some(global));
    assert_eq!(state.cursor.next_local_batch_id, 1);
    assert_eq!(state.tokenizer_fingerprint, "tok-a");
}
