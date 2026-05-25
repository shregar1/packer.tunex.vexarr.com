use std::time::Duration;

use tunex_core::types::TokenizedSample;
use tunex_packer::{
    GreedyPacker, PackerConfig, PackingQueue, PackingScheduler, ParallelPacker, QueuePressure,
};

#[test]
fn scheduler_builds_contiguous_chunks() {
    let scheduler = PackingScheduler::new(3, 2).expect("scheduler");
    let schedule = scheduler.plan(7, 8);

    assert_eq!(schedule.batch_id, 7);
    assert_eq!(schedule.workers, 2);
    assert_eq!(schedule.chunks.len(), 3);
    assert_eq!(schedule.chunks[0].start, 0);
    assert_eq!(schedule.chunks[0].end, 3);
    assert_eq!(schedule.chunks[1].start, 3);
    assert_eq!(schedule.chunks[1].end, 6);
    assert_eq!(schedule.chunks[2].start, 6);
    assert_eq!(schedule.chunks[2].end, 8);
}

#[test]
fn parallel_packer_preserves_chunk_order() {
    let packer = GreedyPacker::new(PackerConfig {
        max_seq_len: 6,
        pad_token_id: 0,
        ignore_index: -100,
        ..PackerConfig::default()
    })
    .expect("packer");
    let runtime = ParallelPacker::new(packer, PackingScheduler::new(2, 2).expect("scheduler"));
    let samples = vec![
        TokenizedSample::new(vec![1, 2]),
        TokenizedSample::new(vec![3]),
        TokenizedSample::new(vec![4, 5]),
        TokenizedSample::new(vec![6]),
    ];

    let result = runtime.pack_ordered(11, &samples).expect("packed");

    assert_eq!(result.outputs.len(), 2);
    assert_eq!(result.outputs[0].batch.token_ids, vec![1, 2, 3, 0, 0, 0]);
    assert_eq!(result.outputs[1].batch.token_ids, vec![4, 5, 6, 0, 0, 0]);
    assert_eq!(result.stats.input_batches, 1);
    assert_eq!(result.stats.output_batches, 2);
    assert_eq!(result.stats.samples, 4);
    assert_eq!(result.stats.tokens, 6);
    assert!(result.stats.tokens_per_second() >= 0.0);
}

#[test]
fn bounded_queue_tracks_backpressure() {
    let queue = PackingQueue::bounded(2);

    queue.try_send(1_u8).expect("first send");
    assert_eq!(
        queue.pressure(Duration::ZERO),
        QueuePressure {
            len: 1,
            capacity: 2,
            wait_time: Duration::ZERO,
        }
    );
    assert_eq!(queue.pressure(Duration::ZERO).ratio(), 0.5);

    queue.try_send(2_u8).expect("second send");
    assert!(queue.pressure(Duration::ZERO).is_full());
    assert!(queue.try_send(3_u8).is_err());
    assert_eq!(queue.recv().expect("first recv"), 1);
}

#[test]
fn parallel_stats_track_padding_waste_and_utilization() {
    let packer = GreedyPacker::new(PackerConfig {
        max_seq_len: 8,
        pad_token_id: 0,
        ..PackerConfig::default()
    })
    .expect("packer");
    let runtime = ParallelPacker::new(packer, PackingScheduler::new(2, 2).expect("scheduler"));
    let samples = vec![
        TokenizedSample::new(vec![1, 2, 3]),
        TokenizedSample::new(vec![4]),
        TokenizedSample::new(vec![5, 6]),
    ];

    let result = runtime.pack_ordered(1, &samples).expect("packed");

    assert_eq!(result.stats.tokens, 6);
    assert_eq!(result.stats.padding_tokens, 10);
    assert!(result.stats.padding_waste_ratio() > 0.0);
    assert!(result.stats.worker_utilization() >= 0.0);
}
