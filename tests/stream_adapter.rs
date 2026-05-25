use std::sync::Arc;

use tunex_core::{cancellation::CancellationToken, types::TokenizedSample};
use tunex_packer::{
    BackpressureAction, BackpressureHook, GreedyPacker, PackerConfig, PackerError,
    PackerStreamAdapter, QueueCapacity, TokenizedBatchInput,
};

#[derive(Debug)]
struct StopOnFull;

impl BackpressureHook for StopOnFull {
    fn on_capacity(&self, capacity: QueueCapacity) -> BackpressureAction {
        if capacity.is_full() {
            BackpressureAction::Stop
        } else {
            BackpressureAction::Continue
        }
    }
}

#[test]
fn stream_adapter_packs_batches_with_ids_and_stats() {
    let packer = GreedyPacker::new(PackerConfig {
        max_seq_len: 6,
        ..PackerConfig::default()
    })
    .expect("packer");
    let mut adapter = PackerStreamAdapter::new(packer, 2);
    let input = TokenizedBatchInput::new(
        42,
        vec![
            TokenizedSample::new(vec![1, 2]),
            TokenizedSample::new(vec![3]),
        ],
        false,
    );

    let envelope = adapter.pack_batch(input).expect("packed");

    assert_eq!(envelope.input_batch_id, 42);
    assert_eq!(envelope.output_batch_id, 0);
    assert_eq!(envelope.output.batch.token_ids, vec![1, 2, 3, 0, 0, 0]);
    assert!(!envelope.partial);
    assert_eq!(envelope.stats.input_batches, 1);
    assert_eq!(envelope.stats.output_batches, 1);
    assert_eq!(envelope.stats.input_samples, 2);
    assert_eq!(envelope.stats.packed_tokens, 3);
}

#[test]
fn stream_adapter_marks_partial_flushes() {
    let packer = GreedyPacker::new(PackerConfig {
        max_seq_len: 4,
        ..PackerConfig::default()
    })
    .expect("packer");
    let mut adapter = PackerStreamAdapter::new(packer, 2);

    let envelope = adapter
        .flush_partial(7, vec![TokenizedSample::new(vec![9])])
        .expect("flush")
        .expect("partial output");

    assert_eq!(envelope.input_batch_id, 7);
    assert!(envelope.partial);
    assert_eq!(envelope.stats.partial_flushes, 1);
}

#[test]
fn stream_adapter_propagates_backpressure_stop() {
    let packer = GreedyPacker::new(PackerConfig::default()).expect("packer");
    let mut adapter = PackerStreamAdapter::new(packer, 1)
        .with_queue_capacity(QueueCapacity::new(1, 1))
        .with_backpressure_hook(Arc::new(StopOnFull));
    let input = TokenizedBatchInput::new(1, vec![TokenizedSample::new(vec![1])], false);

    let error = adapter.pack_batch(input).expect_err("backpressure stop");

    assert!(matches!(error, PackerError::Stream(_)));
    assert_eq!(adapter.stats().backpressure_events, 1);
}

#[test]
fn stream_adapter_observes_cancellation() {
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let packer = GreedyPacker::new(PackerConfig::default()).expect("packer");
    let mut adapter = PackerStreamAdapter::new(packer, 1).with_cancellation(cancellation);
    let input = TokenizedBatchInput::new(1, vec![TokenizedSample::new(vec![1])], false);

    let error = adapter.pack_batch(input).expect_err("cancelled");

    assert!(matches!(error, PackerError::Cancelled));
    assert_eq!(adapter.stats().cancellations, 1);
}
