//! Streaming adapter from tokenized batches to packed batches.

use std::sync::Arc;

use async_trait::async_trait;
use tunex_core::{
    cancellation::CancellationToken,
    events::{EventId, EventKind, EventPublisher, EventSeverity, EventSource, TunexEvent},
};

use crate::{
    packer::TunexSequencePacker,
    stream::{
        BackpressureAction, BackpressureHook, PackedBatchEnvelope, PackerStreamStats,
        QueueCapacity, TokenizedBatchInput,
    },
    PackerError, PackerResult,
};

/// Async-compatible packing stream interface.
#[async_trait]
pub trait AsyncPackingStream {
    /// Packs one tokenized input batch.
    ///
    /// # Errors
    ///
    /// Returns an error when packing, cancellation, or backpressure handling fails.
    async fn pack_next(&mut self, batch: TokenizedBatchInput) -> PackerResult<PackedBatchEnvelope>;
}

/// Tokenizer-to-packer stream adapter.
pub struct PackerStreamAdapter<P> {
    packer: P,
    output_batch_id: u64,
    expected_batch_size: usize,
    queue_capacity: Option<QueueCapacity>,
    backpressure_hook: Option<Arc<dyn BackpressureHook>>,
    event_publisher: Option<Arc<dyn EventPublisher>>,
    cancellation: Option<CancellationToken>,
    stats: PackerStreamStats,
}

impl<P> PackerStreamAdapter<P>
where
    P: TunexSequencePacker,
{
    /// Creates a stream adapter.
    #[must_use]
    pub fn new(packer: P, expected_batch_size: usize) -> Self {
        Self {
            packer,
            output_batch_id: 0,
            expected_batch_size,
            queue_capacity: None,
            backpressure_hook: None,
            event_publisher: None,
            cancellation: None,
            stats: PackerStreamStats::default(),
        }
    }

    /// Sets bounded output queue capacity metadata.
    #[must_use]
    pub const fn with_queue_capacity(mut self, capacity: QueueCapacity) -> Self {
        self.queue_capacity = Some(capacity);
        self
    }

    /// Sets a synchronous backpressure hook.
    #[must_use]
    pub fn with_backpressure_hook(mut self, hook: Arc<dyn BackpressureHook>) -> Self {
        self.backpressure_hook = Some(hook);
        self
    }

    /// Sets a core event publisher.
    #[must_use]
    pub fn with_event_publisher(mut self, publisher: Arc<dyn EventPublisher>) -> Self {
        self.event_publisher = Some(publisher);
        self
    }

    /// Sets cancellation token observed by this stream.
    #[must_use]
    pub fn with_cancellation(mut self, cancellation: CancellationToken) -> Self {
        self.cancellation = Some(cancellation);
        self
    }

    /// Returns stream stats.
    #[must_use]
    pub const fn stats(&self) -> PackerStreamStats {
        self.stats
    }

    /// Packs one tokenized input batch synchronously.
    ///
    /// # Errors
    ///
    /// Returns an error when packing, cancellation, or backpressure handling fails.
    pub fn pack_batch(&mut self, input: TokenizedBatchInput) -> PackerResult<PackedBatchEnvelope> {
        self.check_cancellation(input.batch_id)?;
        self.apply_backpressure()?;

        let partial = input.partial || input.len() < self.expected_batch_size;
        let token_count = input
            .samples
            .iter()
            .map(|sample| sample.token_ids.len())
            .sum();
        self.stats.record_input(input.len(), token_count, partial);

        if input.is_empty() {
            self.stats.record_error();
            self.emit_event(EventKind::StageFailed, input.batch_id, self.output_batch_id);
            return Err(PackerError::Stream(
                "cannot pack an empty tokenized input batch".to_owned(),
            ));
        }

        let output = match self.packer.pack_output(&input.samples) {
            Ok(output) => output,
            Err(error) => {
                self.stats.record_error();
                self.emit_event(EventKind::StageFailed, input.batch_id, self.output_batch_id);
                return Err(error);
            }
        };

        let packed_tokens = output.waste.real_tokens;
        self.stats.record_output(packed_tokens);
        if partial {
            self.stats.record_partial_flush();
        }

        let output_batch_id = self.output_batch_id;
        self.output_batch_id = self.output_batch_id.saturating_add(1);
        self.emit_event(EventKind::BatchProcessed, input.batch_id, output_batch_id);

        Ok(PackedBatchEnvelope {
            input_batch_id: input.batch_id,
            output_batch_id,
            output,
            partial,
            stats: self.stats,
        })
    }

    /// Packs a sequence of tokenized input batches synchronously.
    ///
    /// # Errors
    ///
    /// Returns an error when any batch cannot be packed.
    pub fn pack_batches<I>(&mut self, inputs: I) -> PackerResult<Vec<PackedBatchEnvelope>>
    where
        I: IntoIterator<Item = TokenizedBatchInput>,
    {
        inputs
            .into_iter()
            .map(|batch| self.pack_batch(batch))
            .collect()
    }

    /// Flushes a partial tokenized batch.
    ///
    /// # Errors
    ///
    /// Returns an error when the partial batch cannot be packed.
    pub fn flush_partial(
        &mut self,
        batch_id: u64,
        samples: Vec<tunex_core::types::TokenizedSample>,
    ) -> PackerResult<Option<PackedBatchEnvelope>> {
        if samples.is_empty() {
            return Ok(None);
        }

        self.pack_batch(TokenizedBatchInput::new(batch_id, samples, true))
            .map(Some)
    }

    fn apply_backpressure(&mut self) -> PackerResult<()> {
        let Some(capacity) = self.queue_capacity else {
            return Ok(());
        };

        if capacity.is_full() {
            self.stats.record_backpressure();
            self.emit_event(EventKind::BackpressureApplied, 0, self.output_batch_id);
        }

        let Some(hook) = &self.backpressure_hook else {
            return Ok(());
        };

        match hook.on_capacity(capacity) {
            BackpressureAction::Continue | BackpressureAction::Yield => Ok(()),
            BackpressureAction::Stop => Err(PackerError::Stream(
                "stream stopped by backpressure hook".to_owned(),
            )),
        }
    }

    fn check_cancellation(&mut self, input_batch_id: u64) -> PackerResult<()> {
        let Some(cancellation) = &self.cancellation else {
            return Ok(());
        };

        if !cancellation.is_cancelled() {
            return Ok(());
        }

        self.stats.record_cancellation();
        self.emit_event(EventKind::Cancelled, input_batch_id, self.output_batch_id);
        Err(PackerError::Cancelled)
    }

    fn emit_event(&self, kind: EventKind, input_batch_id: u64, output_batch_id: u64) {
        let Some(publisher) = &self.event_publisher else {
            return;
        };

        let event = TunexEvent::new(
            EventId::new(format!("packer-stream-{input_batch_id}-{output_batch_id}")),
            kind,
            EventSeverity::Info,
        )
        .with_source(
            EventSource::new()
                .with_module("tunex-packer")
                .with_stage("packer")
                .with_batch(input_batch_id.to_string()),
        )
        .with_metadata("input_batch_id", input_batch_id.to_string())
        .with_metadata("output_batch_id", output_batch_id.to_string());

        let _ = publisher.try_publish(event);
    }
}

#[async_trait]
impl<P> AsyncPackingStream for PackerStreamAdapter<P>
where
    P: TunexSequencePacker + Send,
{
    async fn pack_next(&mut self, batch: TokenizedBatchInput) -> PackerResult<PackedBatchEnvelope> {
        self.pack_batch(batch)
    }
}
