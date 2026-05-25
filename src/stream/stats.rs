//! Stream adapter metrics.

/// Packer stream counters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PackerStreamStats {
    /// Tokenized input batches consumed.
    pub input_batches: u64,
    /// Packed output batches emitted.
    pub output_batches: u64,
    /// Tokenized samples consumed.
    pub input_samples: u64,
    /// Tokens consumed.
    pub input_tokens: u64,
    /// Packed tokens emitted.
    pub packed_tokens: u64,
    /// Partial input batches observed.
    pub partial_batches: u64,
    /// Partial flushes emitted.
    pub partial_flushes: u64,
    /// Backpressure applications.
    pub backpressure_events: u64,
    /// Cancellation observations.
    pub cancellations: u64,
    /// Packing errors.
    pub errors: u64,
}

impl PackerStreamStats {
    /// Records one input batch.
    pub fn record_input(&mut self, samples: usize, tokens: usize, partial: bool) {
        self.input_batches = self.input_batches.saturating_add(1);
        self.input_samples = self.input_samples.saturating_add(samples as u64);
        self.input_tokens = self.input_tokens.saturating_add(tokens as u64);
        if partial {
            self.partial_batches = self.partial_batches.saturating_add(1);
        }
    }

    /// Records one output batch.
    pub fn record_output(&mut self, packed_tokens: usize) {
        self.output_batches = self.output_batches.saturating_add(1);
        self.packed_tokens = self.packed_tokens.saturating_add(packed_tokens as u64);
    }

    /// Records a partial flush.
    pub fn record_partial_flush(&mut self) {
        self.partial_flushes = self.partial_flushes.saturating_add(1);
    }

    /// Records backpressure.
    pub fn record_backpressure(&mut self) {
        self.backpressure_events = self.backpressure_events.saturating_add(1);
    }

    /// Records cancellation.
    pub fn record_cancellation(&mut self) {
        self.cancellations = self.cancellations.saturating_add(1);
    }

    /// Records an error.
    pub fn record_error(&mut self) {
        self.errors = self.errors.saturating_add(1);
    }
}
