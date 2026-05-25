//! Packer metrics contracts.

use std::time::Duration;

/// Packer throughput and latency counters.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PackerMetrics {
    /// Input sequences packed.
    pub sequences: u64,
    /// Output tokens emitted.
    pub tokens: u64,
    /// Padding tokens emitted.
    pub padding_tokens: u64,
    /// Total pack latency.
    pub pack_latency: Duration,
}

impl PackerMetrics {
    /// Returns tokens per second.
    #[must_use]
    pub fn tokens_per_second(&self) -> f64 {
        let seconds = self.pack_latency.as_secs_f64();
        if seconds == 0.0 {
            return 0.0;
        }
        self.tokens as f64 / seconds
    }
}
