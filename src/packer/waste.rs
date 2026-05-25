//! Packing waste metrics.

/// Packing efficiency and padding waste summary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PackingWaste {
    /// Number of packed sequences/bins.
    pub packed_sequences: usize,
    /// Original samples placed into bins.
    pub packed_samples: usize,
    /// Real tokens placed into bins.
    pub real_tokens: usize,
    /// Padding tokens needed to fill bins.
    pub padding_tokens: usize,
    /// Oversized samples dropped or skipped.
    pub oversized_samples: usize,
}

impl PackingWaste {
    /// Creates waste metrics.
    #[must_use]
    pub const fn new(
        packed_sequences: usize,
        packed_samples: usize,
        real_tokens: usize,
        max_seq_len: usize,
        oversized_samples: usize,
    ) -> Self {
        let capacity = packed_sequences * max_seq_len;
        Self {
            packed_sequences,
            packed_samples,
            real_tokens,
            padding_tokens: capacity.saturating_sub(real_tokens),
            oversized_samples,
        }
    }

    /// Returns packing efficiency in range 0.0..=1.0.
    #[must_use]
    pub fn efficiency(&self) -> f64 {
        let capacity = self.real_tokens + self.padding_tokens;
        if capacity == 0 {
            return 0.0;
        }
        self.real_tokens as f64 / capacity as f64
    }

    /// Returns padding waste ratio in range 0.0..=1.0.
    #[must_use]
    pub fn padding_waste_ratio(&self) -> f64 {
        1.0 - self.efficiency()
    }

    /// Returns average samples per packed sequence.
    #[must_use]
    pub fn samples_per_packed_sequence(&self) -> f64 {
        if self.packed_sequences == 0 {
            return 0.0;
        }
        self.packed_samples as f64 / self.packed_sequences as f64
    }
}
