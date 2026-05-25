//! Packing strategy selection.

use crate::config::PackingStrategy;

/// Runtime packing strategy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SequencePackingStrategy {
    /// Greedy concatenation.
    Greedy,
    /// Pad-only packing.
    PadOnly,
    /// Best-fit packing placeholder.
    BestFit,
}

impl From<PackingStrategy> for SequencePackingStrategy {
    fn from(strategy: PackingStrategy) -> Self {
        match strategy {
            PackingStrategy::Greedy => Self::Greedy,
            PackingStrategy::PadOnly => Self::PadOnly,
            PackingStrategy::BestFit => Self::BestFit,
        }
    }
}
