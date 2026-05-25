//! Packing strategy configuration.

/// Packing strategy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackingStrategy {
    /// Append sequences greedily until the pack is full.
    Greedy,
    /// Keep sample order and pad to max length without sequence interleaving.
    PadOnly,
    /// Place sequences into the best remaining slot.
    BestFit,
}

impl PackingStrategy {
    /// Parses a strategy name.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "greedy" => Some(Self::Greedy),
            "pad_only" | "pad-only" => Some(Self::PadOnly),
            "best_fit" | "best-fit" => Some(Self::BestFit),
            _ => None,
        }
    }
}
