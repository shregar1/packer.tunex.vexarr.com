//! Position id generation.

/// Position id generation policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PositionIdPolicy {
    /// Positions reset for each sample.
    ResetPerSample,
    /// Positions continue across packed samples.
    Continuous,
}

/// Generates position ids for one sample.
#[must_use]
pub fn position_ids(len: usize, start: usize, policy: PositionIdPolicy) -> Vec<u32> {
    (0..len)
        .map(|position| match policy {
            PositionIdPolicy::ResetPerSample => position,
            PositionIdPolicy::Continuous => start + position,
        })
        .map(|position| u32::try_from(position).unwrap_or(u32::MAX))
        .collect()
}
