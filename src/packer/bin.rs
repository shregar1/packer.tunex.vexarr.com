//! Best-fit packing bins.

use super::sort::SortableSample;

/// One packed sequence bin.
#[derive(Clone, Debug)]
pub struct PackingBin<'a> {
    /// Samples in placement order.
    pub samples: Vec<SortableSample<'a>>,
    /// Used tokens.
    pub used: usize,
    capacity: usize,
}

impl<'a> PackingBin<'a> {
    /// Creates an empty bin.
    #[must_use]
    pub const fn new(capacity: usize) -> Self {
        Self {
            samples: Vec::new(),
            used: 0,
            capacity,
        }
    }

    /// Returns remaining capacity.
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.capacity - self.used
    }

    /// Returns true when sample fits.
    #[must_use]
    pub const fn can_fit(&self, len: usize) -> bool {
        len <= self.remaining()
    }

    /// Pushes one sample into the bin.
    pub fn push(&mut self, sample: SortableSample<'a>) {
        self.used += sample.len;
        self.samples.push(sample);
    }
}

/// Places sorted samples using best-fit decreasing.
#[must_use]
pub fn best_fit_bins<'a>(
    samples: impl IntoIterator<Item = SortableSample<'a>>,
    capacity: usize,
) -> Vec<PackingBin<'a>> {
    let mut bins = Vec::<PackingBin<'a>>::new();
    for sample in samples {
        let mut best_index = None;
        let mut best_remaining = usize::MAX;
        for (index, bin) in bins.iter().enumerate() {
            if bin.can_fit(sample.len) {
                let remaining = bin.remaining() - sample.len;
                if remaining < best_remaining {
                    best_remaining = remaining;
                    best_index = Some(index);
                }
            }
        }

        if let Some(index) = best_index {
            bins[index].push(sample);
        } else {
            let mut bin = PackingBin::new(capacity);
            bin.push(sample);
            bins.push(bin);
        }
    }
    bins
}
