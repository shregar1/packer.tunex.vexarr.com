//! Packed batch alignment helpers.

/// SIMD/GPU-friendly alignment descriptor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Alignment {
    /// Alignment in bytes.
    pub bytes: usize,
}

impl Alignment {
    /// Creates an alignment descriptor.
    #[must_use]
    pub const fn new(bytes: usize) -> Self {
        Self { bytes }
    }

    /// Common 64-byte cache-line alignment.
    #[must_use]
    pub const fn cache_line() -> Self {
        Self { bytes: 64 }
    }
}

/// Rounds capacity up to the requested element alignment.
#[must_use]
pub const fn align_capacity(capacity: usize, alignment: Alignment, element_size: usize) -> usize {
    if capacity == 0 || alignment.bytes == 0 || element_size == 0 {
        return capacity;
    }

    let bytes = capacity * element_size;
    let aligned = bytes.div_ceil(alignment.bytes) * alignment.bytes;
    aligned / element_size
}
