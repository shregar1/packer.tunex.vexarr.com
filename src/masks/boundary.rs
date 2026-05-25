//! Sample boundary masking.

/// Per-token boundary mask. `0` can be used to ignore boundary tokens.
pub type BoundaryMask = Vec<u8>;

/// Builds a sample-boundary mask for a sequence.
#[must_use]
pub fn sample_boundary_mask(len: usize, mask_boundaries: bool) -> BoundaryMask {
    if len == 0 {
        return Vec::new();
    }
    let mut mask = vec![1; len];
    if mask_boundaries {
        mask[0] = 0;
    }
    mask
}
