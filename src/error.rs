//! Packer error model.

use thiserror::Error;
use tunex_core::{error::ErrorKind, TunexError};

/// Result alias for packer operations.
pub type PackerResult<T> = Result<T, PackerError>;

/// Errors produced by sequence packing.
#[derive(Debug, Error)]
pub enum PackerError {
    /// Configuration is invalid.
    #[error("invalid packer config: {0}")]
    InvalidConfig(String),
    /// A sequence exceeds the configured maximum length.
    #[error("sequence length {actual} exceeds max_seq_len {max}")]
    SequenceTooLong {
        /// Actual sequence length.
        actual: usize,
        /// Maximum configured length.
        max: usize,
    },
    /// Integer conversion would overflow.
    #[error("packing index overflow")]
    IndexOverflow,
}

impl From<PackerError> for TunexError {
    fn from(error: PackerError) -> Self {
        TunexError::new(ErrorKind::Packer, error.to_string()).recoverable()
    }
}
