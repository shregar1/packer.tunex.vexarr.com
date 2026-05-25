//! Streaming adapter contracts for tokenizer-to-packer flow.

pub mod adapter;
pub mod backpressure;
pub mod input;
pub mod output;
pub mod stats;

pub use adapter::{AsyncPackingStream, PackerStreamAdapter};
pub use backpressure::{BackpressureAction, BackpressureHook, QueueCapacity};
pub use input::TokenizedBatchInput;
pub use output::PackedBatchEnvelope;
pub use stats::PackerStreamStats;
