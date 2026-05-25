//! High-performance sequence packing engine for TuneX.
//!
//! The v0.1 surface defines cache-friendly packing contracts, initial greedy
//! and best-fit packer implementations, plugin metadata, and a stable prelude.

pub mod batch;
pub mod config;
pub mod error;
pub mod metrics;
pub mod packer;
pub mod plugin;
pub mod prelude;

pub use batch::{
    align_capacity, Alignment, AttentionMask, PackedBatch, PackedBatchBuilder, PackedBatchLayout,
    PackedBatchSlice, PackedBatchView, PositionIds, SequenceBoundary,
};
pub use config::{PackerConfig, PackingStrategy};
pub use error::{PackerError, PackerResult};
pub use metrics::PackerMetrics;
pub use packer::{
    BestFitPacker, GreedyPacker, LabelMaskPolicy, PackedOutput, PackingPlan, PaddingPolicy,
    SequencePackingStrategy, SkippedSample, TunexSequencePacker,
};
pub use plugin::{PackerPlugin, PACKER_PLUGIN_NAME, PACKER_PLUGIN_VERSION};
