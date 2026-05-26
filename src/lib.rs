//! High-performance sequence packing engine for TuneX.
//!
//! The v0.1 surface defines cache-friendly packing contracts, initial greedy
//! and best-fit packer implementations, plugin metadata, and a stable prelude.

pub mod batch;
pub mod cache;
pub mod capabilities;
pub mod config;
pub mod contract;
pub mod error;
pub mod masks;
pub mod metrics;
pub mod packer;
pub mod plugin;
pub mod prelude;
pub mod runtime;
pub mod stream;

pub use batch::{
    align_capacity, Alignment, AttentionMask, PackedBatch, PackedBatchBuilder, PackedBatchLayout,
    PackedBatchSlice, PackedBatchView, PositionIds, SequenceBoundary,
};
pub use cache::{
    BatchFingerprint, CacheTarget, DiskPackingCache, MemoryPackingCache, PackingCacheEntry,
    PackingCacheKey, PackingCachePolicy, PackingCacheStats,
};
pub use capabilities::{
    packer_capabilities, ASSISTANT_ONLY_LOSS, BEST_FIT_PACKING, DETERMINISTIC_PACKING,
    GPU_READY_LAYOUT, GREEDY_PACKING, LOSS_MASK_GENERATION, PACKER_CAPABILITY_NAMES,
    PARALLEL_PACKING, SEQUENCE_PACKING,
};
pub use config::{
    ConfigValidationError, ConfigValidationReport, LabelMode, LayoutConfig, MaskConfig,
    PackerConfig, PackingStrategy, ValidatePackerConfig,
};
pub use contract::PackerContract;
pub use error::{PackerError, PackerResult};
pub use masks::{
    assistant_only_loss_mask, causal_lm_labels, labels_from_loss_mask, packed_attention_mask,
    position_ids, prompt_loss_mask, sample_boundary_mask, BoundaryMask, LossMask,
    PackedAttentionMask, PositionIdPolicy,
};
pub use metrics::PackerMetrics;
pub use packer::{
    BestFitPacker, GreedyPacker, LabelMaskPolicy, PackedOutput, PackingPlan, PaddingPolicy,
    SequencePackingStrategy, SkippedSample, TunexSequencePacker,
};
pub use plugin::{PackerPlugin, PACKER_PLUGIN_NAME, PACKER_PLUGIN_VERSION};
pub use runtime::{
    PackChunk, PackChunkOutput, PackSchedule, PackingQueue, PackingScheduler, PackingWorker,
    ParallelPacker, ParallelPackingResult, ParallelPackingStats, QueuePressure, WorkerPackingStats,
};
pub use stream::{
    AsyncPackingStream, BackpressureAction, BackpressureHook, PackedBatchEnvelope,
    PackerStreamAdapter, PackerStreamStats, QueueCapacity, TokenizedBatchInput,
};
