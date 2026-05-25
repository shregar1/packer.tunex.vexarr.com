//! Convenient public imports for packer users.

pub use crate::batch::{
    align_capacity, Alignment, AttentionMask, PackedBatch, PackedBatchBuilder, PackedBatchLayout,
    PackedBatchSlice, PackedBatchView, PositionIds, SequenceBoundary,
};
pub use crate::config::{
    ConfigValidationError, ConfigValidationReport, LabelMode, LayoutConfig, MaskConfig,
    PackerConfig, PackingStrategy, ValidatePackerConfig,
};
pub use crate::error::{PackerError, PackerResult};
pub use crate::masks::{
    assistant_only_loss_mask, causal_lm_labels, labels_from_loss_mask, packed_attention_mask,
    position_ids, prompt_loss_mask, sample_boundary_mask, BoundaryMask, LossMask,
    PackedAttentionMask, PositionIdPolicy,
};
pub use crate::metrics::PackerMetrics;
pub use crate::packer::{
    best_fit_bins, sort_by_length_desc, BestFitPacker, GreedyPacker, LabelMaskPolicy, PackedOutput,
    PackingBin, PackingPlan, PackingWaste, PaddingPolicy, SequencePackingStrategy, SkippedSample,
    SortableSample, TunexSequencePacker,
};
pub use crate::plugin::{PackerPlugin, PACKER_PLUGIN_NAME, PACKER_PLUGIN_VERSION};
pub use crate::runtime::{
    PackChunk, PackChunkOutput, PackSchedule, PackingQueue, PackingScheduler, PackingWorker,
    ParallelPacker, ParallelPackingResult, ParallelPackingStats, QueuePressure, WorkerPackingStats,
};
