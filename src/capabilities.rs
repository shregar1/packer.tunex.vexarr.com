//! TuneX packer plugin capabilities.

use tunex_core::plugin::PluginCapability;

/// Generic sequence packing support.
pub const SEQUENCE_PACKING: &str = "sequence-packing";
/// Greedy packing strategy support.
pub const GREEDY_PACKING: &str = "greedy-packing";
/// Best-fit packing strategy support.
pub const BEST_FIT_PACKING: &str = "best-fit-packing";
/// GPU transfer-ready contiguous batch layout support.
pub const GPU_READY_LAYOUT: &str = "gpu-ready-layout";
/// Loss mask and label generation support.
pub const LOSS_MASK_GENERATION: &str = "loss-mask-generation";
/// Assistant-only loss masking support.
pub const ASSISTANT_ONLY_LOSS: &str = "assistant-only-loss";
/// Parallel packing runtime support.
pub const PARALLEL_PACKING: &str = "parallel-packing";
/// Deterministic packing order support.
pub const DETERMINISTIC_PACKING: &str = "deterministic-packing";

/// Stable capability list advertised by the packer plugin.
pub const PACKER_CAPABILITY_NAMES: &[&str] = &[
    SEQUENCE_PACKING,
    GREEDY_PACKING,
    BEST_FIT_PACKING,
    GPU_READY_LAYOUT,
    LOSS_MASK_GENERATION,
    ASSISTANT_ONLY_LOSS,
    PARALLEL_PACKING,
    DETERMINISTIC_PACKING,
];

/// Builds core plugin capability values.
#[must_use]
pub fn packer_capabilities() -> Vec<PluginCapability> {
    PACKER_CAPABILITY_NAMES
        .iter()
        .map(|capability| PluginCapability::new(*capability))
        .collect()
}
