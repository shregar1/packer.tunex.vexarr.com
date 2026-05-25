//! Loss mask, label, attention, position, and boundary generation.

pub mod attention;
pub mod boundary;
pub mod labels;
pub mod loss_mask;
pub mod position;

pub use attention::{packed_attention_mask, PackedAttentionMask};
pub use boundary::{sample_boundary_mask, BoundaryMask};
pub use labels::{causal_lm_labels, labels_from_loss_mask};
pub use loss_mask::{assistant_only_loss_mask, prompt_loss_mask, LossMask};
pub use position::{position_ids, PositionIdPolicy};
