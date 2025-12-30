//! # tsai_analysis
//!
//! Analysis utilities for tsai-rs: confusion matrix, top losses, permutation importance.
//!
//! This crate provides tools for analyzing model performance:
//! - Confusion matrix computation and visualization
//! - Top losses identification
//! - Feature and step importance via permutation

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

mod confusion;
mod importance;
mod top_losses;

pub use confusion::{ConfusionMatrix, confusion_matrix};
pub use importance::{feature_importance, step_importance, PermutationImportance};
pub use top_losses::{top_losses, TopLoss};
