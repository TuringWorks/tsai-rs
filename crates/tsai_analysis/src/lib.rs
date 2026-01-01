//! # tsai_analysis
//!
//! Analysis utilities for tsai-rs: confusion matrix, top losses, permutation importance.
//!
//! This crate provides tools for analyzing model performance:
//! - Confusion matrix computation and visualization
//! - Top losses identification
//! - Feature and step importance via permutation
//! - Calibration analysis (ECE, MCE, temperature scaling)
//! - Classification report (per-class precision, recall, F1)

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

mod calibration;
mod confusion;
mod importance;
mod report;
mod top_losses;

pub use calibration::{
    calibration_from_probs, compute_calibration, find_optimal_temperature, temperature_scale,
    CalibrationResult,
};
pub use confusion::{confusion_matrix, ConfusionMatrix};
pub use importance::{feature_importance, step_importance, PermutationImportance};
pub use report::{classification_report, ClassMetrics, ClassificationReport};
pub use top_losses::{top_losses, TopLoss};
