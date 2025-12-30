//! # tsai_explain
//!
//! Explainability tools for tsai-rs: activation/gradient capture, attribution maps.
//!
//! This crate provides:
//! - Activation capture hooks
//! - Gradient capture hooks
//! - Attribution maps (CAM-like for CNNs, attention for transformers)

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

mod activation;
mod attribution;

pub use activation::{ActivationCapture, GradientCapture};
pub use attribution::{
    attention_attribution, grad_cam, input_gradient, integrated_gradients, random_baseline,
    zero_baseline, AttentionAggregation, AttributionMap, AttributionMethod, BaselineType,
    IntegratedGradientsConfig,
};
