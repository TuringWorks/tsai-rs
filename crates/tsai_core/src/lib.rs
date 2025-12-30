//! # tsai_core
//!
//! Core types and traits for tsai-rs time series deep learning.
//!
//! This crate provides:
//! - [`Seed`] for deterministic random number generation
//! - [`TSShape`] for time series tensor shape metadata
//! - [`TSTensor`] wrapper for Burn tensors with shape validation
//! - [`Transform`] trait for data augmentation
//! - Error types and common utilities
//!
//! ## Shape Convention
//!
//! Time series data follows the convention `(B, V, L)`:
//! - `B`: Batch size (number of samples)
//! - `V`: Variables/channels/features
//! - `L`: Sequence length (time steps)
//!
//! ## Example
//!
//! ```rust,ignore
//! use tsai_core::{Seed, TSShape, TSTensor};
//!
//! let seed = Seed::new(42);
//! let shape = TSShape::new(32, 3, 100); // batch=32, vars=3, len=100
//! ```

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

mod error;
mod model_trait;
mod seed;
mod shape;
mod split;
mod tensor;
mod transform;

pub use error::{CoreError, Result};
pub use model_trait::{TSClassificationModel, TSForecastingModel, TSRegressionModel};
pub use seed::Seed;
pub use shape::TSShape;
pub use split::Split;
pub use tensor::{TSBatch, TSMaskTensor, TSTensor};
pub use transform::{Compose, Identity, Transform};

/// Backend type aliases for convenience
pub mod backend {
    #[cfg(feature = "backend-ndarray")]
    pub use burn_ndarray::NdArray;

    #[cfg(feature = "backend-wgpu")]
    pub use burn_wgpu::Wgpu;

    #[cfg(feature = "backend-tch")]
    pub use burn_tch::LibTorch;
}
