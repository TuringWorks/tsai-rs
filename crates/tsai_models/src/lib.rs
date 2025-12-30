//! # tsai_models
//!
//! Model zoo for tsai-rs: CNN, Transformer, ROCKET, RNN, and Tabular architectures.
//!
//! This crate provides deep learning architectures for time series:
//!
//! ## CNN Models
//! - [`InceptionTimePlus`] - InceptionTime with improvements
//! - [`ResNetPlus`] - ResNet adapted for time series
//! - [`XceptionTimePlus`] - Xception-inspired architecture
//! - [`OmniScaleCNN`] - Multi-scale CNN
//! - [`XCMPlus`] - Explainable CNN
//!
//! ## Transformer Models
//! - [`TSTPlus`] - Time Series Transformer
//! - [`TSiTPlus`] - Improved Time Series Transformer with multiple PE options
//! - [`TSPerceiver`] - Perceiver for time series
//! - [`PatchTST`] - Patch-based Transformer
//!
//! ## ROCKET Family
//! - [`MiniRocket`] - Fast random convolutional features
//! - [`MultiRocketPlus`] - Multiple ROCKET kernels
//! - [`HydraPlus`] - Hybrid ROCKET
//!
//! ## RNN Models
//! - [`RNNPlus`] - LSTM/GRU with improvements
//! - [`RNNAttention`] - RNN with attention
//!
//! ## Tabular Models
//! - [`TabTransformer`] - Transformer for tabular data
//! - [`TabFusionTransformer`] - Fusion of time series and tabular

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod checkpoint;
pub mod cnn;
pub mod registry;
pub mod rnn;
pub mod rocket;
pub mod tabular;
pub mod traits;
pub mod transformer;

pub use checkpoint::{
    save_model, load_record, CheckpointError, CheckpointFormat, CheckpointMetadata,
    CheckpointPrecision, ModelCheckpoint,
};
pub use cnn::*;
pub use registry::{default_registry, ModelRegistry, RegistryError, TSModel};
pub use rnn::*;
pub use rocket::*;
pub use tabular::*;
pub use transformer::*;
