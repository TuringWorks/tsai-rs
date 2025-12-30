//! # tsai
//!
//! Time series deep learning in Rust - a feature-parity port of Python tsai.
//!
//! tsai-rs provides a comprehensive toolkit for time series analysis using deep learning:
//!
//! - **Data handling**: Datasets, dataloaders, and preprocessing
//! - **Transforms**: Augmentations, label mixing, and imaging transforms
//! - **Models**: CNN, Transformer, ROCKET, RNN, and Tabular architectures
//! - **Training**: Learner, callbacks, metrics, and schedulers
//! - **Analysis**: Confusion matrix, top losses, importance
//! - **Explainability**: Attribution maps, activation capture
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use tsai::prelude::*;
//!
//! // Load data
//! let x = read_npy("data/X_train.npy")?;
//! let y = read_npy("data/y_train.npy")?;
//! let dataset = TSDataset::from_arrays(x, Some(y))?;
//!
//! // Create dataloaders
//! let (train_ds, valid_ds) = train_test_split(&dataset, 0.2, Seed::new(42))?;
//! let dls = TSDataLoaders::builder(train_ds, valid_ds)
//!     .batch_size(64)
//!     .build()?;
//!
//! // Create model
//! let config = InceptionTimePlusConfig::new(dls.n_vars(), dls.seq_len(), n_classes);
//! let model = config.init(&device);
//!
//! // Train
//! let learner = Learner::new(model, dls, LearnerConfig::default(), &device);
//! learner.fit_one_cycle(25, 1e-3)?;
//! ```
//!
//! ## Feature Flags
//!
//! - `backend-ndarray` (default): CPU backend using ndarray
//! - `backend-wgpu`: GPU backend using WGPU (Metal on macOS, Vulkan on Linux/Windows)
//! - `backend-tch`: PyTorch backend via tch-rs
//! - `wandb`: Weights & Biases integration

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

// Re-export all crates
pub use tsai_analysis as analysis;
pub use tsai_core as core;
pub use tsai_data as data;
pub use tsai_explain as explain;
pub use tsai_models as models;
pub use tsai_train as train;
pub use tsai_transforms as transforms;

/// Prelude module for convenient imports.
///
/// ```rust,ignore
/// use tsai::prelude::*;
/// ```
pub mod prelude {
    // Core types
    pub use tsai_core::{Result, Seed, Split, TSBatch, TSShape, TSTensor, Transform};

    // Data
    pub use tsai_data::{
        read_csv, read_npy, read_npz, read_parquet, train_test_split, train_valid_test_split,
        TSDataLoader, TSDataLoaders, TSDataset, TSDatasets,
    };

    // Transforms
    pub use tsai_transforms::{
        Compose, CutMix1d, CutOut, GaussianNoise, Identity, MagScale, MixUp1d, TimeWarp,
    };

    // Models
    pub use tsai_models::{
        InceptionTimePlus, InceptionTimePlusConfig, MiniRocket, MiniRocketConfig, PatchTST,
        PatchTSTConfig, RNNPlus, RNNPlusConfig, ResNetPlus, ResNetPlusConfig, TSTPlus, TSTConfig,
    };

    // Training
    pub use tsai_train::{
        Accuracy, Callback, CrossEntropyLoss, Learner, LearnerConfig, MSE, MSELoss, Metric,
        OneCycleLR, Scheduler,
    };

    // Analysis
    pub use tsai_analysis::{confusion_matrix, top_losses, ConfusionMatrix};

    // Explain
    pub use tsai_explain::{AttributionMap, AttributionMethod};
}

/// All module for importing everything.
///
/// Mirrors the `from tsai.all import *` pattern from Python.
pub mod all {
    pub use super::prelude::*;

    // Additional exports
    pub use tsai_core::backend;
    pub use tsai_data::{RandomSampler, SequentialSampler, StratifiedSampler};
    pub use tsai_train::{
        CallbackContext, CallbackList, EarlyStoppingCallback, OneCycleLR, ProgressCallback,
    };
    pub use tsai_transforms::{GAFType, RecurrencePlotConfig, TSToGADF, TSToGASF, TSToRP};
}

/// Compatibility module for sklearn-like API.
pub mod compat {
    pub use tsai_train::compat::{
        TSClassifier, TSClassifierConfig, TSForecaster, TSForecasterConfig, TSRegressor,
        TSRegressorConfig,
    };
}
