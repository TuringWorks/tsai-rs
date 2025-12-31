//! # tsai_data
//!
//! Dataset and dataloader implementations for tsai-rs time series deep learning.
//!
//! This crate provides:
//! - [`TSDataset`] for storing time series data
//! - [`TSDatasets`] for train/valid/test split management
//! - [`TSDataLoader`] for batched iteration with shuffling
//! - [`TSDataLoaders`] for paired train/valid dataloaders
//! - I/O utilities for NPY, CSV, and Parquet formats
//!
//! ## Example
//!
//! ```rust,ignore
//! use tsai_data::{TSDataset, TSDataLoader, TSDataLoaders};
//! use tsai_core::Seed;
//!
//! // Load data and create dataloaders
//! let train_ds = TSDataset::from_arrays(x_train, y_train)?;
//! let valid_ds = TSDataset::from_arrays(x_valid, y_valid)?;
//!
//! let dls = TSDataLoaders::new(train_ds, valid_ds)
//!     .batch_size(32)
//!     .seed(Seed::new(42))
//!     .build()?;
//! ```

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

mod dataset;
mod error;
mod io;
mod loader;
mod sampler;
mod splits;
pub mod ucr;

pub use dataset::{TSDataset, TSDatasets};
pub use error::{DataError, Result};
pub use io::{read_npy, read_npz};
#[cfg(feature = "polars-io")]
pub use io::{read_csv, read_parquet};
pub use loader::{TSDataLoader, TSDataLoaderBuilder, TSDataLoaders, TSDataLoadersBuilder};
pub use sampler::{RandomSampler, SequentialSampler, StratifiedSampler};
pub use splits::{train_test_split, train_valid_test_split, SplitStrategy};
pub use ucr::{list_datasets as list_ucr_datasets, UCRDataset, UCRDatasetInfo};

/// Cache directory for downloaded datasets.
pub const CACHE_DIR: &str = ".cache/tsai-rs";

/// Get the default cache directory path.
#[must_use]
pub fn cache_dir() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join(CACHE_DIR)
}
