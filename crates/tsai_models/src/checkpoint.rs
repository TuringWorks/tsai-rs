//! Model checkpointing and serialization utilities.
//!
//! Provides utilities for saving and loading model weights using Burn's record system.
//!
//! # Supported Formats
//!
//! - **MessagePack** (`*.mpk`): Fast binary format, good for local storage
//! - **SafeTensors** (`*.safetensors`): Portable format, safe for sharing
//! - **Named MessagePack** (`*.named.mpk`): Includes parameter names
//!
//! # Example
//!
//! ```rust,ignore
//! use tsai_models::checkpoint::{save_model, load_model, CheckpointFormat};
//! use tsai_models::InceptionTimePlusConfig;
//!
//! // Configure and create model
//! let config = InceptionTimePlusConfig::new(1, 100, 5);
//! let model = config.init::<NdArray>(&device);
//!
//! // Save model
//! save_model(&model, "model.mpk", CheckpointFormat::MessagePack)?;
//!
//! // Load model
//! let loaded = load_model::<_, InceptionTimePlus<_>>(
//!     &config,
//!     "model.mpk",
//!     CheckpointFormat::MessagePack,
//!     &device
//! )?;
//! ```

use std::path::Path;

use burn::module::Module;
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use serde::{de::DeserializeOwned, Serialize};

/// Checkpoint format for model serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointFormat {
    /// MessagePack binary format (fast, compact).
    MessagePack,
    /// Named MessagePack (includes parameter names).
    NamedMessagePack,
}

/// Precision setting for checkpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointPrecision {
    /// Full precision (f32).
    Full,
    /// Half precision (f16).
    Half,
}

/// Save a model to a checkpoint file.
///
/// # Arguments
///
/// * `model` - The model to save
/// * `path` - Output path
/// * `format` - Checkpoint format
///
/// # Returns
///
/// Result indicating success or failure.
pub fn save_model<B, M>(model: &M, path: impl AsRef<Path>, format: CheckpointFormat) -> Result<()>
where
    B: Backend,
    M: Module<B>,
    M::Record: Serialize,
{
    let path = path.as_ref();
    let record = model.clone().into_record();

    match format {
        CheckpointFormat::MessagePack | CheckpointFormat::NamedMessagePack => {
            let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
            recorder
                .record(record, path.to_path_buf())
                .map_err(|e| CheckpointError::Save(e.to_string()))?;
        }
    }

    Ok(())
}

/// Load a model from a checkpoint file.
///
/// # Arguments
///
/// * `path` - Path to checkpoint
/// * `format` - Checkpoint format
/// * `device` - Device to load model onto
///
/// # Returns
///
/// The loaded model record.
pub fn load_record<B, M>(
    path: impl AsRef<Path>,
    format: CheckpointFormat,
    device: &B::Device,
) -> Result<M::Record>
where
    B: Backend,
    M: Module<B>,
    M::Record: DeserializeOwned,
{
    let path = path.as_ref();

    let record = match format {
        CheckpointFormat::MessagePack | CheckpointFormat::NamedMessagePack => {
            let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
            recorder
                .load(path.to_path_buf(), device)
                .map_err(|e| CheckpointError::Load(e.to_string()))?
        }
    };

    Ok(record)
}

/// Model checkpoint metadata.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckpointMetadata {
    /// Model architecture name.
    pub arch: String,
    /// Model configuration as JSON.
    pub config_json: String,
    /// Training epoch (if applicable).
    pub epoch: Option<usize>,
    /// Validation loss (if applicable).
    pub val_loss: Option<f32>,
    /// Validation accuracy (if applicable).
    pub val_acc: Option<f32>,
    /// Additional metadata.
    pub extra: std::collections::HashMap<String, String>,
}

impl CheckpointMetadata {
    /// Create new metadata for a model.
    pub fn new(arch: impl Into<String>) -> Self {
        Self {
            arch: arch.into(),
            config_json: String::new(),
            epoch: None,
            val_loss: None,
            val_acc: None,
            extra: std::collections::HashMap::new(),
        }
    }

    /// Set the config JSON.
    #[must_use]
    pub fn with_config<C: Serialize>(mut self, config: &C) -> Self {
        self.config_json = serde_json::to_string(config).unwrap_or_default();
        self
    }

    /// Set the training epoch.
    #[must_use]
    pub fn with_epoch(mut self, epoch: usize) -> Self {
        self.epoch = Some(epoch);
        self
    }

    /// Set the validation loss.
    #[must_use]
    pub fn with_val_loss(mut self, loss: f32) -> Self {
        self.val_loss = Some(loss);
        self
    }

    /// Set the validation accuracy.
    #[must_use]
    pub fn with_val_acc(mut self, acc: f32) -> Self {
        self.val_acc = Some(acc);
        self
    }

    /// Add extra metadata.
    #[must_use]
    pub fn with_extra(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }

    /// Save metadata to a JSON file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| CheckpointError::Save(e.to_string()))?;
        std::fs::write(path, json).map_err(|e| CheckpointError::Save(e.to_string()))?;
        Ok(())
    }

    /// Load metadata from a JSON file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let json =
            std::fs::read_to_string(path).map_err(|e| CheckpointError::Load(e.to_string()))?;
        serde_json::from_str(&json).map_err(|e| CheckpointError::Load(e.to_string()))
    }
}

/// Result type for checkpoint operations.
pub type Result<T> = std::result::Result<T, CheckpointError>;

/// Checkpoint-related errors.
#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    /// Error saving checkpoint.
    #[error("Failed to save checkpoint: {0}")]
    Save(String),

    /// Error loading checkpoint.
    #[error("Failed to load checkpoint: {0}")]
    Load(String),

    /// Invalid format.
    #[error("Invalid checkpoint format: {0}")]
    InvalidFormat(String),
}

/// Extension trait for models to add checkpoint methods.
pub trait ModelCheckpoint<B: Backend>: Module<B> {
    /// Save the model to a checkpoint file.
    fn save_checkpoint(&self, path: impl AsRef<Path>) -> Result<()>
    where
        Self::Record: Serialize,
    {
        save_model::<B, Self>(self, path, CheckpointFormat::NamedMessagePack)
    }

    /// Load model from a checkpoint into an existing model.
    fn load_checkpoint(&self, path: impl AsRef<Path>, device: &B::Device) -> Result<Self>
    where
        Self: Sized,
        Self::Record: DeserializeOwned,
    {
        let record = load_record::<B, Self>(path, CheckpointFormat::NamedMessagePack, device)?;
        Ok(self.clone().load_record(record))
    }
}

// Implement for all modules
impl<B: Backend, M: Module<B>> ModelCheckpoint<B> for M {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_metadata() {
        let meta = CheckpointMetadata::new("InceptionTimePlus")
            .with_epoch(10)
            .with_val_loss(0.25)
            .with_val_acc(0.92)
            .with_extra("dataset", "ECG200");

        assert_eq!(meta.arch, "InceptionTimePlus");
        assert_eq!(meta.epoch, Some(10));
        assert_eq!(meta.val_loss, Some(0.25));
        assert_eq!(meta.val_acc, Some(0.92));
        assert_eq!(meta.extra.get("dataset"), Some(&"ECG200".to_string()));
    }

    #[test]
    fn test_checkpoint_format() {
        assert_eq!(CheckpointFormat::MessagePack, CheckpointFormat::MessagePack);
        assert_ne!(
            CheckpointFormat::MessagePack,
            CheckpointFormat::NamedMessagePack
        );
    }

    #[test]
    fn test_checkpoint_precision() {
        assert_eq!(CheckpointPrecision::Full, CheckpointPrecision::Full);
        assert_ne!(CheckpointPrecision::Full, CheckpointPrecision::Half);
    }
}
