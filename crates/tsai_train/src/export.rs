//! Learner export and import utilities.
//!
//! Provides functionality to save and load complete trained learners,
//! including model weights, training state, and configuration.
//!
//! # Example
//!
//! ```rust,ignore
//! use tsai_train::export::{save_learner, load_learner, LearnerExport};
//!
//! // Save a trained learner
//! save_learner(&learner, "./my_model")?;
//!
//! // Load a learner
//! let export = LearnerExport::load("./my_model")?;
//! let loaded_model = export.load_model::<B, Model>(&device)?;
//! ```

use std::path::{Path, PathBuf};

use burn::module::Module;
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::error::{Result, TrainError};
use crate::learner::{LearnerConfig, TrainingHistory, TrainingState};

/// Model weights file name.
const MODEL_WEIGHTS_FILE: &str = "model.mpk";
/// Training state file name.
const STATE_FILE: &str = "state.json";
/// Learner config file name.
const CONFIG_FILE: &str = "config.json";
/// Export metadata file name.
const METADATA_FILE: &str = "export_meta.json";

/// Metadata about an exported learner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportMetadata {
    /// Export format version.
    pub version: String,
    /// Model architecture name.
    pub arch: String,
    /// Number of classes (for classification).
    pub n_classes: Option<usize>,
    /// Input sequence length.
    pub seq_len: Option<usize>,
    /// Number of input variables.
    pub n_vars: Option<usize>,
    /// Export timestamp.
    pub timestamp: String,
    /// Best validation loss achieved.
    pub best_val_loss: Option<f32>,
    /// Best validation accuracy achieved.
    pub best_val_acc: Option<f32>,
    /// Total epochs trained.
    pub epochs_trained: usize,
    /// Additional metadata.
    pub extra: std::collections::HashMap<String, String>,
}

impl ExportMetadata {
    /// Create new export metadata.
    pub fn new(arch: impl Into<String>) -> Self {
        // Generate timestamp using std::time
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs().to_string())
            .unwrap_or_else(|_| "0".to_string());

        Self {
            version: "1.0".to_string(),
            arch: arch.into(),
            n_classes: None,
            seq_len: None,
            n_vars: None,
            timestamp,
            best_val_loss: None,
            best_val_acc: None,
            epochs_trained: 0,
            extra: std::collections::HashMap::new(),
        }
    }

    /// Set number of classes.
    #[must_use]
    pub fn with_n_classes(mut self, n: usize) -> Self {
        self.n_classes = Some(n);
        self
    }

    /// Set sequence length.
    #[must_use]
    pub fn with_seq_len(mut self, len: usize) -> Self {
        self.seq_len = Some(len);
        self
    }

    /// Set number of variables.
    #[must_use]
    pub fn with_n_vars(mut self, n: usize) -> Self {
        self.n_vars = Some(n);
        self
    }

    /// Set training stats.
    #[must_use]
    pub fn with_training_stats(mut self, state: &TrainingState) -> Self {
        self.epochs_trained = state.epoch;
        self.best_val_loss = Some(state.best_valid_loss);
        if let Some(last_metrics) = state.history.metrics.last() {
            if let Some(&acc) = last_metrics.get("accuracy") {
                self.best_val_acc = Some(acc);
            }
        }
        self
    }

    /// Add extra metadata.
    #[must_use]
    pub fn with_extra(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }
}

/// A loaded learner export.
#[derive(Debug)]
pub struct LearnerExport {
    /// Export directory path.
    pub path: PathBuf,
    /// Export metadata.
    pub metadata: ExportMetadata,
    /// Training state.
    pub state: TrainingState,
    /// Learner configuration.
    pub config: LearnerConfig,
}

impl LearnerExport {
    /// Load a learner export from a directory.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Load metadata
        let metadata_path = path.join(METADATA_FILE);
        let metadata: ExportMetadata = load_json(&metadata_path)?;

        // Load state
        let state_path = path.join(STATE_FILE);
        let state: TrainingState = load_json(&state_path)?;

        // Load config
        let config_path = path.join(CONFIG_FILE);
        let config: LearnerConfig = load_json(&config_path)?;

        Ok(Self {
            path,
            metadata,
            state,
            config,
        })
    }

    /// Load the model weights.
    ///
    /// # Type Parameters
    ///
    /// * `B` - The backend type
    /// * `M` - The model type (must match what was saved)
    pub fn load_model<B, M>(&self, device: &B::Device) -> Result<M::Record>
    where
        B: Backend,
        M: Module<B>,
        M::Record: DeserializeOwned,
    {
        let model_path = self.path.join(MODEL_WEIGHTS_FILE);
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        recorder
            .load(model_path, device)
            .map_err(|e| TrainError::CheckpointError(format!("Failed to load model: {}", e)))
    }

    /// Get the model weights path.
    pub fn model_path(&self) -> PathBuf {
        self.path.join(MODEL_WEIGHTS_FILE)
    }

    /// Get training history.
    pub fn history(&self) -> &TrainingHistory {
        &self.state.history
    }

    /// Get best validation loss.
    pub fn best_val_loss(&self) -> f32 {
        self.state.best_valid_loss
    }

    /// Get epochs trained.
    pub fn epochs_trained(&self) -> usize {
        self.state.epoch
    }
}

/// Save a model to a directory.
///
/// Creates a directory structure with:
/// - `model.mpk` - Model weights
/// - `state.json` - Training state
/// - `config.json` - Learner configuration
/// - `export_meta.json` - Export metadata
///
/// # Arguments
///
/// * `model` - The model to save
/// * `path` - Output directory path
/// * `config` - Learner configuration
/// * `state` - Training state
/// * `metadata` - Export metadata
pub fn save_model_bundle<B, M>(
    model: &M,
    path: impl AsRef<Path>,
    config: &LearnerConfig,
    state: &TrainingState,
    metadata: &ExportMetadata,
) -> Result<()>
where
    B: Backend,
    M: Module<B>,
    M::Record: Serialize,
{
    let path = path.as_ref();

    // Create output directory
    std::fs::create_dir_all(path)
        .map_err(|e| TrainError::CheckpointError(format!("Failed to create directory: {}", e)))?;

    // Save model weights
    let model_path = path.join(MODEL_WEIGHTS_FILE);
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let record = model.clone().into_record();
    recorder
        .record(record, model_path)
        .map_err(|e| TrainError::CheckpointError(format!("Failed to save model: {}", e)))?;

    // Save training state
    let state_path = path.join(STATE_FILE);
    save_json(state, &state_path)?;

    // Save config
    let config_path = path.join(CONFIG_FILE);
    save_json(config, &config_path)?;

    // Save metadata
    let metadata_path = path.join(METADATA_FILE);
    save_json(metadata, &metadata_path)?;

    Ok(())
}

/// Quick save function for saving just model weights.
///
/// # Arguments
///
/// * `model` - The model to save
/// * `path` - Output file path
pub fn quick_save<B, M>(model: &M, path: impl AsRef<Path>) -> Result<()>
where
    B: Backend,
    M: Module<B>,
    M::Record: Serialize,
{
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let record = model.clone().into_record();
    recorder
        .record(record, path.as_ref().to_path_buf())
        .map_err(|e| TrainError::CheckpointError(format!("Failed to save model: {}", e)))
}

/// Quick load function for loading just model weights.
///
/// # Arguments
///
/// * `path` - Path to model weights file
/// * `device` - Device to load onto
pub fn quick_load<B, M>(path: impl AsRef<Path>, device: &B::Device) -> Result<M::Record>
where
    B: Backend,
    M: Module<B>,
    M::Record: DeserializeOwned,
{
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    recorder
        .load(path.as_ref().to_path_buf(), device)
        .map_err(|e| TrainError::CheckpointError(format!("Failed to load model: {}", e)))
}

/// Helper to save JSON file.
fn save_json<T: Serialize>(data: &T, path: &Path) -> Result<()> {
    let json = serde_json::to_string_pretty(data)
        .map_err(|e| TrainError::SerializationError(format!("Failed to serialize: {}", e)))?;
    std::fs::write(path, json)
        .map_err(|e| TrainError::CheckpointError(format!("Failed to write file: {}", e)))
}

/// Helper to load JSON file.
fn load_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let json = std::fs::read_to_string(path)
        .map_err(|e| TrainError::CheckpointError(format!("Failed to read file: {}", e)))?;
    serde_json::from_str(&json)
        .map_err(|e| TrainError::SerializationError(format!("Failed to deserialize: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_metadata() {
        let meta = ExportMetadata::new("InceptionTimePlus")
            .with_n_classes(5)
            .with_seq_len(100)
            .with_n_vars(3)
            .with_extra("dataset", "NATOPS");

        assert_eq!(meta.arch, "InceptionTimePlus");
        assert_eq!(meta.n_classes, Some(5));
        assert_eq!(meta.seq_len, Some(100));
        assert_eq!(meta.n_vars, Some(3));
        assert_eq!(meta.extra.get("dataset"), Some(&"NATOPS".to_string()));
    }

    #[test]
    fn test_export_metadata_with_training_stats() {
        let mut state = TrainingState::default();
        state.epoch = 25;
        state.best_valid_loss = 0.15;

        let meta = ExportMetadata::new("ResNetPlus").with_training_stats(&state);

        assert_eq!(meta.epochs_trained, 25);
        assert_eq!(meta.best_val_loss, Some(0.15));
    }
}
