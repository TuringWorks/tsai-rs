//! Compatibility facades for sklearn-like API.
//!
//! Provides TSClassifier, TSRegressor, and TSForecaster classes
//! that mirror the Python tsai API.
//!
//! These wrappers provide a simple, sklearn-like interface for time series
//! classification, regression, and forecasting without requiring deep knowledge
//! of Burn's type system.

use burn::prelude::*;
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;
#[allow(unused_imports)]
use burn::module::AutodiffModule;
use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};

use crate::error::{Result, TrainError};
use crate::training::{ClassificationTrainer, ClassificationTrainerConfig, RegressionTrainer, RegressionTrainerConfig};
use tsai_core::Seed;
use tsai_data::{train_test_split, TSDataLoaders, TSDataset};
use tsai_models::{
    InceptionTimePlus, InceptionTimePlusConfig,
    OmniScaleCNN, OmniScaleCNNConfig,
    TSTConfig, TSTPlus,
};

/// Type alias for the training backend (CPU with autodiff).
type TrainBackend = Autodiff<NdArray>;

/// Type alias for the inference backend (CPU only).
type InferBackend = NdArray;

/// Trained model storage for different architectures.
///
/// This enum holds the trained (inference-ready) model for each supported architecture.
enum TrainedModel {
    InceptionTimePlus(InceptionTimePlus<InferBackend>),
    OmniScaleCNN(OmniScaleCNN<InferBackend>),
    TSTPlus(TSTPlus<InferBackend>),
}

impl TrainedModel {
    /// Run inference on the trained model.
    fn forward(&self, x: Tensor<InferBackend, 3>) -> Tensor<InferBackend, 2> {
        match self {
            TrainedModel::InceptionTimePlus(m) => m.forward(x),
            TrainedModel::OmniScaleCNN(m) => m.forward(x),
            TrainedModel::TSTPlus(m) => m.forward(x),
        }
    }
}

/// Configuration for sklearn-like classifiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TSClassifierConfig {
    /// Model architecture name.
    pub arch: String,
    /// Number of epochs.
    pub n_epochs: usize,
    /// Learning rate.
    pub lr: f64,
    /// Batch size.
    pub batch_size: usize,
    /// Validation ratio.
    pub valid_ratio: f32,
    /// Random seed.
    pub seed: u64,
    /// Whether to use GPU.
    pub use_gpu: bool,
}

impl Default for TSClassifierConfig {
    fn default() -> Self {
        Self {
            arch: "InceptionTimePlus".to_string(),
            n_epochs: 25,
            lr: 1e-3,
            batch_size: 64,
            valid_ratio: 0.2,
            seed: 42,
            use_gpu: false,
        }
    }
}

/// Sklearn-like time series classifier.
///
/// This classifier provides a simple, high-level API for time series classification
/// that mirrors the Python tsai library. It handles model creation, training, and
/// inference internally.
///
/// # Supported Architectures
///
/// - `InceptionTimePlus` (default) - InceptionTime with improvements
/// - `OmniScaleCNN` - Multi-scale CNN
/// - `TSTPlus` - Time Series Transformer
///
/// # Example
///
/// ```rust,ignore
/// use tsai_train::compat::{TSClassifier, TSClassifierConfig};
/// use ndarray::Array3;
///
/// let config = TSClassifierConfig::default();
/// let mut clf = TSClassifier::new(config);
///
/// // x: (n_samples, n_vars, seq_len), y: (n_samples, 1)
/// clf.fit(&x_train, &y_train)?;
/// let predictions = clf.predict(&x_test)?;
/// let probabilities = clf.predict_proba(&x_test)?;
/// ```
pub struct TSClassifier {
    config: TSClassifierConfig,
    trained_model: Option<TrainedModel>,
    n_classes: usize,
    n_vars: usize,
    seq_len: usize,
    device: <InferBackend as Backend>::Device,
}

impl TSClassifier {
    /// Create a new classifier.
    pub fn new(config: TSClassifierConfig) -> Self {
        Self {
            config,
            trained_model: None,
            n_classes: 0,
            n_vars: 0,
            seq_len: 0,
            device: Default::default(),
        }
    }

    /// Fit the classifier.
    ///
    /// # Arguments
    ///
    /// * `x` - Input data of shape (n_samples, n_vars, seq_len)
    /// * `y` - Labels of shape (n_samples, 1) with class indices
    ///
    /// # Returns
    ///
    /// Training metrics including final accuracy.
    pub fn fit(&mut self, x: &Array3<f32>, y: &Array2<f32>) -> Result<TrainingMetrics> {
        let (_n_samples, n_vars, seq_len) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        self.n_vars = n_vars;
        self.seq_len = seq_len;

        // Determine number of classes
        let y_flat: Vec<f32> = y.iter().copied().collect();
        let max_class = y_flat.iter().cloned().fold(0.0f32, f32::max) as usize;
        self.n_classes = max_class + 1;

        // Create dataset
        let dataset = TSDataset::from_arrays(x.clone(), Some(y.clone()))?;

        // Split into train/valid
        let seed = Seed::new(self.config.seed);
        let (train_ds, valid_ds) = train_test_split(&dataset, self.config.valid_ratio, seed)?;

        // Create dataloaders
        let dls = TSDataLoaders::builder(train_ds, valid_ds)
            .batch_size(self.config.batch_size)
            .seed(seed)
            .build()?;

        // Train based on architecture
        let metrics = match self.config.arch.as_str() {
            "InceptionTimePlus" | "inception" | "inception_time" => {
                self.train_inception_time(&dls)?
            }
            "OmniScaleCNN" | "omniscale" => {
                self.train_omniscale(&dls)?
            }
            "TSTPlus" | "tst" | "transformer" => {
                self.train_tst(&dls)?
            }
            other => {
                return Err(TrainError::Other(format!(
                    "Unknown architecture '{}'. Supported: InceptionTimePlus, OmniScaleCNN, TSTPlus",
                    other
                )));
            }
        };

        Ok(metrics)
    }

    fn train_inception_time(&mut self, dls: &TSDataLoaders) -> Result<TrainingMetrics> {
        let train_device: <TrainBackend as Backend>::Device = Default::default();

        // Create model config
        let model_config = InceptionTimePlusConfig {
            n_vars: self.n_vars,
            seq_len: self.seq_len,
            n_classes: self.n_classes,
            n_blocks: 6,
            n_filters: 32,
            kernel_sizes: [9, 19, 39],
            bottleneck_dim: 32,
            dropout: 0.0,
        };

        // Initialize model for training
        let model: InceptionTimePlus<TrainBackend> = model_config.init(&train_device);

        // Configure trainer
        let trainer_config = ClassificationTrainerConfig {
            n_epochs: self.config.n_epochs,
            lr: self.config.lr,
            weight_decay: 0.01,
            grad_clip: 1.0,
            verbose: true,
            early_stopping_patience: 0,
            early_stopping_min_delta: 0.001,
        };

        let trainer = ClassificationTrainer::<TrainBackend>::new(trainer_config, train_device);

        // Train with forward functions
        let output = trainer.fit_with_forward(
            model,
            dls,
            |m, x| m.forward(x),
            |m, x| m.forward(x),
        )?;

        // Store the trained model (convert to inference mode)
        let inner_model = output.model.valid();
        self.trained_model = Some(TrainedModel::InceptionTimePlus(inner_model));

        Ok(TrainingMetrics {
            train_losses: output.train_losses,
            valid_losses: output.valid_losses,
            valid_accs: output.valid_accs,
            best_valid_acc: output.best_valid_acc,
            best_epoch: output.best_epoch,
            training_time_secs: output.training_time_secs,
        })
    }

    fn train_omniscale(&mut self, dls: &TSDataLoaders) -> Result<TrainingMetrics> {
        let train_device: <TrainBackend as Backend>::Device = Default::default();

        // Create model config
        let model_config = OmniScaleCNNConfig::new(self.n_vars, self.seq_len, self.n_classes)
            .with_n_filters(64)
            .with_dropout(0.1);

        // Initialize model for training
        let model: OmniScaleCNN<TrainBackend> = model_config.init(&train_device);

        // Configure trainer
        let trainer_config = ClassificationTrainerConfig {
            n_epochs: self.config.n_epochs,
            lr: self.config.lr,
            weight_decay: 0.01,
            grad_clip: 1.0,
            verbose: true,
            early_stopping_patience: 0,
            early_stopping_min_delta: 0.001,
        };

        let trainer = ClassificationTrainer::<TrainBackend>::new(trainer_config, train_device);

        // Train with forward functions
        let output = trainer.fit_with_forward(
            model,
            dls,
            |m, x| m.forward(x),
            |m, x| m.forward(x),
        )?;

        // Store the trained model
        let inner_model = output.model.valid();
        self.trained_model = Some(TrainedModel::OmniScaleCNN(inner_model));

        Ok(TrainingMetrics {
            train_losses: output.train_losses,
            valid_losses: output.valid_losses,
            valid_accs: output.valid_accs,
            best_valid_acc: output.best_valid_acc,
            best_epoch: output.best_epoch,
            training_time_secs: output.training_time_secs,
        })
    }

    fn train_tst(&mut self, dls: &TSDataLoaders) -> Result<TrainingMetrics> {
        let train_device: <TrainBackend as Backend>::Device = Default::default();

        // Create model config
        let d_model = 64;
        let model_config = TSTConfig {
            n_vars: self.n_vars,
            seq_len: self.seq_len,
            n_classes: self.n_classes,
            d_model,
            n_heads: 4,
            n_layers: 3,
            d_ff: d_model * 4,
            dropout: 0.1,
            use_pe: true,
        };

        // Initialize model for training
        let model: TSTPlus<TrainBackend> = model_config.init(&train_device);

        // Configure trainer
        let trainer_config = ClassificationTrainerConfig {
            n_epochs: self.config.n_epochs,
            lr: self.config.lr,
            weight_decay: 0.01,
            grad_clip: 1.0,
            verbose: true,
            early_stopping_patience: 0,
            early_stopping_min_delta: 0.001,
        };

        let trainer = ClassificationTrainer::<TrainBackend>::new(trainer_config, train_device);

        // Train with forward functions
        let output = trainer.fit_with_forward(
            model,
            dls,
            |m, x| m.forward(x),
            |m, x| m.forward(x),
        )?;

        // Store the trained model
        let inner_model = output.model.valid();
        self.trained_model = Some(TrainedModel::TSTPlus(inner_model));

        Ok(TrainingMetrics {
            train_losses: output.train_losses,
            valid_losses: output.valid_losses,
            valid_accs: output.valid_accs,
            best_valid_acc: output.best_valid_acc,
            best_epoch: output.best_epoch,
            training_time_secs: output.training_time_secs,
        })
    }

    /// Predict class labels.
    ///
    /// # Arguments
    ///
    /// * `x` - Input data of shape (n_samples, n_vars, seq_len)
    ///
    /// # Returns
    ///
    /// Class predictions of shape (n_samples, 1)
    pub fn predict(&self, x: &Array3<f32>) -> Result<Array2<i32>> {
        let proba = self.predict_proba(x)?;

        // Get argmax for each sample
        let n_samples = proba.nrows();
        let mut predictions = Array2::zeros((n_samples, 1));

        for i in 0..n_samples {
            let row = proba.row(i);
            let (max_idx, _) = row.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            predictions[[i, 0]] = max_idx as i32;
        }

        Ok(predictions)
    }

    /// Predict class probabilities.
    ///
    /// # Arguments
    ///
    /// * `x` - Input data of shape (n_samples, n_vars, seq_len)
    ///
    /// # Returns
    ///
    /// Class probabilities of shape (n_samples, n_classes)
    pub fn predict_proba(&self, x: &Array3<f32>) -> Result<Array2<f32>> {
        let model = self.trained_model.as_ref().ok_or_else(|| {
            TrainError::Other("Model not fitted. Call fit() first.".to_string())
        })?;

        let (n_samples, n_vars, seq_len) = (x.shape()[0], x.shape()[1], x.shape()[2]);

        // Convert ndarray to Burn tensor
        let data: Vec<f32> = x.iter().copied().collect();
        let tensor_data = burn::tensor::TensorData::new(data, [n_samples, n_vars, seq_len]);
        let tensor: Tensor<InferBackend, 3> = Tensor::from_data(tensor_data, &self.device);

        // Run inference
        let logits = model.forward(tensor);

        // Apply softmax to get probabilities
        let probs = burn::tensor::activation::softmax(logits, 1);

        // Convert back to ndarray
        let probs_data: Vec<f32> = probs.into_data().to_vec().unwrap();
        let result = Array2::from_shape_vec((n_samples, self.n_classes), probs_data)
            .map_err(|e| TrainError::Other(e.to_string()))?;

        Ok(result)
    }

    /// Get the config.
    pub fn config(&self) -> &TSClassifierConfig {
        &self.config
    }

    /// Check if fitted.
    pub fn is_fitted(&self) -> bool {
        self.trained_model.is_some()
    }

    /// Get the number of classes (after fitting).
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }
}

/// Training metrics returned after fitting.
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    /// Training losses per epoch.
    pub train_losses: Vec<f32>,
    /// Validation losses per epoch.
    pub valid_losses: Vec<f32>,
    /// Validation accuracies per epoch.
    pub valid_accs: Vec<f32>,
    /// Best validation accuracy achieved.
    pub best_valid_acc: f32,
    /// Epoch with best validation accuracy.
    pub best_epoch: usize,
    /// Total training time in seconds.
    pub training_time_secs: f64,
}

/// Configuration for sklearn-like regressors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TSRegressorConfig {
    /// Model architecture name.
    pub arch: String,
    /// Number of epochs.
    pub n_epochs: usize,
    /// Learning rate.
    pub lr: f64,
    /// Batch size.
    pub batch_size: usize,
    /// Validation ratio.
    pub valid_ratio: f32,
    /// Random seed.
    pub seed: u64,
}

impl Default for TSRegressorConfig {
    fn default() -> Self {
        Self {
            arch: "InceptionTimePlus".to_string(),
            n_epochs: 25,
            lr: 1e-3,
            batch_size: 64,
            valid_ratio: 0.2,
            seed: 42,
        }
    }
}

/// Trained regression model storage for different architectures.
///
/// Uses the same model architectures as classification but with n_classes=n_outputs.
enum TrainedRegressionModel {
    InceptionTimePlus(InceptionTimePlus<InferBackend>),
    OmniScaleCNN(OmniScaleCNN<InferBackend>),
    TSTPlus(TSTPlus<InferBackend>),
}

impl TrainedRegressionModel {
    /// Run inference on the trained model.
    fn forward(&self, x: Tensor<InferBackend, 3>) -> Tensor<InferBackend, 2> {
        match self {
            TrainedRegressionModel::InceptionTimePlus(m) => m.forward(x),
            TrainedRegressionModel::OmniScaleCNN(m) => m.forward(x),
            TrainedRegressionModel::TSTPlus(m) => m.forward(x),
        }
    }
}

/// Sklearn-like time series regressor.
///
/// This regressor provides a simple, high-level API for time series regression
/// that mirrors the Python tsai library. It handles model creation, training, and
/// inference internally using MSE loss.
///
/// # Supported Architectures
///
/// - `InceptionTimePlus` (default) - InceptionTime with improvements
/// - `OmniScaleCNN` - Multi-scale CNN
/// - `TSTPlus` - Time Series Transformer
///
/// # Example
///
/// ```rust,ignore
/// use tsai_train::compat::{TSRegressor, TSRegressorConfig};
/// use ndarray::{Array3, Array2};
///
/// let config = TSRegressorConfig::default();
/// let mut reg = TSRegressor::new(config);
///
/// // x: (n_samples, n_vars, seq_len), y: (n_samples, n_outputs)
/// reg.fit(&x_train, &y_train)?;
/// let predictions = reg.predict(&x_test)?;
/// ```
pub struct TSRegressor {
    config: TSRegressorConfig,
    trained_model: Option<TrainedRegressionModel>,
    n_outputs: usize,
    n_vars: usize,
    seq_len: usize,
    device: <InferBackend as Backend>::Device,
}

impl TSRegressor {
    /// Create a new regressor.
    pub fn new(config: TSRegressorConfig) -> Self {
        Self {
            config,
            trained_model: None,
            n_outputs: 1,
            n_vars: 0,
            seq_len: 0,
            device: Default::default(),
        }
    }

    /// Fit the regressor.
    ///
    /// # Arguments
    ///
    /// * `x` - Input data of shape (n_samples, n_vars, seq_len)
    /// * `y` - Target values of shape (n_samples, n_outputs)
    ///
    /// # Returns
    ///
    /// Training metrics including final loss.
    pub fn fit(&mut self, x: &Array3<f32>, y: &Array2<f32>) -> Result<RegressionMetrics> {
        let (_n_samples, n_vars, seq_len) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        self.n_vars = n_vars;
        self.seq_len = seq_len;
        self.n_outputs = y.shape()[1];

        // Create dataset
        let dataset = TSDataset::from_arrays(x.clone(), Some(y.clone()))?;

        // Split into train/valid
        let seed = Seed::new(self.config.seed);
        let (train_ds, valid_ds) = train_test_split(&dataset, self.config.valid_ratio, seed)?;

        // Create dataloaders
        let dls = TSDataLoaders::builder(train_ds, valid_ds)
            .batch_size(self.config.batch_size)
            .seed(seed)
            .build()?;

        // Train based on architecture
        let metrics = match self.config.arch.as_str() {
            "InceptionTimePlus" | "inception" | "inception_time" => {
                self.train_inception_time(&dls)?
            }
            "OmniScaleCNN" | "omniscale" => {
                self.train_omniscale(&dls)?
            }
            "TSTPlus" | "tst" | "transformer" => {
                self.train_tst(&dls)?
            }
            other => {
                return Err(TrainError::Other(format!(
                    "Unknown architecture '{}'. Supported: InceptionTimePlus, OmniScaleCNN, TSTPlus",
                    other
                )));
            }
        };

        Ok(metrics)
    }

    fn train_inception_time(&mut self, dls: &TSDataLoaders) -> Result<RegressionMetrics> {
        let train_device: <TrainBackend as Backend>::Device = Default::default();

        // Create model config - use n_outputs instead of n_classes
        let model_config = InceptionTimePlusConfig {
            n_vars: self.n_vars,
            seq_len: self.seq_len,
            n_classes: self.n_outputs, // For regression, this is n_outputs
            n_blocks: 6,
            n_filters: 32,
            kernel_sizes: [9, 19, 39],
            bottleneck_dim: 32,
            dropout: 0.0,
        };

        // Initialize model for training
        let model: InceptionTimePlus<TrainBackend> = model_config.init(&train_device);

        // Configure trainer
        let trainer_config = RegressionTrainerConfig {
            n_epochs: self.config.n_epochs,
            lr: self.config.lr,
            weight_decay: 0.01,
            grad_clip: 1.0,
            verbose: true,
            early_stopping_patience: 0,
            early_stopping_min_delta: 0.0001,
        };

        let trainer = RegressionTrainer::<TrainBackend>::new(trainer_config, train_device);

        // Train with forward functions
        let output = trainer.fit_with_forward(
            model,
            dls,
            |m, x| m.forward(x),
            |m, x| m.forward(x),
        )?;

        // Store the trained model
        let inner_model = output.model.valid();
        self.trained_model = Some(TrainedRegressionModel::InceptionTimePlus(inner_model));

        Ok(RegressionMetrics {
            train_losses: output.train_losses,
            valid_losses: output.valid_losses,
            best_valid_loss: output.best_valid_loss,
            best_epoch: output.best_epoch,
            training_time_secs: output.training_time_secs,
        })
    }

    fn train_omniscale(&mut self, dls: &TSDataLoaders) -> Result<RegressionMetrics> {
        let train_device: <TrainBackend as Backend>::Device = Default::default();

        // Create model config
        let model_config = OmniScaleCNNConfig::new(self.n_vars, self.seq_len, self.n_outputs)
            .with_n_filters(64)
            .with_dropout(0.1);

        // Initialize model for training
        let model: OmniScaleCNN<TrainBackend> = model_config.init(&train_device);

        // Configure trainer
        let trainer_config = RegressionTrainerConfig {
            n_epochs: self.config.n_epochs,
            lr: self.config.lr,
            weight_decay: 0.01,
            grad_clip: 1.0,
            verbose: true,
            early_stopping_patience: 0,
            early_stopping_min_delta: 0.0001,
        };

        let trainer = RegressionTrainer::<TrainBackend>::new(trainer_config, train_device);

        // Train with forward functions
        let output = trainer.fit_with_forward(
            model,
            dls,
            |m, x| m.forward(x),
            |m, x| m.forward(x),
        )?;

        // Store the trained model
        let inner_model = output.model.valid();
        self.trained_model = Some(TrainedRegressionModel::OmniScaleCNN(inner_model));

        Ok(RegressionMetrics {
            train_losses: output.train_losses,
            valid_losses: output.valid_losses,
            best_valid_loss: output.best_valid_loss,
            best_epoch: output.best_epoch,
            training_time_secs: output.training_time_secs,
        })
    }

    fn train_tst(&mut self, dls: &TSDataLoaders) -> Result<RegressionMetrics> {
        let train_device: <TrainBackend as Backend>::Device = Default::default();

        // Create model config
        let d_model = 64;
        let model_config = TSTConfig {
            n_vars: self.n_vars,
            seq_len: self.seq_len,
            n_classes: self.n_outputs, // For regression, this is n_outputs
            d_model,
            n_heads: 4,
            n_layers: 3,
            d_ff: d_model * 4,
            dropout: 0.1,
            use_pe: true,
        };

        // Initialize model for training
        let model: TSTPlus<TrainBackend> = model_config.init(&train_device);

        // Configure trainer
        let trainer_config = RegressionTrainerConfig {
            n_epochs: self.config.n_epochs,
            lr: self.config.lr,
            weight_decay: 0.01,
            grad_clip: 1.0,
            verbose: true,
            early_stopping_patience: 0,
            early_stopping_min_delta: 0.0001,
        };

        let trainer = RegressionTrainer::<TrainBackend>::new(trainer_config, train_device);

        // Train with forward functions
        let output = trainer.fit_with_forward(
            model,
            dls,
            |m, x| m.forward(x),
            |m, x| m.forward(x),
        )?;

        // Store the trained model
        let inner_model = output.model.valid();
        self.trained_model = Some(TrainedRegressionModel::TSTPlus(inner_model));

        Ok(RegressionMetrics {
            train_losses: output.train_losses,
            valid_losses: output.valid_losses,
            best_valid_loss: output.best_valid_loss,
            best_epoch: output.best_epoch,
            training_time_secs: output.training_time_secs,
        })
    }

    /// Predict values.
    ///
    /// # Arguments
    ///
    /// * `x` - Input data of shape (n_samples, n_vars, seq_len)
    ///
    /// # Returns
    ///
    /// Predictions of shape (n_samples, n_outputs)
    pub fn predict(&self, x: &Array3<f32>) -> Result<Array2<f32>> {
        let model = self.trained_model.as_ref().ok_or_else(|| {
            TrainError::Other("Model not fitted. Call fit() first.".to_string())
        })?;

        let (n_samples, n_vars, seq_len) = (x.shape()[0], x.shape()[1], x.shape()[2]);

        // Convert ndarray to Burn tensor
        let data: Vec<f32> = x.iter().copied().collect();
        let tensor_data = burn::tensor::TensorData::new(data, [n_samples, n_vars, seq_len]);
        let tensor: Tensor<InferBackend, 3> = Tensor::from_data(tensor_data, &self.device);

        // Run inference
        let preds = model.forward(tensor);

        // Convert back to ndarray
        let preds_data: Vec<f32> = preds.into_data().to_vec().unwrap();
        let result = Array2::from_shape_vec((n_samples, self.n_outputs), preds_data)
            .map_err(|e| TrainError::Other(e.to_string()))?;

        Ok(result)
    }

    /// Get the config.
    pub fn config(&self) -> &TSRegressorConfig {
        &self.config
    }

    /// Check if fitted.
    pub fn is_fitted(&self) -> bool {
        self.trained_model.is_some()
    }

    /// Get the number of outputs.
    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }
}

/// Regression training metrics returned after fitting.
#[derive(Debug, Clone)]
pub struct RegressionMetrics {
    /// Training losses (MSE) per epoch.
    pub train_losses: Vec<f32>,
    /// Validation losses (MSE) per epoch.
    pub valid_losses: Vec<f32>,
    /// Best validation loss achieved.
    pub best_valid_loss: f32,
    /// Epoch with best validation loss.
    pub best_epoch: usize,
    /// Total training time in seconds.
    pub training_time_secs: f64,
}

/// Configuration for sklearn-like forecasters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TSForecasterConfig {
    /// Model architecture name.
    pub arch: String,
    /// Forecast horizon.
    pub horizon: usize,
    /// Number of epochs.
    pub n_epochs: usize,
    /// Learning rate.
    pub lr: f64,
    /// Batch size.
    pub batch_size: usize,
    /// Validation ratio.
    pub valid_ratio: f32,
    /// Random seed.
    pub seed: u64,
}

impl Default for TSForecasterConfig {
    fn default() -> Self {
        Self {
            arch: "PatchTST".to_string(),
            horizon: 24,
            n_epochs: 25,
            lr: 1e-3,
            batch_size: 64,
            valid_ratio: 0.2,
            seed: 42,
        }
    }
}

/// Trained forecasting model storage for different architectures.
///
/// Uses the same model architectures but with n_classes=horizon for forecasting.
enum TrainedForecastModel {
    InceptionTimePlus(InceptionTimePlus<InferBackend>),
    OmniScaleCNN(OmniScaleCNN<InferBackend>),
    TSTPlus(TSTPlus<InferBackend>),
}

impl TrainedForecastModel {
    /// Run inference on the trained model.
    fn forward(&self, x: Tensor<InferBackend, 3>) -> Tensor<InferBackend, 2> {
        match self {
            TrainedForecastModel::InceptionTimePlus(m) => m.forward(x),
            TrainedForecastModel::OmniScaleCNN(m) => m.forward(x),
            TrainedForecastModel::TSTPlus(m) => m.forward(x),
        }
    }
}

/// Sklearn-like time series forecaster.
///
/// This forecaster provides a simple, high-level API for time series forecasting
/// that mirrors the Python tsai library. It predicts future values given historical
/// data using MSE loss for training.
///
/// # Supported Architectures
///
/// - `InceptionTimePlus` (default) - InceptionTime with improvements
/// - `OmniScaleCNN` - Multi-scale CNN
/// - `TSTPlus` - Time Series Transformer
///
/// # Example
///
/// ```rust,ignore
/// use tsai_train::compat::{TSForecaster, TSForecasterConfig};
/// use ndarray::{Array3, Array2};
///
/// let config = TSForecasterConfig {
///     horizon: 24,
///     ..Default::default()
/// };
/// let mut forecaster = TSForecaster::new(config);
///
/// // x: (n_samples, n_vars, seq_len), y: (n_samples, horizon)
/// forecaster.fit(&x_train, &y_train)?;
/// let predictions = forecaster.predict(&x_test)?;
/// ```
pub struct TSForecaster {
    config: TSForecasterConfig,
    trained_model: Option<TrainedForecastModel>,
    n_vars: usize,
    seq_len: usize,
    device: <InferBackend as Backend>::Device,
}

impl TSForecaster {
    /// Create a new forecaster.
    pub fn new(config: TSForecasterConfig) -> Self {
        Self {
            config,
            trained_model: None,
            n_vars: 0,
            seq_len: 0,
            device: Default::default(),
        }
    }

    /// Fit the forecaster.
    ///
    /// # Arguments
    ///
    /// * `x` - Input data of shape (n_samples, n_vars, seq_len)
    /// * `y` - Future values of shape (n_samples, horizon)
    ///
    /// # Returns
    ///
    /// Training metrics including final loss.
    pub fn fit(&mut self, x: &Array3<f32>, y: &Array2<f32>) -> Result<ForecastMetrics> {
        let (_n_samples, n_vars, seq_len) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        self.n_vars = n_vars;
        self.seq_len = seq_len;

        // Validate horizon matches y shape
        let horizon = y.shape()[1];
        if horizon != self.config.horizon {
            return Err(TrainError::Other(format!(
                "Target shape {} doesn't match configured horizon {}",
                horizon, self.config.horizon
            )));
        }

        // Create dataset
        let dataset = TSDataset::from_arrays(x.clone(), Some(y.clone()))?;

        // Split into train/valid
        let seed = Seed::new(self.config.seed);
        let (train_ds, valid_ds) = train_test_split(&dataset, self.config.valid_ratio, seed)?;

        // Create dataloaders
        let dls = TSDataLoaders::builder(train_ds, valid_ds)
            .batch_size(self.config.batch_size)
            .seed(seed)
            .build()?;

        // Train based on architecture
        let metrics = match self.config.arch.as_str() {
            "InceptionTimePlus" | "inception" | "inception_time" => {
                self.train_inception_time(&dls)?
            }
            "OmniScaleCNN" | "omniscale" => {
                self.train_omniscale(&dls)?
            }
            "TSTPlus" | "tst" | "transformer" => {
                self.train_tst(&dls)?
            }
            other => {
                return Err(TrainError::Other(format!(
                    "Unknown architecture '{}'. Supported: InceptionTimePlus, OmniScaleCNN, TSTPlus",
                    other
                )));
            }
        };

        Ok(metrics)
    }

    fn train_inception_time(&mut self, dls: &TSDataLoaders) -> Result<ForecastMetrics> {
        let train_device: <TrainBackend as Backend>::Device = Default::default();

        // Create model config - use horizon as output dimension
        let model_config = InceptionTimePlusConfig {
            n_vars: self.n_vars,
            seq_len: self.seq_len,
            n_classes: self.config.horizon,
            n_blocks: 6,
            n_filters: 32,
            kernel_sizes: [9, 19, 39],
            bottleneck_dim: 32,
            dropout: 0.0,
        };

        // Initialize model for training
        let model: InceptionTimePlus<TrainBackend> = model_config.init(&train_device);

        // Configure trainer (use RegressionTrainer for MSE loss)
        let trainer_config = RegressionTrainerConfig {
            n_epochs: self.config.n_epochs,
            lr: self.config.lr,
            weight_decay: 0.01,
            grad_clip: 1.0,
            verbose: true,
            early_stopping_patience: 0,
            early_stopping_min_delta: 0.0001,
        };

        let trainer = RegressionTrainer::<TrainBackend>::new(trainer_config, train_device);

        // Train with forward functions
        let output = trainer.fit_with_forward(
            model,
            dls,
            |m, x| m.forward(x),
            |m, x| m.forward(x),
        )?;

        // Store the trained model
        let inner_model = output.model.valid();
        self.trained_model = Some(TrainedForecastModel::InceptionTimePlus(inner_model));

        Ok(ForecastMetrics {
            train_losses: output.train_losses,
            valid_losses: output.valid_losses,
            best_valid_loss: output.best_valid_loss,
            best_epoch: output.best_epoch,
            training_time_secs: output.training_time_secs,
        })
    }

    fn train_omniscale(&mut self, dls: &TSDataLoaders) -> Result<ForecastMetrics> {
        let train_device: <TrainBackend as Backend>::Device = Default::default();

        // Create model config
        let model_config = OmniScaleCNNConfig::new(self.n_vars, self.seq_len, self.config.horizon)
            .with_n_filters(64)
            .with_dropout(0.1);

        // Initialize model for training
        let model: OmniScaleCNN<TrainBackend> = model_config.init(&train_device);

        // Configure trainer
        let trainer_config = RegressionTrainerConfig {
            n_epochs: self.config.n_epochs,
            lr: self.config.lr,
            weight_decay: 0.01,
            grad_clip: 1.0,
            verbose: true,
            early_stopping_patience: 0,
            early_stopping_min_delta: 0.0001,
        };

        let trainer = RegressionTrainer::<TrainBackend>::new(trainer_config, train_device);

        // Train with forward functions
        let output = trainer.fit_with_forward(
            model,
            dls,
            |m, x| m.forward(x),
            |m, x| m.forward(x),
        )?;

        // Store the trained model
        let inner_model = output.model.valid();
        self.trained_model = Some(TrainedForecastModel::OmniScaleCNN(inner_model));

        Ok(ForecastMetrics {
            train_losses: output.train_losses,
            valid_losses: output.valid_losses,
            best_valid_loss: output.best_valid_loss,
            best_epoch: output.best_epoch,
            training_time_secs: output.training_time_secs,
        })
    }

    fn train_tst(&mut self, dls: &TSDataLoaders) -> Result<ForecastMetrics> {
        let train_device: <TrainBackend as Backend>::Device = Default::default();

        // Create model config
        let d_model = 64;
        let model_config = TSTConfig {
            n_vars: self.n_vars,
            seq_len: self.seq_len,
            n_classes: self.config.horizon,
            d_model,
            n_heads: 4,
            n_layers: 3,
            d_ff: d_model * 4,
            dropout: 0.1,
            use_pe: true,
        };

        // Initialize model for training
        let model: TSTPlus<TrainBackend> = model_config.init(&train_device);

        // Configure trainer
        let trainer_config = RegressionTrainerConfig {
            n_epochs: self.config.n_epochs,
            lr: self.config.lr,
            weight_decay: 0.01,
            grad_clip: 1.0,
            verbose: true,
            early_stopping_patience: 0,
            early_stopping_min_delta: 0.0001,
        };

        let trainer = RegressionTrainer::<TrainBackend>::new(trainer_config, train_device);

        // Train with forward functions
        let output = trainer.fit_with_forward(
            model,
            dls,
            |m, x| m.forward(x),
            |m, x| m.forward(x),
        )?;

        // Store the trained model
        let inner_model = output.model.valid();
        self.trained_model = Some(TrainedForecastModel::TSTPlus(inner_model));

        Ok(ForecastMetrics {
            train_losses: output.train_losses,
            valid_losses: output.valid_losses,
            best_valid_loss: output.best_valid_loss,
            best_epoch: output.best_epoch,
            training_time_secs: output.training_time_secs,
        })
    }

    /// Forecast future values.
    ///
    /// # Arguments
    ///
    /// * `x` - Input data of shape (n_samples, n_vars, seq_len)
    ///
    /// # Returns
    ///
    /// Predictions of shape (n_samples, horizon)
    pub fn predict(&self, x: &Array3<f32>) -> Result<Array2<f32>> {
        let model = self.trained_model.as_ref().ok_or_else(|| {
            TrainError::Other("Model not fitted. Call fit() first.".to_string())
        })?;

        let (n_samples, n_vars, seq_len) = (x.shape()[0], x.shape()[1], x.shape()[2]);

        // Convert ndarray to Burn tensor
        let data: Vec<f32> = x.iter().copied().collect();
        let tensor_data = burn::tensor::TensorData::new(data, [n_samples, n_vars, seq_len]);
        let tensor: Tensor<InferBackend, 3> = Tensor::from_data(tensor_data, &self.device);

        // Run inference
        let preds = model.forward(tensor);

        // Convert back to ndarray
        let preds_data: Vec<f32> = preds.into_data().to_vec().unwrap();
        let result = Array2::from_shape_vec((n_samples, self.config.horizon), preds_data)
            .map_err(|e| TrainError::Other(e.to_string()))?;

        Ok(result)
    }

    /// Get the config.
    pub fn config(&self) -> &TSForecasterConfig {
        &self.config
    }

    /// Get the forecast horizon.
    pub fn horizon(&self) -> usize {
        self.config.horizon
    }

    /// Check if fitted.
    pub fn is_fitted(&self) -> bool {
        self.trained_model.is_some()
    }
}

/// Forecasting training metrics returned after fitting.
#[derive(Debug, Clone)]
pub struct ForecastMetrics {
    /// Training losses (MSE) per epoch.
    pub train_losses: Vec<f32>,
    /// Validation losses (MSE) per epoch.
    pub valid_losses: Vec<f32>,
    /// Best validation loss achieved.
    pub best_valid_loss: f32,
    /// Epoch with best validation loss.
    pub best_epoch: usize,
    /// Total training time in seconds.
    pub training_time_secs: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classifier_config() {
        let config = TSClassifierConfig::default();
        assert_eq!(config.arch, "InceptionTimePlus");
        assert_eq!(config.n_epochs, 25);
    }

    #[test]
    fn test_regressor_config() {
        let config = TSRegressorConfig::default();
        assert_eq!(config.batch_size, 64);
    }

    #[test]
    fn test_forecaster_config() {
        let config = TSForecasterConfig::default();
        assert_eq!(config.horizon, 24);
        assert_eq!(config.arch, "PatchTST");
    }
}
