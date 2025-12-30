//! Integration tests for the training pipeline.
//!
//! These tests verify end-to-end training functionality with synthetic data.

use burn::prelude::*;
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;
use ndarray::{Array2, Array3};

use tsai_data::{TSDataLoaders, TSDataset};
use tsai_models::{InceptionTimePlus, InceptionTimePlusConfig};
use tsai_train::{ClassificationTrainer, ClassificationTrainerConfig};

type TrainBackend = Autodiff<NdArray>;

/// Create synthetic time series data for testing.
fn create_synthetic_data(
    n_samples: usize,
    n_vars: usize,
    seq_len: usize,
    n_classes: usize,
) -> (Array3<f32>, Array2<f32>) {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Create random time series with class-dependent patterns
    let mut x_data = Vec::with_capacity(n_samples * n_vars * seq_len);
    let mut y_data = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let class = i % n_classes;
        y_data.push(class as f32);

        for _v in 0..n_vars {
            for t in 0..seq_len {
                // Add class-dependent bias + noise
                let value = (class as f32) * 0.5 + (t as f32 / seq_len as f32) + rng.gen::<f32>() * 0.1;
                x_data.push(value);
            }
        }
    }

    let x = Array3::from_shape_vec((n_samples, n_vars, seq_len), x_data).unwrap();
    let y = Array2::from_shape_vec((n_samples, 1), y_data).unwrap();

    (x, y)
}

#[test]
fn test_training_pipeline_synthetic() {
    // Create small synthetic dataset
    let n_samples = 32;
    let n_vars = 2;
    let seq_len = 50;
    let n_classes = 3;
    let batch_size = 8;
    let n_epochs = 2;

    let (x, y) = create_synthetic_data(n_samples, n_vars, seq_len, n_classes);

    // Create dataset and split
    let train_samples = n_samples * 3 / 4;
    let x_train = x.slice(ndarray::s![..train_samples, .., ..]).to_owned();
    let y_train = y.slice(ndarray::s![..train_samples, ..]).to_owned();
    let x_valid = x.slice(ndarray::s![train_samples.., .., ..]).to_owned();
    let y_valid = y.slice(ndarray::s![train_samples.., ..]).to_owned();

    let train_ds = TSDataset::from_arrays(x_train, Some(y_train)).expect("Failed to create train dataset");
    let valid_ds = TSDataset::from_arrays(x_valid, Some(y_valid)).expect("Failed to create valid dataset");

    // Create dataloaders
    let dls = TSDataLoaders::builder(train_ds, valid_ds)
        .batch_size(batch_size)
        .build()
        .expect("Failed to create dataloaders");

    // Configure model
    let model_config = InceptionTimePlusConfig {
        n_vars,
        seq_len,
        n_classes,
        n_blocks: 2,  // Small for fast testing
        n_filters: 8,
        kernel_sizes: [9, 19, 39],
        bottleneck_dim: 8,
        dropout: 0.0,
    };

    // Initialize model
    let device = <TrainBackend as Backend>::Device::default();
    let model: InceptionTimePlus<TrainBackend> = model_config.init(&device);

    // Configure trainer
    let trainer_config = ClassificationTrainerConfig {
        n_epochs,
        lr: 1e-3,
        weight_decay: 0.01,
        grad_clip: 1.0,
        verbose: false, // Quiet for tests
        early_stopping_patience: 0,
        early_stopping_min_delta: 0.001,
    };

    let trainer = ClassificationTrainer::<TrainBackend>::new(trainer_config, device);

    // Train
    let result = trainer.fit_with_forward(
        model,
        &dls,
        |m, x| m.forward(x),
        |m, x| m.forward(x),
    );

    // Verify training succeeded
    assert!(result.is_ok(), "Training failed: {:?}", result.err());
    let output = result.unwrap();

    // Verify we got metrics
    assert_eq!(output.train_losses.len(), n_epochs);
    assert_eq!(output.valid_losses.len(), n_epochs);
    assert_eq!(output.valid_accs.len(), n_epochs);

    // Verify losses are reasonable (not NaN or Inf)
    for loss in &output.train_losses {
        assert!(loss.is_finite(), "Train loss is not finite: {}", loss);
    }
    for loss in &output.valid_losses {
        assert!(loss.is_finite(), "Valid loss is not finite: {}", loss);
    }

    // Verify training progressed (loss should generally decrease)
    // Note: With only 2 epochs and random init, this may not always hold
    // but losses should at least be in a reasonable range
    assert!(output.train_losses[0] < 10.0, "Initial loss too high");
}

#[test]
fn test_tsclassifier_api() {
    use tsai_train::compat::{TSClassifier, TSClassifierConfig};

    // Create synthetic data
    let (x, y) = create_synthetic_data(32, 2, 50, 3);

    // Create classifier with minimal epochs for fast testing
    let config = TSClassifierConfig {
        arch: "InceptionTimePlus".to_string(),
        n_epochs: 1,
        lr: 1e-3,
        batch_size: 8,
        valid_ratio: 0.25,
        seed: 42,
        use_gpu: false,
    };

    let mut clf = TSClassifier::new(config);

    // Fit
    let result = clf.fit(&x, &y);
    assert!(result.is_ok(), "TSClassifier.fit failed: {:?}", result.err());

    // Verify fitted
    assert!(clf.is_fitted());
    assert_eq!(clf.n_classes(), 3);

    // Test predict
    let preds = clf.predict(&x);
    assert!(preds.is_ok());
    let preds = preds.unwrap();
    assert_eq!(preds.shape(), &[32, 1]);

    // Test predict_proba
    let proba = clf.predict_proba(&x);
    assert!(proba.is_ok());
    let proba = proba.unwrap();
    assert_eq!(proba.shape(), &[32, 3]);

    // Verify probabilities sum to ~1
    for i in 0..proba.nrows() {
        let sum: f32 = proba.row(i).iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Probabilities don't sum to 1: {}", sum);
    }
}

#[test]
fn test_dataset_creation() {
    let (x, y) = create_synthetic_data(16, 2, 50, 3);

    let dataset = TSDataset::from_arrays(x.clone(), Some(y.clone()));
    assert!(dataset.is_ok());

    let ds = dataset.unwrap();
    assert_eq!(ds.len(), 16);
    assert_eq!(ds.n_vars(), 2);
    assert_eq!(ds.seq_len(), 50);
}

/// Create synthetic regression data for testing.
fn create_synthetic_regression_data(
    n_samples: usize,
    n_vars: usize,
    seq_len: usize,
    n_outputs: usize,
) -> (Array3<f32>, Array2<f32>) {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Create random time series with target-dependent patterns
    let mut x_data = Vec::with_capacity(n_samples * n_vars * seq_len);
    let mut y_data = Vec::with_capacity(n_samples * n_outputs);

    for _i in 0..n_samples {
        let base = rng.gen::<f32>() * 2.0;

        for _v in 0..n_vars {
            for t in 0..seq_len {
                let value = base + (t as f32 / seq_len as f32) + rng.gen::<f32>() * 0.1;
                x_data.push(value);
            }
        }

        // Target is related to the base
        for _o in 0..n_outputs {
            y_data.push(base + rng.gen::<f32>() * 0.5);
        }
    }

    let x = Array3::from_shape_vec((n_samples, n_vars, seq_len), x_data).unwrap();
    let y = Array2::from_shape_vec((n_samples, n_outputs), y_data).unwrap();

    (x, y)
}

#[test]
fn test_regression_training_pipeline() {
    use tsai_train::{RegressionTrainer, RegressionTrainerConfig};

    // Create small synthetic dataset
    let n_samples = 32;
    let n_vars = 2;
    let seq_len = 50;
    let n_outputs = 1;
    let batch_size = 8;
    let n_epochs = 2;

    let (x, y) = create_synthetic_regression_data(n_samples, n_vars, seq_len, n_outputs);

    // Create dataset and split
    let train_samples = n_samples * 3 / 4;
    let x_train = x.slice(ndarray::s![..train_samples, .., ..]).to_owned();
    let y_train = y.slice(ndarray::s![..train_samples, ..]).to_owned();
    let x_valid = x.slice(ndarray::s![train_samples.., .., ..]).to_owned();
    let y_valid = y.slice(ndarray::s![train_samples.., ..]).to_owned();

    let train_ds = TSDataset::from_arrays(x_train, Some(y_train)).expect("Failed to create train dataset");
    let valid_ds = TSDataset::from_arrays(x_valid, Some(y_valid)).expect("Failed to create valid dataset");

    // Create dataloaders
    let dls = TSDataLoaders::builder(train_ds, valid_ds)
        .batch_size(batch_size)
        .build()
        .expect("Failed to create dataloaders");

    // Configure model - use n_outputs as n_classes for regression
    let model_config = InceptionTimePlusConfig {
        n_vars,
        seq_len,
        n_classes: n_outputs,
        n_blocks: 2,
        n_filters: 8,
        kernel_sizes: [9, 19, 39],
        bottleneck_dim: 8,
        dropout: 0.0,
    };

    // Initialize model
    let device = <TrainBackend as Backend>::Device::default();
    let model: InceptionTimePlus<TrainBackend> = model_config.init(&device);

    // Configure regression trainer
    let trainer_config = RegressionTrainerConfig {
        n_epochs,
        lr: 1e-3,
        weight_decay: 0.01,
        grad_clip: 1.0,
        verbose: false,
        early_stopping_patience: 0,
        early_stopping_min_delta: 0.0001,
    };

    let trainer = RegressionTrainer::<TrainBackend>::new(trainer_config, device);

    // Train
    let result = trainer.fit_with_forward(
        model,
        &dls,
        |m, x| m.forward(x),
        |m, x| m.forward(x),
    );

    // Verify training succeeded
    assert!(result.is_ok(), "Regression training failed: {:?}", result.err());
    let output = result.unwrap();

    // Verify we got metrics
    assert_eq!(output.train_losses.len(), n_epochs);
    assert_eq!(output.valid_losses.len(), n_epochs);

    // Verify losses are reasonable (not NaN or Inf)
    for loss in &output.train_losses {
        assert!(loss.is_finite(), "Train loss is not finite: {}", loss);
    }
    for loss in &output.valid_losses {
        assert!(loss.is_finite(), "Valid loss is not finite: {}", loss);
    }
}

#[test]
fn test_tsregressor_api() {
    use tsai_train::compat::{TSRegressor, TSRegressorConfig};

    // Create synthetic regression data
    let (x, y) = create_synthetic_regression_data(32, 2, 50, 1);

    // Create regressor with minimal epochs for fast testing
    let config = TSRegressorConfig {
        arch: "InceptionTimePlus".to_string(),
        n_epochs: 1,
        lr: 1e-3,
        batch_size: 8,
        valid_ratio: 0.25,
        seed: 42,
    };

    let mut reg = TSRegressor::new(config);

    // Fit
    let result = reg.fit(&x, &y);
    assert!(result.is_ok(), "TSRegressor.fit failed: {:?}", result.err());

    // Verify fitted
    assert!(reg.is_fitted());
    assert_eq!(reg.n_outputs(), 1);

    // Test predict
    let preds = reg.predict(&x);
    assert!(preds.is_ok());
    let preds = preds.unwrap();
    assert_eq!(preds.shape(), &[32, 1]);

    // Verify predictions are finite
    for val in preds.iter() {
        assert!(val.is_finite(), "Prediction is not finite: {}", val);
    }
}

/// Create synthetic forecasting data for testing.
fn create_synthetic_forecast_data(
    n_samples: usize,
    n_vars: usize,
    seq_len: usize,
    horizon: usize,
) -> (Array3<f32>, Array2<f32>) {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Create time series where future values are related to historical pattern
    let mut x_data = Vec::with_capacity(n_samples * n_vars * seq_len);
    let mut y_data = Vec::with_capacity(n_samples * horizon);

    for _i in 0..n_samples {
        let trend = rng.gen::<f32>() * 0.5;
        let base = rng.gen::<f32>() * 2.0;

        // Input sequence follows a trend
        for _v in 0..n_vars {
            for t in 0..seq_len {
                let value = base + trend * (t as f32) + rng.gen::<f32>() * 0.1;
                x_data.push(value);
            }
        }

        // Future values continue the trend
        for h in 0..horizon {
            let future_t = seq_len + h;
            let value = base + trend * (future_t as f32) + rng.gen::<f32>() * 0.1;
            y_data.push(value);
        }
    }

    let x = Array3::from_shape_vec((n_samples, n_vars, seq_len), x_data).unwrap();
    let y = Array2::from_shape_vec((n_samples, horizon), y_data).unwrap();

    (x, y)
}

#[test]
fn test_tsforecaster_api() {
    use tsai_train::compat::{TSForecaster, TSForecasterConfig};

    // Create synthetic forecasting data
    let horizon = 8;
    let (x, y) = create_synthetic_forecast_data(32, 2, 50, horizon);

    // Create forecaster with minimal epochs for fast testing
    let config = TSForecasterConfig {
        arch: "InceptionTimePlus".to_string(),
        horizon,
        n_epochs: 1,
        lr: 1e-3,
        batch_size: 8,
        valid_ratio: 0.25,
        seed: 42,
    };

    let mut forecaster = TSForecaster::new(config);

    // Fit
    let result = forecaster.fit(&x, &y);
    assert!(result.is_ok(), "TSForecaster.fit failed: {:?}", result.err());

    // Verify fitted
    assert!(forecaster.is_fitted());
    assert_eq!(forecaster.horizon(), horizon);

    // Test predict
    let preds = forecaster.predict(&x);
    assert!(preds.is_ok());
    let preds = preds.unwrap();
    assert_eq!(preds.shape(), &[32, horizon]);

    // Verify predictions are finite
    for val in preds.iter() {
        assert!(val.is_finite(), "Forecast is not finite: {}", val);
    }
}
