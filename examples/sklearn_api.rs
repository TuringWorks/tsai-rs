//! Example demonstrating the sklearn-like API for classification, regression, and forecasting.
//!
//! This example shows how to use TSClassifier, TSRegressor, and TSForecaster
//! with a simple, high-level API similar to scikit-learn.
//!
//! Run with: cargo run --example sklearn_api --release

use ndarray::{Array2, Array3};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use tsai_train::compat::{
    TSClassifier, TSClassifierConfig,
    TSRegressor, TSRegressorConfig,
    TSForecaster, TSForecasterConfig,
};

/// Create synthetic classification data.
fn create_classification_data(
    n_samples: usize,
    n_vars: usize,
    seq_len: usize,
    n_classes: usize,
) -> (Array3<f32>, Array2<f32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let mut x_data = Vec::with_capacity(n_samples * n_vars * seq_len);
    let mut y_data = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let class = i % n_classes;
        y_data.push(class as f32);

        for _v in 0..n_vars {
            for t in 0..seq_len {
                // Class-dependent pattern
                let value = (class as f32) * 0.3 + (t as f32 / seq_len as f32) + rng.gen::<f32>() * 0.1;
                x_data.push(value);
            }
        }
    }

    let x = Array3::from_shape_vec((n_samples, n_vars, seq_len), x_data).unwrap();
    let y = Array2::from_shape_vec((n_samples, 1), y_data).unwrap();

    (x, y)
}

/// Create synthetic regression data.
fn create_regression_data(
    n_samples: usize,
    n_vars: usize,
    seq_len: usize,
) -> (Array3<f32>, Array2<f32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let mut x_data = Vec::with_capacity(n_samples * n_vars * seq_len);
    let mut y_data = Vec::with_capacity(n_samples);

    for _i in 0..n_samples {
        let base = rng.gen::<f32>() * 2.0;

        for _v in 0..n_vars {
            for t in 0..seq_len {
                let value = base + (t as f32 / seq_len as f32) + rng.gen::<f32>() * 0.1;
                x_data.push(value);
            }
        }

        // Target is related to base
        y_data.push(base + rng.gen::<f32>() * 0.5);
    }

    let x = Array3::from_shape_vec((n_samples, n_vars, seq_len), x_data).unwrap();
    let y = Array2::from_shape_vec((n_samples, 1), y_data).unwrap();

    (x, y)
}

/// Create synthetic forecasting data.
fn create_forecast_data(
    n_samples: usize,
    n_vars: usize,
    seq_len: usize,
    horizon: usize,
) -> (Array3<f32>, Array2<f32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let mut x_data = Vec::with_capacity(n_samples * n_vars * seq_len);
    let mut y_data = Vec::with_capacity(n_samples * horizon);

    for _i in 0..n_samples {
        let trend = rng.gen::<f32>() * 0.5;
        let base = rng.gen::<f32>() * 2.0;

        // Historical data
        for _v in 0..n_vars {
            for t in 0..seq_len {
                let value = base + trend * (t as f32) + rng.gen::<f32>() * 0.1;
                x_data.push(value);
            }
        }

        // Future values (continue the trend)
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

fn main() -> anyhow::Result<()> {
    println!("=== tsai-rs sklearn-like API Example ===\n");

    // =====================================================================
    // 1. Classification Example
    // =====================================================================
    println!("1. TSClassifier Example");
    println!("------------------------");

    let (x_clf, y_clf) = create_classification_data(100, 3, 50, 3);
    println!("Data shape: {:?} -> {:?}", x_clf.shape(), y_clf.shape());

    let config = TSClassifierConfig {
        arch: "InceptionTimePlus".to_string(),
        n_epochs: 3,
        lr: 1e-3,
        batch_size: 16,
        valid_ratio: 0.2,
        seed: 42,
        use_gpu: false,
    };

    let mut clf = TSClassifier::new(config);

    println!("Training classifier...");
    let metrics = clf.fit(&x_clf, &y_clf)?;
    println!("Training complete!");
    println!("  Best validation accuracy: {:.2}%", metrics.best_valid_acc * 100.0);
    println!("  Best epoch: {}", metrics.best_epoch + 1);
    println!("  Training time: {:.1}s", metrics.training_time_secs);

    // Make predictions
    let preds = clf.predict(&x_clf)?;
    let proba = clf.predict_proba(&x_clf)?;
    println!("  Prediction shape: {:?}", preds.shape());
    println!("  Probability shape: {:?}", proba.shape());
    println!();

    // =====================================================================
    // 2. Regression Example
    // =====================================================================
    println!("2. TSRegressor Example");
    println!("-----------------------");

    let (x_reg, y_reg) = create_regression_data(100, 3, 50);
    println!("Data shape: {:?} -> {:?}", x_reg.shape(), y_reg.shape());

    let config = TSRegressorConfig {
        arch: "InceptionTimePlus".to_string(),
        n_epochs: 3,
        lr: 1e-3,
        batch_size: 16,
        valid_ratio: 0.2,
        seed: 42,
    };

    let mut reg = TSRegressor::new(config);

    println!("Training regressor...");
    let metrics = reg.fit(&x_reg, &y_reg)?;
    println!("Training complete!");
    println!("  Best validation loss: {:.6}", metrics.best_valid_loss);
    println!("  Best epoch: {}", metrics.best_epoch + 1);
    println!("  Training time: {:.1}s", metrics.training_time_secs);

    // Make predictions
    let preds = reg.predict(&x_reg)?;
    println!("  Prediction shape: {:?}", preds.shape());
    println!();

    // =====================================================================
    // 3. Forecasting Example
    // =====================================================================
    println!("3. TSForecaster Example");
    println!("------------------------");

    let horizon = 12;
    let (x_fcst, y_fcst) = create_forecast_data(100, 3, 50, horizon);
    println!("Data shape: {:?} -> {:?}", x_fcst.shape(), y_fcst.shape());

    let config = TSForecasterConfig {
        arch: "InceptionTimePlus".to_string(),
        horizon,
        n_epochs: 3,
        lr: 1e-3,
        batch_size: 16,
        valid_ratio: 0.2,
        seed: 42,
    };

    let mut forecaster = TSForecaster::new(config);

    println!("Training forecaster...");
    let metrics = forecaster.fit(&x_fcst, &y_fcst)?;
    println!("Training complete!");
    println!("  Best validation loss: {:.6}", metrics.best_valid_loss);
    println!("  Best epoch: {}", metrics.best_epoch + 1);
    println!("  Training time: {:.1}s", metrics.training_time_secs);

    // Make predictions
    let preds = forecaster.predict(&x_fcst)?;
    println!("  Forecast shape: {:?} (horizon={})", preds.shape(), horizon);
    println!();

    println!("=== All examples completed successfully! ===");
    Ok(())
}
