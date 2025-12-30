//! Example: Time Series Forecasting with PatchTST
//!
//! This example demonstrates how to use tsai-rs for time series forecasting
//! using the PatchTST transformer model.
//!
//! Run with: cargo run --example forecasting

use ndarray::{Array2, Array3};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use tsai_core::Seed;
use tsai_train::Scheduler;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Time Series Forecasting with PatchTST ===\n");

    // Forecasting configuration
    let lookback = 96;        // Use 96 past observations
    let horizon = 24;         // Predict 24 future steps
    let n_vars = 3;           // Multivariate: 3 input channels
    let n_samples = 500;

    println!("Forecasting setup:");
    println!("  Lookback window: {} steps", lookback);
    println!("  Forecast horizon: {} steps", horizon);
    println!("  Variables: {}\n", n_vars);

    // Generate synthetic multivariate time series
    let (x_data, y_data) = generate_forecasting_data(n_samples, n_vars, lookback, horizon, 42);

    println!("Generated data:");
    println!("  X shape: {:?} (samples, vars, lookback)", x_data.shape());
    println!("  y shape: {:?} (samples, horizon)\n", y_data.shape());

    // Create dataset
    let dataset = tsai_data::TSDataset::from_arrays(x_data, Some(y_data))?;

    // Split into train/validation/test
    let (train_ds, temp_ds) = tsai_data::train_test_split(&dataset, 0.2, Seed::new(42))?;
    let (valid_ds, test_ds) = tsai_data::train_test_split(&temp_ds, 0.5, Seed::new(42))?;

    println!("Data splits:");
    println!("  Train: {} samples", train_ds.len());
    println!("  Valid: {} samples", valid_ds.len());
    println!("  Test: {} samples\n", test_ds.len());

    // Create dataloaders
    let train_dls = tsai_data::TSDataLoaders::builder(train_ds, valid_ds)
        .batch_size(32)
        .shuffle_train(true)
        .seed(Seed::new(42))
        .build()?;

    println!("Dataloaders created:");
    println!("  Train batches: {}", train_dls.train().n_batches());
    println!("  Valid batches: {}\n", train_dls.valid().n_batches());

    // Configure PatchTST model for forecasting
    let model_config = tsai_models::PatchTSTConfig::for_forecasting(n_vars, lookback, horizon);

    println!("PatchTST model configuration:");
    println!("  n_vars: {}", model_config.n_vars);
    println!("  seq_len: {}", model_config.seq_len);
    println!("  n_outputs (horizon): {}", model_config.n_outputs);
    println!("  patch_len: {}", model_config.patch_len);
    println!("  stride: {}", model_config.stride);
    println!("  n_patches: {}", model_config.n_patches());
    println!("  d_model: {}", model_config.d_model);
    println!("  n_heads: {}", model_config.n_heads);
    println!("  n_layers: {}", model_config.n_layers);
    println!("  dropout: {}\n", model_config.dropout);

    // Configure training
    let train_config = tsai_train::LearnerConfig {
        lr: 1e-4,           // Lower LR for transformers
        weight_decay: 0.05, // Regularization
        grad_clip: 1.0,
        mixed_precision: false,
    };

    // Setup MSE loss for regression
    println!("Loss function: MSE (Mean Squared Error)");
    println!("Metrics: MSE, MAE\n");

    // Show learning rate schedule
    let n_epochs = 50;
    let steps_per_epoch = train_dls.train().n_batches();
    let total_steps = n_epochs * steps_per_epoch;
    let scheduler = tsai_train::OneCycleLR::simple(train_config.lr, total_steps);

    println!("Training schedule:");
    println!("  Epochs: {}", n_epochs);
    println!("  Steps per epoch: {}", steps_per_epoch);
    println!("  Total steps: {}", total_steps);
    println!("  Learning rate schedule:");
    for epoch in [0, 10, 25, 40, 49] {
        let step = epoch * steps_per_epoch;
        println!("    Epoch {}: LR = {:.6}", epoch, scheduler.get_lr(step));
    }

    println!("\n=== Forecasting pipeline ready ===");
    println!("\nNext steps for full training:");
    println!("  1. Initialize model on device");
    println!("  2. Create Learner with MSE loss");
    println!("  3. Add callbacks (EarlyStopping, ModelCheckpoint)");
    println!("  4. Train with fit_one_cycle()");
    println!("  5. Generate forecasts on test set");
    println!("  6. Compute final metrics (MSE, MAE, MAPE)");

    Ok(())
}

/// Generate synthetic data for forecasting.
///
/// Creates multivariate time series with trend, seasonality, and noise,
/// then creates (input, target) pairs for supervised forecasting.
fn generate_forecasting_data(
    n_samples: usize,
    n_vars: usize,
    lookback: usize,
    horizon: usize,
    seed: u64,
) -> (Array3<f32>, Array2<f32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut x = Array3::<f32>::zeros((n_samples, n_vars, lookback));
    let mut y = Array2::<f32>::zeros((n_samples, horizon));

    for i in 0..n_samples {
        let base_t = i as f32;

        // Generate lookback window for each variable
        for v in 0..n_vars {
            let phase: f32 = rng.gen_range(0.0..std::f32::consts::PI);
            for t in 0..lookback {
                let time = base_t + t as f32;
                // Combine trend, seasonality, and noise
                let trend = time * 0.001;
                let seasonal = (time * 0.1 + phase).sin() * 0.5;
                let noise: f32 = rng.gen_range(-0.1..0.1);
                x[[i, v, t]] = trend + seasonal + noise;
            }
        }

        // Generate target (future values of first variable)
        for t in 0..horizon {
            let time = base_t + lookback as f32 + t as f32;
            let trend = time * 0.001;
            let seasonal = (time * 0.1).sin() * 0.5;
            let noise: f32 = rng.gen_range(-0.05..0.05);
            y[[i, t]] = trend + seasonal + noise;
        }
    }

    (x, y)
}
