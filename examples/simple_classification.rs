//! Example: Simple Time Series Classification
//!
//! A minimal example showing how to classify time series using tsai-rs.
//! This demonstrates the core workflow without extensive configuration.
//!
//! Run with: cargo run --example simple_classification

use ndarray::{Array2, Array3};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use tsai_core::Seed;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Simple Time Series Classification ===\n");

    // Step 1: Generate synthetic data
    // In practice, you'd load real data from files
    let (x_train, y_train) = generate_sine_waves(100, 50, 3, 42);
    let (x_test, y_test) = generate_sine_waves(20, 50, 3, 123);

    println!("Training data: {} samples, {} timesteps, {} classes",
             x_train.shape()[0], x_train.shape()[2], 3);
    println!("Test data: {} samples\n", x_test.shape()[0]);

    // Step 2: Create datasets
    let train_dataset = tsai_data::TSDataset::from_arrays(x_train, Some(y_train))?;
    let test_dataset = tsai_data::TSDataset::from_arrays(x_test, Some(y_test))?;

    println!("Train dataset: {} samples", train_dataset.len());
    println!("Test dataset: {} samples\n", test_dataset.len());

    // Step 3: Create dataloaders
    let dls = tsai_data::TSDataLoaders::builder(train_dataset, test_dataset)
        .batch_size(16)
        .shuffle_train(true)
        .seed(Seed::new(42))
        .build()?;

    println!("Train batches: {}", dls.train().n_batches());
    println!("Valid batches: {}\n", dls.valid().n_batches());

    // Step 4: Configure model
    let config = tsai_models::InceptionTimePlusConfig {
        n_vars: 1,
        seq_len: 50,
        n_classes: 3,
        n_blocks: 3,         // Smaller model for this example
        n_filters: 16,
        kernel_sizes: [9, 19, 39],
        bottleneck_dim: 16,
        dropout: 0.1,
    };

    println!("Model: InceptionTimePlus");
    println!("  Blocks: {}, Filters: {}", config.n_blocks, config.n_filters);

    // Step 5: Configure training
    let train_config = tsai_train::LearnerConfig {
        lr: 1e-3,
        weight_decay: 0.01,
        grad_clip: 1.0,
        mixed_precision: false,
    };

    println!("\nTraining config:");
    println!("  Learning rate: {}", train_config.lr);
    println!("  Weight decay: {}\n", train_config.weight_decay);

    // Step 6: Show what training would look like
    println!("Training pipeline ready!");
    println!("In a full implementation, you would:");
    println!("  1. Initialize model: config.init(&device)");
    println!("  2. Create learner: Learner::new(model, dls, loss_fn, optimizer)");
    println!("  3. Train: learner.fit_one_cycle(epochs, lr)");
    println!("  4. Evaluate: learner.validate()");

    // Step 7: Demonstrate metrics
    println!("\nSimulating predictions...");
    let preds = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1];
    let targets = vec![0, 1, 2, 0, 1, 1, 0, 2, 2, 0, 1, 2, 1, 1, 2, 0, 0, 2, 0, 1];

    let cm = tsai_analysis::confusion_matrix(&preds, &targets, 3);
    println!("Accuracy: {:.1}%", cm.accuracy() * 100.0);
    println!("Macro F1: {:.4}", cm.macro_f1());

    println!("\n=== Example complete ===");
    Ok(())
}

/// Generate synthetic sine wave data with different frequencies per class.
fn generate_sine_waves(
    n_samples: usize,
    seq_len: usize,
    n_classes: usize,
    seed: u64,
) -> (Array3<f32>, Array2<f32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut x = Array3::<f32>::zeros((n_samples, 1, seq_len));
    let mut y = Array2::<f32>::zeros((n_samples, 1));

    for i in 0..n_samples {
        let class = i % n_classes;
        y[[i, 0]] = class as f32;

        // Different frequency for each class
        let freq = (class + 1) as f32 * 0.2;
        for t in 0..seq_len {
            let noise: f32 = rng.gen_range(-0.1..0.1);
            x[[i, 0, t]] = (freq * t as f32).sin() + noise;
        }
    }

    (x, y)
}
