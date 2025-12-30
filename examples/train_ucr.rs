//! Example: Train InceptionTimePlus on UCR Dataset
//!
//! This example demonstrates end-to-end training using tsai-rs:
//! - Loading UCR time series datasets
//! - Creating dataloaders with shuffling
//! - Training InceptionTimePlus with OneCycleLR
//! - Tracking metrics and best model
//!
//! Run with: cargo run --example train_ucr --release
//!
//! For a quick test, use a small dataset like ECG200:
//!   cargo run --example train_ucr --release -- --dataset ECG200 --epochs 5

use anyhow::{Context, Result};
use burn::prelude::*;
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;

use tsai_core::Seed;
use tsai_data::ucr::UCRDataset;
use tsai_data::TSDataLoaders;
use tsai_models::{InceptionTimePlus, InceptionTimePlusConfig};
use tsai_train::{ClassificationTrainer, ClassificationTrainerConfig, evaluate_classification};

/// Backend type for training (NdArray with autodiff).
type TrainBackend = Autodiff<NdArray>;

fn main() -> Result<()> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let dataset_name = args
        .iter()
        .position(|s| s == "--dataset")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("ECG200");

    let epochs: usize = args
        .iter()
        .position(|s| s == "--epochs")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(25);

    let batch_size: usize = args
        .iter()
        .position(|s| s == "--batch-size")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);

    let lr: f64 = args
        .iter()
        .position(|s| s == "--lr")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-3);

    let seed: u64 = args
        .iter()
        .position(|s| s == "--seed")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           tsai-rs: Time Series Classification                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // =========================================================================
    // Step 1: Load Dataset
    // =========================================================================
    println!("Step 1: Loading UCR Dataset");
    println!("─────────────────────────────────────────────────────────────────");

    println!("  Dataset: {}", dataset_name);
    println!("  (First run will download from timeseriesclassification.com)");
    println!();

    let ucr = UCRDataset::load(dataset_name, None)
        .context(format!("Failed to load dataset '{}'", dataset_name))?;

    println!("  ✓ Dataset loaded successfully!");
    println!("    Train samples: {}", ucr.train.len());
    println!("    Test samples:  {}", ucr.test.len());
    println!("    Sequence length: {}", ucr.seq_len);
    println!("    Classes: {}", ucr.n_classes);
    println!("    Variables: {}", ucr.train.n_vars());
    println!();

    // =========================================================================
    // Step 2: Create DataLoaders
    // =========================================================================
    println!("Step 2: Creating DataLoaders");
    println!("─────────────────────────────────────────────────────────────────");

    let dls = TSDataLoaders::builder(ucr.train.clone(), ucr.test.clone())
        .batch_size(batch_size)
        .shuffle_train(true)
        .seed(Seed::new(seed))
        .build()
        .context("Failed to create dataloaders")?;

    println!("  Batch size: {}", batch_size);
    println!("  Train batches: {}", dls.train().n_batches());
    println!("  Valid batches: {}", dls.valid().n_batches());
    println!("  Seed: {}", seed);
    println!();

    // =========================================================================
    // Step 3: Configure Model
    // =========================================================================
    println!("Step 3: Configuring InceptionTimePlus Model");
    println!("─────────────────────────────────────────────────────────────────");

    let model_config = InceptionTimePlusConfig {
        n_vars: ucr.train.n_vars(),
        seq_len: ucr.seq_len,
        n_classes: ucr.n_classes,
        n_blocks: 6,
        n_filters: 32,
        kernel_sizes: [9, 19, 39],
        bottleneck_dim: 32,
        dropout: 0.0,
    };

    println!("  Architecture: InceptionTimePlus");
    println!("  Inception blocks: {}", model_config.n_blocks);
    println!("  Filters per branch: {}", model_config.n_filters);
    println!("  Kernel sizes: {:?}", model_config.kernel_sizes);
    println!("  Bottleneck dim: {}", model_config.bottleneck_dim);

    // Count parameters (approximate)
    let total_filters = model_config.n_filters * 4; // 4 branches per block
    let approx_params = model_config.n_blocks * total_filters * 100; // rough estimate
    println!("  Estimated parameters: ~{}K", approx_params / 1000);
    println!();

    // =========================================================================
    // Step 4: Initialize Model and Trainer
    // =========================================================================
    println!("Step 4: Initializing Training");
    println!("─────────────────────────────────────────────────────────────────");

    let device = <TrainBackend as Backend>::Device::default();
    let model: InceptionTimePlus<TrainBackend> = InceptionTimePlus::new(model_config, &device);

    let trainer_config = ClassificationTrainerConfig {
        n_epochs: epochs,
        lr,
        weight_decay: 0.01,
        grad_clip: 1.0,
        verbose: true,
        early_stopping_patience: 5, // Stop if no improvement for 5 epochs
        early_stopping_min_delta: 0.001,
    };

    println!("  Epochs: {}", epochs);
    println!("  Learning rate: {}", lr);
    println!("  Weight decay: {}", trainer_config.weight_decay);
    println!("  Gradient clipping: {}", trainer_config.grad_clip);
    println!("  Early stopping: patience={}", trainer_config.early_stopping_patience);
    println!("  Optimizer: AdamW");
    println!("  Scheduler: OneCycleLR");
    println!();

    let trainer = ClassificationTrainer::<TrainBackend>::new(trainer_config, device);

    // =========================================================================
    // Step 5: Train!
    // =========================================================================
    println!("Step 5: Training");
    println!("─────────────────────────────────────────────────────────────────");
    println!();

    let result = trainer
        .fit_with_forward(
            model,
            &dls,
            |model, x| model.forward(x),
            |model, x| model.forward(x),
        )
        .context("Training failed")?;

    // =========================================================================
    // Step 6: Results
    // =========================================================================
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                     Training Results                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Dataset: {}", dataset_name);
    println!("  Best validation accuracy: {:.2}%", result.best_valid_acc * 100.0);
    println!("  Best epoch: {}", result.best_epoch + 1);
    println!("  Total training time: {:.1}s", result.training_time_secs);
    println!();

    // Print loss curve summary
    println!("  Loss curve (first -> last):");
    if let (Some(first), Some(last)) = (result.train_losses.first(), result.train_losses.last()) {
        println!("    Train loss: {:.4} -> {:.4}", first, last);
    }
    if let (Some(first), Some(last)) = (result.valid_losses.first(), result.valid_losses.last()) {
        println!("    Valid loss: {:.4} -> {:.4}", first, last);
    }
    if let (Some(first), Some(last)) = (result.valid_accs.first(), result.valid_accs.last()) {
        println!("    Valid acc:  {:.2}% -> {:.2}%", first * 100.0, last * 100.0);
    }
    println!();

    // =========================================================================
    // Step 7: Evaluate and show confusion matrix
    // =========================================================================
    println!("Step 7: Final Evaluation on Test Set");
    println!("─────────────────────────────────────────────────────────────────");

    let eval_result = evaluate_classification::<TrainBackend, _, _>(
        &result.model,
        &dls,
        |model, x| model.forward(x),
    ).context("Evaluation failed")?;

    eval_result.print_summary();
    eval_result.print_confusion_matrix(ucr.n_classes);

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    Training Complete!                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    Ok(())
}
