//! Example: Train InceptionTimePlus on UCR Dataset with MLX (Apple Silicon)
//!
//! This example demonstrates GPU-accelerated training using tsai-rs with MLX backend:
//! - Loading UCR time series datasets
//! - Creating dataloaders with shuffling
//! - Training InceptionTimePlus with OneCycleLR on Apple Silicon GPU via MLX
//!
//! Run with (macOS only):
//!   cargo run --example train_ucr_mlx --release --features backend-mlx
//!
//! For a quick test, use a small dataset like ECG200:
//!   cargo run --example train_ucr_mlx --release --features backend-mlx -- --dataset ECG200 --epochs 10
//!
//! Note: This requires an Apple Silicon Mac (M1/M2/M3/M4).

#[cfg(all(target_os = "macos", feature = "backend-mlx"))]
use anyhow::{Context, Result};
#[cfg(all(target_os = "macos", feature = "backend-mlx"))]
use burn_autodiff::Autodiff;
#[cfg(all(target_os = "macos", feature = "backend-mlx"))]
use burn_mlx::{Mlx, MlxDevice};

#[cfg(all(target_os = "macos", feature = "backend-mlx"))]
use tsai_core::Seed;
#[cfg(all(target_os = "macos", feature = "backend-mlx"))]
use tsai_data::ucr::UCRDataset;
#[cfg(all(target_os = "macos", feature = "backend-mlx"))]
use tsai_data::TSDataLoaders;
#[cfg(all(target_os = "macos", feature = "backend-mlx"))]
use tsai_models::{InceptionTimePlus, InceptionTimePlusConfig};
#[cfg(all(target_os = "macos", feature = "backend-mlx"))]
use tsai_train::{ClassificationTrainer, ClassificationTrainerConfig, evaluate_classification};

/// Backend type for training (MLX with autodiff - native Apple Silicon).
#[cfg(all(target_os = "macos", feature = "backend-mlx"))]
type TrainBackend = Autodiff<Mlx>;

#[cfg(all(target_os = "macos", feature = "backend-mlx"))]
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

    println!("==============================================================");
    println!("     tsai-rs: Time Series Classification (MLX - Apple GPU)    ");
    println!("==============================================================");
    println!();

    // =========================================================================
    // Step 1: Load Dataset
    // =========================================================================
    println!("Step 1: Loading UCR Dataset");
    println!("-------------------------------------------------------------");

    println!("  Dataset: {}", dataset_name);
    println!("  (First run will download from timeseriesclassification.com)");
    println!();

    let ucr = UCRDataset::load(dataset_name, None)
        .context(format!("Failed to load dataset '{}'", dataset_name))?;

    println!("  Dataset loaded successfully!");
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
    println!("-------------------------------------------------------------");

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
    println!("-------------------------------------------------------------");

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
    // Step 4: Initialize Model and Trainer with MLX (Apple Silicon GPU)
    // =========================================================================
    println!("Step 4: Initializing Training (MLX - Apple Silicon)");
    println!("-------------------------------------------------------------");

    // Use default MLX device (GPU on Apple Silicon)
    let device = MlxDevice::default();
    println!("  Device: {:?}", device);
    println!("  Backend: MLX (Native Apple Silicon GPU)");
    println!("  Features: Unified Memory, Lazy Evaluation");

    let model: InceptionTimePlus<TrainBackend> = InceptionTimePlus::new(model_config, &device);

    let trainer_config = ClassificationTrainerConfig {
        n_epochs: epochs,
        lr,
        weight_decay: 0.01,
        grad_clip: 1.0,
        verbose: true,
        early_stopping_patience: 5,
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

    let trainer = ClassificationTrainer::<TrainBackend>::new(trainer_config, device.clone());

    // =========================================================================
    // Step 5: Train!
    // =========================================================================
    println!("Step 5: Training on MLX (Apple Silicon GPU)");
    println!("-------------------------------------------------------------");
    println!();

    let start_time = std::time::Instant::now();

    let result = trainer
        .fit_with_forward(
            model,
            &dls,
            |model, x| model.forward(x),
            |model, x| model.forward(x),
        )
        .context("Training failed")?;

    let total_time = start_time.elapsed();

    // =========================================================================
    // Step 6: Results
    // =========================================================================
    println!();
    println!("==============================================================");
    println!("                     Training Results                         ");
    println!("==============================================================");
    println!();
    println!("  Dataset: {}", dataset_name);
    println!("  Backend: MLX (Apple Silicon GPU)");
    println!("  Best validation accuracy: {:.2}%", result.best_valid_acc * 100.0);
    println!("  Best epoch: {}", result.best_epoch + 1);
    println!("  Total training time: {:.1}s", result.training_time_secs);
    println!("  Wall clock time: {:.1}s", total_time.as_secs_f64());
    println!("  Time per epoch: {:.2}s", result.training_time_secs / epochs as f64);
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
    println!("-------------------------------------------------------------");

    let eval_result = evaluate_classification::<TrainBackend, _, _>(
        &result.model,
        &dls,
        |model, x| model.forward(x),
    ).context("Evaluation failed")?;

    eval_result.print_summary();
    eval_result.print_confusion_matrix(ucr.n_classes);

    println!();
    println!("==============================================================");
    println!("      Training Complete! (MLX - Apple Silicon GPU)            ");
    println!("==============================================================");

    Ok(())
}

#[cfg(not(all(target_os = "macos", feature = "backend-mlx")))]
fn main() {
    #[cfg(not(target_os = "macos"))]
    {
        println!("This example requires macOS with Apple Silicon (M1/M2/M3/M4).");
        println!("MLX is only available on Apple platforms.");
    }
    #[cfg(all(target_os = "macos", not(feature = "backend-mlx")))]
    {
        println!("This example requires the 'backend-mlx' feature.");
        println!("Run with: cargo run --example train_ucr_mlx --release --features backend-mlx");
    }
}
