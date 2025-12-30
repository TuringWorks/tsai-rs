//! Example: UCR Classification with InceptionTimePlus
//!
//! This example demonstrates how to use tsai-rs for time series classification
//! using the InceptionTimePlus model on UCR Archive datasets.
//!
//! Run with: cargo run --example ucr_inception_time
//!
//! Note: First run will download the dataset (~1MB for ECG200).

use ndarray::{Array2, Array3};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use tsai_data::ucr::UCRDataset;
use tsai_train::Scheduler;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== tsai-rs: UCR Classification with InceptionTimePlus ===\n");

    // Choose dataset - ECG200 is small and downloads quickly
    let dataset_name = "ECG200";

    println!("Loading UCR dataset: {}", dataset_name);
    println!("(First run will download from timeseriesclassification.com)\n");

    // Load the UCR dataset
    let ucr = match UCRDataset::load(dataset_name, None) {
        Ok(ds) => ds,
        Err(e) => {
            println!("Failed to load dataset: {}", e);
            println!("\nFalling back to synthetic data...\n");
            return run_with_synthetic_data();
        }
    };

    println!("Dataset loaded successfully!");
    println!("─────────────────────────────────────────");
    println!("  Name:            {}", ucr.name);
    println!("  Classes:         {}", ucr.n_classes);
    println!("  Sequence length: {}", ucr.seq_len);
    println!("  Train samples:   {}", ucr.train.len());
    println!("  Test samples:    {}", ucr.test.len());
    println!();

    // Get dimensions
    let n_vars = ucr.train.n_vars();
    let seq_len = ucr.seq_len;
    let n_classes = ucr.n_classes;
    let batch_size = 16;
    let seed = 42u64;

    println!("Configuration:");
    println!("  Variables: {}", n_vars);
    println!("  Sequence length: {}", seq_len);
    println!("  Classes: {}", n_classes);
    println!("  Batch size: {}", batch_size);
    println!("  Seed: {}\n", seed);

    // Create dataloaders from train/test splits
    println!("Creating dataloaders...");
    let dls = tsai_data::TSDataLoaders::builder(ucr.train.clone(), ucr.test.clone())
        .batch_size(batch_size)
        .shuffle_train(true)
        .seed(tsai_core::Seed::new(seed))
        .build()?;
    println!("  Train batches: {}", dls.train().n_batches());
    println!("  Test batches: {}\n", dls.valid().n_batches());

    // Configure model
    println!("Configuring InceptionTimePlus model...");
    let model_config = tsai_models::InceptionTimePlusConfig {
        n_vars,
        seq_len,
        n_classes,
        n_blocks: 6,
        n_filters: 32,
        kernel_sizes: [9, 19, 39],
        bottleneck_dim: 32,
        dropout: 0.0,
    };
    println!("  Blocks: {}", model_config.n_blocks);
    println!("  Filters: {}", model_config.n_filters);
    println!("  Kernel sizes: {:?}", model_config.kernel_sizes);
    println!("  Bottleneck dim: {}\n", model_config.bottleneck_dim);

    // Configure training
    println!("Configuring training...");
    let train_config = tsai_train::LearnerConfig {
        lr: 1e-3,
        weight_decay: 0.01,
        grad_clip: 1.0,
        mixed_precision: false,
    };
    println!("  Learning rate: {}", train_config.lr);
    println!("  Weight decay: {}", train_config.weight_decay);
    println!("  Gradient clipping: {}\n", train_config.grad_clip);

    // Configure scheduler
    println!("Configuring OneCycle scheduler...");
    let n_epochs = 25;
    let steps_per_epoch = dls.train().n_batches();
    let total_steps = n_epochs * steps_per_epoch;
    let scheduler = tsai_train::OneCycleLR::simple(train_config.lr, total_steps);

    println!("  Epochs: {}", n_epochs);
    println!("  Steps per epoch: {}", steps_per_epoch);
    println!("  Total steps: {}", total_steps);
    println!("  Sample LRs:");
    for pct in [0, 25, 50, 75, 100] {
        let step = (pct * total_steps) / 100;
        println!("    Step {} ({}%): {:.6}", step, pct, scheduler.get_lr(step.min(total_steps - 1)));
    }
    println!();

    // Demonstrate transforms
    println!("Available augmentation transforms:");
    println!("  - GaussianNoise(std=0.1)");
    println!("  - TimeWarp(magnitude=0.2)");
    println!("  - MagScale(factor=1.2)");
    println!("  - CutOut(max_cut_ratio=0.2)");
    println!("  - MixUp1d(alpha=0.4)");
    println!("  - CutMix1d(alpha=1.0)");
    println!("  - TSRandomShift, TSHorizontalFlip, TSVerticalFlip\n");

    // Demonstrate imaging transforms
    println!("Demonstrating imaging transform (GASF)...");
    let sample_series: Vec<f32> = (0..20).map(|i| (i as f32 * 0.1).sin()).collect();
    let gasf = tsai_transforms::TSToGASF::new(20);
    let gasf_image = gasf.compute(&sample_series);
    println!("  Input series length: {}", sample_series.len());
    println!("  GASF output shape: {}x{}\n", gasf_image.len(), gasf_image[0].len());

    // Demonstrate analysis tools with simulated predictions
    println!("Demonstrating analysis tools (simulated predictions)...");

    // Simulate some predictions
    let preds = vec![0, 0, 1, 1, 0, 1, 0, 1, 0, 1];
    let targets = vec![0, 1, 1, 1, 0, 0, 0, 1, 0, 1];
    let cm = tsai_analysis::confusion_matrix(&preds, &targets, n_classes);
    println!("  Confusion matrix:");
    println!("    Accuracy: {:.2}%", cm.accuracy() * 100.0);
    println!("    Macro F1: {:.4}", cm.macro_f1());

    // Top losses
    let losses = vec![0.1, 0.5, 0.3, 0.9, 0.2, 0.15, 0.8, 0.05, 0.4, 0.6];
    let probs = vec![0.9, 0.6, 0.8, 0.55, 0.85, 0.88, 0.58, 0.95, 0.7, 0.5];
    let top = tsai_analysis::top_losses(&losses, &targets, &preds, &probs, 3);
    println!("\n  Top 3 losses:");
    for (i, tl) in top.iter().enumerate() {
        println!(
            "    {}: idx={}, loss={:.4}, true={}, pred={}",
            i + 1,
            tl.index,
            tl.loss,
            tl.target,
            tl.pred
        );
    }

    // Available models
    println!("\n\nAvailable model architectures:");
    println!("  CNN:");
    println!("    - InceptionTimePlus: Multi-scale inception blocks");
    println!("    - XceptionTime: Depthwise separable convolutions");
    println!("    - FCN: Fully convolutional network");
    println!("    - ResNetPlus: Residual network");
    println!("    - OmniScaleCNN: Multi-scale parallel branches");
    println!("    - XCMPlus: Explainable CNN");
    println!("  Transformer:");
    println!("    - PatchTST: Patch-based transformer");
    println!("    - TST: Time series transformer");
    println!("    - TSiT: Time series image transformer");
    println!("    - TSPerceiver: Perceiver for time series");
    println!("  RNN:");
    println!("    - RNNPlus: LSTM/GRU with attention");
    println!("    - RNNAttention: RNN with self-attention");
    println!("  ROCKET:");
    println!("    - Rocket: Random convolutional kernels");
    println!("    - MiniRocket: Optimized fixed kernels");
    println!("    - MultiRocket: Multiple kernel types");
    println!("    - HydraPlus: Hybrid dictionary approach");

    println!("\n=== Example completed successfully! ===");
    println!("\nNote: This example demonstrates the API and data structures.");
    println!("Full training requires the training loop to be executed with");
    println!("actual forward/backward passes using the Burn framework.");

    Ok(())
}

/// Fallback: run with synthetic data if UCR download fails.
fn run_with_synthetic_data() -> Result<(), Box<dyn std::error::Error>> {
    let n_samples = 200;
    let n_vars = 1;
    let seq_len = 96;
    let n_classes = 2;
    let batch_size = 32;
    let seed = 42u64;

    println!("Using synthetic data:");
    println!("  Samples: {}", n_samples);
    println!("  Variables: {}", n_vars);
    println!("  Sequence length: {}", seq_len);
    println!("  Classes: {}\n", n_classes);

    // Generate synthetic data
    let (x, y) = generate_synthetic_data(n_samples, n_vars, seq_len, n_classes, seed);

    // Create dataset
    let dataset = tsai_data::TSDataset::from_arrays(x, Some(y))?;

    // Split into train/validation
    let (train_ds, valid_ds) =
        tsai_data::train_test_split(&dataset, 0.2, tsai_core::Seed::new(seed))?;

    println!("  Train size: {}", train_ds.len());
    println!("  Valid size: {}\n", valid_ds.len());

    // Create dataloaders
    let dls = tsai_data::TSDataLoaders::builder(train_ds, valid_ds)
        .batch_size(batch_size)
        .shuffle_train(true)
        .seed(tsai_core::Seed::new(seed))
        .build()?;

    println!("  Train batches: {}", dls.train().n_batches());
    println!("  Valid batches: {}", dls.valid().n_batches());

    println!("\n=== Synthetic data example completed! ===");
    Ok(())
}

/// Generate synthetic time series data for classification.
fn generate_synthetic_data(
    n_samples: usize,
    n_vars: usize,
    seq_len: usize,
    n_classes: usize,
    seed: u64,
) -> (Array3<f32>, Array2<f32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut x = Array3::<f32>::zeros((n_samples, n_vars, seq_len));
    let mut y = Array2::<f32>::zeros((n_samples, 1));

    for i in 0..n_samples {
        let class = i % n_classes;
        y[[i, 0]] = class as f32;

        // Generate class-specific patterns
        let freq = (class + 1) as f32 * 0.1;
        let amplitude = 1.0 + (class as f32 * 0.2);

        for v in 0..n_vars {
            for t in 0..seq_len {
                let base = amplitude * (freq * t as f32).sin();
                let noise: f32 = rng.gen_range(-0.1..0.1);
                x[[i, v, t]] = base + noise;
            }
        }
    }

    (x, y)
}
