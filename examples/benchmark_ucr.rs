//! Benchmark: Run InceptionTimePlus on multiple UCR datasets
//!
//! This example trains InceptionTimePlus on multiple UCR datasets
//! and reports accuracy for benchmarking against Python tsai.
//!
//! Run with: cargo run --example benchmark_ucr --release
//!
//! For a quick test:
//!   cargo run --example benchmark_ucr --release -- --datasets ECG200,Chinatown,Coffee

use anyhow::{Context, Result};
use burn::prelude::*;
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;
use std::time::Instant;

use tsai_core::Seed;
use tsai_data::ucr::UCRDataset;
use tsai_data::TSDataLoaders;
use tsai_models::{InceptionTimePlus, InceptionTimePlusConfig};
use tsai_train::{ClassificationTrainer, ClassificationTrainerConfig};

/// Backend type for training (NdArray with autodiff).
type TrainBackend = Autodiff<NdArray>;

/// Result for a single dataset benchmark.
#[derive(Debug, Clone)]
struct BenchmarkResult {
    dataset: String,
    train_samples: usize,
    test_samples: usize,
    seq_len: usize,
    n_classes: usize,
    best_acc: f32,
    final_acc: f32,
    best_epoch: usize,
    training_time_secs: f64,
}

fn main() -> Result<()> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    let datasets_arg = args
        .iter()
        .position(|s| s == "--datasets")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str());

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
        .unwrap_or(64);

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

    // Default datasets for quick benchmarking (small/medium size)
    let default_datasets = vec![
        "Chinatown",     // 20 train, 345 test, 24 len, 2 classes
        "ECG200",        // 100 train, 100 test, 96 len, 2 classes
        "Coffee",        // 28 train, 28 test, 286 len, 2 classes
        "GunPoint",      // 50 train, 150 test, 150 len, 2 classes
        "ItalyPowerDemand", // 67 train, 1029 test, 24 len, 2 classes
        "SyntheticControl", // 300 train, 300 test, 60 len, 6 classes
        "Trace",         // 100 train, 100 test, 275 len, 4 classes
        "TwoLeadECG",    // 23 train, 1139 test, 82 len, 2 classes
    ];

    let datasets: Vec<&str> = match datasets_arg {
        Some(arg) => arg.split(',').collect(),
        None => default_datasets,
    };

    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║              tsai-rs: UCR Benchmark with InceptionTimePlus               ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Configuration:");
    println!("  Datasets: {} total", datasets.len());
    println!("  Epochs: {}", epochs);
    println!("  Batch size: {}", batch_size);
    println!("  Learning rate: {}", lr);
    println!("  Seed: {}", seed);
    println!();
    println!("─────────────────────────────────────────────────────────────────────────────");
    println!();

    let total_start = Instant::now();
    let mut results: Vec<BenchmarkResult> = Vec::new();
    let mut failed: Vec<(String, String)> = Vec::new();

    for (i, &dataset_name) in datasets.iter().enumerate() {
        println!(
            "[{}/{}] Training on: {}",
            i + 1,
            datasets.len(),
            dataset_name
        );

        match run_benchmark(dataset_name, epochs, batch_size, lr, seed) {
            Ok(result) => {
                println!(
                    "       Best acc: {:.2}% (epoch {}), Time: {:.1}s",
                    result.best_acc * 100.0,
                    result.best_epoch + 1,
                    result.training_time_secs
                );
                results.push(result);
            }
            Err(e) => {
                println!("       FAILED: {}", e);
                failed.push((dataset_name.to_string(), e.to_string()));
            }
        }
        println!();
    }

    let total_time = total_start.elapsed().as_secs_f64();

    // Print summary table
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("                              BENCHMARK RESULTS                                 ");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!();
    println!(
        "{:<25} {:>6} {:>6} {:>5} {:>4} {:>8} {:>8} {:>8}",
        "Dataset", "Train", "Test", "Len", "C", "Best%", "Final%", "Time(s)"
    );
    println!("{}", "─".repeat(79));

    for r in &results {
        println!(
            "{:<25} {:>6} {:>6} {:>5} {:>4} {:>7.2}% {:>7.2}% {:>8.1}",
            r.dataset,
            r.train_samples,
            r.test_samples,
            r.seq_len,
            r.n_classes,
            r.best_acc * 100.0,
            r.final_acc * 100.0,
            r.training_time_secs
        );
    }

    println!("{}", "─".repeat(79));

    // Calculate average accuracy
    if !results.is_empty() {
        let avg_best: f32 = results.iter().map(|r| r.best_acc).sum::<f32>() / results.len() as f32;
        let avg_final: f32 =
            results.iter().map(|r| r.final_acc).sum::<f32>() / results.len() as f32;
        let total_train_time: f64 = results.iter().map(|r| r.training_time_secs).sum();

        println!(
            "{:<25} {:>6} {:>6} {:>5} {:>4} {:>7.2}% {:>7.2}% {:>8.1}",
            "AVERAGE", "", "", "", "", avg_best * 100.0, avg_final * 100.0, total_train_time
        );
    }

    println!();
    println!("Total benchmark time: {:.1}s", total_time);
    println!(
        "Successful: {}, Failed: {}",
        results.len(),
        failed.len()
    );

    if !failed.is_empty() {
        println!("\nFailed datasets:");
        for (name, err) in &failed {
            println!("  - {}: {}", name, err);
        }
    }

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                         Benchmark Complete!                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");

    Ok(())
}

fn run_benchmark(
    dataset_name: &str,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    seed: u64,
) -> Result<BenchmarkResult> {
    // Load dataset
    let ucr = UCRDataset::load(dataset_name, None)
        .context(format!("Failed to load dataset '{}'", dataset_name))?;

    // Adjust batch size if larger than training set
    let actual_batch_size = batch_size.min(ucr.train.len());

    // Create dataloaders
    let dls = TSDataLoaders::builder(ucr.train.clone(), ucr.test.clone())
        .batch_size(actual_batch_size)
        .shuffle_train(true)
        .seed(Seed::new(seed))
        .build()
        .context("Failed to create dataloaders")?;

    // Configure model
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

    // Initialize model
    let device = <TrainBackend as Backend>::Device::default();
    let model: InceptionTimePlus<TrainBackend> = InceptionTimePlus::new(model_config, &device);

    // Configure trainer (no early stopping for benchmarking)
    let trainer_config = ClassificationTrainerConfig {
        n_epochs: epochs,
        lr,
        weight_decay: 0.01,
        grad_clip: 1.0,
        verbose: false, // Quiet mode for benchmarking
        early_stopping_patience: 0, // Disabled
        early_stopping_min_delta: 0.001,
    };

    let trainer = ClassificationTrainer::<TrainBackend>::new(trainer_config, device);

    // Train
    let result = trainer
        .fit_with_forward(
            model,
            &dls,
            |model, x| model.forward(x),
            |model, x| model.forward(x),
        )
        .context("Training failed")?;

    Ok(BenchmarkResult {
        dataset: dataset_name.to_string(),
        train_samples: ucr.train.len(),
        test_samples: ucr.test.len(),
        seq_len: ucr.seq_len,
        n_classes: ucr.n_classes,
        best_acc: result.best_valid_acc,
        final_acc: *result.valid_accs.last().unwrap_or(&0.0),
        best_epoch: result.best_epoch,
        training_time_secs: result.training_time_secs,
    })
}
