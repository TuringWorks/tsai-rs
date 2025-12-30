//! Example: Compare different model architectures on a UCR dataset
//!
//! This example trains multiple architectures on the same dataset
//! and compares their performance.
//!
//! Run with: cargo run --example compare_models --release
//!
//! Options:
//!   --dataset DATASET  Dataset name (default: ECG200)
//!   --epochs N         Number of epochs (default: 15)

use anyhow::{Context, Result};
use burn::prelude::*;
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;
use std::time::Instant;

use tsai_core::Seed;
use tsai_data::ucr::UCRDataset;
use tsai_data::TSDataLoaders;
use tsai_models::{
    FCN, FCNConfig, InceptionTimePlus, InceptionTimePlusConfig, ResNetPlus, ResNetPlusConfig,
    TSTConfig, TSTPlus, XceptionTime, XceptionTimeConfig,
};
use tsai_train::{ClassificationTrainer, ClassificationTrainerConfig};

/// Backend type for training.
type TrainBackend = Autodiff<NdArray>;

/// Result for a model benchmark.
#[derive(Debug, Clone)]
struct ModelResult {
    name: String,
    best_acc: f32,
    final_acc: f32,
    best_epoch: usize,
    training_time_secs: f64,
}

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
        .unwrap_or(15);

    let seed: u64 = args
        .iter()
        .position(|s| s == "--seed")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);

    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║            tsai-rs: Model Architecture Comparison                        ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Load dataset
    println!("Loading dataset: {}", dataset_name);
    let ucr = UCRDataset::load(dataset_name, None)
        .context(format!("Failed to load dataset '{}'", dataset_name))?;

    println!("  Train samples: {}", ucr.train.len());
    println!("  Test samples: {}", ucr.test.len());
    println!("  Sequence length: {}", ucr.seq_len);
    println!("  Classes: {}", ucr.n_classes);
    println!("  Epochs: {}", epochs);
    println!();

    // Create dataloaders
    let batch_size = 64.min(ucr.train.len());
    let dls = TSDataLoaders::builder(ucr.train.clone(), ucr.test.clone())
        .batch_size(batch_size)
        .shuffle_train(true)
        .seed(Seed::new(seed))
        .build()
        .context("Failed to create dataloaders")?;

    let mut results: Vec<ModelResult> = Vec::new();
    let total_start = Instant::now();

    // Train each model
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("                           Training Models                                      ");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!();

    // 1. InceptionTimePlus
    println!("[1/5] Training InceptionTimePlus...");
    if let Ok(result) = train_inception(&dls, &ucr, epochs, seed) {
        println!(
            "      Best: {:.2}% (epoch {}), Time: {:.1}s",
            result.best_acc * 100.0,
            result.best_epoch + 1,
            result.training_time_secs
        );
        results.push(result);
    } else {
        println!("      FAILED");
    }
    println!();

    // 2. FCN
    println!("[2/5] Training FCN...");
    if let Ok(result) = train_fcn(&dls, &ucr, epochs, seed) {
        println!(
            "      Best: {:.2}% (epoch {}), Time: {:.1}s",
            result.best_acc * 100.0,
            result.best_epoch + 1,
            result.training_time_secs
        );
        results.push(result);
    } else {
        println!("      FAILED");
    }
    println!();

    // 3. ResNetPlus
    println!("[3/5] Training ResNetPlus...");
    if let Ok(result) = train_resnet(&dls, &ucr, epochs, seed) {
        println!(
            "      Best: {:.2}% (epoch {}), Time: {:.1}s",
            result.best_acc * 100.0,
            result.best_epoch + 1,
            result.training_time_secs
        );
        results.push(result);
    } else {
        println!("      FAILED");
    }
    println!();

    // 4. XceptionTime
    println!("[4/5] Training XceptionTime...");
    if let Ok(result) = train_xception(&dls, &ucr, epochs, seed) {
        println!(
            "      Best: {:.2}% (epoch {}), Time: {:.1}s",
            result.best_acc * 100.0,
            result.best_epoch + 1,
            result.training_time_secs
        );
        results.push(result);
    } else {
        println!("      FAILED");
    }
    println!();

    // 5. TSTPlus (Transformer)
    println!("[5/5] Training TSTPlus (Transformer)...");
    if let Ok(result) = train_tst(&dls, &ucr, epochs, seed) {
        println!(
            "      Best: {:.2}% (epoch {}), Time: {:.1}s",
            result.best_acc * 100.0,
            result.best_epoch + 1,
            result.training_time_secs
        );
        results.push(result);
    } else {
        println!("      FAILED");
    }
    println!();

    let total_time = total_start.elapsed().as_secs_f64();

    // Print results table
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("                              COMPARISON RESULTS                                ");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!();
    println!("Dataset: {} ({} train, {} test, {} classes)",
             dataset_name, ucr.train.len(), ucr.test.len(), ucr.n_classes);
    println!();
    println!(
        "{:<20} {:>10} {:>10} {:>12} {:>10}",
        "Model", "Best Acc", "Final Acc", "Best Epoch", "Time (s)"
    );
    println!("{}", "─".repeat(65));

    // Sort by best accuracy
    results.sort_by(|a, b| b.best_acc.partial_cmp(&a.best_acc).unwrap());

    for (i, r) in results.iter().enumerate() {
        let marker = if i == 0 { " *" } else { "" };
        println!(
            "{:<20} {:>9.2}% {:>9.2}% {:>12} {:>10.1}{}",
            r.name,
            r.best_acc * 100.0,
            r.final_acc * 100.0,
            r.best_epoch + 1,
            r.training_time_secs,
            marker
        );
    }

    println!("{}", "─".repeat(65));
    println!();
    println!("* = Best performing model");
    println!("Total comparison time: {:.1}s", total_time);
    println!();

    if let Some(best) = results.first() {
        println!("Winner: {} with {:.2}% accuracy", best.name, best.best_acc * 100.0);
    }

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                        Comparison Complete!                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");

    Ok(())
}

fn train_inception(dls: &TSDataLoaders, ucr: &UCRDataset, epochs: usize, _seed: u64) -> Result<ModelResult> {
    let config = InceptionTimePlusConfig {
        n_vars: ucr.train.n_vars(),
        seq_len: ucr.seq_len,
        n_classes: ucr.n_classes,
        n_blocks: 6,
        n_filters: 32,
        kernel_sizes: [9, 19, 39],
        bottleneck_dim: 32,
        dropout: 0.0,
    };

    let device = <TrainBackend as Backend>::Device::default();
    let model: InceptionTimePlus<TrainBackend> = InceptionTimePlus::new(config, &device);

    let trainer_config = ClassificationTrainerConfig {
        n_epochs: epochs,
        lr: 1e-3,
        weight_decay: 0.01,
        grad_clip: 1.0,
        verbose: false,
        early_stopping_patience: 0,
        early_stopping_min_delta: 0.001,
    };

    let trainer = ClassificationTrainer::<TrainBackend>::new(trainer_config, device);

    let result = trainer.fit_with_forward(
        model,
        dls,
        |model, x| model.forward(x),
        |model, x| model.forward(x),
    )?;

    Ok(ModelResult {
        name: "InceptionTimePlus".to_string(),
        best_acc: result.best_valid_acc,
        final_acc: *result.valid_accs.last().unwrap_or(&0.0),
        best_epoch: result.best_epoch,
        training_time_secs: result.training_time_secs,
    })
}

fn train_fcn(dls: &TSDataLoaders, ucr: &UCRDataset, epochs: usize, _seed: u64) -> Result<ModelResult> {
    let config = FCNConfig {
        n_vars: ucr.train.n_vars(),
        seq_len: ucr.seq_len,
        n_classes: ucr.n_classes,
        n_filters_1: 128,
        n_filters_2: 256,
        n_filters_3: 128,
        kernel_size_1: 7,
        kernel_size_2: 5,
        kernel_size_3: 3,
    };

    let device = <TrainBackend as Backend>::Device::default();
    let model: FCN<TrainBackend> = FCN::new(config, &device);

    let trainer_config = ClassificationTrainerConfig {
        n_epochs: epochs,
        lr: 1e-3,
        weight_decay: 0.01,
        grad_clip: 1.0,
        verbose: false,
        early_stopping_patience: 0,
        early_stopping_min_delta: 0.001,
    };

    let trainer = ClassificationTrainer::<TrainBackend>::new(trainer_config, device);

    let result = trainer.fit_with_forward(
        model,
        dls,
        |model, x| model.forward(x),
        |model, x| model.forward(x),
    )?;

    Ok(ModelResult {
        name: "FCN".to_string(),
        best_acc: result.best_valid_acc,
        final_acc: *result.valid_accs.last().unwrap_or(&0.0),
        best_epoch: result.best_epoch,
        training_time_secs: result.training_time_secs,
    })
}

fn train_resnet(dls: &TSDataLoaders, ucr: &UCRDataset, epochs: usize, _seed: u64) -> Result<ModelResult> {
    let config = ResNetPlusConfig {
        n_vars: ucr.train.n_vars(),
        seq_len: ucr.seq_len,
        n_classes: ucr.n_classes,
        n_blocks: 3,
        n_filters: vec![64, 128, 128],
        kernel_size: 7,
    };

    let device = <TrainBackend as Backend>::Device::default();
    let model: ResNetPlus<TrainBackend> = ResNetPlus::new(config, &device);

    let trainer_config = ClassificationTrainerConfig {
        n_epochs: epochs,
        lr: 1e-3,
        weight_decay: 0.01,
        grad_clip: 1.0,
        verbose: false,
        early_stopping_patience: 0,
        early_stopping_min_delta: 0.001,
    };

    let trainer = ClassificationTrainer::<TrainBackend>::new(trainer_config, device);

    let result = trainer.fit_with_forward(
        model,
        dls,
        |model, x| model.forward(x),
        |model, x| model.forward(x),
    )?;

    Ok(ModelResult {
        name: "ResNetPlus".to_string(),
        best_acc: result.best_valid_acc,
        final_acc: *result.valid_accs.last().unwrap_or(&0.0),
        best_epoch: result.best_epoch,
        training_time_secs: result.training_time_secs,
    })
}

fn train_xception(dls: &TSDataLoaders, ucr: &UCRDataset, epochs: usize, _seed: u64) -> Result<ModelResult> {
    let config = XceptionTimeConfig {
        n_vars: ucr.train.n_vars(),
        seq_len: ucr.seq_len,
        n_classes: ucr.n_classes,
        n_blocks: 4,
        n_filters: 64,
        kernel_size: 39,
        dropout: 0.0,
    };

    let device = <TrainBackend as Backend>::Device::default();
    let model: XceptionTime<TrainBackend> = XceptionTime::new(config, &device);

    let trainer_config = ClassificationTrainerConfig {
        n_epochs: epochs,
        lr: 1e-3,
        weight_decay: 0.01,
        grad_clip: 1.0,
        verbose: false,
        early_stopping_patience: 0,
        early_stopping_min_delta: 0.001,
    };

    let trainer = ClassificationTrainer::<TrainBackend>::new(trainer_config, device);

    let result = trainer.fit_with_forward(
        model,
        dls,
        |model, x| model.forward(x),
        |model, x| model.forward(x),
    )?;

    Ok(ModelResult {
        name: "XceptionTime".to_string(),
        best_acc: result.best_valid_acc,
        final_acc: *result.valid_accs.last().unwrap_or(&0.0),
        best_epoch: result.best_epoch,
        training_time_secs: result.training_time_secs,
    })
}

fn train_tst(dls: &TSDataLoaders, ucr: &UCRDataset, epochs: usize, _seed: u64) -> Result<ModelResult> {
    let config = TSTConfig {
        n_vars: ucr.train.n_vars(),
        seq_len: ucr.seq_len,
        n_classes: ucr.n_classes,
        d_model: 128,
        n_heads: 4,
        n_layers: 3,
        d_ff: 256,
        dropout: 0.1,
        use_pe: true,
    };

    let device = <TrainBackend as Backend>::Device::default();
    let model: TSTPlus<TrainBackend> = TSTPlus::new(config, &device);

    let trainer_config = ClassificationTrainerConfig {
        n_epochs: epochs,
        lr: 1e-3,
        weight_decay: 0.01,
        grad_clip: 1.0,
        verbose: false,
        early_stopping_patience: 0,
        early_stopping_min_delta: 0.001,
    };

    let trainer = ClassificationTrainer::<TrainBackend>::new(trainer_config, device);

    let result = trainer.fit_with_forward(
        model,
        dls,
        |model, x| model.forward(x),
        |model, x| model.forward(x),
    )?;

    Ok(ModelResult {
        name: "TSTPlus".to_string(),
        best_acc: result.best_valid_acc,
        final_acc: *result.valid_accs.last().unwrap_or(&0.0),
        best_epoch: result.best_epoch,
        training_time_secs: result.training_time_secs,
    })
}
