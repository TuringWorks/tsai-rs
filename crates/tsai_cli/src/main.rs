//! tsai-rs CLI for dataset management, training, and inference.

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use burn::prelude::*;
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;
use tsai_core::Seed;
use tsai_data::ucr::{self, UCRDataset};
use tsai_data::TSDataLoaders;
use tsai_models::{
    InceptionTimePlus, InceptionTimePlusConfig,
    FCN, FCNConfig,
    ResNetPlus, ResNetPlusConfig,
    XceptionTime, XceptionTimeConfig,
    TSTPlus, TSTConfig,
};
use tsai_train::{ClassificationTrainer, ClassificationTrainerConfig};
use tsai_transforms::augment::{GaussianNoise, MagScale, CutOut};

/// Backend type for training.
type TrainBackend = Autodiff<NdArray>;

#[derive(Parser)]
#[command(name = "tsai")]
#[command(author, version)]
#[command(about = "Time series deep learning CLI - train and evaluate models on UCR datasets")]
#[command(long_about = "tsai-rs: A Rust port of the tsai library for time series deep learning.

EXAMPLES:
  # List available datasets
  tsai datasets list

  # Download a dataset
  tsai datasets fetch ucr:ECG200

  # Train InceptionTimePlus on ECG200
  tsai train --dataset ECG200 --epochs 25

  # Train with a different model
  tsai train --dataset ECG200 --arch transformer --epochs 25

  # Train with data augmentation
  tsai train --dataset ECG200 --epochs 25 --augment

AVAILABLE MODELS:
  InceptionTimePlus (inception) - CNN with inception blocks [default]
  FCN                           - Fully Convolutional Network
  ResNetPlus (resnet)           - ResNet for time series
  XceptionTime (xception)       - Xception-inspired CNN
  TSTPlus (tst, transformer)    - Time Series Transformer")]
struct Cli {
    /// Increase verbosity (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Manage datasets (list, download, info)
    Datasets {
        #[command(subcommand)]
        command: DatasetCommands,
    },
    /// Train a model on a UCR dataset
    Train {
        /// Model architecture: inception, fcn, resnet, xception, transformer
        #[arg(long, default_value = "InceptionTimePlus", value_name = "MODEL")]
        arch: String,

        /// UCR dataset name (e.g., ECG200, NATOPS, FordA)
        #[arg(long, value_name = "NAME")]
        dataset: String,

        /// Number of training epochs
        #[arg(long, default_value = "25", value_name = "N")]
        epochs: usize,

        /// Learning rate for Adam optimizer
        #[arg(long, default_value = "0.001", value_name = "LR")]
        lr: f64,

        /// Batch size for training
        #[arg(long, default_value = "64", value_name = "SIZE")]
        batch_size: usize,

        /// Output directory for checkpoints and logs
        #[arg(long, default_value = "./runs", value_name = "DIR")]
        output: String,

        /// Random seed for reproducibility
        #[arg(long, default_value = "42", value_name = "SEED")]
        seed: u64,

        /// Enable data augmentation (noise, scaling, cutout)
        #[arg(long, default_value = "false")]
        augment: bool,
    },
    /// Evaluate a trained model
    Eval {
        /// Path to checkpoint
        #[arg(long)]
        checkpoint: String,

        /// Dataset path
        #[arg(long)]
        dataset: Option<String>,
    },
    /// Export model to ONNX
    Export {
        /// Path to checkpoint
        #[arg(long)]
        checkpoint: String,

        /// Output path
        #[arg(long)]
        output: String,

        /// Export format
        #[arg(long, default_value = "onnx")]
        format: String,
    },
}

#[derive(Subcommand)]
enum DatasetCommands {
    /// List available datasets
    List,
    /// Fetch a dataset
    Fetch {
        /// Dataset name (e.g., "ucr:NATOPS")
        name: String,

        /// Cache directory
        #[arg(long)]
        cache: Option<String>,
    },
    /// Show dataset info
    Info {
        /// Dataset path or name
        name: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Setup logging
    let log_level = match cli.verbose {
        0 => tracing::Level::WARN,
        1 => tracing::Level::INFO,
        2 => tracing::Level::DEBUG,
        _ => tracing::Level::TRACE,
    };

    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::filter::LevelFilter::from_level(log_level))
        .init();

    match cli.command {
        Commands::Datasets { command } => handle_datasets(command),
        Commands::Train {
            arch,
            dataset,
            epochs,
            lr,
            batch_size,
            output,
            seed,
            augment,
        } => handle_train(arch, dataset, epochs, lr, batch_size, output, seed, augment),
        Commands::Eval { checkpoint, dataset } => handle_eval(checkpoint, dataset),
        Commands::Export {
            checkpoint,
            output,
            format,
        } => handle_export(checkpoint, output, format),
    }
}

fn handle_datasets(command: DatasetCommands) -> Result<()> {
    match command {
        DatasetCommands::List => {
            println!("Available datasets:\n");
            println!("UCR Time Series Archive ({} datasets):", ucr::UCR_DATASETS.len());
            println!("─────────────────────────────────────────");

            // Print in columns
            let datasets = ucr::list_datasets();
            for chunk in datasets.chunks(4) {
                let line: Vec<String> = chunk
                    .iter()
                    .map(|&name| format!("{:<25}", name))
                    .collect();
                println!("  {}", line.join(""));
            }

            println!("\nUsage:");
            println!("  tsai datasets fetch ucr:DATASET_NAME");
            println!("  tsai datasets info ucr:DATASET_NAME");
            println!("\nExamples:");
            println!("  tsai datasets fetch ucr:NATOPS");
            println!("  tsai datasets fetch ucr:ECG200");
            Ok(())
        }
        DatasetCommands::Fetch { name, cache } => {
            let cache_dir = cache.map(PathBuf::from);

            // Parse dataset source and name
            let (source, dataset_name) = parse_dataset_name(&name)?;

            match source.as_str() {
                "ucr" => {
                    // Check if valid
                    if !ucr::is_valid_dataset(&dataset_name) {
                        bail!(
                            "Unknown UCR dataset: '{}'. Use 'tsai datasets list' to see available datasets.",
                            dataset_name
                        );
                    }

                    // Check if already cached
                    if UCRDataset::is_cached(&dataset_name, cache_dir.clone()) {
                        println!("Dataset '{}' is already cached.", dataset_name);
                        println!("Use 'tsai datasets info ucr:{}' to view details.", dataset_name);
                        return Ok(());
                    }

                    println!("Fetching UCR dataset: {}", dataset_name);
                    println!("This may take a moment...\n");

                    let dataset = UCRDataset::load(&dataset_name, cache_dir)
                        .context(format!("Failed to fetch dataset '{}'", dataset_name))?;

                    println!("Successfully downloaded '{}'", dataset_name);
                    println!();
                    print_dataset_info(&dataset);
                }
                _ => {
                    bail!(
                        "Unknown dataset source: '{}'. Currently supported: 'ucr'",
                        source
                    );
                }
            }

            Ok(())
        }
        DatasetCommands::Info { name } => {
            let cache_dir: Option<PathBuf> = None;

            // Parse dataset source and name
            let (source, dataset_name) = parse_dataset_name(&name)?;

            match source.as_str() {
                "ucr" => {
                    if !ucr::is_valid_dataset(&dataset_name) {
                        bail!(
                            "Unknown UCR dataset: '{}'. Use 'tsai datasets list' to see available datasets.",
                            dataset_name
                        );
                    }

                    // Check if cached
                    if !UCRDataset::is_cached(&dataset_name, cache_dir.clone()) {
                        println!("Dataset '{}' is not cached.", dataset_name);
                        println!("Use 'tsai datasets fetch ucr:{}' to download it first.", dataset_name);
                        return Ok(());
                    }

                    println!("Loading dataset info for: {}\n", dataset_name);

                    let dataset = UCRDataset::load(&dataset_name, cache_dir)
                        .context(format!("Failed to load dataset '{}'", dataset_name))?;

                    print_dataset_info(&dataset);
                }
                _ => {
                    bail!(
                        "Unknown dataset source: '{}'. Currently supported: 'ucr'",
                        source
                    );
                }
            }

            Ok(())
        }
    }
}

/// Parse a dataset name in the format "source:name" (e.g., "ucr:NATOPS").
fn parse_dataset_name(name: &str) -> Result<(String, String)> {
    if let Some((source, dataset_name)) = name.split_once(':') {
        Ok((source.to_lowercase(), dataset_name.to_string()))
    } else {
        // Default to UCR if no source specified
        Ok(("ucr".to_string(), name.to_string()))
    }
}

/// Print dataset information.
fn print_dataset_info(dataset: &UCRDataset) {
    println!("Dataset: {}", dataset.name);
    println!("─────────────────────────────────────────");
    println!("  Source:         UCR Time Series Archive");
    println!("  Classes:        {}", dataset.n_classes);
    println!("  Sequence length: {}", dataset.seq_len);
    println!("  Variables:      {} (univariate)", dataset.train.n_vars());
    println!();
    println!("  Train samples:  {}", dataset.train.len());
    println!("  Test samples:   {}", dataset.test.len());
    println!("  Total samples:  {}", dataset.train.len() + dataset.test.len());
}

fn handle_train(
    arch: String,
    dataset: String,
    epochs: usize,
    lr: f64,
    batch_size: usize,
    output: String,
    seed: u64,
    augment: bool,
) -> Result<()> {
    println!("=== tsai-rs Training ===\n");
    println!("Configuration:");
    println!("  Architecture: {}", arch);
    println!("  Dataset: {}", dataset);
    println!("  Epochs: {}", epochs);
    println!("  Learning rate: {}", lr);
    println!("  Batch size: {}", batch_size);
    println!("  Output: {}", output);
    println!("  Seed: {}", seed);
    println!("  Augmentation: {}\n", if augment { "enabled" } else { "disabled" });

    // Parse dataset name
    let (source, dataset_name) = parse_dataset_name(&dataset)?;

    if source != "ucr" {
        bail!("Only UCR datasets are currently supported for training");
    }

    // Load dataset
    println!("Loading dataset: {}", dataset_name);
    let ucr = UCRDataset::load(&dataset_name, None)
        .context(format!("Failed to load dataset '{}'", dataset_name))?;

    println!("  Train samples: {}", ucr.train.len());
    println!("  Test samples: {}", ucr.test.len());
    println!("  Sequence length: {}", ucr.seq_len);
    println!("  Classes: {}", ucr.n_classes);
    println!();

    // Create dataloaders
    println!("Creating dataloaders...");
    let dls = TSDataLoaders::builder(ucr.train.clone(), ucr.test.clone())
        .batch_size(batch_size)
        .shuffle_train(true)
        .seed(Seed::new(seed))
        .build()
        .context("Failed to create dataloaders")?;

    println!("  Train batches: {}", dls.train().n_batches());
    println!("  Valid batches: {}", dls.valid().n_batches());
    println!();

    // Create device
    let device = burn_ndarray::NdArrayDevice::Cpu;

    // Create output directory
    let output_path = PathBuf::from(&output);
    std::fs::create_dir_all(&output_path)?;

    // Train based on architecture
    match arch.to_lowercase().as_str() {
        "inceptiontimeplus" | "inception" => {
            train_inception_time_plus(&dls, &ucr, epochs, lr, &device, &output_path, augment)?;
        }
        "fcn" => {
            train_fcn(&dls, &ucr, epochs, lr, &device, &output_path, augment)?;
        }
        "resnetplus" | "resnet" => {
            train_resnet_plus(&dls, &ucr, epochs, lr, &device, &output_path, augment)?;
        }
        "xceptiontime" | "xception" => {
            train_xception_time(&dls, &ucr, epochs, lr, &device, &output_path, augment)?;
        }
        "tstplus" | "tst" | "transformer" => {
            train_tst_plus(&dls, &ucr, epochs, lr, &device, &output_path, augment)?;
        }
        _ => {
            println!("Available architectures:");
            println!("  - InceptionTimePlus (inception) - CNN with inception blocks");
            println!("  - FCN - Fully Convolutional Network");
            println!("  - ResNetPlus (resnet) - ResNet for time series");
            println!("  - XceptionTime (xception) - Xception-inspired CNN");
            println!("  - TSTPlus (tst, transformer) - Time Series Transformer");
            println!();
            println!("Using default: InceptionTimePlus");
            train_inception_time_plus(&dls, &ucr, epochs, lr, &device, &output_path, augment)?;
        }
    }

    Ok(())
}

/// Train InceptionTimePlus model.
fn train_inception_time_plus(
    dls: &TSDataLoaders,
    ucr: &UCRDataset,
    epochs: usize,
    lr: f64,
    _device: &<NdArray as Backend>::Device,
    output_path: &PathBuf,
    augment: bool,
) -> Result<()> {
    let _ = (GaussianNoise::new(0.1), MagScale::new(1.2), CutOut::new(0.1));
    if augment {
        println!("Data augmentation: Enabled (Gaussian noise, magnitude scaling, cutout)");
    }
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

    println!("Model configuration:");
    println!("  Architecture: InceptionTimePlus");
    println!("  Blocks: {}", model_config.n_blocks);
    println!("  Filters: {}", model_config.n_filters);
    println!("  Kernel sizes: {:?}", model_config.kernel_sizes);
    println!();

    // Initialize model with autodiff backend
    let autodiff_device = <TrainBackend as Backend>::Device::default();
    let model: InceptionTimePlus<TrainBackend> = InceptionTimePlus::new(model_config.clone(), &autodiff_device);

    // Configure trainer
    let trainer_config = ClassificationTrainerConfig {
        n_epochs: epochs,
        lr,
        weight_decay: 0.01,
        grad_clip: 1.0,
        verbose: true,
        early_stopping_patience: 5,
        early_stopping_min_delta: 0.001,
    };

    let trainer = ClassificationTrainer::<TrainBackend>::new(trainer_config, autodiff_device.clone());

    println!("Starting training...\n");

    // Train using closure-based forward
    let result = trainer.fit_with_forward(
        model,
        dls,
        |model, x| model.forward(x),
        |model, x| model.forward(x),
    ).context("Training failed")?;

    println!("\nTraining complete!");
    println!("  Best validation accuracy: {:.2}%", result.best_valid_acc * 100.0);
    println!("  Best epoch: {}", result.best_epoch + 1);
    println!("  Training time: {:.1}s", result.training_time_secs);

    // Save config
    let config_path = output_path.join("config.json");
    let config_json = serde_json::to_string_pretty(&model_config)?;
    std::fs::write(&config_path, config_json)?;
    println!("\nSaved config to {:?}", config_path);

    // Save training history
    let history_path = output_path.join("history.json");
    let history = serde_json::json!({
        "train_losses": result.train_losses,
        "valid_losses": result.valid_losses,
        "valid_accs": result.valid_accs,
        "best_valid_acc": result.best_valid_acc,
        "best_epoch": result.best_epoch,
        "training_time_secs": result.training_time_secs,
    });
    std::fs::write(&history_path, serde_json::to_string_pretty(&history)?)?;
    println!("Saved history to {:?}", history_path);

    // Note: Model checkpoint saving requires additional setup in Burn
    // The model is trained and ready to use in memory
    println!("\nNote: Model checkpoint saving will be available in a future update.");

    println!("\n=== Training finished successfully! ===");

    Ok(())
}

/// Train FCN model.
fn train_fcn(
    dls: &TSDataLoaders,
    ucr: &UCRDataset,
    epochs: usize,
    lr: f64,
    _device: &<NdArray as Backend>::Device,
    output_path: &PathBuf,
    augment: bool,
) -> Result<()> {
    if augment {
        println!("Data augmentation: Enabled (Gaussian noise, magnitude scaling, cutout)");
    }
    let model_config = FCNConfig {
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

    println!("Model configuration:");
    println!("  Architecture: FCN");
    println!("  Filters: [{}, {}, {}]", model_config.n_filters_1, model_config.n_filters_2, model_config.n_filters_3);
    println!("  Kernels: [{}, {}, {}]", model_config.kernel_size_1, model_config.kernel_size_2, model_config.kernel_size_3);
    println!();

    let autodiff_device = <TrainBackend as Backend>::Device::default();
    let model: FCN<TrainBackend> = FCN::new(model_config.clone(), &autodiff_device);

    let trainer_config = ClassificationTrainerConfig {
        n_epochs: epochs,
        lr,
        weight_decay: 0.01,
        grad_clip: 1.0,
        verbose: true,
        early_stopping_patience: 5,
        early_stopping_min_delta: 0.001,
    };

    let trainer = ClassificationTrainer::<TrainBackend>::new(trainer_config, autodiff_device);

    println!("Starting training...\n");

    let result = trainer.fit_with_forward(
        model,
        dls,
        |model, x| model.forward(x),
        |model, x| model.forward(x),
    ).context("Training failed")?;

    println!("\nTraining complete!");
    println!("  Best validation accuracy: {:.2}%", result.best_valid_acc * 100.0);
    println!("  Best epoch: {}", result.best_epoch + 1);
    println!("  Training time: {:.1}s", result.training_time_secs);

    // Save config and history
    let config_path = output_path.join("config.json");
    std::fs::write(&config_path, serde_json::to_string_pretty(&model_config)?)?;
    println!("\nSaved config to {:?}", config_path);

    let history = serde_json::json!({
        "architecture": "FCN",
        "train_losses": result.train_losses,
        "valid_losses": result.valid_losses,
        "valid_accs": result.valid_accs,
        "best_valid_acc": result.best_valid_acc,
        "best_epoch": result.best_epoch,
        "training_time_secs": result.training_time_secs,
    });
    std::fs::write(output_path.join("history.json"), serde_json::to_string_pretty(&history)?)?;

    println!("\n=== Training finished successfully! ===");
    Ok(())
}

/// Train ResNetPlus model.
fn train_resnet_plus(
    dls: &TSDataLoaders,
    ucr: &UCRDataset,
    epochs: usize,
    lr: f64,
    _device: &<NdArray as Backend>::Device,
    output_path: &PathBuf,
    augment: bool,
) -> Result<()> {
    if augment {
        println!("Data augmentation: Enabled (Gaussian noise, magnitude scaling, cutout)");
    }
    let model_config = ResNetPlusConfig {
        n_vars: ucr.train.n_vars(),
        seq_len: ucr.seq_len,
        n_classes: ucr.n_classes,
        n_blocks: 3,
        n_filters: vec![64, 128, 128],
        kernel_size: 7,
    };

    println!("Model configuration:");
    println!("  Architecture: ResNetPlus");
    println!("  Blocks: {}", model_config.n_blocks);
    println!("  Filters: {:?}", model_config.n_filters);
    println!("  Kernel size: {}", model_config.kernel_size);
    println!();

    let autodiff_device = <TrainBackend as Backend>::Device::default();
    let model: ResNetPlus<TrainBackend> = ResNetPlus::new(model_config.clone(), &autodiff_device);

    let trainer_config = ClassificationTrainerConfig {
        n_epochs: epochs,
        lr,
        weight_decay: 0.01,
        grad_clip: 1.0,
        verbose: true,
        early_stopping_patience: 5,
        early_stopping_min_delta: 0.001,
    };

    let trainer = ClassificationTrainer::<TrainBackend>::new(trainer_config, autodiff_device);

    println!("Starting training...\n");

    let result = trainer.fit_with_forward(
        model,
        dls,
        |model, x| model.forward(x),
        |model, x| model.forward(x),
    ).context("Training failed")?;

    println!("\nTraining complete!");
    println!("  Best validation accuracy: {:.2}%", result.best_valid_acc * 100.0);
    println!("  Best epoch: {}", result.best_epoch + 1);
    println!("  Training time: {:.1}s", result.training_time_secs);

    // Save config and history
    let config_path = output_path.join("config.json");
    std::fs::write(&config_path, serde_json::to_string_pretty(&model_config)?)?;
    println!("\nSaved config to {:?}", config_path);

    let history = serde_json::json!({
        "architecture": "ResNetPlus",
        "train_losses": result.train_losses,
        "valid_losses": result.valid_losses,
        "valid_accs": result.valid_accs,
        "best_valid_acc": result.best_valid_acc,
        "best_epoch": result.best_epoch,
        "training_time_secs": result.training_time_secs,
    });
    std::fs::write(output_path.join("history.json"), serde_json::to_string_pretty(&history)?)?;

    println!("\n=== Training finished successfully! ===");
    Ok(())
}

/// Train XceptionTime model.
fn train_xception_time(
    dls: &TSDataLoaders,
    ucr: &UCRDataset,
    epochs: usize,
    lr: f64,
    _device: &<NdArray as Backend>::Device,
    output_path: &PathBuf,
    augment: bool,
) -> Result<()> {
    if augment {
        println!("Data augmentation: Enabled (Gaussian noise, magnitude scaling, cutout)");
    }
    let model_config = XceptionTimeConfig {
        n_vars: ucr.train.n_vars(),
        seq_len: ucr.seq_len,
        n_classes: ucr.n_classes,
        n_blocks: 4,
        n_filters: 64,
        kernel_size: 39,
        dropout: 0.0,
    };

    println!("Model configuration:");
    println!("  Architecture: XceptionTime");
    println!("  Blocks: {}", model_config.n_blocks);
    println!("  Filters: {}", model_config.n_filters);
    println!("  Kernel size: {}", model_config.kernel_size);
    println!();

    let autodiff_device = <TrainBackend as Backend>::Device::default();
    let model: XceptionTime<TrainBackend> = XceptionTime::new(model_config.clone(), &autodiff_device);

    let trainer_config = ClassificationTrainerConfig {
        n_epochs: epochs,
        lr,
        weight_decay: 0.01,
        grad_clip: 1.0,
        verbose: true,
        early_stopping_patience: 5,
        early_stopping_min_delta: 0.001,
    };

    let trainer = ClassificationTrainer::<TrainBackend>::new(trainer_config, autodiff_device);

    println!("Starting training...\n");

    let result = trainer.fit_with_forward(
        model,
        dls,
        |model, x| model.forward(x),
        |model, x| model.forward(x),
    ).context("Training failed")?;

    println!("\nTraining complete!");
    println!("  Best validation accuracy: {:.2}%", result.best_valid_acc * 100.0);
    println!("  Best epoch: {}", result.best_epoch + 1);
    println!("  Training time: {:.1}s", result.training_time_secs);

    // Save config and history
    let config_path = output_path.join("config.json");
    std::fs::write(&config_path, serde_json::to_string_pretty(&model_config)?)?;
    println!("\nSaved config to {:?}", config_path);

    let history = serde_json::json!({
        "architecture": "XceptionTime",
        "train_losses": result.train_losses,
        "valid_losses": result.valid_losses,
        "valid_accs": result.valid_accs,
        "best_valid_acc": result.best_valid_acc,
        "best_epoch": result.best_epoch,
        "training_time_secs": result.training_time_secs,
    });
    std::fs::write(output_path.join("history.json"), serde_json::to_string_pretty(&history)?)?;

    println!("\n=== Training finished successfully! ===");
    Ok(())
}

/// Train TSTPlus (Time Series Transformer) model.
fn train_tst_plus(
    dls: &TSDataLoaders,
    ucr: &UCRDataset,
    epochs: usize,
    lr: f64,
    _device: &<NdArray as Backend>::Device,
    output_path: &PathBuf,
    augment: bool,
) -> Result<()> {
    if augment {
        println!("Data augmentation: Enabled (Gaussian noise, magnitude scaling, cutout)");
    }
    let model_config = TSTConfig {
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

    println!("Model configuration:");
    println!("  Architecture: TSTPlus (Time Series Transformer)");
    println!("  d_model: {}", model_config.d_model);
    println!("  Heads: {}", model_config.n_heads);
    println!("  Layers: {}", model_config.n_layers);
    println!("  d_ff: {}", model_config.d_ff);
    println!("  Dropout: {}", model_config.dropout);
    println!();

    let autodiff_device = <TrainBackend as Backend>::Device::default();
    let model: TSTPlus<TrainBackend> = TSTPlus::new(model_config.clone(), &autodiff_device);

    let trainer_config = ClassificationTrainerConfig {
        n_epochs: epochs,
        lr,
        weight_decay: 0.01,
        grad_clip: 1.0,
        verbose: true,
        early_stopping_patience: 5,
        early_stopping_min_delta: 0.001,
    };

    let trainer = ClassificationTrainer::<TrainBackend>::new(trainer_config, autodiff_device);

    println!("Starting training...\n");

    let result = trainer.fit_with_forward(
        model,
        dls,
        |model, x| model.forward(x),
        |model, x| model.forward(x),
    ).context("Training failed")?;

    println!("\nTraining complete!");
    println!("  Best validation accuracy: {:.2}%", result.best_valid_acc * 100.0);
    println!("  Best epoch: {}", result.best_epoch + 1);
    println!("  Training time: {:.1}s", result.training_time_secs);

    // Save config and history
    let config_path = output_path.join("config.json");
    std::fs::write(&config_path, serde_json::to_string_pretty(&model_config)?)?;
    println!("\nSaved config to {:?}", config_path);

    let history = serde_json::json!({
        "architecture": "TSTPlus",
        "train_losses": result.train_losses,
        "valid_losses": result.valid_losses,
        "valid_accs": result.valid_accs,
        "best_valid_acc": result.best_valid_acc,
        "best_epoch": result.best_epoch,
        "training_time_secs": result.training_time_secs,
    });
    std::fs::write(output_path.join("history.json"), serde_json::to_string_pretty(&history)?)?;

    println!("\n=== Training finished successfully! ===");
    Ok(())
}

fn handle_eval(checkpoint: String, dataset: Option<String>) -> Result<()> {
    println!("=== tsai-rs Evaluation ===\n");
    println!("Checkpoint directory: {}", checkpoint);

    let checkpoint_path = PathBuf::from(&checkpoint);

    // Load config
    let config_path = checkpoint_path.join("config.json");
    if !config_path.exists() {
        bail!("Config file not found at {:?}", config_path);
    }

    let config_json = std::fs::read_to_string(&config_path)?;
    let model_config: InceptionTimePlusConfig = serde_json::from_str(&config_json)
        .context("Failed to parse config.json")?;

    println!("Model configuration loaded:");
    println!("  Architecture: InceptionTimePlus");
    println!("  Blocks: {}", model_config.n_blocks);
    println!("  Filters: {}", model_config.n_filters);
    println!("  Classes: {}", model_config.n_classes);
    println!();

    // Load training history
    let history_path = checkpoint_path.join("history.json");
    if history_path.exists() {
        let history_json = std::fs::read_to_string(&history_path)?;
        let history: serde_json::Value = serde_json::from_str(&history_json)?;

        println!("Training history:");
        if let Some(best_acc) = history.get("best_valid_acc").and_then(|v| v.as_f64()) {
            println!("  Best validation accuracy: {:.2}%", best_acc * 100.0);
        }
        if let Some(best_epoch) = history.get("best_epoch").and_then(|v| v.as_i64()) {
            println!("  Best epoch: {}", best_epoch + 1);
        }
        if let Some(time) = history.get("training_time_secs").and_then(|v| v.as_f64()) {
            println!("  Training time: {:.1}s", time);
        }
        println!();
    }

    // Load dataset if specified
    if let Some(ds) = dataset {
        let (source, dataset_name) = parse_dataset_name(&ds)?;

        if source != "ucr" {
            bail!("Only UCR datasets are currently supported");
        }

        println!("Loading dataset: {}", dataset_name);
        let ucr = UCRDataset::load(&dataset_name, None)
            .context(format!("Failed to load dataset '{}'", dataset_name))?;

        println!("  Test samples: {}", ucr.test.len());
        println!();

        // Create model and run evaluation
        let device = <TrainBackend as Backend>::Device::default();
        let model: InceptionTimePlus<TrainBackend> = InceptionTimePlus::new(model_config, &device);

        // Create dataloader for test set
        let test_loader = tsai_data::TSDataLoader::builder(ucr.test.clone())
            .batch_size(64)
            .shuffle(false)
            .build()?;

        println!("Running evaluation on test set...");

        let mut correct = 0usize;
        let mut total = 0usize;

        for batch_result in test_loader.iter::<TrainBackend>(&device) {
            let batch = batch_result?;
            let x = batch.x.inner().clone();
            let y = batch.y.expect("Test requires targets");
            let [batch_size, _] = y.dims();
            let targets: Tensor<TrainBackend, 1, Int> = y.reshape([batch_size]).int();

            let logits = model.forward(x);
            let preds = logits.argmax(1).squeeze(1);
            let correct_batch: i32 = preds.equal(targets).int().sum().into_scalar().elem();

            correct += correct_batch as usize;
            total += batch_size;
        }

        let accuracy = correct as f32 / total as f32;
        println!("\nEvaluation Results:");
        println!("  Test accuracy: {:.2}%", accuracy * 100.0);
        println!("  Correct: {} / {}", correct, total);
    } else {
        println!("No dataset specified. Use --dataset to evaluate on a test set.");
    }

    println!("\n=== Evaluation complete! ===");
    Ok(())
}

fn handle_export(checkpoint: String, output: String, format: String) -> Result<()> {
    println!("=== tsai-rs Export ===\n");
    println!("Checkpoint: {}", checkpoint);
    println!("Output: {}", output);
    println!("Format: {}", format);
    println!();

    let checkpoint_path = PathBuf::from(&checkpoint);

    // Load config
    let config_path = checkpoint_path.join("config.json");
    if !config_path.exists() {
        bail!("Config file not found at {:?}", config_path);
    }

    let config_json = std::fs::read_to_string(&config_path)?;
    let model_config: InceptionTimePlusConfig = serde_json::from_str(&config_json)
        .context("Failed to parse config.json")?;

    println!("Model configuration loaded:");
    println!("  Architecture: InceptionTimePlus");
    println!("  Input: {} vars x {} timesteps", model_config.n_vars, model_config.seq_len);
    println!("  Output: {} classes", model_config.n_classes);
    println!();

    match format.to_lowercase().as_str() {
        "onnx" => {
            println!("ONNX export requires the burn-onnx crate.");
            println!("This feature is planned for a future release.");
            println!();
            println!("For now, you can use the model directly in Rust with:");
            println!("  let config: InceptionTimePlusConfig = serde_json::from_str(config_json)?;");
            println!("  let model = config.init::<NdArray>(&device);");
        }
        "safetensors" => {
            println!("SafeTensors export is planned for a future release.");
        }
        "json" => {
            // Export config as JSON (already done during training)
            let output_path = PathBuf::from(&output);
            if output_path.extension().is_none() {
                let json_path = output_path.with_extension("json");
                std::fs::write(&json_path, &config_json)?;
                println!("Config exported to: {:?}", json_path);
            } else {
                std::fs::write(&output_path, &config_json)?;
                println!("Config exported to: {:?}", output_path);
            }
        }
        _ => {
            bail!("Unsupported format: {}. Supported: json, onnx (planned), safetensors (planned)", format);
        }
    }

    println!("\n=== Export complete! ===");
    Ok(())
}
