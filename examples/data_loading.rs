//! Example: Data Loading from Various Formats
//!
//! This example demonstrates how to load time series data from:
//! - NumPy files (.npy, .npz)
//! - CSV files
//! - Parquet files
//! - In-memory arrays
//!
//! Run with: cargo run --example data_loading

use ndarray::{Array2, Array3};
use std::path::PathBuf;
use tsai_core::Seed;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Data Loading Examples ===\n");

    // Example 1: Create data from in-memory arrays
    println!("1. Loading from in-memory arrays");
    println!("---------------------------------");
    from_arrays_example()?;

    // Example 2: Loading from NumPy files (demonstration)
    println!("\n2. Loading from NumPy files");
    println!("---------------------------");
    from_numpy_example();

    // Example 3: Loading from CSV files (demonstration)
    println!("\n3. Loading from CSV files");
    println!("-------------------------");
    from_csv_example();

    // Example 4: Loading from Parquet files (demonstration)
    println!("\n4. Loading from Parquet files");
    println!("-----------------------------");
    from_parquet_example();

    // Example 5: Data preprocessing
    println!("\n5. Data preprocessing");
    println!("---------------------");
    preprocessing_example()?;

    println!("\n=== Data loading examples complete ===");
    Ok(())
}

/// Example 1: Creating datasets from in-memory arrays
fn from_arrays_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create sample data: 100 samples, 3 variables, 50 timesteps
    let x = Array3::<f32>::zeros((100, 3, 50));
    let y = Array2::<f32>::zeros((100, 1));

    println!("Creating dataset from arrays:");
    println!("  X shape: {:?}", x.shape());
    println!("  y shape: {:?}", y.shape());

    // Create dataset
    let dataset = tsai_data::TSDataset::from_arrays(x, Some(y))?;

    println!("  Dataset size: {}", dataset.len());
    println!("  Has targets: {}", dataset.has_targets());

    // Access individual samples
    if let Ok((x_sample, y_sample)) = dataset.get(0) {
        println!("  First sample X shape: {:?}", x_sample.shape());
        if let Some(y) = y_sample {
            println!("  First sample y shape: {:?}", y.shape());
        }
    }

    // Create dataset without targets (for inference)
    let x_only = Array3::<f32>::zeros((50, 3, 50));
    let inference_dataset = tsai_data::TSDataset::from_arrays(x_only, None)?;
    println!("  Inference dataset (no targets): {} samples", inference_dataset.len());

    Ok(())
}

/// Example 2: Loading from NumPy files
fn from_numpy_example() {
    println!("To load from NumPy files:\n");

    println!(r#"```rust
// Load single .npy file
let x: Array3<f32> = tsai_data::read_npy("data/X_train.npy")?;
let y: Array2<f32> = tsai_data::read_npy("data/y_train.npy")?;

// Load from .npz archive
let data = tsai_data::read_npz("data/dataset.npz")?;
// Access arrays: data["X"], data["y"]

// Create dataset
let dataset = tsai_data::TSDataset::from_arrays(x, Some(y))?;
```"#);

    // Show expected shapes
    println!("\nExpected array shapes:");
    println!("  X: (n_samples, n_variables, sequence_length)");
    println!("  y: (n_samples, n_outputs)");

    // Demonstrate cache directory
    println!("\nCache directory: {:?}", tsai_data::cache_dir());
}

/// Example 3: Loading from CSV files
fn from_csv_example() {
    println!("To load from CSV files:\n");

    println!(r#"```rust
// Load CSV with time series data
// Expected format: each row is a flattened time series
let df = tsai_data::read_csv("data/train.csv")?;

// For multi-variable time series, reshape accordingly:
// CSV shape: (n_samples, n_vars * seq_len)
// Reshape to: (n_samples, n_vars, seq_len)

// Using TSDataset builder
let dataset = tsai_data::TSDataset::builder()
    .from_csv("data/train.csv")?
    .target_column("label")
    .build()?;
```"#);

    println!("\nCSV format tips:");
    println!("  - Wide format: each row = one sample, columns = features");
    println!("  - Long format: (sample_id, time, var, value) - needs reshaping");
    println!("  - Target column can be named (e.g., 'label', 'target', 'class')");
}

/// Example 4: Loading from Parquet files
fn from_parquet_example() {
    println!("To load from Parquet files:\n");

    println!(r#"```rust
// Load Parquet file
let data = tsai_data::read_parquet("data/train.parquet")?;

// Parquet is efficient for large datasets
// Supports compression and columnar storage

// Example with partitioned data
let train = tsai_data::read_parquet("data/train.parquet")?;
let test = tsai_data::read_parquet("data/test.parquet")?;
```"#);

    println!("\nParquet benefits:");
    println!("  - Efficient compression");
    println!("  - Fast columnar reads");
    println!("  - Schema preservation");
    println!("  - Good for large datasets");
}

/// Example 5: Data preprocessing
fn preprocessing_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create sample data
    let x = create_sample_data(100, 3, 50);
    let y = Array2::<f32>::zeros((100, 1));

    let dataset = tsai_data::TSDataset::from_arrays(x, Some(y))?;

    // Split data
    println!("Train/test split:");
    let (train_ds, test_ds) = tsai_data::train_test_split(&dataset, 0.2, Seed::new(42))?;
    println!("  Train: {} samples (80%)", train_ds.len());
    println!("  Test: {} samples (20%)", test_ds.len());

    // Three-way split
    println!("\nTrain/valid/test split:");
    let (train2, valid, test2) = tsai_data::train_valid_test_split(
        &dataset,
        0.15,  // 15% validation
        0.15,  // 15% test
        Seed::new(42),
    )?;
    println!("  Train: {} samples (70%)", train2.len());
    println!("  Valid: {} samples (15%)", valid.len());
    println!("  Test: {} samples (15%)", test2.len());

    // Create dataloaders with different configurations
    println!("\nDataLoader configurations:");

    // Basic dataloader
    let basic_dls = tsai_data::TSDataLoaders::builder(train_ds.clone(), test_ds.clone())
        .batch_size(32)
        .build()?;
    println!("  Basic: batch_size=32");

    // With stratified sampling
    let stratified_dls = tsai_data::TSDataLoaders::builder(train_ds.clone(), test_ds.clone())
        .batch_size(16)
        .shuffle_train(true)
        .seed(Seed::new(42))
        .build()?;
    println!("  Stratified: batch_size=16, shuffled, seeded");

    // Show batch info
    println!("\nBatch information:");
    println!("  Train batches: {}", basic_dls.train().n_batches());
    println!("  Valid batches: {}", basic_dls.valid().n_batches());

    Ok(())
}

/// Helper to create sample data
fn create_sample_data(n_samples: usize, n_vars: usize, seq_len: usize) -> Array3<f32> {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut x = Array3::<f32>::zeros((n_samples, n_vars, seq_len));

    for i in 0..n_samples {
        for v in 0..n_vars {
            for t in 0..seq_len {
                x[[i, v, t]] = rng.gen_range(-1.0..1.0);
            }
        }
    }

    x
}
