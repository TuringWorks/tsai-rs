# tsai-rs

**Time series deep learning in Rust** — a feature-parity port of Python [tsai](https://github.com/timeseriesAI/tsai).

[![Crates.io](https://img.shields.io/crates/v/tsai.svg)](https://crates.io/crates/tsai)
[![Documentation](https://docs.rs/tsai/badge.svg)](https://docs.rs/tsai)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

## Features

- **Comprehensive Model Zoo**: 16+ architectures including InceptionTimePlus, PatchTST, TSTPlus, MiniRocket, RNNPlus, FCN, and more
- **Data Augmentation**: 17+ time series transforms (noise, warping, masking, SpecAugment, etc.)
- **Training Framework**: 8 callbacks, 9 schedulers, 9 metrics, 5 loss functions, and checkpointing
- **Analysis Tools**: Confusion matrix, top losses, permutation importance
- **Explainability**: GradCAM, attention visualization, attribution maps
- **Multiple Backends**: CPU (ndarray), GPU (WGPU/Metal), Apple MLX, or PyTorch (tch)
- **Python Bindings**: Use from Python via `tsai_rs` package

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
tsai = "0.1"
```

### Classification Example

```rust
use tsai::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load data
    let x_train = read_npy("data/X_train.npy")?;
    let y_train = read_npy("data/y_train.npy")?;

    // Create dataset and split
    let dataset = TSDataset::from_arrays(x_train, Some(y_train))?;
    let (train_ds, valid_ds) = train_test_split(&dataset, 0.2, Seed::new(42))?;

    // Create dataloaders
    let dls = TSDataLoaders::builder(train_ds, valid_ds)
        .batch_size(64)
        .seed(Seed::new(42))
        .build()?;

    // Configure model
    let config = InceptionTimePlusConfig::new(
        dls.n_vars(),    // number of input variables
        dls.seq_len(),   // sequence length
        5,               // number of classes
    );

    // Initialize and train (requires backend-specific code)
    println!("Model configured: {:?}", config);

    Ok(())
}
```

### Using sklearn-like API

```rust
use tsai::compat::{TSClassifier, TSClassifierConfig};

let mut clf = TSClassifier::new(TSClassifierConfig {
    arch: "InceptionTimePlus".to_string(),
    n_epochs: 25,
    lr: 1e-3,
    ..Default::default()
});

clf.fit(&x_train, &y_train)?;
let predictions = clf.predict(&x_test)?;
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `backend-ndarray` (default) | CPU backend using ndarray |
| `backend-wgpu` | GPU backend (Metal on macOS, Vulkan on Linux/Windows) |
| `backend-mlx` | Native Apple Silicon GPU via MLX (macOS only) |
| `backend-tch` | PyTorch backend via tch-rs |
| `wandb` | Weights & Biases integration |

Enable GPU support:

```toml
# Cross-platform GPU (recommended for most users)
[dependencies]
tsai = { version = "0.1", features = ["backend-wgpu"] }

# Native Apple Silicon (M1/M2/M3/M4 Macs)
[dependencies]
tsai = { version = "0.1", features = ["backend-mlx"] }
```

## Model Zoo

### CNN Models
- `InceptionTimePlus` - InceptionTime with improvements
- `ResNetPlus` - ResNet adapted for time series
- `XceptionTimePlus` - Xception-inspired architecture
- `OmniScaleCNN` - Multi-scale CNN
- `XCMPlus` - Explainable CNN
- `FCN` - Fully Convolutional Network

### Transformer Models
- `TSTPlus` - Time Series Transformer
- `TSiTPlus` - Improved TS Transformer with multiple PE options
- `TSPerceiver` - Perceiver for time series
- `PatchTST` - Patch-based Transformer

### ROCKET Family
- `MiniRocket` - Fast random convolutional features

### RNN Models
- `RNNPlus` - LSTM/GRU with improvements

### Tabular Models
- `TabTransformer` - Transformer for tabular data

## Data Formats

tsai-rs supports multiple data formats:

```rust
// NumPy
let x = read_npy("data.npy")?;
let (x, y) = read_npz("data.npz")?;

// CSV
let dataset = read_csv("data.csv", n_vars, seq_len, has_labels)?;

// Parquet
let dataset = read_parquet("data.parquet", &x_cols, y_col, n_vars, seq_len)?;
```

## Transforms

Apply data augmentation during training:

```rust
use tsai::transforms::{Compose, GaussianNoise, TimeWarp, MagScale};

let transform = Compose::new()
    .add(GaussianNoise::new(0.1))
    .add(TimeWarp::new(0.2))
    .add(MagScale::new(1.2));
```

Available transforms include:
- Noise: `GaussianNoise`
- Warping: `TimeWarp`, `WindowWarp`, `MagWarp`
- Masking: `CutOut`, `FrequencyMask`, `TimeMask`
- Temporal: `HorizontalFlip`, `RandomShift`, `Rotation`, `Permutation`
- SpecAugment: `SpecAugment`, `TSRandomShift`, `TSHorizontalFlip`, `TSVerticalFlip`
- Mixing: `MixUp1d`, `CutMix1d`, `IntraClassCutMix1d`
- Imaging: `TSToRP`, `TSToGASF`, `TSToGADF`, `TSToMTF`

## CLI

```bash
# Install CLI
cargo install tsai_cli

# List available datasets
tsai datasets list

# Fetch a dataset
tsai datasets fetch ucr:NATOPS

# Train a model
tsai train --arch InceptionTimePlus --dataset ./data --epochs 25

# Evaluate
tsai eval --checkpoint ./runs/best_model.safetensors
```

## Examples

See the `examples/` directory for more:

- `ucr_inception_time.rs` - UCR classification with InceptionTimePlus
- `simple_classification.rs` - Basic classification example
- `forecasting.rs` - Time series forecasting
- `sklearn_api.rs` - sklearn-like API demonstration
- `train_ucr_metal.rs` - GPU training with WGPU/Metal backend
- `train_ucr_mlx.rs` - Apple MLX backend example
- `compare_models.rs` - Model comparison on UCR datasets

## Python Bindings

Use tsai-rs from Python via the `tsai_rs` package:

```bash
# Build from source (requires Rust)
cd crates/tsai_python
pip install maturin
maturin develop --release
```

```python
import tsai_rs
import numpy as np

# Configure a model
config = tsai_rs.InceptionTimePlusConfig(n_vars=1, seq_len=100, n_classes=5)

# Compute confusion matrix
preds = np.array([0, 1, 2, 0, 1])
targets = np.array([0, 1, 1, 0, 2])
cm = tsai_rs.confusion_matrix(preds, targets, n_classes=3)
print(f"Accuracy: {cm.accuracy():.2%}")

# Time series to image transforms
series = np.sin(np.linspace(0, 4*np.pi, 50)).astype(np.float32)
gasf_image = tsai_rs.compute_gasf(series)
```

## Benchmarks

Run benchmarks:

```bash
cargo bench
```

## License

Apache-2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [tsai](https://github.com/timeseriesAI/tsai) - The original Python library
- [Burn](https://github.com/tracel-ai/burn) - Rust deep learning framework
- Research papers cited in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

## Contributing

Contributions welcome! Please read our contributing guidelines before submitting PRs.
