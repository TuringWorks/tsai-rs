# tsai_rs - Python bindings for tsai-rs

Python bindings for the tsai-rs time series deep learning library.

## Installation

### From source (requires Rust)

```bash
# Install maturin
pip install maturin

# Build and install
cd crates/tsai_python
maturin develop --release
```

### Prerequisites

- Python >= 3.8
- Rust >= 1.75
- maturin >= 1.4

## Quick Start

```python
import tsai_rs
import numpy as np

# Configure a model
config = tsai_rs.InceptionTimePlusConfig(
    n_vars=1,
    seq_len=100,
    n_classes=5
)
print(config)
# InceptionTimePlusConfig(n_vars=1, seq_len=100, n_classes=5, n_blocks=6, n_filters=32)

# Compute confusion matrix from predictions
preds = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
targets = np.array([0, 1, 1, 0, 2, 2, 0, 0, 2, 1])
cm = tsai_rs.confusion_matrix(preds, targets, n_classes=3)
print(f"Accuracy: {cm.accuracy():.2%}")
print(f"Macro F1: {cm.macro_f1():.4f}")

# Convert time series to image (GASF)
series = np.sin(np.linspace(0, 4*np.pi, 50)).astype(np.float32)
gasf_image = tsai_rs.compute_gasf(series)
print(f"GASF shape: {gasf_image.shape}")  # (50, 50)
```

## Features

### Model Configurations

- `InceptionTimePlusConfig` - InceptionTime CNN configuration
- `PatchTSTConfig` - Patch Time Series Transformer configuration
- `RNNPlusConfig` - RNN (LSTM/GRU) configuration

### Training Utilities

- `LearnerConfig` - Training hyperparameters
- `OneCycleLR` - One-cycle learning rate scheduler

### Analysis Tools

- `confusion_matrix()` - Compute confusion matrix with metrics
- `top_losses()` - Find hardest examples

### Time Series to Image Transforms

- `compute_gasf()` - Gramian Angular Summation Field
- `compute_gadf()` - Gramian Angular Difference Field
- `compute_recurrence_plot()` - Recurrence Plot

## Examples

See the `examples/` directory for complete examples:

- `classification_example.py` - Time series classification workflow
- `forecasting_example.py` - Time series forecasting workflow

## License

Apache-2.0
