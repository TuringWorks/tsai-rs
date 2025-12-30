#!/usr/bin/env python3
"""
Time Series Classification Example using tsai_rs

This example demonstrates how to use tsai_rs for time series classification:
1. Generate synthetic classification data
2. Configure an InceptionTimePlus model
3. Set up training configuration
4. Evaluate predictions using confusion matrix and top losses
5. Visualize time series as images (GASF)

Prerequisites:
    pip install numpy matplotlib
    cd crates/tsai_python && maturin develop --release
"""

import numpy as np
import tsai_rs

# Optional: for visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def generate_synthetic_data(n_samples: int, seq_len: int, n_classes: int, seed: int = 42):
    """Generate synthetic time series classification data.

    Each class has a distinct frequency pattern.
    """
    np.random.seed(seed)

    X = np.zeros((n_samples, 1, seq_len), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)

    for i in range(n_samples):
        class_idx = i % n_classes
        y[i] = class_idx

        # Different frequency for each class
        freq = (class_idx + 1) * 0.2
        t = np.linspace(0, 4 * np.pi, seq_len)
        X[i, 0, :] = np.sin(freq * t) + np.random.randn(seq_len) * 0.1

    return X, y


def main():
    print("=" * 60)
    print("Time Series Classification with tsai_rs")
    print("=" * 60)
    print()

    # Configuration
    n_samples = 200
    seq_len = 100
    n_classes = 5

    print(f"Configuration:")
    print(f"  Samples: {n_samples}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Classes: {n_classes}")
    print()

    # Generate synthetic data
    print("Generating synthetic data...")
    X_train, y_train = generate_synthetic_data(160, seq_len, n_classes, seed=42)
    X_test, y_test = generate_synthetic_data(40, seq_len, n_classes, seed=123)

    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    print()

    # Configure model
    print("Configuring InceptionTimePlus model...")
    model_config = tsai_rs.InceptionTimePlusConfig(
        n_vars=1,
        seq_len=seq_len,
        n_classes=n_classes,
        n_blocks=6,
        n_filters=32,
        bottleneck_dim=32,
        dropout=0.1
    )
    print(f"  {model_config}")
    print(f"  Config JSON: {model_config.to_json()[:100]}...")
    print()

    # Configure training
    print("Configuring training...")
    train_config = tsai_rs.LearnerConfig(
        lr=1e-3,
        weight_decay=0.01,
        grad_clip=1.0,
        mixed_precision=False
    )
    print(f"  {train_config}")
    print()

    # Learning rate schedule
    print("Learning rate schedule (One-Cycle)...")
    n_epochs = 50
    steps_per_epoch = 160 // 32  # Assuming batch size 32
    total_steps = n_epochs * steps_per_epoch
    scheduler = tsai_rs.OneCycleLR.simple(train_config.lr, total_steps)

    print(f"  Total steps: {total_steps}")
    for epoch in [0, 10, 25, 40, 49]:
        step = epoch * steps_per_epoch
        print(f"  Epoch {epoch:2d}: LR = {scheduler.get_lr(step):.6f}")
    print()

    # Simulate predictions (in practice, these would come from training)
    print("Simulating model predictions...")
    np.random.seed(42)

    # Simulate predictions with 80% accuracy
    y_pred = y_test.copy()
    n_errors = int(0.2 * len(y_test))
    error_indices = np.random.choice(len(y_test), n_errors, replace=False)
    for idx in error_indices:
        y_pred[idx] = (y_pred[idx] + np.random.randint(1, n_classes)) % n_classes

    # Simulate probabilities
    probs = np.random.uniform(0.5, 1.0, len(y_test)).astype(np.float32)
    losses = 1.0 - probs

    # Compute confusion matrix
    print("\nEvaluating predictions...")
    cm = tsai_rs.confusion_matrix(y_pred, y_test, n_classes)

    print(f"  {cm}")
    print(f"  Accuracy: {cm.accuracy():.2%}")
    print(f"  Macro F1: {cm.macro_f1():.4f}")
    print()

    # Per-class metrics
    print("  Per-class metrics:")
    for c in range(n_classes):
        print(f"    Class {c}: P={cm.precision(c):.3f}, R={cm.recall(c):.3f}, F1={cm.f1(c):.3f}")
    print()

    # Find top losses
    print("Top 5 hardest examples:")
    top = tsai_rs.top_losses(losses, y_test, y_pred, probs, k=5)
    for i, t in enumerate(top):
        print(f"  {i+1}. {t}")
    print()

    # Time series to image transformation
    print("Converting time series to GASF image...")
    sample_series = X_test[0, 0, :].astype(np.float32)
    gasf_image = tsai_rs.compute_gasf(sample_series)
    gadf_image = tsai_rs.compute_gadf(sample_series)
    rp_image = tsai_rs.compute_recurrence_plot(sample_series, threshold=0.2)

    print(f"  Input series shape: ({len(sample_series)},)")
    print(f"  GASF image shape: {gasf_image.shape}")
    print(f"  GADF image shape: {gadf_image.shape}")
    print(f"  Recurrence plot shape: {rp_image.shape}")
    print()

    # Optional visualization
    if HAS_MATPLOTLIB:
        print("Generating visualizations...")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Time series
        axes[0, 0].plot(sample_series)
        axes[0, 0].set_title("Original Time Series")
        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel("Value")

        # GASF
        im1 = axes[0, 1].imshow(gasf_image, cmap='viridis', aspect='auto')
        axes[0, 1].set_title("GASF (Gramian Angular Summation Field)")
        plt.colorbar(im1, ax=axes[0, 1])

        # GADF
        im2 = axes[1, 0].imshow(gadf_image, cmap='viridis', aspect='auto')
        axes[1, 0].set_title("GADF (Gramian Angular Difference Field)")
        plt.colorbar(im2, ax=axes[1, 0])

        # Recurrence Plot
        im3 = axes[1, 1].imshow(rp_image, cmap='binary', aspect='auto')
        axes[1, 1].set_title("Recurrence Plot")
        plt.colorbar(im3, ax=axes[1, 1])

        plt.tight_layout()
        plt.savefig("classification_example.png", dpi=150)
        print("  Saved visualization to classification_example.png")
    else:
        print("  (matplotlib not installed, skipping visualization)")

    print()
    print("=" * 60)
    print("Classification example complete!")
    print("=" * 60)

    # Version info
    print(f"\ntsai_rs version: {tsai_rs.version()}")


if __name__ == "__main__":
    main()
