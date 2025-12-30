#!/usr/bin/env python3
"""
Time Series Forecasting Example using tsai_rs

This example demonstrates how to use tsai_rs for time series forecasting:
1. Generate synthetic forecasting data
2. Configure a PatchTST model for forecasting
3. Set up training configuration with appropriate schedule
4. Compare different model architectures

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


def generate_forecasting_data(
    n_samples: int,
    n_vars: int,
    lookback: int,
    horizon: int,
    seed: int = 42
):
    """Generate synthetic multivariate time series for forecasting.

    Creates data with trend, seasonality, and noise.
    """
    np.random.seed(seed)

    X = np.zeros((n_samples, n_vars, lookback), dtype=np.float32)
    y = np.zeros((n_samples, horizon), dtype=np.float32)

    for i in range(n_samples):
        base_t = i * 0.1

        # Generate lookback window for each variable
        for v in range(n_vars):
            phase = np.random.uniform(0, np.pi)
            for t in range(lookback):
                time = base_t + t * 0.1
                trend = time * 0.01
                seasonal = np.sin(time + phase) * 0.5
                noise = np.random.randn() * 0.1
                X[i, v, t] = trend + seasonal + noise

        # Generate target (future values of first variable)
        for t in range(horizon):
            time = base_t + (lookback + t) * 0.1
            trend = time * 0.01
            seasonal = np.sin(time) * 0.5
            noise = np.random.randn() * 0.05
            y[i, t] = trend + seasonal + noise

    return X, y


def main():
    print("=" * 60)
    print("Time Series Forecasting with tsai_rs")
    print("=" * 60)
    print()

    # Configuration
    n_samples = 500
    n_vars = 3          # Multivariate input
    lookback = 96       # Use 96 past observations
    horizon = 24        # Predict 24 future steps

    print("Forecasting Configuration:")
    print(f"  Samples: {n_samples}")
    print(f"  Variables: {n_vars}")
    print(f"  Lookback window: {lookback} steps")
    print(f"  Forecast horizon: {horizon} steps")
    print()

    # Generate data
    print("Generating synthetic data...")
    X_train, y_train = generate_forecasting_data(400, n_vars, lookback, horizon, seed=42)
    X_val, y_val = generate_forecasting_data(50, n_vars, lookback, horizon, seed=123)
    X_test, y_test = generate_forecasting_data(50, n_vars, lookback, horizon, seed=456)

    print(f"  Train: {X_train.shape[0]} samples, X={X_train.shape}, y={y_train.shape}")
    print(f"  Valid: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    print()

    # Configure PatchTST model
    print("Configuring PatchTST model for forecasting...")
    patchtst_config = tsai_rs.PatchTSTConfig.for_forecasting(n_vars, lookback, horizon)

    print(f"  {patchtst_config}")
    print(f"  n_vars: {patchtst_config.n_vars}")
    print(f"  seq_len: {patchtst_config.seq_len}")
    print(f"  n_outputs (horizon): {patchtst_config.n_outputs}")
    print(f"  patch_len: {patchtst_config.patch_len}")
    print(f"  stride: {patchtst_config.stride}")
    print(f"  n_patches: {patchtst_config.n_patches}")
    print(f"  d_model: {patchtst_config.d_model}")
    print(f"  n_heads: {patchtst_config.n_heads}")
    print(f"  n_layers: {patchtst_config.n_layers}")
    print()

    # Compare with RNN model
    print("Configuring RNNPlus model for comparison...")
    rnn_config = tsai_rs.RNNPlusConfig(
        n_vars=n_vars,
        seq_len=lookback,
        n_outputs=horizon,
        hidden_dim=128,
        n_layers=2,
        rnn_type="lstm",
        bidirectional=True,
        dropout=0.1
    )
    print(f"  {rnn_config}")
    print()

    # Training configuration
    print("Training configuration...")
    train_config = tsai_rs.LearnerConfig(
        lr=1e-4,           # Lower LR for transformers
        weight_decay=0.05, # Regularization
        grad_clip=1.0,
        mixed_precision=False
    )
    print(f"  {train_config}")
    print()

    # Learning rate schedule
    print("Learning rate schedule...")
    n_epochs = 100
    batch_size = 32
    steps_per_epoch = 400 // batch_size
    total_steps = n_epochs * steps_per_epoch

    scheduler = tsai_rs.OneCycleLR.simple(train_config.lr, total_steps)

    print(f"  Epochs: {n_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print()

    # Get full learning rate schedule
    lr_schedule = scheduler.get_lr_schedule(total_steps)
    print(f"  LR schedule shape: {lr_schedule.shape}")
    print(f"  LR range: [{lr_schedule.min():.2e}, {lr_schedule.max():.2e}]")
    print()

    # Show sample learning rates at key epochs
    print("  LR at key epochs:")
    for epoch in [0, 25, 50, 75, 99]:
        step = epoch * steps_per_epoch
        print(f"    Epoch {epoch:3d}: LR = {scheduler.get_lr(step):.2e}")
    print()

    # Simulated forecasting evaluation
    print("Simulating forecasting predictions...")

    # In practice, predictions would come from trained model
    # Here we simulate with simple baseline predictions
    y_pred = np.zeros_like(y_test)
    for i in range(len(y_test)):
        # Naive baseline: repeat last value
        y_pred[i, :] = X_test[i, 0, -1]

    # Compute metrics
    mse = np.mean((y_pred - y_test) ** 2)
    mae = np.mean(np.abs(y_pred - y_test))
    rmse = np.sqrt(mse)

    print(f"  Baseline metrics (naive last-value):")
    print(f"    MSE: {mse:.6f}")
    print(f"    MAE: {mae:.6f}")
    print(f"    RMSE: {rmse:.6f}")
    print()

    # Time series imaging for feature extraction
    print("Time series imaging (can be used for CNN-based forecasting)...")

    sample_series = X_test[0, 0, :].astype(np.float32)

    gasf = tsai_rs.compute_gasf(sample_series)
    gadf = tsai_rs.compute_gadf(sample_series)
    rp = tsai_rs.compute_recurrence_plot(sample_series, threshold=0.15)

    print(f"  GASF shape: {gasf.shape}")
    print(f"  GADF shape: {gadf.shape}")
    print(f"  Recurrence plot shape: {rp.shape}")
    print()

    # Optional visualization
    if HAS_MATPLOTLIB:
        print("Generating visualizations...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Sample forecasting example
        sample_idx = 0
        x_plot = np.arange(lookback)
        y_plot = np.arange(lookback, lookback + horizon)

        axes[0, 0].plot(x_plot, X_test[sample_idx, 0, :], 'b-', label='History (var 0)', linewidth=2)
        axes[0, 0].plot(y_plot, y_test[sample_idx, :], 'g-', label='Ground truth', linewidth=2)
        axes[0, 0].plot(y_plot, y_pred[sample_idx, :], 'r--', label='Prediction (baseline)', linewidth=2)
        axes[0, 0].axvline(x=lookback, color='gray', linestyle='--', alpha=0.5)
        axes[0, 0].set_title("Forecasting Example")
        axes[0, 0].set_xlabel("Time step")
        axes[0, 0].set_ylabel("Value")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Learning rate schedule
        epochs = np.arange(n_epochs)
        epoch_lrs = [scheduler.get_lr(e * steps_per_epoch) for e in epochs]
        axes[0, 1].plot(epochs, epoch_lrs, 'b-', linewidth=2)
        axes[0, 1].set_title("One-Cycle Learning Rate Schedule")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Learning Rate")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')

        # Multivariate input
        for v in range(n_vars):
            axes[1, 0].plot(X_test[sample_idx, v, :], label=f'Variable {v}', alpha=0.8)
        axes[1, 0].set_title("Multivariate Input")
        axes[1, 0].set_xlabel("Time step")
        axes[1, 0].set_ylabel("Value")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # GASF image
        im = axes[1, 1].imshow(gasf, cmap='viridis', aspect='auto')
        axes[1, 1].set_title("GASF Image of First Variable")
        plt.colorbar(im, ax=axes[1, 1])

        plt.tight_layout()
        plt.savefig("forecasting_example.png", dpi=150)
        print("  Saved visualization to forecasting_example.png")
    else:
        print("  (matplotlib not installed, skipping visualization)")

    print()
    print("=" * 60)
    print("Forecasting example complete!")
    print("=" * 60)
    print()

    # Summary of model comparison
    print("Model Architecture Comparison:")
    print("-" * 50)
    print(f"PatchTST (Transformer):")
    print(f"  - Patches: {patchtst_config.n_patches}")
    print(f"  - Model dim: {patchtst_config.d_model}")
    print(f"  - Attention heads: {patchtst_config.n_heads}")
    print(f"  - Layers: {patchtst_config.n_layers}")
    print()
    print(f"RNNPlus (LSTM):")
    print(f"  - Hidden dim: 128")
    print(f"  - Layers: 2")
    print(f"  - Bidirectional: Yes")
    print()

    print(f"tsai_rs version: {tsai_rs.version()}")


if __name__ == "__main__":
    main()
