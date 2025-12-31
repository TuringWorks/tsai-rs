//! Temporal Convolutional Network (TCN).
//!
//! TCN uses causal dilated convolutions for sequence modeling,
//! enabling efficient training with long-range dependencies.
//!
//! Key features:
//! - Causal convolutions (no information leakage from future)
//! - Dilated convolutions with exponentially increasing dilation
//! - Residual connections for gradient flow
//! - Flexible receptive field via kernel size and dilation
//!
//! Reference: "An Empirical Evaluation of Generic Convolutional and Recurrent
//! Networks for Sequence Modeling" by Bai et al. (2018)

use burn::nn::{
    conv::{Conv1d, Conv1dConfig},
    Dropout, DropoutConfig, Linear, LinearConfig, Relu,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use serde::{Deserialize, Serialize};

/// Configuration for a TCN residual block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TCNBlockConfig {
    /// Input channels.
    pub in_channels: usize,
    /// Output channels.
    pub out_channels: usize,
    /// Kernel size for convolutions.
    pub kernel_size: usize,
    /// Dilation factor.
    pub dilation: usize,
    /// Dropout rate.
    pub dropout: f64,
}

impl TCNBlockConfig {
    /// Create a new TCN block config.
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, dilation: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            dilation,
            dropout: 0.1,
        }
    }

    /// Set dropout rate.
    #[must_use]
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Initialize the block.
    pub fn init<B: Backend>(&self, device: &B::Device) -> TCNBlock<B> {
        TCNBlock::new(self.clone(), device)
    }
}

/// TCN residual block with causal dilated convolutions.
///
/// Structure:
/// ```text
/// Input -> Conv1 -> ReLU -> Dropout -> Conv2 -> ReLU -> Dropout -> Add -> Output
///   |                                                               ^
///   +-------------------- (1x1 conv if needed) ---------------------+
/// ```
#[derive(Module, Debug)]
pub struct TCNBlock<B: Backend> {
    /// First dilated convolution.
    conv1: Conv1d<B>,
    /// Second dilated convolution.
    conv2: Conv1d<B>,
    /// Dropout layer.
    dropout: Dropout,
    /// Residual connection (1x1 conv if channels differ).
    residual: Option<Conv1d<B>>,
    /// Padding for causal convolution.
    #[module(skip)]
    padding: usize,
}

impl<B: Backend> TCNBlock<B> {
    /// Create a new TCN block.
    pub fn new(config: TCNBlockConfig, device: &B::Device) -> Self {
        // Causal padding: (kernel_size - 1) * dilation
        let padding = (config.kernel_size - 1) * config.dilation;

        // First conv: dilated causal convolution
        let conv1 = Conv1dConfig::new(config.in_channels, config.out_channels, config.kernel_size)
            .with_dilation(config.dilation)
            .with_padding(burn::nn::PaddingConfig1d::Explicit(padding))
            .init(device);

        // Second conv: same configuration
        let conv2 = Conv1dConfig::new(config.out_channels, config.out_channels, config.kernel_size)
            .with_dilation(config.dilation)
            .with_padding(burn::nn::PaddingConfig1d::Explicit(padding))
            .init(device);

        // Residual connection if channels differ
        let residual = if config.in_channels != config.out_channels {
            Some(
                Conv1dConfig::new(config.in_channels, config.out_channels, 1)
                    .init(device),
            )
        } else {
            None
        };

        let dropout = DropoutConfig::new(config.dropout).init();

        Self {
            conv1,
            conv2,
            dropout,
            residual,
            padding,
        }
    }

    /// Forward pass.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_batch, _, seq_len] = x.dims();

        // First convolution + ReLU + dropout
        let out = self.conv1.forward(x.clone());
        // Trim to make it causal (remove future context)
        let out_dims = out.dims();
        let out = out.slice([0..out_dims[0], 0..out_dims[1], 0..seq_len]);
        let out = Relu::new().forward(out);
        let out = self.dropout.forward(out);

        // Second convolution + ReLU + dropout
        let out = self.conv2.forward(out);
        let out_dims = out.dims();
        let out = out.slice([0..out_dims[0], 0..out_dims[1], 0..seq_len]);
        let out = Relu::new().forward(out);
        let out = self.dropout.forward(out);

        // Residual connection
        let residual = match &self.residual {
            Some(conv) => conv.forward(x),
            None => x,
        };

        // Add residual and apply ReLU
        Relu::new().forward(out + residual)
    }
}

/// Configuration for TCN model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TCNConfig {
    /// Number of input variables/channels.
    pub n_vars: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// Number of channels per layer.
    pub n_channels: Vec<usize>,
    /// Kernel size.
    pub kernel_size: usize,
    /// Dropout rate.
    pub dropout: f64,
}

impl Default for TCNConfig {
    fn default() -> Self {
        Self {
            n_vars: 1,
            seq_len: 100,
            n_classes: 2,
            n_channels: vec![64, 64, 64, 64],
            kernel_size: 3,
            dropout: 0.1,
        }
    }
}

impl TCNConfig {
    /// Create a new TCN config.
    pub fn new(n_vars: usize, seq_len: usize, n_classes: usize) -> Self {
        Self {
            n_vars,
            seq_len,
            n_classes,
            ..Default::default()
        }
    }

    /// Set the number of channels per layer.
    #[must_use]
    pub fn with_channels(mut self, channels: Vec<usize>) -> Self {
        self.n_channels = channels;
        self
    }

    /// Set the kernel size.
    #[must_use]
    pub fn with_kernel_size(mut self, kernel_size: usize) -> Self {
        self.kernel_size = kernel_size;
        self
    }

    /// Set the dropout rate.
    #[must_use]
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Calculate the receptive field.
    pub fn receptive_field(&self) -> usize {
        let n_layers = self.n_channels.len();
        // Each layer doubles the dilation
        // Receptive field = 1 + 2 * (kernel_size - 1) * sum(2^i for i in 0..n_layers)
        let dilation_sum: usize = (0..n_layers).map(|i| 1 << i).sum();
        1 + 2 * (self.kernel_size - 1) * dilation_sum
    }

    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> TCN<B> {
        TCN::new(self.clone(), device)
    }
}

/// Temporal Convolutional Network for time series classification.
///
/// Uses stacked residual blocks with exponentially increasing dilations
/// to capture patterns at multiple time scales.
///
/// # Architecture
///
/// ```text
/// Input (B, V, L) -> [TCN Block d=1] -> [TCN Block d=2] -> ... -> [TCN Block d=2^n]
///                 -> Global Average Pool -> Linear -> Output
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use tsai_models::cnn::{TCN, TCNConfig};
///
/// let config = TCNConfig::new(3, 100, 5)
///     .with_channels(vec![64, 64, 128, 128])
///     .with_kernel_size(3);
/// let model = config.init::<NdArray>(&device);
///
/// let x = Tensor::random([32, 3, 100], Distribution::Normal(0.0, 1.0), &device);
/// let output = model.forward(x);
/// // output shape: [32, 5]
/// ```
#[derive(Module, Debug)]
pub struct TCN<B: Backend> {
    /// TCN blocks with increasing dilation.
    blocks: Vec<TCNBlock<B>>,
    /// Final classifier.
    classifier: Linear<B>,
}

impl<B: Backend> TCN<B> {
    /// Create a new TCN model.
    pub fn new(config: TCNConfig, device: &B::Device) -> Self {
        let mut blocks = Vec::new();
        let n_layers = config.n_channels.len();

        for i in 0..n_layers {
            let in_channels = if i == 0 {
                config.n_vars
            } else {
                config.n_channels[i - 1]
            };
            let out_channels = config.n_channels[i];
            let dilation = 1 << i; // Exponential dilation: 1, 2, 4, 8, ...

            let block_config = TCNBlockConfig::new(in_channels, out_channels, config.kernel_size, dilation)
                .with_dropout(config.dropout);
            blocks.push(block_config.init(device));
        }

        let final_channels = *config.n_channels.last().unwrap_or(&64);
        let classifier = LinearConfig::new(final_channels, config.n_classes).init(device);

        Self { blocks, classifier }
    }

    /// Forward pass.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let mut out = x;

        // Apply TCN blocks
        for block in &self.blocks {
            out = block.forward(out);
        }

        // Global average pooling over time dimension
        // Shape: (B, C, L) -> (B, C)
        let [batch, channels, _] = out.dims();
        let out: Tensor<B, 2> = out.mean_dim(2).reshape([batch, channels]);

        // Classifier
        self.classifier.forward(out)
    }

    /// Forward pass returning probabilities.
    pub fn forward_probs(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let logits = self.forward(x);
        softmax(logits, 1)
    }

    /// Get the number of layers.
    pub fn num_layers(&self) -> usize {
        self.blocks.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tcn_config_default() {
        let config = TCNConfig::default();
        assert_eq!(config.n_vars, 1);
        assert_eq!(config.kernel_size, 3);
        assert_eq!(config.n_channels.len(), 4);
    }

    #[test]
    fn test_tcn_config_new() {
        let config = TCNConfig::new(3, 200, 10);
        assert_eq!(config.n_vars, 3);
        assert_eq!(config.seq_len, 200);
        assert_eq!(config.n_classes, 10);
    }

    #[test]
    fn test_tcn_config_builder() {
        let config = TCNConfig::new(3, 100, 5)
            .with_channels(vec![32, 64, 128])
            .with_kernel_size(5)
            .with_dropout(0.2);

        assert_eq!(config.n_channels, vec![32, 64, 128]);
        assert_eq!(config.kernel_size, 5);
        assert_eq!(config.dropout, 0.2);
    }

    #[test]
    fn test_receptive_field() {
        // 4 layers with kernel_size=3
        // Dilations: 1, 2, 4, 8
        // RF = 1 + 2 * (3-1) * (1+2+4+8) = 1 + 4 * 15 = 61
        let config = TCNConfig::default();
        assert_eq!(config.receptive_field(), 61);

        // 3 layers with kernel_size=5
        // Dilations: 1, 2, 4
        // RF = 1 + 2 * (5-1) * (1+2+4) = 1 + 8 * 7 = 57
        let config = TCNConfig::new(1, 100, 2)
            .with_channels(vec![64, 64, 64])
            .with_kernel_size(5);
        assert_eq!(config.receptive_field(), 57);
    }

    #[test]
    fn test_tcn_block_config() {
        let config = TCNBlockConfig::new(32, 64, 3, 4)
            .with_dropout(0.2);
        assert_eq!(config.in_channels, 32);
        assert_eq!(config.out_channels, 64);
        assert_eq!(config.kernel_size, 3);
        assert_eq!(config.dilation, 4);
        assert_eq!(config.dropout, 0.2);
    }
}
