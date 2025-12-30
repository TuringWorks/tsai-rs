//! Fully Convolutional Network (FCN) for time series classification.
//!
//! Based on the paper "Time Series Classification from Scratch with
//! Deep Neural Networks: A Strong Baseline" by Wang et al. (2017).
//!
//! The FCN architecture consists of three convolutional blocks followed
//! by global average pooling and a softmax classifier.

use burn::nn::{
    conv::{Conv1d, Conv1dConfig},
    pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig},
    BatchNorm, BatchNormConfig, Linear, LinearConfig, Relu,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use serde::{Deserialize, Serialize};

/// Configuration for the FCN model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FCNConfig {
    /// Number of input variables/channels.
    pub n_vars: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// Number of filters in the first conv layer.
    pub n_filters_1: usize,
    /// Number of filters in the second conv layer.
    pub n_filters_2: usize,
    /// Number of filters in the third conv layer.
    pub n_filters_3: usize,
    /// Kernel size for the first conv layer.
    pub kernel_size_1: usize,
    /// Kernel size for the second conv layer.
    pub kernel_size_2: usize,
    /// Kernel size for the third conv layer.
    pub kernel_size_3: usize,
}

impl Default for FCNConfig {
    fn default() -> Self {
        Self {
            n_vars: 1,
            seq_len: 100,
            n_classes: 2,
            n_filters_1: 128,
            n_filters_2: 256,
            n_filters_3: 128,
            kernel_size_1: 8,
            kernel_size_2: 5,
            kernel_size_3: 3,
        }
    }
}

impl FCNConfig {
    /// Create a new config with specified dimensions.
    pub fn new(n_vars: usize, seq_len: usize, n_classes: usize) -> Self {
        Self {
            n_vars,
            seq_len,
            n_classes,
            ..Default::default()
        }
    }

    /// Set the number of filters for all layers.
    #[must_use]
    pub fn with_filters(mut self, n_filters_1: usize, n_filters_2: usize, n_filters_3: usize) -> Self {
        self.n_filters_1 = n_filters_1;
        self.n_filters_2 = n_filters_2;
        self.n_filters_3 = n_filters_3;
        self
    }

    /// Set the kernel sizes for all layers.
    #[must_use]
    pub fn with_kernel_sizes(mut self, k1: usize, k2: usize, k3: usize) -> Self {
        self.kernel_size_1 = k1;
        self.kernel_size_2 = k2;
        self.kernel_size_3 = k3;
        self
    }

    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> FCN<B> {
        FCN::new(self.clone(), device)
    }
}

/// A single convolutional block: Conv1d -> BatchNorm -> ReLU
#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    /// Convolutional layer.
    conv: Conv1d<B>,
    /// Batch normalization.
    bn: BatchNorm<B, 1>,
}

impl<B: Backend> ConvBlock<B> {
    /// Create a new convolutional block.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        device: &B::Device,
    ) -> Self {
        let conv = Conv1dConfig::new(in_channels, out_channels, kernel_size)
            .with_padding(burn::nn::PaddingConfig1d::Same)
            .with_bias(false)
            .init(device);

        let bn = BatchNormConfig::new(out_channels).init(device);

        Self { conv, bn }
    }

    /// Forward pass through the block.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let out = self.conv.forward(x);
        let out = self.bn.forward(out);
        Relu::new().forward(out)
    }
}

/// Fully Convolutional Network for time series classification.
///
/// Architecture:
/// - Conv1d(n_vars, 128, kernel=8) -> BatchNorm -> ReLU
/// - Conv1d(128, 256, kernel=5) -> BatchNorm -> ReLU
/// - Conv1d(256, 128, kernel=3) -> BatchNorm -> ReLU
/// - Global Average Pooling
/// - Linear(128, n_classes)
///
/// # Example
///
/// ```rust,ignore
/// use tsai_models::cnn::FCN;
///
/// let config = FCNConfig::new(3, 100, 5);
/// let model = config.init::<NdArray>(&device);
///
/// let x = Tensor::random([32, 3, 100], Distribution::Normal(0.0, 1.0), &device);
/// let output = model.forward(x);
/// // output shape: [32, 5]
/// ```
#[derive(Module, Debug)]
pub struct FCN<B: Backend> {
    /// First convolutional block.
    block1: ConvBlock<B>,
    /// Second convolutional block.
    block2: ConvBlock<B>,
    /// Third convolutional block.
    block3: ConvBlock<B>,
    /// Global average pooling.
    gap: AdaptiveAvgPool1d,
    /// Final linear classifier.
    fc: Linear<B>,
}

impl<B: Backend> FCN<B> {
    /// Create a new FCN model.
    pub fn new(config: FCNConfig, device: &B::Device) -> Self {
        let block1 = ConvBlock::new(
            config.n_vars,
            config.n_filters_1,
            config.kernel_size_1,
            device,
        );

        let block2 = ConvBlock::new(
            config.n_filters_1,
            config.n_filters_2,
            config.kernel_size_2,
            device,
        );

        let block3 = ConvBlock::new(
            config.n_filters_2,
            config.n_filters_3,
            config.kernel_size_3,
            device,
        );

        let gap = AdaptiveAvgPool1dConfig::new(1).init();
        let fc = LinearConfig::new(config.n_filters_3, config.n_classes).init(device);

        Self {
            block1,
            block2,
            block3,
            gap,
            fc,
        }
    }

    /// Forward pass.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape (batch, vars, seq_len)
    ///
    /// # Returns
    ///
    /// Output tensor of shape (batch, n_classes) with logits
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        // Three conv blocks
        let out = self.block1.forward(x);
        let out = self.block2.forward(out);
        let out = self.block3.forward(out);

        // Global average pooling
        let out = self.gap.forward(out);

        // Flatten and classify
        let [batch, channels, _] = out.dims();
        let out = out.reshape([batch, channels]);
        self.fc.forward(out)
    }

    /// Forward pass returning probabilities.
    pub fn forward_probs(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let logits = self.forward(x);
        softmax(logits, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fcn_config_default() {
        let config = FCNConfig::default();
        assert_eq!(config.n_vars, 1);
        assert_eq!(config.n_filters_1, 128);
        assert_eq!(config.n_filters_2, 256);
        assert_eq!(config.n_filters_3, 128);
        assert_eq!(config.kernel_size_1, 8);
        assert_eq!(config.kernel_size_2, 5);
        assert_eq!(config.kernel_size_3, 3);
    }

    #[test]
    fn test_fcn_config_new() {
        let config = FCNConfig::new(3, 200, 10);
        assert_eq!(config.n_vars, 3);
        assert_eq!(config.seq_len, 200);
        assert_eq!(config.n_classes, 10);
    }

    #[test]
    fn test_fcn_config_builder() {
        let config = FCNConfig::new(3, 100, 5)
            .with_filters(64, 128, 64)
            .with_kernel_sizes(7, 5, 3);

        assert_eq!(config.n_filters_1, 64);
        assert_eq!(config.n_filters_2, 128);
        assert_eq!(config.n_filters_3, 64);
        assert_eq!(config.kernel_size_1, 7);
    }
}
