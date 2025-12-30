//! XceptionTime model architecture.
//!
//! Based on the Xception architecture adapted for time series classification.
//! Uses depthwise separable convolutions for efficient feature extraction.
//!
//! Reference: "Xception: Deep Learning with Depthwise Separable Convolutions"
//! by Chollet (2017), adapted for 1D time series.

use burn::nn::{
    conv::{Conv1d, Conv1dConfig},
    pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig, MaxPool1d, MaxPool1dConfig},
    BatchNorm, BatchNormConfig, Linear, LinearConfig, Relu,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use serde::{Deserialize, Serialize};

/// Configuration for XceptionTime model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XceptionTimeConfig {
    /// Number of input variables/channels.
    pub n_vars: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// Number of filters in each block.
    pub n_filters: usize,
    /// Kernel size for separable convolutions.
    pub kernel_size: usize,
    /// Number of Xception blocks.
    pub n_blocks: usize,
    /// Dropout rate.
    pub dropout: f64,
}

impl Default for XceptionTimeConfig {
    fn default() -> Self {
        Self {
            n_vars: 1,
            seq_len: 100,
            n_classes: 2,
            n_filters: 128,
            kernel_size: 39,
            n_blocks: 4,
            dropout: 0.0,
        }
    }
}

impl XceptionTimeConfig {
    /// Create a new config with specified dimensions.
    pub fn new(n_vars: usize, seq_len: usize, n_classes: usize) -> Self {
        Self {
            n_vars,
            seq_len,
            n_classes,
            ..Default::default()
        }
    }

    /// Set the number of filters.
    #[must_use]
    pub fn with_filters(mut self, n_filters: usize) -> Self {
        self.n_filters = n_filters;
        self
    }

    /// Set the kernel size.
    #[must_use]
    pub fn with_kernel_size(mut self, kernel_size: usize) -> Self {
        self.kernel_size = kernel_size;
        self
    }

    /// Set the number of blocks.
    #[must_use]
    pub fn with_n_blocks(mut self, n_blocks: usize) -> Self {
        self.n_blocks = n_blocks;
        self
    }

    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> XceptionTime<B> {
        XceptionTime::new(self.clone(), device)
    }
}

/// Depthwise separable convolution block.
///
/// Consists of:
/// 1. Depthwise convolution (groups = in_channels)
/// 2. Pointwise convolution (1x1 conv)
/// 3. Batch normalization
/// 4. ReLU activation
#[derive(Module, Debug)]
pub struct SeparableConv1d<B: Backend> {
    /// Depthwise convolution.
    depthwise: Conv1d<B>,
    /// Pointwise convolution.
    pointwise: Conv1d<B>,
    /// Batch normalization.
    bn: BatchNorm<B, 1>,
}

impl<B: Backend> SeparableConv1d<B> {
    /// Create a new separable convolution.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        device: &B::Device,
    ) -> Self {
        // Depthwise: each input channel is convolved separately
        let depthwise = Conv1dConfig::new(in_channels, in_channels, kernel_size)
            .with_padding(burn::nn::PaddingConfig1d::Same)
            .with_groups(in_channels)
            .with_bias(false)
            .init(device);

        // Pointwise: 1x1 convolution to mix channels
        let pointwise = Conv1dConfig::new(in_channels, out_channels, 1)
            .with_bias(false)
            .init(device);

        let bn = BatchNormConfig::new(out_channels).init(device);

        Self {
            depthwise,
            pointwise,
            bn,
        }
    }

    /// Forward pass.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let out = self.depthwise.forward(x);
        let out = self.pointwise.forward(out);
        let out = self.bn.forward(out);
        Relu::new().forward(out)
    }
}

/// Xception block with residual connection.
#[derive(Module, Debug)]
pub struct XceptionBlock<B: Backend> {
    /// First separable convolution.
    sep_conv1: SeparableConv1d<B>,
    /// Second separable convolution.
    sep_conv2: SeparableConv1d<B>,
    /// Residual convolution (for channel matching).
    residual_conv: Conv1d<B>,
    /// Residual batch norm.
    residual_bn: BatchNorm<B, 1>,
    /// Max pooling.
    maxpool: MaxPool1d,
}

impl<B: Backend> XceptionBlock<B> {
    /// Create a new Xception block.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        device: &B::Device,
    ) -> Self {
        let sep_conv1 = SeparableConv1d::new(in_channels, out_channels, kernel_size, device);
        let sep_conv2 = SeparableConv1d::new(out_channels, out_channels, kernel_size, device);

        let residual_conv = Conv1dConfig::new(in_channels, out_channels, 1)
            .with_bias(false)
            .init(device);
        let residual_bn = BatchNormConfig::new(out_channels).init(device);

        let maxpool = MaxPool1dConfig::new(3)
            .with_padding(burn::nn::PaddingConfig1d::Same)
            .with_stride(1)
            .init();

        Self {
            sep_conv1,
            sep_conv2,
            residual_conv,
            residual_bn,
            maxpool,
        }
    }

    /// Forward pass with residual connection.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Main path
        let out = self.sep_conv1.forward(x.clone());
        let out = self.sep_conv2.forward(out);
        let out = self.maxpool.forward(out);

        // Residual path
        let residual = self.residual_conv.forward(x);
        let residual = self.residual_bn.forward(residual);

        // Add residual
        Relu::new().forward(out + residual)
    }
}

/// XceptionTime model for time series classification.
///
/// Architecture:
/// - Entry flow: Initial convolution
/// - Middle flow: Multiple Xception blocks with separable convolutions
/// - Exit flow: Global average pooling + classifier
///
/// # Example
///
/// ```rust,ignore
/// use tsai_models::cnn::XceptionTime;
///
/// let config = XceptionTimeConfig::new(3, 100, 5)
///     .with_filters(128)
///     .with_n_blocks(4);
/// let model = config.init::<NdArray>(&device);
///
/// let x = Tensor::random([32, 3, 100], Distribution::Normal(0.0, 1.0), &device);
/// let output = model.forward(x);
/// // output shape: [32, 5]
/// ```
#[derive(Module, Debug)]
pub struct XceptionTime<B: Backend> {
    /// Entry convolution.
    entry_conv: Conv1d<B>,
    /// Entry batch norm.
    entry_bn: BatchNorm<B, 1>,
    /// Xception blocks.
    blocks: Vec<XceptionBlock<B>>,
    /// Global average pooling.
    gap: AdaptiveAvgPool1d,
    /// Final classifier.
    fc: Linear<B>,
}

impl<B: Backend> XceptionTime<B> {
    /// Create a new XceptionTime model.
    pub fn new(config: XceptionTimeConfig, device: &B::Device) -> Self {
        // Entry flow
        let entry_conv = Conv1dConfig::new(config.n_vars, config.n_filters, config.kernel_size)
            .with_padding(burn::nn::PaddingConfig1d::Same)
            .with_bias(false)
            .init(device);
        let entry_bn = BatchNormConfig::new(config.n_filters).init(device);

        // Middle flow: Xception blocks
        let mut blocks = Vec::new();
        for i in 0..config.n_blocks {
            let in_channels = if i == 0 {
                config.n_filters
            } else {
                config.n_filters * 2_usize.pow(i as u32 - 1).min(4)
            };
            let out_channels = config.n_filters * 2_usize.pow(i as u32).min(4);

            blocks.push(XceptionBlock::new(
                in_channels,
                out_channels,
                config.kernel_size,
                device,
            ));
        }

        let final_channels = config.n_filters * 2_usize.pow((config.n_blocks - 1) as u32).min(4);

        // Exit flow
        let gap = AdaptiveAvgPool1dConfig::new(1).init();
        let fc = LinearConfig::new(final_channels, config.n_classes).init(device);

        Self {
            entry_conv,
            entry_bn,
            blocks,
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
        // Entry flow
        let mut out = self.entry_conv.forward(x);
        out = self.entry_bn.forward(out);
        out = Relu::new().forward(out);

        // Middle flow
        for block in &self.blocks {
            out = block.forward(out);
        }

        // Exit flow
        let out = self.gap.forward(out);
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
    fn test_xception_config_default() {
        let config = XceptionTimeConfig::default();
        assert_eq!(config.n_vars, 1);
        assert_eq!(config.n_filters, 128);
        assert_eq!(config.kernel_size, 39);
        assert_eq!(config.n_blocks, 4);
    }

    #[test]
    fn test_xception_config_builder() {
        let config = XceptionTimeConfig::new(3, 200, 10)
            .with_filters(64)
            .with_kernel_size(15)
            .with_n_blocks(3);

        assert_eq!(config.n_vars, 3);
        assert_eq!(config.seq_len, 200);
        assert_eq!(config.n_classes, 10);
        assert_eq!(config.n_filters, 64);
        assert_eq!(config.kernel_size, 15);
        assert_eq!(config.n_blocks, 3);
    }
}
