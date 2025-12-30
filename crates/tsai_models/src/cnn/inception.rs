//! InceptionTime model architecture.
//!
//! Based on the paper "InceptionTime: Finding AlexNet for Time Series Classification"
//! by Fawaz et al. (2020).

use burn::nn::{
    conv::{Conv1d, Conv1dConfig},
    pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig, MaxPool1d, MaxPool1dConfig},
    BatchNorm, BatchNormConfig, Linear, LinearConfig, Relu,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use serde::{Deserialize, Serialize};

/// Configuration for InceptionTimePlus model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InceptionTimePlusConfig {
    /// Number of input variables/channels.
    pub n_vars: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// Number of inception blocks.
    pub n_blocks: usize,
    /// Number of filters in each branch.
    pub n_filters: usize,
    /// Kernel sizes for the three conv branches.
    pub kernel_sizes: [usize; 3],
    /// Bottleneck dimension (if > 0, adds bottleneck layer).
    pub bottleneck_dim: usize,
    /// Dropout rate.
    pub dropout: f64,
}

impl Default for InceptionTimePlusConfig {
    fn default() -> Self {
        Self {
            n_vars: 1,
            seq_len: 100,
            n_classes: 2,
            n_blocks: 6,
            n_filters: 32,
            kernel_sizes: [9, 19, 39],
            bottleneck_dim: 32,
            dropout: 0.0,
        }
    }
}

impl InceptionTimePlusConfig {
    /// Create a new config with specified dimensions.
    pub fn new(n_vars: usize, seq_len: usize, n_classes: usize) -> Self {
        Self {
            n_vars,
            seq_len,
            n_classes,
            ..Default::default()
        }
    }

    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> InceptionTimePlus<B> {
        InceptionTimePlus::new(self.clone(), device)
    }
}

/// Inception block with multiple parallel convolution branches.
#[derive(Module, Debug)]
pub struct InceptionBlock<B: Backend> {
    /// Bottleneck convolution (optional).
    bottleneck: Option<Conv1d<B>>,
    /// First branch convolution.
    conv1: Conv1d<B>,
    /// Second branch convolution.
    conv2: Conv1d<B>,
    /// Third branch convolution.
    conv3: Conv1d<B>,
    /// Max pooling branch.
    maxpool: MaxPool1d,
    /// Conv for maxpool branch.
    conv_maxpool: Conv1d<B>,
    /// Batch normalization.
    bn: BatchNorm<B, 1>,
}

impl<B: Backend> InceptionBlock<B> {
    /// Create a new inception block.
    pub fn new(
        in_channels: usize,
        n_filters: usize,
        kernel_sizes: [usize; 3],
        bottleneck_dim: usize,
        device: &B::Device,
    ) -> Self {
        let (conv_in, bottleneck) = if bottleneck_dim > 0 {
            let bn_conv = Conv1dConfig::new(in_channels, bottleneck_dim, 1)
                .with_bias(false)
                .init(device);
            (bottleneck_dim, Some(bn_conv))
        } else {
            (in_channels, None)
        };

        // Three parallel conv branches with different kernel sizes
        let conv1 = Conv1dConfig::new(conv_in, n_filters, kernel_sizes[0])
            .with_padding(burn::nn::PaddingConfig1d::Same)
            .with_bias(false)
            .init(device);

        let conv2 = Conv1dConfig::new(conv_in, n_filters, kernel_sizes[1])
            .with_padding(burn::nn::PaddingConfig1d::Same)
            .with_bias(false)
            .init(device);

        let conv3 = Conv1dConfig::new(conv_in, n_filters, kernel_sizes[2])
            .with_padding(burn::nn::PaddingConfig1d::Same)
            .with_bias(false)
            .init(device);

        // Max pooling branch
        let maxpool = MaxPool1dConfig::new(3)
            .with_padding(burn::nn::PaddingConfig1d::Same)
            .with_stride(1)
            .init();

        let conv_maxpool = Conv1dConfig::new(in_channels, n_filters, 1)
            .with_bias(false)
            .init(device);

        // Total output channels = n_filters * 4 (3 conv branches + 1 maxpool)
        let out_channels = n_filters * 4;
        let bn = BatchNormConfig::new(out_channels).init(device);

        Self {
            bottleneck,
            conv1,
            conv2,
            conv3,
            maxpool,
            conv_maxpool,
            bn,
        }
    }

    /// Forward pass.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Apply bottleneck if present
        let x_bn = if let Some(ref bottleneck) = self.bottleneck {
            bottleneck.forward(x.clone())
        } else {
            x.clone()
        };

        // Parallel branches
        let out1 = self.conv1.forward(x_bn.clone());
        let out2 = self.conv2.forward(x_bn.clone());
        let out3 = self.conv3.forward(x_bn);

        // Max pooling branch (uses original input)
        let out_pool = self.maxpool.forward(x);
        let out_pool = self.conv_maxpool.forward(out_pool);

        // Concatenate along channel dimension
        let out = Tensor::cat(vec![out1, out2, out3, out_pool], 1);

        // Batch norm + ReLU
        let out = self.bn.forward(out);
        Relu::new().forward(out)
    }
}

/// InceptionTimePlus model for time series classification.
#[derive(Module, Debug)]
pub struct InceptionTimePlus<B: Backend> {
    /// Inception blocks.
    blocks: Vec<InceptionBlock<B>>,
    /// Residual convolutions for shortcut connections.
    residual_convs: Vec<Conv1d<B>>,
    /// Residual batch norms.
    residual_bns: Vec<BatchNorm<B, 1>>,
    /// Global average pooling.
    gap: AdaptiveAvgPool1d,
    /// Final linear classifier.
    fc: Linear<B>,
}

impl<B: Backend> InceptionTimePlus<B> {
    /// Create a new InceptionTimePlus model.
    pub fn new(config: InceptionTimePlusConfig, device: &B::Device) -> Self {
        let mut blocks = Vec::new();
        let mut residual_convs = Vec::new();
        let mut residual_bns = Vec::new();

        let n_filters = config.n_filters;
        let out_channels = n_filters * 4; // Each block outputs 4 * n_filters

        for i in 0..config.n_blocks {
            let in_channels = if i == 0 {
                config.n_vars
            } else {
                out_channels
            };

            let block = InceptionBlock::new(
                in_channels,
                n_filters,
                config.kernel_sizes,
                config.bottleneck_dim,
                device,
            );
            blocks.push(block);

            // Residual connection every 3 blocks
            if (i + 1) % 3 == 0 {
                let res_in = if i < 3 { config.n_vars } else { out_channels };
                let res_conv = Conv1dConfig::new(res_in, out_channels, 1)
                    .with_bias(false)
                    .init(device);
                let res_bn = BatchNormConfig::new(out_channels).init(device);
                residual_convs.push(res_conv);
                residual_bns.push(res_bn);
            }
        }

        let gap = AdaptiveAvgPool1dConfig::new(1).init();
        let fc = LinearConfig::new(out_channels, config.n_classes).init(device);

        Self {
            blocks,
            residual_convs,
            residual_bns,
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
    /// Output tensor of shape (batch, n_classes)
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let mut out = x.clone();
        let mut residual = x;
        let mut res_idx = 0;

        for (i, block) in self.blocks.iter().enumerate() {
            out = block.forward(out);

            // Apply residual connection every 3 blocks
            if (i + 1) % 3 == 0 && res_idx < self.residual_convs.len() {
                let res = self.residual_convs[res_idx].forward(residual.clone());
                let res = self.residual_bns[res_idx].forward(res);
                out = out + res;
                out = Relu::new().forward(out);
                residual = out.clone();
                res_idx += 1;
            }
        }

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
    fn test_config_default() {
        let config = InceptionTimePlusConfig::default();
        assert_eq!(config.n_blocks, 6);
        assert_eq!(config.n_filters, 32);
    }

    // Backend-specific tests would go here
}
