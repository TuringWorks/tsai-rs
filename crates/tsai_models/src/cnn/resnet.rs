//! ResNet model architecture for time series.

use burn::nn::{
    conv::{Conv1d, Conv1dConfig},
    pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig},
    BatchNorm, BatchNormConfig, Linear, LinearConfig, Relu,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use serde::{Deserialize, Serialize};

/// Configuration for ResNetPlus model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResNetPlusConfig {
    /// Number of input variables/channels.
    pub n_vars: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// Number of residual blocks.
    pub n_blocks: usize,
    /// Number of filters in each layer.
    pub n_filters: Vec<usize>,
    /// Kernel size.
    pub kernel_size: usize,
}

impl Default for ResNetPlusConfig {
    fn default() -> Self {
        Self {
            n_vars: 1,
            seq_len: 100,
            n_classes: 2,
            n_blocks: 3,
            n_filters: vec![64, 128, 128],
            kernel_size: 8,
        }
    }
}

impl ResNetPlusConfig {
    /// Create a new config.
    pub fn new(n_vars: usize, seq_len: usize, n_classes: usize) -> Self {
        Self {
            n_vars,
            seq_len,
            n_classes,
            ..Default::default()
        }
    }

    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResNetPlus<B> {
        ResNetPlus::new(self.clone(), device)
    }
}

/// Residual block with two convolutions and skip connection.
#[derive(Module, Debug)]
pub struct ResNetBlock<B: Backend> {
    conv1: Conv1d<B>,
    bn1: BatchNorm<B, 1>,
    conv2: Conv1d<B>,
    bn2: BatchNorm<B, 1>,
    conv3: Conv1d<B>,
    bn3: BatchNorm<B, 1>,
    shortcut: Option<Conv1d<B>>,
    shortcut_bn: Option<BatchNorm<B, 1>>,
}

impl<B: Backend> ResNetBlock<B> {
    /// Create a new residual block.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        device: &B::Device,
    ) -> Self {
        let conv1 = Conv1dConfig::new(in_channels, out_channels, kernel_size)
            .with_padding(burn::nn::PaddingConfig1d::Same)
            .with_bias(false)
            .init(device);
        let bn1 = BatchNormConfig::new(out_channels).init(device);

        let conv2 = Conv1dConfig::new(out_channels, out_channels, kernel_size)
            .with_padding(burn::nn::PaddingConfig1d::Same)
            .with_bias(false)
            .init(device);
        let bn2 = BatchNormConfig::new(out_channels).init(device);

        let conv3 = Conv1dConfig::new(out_channels, out_channels, kernel_size)
            .with_padding(burn::nn::PaddingConfig1d::Same)
            .with_bias(false)
            .init(device);
        let bn3 = BatchNormConfig::new(out_channels).init(device);

        // Shortcut connection if dimensions differ
        let (shortcut, shortcut_bn) = if in_channels != out_channels {
            let sc = Conv1dConfig::new(in_channels, out_channels, 1)
                .with_bias(false)
                .init(device);
            let sc_bn = BatchNormConfig::new(out_channels).init(device);
            (Some(sc), Some(sc_bn))
        } else {
            (None, None)
        };

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            shortcut,
            shortcut_bn,
        }
    }

    /// Forward pass.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let relu = Relu::new();

        let out = self.conv1.forward(x.clone());
        let out = self.bn1.forward(out);
        let out = relu.forward(out);

        let out = self.conv2.forward(out);
        let out = self.bn2.forward(out);
        let out = relu.forward(out);

        let out = self.conv3.forward(out);
        let out = self.bn3.forward(out);

        // Shortcut
        let shortcut = if let (Some(ref sc), Some(ref sc_bn)) = (&self.shortcut, &self.shortcut_bn)
        {
            let s = sc.forward(x);
            sc_bn.forward(s)
        } else {
            x
        };

        let out = out + shortcut;
        relu.forward(out)
    }
}

/// ResNetPlus model for time series classification.
#[derive(Module, Debug)]
pub struct ResNetPlus<B: Backend> {
    blocks: Vec<ResNetBlock<B>>,
    gap: AdaptiveAvgPool1d,
    fc: Linear<B>,
}

impl<B: Backend> ResNetPlus<B> {
    /// Create a new ResNetPlus model.
    pub fn new(config: ResNetPlusConfig, device: &B::Device) -> Self {
        let mut blocks = Vec::new();

        let mut in_channels = config.n_vars;
        for &out_channels in &config.n_filters {
            let block = ResNetBlock::new(in_channels, out_channels, config.kernel_size, device);
            blocks.push(block);
            in_channels = out_channels;
        }

        let gap = AdaptiveAvgPool1dConfig::new(1).init();
        let final_channels = *config.n_filters.last().unwrap_or(&64);
        let fc = LinearConfig::new(final_channels, config.n_classes).init(device);

        Self { blocks, gap, fc }
    }

    /// Forward pass.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let mut out = x;

        for block in &self.blocks {
            out = block.forward(out);
        }

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
    fn test_resnet_config() {
        let config = ResNetPlusConfig::default();
        assert_eq!(config.n_blocks, 3);
        assert_eq!(config.kernel_size, 8);
    }
}
