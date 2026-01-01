//! ResCNN model architecture for time series.
//!
//! ResCNN (Residual CNN) is a simpler variant of ResNet for 1D time series,
//! using basic residual blocks with two convolutions instead of three.

use burn::nn::{
    conv::{Conv1d, Conv1dConfig},
    pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig},
    BatchNorm, BatchNormConfig, Linear, LinearConfig, Relu,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use serde::{Deserialize, Serialize};

/// Configuration for ResCNN model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResCNNConfig {
    /// Number of input variables/channels.
    pub n_vars: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// Number of filters in each block.
    pub n_filters: Vec<usize>,
    /// Kernel sizes for each block.
    pub kernel_sizes: Vec<usize>,
}

impl Default for ResCNNConfig {
    fn default() -> Self {
        Self {
            n_vars: 1,
            seq_len: 100,
            n_classes: 2,
            n_filters: vec![64, 64, 64],
            kernel_sizes: vec![8, 5, 3],
        }
    }
}

impl ResCNNConfig {
    /// Create a new config.
    pub fn new(n_vars: usize, seq_len: usize, n_classes: usize) -> Self {
        Self {
            n_vars,
            seq_len,
            n_classes,
            ..Default::default()
        }
    }

    /// Set the number of filters.
    pub fn with_n_filters(mut self, n_filters: Vec<usize>) -> Self {
        self.n_filters = n_filters;
        self
    }

    /// Set the kernel sizes.
    pub fn with_kernel_sizes(mut self, kernel_sizes: Vec<usize>) -> Self {
        self.kernel_sizes = kernel_sizes;
        self
    }

    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResCNN<B> {
        ResCNN::new(self.clone(), device)
    }
}

/// Basic residual block with two convolutions and skip connection.
#[derive(Module, Debug)]
pub struct ResCNNBlock<B: Backend> {
    conv1: Conv1d<B>,
    bn1: BatchNorm<B, 1>,
    conv2: Conv1d<B>,
    bn2: BatchNorm<B, 1>,
    shortcut: Option<Conv1d<B>>,
    shortcut_bn: Option<BatchNorm<B, 1>>,
}

impl<B: Backend> ResCNNBlock<B> {
    /// Create a new ResCNN block.
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
            shortcut,
            shortcut_bn,
        }
    }

    /// Forward pass.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let relu = Relu::new();

        // First conv + bn + relu
        let out = self.conv1.forward(x.clone());
        let out = self.bn1.forward(out);
        let out = relu.forward(out);

        // Second conv + bn (no relu before addition)
        let out = self.conv2.forward(out);
        let out = self.bn2.forward(out);

        // Shortcut
        let shortcut = if let (Some(ref sc), Some(ref sc_bn)) = (&self.shortcut, &self.shortcut_bn)
        {
            let s = sc.forward(x);
            sc_bn.forward(s)
        } else {
            x
        };

        // Residual addition + relu
        let out = out + shortcut;
        relu.forward(out)
    }
}

/// ResCNN model for time series classification.
///
/// A simpler variant of ResNet using basic residual blocks with
/// two convolutions instead of three, suitable for 1D time series.
#[derive(Module, Debug)]
pub struct ResCNN<B: Backend> {
    blocks: Vec<ResCNNBlock<B>>,
    gap: AdaptiveAvgPool1d,
    fc: Linear<B>,
}

impl<B: Backend> ResCNN<B> {
    /// Create a new ResCNN model.
    pub fn new(config: ResCNNConfig, device: &B::Device) -> Self {
        let mut blocks = Vec::new();

        let mut in_channels = config.n_vars;
        for (i, &out_channels) in config.n_filters.iter().enumerate() {
            let kernel_size = config
                .kernel_sizes
                .get(i)
                .copied()
                .unwrap_or(*config.kernel_sizes.last().unwrap_or(&3));
            let block = ResCNNBlock::new(in_channels, out_channels, kernel_size, device);
            blocks.push(block);
            in_channels = out_channels;
        }

        let gap = AdaptiveAvgPool1dConfig::new(1).init();
        let final_channels = *config.n_filters.last().unwrap_or(&64);
        let fc = LinearConfig::new(final_channels, config.n_classes).init(device);

        Self { blocks, gap, fc }
    }

    /// Forward pass returning logits.
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
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_rescnn_config() {
        let config = ResCNNConfig::default();
        assert_eq!(config.n_filters.len(), 3);
        assert_eq!(config.kernel_sizes, vec![8, 5, 3]);
    }

    #[test]
    fn test_rescnn_forward() {
        let device = Default::default();
        let config = ResCNNConfig::new(3, 100, 5);
        let model: ResCNN<TestBackend> = config.init(&device);

        let x = Tensor::<TestBackend, 3>::zeros([4, 3, 100], &device);
        let out = model.forward(x);
        assert_eq!(out.dims(), [4, 5]);
    }

    #[test]
    fn test_rescnn_block() {
        let device = Default::default();
        let block: ResCNNBlock<TestBackend> = ResCNNBlock::new(3, 64, 5, &device);

        let x = Tensor::<TestBackend, 3>::zeros([4, 3, 100], &device);
        let out = block.forward(x);
        assert_eq!(out.dims(), [4, 64, 100]);
    }
}
