//! XCM (eXplainable Convolutional neural network for Multivariate time series) model.

use burn::nn::{
    conv::{Conv1d, Conv1dConfig},
    pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig},
    BatchNorm, BatchNormConfig, Linear, LinearConfig, Relu,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use serde::{Deserialize, Serialize};

/// Configuration for XCMPlus model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XCMPlusConfig {
    /// Number of input variables/channels.
    pub n_vars: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// Number of filters.
    pub n_filters: usize,
    /// Window sizes for multi-scale convolutions.
    pub window_sizes: Vec<usize>,
}

impl Default for XCMPlusConfig {
    fn default() -> Self {
        Self {
            n_vars: 1,
            seq_len: 100,
            n_classes: 2,
            n_filters: 128,
            window_sizes: vec![10, 20, 40],
        }
    }
}

impl XCMPlusConfig {
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
    pub fn init<B: Backend>(&self, device: &B::Device) -> XCMPlus<B> {
        XCMPlus::new(self.clone(), device)
    }
}

/// XCMPlus model for explainable time series classification.
#[derive(Module, Debug)]
pub struct XCMPlus<B: Backend> {
    /// Time-wise convolutions (one per window size).
    time_convs: Vec<Conv1d<B>>,
    time_bns: Vec<BatchNorm<B, 1>>,
    /// Variable-wise convolution.
    var_conv: Conv1d<B>,
    var_bn: BatchNorm<B, 1>,
    /// Global average pooling.
    gap: AdaptiveAvgPool1d,
    /// Final classifier.
    fc: Linear<B>,
}

impl<B: Backend> XCMPlus<B> {
    /// Create a new XCMPlus model.
    pub fn new(config: XCMPlusConfig, device: &B::Device) -> Self {
        let mut time_convs = Vec::new();
        let mut time_bns = Vec::new();

        // Create multi-scale time-wise convolutions
        for &window_size in &config.window_sizes {
            let conv = Conv1dConfig::new(config.n_vars, config.n_filters, window_size)
                .with_padding(burn::nn::PaddingConfig1d::Same)
                .with_bias(false)
                .init(device);
            let bn = BatchNormConfig::new(config.n_filters).init(device);
            time_convs.push(conv);
            time_bns.push(bn);
        }

        // Variable-wise convolution
        let var_conv = Conv1dConfig::new(config.seq_len, config.n_filters, 1)
            .with_bias(false)
            .init(device);
        let var_bn = BatchNormConfig::new(config.n_filters).init(device);

        let gap = AdaptiveAvgPool1dConfig::new(1).init();

        // Combined features from all branches
        let combined_features = config.n_filters * (config.window_sizes.len() + 1);
        let fc = LinearConfig::new(combined_features, config.n_classes).init(device);

        Self {
            time_convs,
            time_bns,
            var_conv,
            var_bn,
            gap,
            fc,
        }
    }

    /// Forward pass.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let relu = Relu::new();
        let mut features = Vec::new();

        // Time-wise convolutions
        for (conv, bn) in self.time_convs.iter().zip(&self.time_bns) {
            let out = conv.forward(x.clone());
            let out = bn.forward(out);
            let out = relu.forward(out);
            let out = self.gap.forward(out);
            features.push(out);
        }

        // Variable-wise convolution (transpose first)
        let x_t = x.swap_dims(1, 2); // (B, L, V)
        let var_out = self.var_conv.forward(x_t);
        let var_out = self.var_bn.forward(var_out);
        let var_out = relu.forward(var_out);
        let var_out = self.gap.forward(var_out);
        features.push(var_out);

        // Concatenate all features
        let combined = Tensor::cat(features, 1);
        let [batch, channels, _] = combined.dims();
        let combined = combined.reshape([batch, channels]);

        self.fc.forward(combined)
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
    fn test_xcm_config() {
        let config = XCMPlusConfig::default();
        assert_eq!(config.n_filters, 128);
        assert_eq!(config.window_sizes.len(), 3);
    }
}
