//! mWDN: Multilevel Wavelet Decomposition Network.
//!
//! A model that uses discrete wavelet transform (DWT) to decompose time series
//! at multiple levels, extracting both approximation (low-frequency) and
//! detail (high-frequency) coefficients at each level.
//!
//! Reference: "Time Series Classification Using Multi-Channels Deep
//! Convolutional Neural Networks" by Zheng et al.

use burn::nn::{
    conv::{Conv1d, Conv1dConfig},
    pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig},
    BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, Relu,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use serde::{Deserialize, Serialize};

/// Wavelet type for decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WaveletType {
    /// Haar wavelet (simple average/difference).
    #[default]
    Haar,
    /// Daubechies-2 wavelet.
    Db2,
    /// Daubechies-4 wavelet.
    Db4,
}

/// Configuration for mWDN model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MWDNConfig {
    /// Number of input variables/channels.
    pub n_vars: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// Number of wavelet decomposition levels.
    pub n_levels: usize,
    /// Wavelet type.
    pub wavelet: WaveletType,
    /// Hidden size for feature processing.
    pub hidden_size: usize,
    /// Dropout rate.
    pub dropout: f64,
}

impl Default for MWDNConfig {
    fn default() -> Self {
        Self {
            n_vars: 1,
            seq_len: 100,
            n_classes: 2,
            n_levels: 3,
            wavelet: WaveletType::Haar,
            hidden_size: 128,
            dropout: 0.1,
        }
    }
}

impl MWDNConfig {
    /// Create a new config.
    pub fn new(n_vars: usize, seq_len: usize, n_classes: usize) -> Self {
        Self {
            n_vars,
            seq_len,
            n_classes,
            ..Default::default()
        }
    }

    /// Set number of decomposition levels.
    #[must_use]
    pub fn with_n_levels(mut self, n_levels: usize) -> Self {
        self.n_levels = n_levels;
        self
    }

    /// Set wavelet type.
    #[must_use]
    pub fn with_wavelet(mut self, wavelet: WaveletType) -> Self {
        self.wavelet = wavelet;
        self
    }

    /// Set hidden size.
    #[must_use]
    pub fn with_hidden_size(mut self, hidden_size: usize) -> Self {
        self.hidden_size = hidden_size;
        self
    }

    /// Set dropout rate.
    #[must_use]
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> MWDN<B> {
        MWDN::new(self.clone(), device)
    }
}

/// Convolutional block for processing wavelet coefficients.
#[derive(Module, Debug)]
struct WaveletConvBlock<B: Backend> {
    conv1: Conv1d<B>,
    bn1: BatchNorm<B, 1>,
    conv2: Conv1d<B>,
    bn2: BatchNorm<B, 1>,
}

impl<B: Backend> WaveletConvBlock<B> {
    fn new(in_channels: usize, out_channels: usize, device: &B::Device) -> Self {
        let conv1 = Conv1dConfig::new(in_channels, out_channels, 3)
            .with_padding(burn::nn::PaddingConfig1d::Same)
            .init(device);
        let bn1 = BatchNormConfig::new(out_channels).init(device);
        let conv2 = Conv1dConfig::new(out_channels, out_channels, 3)
            .with_padding(burn::nn::PaddingConfig1d::Same)
            .init(device);
        let bn2 = BatchNormConfig::new(out_channels).init(device);

        Self { conv1, bn1, conv2, bn2 }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let out = self.conv1.forward(x);
        let out = self.bn1.forward(out);
        let out = Relu::new().forward(out);
        let out = self.conv2.forward(out);
        let out = self.bn2.forward(out);
        Relu::new().forward(out)
    }
}

/// mWDN: Multilevel Wavelet Decomposition Network.
///
/// This model applies discrete wavelet transform to decompose the input
/// at multiple resolution levels, then processes each level with
/// convolutional layers before combining for classification.
///
/// # Architecture
///
/// ```text
/// Input (B, V, L)
///       |
///       +---> Level 1: [DWT] -> [Approx: A1] + [Detail: D1]
///       |                           |              |
///       |                       ConvBlock      ConvBlock
///       |                           |              |
///       +---> Level 2: [DWT(A1)] -> [A2] + [D2]   |
///       |                            |      |      |
///       |                        ConvBlock ConvBlock
///       |                            |      |      |
///       +---> Level 3: [DWT(A2)] -> [A3] + [D3]   |
///                                    |      |      |
///                                ConvBlock ConvBlock
///                                    |      |      |
///                                   GAP    GAP    ...
///                                    \      |     /
///                                     Concatenate
///                                         |
///                                      Linear
///                                         |
///                                      Output
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use tsai_models::cnn::{MWDN, MWDNConfig, WaveletType};
///
/// let config = MWDNConfig::new(3, 100, 5)
///     .with_n_levels(4)
///     .with_wavelet(WaveletType::Haar);
/// let model = config.init::<NdArray>(&device);
///
/// let x = Tensor::random([32, 3, 100], Distribution::Normal(0.0, 1.0), &device);
/// let output = model.forward(x);
/// // output shape: [32, 5]
/// ```
#[derive(Module, Debug)]
pub struct MWDN<B: Backend> {
    /// Convolution blocks for approximation coefficients at each level.
    approx_blocks: Vec<WaveletConvBlock<B>>,
    /// Convolution blocks for detail coefficients at each level.
    detail_blocks: Vec<WaveletConvBlock<B>>,
    /// Global average pooling.
    gap: AdaptiveAvgPool1d,
    /// Dropout layer.
    dropout: Dropout,
    /// Final classifier.
    classifier: Linear<B>,
    /// Number of levels.
    #[module(skip)]
    n_levels: usize,
    /// Hidden size for pooled features.
    #[module(skip)]
    hidden_size: usize,
}

impl<B: Backend> MWDN<B> {
    /// Create a new mWDN model.
    pub fn new(config: MWDNConfig, device: &B::Device) -> Self {
        let mut approx_blocks = Vec::new();
        let mut detail_blocks = Vec::new();

        // Create conv blocks for each level
        for _ in 0..config.n_levels {
            approx_blocks.push(WaveletConvBlock::new(config.n_vars, config.hidden_size, device));
            detail_blocks.push(WaveletConvBlock::new(config.n_vars, config.hidden_size, device));
        }

        let gap = AdaptiveAvgPool1dConfig::new(1).init();
        let dropout = DropoutConfig::new(config.dropout).init();

        // Classifier: n_levels * 2 (approx + detail) * hidden_size
        let classifier_in = config.n_levels * 2 * config.hidden_size;
        let classifier = LinearConfig::new(classifier_in, config.n_classes).init(device);

        Self {
            approx_blocks,
            detail_blocks,
            gap,
            dropout,
            classifier,
            n_levels: config.n_levels,
            hidden_size: config.hidden_size,
        }
    }

    /// Apply Haar wavelet decomposition.
    /// Returns (approximation, detail) coefficients.
    fn haar_dwt(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [batch, channels, seq_len] = x.dims();
        let new_len = seq_len / 2;

        if new_len == 0 {
            // If sequence is too short, just return the input as both
            return (x.clone(), x);
        }

        // Reshape for pairwise operations: (B, C, L) -> (B, C, L/2, 2)
        let truncated_len = new_len * 2;
        let x_truncated = x.clone().slice([0..batch, 0..channels, 0..truncated_len]);
        let x_reshaped = x_truncated.reshape([batch, channels, new_len, 2]);

        // Get even and odd samples
        let x_even = x_reshaped.clone().slice([0..batch, 0..channels, 0..new_len, 0..1])
            .reshape([batch, channels, new_len]);
        let x_odd = x_reshaped.slice([0..batch, 0..channels, 0..new_len, 1..2])
            .reshape([batch, channels, new_len]);

        // Haar wavelet: approx = (even + odd) / sqrt(2), detail = (even - odd) / sqrt(2)
        let sqrt2 = std::f32::consts::SQRT_2;
        let approx = (x_even.clone() + x_odd.clone()) / sqrt2;
        let detail = (x_even - x_odd) / sqrt2;

        (approx, detail)
    }

    /// Forward pass.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, _channels, _seq_len] = x.dims();

        let mut all_features = Vec::new();
        let mut current = x;

        // Apply wavelet decomposition at each level
        for level in 0..self.n_levels {
            let (approx, detail) = self.haar_dwt(current.clone());

            // Process approximation coefficients
            let approx_features = self.approx_blocks[level].forward(approx.clone());
            let approx_pooled = self.gap.forward(approx_features);
            let approx_pooled = approx_pooled.reshape([batch_size, self.hidden_size]);

            // Process detail coefficients
            let detail_features = self.detail_blocks[level].forward(detail);
            let detail_pooled = self.gap.forward(detail_features);
            let detail_pooled = detail_pooled.reshape([batch_size, self.hidden_size]);

            all_features.push(approx_pooled);
            all_features.push(detail_pooled);

            // Use approximation for next level
            current = approx;
        }

        // Concatenate all features
        let combined = Tensor::cat(all_features, 1);
        let combined = self.dropout.forward(combined);

        // Classify
        self.classifier.forward(combined)
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
    fn test_mwdn_config_default() {
        let config = MWDNConfig::default();
        assert_eq!(config.n_levels, 3);
        assert_eq!(config.wavelet, WaveletType::Haar);
        assert_eq!(config.hidden_size, 128);
    }

    #[test]
    fn test_mwdn_config_new() {
        let config = MWDNConfig::new(3, 200, 10);
        assert_eq!(config.n_vars, 3);
        assert_eq!(config.seq_len, 200);
        assert_eq!(config.n_classes, 10);
    }

    #[test]
    fn test_mwdn_config_builder() {
        let config = MWDNConfig::new(3, 100, 5)
            .with_n_levels(4)
            .with_wavelet(WaveletType::Db2)
            .with_hidden_size(64)
            .with_dropout(0.2);

        assert_eq!(config.n_levels, 4);
        assert_eq!(config.wavelet, WaveletType::Db2);
        assert_eq!(config.hidden_size, 64);
        assert_eq!(config.dropout, 0.2);
    }
}
