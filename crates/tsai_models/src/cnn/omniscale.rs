//! OmniScaleCNN (OS-CNN) for time series classification.
//!
//! Multi-scale convolutional architecture that captures patterns at different
//! temporal scales simultaneously.
//!
//! Based on the paper "Omni-Scale CNNs: a simple and effective kernel size
//! configuration for time series classification" by Tang et al. (2020).

use burn::nn::{
    conv::{Conv1d, Conv1dConfig},
    pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig},
    BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, Relu,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use serde::{Deserialize, Serialize};

/// Configuration for OmniScaleCNN model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OmniScaleCNNConfig {
    /// Number of input variables/channels.
    pub n_vars: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// Number of filters per scale.
    pub n_filters: usize,
    /// Kernel sizes to use (e.g., [1, 2, 4, 8, 16]).
    /// If empty, will auto-generate based on sequence length.
    pub kernel_sizes: Vec<usize>,
    /// Number of convolutional layers per scale.
    pub n_layers: usize,
    /// Dropout rate.
    pub dropout: f64,
    /// Maximum kernel size (for auto-generation).
    pub max_kernel_size: Option<usize>,
}

impl Default for OmniScaleCNNConfig {
    fn default() -> Self {
        Self {
            n_vars: 1,
            seq_len: 100,
            n_classes: 2,
            n_filters: 64,
            kernel_sizes: vec![],  // Will auto-generate
            n_layers: 1,
            dropout: 0.1,
            max_kernel_size: None,
        }
    }
}

impl OmniScaleCNNConfig {
    /// Create a new config with specified dimensions.
    pub fn new(n_vars: usize, seq_len: usize, n_classes: usize) -> Self {
        Self {
            n_vars,
            seq_len,
            n_classes,
            ..Default::default()
        }
    }

    /// Set the number of filters per scale.
    #[must_use]
    pub fn with_n_filters(mut self, n_filters: usize) -> Self {
        self.n_filters = n_filters;
        self
    }

    /// Set custom kernel sizes.
    #[must_use]
    pub fn with_kernel_sizes(mut self, kernel_sizes: Vec<usize>) -> Self {
        self.kernel_sizes = kernel_sizes;
        self
    }

    /// Set the number of layers per scale.
    #[must_use]
    pub fn with_n_layers(mut self, n_layers: usize) -> Self {
        self.n_layers = n_layers;
        self
    }

    /// Set dropout rate.
    #[must_use]
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Set maximum kernel size for auto-generation.
    #[must_use]
    pub fn with_max_kernel_size(mut self, max_kernel_size: usize) -> Self {
        self.max_kernel_size = Some(max_kernel_size);
        self
    }

    /// Get kernel sizes (auto-generate if not specified).
    ///
    /// Kernel sizes are always odd to support "Same" padding in convolutions.
    pub fn get_kernel_sizes(&self) -> Vec<usize> {
        if !self.kernel_sizes.is_empty() {
            return self.kernel_sizes.clone();
        }

        // Auto-generate kernel sizes: 1, 3, 5, 9, 17, ... (2^n + 1 to ensure odd)
        // This captures different temporal scales while being compatible with "Same" padding
        let max_k = self.max_kernel_size.unwrap_or(self.seq_len / 2).max(1);
        let mut sizes = vec![1, 3, 5]; // Small odd kernel sizes
        let mut k = 8;
        while k < max_k {
            sizes.push(k + 1); // 9, 17, 33, ... (always odd)
            k *= 2;
        }
        sizes
    }

    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> OmniScaleCNN<B> {
        OmniScaleCNN::new(self.clone(), device)
    }
}

/// A single scale branch: multiple Conv1d -> BatchNorm -> ReLU layers.
#[derive(Module, Debug)]
struct ScaleBranch<B: Backend> {
    /// Convolutional layers.
    convs: Vec<Conv1d<B>>,
    /// Batch normalization layers.
    bns: Vec<BatchNorm<B, 1>>,
}

impl<B: Backend> ScaleBranch<B> {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        n_layers: usize,
        device: &B::Device,
    ) -> Self {
        let mut convs = Vec::new();
        let mut bns = Vec::new();

        for i in 0..n_layers {
            let in_ch = if i == 0 { in_channels } else { out_channels };

            let conv = Conv1dConfig::new(in_ch, out_channels, kernel_size)
                .with_padding(burn::nn::PaddingConfig1d::Same)
                .with_bias(false)
                .init(device);

            let bn = BatchNormConfig::new(out_channels).init(device);

            convs.push(conv);
            bns.push(bn);
        }

        Self { convs, bns }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut out = x;
        for (conv, bn) in self.convs.iter().zip(self.bns.iter()) {
            out = conv.forward(out);
            out = bn.forward(out);
            out = Relu::new().forward(out);
        }
        out
    }
}

/// OmniScaleCNN (OS-CNN) for time series classification.
///
/// Uses multiple parallel convolutional branches with different kernel sizes
/// to capture patterns at different temporal scales. Outputs from all scales
/// are concatenated before global pooling and classification.
///
/// # Architecture
///
/// ```text
/// Input -> [Scale1(k=1)] -> Concat -> GAP -> Dropout -> Linear -> Output
///       -> [Scale2(k=2)] ->
///       -> [Scale3(k=4)] ->
///       -> [Scale4(k=8)] ->
///       -> ...
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use tsai_models::cnn::OmniScaleCNN;
///
/// let config = OmniScaleCNNConfig::new(3, 100, 5)
///     .with_n_filters(32)
///     .with_kernel_sizes(vec![1, 2, 4, 8, 16]);
/// let model = config.init::<NdArray>(&device);
///
/// let x = Tensor::random([32, 3, 100], Distribution::Normal(0.0, 1.0), &device);
/// let output = model.forward(x);
/// // output shape: [32, 5]
/// ```
#[derive(Module, Debug)]
pub struct OmniScaleCNN<B: Backend> {
    /// Scale branches (one per kernel size).
    branches: Vec<ScaleBranch<B>>,
    /// Global average pooling.
    gap: AdaptiveAvgPool1d,
    /// Dropout.
    dropout: Dropout,
    /// Final classifier.
    fc: Linear<B>,
    /// Number of scales (for reshaping).
    n_scales: usize,
    /// Filters per scale.
    n_filters: usize,
}

impl<B: Backend> OmniScaleCNN<B> {
    /// Create a new OmniScaleCNN model.
    pub fn new(config: OmniScaleCNNConfig, device: &B::Device) -> Self {
        let kernel_sizes = config.get_kernel_sizes();
        let n_scales = kernel_sizes.len();

        // Create a branch for each kernel size
        let branches: Vec<_> = kernel_sizes
            .iter()
            .map(|&k| {
                ScaleBranch::new(
                    config.n_vars,
                    config.n_filters,
                    k,
                    config.n_layers,
                    device,
                )
            })
            .collect();

        let gap = AdaptiveAvgPool1dConfig::new(1).init();
        let dropout = DropoutConfig::new(config.dropout).init();

        // Total channels after concatenation
        let total_channels = config.n_filters * n_scales;
        let fc = LinearConfig::new(total_channels, config.n_classes).init(device);

        Self {
            branches,
            gap,
            dropout,
            fc,
            n_scales,
            n_filters: config.n_filters,
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
        let [batch, _vars, _seq_len] = x.dims();

        // Process each scale branch
        let mut scale_outputs: Vec<Tensor<B, 3>> = Vec::with_capacity(self.n_scales);
        for branch in &self.branches {
            let out = branch.forward(x.clone());
            scale_outputs.push(out);
        }

        // Concatenate all scale outputs along channel dimension
        let concatenated = Tensor::cat(scale_outputs, 1);

        // Global average pooling
        let pooled = self.gap.forward(concatenated);

        // Flatten
        let [_, channels, _] = pooled.dims();
        let flat = pooled.reshape([batch, channels]);

        // Dropout and classify
        let out = self.dropout.forward(flat);
        self.fc.forward(out)
    }

    /// Forward pass returning probabilities.
    pub fn forward_probs(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let logits = self.forward(x);
        softmax(logits, 1)
    }

    /// Get the number of scales used.
    pub fn n_scales(&self) -> usize {
        self.n_scales
    }

    /// Get the number of filters per scale.
    pub fn n_filters(&self) -> usize {
        self.n_filters
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_omniscale_config_default() {
        let config = OmniScaleCNNConfig::default();
        assert_eq!(config.n_vars, 1);
        assert_eq!(config.n_filters, 64);
        assert_eq!(config.n_layers, 1);
        assert_eq!(config.dropout, 0.1);
    }

    #[test]
    fn test_omniscale_config_new() {
        let config = OmniScaleCNNConfig::new(3, 200, 10);
        assert_eq!(config.n_vars, 3);
        assert_eq!(config.seq_len, 200);
        assert_eq!(config.n_classes, 10);
    }

    #[test]
    fn test_omniscale_config_builder() {
        let config = OmniScaleCNNConfig::new(3, 100, 5)
            .with_n_filters(32)
            .with_kernel_sizes(vec![1, 2, 4, 8])
            .with_n_layers(2)
            .with_dropout(0.2);

        assert_eq!(config.n_filters, 32);
        assert_eq!(config.kernel_sizes, vec![1, 2, 4, 8]);
        assert_eq!(config.n_layers, 2);
        assert_eq!(config.dropout, 0.2);
    }

    #[test]
    fn test_auto_kernel_sizes() {
        let config = OmniScaleCNNConfig::new(1, 100, 2);
        let sizes = config.get_kernel_sizes();

        // Should generate odd kernel sizes: 1, 3, 5, 9, 17, 33
        // (compatible with "Same" padding)
        assert_eq!(sizes, vec![1, 3, 5, 9, 17, 33]);
    }

    #[test]
    fn test_auto_kernel_sizes_short_seq() {
        let config = OmniScaleCNNConfig::new(1, 10, 2);
        let sizes = config.get_kernel_sizes();

        // Should generate base odd sizes only (max_k = 5)
        assert_eq!(sizes, vec![1, 3, 5]);
    }

    #[test]
    fn test_custom_kernel_sizes() {
        let config = OmniScaleCNNConfig::new(1, 100, 2)
            .with_kernel_sizes(vec![3, 5, 7, 11]);
        let sizes = config.get_kernel_sizes();

        assert_eq!(sizes, vec![3, 5, 7, 11]);
    }

    #[test]
    fn test_max_kernel_size_limit() {
        let config = OmniScaleCNNConfig::new(1, 1000, 2)
            .with_max_kernel_size(16);
        let sizes = config.get_kernel_sizes();

        // Should generate odd sizes up to max_k = 16: 1, 3, 5, 9 (8+1)
        assert_eq!(sizes, vec![1, 3, 5, 9]);
    }
}
