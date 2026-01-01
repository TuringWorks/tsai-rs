//! MLP: Multilayer Perceptron for time series classification.
//!
//! A simple but effective baseline model using fully connected layers.

use burn::nn::{
    pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig},
    BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, Relu,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use serde::{Deserialize, Serialize};

/// Configuration for MLP model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLPConfig {
    /// Number of input variables/channels.
    pub n_vars: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// Hidden layer sizes.
    pub hidden_sizes: Vec<usize>,
    /// Dropout rate.
    pub dropout: f64,
    /// Whether to use batch normalization.
    pub use_bn: bool,
    /// Pooling strategy: "flatten" or "gap" (global average pooling).
    pub pool: String,
}

impl Default for MLPConfig {
    fn default() -> Self {
        Self {
            n_vars: 1,
            seq_len: 100,
            n_classes: 2,
            hidden_sizes: vec![500, 500, 500],
            dropout: 0.1,
            use_bn: false,
            pool: "flatten".to_string(),
        }
    }
}

impl MLPConfig {
    /// Create a new config.
    pub fn new(n_vars: usize, seq_len: usize, n_classes: usize) -> Self {
        Self {
            n_vars,
            seq_len,
            n_classes,
            ..Default::default()
        }
    }

    /// Set hidden layer sizes.
    #[must_use]
    pub fn with_hidden_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.hidden_sizes = sizes;
        self
    }

    /// Set dropout rate.
    #[must_use]
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Enable or disable batch normalization.
    #[must_use]
    pub fn with_bn(mut self, use_bn: bool) -> Self {
        self.use_bn = use_bn;
        self
    }

    /// Set pooling strategy.
    #[must_use]
    pub fn with_pool(mut self, pool: &str) -> Self {
        self.pool = pool.to_string();
        self
    }

    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> MLP<B> {
        MLP::new(self.clone(), device)
    }
}

/// MLP block with optional batch normalization.
#[derive(Module, Debug)]
struct MLPBlock<B: Backend> {
    linear: Linear<B>,
    bn: Option<BatchNorm<B, 1>>,
    dropout: Dropout,
}

impl<B: Backend> MLPBlock<B> {
    fn new(in_features: usize, out_features: usize, dropout: f64, use_bn: bool, device: &B::Device) -> Self {
        let linear = LinearConfig::new(in_features, out_features).init(device);
        let bn = if use_bn {
            Some(BatchNormConfig::new(out_features).init(device))
        } else {
            None
        };
        let dropout_layer = DropoutConfig::new(dropout).init();

        Self { linear, bn, dropout: dropout_layer }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let out = self.linear.forward(x);
        let out = Relu::new().forward(out);
        let out = if let Some(ref bn) = self.bn {
            // Reshape for BatchNorm1d: (B, F) -> (B, F, 1) -> (B, F)
            let [batch, features] = out.dims();
            let out = out.reshape([batch, features, 1]);
            let out = bn.forward(out);
            out.reshape([batch, features])
        } else {
            out
        };
        self.dropout.forward(out)
    }
}

/// MLP: Multilayer Perceptron for time series classification.
///
/// A simple baseline model that flattens the time series input
/// and passes it through fully connected layers.
///
/// # Architecture
///
/// ```text
/// Input (B, V, L)
///       |
///       +---> [Flatten or GAP] -> (B, F)
///       |
///       +---> [Linear + ReLU + BN? + Dropout] x N
///       |
///       +---> [Linear] -> Output (B, n_classes)
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use tsai_models::cnn::{MLP, MLPConfig};
///
/// let config = MLPConfig::new(3, 100, 5)
///     .with_hidden_sizes(vec![256, 128, 64])
///     .with_dropout(0.2)
///     .with_bn(true);
/// let model = config.init::<NdArray>(&device);
///
/// let x = Tensor::random([32, 3, 100], Distribution::Normal(0.0, 1.0), &device);
/// let output = model.forward(x);
/// // output shape: [32, 5]
/// ```
#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    /// Optional global average pooling.
    gap: Option<AdaptiveAvgPool1d>,
    /// Hidden layers.
    blocks: Vec<MLPBlock<B>>,
    /// Output layer.
    head: Linear<B>,
    /// Input features (after flatten/pool).
    #[module(skip)]
    in_features: usize,
    /// Whether to use GAP.
    #[module(skip)]
    use_gap: bool,
}

impl<B: Backend> MLP<B> {
    /// Create a new MLP model.
    pub fn new(config: MLPConfig, device: &B::Device) -> Self {
        let use_gap = config.pool == "gap";

        let (gap, in_features) = if use_gap {
            (Some(AdaptiveAvgPool1dConfig::new(1).init()), config.n_vars)
        } else {
            (None, config.n_vars * config.seq_len)
        };

        // Build hidden layers
        let mut blocks = Vec::new();
        let mut prev_size = in_features;
        for &hidden_size in &config.hidden_sizes {
            blocks.push(MLPBlock::new(prev_size, hidden_size, config.dropout, config.use_bn, device));
            prev_size = hidden_size;
        }

        // Output layer
        let head = LinearConfig::new(prev_size, config.n_classes).init(device);

        Self {
            gap,
            blocks,
            head,
            in_features,
            use_gap,
        }
    }

    /// Forward pass.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, n_vars, seq_len] = x.dims();

        // Flatten or GAP
        let out = if self.use_gap {
            let out = self.gap.as_ref().unwrap().forward(x);
            out.reshape([batch_size, n_vars])
        } else {
            x.reshape([batch_size, n_vars * seq_len])
        };

        // Apply hidden layers
        let mut out = out;
        for block in &self.blocks {
            out = block.forward(out);
        }

        // Output
        self.head.forward(out)
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
    fn test_mlp_config_default() {
        let config = MLPConfig::default();
        assert_eq!(config.hidden_sizes, vec![500, 500, 500]);
        assert_eq!(config.dropout, 0.1);
        assert!(!config.use_bn);
    }

    #[test]
    fn test_mlp_config_new() {
        let config = MLPConfig::new(3, 200, 10);
        assert_eq!(config.n_vars, 3);
        assert_eq!(config.seq_len, 200);
        assert_eq!(config.n_classes, 10);
    }

    #[test]
    fn test_mlp_config_builder() {
        let config = MLPConfig::new(3, 100, 5)
            .with_hidden_sizes(vec![256, 128])
            .with_dropout(0.3)
            .with_bn(true)
            .with_pool("gap");

        assert_eq!(config.hidden_sizes, vec![256, 128]);
        assert_eq!(config.dropout, 0.3);
        assert!(config.use_bn);
        assert_eq!(config.pool, "gap");
    }
}
