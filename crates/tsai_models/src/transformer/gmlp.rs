//! gMLP: Gated MLP for time series classification.
//!
//! A transformer alternative that uses gated linear units and spatial
//! gating instead of self-attention.
//!
//! Reference: "Pay Attention to MLPs" by Liu et al. (2021)

use burn::nn::{
    pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig},
    Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
};
use burn::prelude::*;
use burn::tensor::activation::{gelu, softmax};
use serde::{Deserialize, Serialize};

/// Configuration for gMLP model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GMLPConfig {
    /// Number of input variables/channels.
    pub n_vars: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// Model dimension (hidden size).
    pub d_model: usize,
    /// Feedforward expansion factor.
    pub ff_mult: usize,
    /// Number of gMLP blocks.
    pub n_layers: usize,
    /// Dropout rate.
    pub dropout: f64,
}

impl Default for GMLPConfig {
    fn default() -> Self {
        Self {
            n_vars: 1,
            seq_len: 100,
            n_classes: 2,
            d_model: 128,
            ff_mult: 4,
            n_layers: 4,
            dropout: 0.1,
        }
    }
}

impl GMLPConfig {
    /// Create a new config.
    pub fn new(n_vars: usize, seq_len: usize, n_classes: usize) -> Self {
        Self {
            n_vars,
            seq_len,
            n_classes,
            ..Default::default()
        }
    }

    /// Set model dimension.
    #[must_use]
    pub fn with_d_model(mut self, d_model: usize) -> Self {
        self.d_model = d_model;
        self
    }

    /// Set feedforward expansion factor.
    #[must_use]
    pub fn with_ff_mult(mut self, ff_mult: usize) -> Self {
        self.ff_mult = ff_mult;
        self
    }

    /// Set number of layers.
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

    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> GMLP<B> {
        GMLP::new(self.clone(), device)
    }
}

/// Spatial Gating Unit (SGU) for gMLP.
///
/// Performs spatial mixing through a gated mechanism.
#[derive(Module, Debug)]
struct SpatialGatingUnit<B: Backend> {
    norm: LayerNorm<B>,
    proj: Linear<B>,
    #[module(skip)]
    seq_len: usize,
}

impl<B: Backend> SpatialGatingUnit<B> {
    fn new(d_ff: usize, seq_len: usize, device: &B::Device) -> Self {
        let norm = LayerNormConfig::new(d_ff / 2).init(device);
        // Project from seq_len to seq_len (spatial mixing)
        let proj = LinearConfig::new(seq_len, seq_len).init(device);

        Self { norm, proj, seq_len }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, d_ff] = x.dims();
        let half_d = d_ff / 2;

        // Split into two halves along feature dimension
        let u = x.clone().slice([0..batch, 0..seq_len, 0..half_d]);
        let v = x.slice([0..batch, 0..seq_len, half_d..d_ff]);

        // Apply spatial gating to v
        let v = self.norm.forward(v);
        // Transpose for spatial projection: (B, L, D/2) -> (B, D/2, L)
        let v = v.swap_dims(1, 2);
        let v = self.proj.forward(v);
        // Transpose back: (B, D/2, L) -> (B, L, D/2)
        let v = v.swap_dims(1, 2);

        // Gate: u * v
        u * v
    }
}

/// gMLP block with channel projection and spatial gating.
#[derive(Module, Debug)]
struct GMLPBlock<B: Backend> {
    norm: LayerNorm<B>,
    proj_in: Linear<B>,
    sgu: SpatialGatingUnit<B>,
    proj_out: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> GMLPBlock<B> {
    fn new(d_model: usize, d_ff: usize, seq_len: usize, dropout: f64, device: &B::Device) -> Self {
        let norm = LayerNormConfig::new(d_model).init(device);
        let proj_in = LinearConfig::new(d_model, d_ff).init(device);
        let sgu = SpatialGatingUnit::new(d_ff, seq_len, device);
        let proj_out = LinearConfig::new(d_ff / 2, d_model).init(device);
        let dropout_layer = DropoutConfig::new(dropout).init();

        Self {
            norm,
            proj_in,
            sgu,
            proj_out,
            dropout: dropout_layer,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();

        // Normalize
        let out = self.norm.forward(x);

        // Project up
        let out = self.proj_in.forward(out);
        let out = gelu(out);

        // Spatial gating
        let out = self.sgu.forward(out);

        // Project down
        let out = self.proj_out.forward(out);
        let out = self.dropout.forward(out);

        // Residual connection
        residual + out
    }
}

/// gMLP: Gated MLP for time series classification.
///
/// Uses spatial gating units instead of self-attention for mixing
/// information across the sequence dimension.
///
/// # Architecture
///
/// ```text
/// Input (B, V, L)
///       |
///       +---> [Permute to (B, L, V)] -> [Proj to d_model]
///       |
///       +---> [gMLP Block 1] -> [gMLP Block 2] -> ... -> [gMLP Block N]
///       |
///       +---> [Global Average Pool] -> (B, D)
///       |
///       +---> [Linear] -> Output (B, n_classes)
/// ```
///
/// Each gMLP Block:
/// ```text
/// x -> [LayerNorm] -> [Linear up] -> [GELU] -> [SGU] -> [Linear down] -> + x
///                                                |
///                                        Spatial Gating Unit:
///                                        split -> (u, v)
///                                        v = Norm(v)
///                                        v = SpatialProj(v)
///                                        output = u * v
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use tsai_models::transformer::{GMLP, GMLPConfig};
///
/// let config = GMLPConfig::new(3, 100, 5)
///     .with_d_model(256)
///     .with_n_layers(6);
/// let model = config.init::<NdArray>(&device);
///
/// let x = Tensor::random([32, 3, 100], Distribution::Normal(0.0, 1.0), &device);
/// let output = model.forward(x);
/// // output shape: [32, 5]
/// ```
#[derive(Module, Debug)]
pub struct GMLP<B: Backend> {
    /// Input projection.
    input_proj: Linear<B>,
    /// gMLP blocks.
    blocks: Vec<GMLPBlock<B>>,
    /// Final layer norm.
    final_norm: LayerNorm<B>,
    /// Global average pooling.
    gap: AdaptiveAvgPool1d,
    /// Classification head.
    head: Linear<B>,
    /// Model dimension.
    #[module(skip)]
    d_model: usize,
}

impl<B: Backend> GMLP<B> {
    /// Create a new gMLP model.
    pub fn new(config: GMLPConfig, device: &B::Device) -> Self {
        let d_ff = config.d_model * config.ff_mult;

        // Input projection
        let input_proj = LinearConfig::new(config.n_vars, config.d_model).init(device);

        // gMLP blocks
        let blocks: Vec<_> = (0..config.n_layers)
            .map(|_| GMLPBlock::new(config.d_model, d_ff, config.seq_len, config.dropout, device))
            .collect();

        let final_norm = LayerNormConfig::new(config.d_model).init(device);
        let gap = AdaptiveAvgPool1dConfig::new(1).init();
        let head = LinearConfig::new(config.d_model, config.n_classes).init(device);

        Self {
            input_proj,
            blocks,
            final_norm,
            gap,
            head,
            d_model: config.d_model,
        }
    }

    /// Forward pass.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, _n_vars, _seq_len] = x.dims();

        // Transpose: (B, V, L) -> (B, L, V)
        let out = x.swap_dims(1, 2);

        // Project to d_model
        let out = self.input_proj.forward(out);

        // Apply gMLP blocks
        let mut out = out;
        for block in &self.blocks {
            out = block.forward(out);
        }

        // Final normalization
        let out = self.final_norm.forward(out);

        // Transpose for pooling: (B, L, D) -> (B, D, L)
        let out = out.swap_dims(1, 2);

        // Global average pooling
        let out = self.gap.forward(out);
        let out = out.reshape([batch_size, self.d_model]);

        // Classification
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
    fn test_gmlp_config_default() {
        let config = GMLPConfig::default();
        assert_eq!(config.d_model, 128);
        assert_eq!(config.ff_mult, 4);
        assert_eq!(config.n_layers, 4);
    }

    #[test]
    fn test_gmlp_config_new() {
        let config = GMLPConfig::new(3, 200, 10);
        assert_eq!(config.n_vars, 3);
        assert_eq!(config.seq_len, 200);
        assert_eq!(config.n_classes, 10);
    }

    #[test]
    fn test_gmlp_config_builder() {
        let config = GMLPConfig::new(3, 100, 5)
            .with_d_model(256)
            .with_ff_mult(6)
            .with_n_layers(8)
            .with_dropout(0.2);

        assert_eq!(config.d_model, 256);
        assert_eq!(config.ff_mult, 6);
        assert_eq!(config.n_layers, 8);
        assert_eq!(config.dropout, 0.2);
    }
}
