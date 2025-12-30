//! TSPerceiver: Perceiver architecture for time series.
//!
//! Uses cross-attention to map variable-length time series to a fixed-size
//! latent representation, which is then processed by self-attention layers.
//!
//! Based on the Perceiver architecture by Jaegle et al. (2021),
//! adapted for time series classification.

use burn::nn::{
    attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
    Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use serde::{Deserialize, Serialize};

/// Configuration for TSPerceiver model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TSPerceiverConfig {
    /// Number of input variables/channels.
    pub n_vars: usize,
    /// Sequence length (can vary at inference).
    pub seq_len: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// Latent dimension.
    pub d_latent: usize,
    /// Number of latent vectors.
    pub n_latents: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of cross-attention layers.
    pub n_cross_layers: usize,
    /// Number of self-attention layers per cross-attention.
    pub n_self_layers: usize,
    /// Feedforward dimension multiplier.
    pub ff_mult: usize,
    /// Dropout rate.
    pub dropout: f64,
    /// Whether to use weight sharing for cross-attention layers.
    pub share_weights: bool,
}

impl Default for TSPerceiverConfig {
    fn default() -> Self {
        Self {
            n_vars: 1,
            seq_len: 100,
            n_classes: 2,
            d_latent: 128,
            n_latents: 32,
            n_heads: 8,
            n_cross_layers: 2,
            n_self_layers: 4,
            ff_mult: 4,
            dropout: 0.1,
            share_weights: true,
        }
    }
}

impl TSPerceiverConfig {
    /// Create a new config.
    pub fn new(n_vars: usize, seq_len: usize, n_classes: usize) -> Self {
        Self {
            n_vars,
            seq_len,
            n_classes,
            ..Default::default()
        }
    }

    /// Set latent dimension.
    #[must_use]
    pub fn with_d_latent(mut self, d_latent: usize) -> Self {
        self.d_latent = d_latent;
        self
    }

    /// Set number of latent vectors.
    #[must_use]
    pub fn with_n_latents(mut self, n_latents: usize) -> Self {
        self.n_latents = n_latents;
        self
    }

    /// Set number of attention heads.
    #[must_use]
    pub fn with_n_heads(mut self, n_heads: usize) -> Self {
        self.n_heads = n_heads;
        self
    }

    /// Set number of cross-attention layers.
    #[must_use]
    pub fn with_n_cross_layers(mut self, n_cross_layers: usize) -> Self {
        self.n_cross_layers = n_cross_layers;
        self
    }

    /// Set number of self-attention layers per cross-attention.
    #[must_use]
    pub fn with_n_self_layers(mut self, n_self_layers: usize) -> Self {
        self.n_self_layers = n_self_layers;
        self
    }

    /// Set dropout rate.
    #[must_use]
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Set weight sharing for cross-attention layers.
    #[must_use]
    pub fn with_share_weights(mut self, share_weights: bool) -> Self {
        self.share_weights = share_weights;
        self
    }

    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> TSPerceiver<B> {
        TSPerceiver::new(self.clone(), device)
    }
}

/// Cross-attention block: latent attends to input.
#[derive(Module, Debug)]
struct CrossAttentionBlock<B: Backend> {
    /// Query projection for latent.
    latent_norm: LayerNorm<B>,
    /// Key/Value projection for input.
    input_norm: LayerNorm<B>,
    /// Cross-attention layer.
    attention: MultiHeadAttention<B>,
    /// Feedforward layers.
    ff_norm: LayerNorm<B>,
    ff_linear1: Linear<B>,
    ff_linear2: Linear<B>,
    /// Dropout.
    dropout: Dropout,
}

impl<B: Backend> CrossAttentionBlock<B> {
    fn new(d_latent: usize, d_input: usize, n_heads: usize, d_ff: usize, dropout: f64, device: &B::Device) -> Self {
        let latent_norm = LayerNormConfig::new(d_latent).init(device);
        let input_norm = LayerNormConfig::new(d_input).init(device);

        // Cross-attention: queries from latent, keys/values from input
        let attention = MultiHeadAttentionConfig::new(d_latent, n_heads)
            .with_dropout(dropout)
            .init(device);

        let ff_norm = LayerNormConfig::new(d_latent).init(device);
        let ff_linear1 = LinearConfig::new(d_latent, d_ff).init(device);
        let ff_linear2 = LinearConfig::new(d_ff, d_latent).init(device);
        let dropout_layer = DropoutConfig::new(dropout).init();

        Self {
            latent_norm,
            input_norm,
            attention,
            ff_norm,
            ff_linear1,
            ff_linear2,
            dropout: dropout_layer,
        }
    }

    fn forward(&self, latent: Tensor<B, 3>, input: Tensor<B, 3>, input_proj: &Linear<B>) -> Tensor<B, 3> {
        // Pre-norm
        let latent_normed = self.latent_norm.forward(latent.clone());
        let input_normed = self.input_norm.forward(input);

        // Project input to latent dimension
        let input_projected = input_proj.forward(input_normed);

        // Cross-attention: latent queries attend to input keys/values
        let attn_input = MhaInput::new(latent_normed, input_projected.clone(), input_projected);
        let attn_out = self.attention.forward(attn_input).context;
        let latent = latent + self.dropout.forward(attn_out);

        // Feedforward
        let normed = self.ff_norm.forward(latent.clone());
        let ff_out = self.ff_linear1.forward(normed);
        let ff_out = Gelu::new().forward(ff_out);
        let ff_out = self.dropout.forward(ff_out);
        let ff_out = self.ff_linear2.forward(ff_out);

        latent + self.dropout.forward(ff_out)
    }
}

/// Self-attention block for latent processing.
#[derive(Module, Debug)]
struct SelfAttentionBlock<B: Backend> {
    /// Layer normalization before attention.
    attn_norm: LayerNorm<B>,
    /// Self-attention layer.
    attention: MultiHeadAttention<B>,
    /// Feedforward layers.
    ff_norm: LayerNorm<B>,
    ff_linear1: Linear<B>,
    ff_linear2: Linear<B>,
    /// Dropout.
    dropout: Dropout,
}

impl<B: Backend> SelfAttentionBlock<B> {
    fn new(d_latent: usize, n_heads: usize, d_ff: usize, dropout: f64, device: &B::Device) -> Self {
        let attn_norm = LayerNormConfig::new(d_latent).init(device);
        let attention = MultiHeadAttentionConfig::new(d_latent, n_heads)
            .with_dropout(dropout)
            .init(device);

        let ff_norm = LayerNormConfig::new(d_latent).init(device);
        let ff_linear1 = LinearConfig::new(d_latent, d_ff).init(device);
        let ff_linear2 = LinearConfig::new(d_ff, d_latent).init(device);
        let dropout_layer = DropoutConfig::new(dropout).init();

        Self {
            attn_norm,
            attention,
            ff_norm,
            ff_linear1,
            ff_linear2,
            dropout: dropout_layer,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Pre-norm self-attention
        let normed = self.attn_norm.forward(x.clone());
        let attn_input = MhaInput::self_attn(normed);
        let attn_out = self.attention.forward(attn_input).context;
        let x = x + self.dropout.forward(attn_out);

        // Feedforward
        let normed = self.ff_norm.forward(x.clone());
        let ff_out = self.ff_linear1.forward(normed);
        let ff_out = Gelu::new().forward(ff_out);
        let ff_out = self.dropout.forward(ff_out);
        let ff_out = self.ff_linear2.forward(ff_out);

        x + self.dropout.forward(ff_out)
    }
}

/// TSPerceiver: Perceiver architecture for time series classification.
///
/// Maps variable-length time series to a fixed-size latent representation
/// using cross-attention, then processes with self-attention layers.
///
/// # Architecture
///
/// ```text
/// Input (B, V, L) -> Project -> Cross-Attn with Latents -> Self-Attn x N
///                                      ^
///                                      |
///                             Learnable Latents (N, D)
///
/// Latent Output -> Mean Pool -> Linear -> Class Logits
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use tsai_models::transformer::TSPerceiver;
///
/// let config = TSPerceiverConfig::new(3, 100, 5)
///     .with_d_latent(64)
///     .with_n_latents(16);
/// let model = config.init::<NdArray>(&device);
///
/// let x = Tensor::random([32, 3, 100], Distribution::Normal(0.0, 1.0), &device);
/// let output = model.forward(x);
/// // output shape: [32, 5]
/// ```
#[derive(Module, Debug)]
pub struct TSPerceiver<B: Backend> {
    /// Input projection to latent dimension.
    input_proj: Linear<B>,
    /// Cross-attention blocks.
    cross_attn_blocks: Vec<CrossAttentionBlock<B>>,
    /// Self-attention blocks.
    self_attn_blocks: Vec<SelfAttentionBlock<B>>,
    /// Final layer norm.
    final_norm: LayerNorm<B>,
    /// Classification head.
    head: Linear<B>,
    /// Head dropout.
    head_dropout: Dropout,
    /// Number of cross-attention layers.
    n_cross_layers: usize,
    /// Number of self-attention layers per cross.
    n_self_layers: usize,
    /// Latent dimension.
    d_latent: usize,
    /// Number of latents.
    n_latents: usize,
}

impl<B: Backend> TSPerceiver<B> {
    /// Create a new TSPerceiver model.
    pub fn new(config: TSPerceiverConfig, device: &B::Device) -> Self {
        let d_ff = config.d_latent * config.ff_mult;

        // Input projection: n_vars -> d_latent
        let input_proj = LinearConfig::new(config.n_vars, config.d_latent).init(device);

        // Cross-attention blocks
        let n_cross = if config.share_weights { 1 } else { config.n_cross_layers };
        let cross_attn_blocks: Vec<_> = (0..n_cross)
            .map(|_| {
                CrossAttentionBlock::new(
                    config.d_latent,
                    config.d_latent,
                    config.n_heads,
                    d_ff,
                    config.dropout,
                    device,
                )
            })
            .collect();

        // Self-attention blocks (shared across cross-attention layers)
        let self_attn_blocks: Vec<_> = (0..config.n_self_layers)
            .map(|_| {
                SelfAttentionBlock::new(
                    config.d_latent,
                    config.n_heads,
                    d_ff,
                    config.dropout,
                    device,
                )
            })
            .collect();

        let final_norm = LayerNormConfig::new(config.d_latent).init(device);
        let head = LinearConfig::new(config.d_latent, config.n_classes).init(device);
        let head_dropout = DropoutConfig::new(config.dropout).init();

        Self {
            input_proj,
            cross_attn_blocks,
            self_attn_blocks,
            final_norm,
            head,
            head_dropout,
            n_cross_layers: config.n_cross_layers,
            n_self_layers: config.n_self_layers,
            d_latent: config.d_latent,
            n_latents: config.n_latents,
        }
    }

    /// Initialize latent array (learnable, initialized from normal distribution).
    fn init_latents(&self, batch_size: usize, device: &B::Device) -> Tensor<B, 3> {
        // Initialize latents with small random values
        Tensor::random(
            [batch_size, self.n_latents, self.d_latent],
            burn::tensor::Distribution::Normal(0.0, 0.02),
            device,
        )
    }

    /// Forward pass.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch, _n_vars, _seq_len] = x.dims();
        let device = x.device();

        // Transpose to (batch, seq_len, n_vars)
        let x = x.swap_dims(1, 2);

        // Project input to latent dimension
        let input = self.input_proj.forward(x);

        // Initialize latent array
        let mut latent = self.init_latents(batch, &device);

        // Apply cross-attention and self-attention layers
        for i in 0..self.n_cross_layers {
            // Cross-attention: latent attends to input
            let cross_idx = if self.cross_attn_blocks.len() == 1 { 0 } else { i };
            latent = self.cross_attn_blocks[cross_idx].forward(
                latent,
                input.clone(),
                &self.input_proj,
            );

            // Self-attention on latent
            for self_attn in &self.self_attn_blocks {
                latent = self_attn.forward(latent);
            }
        }

        // Final norm
        let latent = self.final_norm.forward(latent);

        // Mean pool over latent vectors
        let pooled = latent.mean_dim(1).reshape([batch, self.d_latent]);

        // Classify
        let out = self.head_dropout.forward(pooled);
        self.head.forward(out)
    }

    /// Forward pass returning probabilities.
    pub fn forward_probs(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let logits = self.forward(x);
        softmax(logits, 1)
    }

    /// Get number of latent vectors.
    pub fn n_latents(&self) -> usize {
        self.n_latents
    }

    /// Get latent dimension.
    pub fn d_latent(&self) -> usize {
        self.d_latent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perceiver_config_default() {
        let config = TSPerceiverConfig::default();
        assert_eq!(config.d_latent, 128);
        assert_eq!(config.n_latents, 32);
        assert_eq!(config.n_heads, 8);
        assert_eq!(config.n_cross_layers, 2);
        assert_eq!(config.n_self_layers, 4);
        assert!(config.share_weights);
    }

    #[test]
    fn test_perceiver_config_new() {
        let config = TSPerceiverConfig::new(3, 200, 10);
        assert_eq!(config.n_vars, 3);
        assert_eq!(config.seq_len, 200);
        assert_eq!(config.n_classes, 10);
    }

    #[test]
    fn test_perceiver_config_builder() {
        let config = TSPerceiverConfig::new(3, 100, 5)
            .with_d_latent(64)
            .with_n_latents(16)
            .with_n_heads(4)
            .with_n_cross_layers(3)
            .with_n_self_layers(2)
            .with_dropout(0.2)
            .with_share_weights(false);

        assert_eq!(config.d_latent, 64);
        assert_eq!(config.n_latents, 16);
        assert_eq!(config.n_heads, 4);
        assert_eq!(config.n_cross_layers, 3);
        assert_eq!(config.n_self_layers, 2);
        assert_eq!(config.dropout, 0.2);
        assert!(!config.share_weights);
    }
}
