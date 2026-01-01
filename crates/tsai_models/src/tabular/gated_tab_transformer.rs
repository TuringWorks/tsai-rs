//! GatedTabTransformer: Gated variant of TabTransformer.
//!
//! Uses gated linear units (GEGLU) in the transformer feedforward layers
//! for enhanced performance on tabular data.
//!
//! Reference: "Language Models are Few-Shot Learners" (GEGLU activation)

use burn::nn::{
    attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig,
    Linear, LinearConfig,
};
use burn::prelude::*;
use burn::tensor::activation::{gelu, softmax};
use serde::{Deserialize, Serialize};

/// Configuration for GatedTabTransformer model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatedTabTransformerConfig {
    /// Number of continuous features.
    pub n_continuous: usize,
    /// Number of categorical features.
    pub n_categorical: usize,
    /// Cardinalities for each categorical feature.
    pub cat_cardinalities: Vec<usize>,
    /// Number of output classes.
    pub n_classes: usize,
    /// Model dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Feedforward expansion factor.
    pub ff_mult: usize,
    /// Dropout rate.
    pub dropout: f64,
    /// Whether to apply attention to continuous features.
    pub attn_on_continuous: bool,
}

impl Default for GatedTabTransformerConfig {
    fn default() -> Self {
        Self {
            n_continuous: 10,
            n_categorical: 5,
            cat_cardinalities: vec![10, 20, 30, 40, 50],
            n_classes: 2,
            d_model: 64,
            n_heads: 4,
            n_layers: 2,
            ff_mult: 4,
            dropout: 0.1,
            attn_on_continuous: true,
        }
    }
}

impl GatedTabTransformerConfig {
    /// Create a new config.
    pub fn new(n_continuous: usize, n_categorical: usize, n_classes: usize) -> Self {
        Self {
            n_continuous,
            n_categorical,
            n_classes,
            ..Default::default()
        }
    }

    /// Set categorical cardinalities.
    #[must_use]
    pub fn with_cardinalities(mut self, cardinalities: Vec<usize>) -> Self {
        self.cat_cardinalities = cardinalities;
        self
    }

    /// Set model dimension.
    #[must_use]
    pub fn with_d_model(mut self, d_model: usize) -> Self {
        self.d_model = d_model;
        self
    }

    /// Set number of attention heads.
    #[must_use]
    pub fn with_n_heads(mut self, n_heads: usize) -> Self {
        self.n_heads = n_heads;
        self
    }

    /// Set number of layers.
    #[must_use]
    pub fn with_n_layers(mut self, n_layers: usize) -> Self {
        self.n_layers = n_layers;
        self
    }

    /// Set feedforward expansion factor.
    #[must_use]
    pub fn with_ff_mult(mut self, ff_mult: usize) -> Self {
        self.ff_mult = ff_mult;
        self
    }

    /// Set dropout rate.
    #[must_use]
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Set whether to apply attention to continuous features.
    #[must_use]
    pub fn with_attn_on_continuous(mut self, attn_on_continuous: bool) -> Self {
        self.attn_on_continuous = attn_on_continuous;
        self
    }

    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> GatedTabTransformer<B> {
        GatedTabTransformer::new(self.clone(), device)
    }
}

/// GEGLU (Gated GELU Linear Unit) feedforward layer.
///
/// Splits the input and applies gating: x1 * GELU(x2)
#[derive(Module, Debug)]
struct GEGLU<B: Backend> {
    proj: Linear<B>,
}

impl<B: Backend> GEGLU<B> {
    fn new(in_features: usize, out_features: usize, device: &B::Device) -> Self {
        // Project to 2x size for gating
        let proj = LinearConfig::new(in_features, out_features * 2).init(device);
        Self { proj }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let out = self.proj.forward(x);
        let [batch, seq, features] = out.dims();
        let half = features / 2;

        // Split into two halves
        let x1 = out.clone().slice([0..batch, 0..seq, 0..half]);
        let x2 = out.slice([0..batch, 0..seq, half..features]);

        // Gated activation: x1 * GELU(x2)
        x1 * gelu(x2)
    }
}

/// Gated transformer encoder layer with GEGLU feedforward.
#[derive(Module, Debug)]
struct GatedEncoderLayer<B: Backend> {
    attention: MultiHeadAttention<B>,
    norm1: LayerNorm<B>,
    geglu: GEGLU<B>,
    ff_out: Linear<B>,
    norm2: LayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> GatedEncoderLayer<B> {
    fn new(d_model: usize, n_heads: usize, ff_mult: usize, dropout: f64, device: &B::Device) -> Self {
        let attention = MultiHeadAttentionConfig::new(d_model, n_heads)
            .with_dropout(dropout)
            .init(device);
        let norm1 = LayerNormConfig::new(d_model).init(device);

        // GEGLU feedforward
        let d_ff = d_model * ff_mult;
        let geglu = GEGLU::new(d_model, d_ff, device);
        let ff_out = LinearConfig::new(d_ff, d_model).init(device);

        let norm2 = LayerNormConfig::new(d_model).init(device);
        let dropout_layer = DropoutConfig::new(dropout).init();

        Self {
            attention,
            norm1,
            geglu,
            ff_out,
            norm2,
            dropout: dropout_layer,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Self-attention with residual
        let attn_input = MhaInput::self_attn(x.clone());
        let attn_out = self.attention.forward(attn_input).context;
        let x = self.norm1.forward(x + self.dropout.forward(attn_out));

        // Gated feedforward with residual
        let ff_out = self.geglu.forward(x.clone());
        let ff_out = self.ff_out.forward(ff_out);

        self.norm2.forward(x + self.dropout.forward(ff_out))
    }
}

/// GatedTabTransformer for tabular data classification.
///
/// Uses GEGLU (Gated GELU Linear Unit) activations in the feedforward
/// layers of the transformer for enhanced expressiveness.
///
/// # Architecture
///
/// ```text
/// Continuous (B, N_cont)     Categorical (B, N_cat)
///       |                          |
///       +---> [Linear]             +---> [Embeddings]
///       |         |                       |
///       |         v                       v
///       |     (B, 1, D)            (B, N_cat, D)
///       |         |                       |
///       |         +--------+------+-------+
///       |                  |
///       |                  v
///       |          [Gated Transformer Layers]
///       |                  |
///       |                  v
///       |             [Flatten]
///       |                  |
///       +----------------> +
///                          |
///                          v
///                     [Linear]
///                          |
///                    Output (B, n_classes)
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use tsai_models::tabular::{GatedTabTransformer, GatedTabTransformerConfig};
///
/// let config = GatedTabTransformerConfig::new(10, 5, 3)
///     .with_cardinalities(vec![10, 20, 30, 40, 50])
///     .with_d_model(128)
///     .with_n_layers(4);
/// let model = config.init::<NdArray>(&device);
/// ```
#[derive(Module, Debug)]
pub struct GatedTabTransformer<B: Backend> {
    /// Embeddings for categorical features.
    cat_embeddings: Vec<Embedding<B>>,
    /// Projection for continuous features.
    cont_proj: Linear<B>,
    /// Gated transformer encoder layers.
    encoder_layers: Vec<GatedEncoderLayer<B>>,
    /// Final normalization.
    final_norm: LayerNorm<B>,
    /// Continuous feature MLP (when not using attn_on_continuous).
    cont_mlp: Option<Linear<B>>,
    /// Final classifier.
    head: Linear<B>,
    /// Model dimension.
    #[module(skip)]
    d_model: usize,
    /// Number of categorical features.
    #[module(skip)]
    n_categorical: usize,
    /// Whether to apply attention to continuous features.
    #[module(skip)]
    attn_on_continuous: bool,
}

impl<B: Backend> GatedTabTransformer<B> {
    /// Create a new GatedTabTransformer model.
    pub fn new(config: GatedTabTransformerConfig, device: &B::Device) -> Self {
        // Create embeddings for each categorical feature
        let cat_embeddings: Vec<_> = config
            .cat_cardinalities
            .iter()
            .map(|&card| EmbeddingConfig::new(card, config.d_model).init(device))
            .collect();

        // Projection for continuous features
        let cont_proj = LinearConfig::new(config.n_continuous.max(1), config.d_model).init(device);

        // Encoder layers
        let encoder_layers: Vec<_> = (0..config.n_layers)
            .map(|_| {
                GatedEncoderLayer::new(
                    config.d_model,
                    config.n_heads,
                    config.ff_mult,
                    config.dropout,
                    device,
                )
            })
            .collect();

        let final_norm = LayerNormConfig::new(config.d_model).init(device);

        // Optional MLP for continuous features when not applying attention
        let cont_mlp = if !config.attn_on_continuous && config.n_continuous > 0 {
            Some(LinearConfig::new(config.d_model, config.d_model).init(device))
        } else {
            None
        };

        // Output head
        let n_tokens = if config.attn_on_continuous {
            config.n_categorical + 1 // +1 for continuous token
        } else {
            config.n_categorical
        };
        let head_input = if config.attn_on_continuous {
            config.d_model * n_tokens
        } else {
            config.d_model * n_tokens + config.d_model // transformer output + cont MLP output
        };
        let head = LinearConfig::new(head_input, config.n_classes).init(device);

        Self {
            cat_embeddings,
            cont_proj,
            encoder_layers,
            final_norm,
            cont_mlp,
            head,
            d_model: config.d_model,
            n_categorical: config.n_categorical,
            attn_on_continuous: config.attn_on_continuous,
        }
    }

    /// Forward pass.
    ///
    /// # Arguments
    ///
    /// * `x_continuous` - Continuous features (batch, n_continuous)
    /// * `x_categorical` - Categorical features (batch, n_categorical) as indices
    pub fn forward(
        &self,
        x_continuous: Tensor<B, 2>,
        x_categorical: Tensor<B, 2, Int>,
    ) -> Tensor<B, 2> {
        let [batch, _] = x_continuous.dims();

        // Project continuous features
        let cont_embedded = self.cont_proj.forward(x_continuous);
        let cont_token = cont_embedded.clone().reshape([batch, 1, self.d_model]);

        // Embed categorical features
        let mut cat_tokens = Vec::new();
        for (i, embedding) in self.cat_embeddings.iter().enumerate() {
            if i < self.n_categorical {
                let cat_col = x_categorical.clone().slice([0..batch, i..(i + 1)]);
                let embedded = embedding.forward(cat_col); // (batch, 1, d_model)
                cat_tokens.push(embedded);
            }
        }

        // Combine tokens for transformer
        let transformer_input = if self.attn_on_continuous {
            let mut all_tokens = vec![cont_token];
            all_tokens.extend(cat_tokens);
            Tensor::cat(all_tokens, 1) // (batch, n_cat + 1, d_model)
        } else {
            Tensor::cat(cat_tokens, 1) // (batch, n_cat, d_model)
        };

        // Apply gated transformer layers
        let mut x = transformer_input;
        for layer in &self.encoder_layers {
            x = layer.forward(x);
        }

        // Final normalization
        let x = self.final_norm.forward(x);

        // Flatten transformer output
        let [_, n_tokens, d_model] = x.dims();
        let transformer_out = x.reshape([batch, n_tokens * d_model]);

        // Combine with continuous MLP output if not using attn_on_continuous
        let final_features = if let Some(ref cont_mlp) = self.cont_mlp {
            let cont_out = cont_mlp.forward(cont_embedded);
            let cont_out = gelu(cont_out);
            Tensor::cat(vec![transformer_out, cont_out], 1)
        } else {
            transformer_out
        };

        self.head.forward(final_features)
    }

    /// Forward pass returning probabilities.
    pub fn forward_probs(
        &self,
        x_continuous: Tensor<B, 2>,
        x_categorical: Tensor<B, 2, Int>,
    ) -> Tensor<B, 2> {
        let logits = self.forward(x_continuous, x_categorical);
        softmax(logits, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gated_tab_transformer_config_default() {
        let config = GatedTabTransformerConfig::default();
        assert_eq!(config.n_continuous, 10);
        assert_eq!(config.n_categorical, 5);
        assert_eq!(config.ff_mult, 4);
        assert!(config.attn_on_continuous);
    }

    #[test]
    fn test_gated_tab_transformer_config_new() {
        let config = GatedTabTransformerConfig::new(20, 8, 10);
        assert_eq!(config.n_continuous, 20);
        assert_eq!(config.n_categorical, 8);
        assert_eq!(config.n_classes, 10);
    }

    #[test]
    fn test_gated_tab_transformer_config_builder() {
        let config = GatedTabTransformerConfig::new(10, 5, 3)
            .with_d_model(128)
            .with_n_heads(8)
            .with_n_layers(4)
            .with_ff_mult(6)
            .with_dropout(0.2)
            .with_attn_on_continuous(false);

        assert_eq!(config.d_model, 128);
        assert_eq!(config.n_heads, 8);
        assert_eq!(config.n_layers, 4);
        assert_eq!(config.ff_mult, 6);
        assert_eq!(config.dropout, 0.2);
        assert!(!config.attn_on_continuous);
    }
}
