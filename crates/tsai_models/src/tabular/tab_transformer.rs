//! TabTransformer model for tabular + time series data.

use burn::nn::{
    attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
    Embedding, EmbeddingConfig, Linear, LinearConfig, LayerNorm, LayerNormConfig, Relu,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use serde::{Deserialize, Serialize};

/// Configuration for TabTransformer model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TabTransformerConfig {
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
    /// Dropout rate.
    pub dropout: f64,
}

impl Default for TabTransformerConfig {
    fn default() -> Self {
        Self {
            n_continuous: 10,
            n_categorical: 5,
            cat_cardinalities: vec![10, 20, 30, 40, 50],
            n_classes: 2,
            d_model: 64,
            n_heads: 4,
            n_layers: 2,
            dropout: 0.1,
        }
    }
}

impl TabTransformerConfig {
    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> TabTransformer<B> {
        TabTransformer::new(self.clone(), device)
    }
}

/// Transformer encoder layer for tabular data.
#[derive(Module, Debug)]
struct TabEncoderLayer<B: Backend> {
    attention: MultiHeadAttention<B>,
    norm1: LayerNorm<B>,
    ff_linear1: Linear<B>,
    ff_linear2: Linear<B>,
    norm2: LayerNorm<B>,
}

impl<B: Backend> TabEncoderLayer<B> {
    fn new(d_model: usize, n_heads: usize, dropout: f64, device: &B::Device) -> Self {
        let attention = MultiHeadAttentionConfig::new(d_model, n_heads)
            .with_dropout(dropout)
            .init(device);
        let norm1 = LayerNormConfig::new(d_model).init(device);
        let ff_linear1 = LinearConfig::new(d_model, d_model * 4).init(device);
        let ff_linear2 = LinearConfig::new(d_model * 4, d_model).init(device);
        let norm2 = LayerNormConfig::new(d_model).init(device);

        Self {
            attention,
            norm1,
            ff_linear1,
            ff_linear2,
            norm2,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Self-attention
        let attn_input = MhaInput::self_attn(x.clone());
        let attn_out = self.attention.forward(attn_input).context;
        let x = self.norm1.forward(x + attn_out);

        // Feedforward
        let ff_out = self.ff_linear1.forward(x.clone());
        let ff_out = Relu::new().forward(ff_out);
        let ff_out = self.ff_linear2.forward(ff_out);

        self.norm2.forward(x + ff_out)
    }
}

/// TabTransformer for tabular data classification.
#[derive(Module, Debug)]
pub struct TabTransformer<B: Backend> {
    /// Embeddings for categorical features.
    cat_embeddings: Vec<Embedding<B>>,
    /// Projection for continuous features.
    cont_proj: Linear<B>,
    /// Transformer encoder layers.
    encoder_layers: Vec<TabEncoderLayer<B>>,
    /// Final classifier.
    head: Linear<B>,
}

impl<B: Backend> TabTransformer<B> {
    /// Create a new TabTransformer model.
    pub fn new(config: TabTransformerConfig, device: &B::Device) -> Self {
        // Create embeddings for each categorical feature
        let cat_embeddings: Vec<_> = config
            .cat_cardinalities
            .iter()
            .map(|&card| EmbeddingConfig::new(card, config.d_model).init(device))
            .collect();

        // Projection for continuous features
        let cont_proj = LinearConfig::new(config.n_continuous, config.d_model).init(device);

        // Encoder layers
        let encoder_layers: Vec<_> = (0..config.n_layers)
            .map(|_| TabEncoderLayer::new(config.d_model, config.n_heads, config.dropout, device))
            .collect();

        // Output head
        let total_features = config.n_categorical + 1; // +1 for continuous
        let head = LinearConfig::new(config.d_model * total_features, config.n_classes).init(device);

        Self {
            cat_embeddings,
            cont_proj,
            encoder_layers,
            head,
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
        let cont_embedded = cont_embedded.unsqueeze_dim(1); // (batch, 1, d_model)

        // Embed categorical features
        let mut cat_embeddings = Vec::new();
        for (i, embedding) in self.cat_embeddings.iter().enumerate() {
            let cat_col = x_categorical.clone().slice([0..batch, i..(i + 1)]);
            let embedded = embedding.forward(cat_col); // (batch, 1, d_model)
            cat_embeddings.push(embedded);
        }

        // Concatenate all features: (batch, n_categorical + 1, d_model)
        let mut all_features = vec![cont_embedded];
        all_features.extend(cat_embeddings);
        let features = Tensor::cat(all_features, 1);

        // Apply transformer encoder to categorical embeddings
        let mut x = features;
        for layer in &self.encoder_layers {
            x = layer.forward(x);
        }

        // Flatten and classify
        let [_, n_feats, d_model] = x.dims();
        let x = x.reshape([batch, n_feats * d_model]);
        self.head.forward(x)
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
    fn test_tab_transformer_config() {
        let config = TabTransformerConfig::default();
        assert_eq!(config.n_continuous, 10);
        assert_eq!(config.n_categorical, 5);
    }
}
