//! TabFusionTransformer: Fusion of time series and tabular data.
//!
//! Combines time series features with tabular (categorical + continuous)
//! features using transformer attention for multi-modal learning.

use burn::nn::{
    attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
    conv::{Conv1d, Conv1dConfig},
    pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig},
    BatchNorm, BatchNormConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig,
    Linear, LinearConfig, Relu, Dropout, DropoutConfig,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use serde::{Deserialize, Serialize};

/// Configuration for TabFusionTransformer model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TabFusionTransformerConfig {
    /// Number of time series input variables.
    pub n_ts_vars: usize,
    /// Time series sequence length.
    pub ts_seq_len: usize,
    /// Number of continuous tabular features.
    pub n_continuous: usize,
    /// Number of categorical tabular features.
    pub n_categorical: usize,
    /// Cardinalities for each categorical feature.
    pub cat_cardinalities: Vec<usize>,
    /// Number of output classes.
    pub n_classes: usize,
    /// Model dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of fusion transformer layers.
    pub n_layers: usize,
    /// Dropout rate.
    pub dropout: f64,
    /// Time series conv filters.
    pub ts_filters: Vec<usize>,
}

impl Default for TabFusionTransformerConfig {
    fn default() -> Self {
        Self {
            n_ts_vars: 1,
            ts_seq_len: 100,
            n_continuous: 10,
            n_categorical: 5,
            cat_cardinalities: vec![10, 20, 30, 40, 50],
            n_classes: 2,
            d_model: 128,
            n_heads: 8,
            n_layers: 2,
            dropout: 0.1,
            ts_filters: vec![64, 128],
        }
    }
}

impl TabFusionTransformerConfig {
    /// Create a new config.
    pub fn new(n_ts_vars: usize, ts_seq_len: usize, n_classes: usize) -> Self {
        Self {
            n_ts_vars,
            ts_seq_len,
            n_classes,
            ..Default::default()
        }
    }

    /// Set number of continuous features.
    #[must_use]
    pub fn with_n_continuous(mut self, n: usize) -> Self {
        self.n_continuous = n;
        self
    }

    /// Set categorical features.
    #[must_use]
    pub fn with_categorical(mut self, n_categorical: usize, cardinalities: Vec<usize>) -> Self {
        self.n_categorical = n_categorical;
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

    /// Set dropout rate.
    #[must_use]
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Set time series conv filters.
    #[must_use]
    pub fn with_ts_filters(mut self, filters: Vec<usize>) -> Self {
        self.ts_filters = filters;
        self
    }

    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> TabFusionTransformer<B> {
        TabFusionTransformer::new(self.clone(), device)
    }
}

/// Conv block for time series feature extraction.
#[derive(Module, Debug)]
struct TSConvBlock<B: Backend> {
    conv: Conv1d<B>,
    bn: BatchNorm<B, 1>,
}

impl<B: Backend> TSConvBlock<B> {
    fn new(in_channels: usize, out_channels: usize, device: &B::Device) -> Self {
        let conv = Conv1dConfig::new(in_channels, out_channels, 3)
            .with_padding(burn::nn::PaddingConfig1d::Same)
            .init(device);
        let bn = BatchNormConfig::new(out_channels).init(device);
        Self { conv, bn }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let out = self.conv.forward(x);
        let out = self.bn.forward(out);
        Relu::new().forward(out)
    }
}

/// Fusion transformer layer.
#[derive(Module, Debug)]
struct FusionLayer<B: Backend> {
    attention: MultiHeadAttention<B>,
    norm1: LayerNorm<B>,
    ff_linear1: Linear<B>,
    ff_linear2: Linear<B>,
    norm2: LayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> FusionLayer<B> {
    fn new(d_model: usize, n_heads: usize, dropout: f64, device: &B::Device) -> Self {
        let attention = MultiHeadAttentionConfig::new(d_model, n_heads)
            .with_dropout(dropout)
            .init(device);
        let norm1 = LayerNormConfig::new(d_model).init(device);
        let ff_linear1 = LinearConfig::new(d_model, d_model * 4).init(device);
        let ff_linear2 = LinearConfig::new(d_model * 4, d_model).init(device);
        let norm2 = LayerNormConfig::new(d_model).init(device);
        let dropout_layer = DropoutConfig::new(dropout).init();

        Self {
            attention,
            norm1,
            ff_linear1,
            ff_linear2,
            norm2,
            dropout: dropout_layer,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Self-attention
        let attn_input = MhaInput::self_attn(x.clone());
        let attn_out = self.attention.forward(attn_input).context;
        let x = self.norm1.forward(x + self.dropout.forward(attn_out));

        // Feedforward
        let ff_out = self.ff_linear1.forward(x.clone());
        let ff_out = Relu::new().forward(ff_out);
        let ff_out = self.dropout.forward(ff_out);
        let ff_out = self.ff_linear2.forward(ff_out);

        self.norm2.forward(x + self.dropout.forward(ff_out))
    }
}

/// TabFusionTransformer for multi-modal time series + tabular classification.
///
/// Fuses time series features with tabular (categorical + continuous)
/// features using transformer cross-attention.
///
/// # Architecture
///
/// ```text
/// Time Series Input (B, V, L)     Tabular Input (B, N_tab)
///       |                               |
///       +---> [Conv Blocks] -> GAP      +---> [Embeddings] (categorical)
///       |          |                    |
///       |          v                    +---> [Linear] (continuous)
///       |      (B, D_ts)                        |
///       |          |                            v
///       |          +-------> Concatenate <------+
///       |                         |
///       |                         v
///       |                 [Fusion Transformer Layers]
///       |                         |
///       |                         v
///       |                    [Linear]
///       |                         |
///       +------------------> Output (B, n_classes)
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use tsai_models::tabular::{TabFusionTransformer, TabFusionTransformerConfig};
///
/// let config = TabFusionTransformerConfig::new(3, 100, 5)
///     .with_n_continuous(10)
///     .with_categorical(5, vec![10, 20, 30, 40, 50]);
/// let model = config.init::<NdArray>(&device);
/// ```
#[derive(Module, Debug)]
pub struct TabFusionTransformer<B: Backend> {
    /// Time series conv blocks.
    ts_conv_blocks: Vec<TSConvBlock<B>>,
    /// Global average pooling for time series.
    ts_gap: AdaptiveAvgPool1d,
    /// Time series projection to d_model.
    ts_proj: Linear<B>,
    /// Categorical embeddings.
    cat_embeddings: Vec<Embedding<B>>,
    /// Continuous feature projection.
    cont_proj: Linear<B>,
    /// Fusion transformer layers.
    fusion_layers: Vec<FusionLayer<B>>,
    /// Final layer norm.
    final_norm: LayerNorm<B>,
    /// Classification head.
    head: Linear<B>,
    /// Dropout.
    dropout: Dropout,
    /// Model dimension.
    #[module(skip)]
    d_model: usize,
    /// Number of categorical features.
    #[module(skip)]
    n_categorical: usize,
    /// Number of continuous features.
    #[module(skip)]
    n_continuous: usize,
}

impl<B: Backend> TabFusionTransformer<B> {
    /// Create a new TabFusionTransformer model.
    pub fn new(config: TabFusionTransformerConfig, device: &B::Device) -> Self {
        // Time series conv blocks
        let mut ts_conv_blocks = Vec::new();
        let mut in_channels = config.n_ts_vars;
        for &filters in &config.ts_filters {
            ts_conv_blocks.push(TSConvBlock::new(in_channels, filters, device));
            in_channels = filters;
        }

        let ts_gap = AdaptiveAvgPool1dConfig::new(1).init();
        let ts_out_channels = *config.ts_filters.last().unwrap_or(&config.n_ts_vars);
        let ts_proj = LinearConfig::new(ts_out_channels, config.d_model).init(device);

        // Categorical embeddings
        let cat_embeddings: Vec<_> = config
            .cat_cardinalities
            .iter()
            .map(|&card| EmbeddingConfig::new(card, config.d_model).init(device))
            .collect();

        // Continuous projection
        let cont_proj = LinearConfig::new(config.n_continuous.max(1), config.d_model).init(device);

        // Fusion transformer layers
        let fusion_layers: Vec<_> = (0..config.n_layers)
            .map(|_| FusionLayer::new(config.d_model, config.n_heads, config.dropout, device))
            .collect();

        let final_norm = LayerNormConfig::new(config.d_model).init(device);

        // Total number of tokens: 1 (ts) + n_categorical + 1 (continuous)
        let n_tokens = 1 + config.n_categorical + (if config.n_continuous > 0 { 1 } else { 0 });
        let head = LinearConfig::new(config.d_model * n_tokens, config.n_classes).init(device);

        let dropout = DropoutConfig::new(config.dropout).init();

        Self {
            ts_conv_blocks,
            ts_gap,
            ts_proj,
            cat_embeddings,
            cont_proj,
            fusion_layers,
            final_norm,
            head,
            dropout,
            d_model: config.d_model,
            n_categorical: config.n_categorical,
            n_continuous: config.n_continuous,
        }
    }

    /// Forward pass.
    ///
    /// # Arguments
    ///
    /// * `ts` - Time series input (B, V, L)
    /// * `continuous` - Continuous features (B, N_cont)
    /// * `categorical` - Categorical features (B, N_cat) as integer indices
    pub fn forward(
        &self,
        ts: Tensor<B, 3>,
        continuous: Option<Tensor<B, 2>>,
        categorical: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 2> {
        let [batch_size, _, _] = ts.dims();
        let _device = ts.device();

        // Process time series
        let mut ts_out = ts;
        for block in &self.ts_conv_blocks {
            ts_out = block.forward(ts_out);
        }
        let ts_out = self.ts_gap.forward(ts_out);
        let ts_channels = ts_out.dims()[1];
        let ts_out = ts_out.reshape([batch_size, ts_channels]);
        let ts_out = self.ts_proj.forward(ts_out);
        let ts_token = ts_out.reshape([batch_size, 1, self.d_model]);

        // Collect all tokens
        let mut tokens = vec![ts_token];

        // Process categorical features
        if let Some(cat) = categorical {
            for (i, emb) in self.cat_embeddings.iter().enumerate() {
                if i < self.n_categorical {
                    // Extract the i-th categorical feature as 2D tensor (B, 1)
                    let cat_i = cat.clone().slice([0..batch_size, i..i + 1]);
                    // Embedding forward expects (B, seq) and returns (B, seq, d_model)
                    let cat_emb = emb.forward(cat_i);
                    // Result is (B, 1, d_model), which is already what we need
                    tokens.push(cat_emb);
                }
            }
        }

        // Process continuous features
        if let Some(cont) = continuous {
            let cont_emb = self.cont_proj.forward(cont);
            let cont_token = cont_emb.reshape([batch_size, 1, self.d_model]);
            tokens.push(cont_token);
        }

        // Concatenate all tokens: (B, N_tokens, D)
        let combined = Tensor::cat(tokens, 1);
        let combined = self.dropout.forward(combined);

        // Apply fusion transformer layers
        let mut out = combined;
        for layer in &self.fusion_layers {
            out = layer.forward(out);
        }

        // Final normalization
        let out = self.final_norm.forward(out);

        // Flatten and classify
        let n_tokens = out.dims()[1];
        let out = out.reshape([batch_size, n_tokens * self.d_model]);
        self.head.forward(out)
    }

    /// Forward pass returning probabilities.
    pub fn forward_probs(
        &self,
        ts: Tensor<B, 3>,
        continuous: Option<Tensor<B, 2>>,
        categorical: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 2> {
        let logits = self.forward(ts, continuous, categorical);
        softmax(logits, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tab_fusion_config_default() {
        let config = TabFusionTransformerConfig::default();
        assert_eq!(config.n_ts_vars, 1);
        assert_eq!(config.n_continuous, 10);
        assert_eq!(config.n_categorical, 5);
        assert_eq!(config.d_model, 128);
    }

    #[test]
    fn test_tab_fusion_config_new() {
        let config = TabFusionTransformerConfig::new(3, 200, 10);
        assert_eq!(config.n_ts_vars, 3);
        assert_eq!(config.ts_seq_len, 200);
        assert_eq!(config.n_classes, 10);
    }

    #[test]
    fn test_tab_fusion_config_builder() {
        let config = TabFusionTransformerConfig::new(3, 100, 5)
            .with_n_continuous(20)
            .with_categorical(3, vec![5, 10, 15])
            .with_d_model(256)
            .with_n_layers(4);

        assert_eq!(config.n_continuous, 20);
        assert_eq!(config.n_categorical, 3);
        assert_eq!(config.cat_cardinalities, vec![5, 10, 15]);
        assert_eq!(config.d_model, 256);
        assert_eq!(config.n_layers, 4);
    }
}
