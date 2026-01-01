//! ConvTranPlus: Convolution + Transformer hybrid model.
//!
//! Combines the local feature extraction of convolutions with
//! the global attention mechanism of transformers.
//!
//! Reference: "ConvTran: Improving Position Encoding of Transformers for
//! Multivariate Time Series Classification"

use burn::nn::{
    attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
    conv::{Conv1d, Conv1dConfig},
    pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig},
    BatchNorm, BatchNormConfig, Dropout, DropoutConfig, LayerNorm, LayerNormConfig,
    Linear, LinearConfig, Relu,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use serde::{Deserialize, Serialize};

/// Configuration for ConvTranPlus model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvTranPlusConfig {
    /// Number of input variables/channels.
    pub n_vars: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// Conv filter sizes for each conv layer.
    pub conv_filters: Vec<usize>,
    /// Conv kernel sizes.
    pub conv_kernels: Vec<usize>,
    /// Model dimension for transformer.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Feedforward dimension.
    pub d_ff: usize,
    /// Dropout rate.
    pub dropout: f64,
    /// Whether to use positional encoding.
    pub use_pe: bool,
}

impl Default for ConvTranPlusConfig {
    fn default() -> Self {
        Self {
            n_vars: 1,
            seq_len: 100,
            n_classes: 2,
            conv_filters: vec![64, 128],
            conv_kernels: vec![7, 5],
            d_model: 128,
            n_heads: 8,
            n_layers: 2,
            d_ff: 256,
            dropout: 0.1,
            use_pe: true,
        }
    }
}

impl ConvTranPlusConfig {
    /// Create a new config.
    pub fn new(n_vars: usize, seq_len: usize, n_classes: usize) -> Self {
        Self {
            n_vars,
            seq_len,
            n_classes,
            ..Default::default()
        }
    }

    /// Set conv filter sizes.
    #[must_use]
    pub fn with_conv_filters(mut self, filters: Vec<usize>) -> Self {
        self.conv_filters = filters;
        self
    }

    /// Set conv kernel sizes.
    #[must_use]
    pub fn with_conv_kernels(mut self, kernels: Vec<usize>) -> Self {
        self.conv_kernels = kernels;
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

    /// Set number of transformer layers.
    #[must_use]
    pub fn with_n_layers(mut self, n_layers: usize) -> Self {
        self.n_layers = n_layers;
        self
    }

    /// Set feedforward dimension.
    #[must_use]
    pub fn with_d_ff(mut self, d_ff: usize) -> Self {
        self.d_ff = d_ff;
        self
    }

    /// Set dropout rate.
    #[must_use]
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Set whether to use positional encoding.
    #[must_use]
    pub fn with_pe(mut self, use_pe: bool) -> Self {
        self.use_pe = use_pe;
        self
    }

    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvTranPlus<B> {
        ConvTranPlus::new(self.clone(), device)
    }
}

/// Convolutional block with conv, batch norm, and ReLU.
#[derive(Module, Debug)]
struct ConvBlock<B: Backend> {
    conv: Conv1d<B>,
    bn: BatchNorm<B, 1>,
}

impl<B: Backend> ConvBlock<B> {
    fn new(in_channels: usize, out_channels: usize, kernel_size: usize, device: &B::Device) -> Self {
        let conv = Conv1dConfig::new(in_channels, out_channels, kernel_size)
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

/// Transformer encoder layer.
#[derive(Module, Debug)]
struct TransformerLayer<B: Backend> {
    attention: MultiHeadAttention<B>,
    norm1: LayerNorm<B>,
    ff_linear1: Linear<B>,
    ff_linear2: Linear<B>,
    norm2: LayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> TransformerLayer<B> {
    fn new(d_model: usize, n_heads: usize, d_ff: usize, dropout: f64, device: &B::Device) -> Self {
        let attention = MultiHeadAttentionConfig::new(d_model, n_heads)
            .with_dropout(dropout)
            .init(device);
        let norm1 = LayerNormConfig::new(d_model).init(device);
        let ff_linear1 = LinearConfig::new(d_model, d_ff).init(device);
        let ff_linear2 = LinearConfig::new(d_ff, d_model).init(device);
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

/// ConvTranPlus: Convolution + Transformer hybrid.
///
/// This model first extracts local features using convolutional layers,
/// then captures global dependencies with transformer attention.
///
/// # Architecture
///
/// ```text
/// Input (B, V, L)
///       |
///       +---> [Conv1->BN->ReLU] -> [Conv2->BN->ReLU] -> ... -> (B, C, L)
///       |
///       +---> [Permute to (B, L, C)] -> [Proj to d_model] -> (B, L, D)
///       |
///       +---> [+ Positional Encoding] (optional)
///       |
///       +---> [Transformer Layer 1] -> [Transformer Layer 2] -> ... -> (B, L, D)
///       |
///       +---> [Global Average Pool] -> (B, D)
///       |
///       +---> [Linear] -> Output (B, n_classes)
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use tsai_models::hybrid::{ConvTranPlus, ConvTranPlusConfig};
///
/// let config = ConvTranPlusConfig::new(3, 100, 5)
///     .with_conv_filters(vec![64, 128])
///     .with_n_layers(3);
/// let model = config.init::<NdArray>(&device);
///
/// let x = Tensor::random([32, 3, 100], Distribution::Normal(0.0, 1.0), &device);
/// let output = model.forward(x);
/// // output shape: [32, 5]
/// ```
#[derive(Module, Debug)]
pub struct ConvTranPlus<B: Backend> {
    /// Convolutional layers.
    conv_blocks: Vec<ConvBlock<B>>,
    /// Projection to d_model.
    proj: Linear<B>,
    /// Transformer layers.
    transformer_layers: Vec<TransformerLayer<B>>,
    /// Global average pooling.
    gap: AdaptiveAvgPool1d,
    /// Classification head.
    head: Linear<B>,
    /// Dropout.
    dropout: Dropout,
    /// Model dimension.
    #[module(skip)]
    d_model: usize,
    /// Whether to use positional encoding.
    #[module(skip)]
    use_pe: bool,
    /// Sequence length after convolutions (for PE).
    #[module(skip)]
    seq_len: usize,
}

impl<B: Backend> ConvTranPlus<B> {
    /// Create a new ConvTranPlus model.
    pub fn new(config: ConvTranPlusConfig, device: &B::Device) -> Self {
        // Build conv blocks
        let mut conv_blocks = Vec::new();
        let mut in_channels = config.n_vars;

        for (&filters, &kernel) in config.conv_filters.iter().zip(&config.conv_kernels) {
            conv_blocks.push(ConvBlock::new(in_channels, filters, kernel, device));
            in_channels = filters;
        }

        // Projection from conv output channels to d_model
        let conv_out_channels = *config.conv_filters.last().unwrap_or(&config.n_vars);
        let proj = LinearConfig::new(conv_out_channels, config.d_model).init(device);

        // Transformer layers
        let transformer_layers: Vec<_> = (0..config.n_layers)
            .map(|_| {
                TransformerLayer::new(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    config.dropout,
                    device,
                )
            })
            .collect();

        let gap = AdaptiveAvgPool1dConfig::new(1).init();
        let head = LinearConfig::new(config.d_model, config.n_classes).init(device);
        let dropout = DropoutConfig::new(config.dropout).init();

        Self {
            conv_blocks,
            proj,
            transformer_layers,
            gap,
            head,
            dropout,
            d_model: config.d_model,
            use_pe: config.use_pe,
            seq_len: config.seq_len,
        }
    }

    /// Generate sinusoidal positional encoding.
    fn positional_encoding<B2: Backend>(&self, seq_len: usize, device: &B2::Device) -> Tensor<B2, 2> {
        let d_model = self.d_model;
        let mut pe = vec![0.0f32; seq_len * d_model];

        for pos in 0..seq_len {
            for i in 0..d_model / 2 {
                let angle = pos as f32 / (10000.0f32).powf((2 * i) as f32 / d_model as f32);
                pe[pos * d_model + 2 * i] = angle.sin();
                pe[pos * d_model + 2 * i + 1] = angle.cos();
            }
        }

        Tensor::<B2, 1>::from_floats(pe.as_slice(), device).reshape([seq_len, d_model])
    }

    /// Forward pass.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, _n_vars, seq_len] = x.dims();
        let device = x.device();

        // Apply conv blocks: (B, V, L) -> (B, C, L)
        let mut out = x;
        for block in &self.conv_blocks {
            out = block.forward(out);
        }

        // Transpose for transformer: (B, C, L) -> (B, L, C)
        let out = out.swap_dims(1, 2);

        // Project to d_model: (B, L, C) -> (B, L, D)
        let out = self.proj.forward(out);

        // Add positional encoding
        let out = if self.use_pe {
            let pe = self.positional_encoding::<B>(seq_len, &device);
            // Broadcast PE to batch: (L, D) -> (1, L, D) -> (B, L, D)
            let pe = pe.unsqueeze::<3>();
            out + pe
        } else {
            out
        };

        let out = self.dropout.forward(out);

        // Apply transformer layers
        let mut out = out;
        for layer in &self.transformer_layers {
            out = layer.forward(out);
        }

        // Transpose back for GAP: (B, L, D) -> (B, D, L)
        let out = out.swap_dims(1, 2);

        // Global average pooling: (B, D, L) -> (B, D, 1) -> (B, D)
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
    fn test_convtranplus_config_default() {
        let config = ConvTranPlusConfig::default();
        assert_eq!(config.conv_filters, vec![64, 128]);
        assert_eq!(config.d_model, 128);
        assert_eq!(config.n_layers, 2);
        assert!(config.use_pe);
    }

    #[test]
    fn test_convtranplus_config_new() {
        let config = ConvTranPlusConfig::new(3, 200, 10);
        assert_eq!(config.n_vars, 3);
        assert_eq!(config.seq_len, 200);
        assert_eq!(config.n_classes, 10);
    }

    #[test]
    fn test_convtranplus_config_builder() {
        let config = ConvTranPlusConfig::new(3, 100, 5)
            .with_conv_filters(vec![32, 64, 128])
            .with_d_model(256)
            .with_n_heads(4)
            .with_n_layers(4)
            .with_dropout(0.2);

        assert_eq!(config.conv_filters, vec![32, 64, 128]);
        assert_eq!(config.d_model, 256);
        assert_eq!(config.n_heads, 4);
        assert_eq!(config.n_layers, 4);
        assert_eq!(config.dropout, 0.2);
    }
}
