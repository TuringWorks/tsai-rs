//! Time Series Transformer (TSTPlus) model.

use burn::nn::{
    attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
    Dropout, DropoutConfig, Linear, LinearConfig, LayerNorm, LayerNormConfig, Relu,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use serde::{Deserialize, Serialize};

/// Configuration for TSTPlus model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TSTConfig {
    /// Number of input variables/channels.
    pub n_vars: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// Model dimension.
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

impl Default for TSTConfig {
    fn default() -> Self {
        Self {
            n_vars: 1,
            seq_len: 100,
            n_classes: 2,
            d_model: 128,
            n_heads: 8,
            n_layers: 3,
            d_ff: 256,
            dropout: 0.1,
            use_pe: true,
        }
    }
}

impl TSTConfig {
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
    pub fn init<B: Backend>(&self, device: &B::Device) -> TSTPlus<B> {
        TSTPlus::new(self.clone(), device)
    }
}

/// Transformer encoder layer.
#[derive(Module, Debug)]
struct TSTEncoderLayer<B: Backend> {
    attention: MultiHeadAttention<B>,
    norm1: LayerNorm<B>,
    ff_linear1: Linear<B>,
    ff_linear2: Linear<B>,
    norm2: LayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> TSTEncoderLayer<B> {
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

/// Time Series Transformer (TSTPlus) for classification.
#[derive(Module, Debug)]
pub struct TSTPlus<B: Backend> {
    /// Input projection.
    input_proj: Linear<B>,
    /// Transformer encoder layers.
    encoder_layers: Vec<TSTEncoderLayer<B>>,
    /// Classification head.
    head: Linear<B>,
    /// Dropout.
    dropout: Dropout,
}

impl<B: Backend> TSTPlus<B> {
    /// Create a new TSTPlus model.
    pub fn new(config: TSTConfig, device: &B::Device) -> Self {
        // Input projection: n_vars -> d_model
        let input_proj = LinearConfig::new(config.n_vars, config.d_model).init(device);

        // Encoder layers
        let encoder_layers: Vec<_> = (0..config.n_layers)
            .map(|_| {
                TSTEncoderLayer::new(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    config.dropout,
                    device,
                )
            })
            .collect();

        // Classification head
        let head = LinearConfig::new(config.d_model, config.n_classes).init(device);

        let dropout = DropoutConfig::new(config.dropout).init();

        Self {
            input_proj,
            encoder_layers,
            head,
            dropout,
        }
    }

    /// Create sinusoidal positional encoding.
    fn create_positional_encoding(seq_len: usize, d_model: usize, device: &B::Device) -> Tensor<B, 2> {
        let mut pe = vec![0.0f32; seq_len * d_model];

        for pos in 0..seq_len {
            for i in 0..d_model {
                let angle = pos as f32 / (10000.0f32).powf((2 * (i / 2)) as f32 / d_model as f32);
                pe[pos * d_model + i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }

        Tensor::<B, 1>::from_floats(pe.as_slice(), device).reshape([seq_len, d_model])
    }

    /// Forward pass.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch, _n_vars, seq_len] = x.dims();
        let device = x.device();

        // Transpose to (batch, seq_len, n_vars)
        let x = x.swap_dims(1, 2);

        // Project to d_model
        let x = self.input_proj.forward(x);
        let [_, _, d_model] = x.dims();

        // Add positional encoding
        let pos_encoding = Self::create_positional_encoding(seq_len, d_model, &device);
        let x = x + pos_encoding.unsqueeze::<3>();

        // Apply dropout
        let mut x = self.dropout.forward(x);

        // Transformer encoder
        for layer in &self.encoder_layers {
            x = layer.forward(x);
        }

        // Global average pooling over sequence dimension
        let x = x.mean_dim(1);

        // Reshape and classify
        let x = x.reshape([batch, d_model]);
        self.head.forward(x)
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
    fn test_tst_config() {
        let config = TSTConfig::default();
        assert_eq!(config.d_model, 128);
        assert_eq!(config.n_heads, 8);
    }
}
