//! TransformerRNNPlus: Transformer + RNN hybrid model.
//!
//! Combines transformer self-attention with RNN sequential modeling
//! for time series classification.
//!
//! The model first applies transformer attention to capture global
//! dependencies, then uses RNN layers to model sequential patterns.

use burn::nn::{
    attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
    gru::{Gru, GruConfig},
    lstm::{Lstm, LstmConfig},
    Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, Relu,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use serde::{Deserialize, Serialize};

/// RNN type for the hybrid model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TransformerRNNType {
    /// Use LSTM for the RNN component.
    #[default]
    LSTM,
    /// Use GRU for the RNN component.
    GRU,
}

/// Configuration for TransformerRNNPlus model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerRNNPlusConfig {
    /// Number of input variables/channels.
    pub n_vars: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// Model dimension for transformer.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of transformer layers.
    pub n_transformer_layers: usize,
    /// Feedforward dimension in transformer.
    pub d_ff: usize,
    /// RNN type (LSTM or GRU).
    pub rnn_type: TransformerRNNType,
    /// RNN hidden size.
    pub rnn_hidden_size: usize,
    /// Number of RNN layers.
    pub n_rnn_layers: usize,
    /// Dropout rate.
    pub dropout: f64,
    /// Whether to use bidirectional RNN.
    pub bidirectional: bool,
}

impl Default for TransformerRNNPlusConfig {
    fn default() -> Self {
        Self {
            n_vars: 1,
            seq_len: 100,
            n_classes: 2,
            d_model: 128,
            n_heads: 8,
            n_transformer_layers: 2,
            d_ff: 256,
            rnn_type: TransformerRNNType::LSTM,
            rnn_hidden_size: 128,
            n_rnn_layers: 1,
            dropout: 0.1,
            bidirectional: false,
        }
    }
}

impl TransformerRNNPlusConfig {
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

    /// Set number of attention heads.
    #[must_use]
    pub fn with_n_heads(mut self, n_heads: usize) -> Self {
        self.n_heads = n_heads;
        self
    }

    /// Set number of transformer layers.
    #[must_use]
    pub fn with_n_transformer_layers(mut self, n_layers: usize) -> Self {
        self.n_transformer_layers = n_layers;
        self
    }

    /// Set RNN type.
    #[must_use]
    pub fn with_rnn_type(mut self, rnn_type: TransformerRNNType) -> Self {
        self.rnn_type = rnn_type;
        self
    }

    /// Set RNN hidden size.
    #[must_use]
    pub fn with_rnn_hidden_size(mut self, hidden_size: usize) -> Self {
        self.rnn_hidden_size = hidden_size;
        self
    }

    /// Set number of RNN layers.
    #[must_use]
    pub fn with_n_rnn_layers(mut self, n_layers: usize) -> Self {
        self.n_rnn_layers = n_layers;
        self
    }

    /// Set dropout rate.
    #[must_use]
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Set bidirectional mode.
    #[must_use]
    pub fn with_bidirectional(mut self, bidirectional: bool) -> Self {
        self.bidirectional = bidirectional;
        self
    }

    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerRNNPlus<B> {
        TransformerRNNPlus::new(self.clone(), device)
    }
}

/// Transformer encoder layer.
#[derive(Module, Debug)]
struct TransformerEncoderLayer<B: Backend> {
    attention: MultiHeadAttention<B>,
    norm1: LayerNorm<B>,
    ff_linear1: Linear<B>,
    ff_linear2: Linear<B>,
    norm2: LayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> TransformerEncoderLayer<B> {
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
        // Self-attention with residual
        let attn_input = MhaInput::self_attn(x.clone());
        let attn_out = self.attention.forward(attn_input).context;
        let x = self.norm1.forward(x + self.dropout.forward(attn_out));

        // Feedforward with residual
        let ff_out = self.ff_linear1.forward(x.clone());
        let ff_out = Relu::new().forward(ff_out);
        let ff_out = self.dropout.forward(ff_out);
        let ff_out = self.ff_linear2.forward(ff_out);

        self.norm2.forward(x + self.dropout.forward(ff_out))
    }
}

/// TransformerRNNPlus: Transformer + RNN hybrid.
///
/// Combines the global attention of transformers with the sequential
/// modeling capabilities of RNNs.
///
/// # Architecture
///
/// ```text
/// Input (B, V, L)
///       |
///       +---> [Permute to (B, L, V)] -> [Proj to d_model]
///       |
///       +---> [+ Positional Encoding]
///       |
///       +---> [Transformer Layer 1] -> [Transformer Layer 2] -> ...
///       |
///       +---> [RNN (LSTM/GRU)] -> [Last hidden state]
///       |
///       +---> [Linear] -> Output (B, n_classes)
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use tsai_models::hybrid::{TransformerRNNPlus, TransformerRNNPlusConfig, TransformerRNNType};
///
/// let config = TransformerRNNPlusConfig::new(3, 100, 5)
///     .with_rnn_type(TransformerRNNType::LSTM)
///     .with_n_transformer_layers(2)
///     .with_n_rnn_layers(2);
/// let model = config.init::<NdArray>(&device);
///
/// let x = Tensor::random([32, 3, 100], Distribution::Normal(0.0, 1.0), &device);
/// let output = model.forward(x);
/// // output shape: [32, 5]
/// ```
#[derive(Module, Debug)]
pub struct TransformerRNNPlus<B: Backend> {
    /// Input projection.
    input_proj: Linear<B>,
    /// Transformer encoder layers.
    transformer_layers: Vec<TransformerEncoderLayer<B>>,
    /// LSTM layer (if using LSTM).
    lstm: Option<Lstm<B>>,
    /// GRU layer (if using GRU).
    gru: Option<Gru<B>>,
    /// Dropout.
    dropout: Dropout,
    /// Classification head.
    head: Linear<B>,
    /// Model dimension.
    #[module(skip)]
    d_model: usize,
    /// RNN hidden size.
    #[module(skip)]
    rnn_hidden_size: usize,
    /// Whether bidirectional.
    #[module(skip)]
    bidirectional: bool,
}

impl<B: Backend> TransformerRNNPlus<B> {
    /// Create a new TransformerRNNPlus model.
    pub fn new(config: TransformerRNNPlusConfig, device: &B::Device) -> Self {
        // Input projection
        let input_proj = LinearConfig::new(config.n_vars, config.d_model).init(device);

        // Transformer layers
        let transformer_layers: Vec<_> = (0..config.n_transformer_layers)
            .map(|_| {
                TransformerEncoderLayer::new(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    config.dropout,
                    device,
                )
            })
            .collect();

        // RNN layer
        let (lstm, gru) = match config.rnn_type {
            TransformerRNNType::LSTM => {
                let lstm = LstmConfig::new(config.d_model, config.rnn_hidden_size, config.bidirectional)
                    .init(device);
                (Some(lstm), None)
            }
            TransformerRNNType::GRU => {
                let gru = GruConfig::new(config.d_model, config.rnn_hidden_size, config.bidirectional)
                    .init(device);
                (None, Some(gru))
            }
        };

        let dropout = DropoutConfig::new(config.dropout).init();

        // Output size depends on bidirectional
        let rnn_out_size = if config.bidirectional {
            config.rnn_hidden_size * 2
        } else {
            config.rnn_hidden_size
        };
        let head = LinearConfig::new(rnn_out_size, config.n_classes).init(device);

        Self {
            input_proj,
            transformer_layers,
            lstm,
            gru,
            dropout,
            head,
            d_model: config.d_model,
            rnn_hidden_size: config.rnn_hidden_size,
            bidirectional: config.bidirectional,
        }
    }

    /// Generate sinusoidal positional encoding.
    fn positional_encoding(&self, seq_len: usize, device: &B::Device) -> Tensor<B, 2> {
        let d_model = self.d_model;
        let mut pe = vec![0.0f32; seq_len * d_model];

        for pos in 0..seq_len {
            for i in 0..d_model / 2 {
                let angle = pos as f32 / (10000.0f32).powf((2 * i) as f32 / d_model as f32);
                pe[pos * d_model + 2 * i] = angle.sin();
                pe[pos * d_model + 2 * i + 1] = angle.cos();
            }
        }

        Tensor::<B, 1>::from_floats(pe.as_slice(), device).reshape([seq_len, d_model])
    }

    /// Forward pass.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, _n_vars, seq_len] = x.dims();
        let device = x.device();

        // Transpose: (B, V, L) -> (B, L, V)
        let out = x.swap_dims(1, 2);

        // Project to d_model: (B, L, V) -> (B, L, D)
        let out = self.input_proj.forward(out);

        // Add positional encoding
        let pe = self.positional_encoding(seq_len, &device);
        let pe = pe.unsqueeze::<3>();
        let out = out + pe;

        let out = self.dropout.forward(out);

        // Apply transformer layers
        let mut out = out;
        for layer in &self.transformer_layers {
            out = layer.forward(out);
        }

        // Apply RNN and get last hidden state
        let rnn_out = if let Some(ref lstm) = self.lstm {
            let (output, _) = lstm.forward(out, None);
            // Take last time step
            let last_idx = seq_len - 1;
            let hidden_size = output.dims()[2];
            output
                .slice([0..batch_size, last_idx..last_idx + 1, 0..hidden_size])
                .reshape([batch_size, hidden_size])
        } else if let Some(ref gru) = self.gru {
            let output = gru.forward(out, None);
            let last_idx = seq_len - 1;
            let hidden_size = output.dims()[2];
            output
                .slice([0..batch_size, last_idx..last_idx + 1, 0..hidden_size])
                .reshape([batch_size, hidden_size])
        } else {
            panic!("No RNN layer configured");
        };

        let rnn_out = self.dropout.forward(rnn_out);

        // Classification
        self.head.forward(rnn_out)
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
    fn test_transformer_rnn_config_default() {
        let config = TransformerRNNPlusConfig::default();
        assert_eq!(config.d_model, 128);
        assert_eq!(config.n_transformer_layers, 2);
        assert_eq!(config.rnn_type, TransformerRNNType::LSTM);
    }

    #[test]
    fn test_transformer_rnn_config_new() {
        let config = TransformerRNNPlusConfig::new(3, 200, 10);
        assert_eq!(config.n_vars, 3);
        assert_eq!(config.seq_len, 200);
        assert_eq!(config.n_classes, 10);
    }

    #[test]
    fn test_transformer_rnn_config_builder() {
        let config = TransformerRNNPlusConfig::new(3, 100, 5)
            .with_rnn_type(TransformerRNNType::GRU)
            .with_n_transformer_layers(4)
            .with_n_rnn_layers(2)
            .with_bidirectional(true);

        assert_eq!(config.rnn_type, TransformerRNNType::GRU);
        assert_eq!(config.n_transformer_layers, 4);
        assert_eq!(config.n_rnn_layers, 2);
        assert!(config.bidirectional);
    }
}
