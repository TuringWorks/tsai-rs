//! RNN with Attention for time series classification.
//!
//! Combines LSTM/GRU with attention mechanism to learn which timesteps
//! are most relevant for classification.
//!
//! Reference: "Attention-Based Recurrent Neural Network Models for
//! Joint Intent Detection and Slot Filling" (Liu & Lane, 2016)

use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig, Lstm, LstmConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use serde::{Deserialize, Serialize};

use super::RNNType;

/// Type of attention mechanism.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionType {
    /// Additive attention (Bahdanau-style).
    Additive,
    /// Dot-product attention (Luong-style).
    DotProduct,
    /// Scaled dot-product attention.
    ScaledDotProduct,
}

impl Default for AttentionType {
    fn default() -> Self {
        Self::ScaledDotProduct
    }
}

/// Configuration for RNN with Attention model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RNNAttentionConfig {
    /// Number of input variables/channels.
    pub n_vars: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Number of RNN layers.
    pub n_layers: usize,
    /// Type of RNN (LSTM or GRU).
    pub rnn_type: RNNType,
    /// Whether to use bidirectional RNN.
    pub bidirectional: bool,
    /// Type of attention mechanism.
    pub attention_type: AttentionType,
    /// Attention dimension for additive attention.
    pub attention_dim: usize,
    /// Dropout rate.
    pub dropout: f64,
}

impl Default for RNNAttentionConfig {
    fn default() -> Self {
        Self {
            n_vars: 1,
            seq_len: 100,
            n_classes: 2,
            hidden_size: 128,
            n_layers: 1,
            rnn_type: RNNType::LSTM,
            bidirectional: true,
            attention_type: AttentionType::ScaledDotProduct,
            attention_dim: 64,
            dropout: 0.1,
        }
    }
}

impl RNNAttentionConfig {
    /// Create a new config.
    pub fn new(n_vars: usize, seq_len: usize, n_classes: usize) -> Self {
        Self {
            n_vars,
            seq_len,
            n_classes,
            ..Default::default()
        }
    }

    /// Set hidden size.
    #[must_use]
    pub fn with_hidden_size(mut self, hidden_size: usize) -> Self {
        self.hidden_size = hidden_size;
        self
    }

    /// Set bidirectional.
    #[must_use]
    pub fn with_bidirectional(mut self, bidirectional: bool) -> Self {
        self.bidirectional = bidirectional;
        self
    }

    /// Set attention type.
    #[must_use]
    pub fn with_attention_type(mut self, attention_type: AttentionType) -> Self {
        self.attention_type = attention_type;
        self
    }

    /// Get output dimension based on bidirectional setting.
    fn output_dim(&self) -> usize {
        if self.bidirectional {
            self.hidden_size * 2
        } else {
            self.hidden_size
        }
    }

    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> RNNAttention<B> {
        RNNAttention::new(self.clone(), device)
    }
}

/// Additive attention layer.
#[derive(Module, Debug)]
pub struct AdditiveAttention<B: Backend> {
    /// Query projection.
    query_proj: Linear<B>,
    /// Key projection.
    key_proj: Linear<B>,
    /// Score vector.
    score_proj: Linear<B>,
}

impl<B: Backend> AdditiveAttention<B> {
    /// Create additive attention.
    pub fn new(hidden_dim: usize, attention_dim: usize, device: &B::Device) -> Self {
        let query_proj = LinearConfig::new(hidden_dim, attention_dim)
            .with_bias(false)
            .init(device);
        let key_proj = LinearConfig::new(hidden_dim, attention_dim)
            .with_bias(false)
            .init(device);
        let score_proj = LinearConfig::new(attention_dim, 1)
            .with_bias(false)
            .init(device);

        Self {
            query_proj,
            key_proj,
            score_proj,
        }
    }

    /// Compute attention weights.
    ///
    /// # Arguments
    /// * `query` - Query tensor of shape (batch, hidden_dim)
    /// * `keys` - Key tensor of shape (batch, seq_len, hidden_dim)
    ///
    /// # Returns
    /// Attention weights of shape (batch, seq_len)
    pub fn forward(&self, query: Tensor<B, 2>, keys: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch, seq_len, _] = keys.dims();

        // Project query: (batch, hidden_dim) -> (batch, attention_dim)
        let query_proj = self.query_proj.forward(query);
        // Expand query to (batch, 1, attention_dim)
        let query_proj = query_proj.unsqueeze_dim(1);

        // Project keys: (batch, seq_len, hidden_dim) -> (batch, seq_len, attention_dim)
        let [batch_k, seq_len_k, hidden_k] = keys.dims();
        let keys_flat = keys.reshape([batch_k * seq_len_k, hidden_k]);
        let keys_proj = self.key_proj.forward(keys_flat);
        let attention_dim = query_proj.dims()[2];
        let keys_proj = keys_proj.reshape([batch, seq_len, attention_dim]);

        // Add and apply tanh
        let combined = query_proj + keys_proj;
        let combined = combined.tanh();

        // Project to scores: (batch, seq_len, attention_dim) -> (batch, seq_len, 1)
        let [batch_c, seq_len_c, att_dim] = combined.dims();
        let combined_flat = combined.reshape([batch_c * seq_len_c, att_dim]);
        let scores = self.score_proj.forward(combined_flat);
        let scores = scores.reshape([batch, seq_len]);

        // Softmax to get attention weights
        softmax(scores, 1)
    }
}

/// RNN with Attention for time series classification.
///
/// Architecture:
/// 1. RNN (LSTM/GRU) processes the sequence
/// 2. Attention mechanism computes importance weights for each timestep
/// 3. Weighted sum of RNN outputs produces context vector
/// 4. Linear classifier on context vector
///
/// # Example
///
/// ```rust,ignore
/// use tsai_models::rnn::{RNNAttention, RNNAttentionConfig, AttentionType};
///
/// let config = RNNAttentionConfig::new(3, 100, 5)
///     .with_hidden_size(128)
///     .with_bidirectional(true)
///     .with_attention_type(AttentionType::ScaledDotProduct);
/// let model = config.init::<NdArray>(&device);
///
/// let x = Tensor::random([32, 3, 100], Distribution::Normal(0.0, 1.0), &device);
/// let output = model.forward(x);
/// // output shape: [32, 5]
/// ```
#[derive(Module, Debug)]
pub struct RNNAttention<B: Backend> {
    /// LSTM layer.
    lstm: Lstm<B>,
    /// Additive attention (if used).
    additive_attention: Option<AdditiveAttention<B>>,
    /// Query projection for dot-product attention.
    query_proj: Option<Linear<B>>,
    /// Dropout layer.
    dropout: Dropout,
    /// Final classifier.
    fc: Linear<B>,
    /// Use scaled dot-product (vs regular dot-product or additive).
    /// 0 = Additive, 1 = DotProduct, 2 = ScaledDotProduct
    #[module(skip)]
    attention_mode: u8,
    /// Hidden dimension (after bidirectional).
    #[module(skip)]
    hidden_dim: usize,
}

impl<B: Backend> RNNAttention<B> {
    /// Create a new RNN with Attention model.
    pub fn new(config: RNNAttentionConfig, device: &B::Device) -> Self {
        let lstm = LstmConfig::new(config.n_vars, config.hidden_size, config.bidirectional)
            .init(device);

        let hidden_dim = config.output_dim();

        // Convert AttentionType to u8 mode
        let attention_mode = match config.attention_type {
            AttentionType::Additive => 0,
            AttentionType::DotProduct => 1,
            AttentionType::ScaledDotProduct => 2,
        };

        // Create attention components based on type
        let (additive_attention, query_proj) = match config.attention_type {
            AttentionType::Additive => {
                let attn = AdditiveAttention::new(hidden_dim, config.attention_dim, device);
                (Some(attn), None)
            }
            AttentionType::DotProduct | AttentionType::ScaledDotProduct => {
                let query = LinearConfig::new(hidden_dim, hidden_dim)
                    .with_bias(false)
                    .init(device);
                (None, Some(query))
            }
        };

        let dropout = DropoutConfig::new(config.dropout).init();
        let fc = LinearConfig::new(hidden_dim, config.n_classes).init(device);

        Self {
            lstm,
            additive_attention,
            query_proj,
            dropout,
            fc,
            attention_mode,
            hidden_dim,
        }
    }

    /// Compute attention weights using dot-product attention.
    fn dot_product_attention(&self, outputs: Tensor<B, 3>, scale: bool) -> Tensor<B, 2> {
        let [batch, seq_len, hidden_dim] = outputs.dims();

        // Use mean of outputs as query
        let query = outputs.clone().mean_dim(1);

        // Optionally project query
        let query = if let Some(ref proj) = self.query_proj {
            proj.forward(query)
        } else {
            query
        };

        // Compute attention scores: (batch, hidden) @ (batch, hidden, seq_len) = (batch, seq_len)
        let query = query.unsqueeze_dim(1); // (batch, 1, hidden)
        let keys = outputs.swap_dims(1, 2); // (batch, hidden, seq_len)
        let scores = query.matmul(keys).reshape([batch, seq_len]); // (batch, seq_len)

        // Scale if using scaled dot-product
        let scores = if scale {
            scores / (hidden_dim as f32).sqrt()
        } else {
            scores
        };

        softmax(scores, 1)
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
        let [batch, _n_vars, _seq_len] = x.dims();

        // Transpose to (batch, seq_len, n_vars) for RNN
        let x = x.swap_dims(1, 2);

        // Apply LSTM: outputs shape (batch, seq_len, hidden_dim)
        let (outputs, _) = self.lstm.forward(x, None);

        // Get dimensions from output tensor
        let [_, seq_len, hidden_dim] = outputs.dims();

        // Compute attention weights based on attention mode
        // 0 = Additive, 1 = DotProduct, 2 = ScaledDotProduct
        let attention_weights = match self.attention_mode {
            0 => {
                // Additive: Use last hidden state as query
                let last_hidden =
                    outputs
                        .clone()
                        .slice([0..batch, (seq_len - 1)..seq_len, 0..hidden_dim]);
                let last_hidden = last_hidden.reshape([batch, hidden_dim]);

                self.additive_attention
                    .as_ref()
                    .unwrap()
                    .forward(last_hidden, outputs.clone())
            }
            1 => self.dot_product_attention(outputs.clone(), false),
            _ => self.dot_product_attention(outputs.clone(), true),
        };

        // Apply attention: weighted sum of outputs
        // weights: (batch, seq_len), outputs: (batch, seq_len, hidden_dim)
        let weights = attention_weights.unsqueeze_dim(2); // (batch, seq_len, 1)
        let context = outputs * weights; // (batch, seq_len, hidden_dim)
        let context = context.sum_dim(1).squeeze(1); // (batch, hidden_dim)

        // Apply dropout and classify
        let context = self.dropout.forward(context);
        self.fc.forward(context)
    }

    /// Forward pass returning probabilities.
    pub fn forward_probs(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let logits = self.forward(x);
        softmax(logits, 1)
    }

    /// Get attention weights for interpretability.
    ///
    /// Returns attention weights of shape (batch, seq_len) showing
    /// which timesteps the model focuses on.
    pub fn get_attention_weights(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch, _n_vars, _seq_len] = x.dims();

        // Transpose to (batch, seq_len, n_vars) for RNN
        let x = x.swap_dims(1, 2);

        // Apply LSTM
        let (outputs, _) = self.lstm.forward(x, None);

        let [_, seq_len, hidden_dim] = outputs.dims();

        // Compute attention weights based on attention mode
        // 0 = Additive, 1 = DotProduct, 2 = ScaledDotProduct
        match self.attention_mode {
            0 => {
                let last_hidden =
                    outputs
                        .clone()
                        .slice([0..batch, (seq_len - 1)..seq_len, 0..hidden_dim]);
                let last_hidden = last_hidden.reshape([batch, hidden_dim]);

                self.additive_attention
                    .as_ref()
                    .unwrap()
                    .forward(last_hidden, outputs)
            }
            1 => self.dot_product_attention(outputs, false),
            _ => self.dot_product_attention(outputs, true),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rnn_attention_config_default() {
        let config = RNNAttentionConfig::default();
        assert_eq!(config.hidden_size, 128);
        assert_eq!(config.attention_type, AttentionType::ScaledDotProduct);
        assert!(config.bidirectional);
    }

    #[test]
    fn test_rnn_attention_config_builder() {
        let config = RNNAttentionConfig::new(3, 200, 10)
            .with_hidden_size(256)
            .with_bidirectional(false)
            .with_attention_type(AttentionType::Additive);

        assert_eq!(config.n_vars, 3);
        assert_eq!(config.seq_len, 200);
        assert_eq!(config.n_classes, 10);
        assert_eq!(config.hidden_size, 256);
        assert!(!config.bidirectional);
        assert_eq!(config.attention_type, AttentionType::Additive);
    }

    #[test]
    fn test_output_dim() {
        let config = RNNAttentionConfig::default();
        assert_eq!(config.output_dim(), 256); // bidirectional by default

        let config_uni = RNNAttentionConfig {
            bidirectional: false,
            ..Default::default()
        };
        assert_eq!(config_uni.output_dim(), 128);
    }
}
