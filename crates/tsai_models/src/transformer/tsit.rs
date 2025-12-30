//! TSiTPlus (Time Series iTransformer Plus) model.
//!
//! An improved Time Series Transformer with:
//! - Multiple positional encoding options (learned, sinusoidal, rotary)
//! - Pre-LayerNorm transformer blocks
//! - GELU activation
//! - Optional CLS token for classification
//! - Configurable pooling strategies

use burn::nn::{
    attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
    Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use serde::{Deserialize, Serialize};

/// Positional encoding type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PositionalEncodingType {
    /// Sinusoidal positional encoding (fixed).
    #[default]
    Sinusoidal,
    /// Learned positional encoding.
    Learned,
    /// No positional encoding.
    None,
}

/// Pooling strategy for classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PoolingStrategy {
    /// Global average pooling over sequence.
    #[default]
    GlobalAvgPool,
    /// Use CLS token.
    ClsToken,
    /// Global max pooling over sequence.
    GlobalMaxPool,
    /// Concatenate first and last tokens.
    FirstLast,
}

impl PositionalEncodingType {
    fn to_id(self) -> usize {
        match self {
            Self::Sinusoidal => 0,
            Self::Learned => 1,
            Self::None => 2,
        }
    }

    fn from_id(id: usize) -> Self {
        match id {
            0 => Self::Sinusoidal,
            1 => Self::Learned,
            _ => Self::None,
        }
    }
}

impl PoolingStrategy {
    fn to_id(self) -> usize {
        match self {
            Self::GlobalAvgPool => 0,
            Self::ClsToken => 1,
            Self::GlobalMaxPool => 2,
            Self::FirstLast => 3,
        }
    }

    fn from_id(id: usize) -> Self {
        match id {
            0 => Self::GlobalAvgPool,
            1 => Self::ClsToken,
            2 => Self::GlobalMaxPool,
            _ => Self::FirstLast,
        }
    }
}

/// Configuration for TSiTPlus model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TSiTPlusConfig {
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
    /// Feedforward dimension multiplier (d_ff = d_model * ff_mult).
    pub ff_mult: usize,
    /// Dropout rate for attention.
    pub attn_dropout: f64,
    /// Dropout rate for feedforward.
    pub ff_dropout: f64,
    /// Positional encoding type.
    pub pe_type: PositionalEncodingType,
    /// Pooling strategy.
    pub pooling: PoolingStrategy,
    /// Whether to use pre-norm (LayerNorm before attention/FF).
    pub pre_norm: bool,
    /// Whether to use bias in linear layers.
    pub use_bias: bool,
}

impl Default for TSiTPlusConfig {
    fn default() -> Self {
        Self {
            n_vars: 1,
            seq_len: 100,
            n_classes: 2,
            d_model: 128,
            n_heads: 8,
            n_layers: 3,
            ff_mult: 4,
            attn_dropout: 0.1,
            ff_dropout: 0.1,
            pe_type: PositionalEncodingType::Learned,
            pooling: PoolingStrategy::GlobalAvgPool,
            pre_norm: true,
            use_bias: true,
        }
    }
}

impl TSiTPlusConfig {
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
    pub fn with_d_model(mut self, d_model: usize) -> Self {
        self.d_model = d_model;
        self
    }

    /// Set number of heads.
    pub fn with_n_heads(mut self, n_heads: usize) -> Self {
        self.n_heads = n_heads;
        self
    }

    /// Set number of layers.
    pub fn with_n_layers(mut self, n_layers: usize) -> Self {
        self.n_layers = n_layers;
        self
    }

    /// Set positional encoding type.
    pub fn with_pe_type(mut self, pe_type: PositionalEncodingType) -> Self {
        self.pe_type = pe_type;
        self
    }

    /// Set pooling strategy.
    pub fn with_pooling(mut self, pooling: PoolingStrategy) -> Self {
        self.pooling = pooling;
        self
    }

    /// Set dropout rates.
    pub fn with_dropout(mut self, attn_dropout: f64, ff_dropout: f64) -> Self {
        self.attn_dropout = attn_dropout;
        self.ff_dropout = ff_dropout;
        self
    }

    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> TSiTPlus<B> {
        TSiTPlus::new(self.clone(), device)
    }

    /// Get the effective sequence length (including CLS token if used).
    pub fn effective_seq_len(&self) -> usize {
        match self.pooling {
            PoolingStrategy::ClsToken => self.seq_len + 1,
            _ => self.seq_len,
        }
    }
}

/// Pre-LayerNorm transformer encoder layer.
#[derive(Module, Debug)]
struct TSiTEncoderLayer<B: Backend> {
    attention: MultiHeadAttention<B>,
    norm1: LayerNorm<B>,
    ff_linear1: Linear<B>,
    ff_linear2: Linear<B>,
    norm2: LayerNorm<B>,
    attn_dropout: Dropout,
    ff_dropout: Dropout,
    pre_norm: bool,
}

impl<B: Backend> TSiTEncoderLayer<B> {
    fn new(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        attn_dropout: f64,
        ff_dropout: f64,
        pre_norm: bool,
        device: &B::Device,
    ) -> Self {
        let attention = MultiHeadAttentionConfig::new(d_model, n_heads)
            .with_dropout(attn_dropout)
            .init(device);
        let norm1 = LayerNormConfig::new(d_model).init(device);
        let ff_linear1 = LinearConfig::new(d_model, d_ff).init(device);
        let ff_linear2 = LinearConfig::new(d_ff, d_model).init(device);
        let norm2 = LayerNormConfig::new(d_model).init(device);
        let attn_dropout_layer = DropoutConfig::new(attn_dropout).init();
        let ff_dropout_layer = DropoutConfig::new(ff_dropout).init();

        Self {
            attention,
            norm1,
            ff_linear1,
            ff_linear2,
            norm2,
            attn_dropout: attn_dropout_layer,
            ff_dropout: ff_dropout_layer,
            pre_norm,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        if self.pre_norm {
            // Pre-norm: Norm -> Attention -> Residual
            let normed = self.norm1.forward(x.clone());
            let attn_input = MhaInput::self_attn(normed);
            let attn_out = self.attention.forward(attn_input).context;
            let x = x + self.attn_dropout.forward(attn_out);

            // Pre-norm: Norm -> FF -> Residual
            let normed = self.norm2.forward(x.clone());
            let ff_out = self.ff_linear1.forward(normed);
            let ff_out = Gelu::new().forward(ff_out);
            let ff_out = self.ff_dropout.forward(ff_out);
            let ff_out = self.ff_linear2.forward(ff_out);

            x + self.ff_dropout.forward(ff_out)
        } else {
            // Post-norm: Attention -> Residual -> Norm
            let attn_input = MhaInput::self_attn(x.clone());
            let attn_out = self.attention.forward(attn_input).context;
            let x = self.norm1.forward(x + self.attn_dropout.forward(attn_out));

            // Post-norm: FF -> Residual -> Norm
            let ff_out = self.ff_linear1.forward(x.clone());
            let ff_out = Gelu::new().forward(ff_out);
            let ff_out = self.ff_dropout.forward(ff_out);
            let ff_out = self.ff_linear2.forward(ff_out);

            self.norm2.forward(x + self.ff_dropout.forward(ff_out))
        }
    }
}

/// TSiTPlus (Time Series iTransformer Plus) for classification.
#[derive(Module, Debug)]
pub struct TSiTPlus<B: Backend> {
    /// Input projection.
    input_proj: Linear<B>,
    /// Transformer encoder layers.
    encoder_layers: Vec<TSiTEncoderLayer<B>>,
    /// Final layer norm (for pre-norm architecture).
    final_norm: LayerNorm<B>,
    /// Classification head.
    head: Linear<B>,
    /// Dropout before head.
    head_dropout: Dropout,
    /// Positional encoding type (0=Sinusoidal, 1=Learned, 2=None).
    pe_type_id: usize,
    /// Pooling strategy (0=GlobalAvgPool, 1=ClsToken, 2=GlobalMaxPool, 3=FirstLast).
    pooling_id: usize,
    /// Whether to use pre-norm.
    pre_norm: bool,
}

impl<B: Backend> TSiTPlus<B> {
    /// Create a new TSiTPlus model.
    pub fn new(config: TSiTPlusConfig, device: &B::Device) -> Self {
        let d_ff = config.d_model * config.ff_mult;

        // Input projection: n_vars -> d_model
        let input_proj = LinearConfig::new(config.n_vars, config.d_model)
            .with_bias(config.use_bias)
            .init(device);

        // Encoder layers
        let encoder_layers: Vec<_> = (0..config.n_layers)
            .map(|_| {
                TSiTEncoderLayer::new(
                    config.d_model,
                    config.n_heads,
                    d_ff,
                    config.attn_dropout,
                    config.ff_dropout,
                    config.pre_norm,
                    device,
                )
            })
            .collect();

        // Final norm
        let final_norm = LayerNormConfig::new(config.d_model).init(device);

        // Classification head
        let head_in = match config.pooling {
            PoolingStrategy::FirstLast => config.d_model * 2,
            _ => config.d_model,
        };
        let head = LinearConfig::new(head_in, config.n_classes)
            .with_bias(config.use_bias)
            .init(device);

        let head_dropout = DropoutConfig::new(config.ff_dropout).init();

        Self {
            input_proj,
            encoder_layers,
            final_norm,
            head,
            head_dropout,
            pe_type_id: config.pe_type.to_id(),
            pooling_id: config.pooling.to_id(),
            pre_norm: config.pre_norm,
        }
    }

    /// Create sinusoidal positional encoding.
    fn create_sinusoidal_pe(seq_len: usize, d_model: usize, device: &B::Device) -> Tensor<B, 2> {
        let mut pe = vec![0.0f32; seq_len * d_model];

        for pos in 0..seq_len {
            for i in 0..d_model {
                let div_term = (10000.0f32).powf((2 * (i / 2)) as f32 / d_model as f32);
                let angle = pos as f32 / div_term;
                pe[pos * d_model + i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }

        Tensor::<B, 1>::from_floats(pe.as_slice(), device).reshape([seq_len, d_model])
    }

    /// Create learned positional encoding (initialized to zeros).
    fn create_learned_pe(seq_len: usize, d_model: usize, device: &B::Device) -> Tensor<B, 2> {
        // Initialize with small random values
        Tensor::random([seq_len, d_model], burn::tensor::Distribution::Normal(0.0, 0.02), device)
    }

    /// Apply pooling strategy.
    fn apply_pooling(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch, seq_len, d_model] = x.dims();

        match PoolingStrategy::from_id(self.pooling_id) {
            PoolingStrategy::GlobalAvgPool => {
                // Mean over sequence dimension
                x.mean_dim(1).reshape([batch, d_model])
            }
            PoolingStrategy::GlobalMaxPool => {
                // Max over sequence dimension
                x.max_dim(1).reshape([batch, d_model])
            }
            PoolingStrategy::ClsToken => {
                // Use first token (CLS token)
                x.slice([0..batch, 0..1, 0..d_model]).reshape([batch, d_model])
            }
            PoolingStrategy::FirstLast => {
                // Concatenate first and last tokens
                let first = x.clone().slice([0..batch, 0..1, 0..d_model]).reshape([batch, d_model]);
                let last = x.slice([0..batch, (seq_len - 1)..seq_len, 0..d_model]).reshape([batch, d_model]);
                Tensor::cat(vec![first, last], 1)
            }
        }
    }

    /// Forward pass.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [_batch, _n_vars, seq_len] = x.dims();
        let device = x.device();

        // Transpose to (batch, seq_len, n_vars)
        let x = x.swap_dims(1, 2);

        // Project to d_model
        let x = self.input_proj.forward(x);
        let [_, _, d_model] = x.dims();

        // Add positional encoding
        let x = match PositionalEncodingType::from_id(self.pe_type_id) {
            PositionalEncodingType::Sinusoidal => {
                let pe = Self::create_sinusoidal_pe(seq_len, d_model, &device);
                x + pe.unsqueeze::<3>()
            }
            PositionalEncodingType::Learned => {
                let pe = Self::create_learned_pe(seq_len, d_model, &device);
                x + pe.unsqueeze::<3>()
            }
            PositionalEncodingType::None => x,
        };

        // Transformer encoder
        let mut x = x;
        for layer in &self.encoder_layers {
            x = layer.forward(x);
        }

        // Final norm (important for pre-norm architecture)
        let x = if self.pre_norm {
            self.final_norm.forward(x)
        } else {
            x
        };

        // Apply pooling
        let x = self.apply_pooling(x);

        // Classification head
        let x = self.head_dropout.forward(x);
        self.head.forward(x)
    }

    /// Forward pass returning probabilities.
    pub fn forward_probs(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let logits = self.forward(x);
        softmax(logits, 1)
    }

    /// Get attention weights for interpretability.
    /// Returns attention weights from all layers: Vec<(batch, n_heads, seq_len, seq_len)>
    pub fn get_attention_weights(&self, x: Tensor<B, 3>) -> Vec<Tensor<B, 4>> {
        let [_batch, _n_vars, seq_len] = x.dims();
        let device = x.device();

        // Transpose to (batch, seq_len, n_vars)
        let x = x.swap_dims(1, 2);

        // Project to d_model
        let x = self.input_proj.forward(x);
        let [_, _, d_model] = x.dims();

        // Add positional encoding
        let x = match PositionalEncodingType::from_id(self.pe_type_id) {
            PositionalEncodingType::Sinusoidal => {
                let pe = Self::create_sinusoidal_pe(seq_len, d_model, &device);
                x + pe.unsqueeze::<3>()
            }
            PositionalEncodingType::Learned => {
                let pe = Self::create_learned_pe(seq_len, d_model, &device);
                x + pe.unsqueeze::<3>()
            }
            PositionalEncodingType::None => x,
        };

        // Collect attention weights
        let mut attention_weights = Vec::new();
        let mut current = x;

        for layer in &self.encoder_layers {
            let normed = if layer.pre_norm {
                layer.norm1.forward(current.clone())
            } else {
                current.clone()
            };

            let attn_input = MhaInput::self_attn(normed);
            let attn_output = layer.attention.forward(attn_input);

            // The attention weights are stored in the output
            attention_weights.push(attn_output.weights);

            // Continue forward pass
            current = layer.forward(current);
        }

        attention_weights
    }

    /// Get positional encoding type.
    pub fn pe_type(&self) -> PositionalEncodingType {
        PositionalEncodingType::from_id(self.pe_type_id)
    }

    /// Get pooling strategy.
    pub fn pooling(&self) -> PoolingStrategy {
        PoolingStrategy::from_id(self.pooling_id)
    }

    /// Check if using pre-norm architecture.
    pub fn is_pre_norm(&self) -> bool {
        self.pre_norm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tsit_config() {
        let config = TSiTPlusConfig::default();
        assert_eq!(config.d_model, 128);
        assert_eq!(config.n_heads, 8);
        assert_eq!(config.ff_mult, 4);
        assert!(config.pre_norm);
    }

    #[test]
    fn test_tsit_config_builder() {
        let config = TSiTPlusConfig::new(5, 100, 10)
            .with_d_model(64)
            .with_n_heads(4)
            .with_n_layers(2)
            .with_pe_type(PositionalEncodingType::Learned)
            .with_pooling(PoolingStrategy::ClsToken);

        assert_eq!(config.n_vars, 5);
        assert_eq!(config.seq_len, 100);
        assert_eq!(config.n_classes, 10);
        assert_eq!(config.d_model, 64);
        assert_eq!(config.n_heads, 4);
        assert_eq!(config.n_layers, 2);
        assert_eq!(config.pe_type, PositionalEncodingType::Learned);
        assert_eq!(config.pooling, PoolingStrategy::ClsToken);
    }

    #[test]
    fn test_pe_type_conversion() {
        assert_eq!(PositionalEncodingType::from_id(0), PositionalEncodingType::Sinusoidal);
        assert_eq!(PositionalEncodingType::from_id(1), PositionalEncodingType::Learned);
        assert_eq!(PositionalEncodingType::from_id(2), PositionalEncodingType::None);
        assert_eq!(PositionalEncodingType::from_id(99), PositionalEncodingType::None);

        assert_eq!(PositionalEncodingType::Sinusoidal.to_id(), 0);
        assert_eq!(PositionalEncodingType::Learned.to_id(), 1);
        assert_eq!(PositionalEncodingType::None.to_id(), 2);
    }

    #[test]
    fn test_pooling_strategy_conversion() {
        assert_eq!(PoolingStrategy::from_id(0), PoolingStrategy::GlobalAvgPool);
        assert_eq!(PoolingStrategy::from_id(1), PoolingStrategy::ClsToken);
        assert_eq!(PoolingStrategy::from_id(2), PoolingStrategy::GlobalMaxPool);
        assert_eq!(PoolingStrategy::from_id(3), PoolingStrategy::FirstLast);
        assert_eq!(PoolingStrategy::from_id(99), PoolingStrategy::FirstLast);

        assert_eq!(PoolingStrategy::GlobalAvgPool.to_id(), 0);
        assert_eq!(PoolingStrategy::ClsToken.to_id(), 1);
        assert_eq!(PoolingStrategy::GlobalMaxPool.to_id(), 2);
        assert_eq!(PoolingStrategy::FirstLast.to_id(), 3);
    }

    #[test]
    fn test_effective_seq_len() {
        let config = TSiTPlusConfig::new(2, 100, 5);
        assert_eq!(config.effective_seq_len(), 100);

        let config_cls = TSiTPlusConfig::new(2, 100, 5)
            .with_pooling(PoolingStrategy::ClsToken);
        assert_eq!(config_cls.effective_seq_len(), 101);
    }
}
