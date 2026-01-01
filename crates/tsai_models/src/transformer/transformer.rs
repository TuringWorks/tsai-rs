//! Base Transformer model for time series.
//!
//! This is a vanilla Transformer encoder for time series classification/regression,
//! similar to the base TransformerModel in Python tsai.

use burn::module::Param;
use burn::nn::{
    attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
    Dropout, DropoutConfig, Linear, LinearConfig, LayerNorm, LayerNormConfig,
};
use burn::prelude::*;
use burn::tensor::activation::{gelu, softmax};
use serde::{Deserialize, Serialize};

/// Configuration for the base Transformer model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerModelConfig {
    /// Number of input variables/channels.
    pub n_vars: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Number of output classes (for classification) or targets (for regression).
    pub n_outputs: usize,
    /// Model dimension (embedding size).
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of transformer encoder layers.
    pub n_layers: usize,
    /// Feedforward network dimension (typically 4 * d_model).
    pub d_ff: usize,
    /// Dropout rate.
    pub dropout: f64,
    /// Attention dropout rate.
    pub attn_dropout: f64,
    /// Whether to use positional encoding.
    pub use_pos_encoding: bool,
    /// Type of positional encoding.
    pub pos_encoding_type: PositionalEncodingType,
    /// Whether to use pre-layer normalization (more stable training).
    pub pre_norm: bool,
    /// Activation function for feedforward network.
    pub activation: ActivationType,
    /// How to aggregate sequence for output (mean, first, last, flatten).
    pub aggregation: AggregationType,
}

/// Type of positional encoding.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum PositionalEncodingType {
    /// Sinusoidal positional encoding (original Transformer).
    #[default]
    Sinusoidal,
    /// Learnable positional embeddings.
    Learnable,
    /// No positional encoding.
    None,
}

/// Activation function type.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum ActivationType {
    /// ReLU activation.
    ReLU,
    /// GELU activation (default, better for transformers).
    #[default]
    GELU,
}

/// Aggregation type for output.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum AggregationType {
    /// Mean pooling over sequence dimension.
    #[default]
    Mean,
    /// Use first token (CLS-style).
    First,
    /// Use last token.
    Last,
    /// Flatten all tokens.
    Flatten,
}

impl Default for TransformerModelConfig {
    fn default() -> Self {
        Self {
            n_vars: 1,
            seq_len: 100,
            n_outputs: 2,
            d_model: 128,
            n_heads: 8,
            n_layers: 3,
            d_ff: 512,
            dropout: 0.1,
            attn_dropout: 0.0,
            use_pos_encoding: true,
            pos_encoding_type: PositionalEncodingType::Sinusoidal,
            pre_norm: false,
            activation: ActivationType::GELU,
            aggregation: AggregationType::Mean,
        }
    }
}

impl TransformerModelConfig {
    /// Create a new config with basic parameters.
    pub fn new(n_vars: usize, seq_len: usize, n_outputs: usize) -> Self {
        Self {
            n_vars,
            seq_len,
            n_outputs,
            ..Default::default()
        }
    }

    /// Set model dimension.
    pub fn with_d_model(mut self, d_model: usize) -> Self {
        self.d_model = d_model;
        self
    }

    /// Set number of attention heads.
    pub fn with_n_heads(mut self, n_heads: usize) -> Self {
        self.n_heads = n_heads;
        self
    }

    /// Set number of layers.
    pub fn with_n_layers(mut self, n_layers: usize) -> Self {
        self.n_layers = n_layers;
        self
    }

    /// Set feedforward dimension.
    pub fn with_d_ff(mut self, d_ff: usize) -> Self {
        self.d_ff = d_ff;
        self
    }

    /// Set dropout rate.
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Set positional encoding type.
    pub fn with_pos_encoding(mut self, pos_type: PositionalEncodingType) -> Self {
        self.pos_encoding_type = pos_type;
        self.use_pos_encoding = !matches!(pos_type, PositionalEncodingType::None);
        self
    }

    /// Enable pre-layer normalization.
    pub fn with_pre_norm(mut self, pre_norm: bool) -> Self {
        self.pre_norm = pre_norm;
        self
    }

    /// Set activation function.
    pub fn with_activation(mut self, activation: ActivationType) -> Self {
        self.activation = activation;
        self
    }

    /// Set aggregation type.
    pub fn with_aggregation(mut self, aggregation: AggregationType) -> Self {
        self.aggregation = aggregation;
        self
    }

    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerModel<B> {
        TransformerModel::new(self.clone(), device)
    }
}

/// Transformer encoder layer with pre-norm or post-norm option.
#[derive(Module, Debug)]
pub struct TransformerEncoderLayer<B: Backend> {
    attention: MultiHeadAttention<B>,
    norm1: LayerNorm<B>,
    ff_linear1: Linear<B>,
    ff_linear2: Linear<B>,
    norm2: LayerNorm<B>,
    dropout1: Dropout,
    dropout2: Dropout,
    #[module(skip)]
    pre_norm: bool,
    #[module(skip)]
    use_gelu: bool,
}

impl<B: Backend> TransformerEncoderLayer<B> {
    fn new(config: &TransformerModelConfig, device: &B::Device) -> Self {
        let attention = MultiHeadAttentionConfig::new(config.d_model, config.n_heads)
            .with_dropout(config.attn_dropout)
            .init(device);
        let norm1 = LayerNormConfig::new(config.d_model).init(device);
        let ff_linear1 = LinearConfig::new(config.d_model, config.d_ff).init(device);
        let ff_linear2 = LinearConfig::new(config.d_ff, config.d_model).init(device);
        let norm2 = LayerNormConfig::new(config.d_model).init(device);
        let dropout1 = DropoutConfig::new(config.dropout).init();
        let dropout2 = DropoutConfig::new(config.dropout).init();

        Self {
            attention,
            norm1,
            ff_linear1,
            ff_linear2,
            norm2,
            dropout1,
            dropout2,
            pre_norm: config.pre_norm,
            use_gelu: matches!(config.activation, ActivationType::GELU),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        if self.pre_norm {
            // Pre-norm: norm -> attention -> residual
            let x_norm = self.norm1.forward(x.clone());
            let attn_input = MhaInput::self_attn(x_norm);
            let attn_out = self.attention.forward(attn_input).context;
            let x = x + self.dropout1.forward(attn_out);

            // Feedforward with pre-norm
            let x_norm = self.norm2.forward(x.clone());
            let ff_out = self.ff_linear1.forward(x_norm);
            let ff_out = if self.use_gelu {
                gelu(ff_out)
            } else {
                ff_out.clamp_min(0.0) // ReLU
            };
            let ff_out = self.dropout2.forward(ff_out);
            let ff_out = self.ff_linear2.forward(ff_out);

            x + self.dropout2.forward(ff_out)
        } else {
            // Post-norm (original Transformer): attention -> residual -> norm
            let attn_input = MhaInput::self_attn(x.clone());
            let attn_out = self.attention.forward(attn_input).context;
            let x = self.norm1.forward(x + self.dropout1.forward(attn_out));

            // Feedforward
            let ff_out = self.ff_linear1.forward(x.clone());
            let ff_out = if self.use_gelu {
                gelu(ff_out)
            } else {
                ff_out.clamp_min(0.0) // ReLU
            };
            let ff_out = self.dropout2.forward(ff_out);
            let ff_out = self.ff_linear2.forward(ff_out);

            self.norm2.forward(x + self.dropout2.forward(ff_out))
        }
    }
}

/// Base Transformer model for time series.
///
/// This is a vanilla transformer encoder that can be used for:
/// - Time series classification
/// - Time series regression
/// - As a backbone for more complex models
///
/// # Example
///
/// ```rust,ignore
/// use tsai_models::transformer::{TransformerModel, TransformerModelConfig};
///
/// let config = TransformerModelConfig::new(3, 100, 5)  // 3 vars, 100 timesteps, 5 classes
///     .with_d_model(64)
///     .with_n_layers(4)
///     .with_n_heads(4);
///
/// let model = config.init::<NdArray>(&device);
/// let output = model.forward(input);  // [batch, n_classes]
/// ```
#[derive(Module, Debug)]
pub struct TransformerModel<B: Backend> {
    /// Input embedding (projects n_vars to d_model).
    input_embedding: Linear<B>,
    /// Learnable positional encoding (optional).
    pos_embedding: Option<Param<Tensor<B, 2>>>,
    /// Input dropout.
    input_dropout: Dropout,
    /// Transformer encoder layers.
    encoder_layers: Vec<TransformerEncoderLayer<B>>,
    /// Final layer normalization (for pre-norm models).
    final_norm: Option<LayerNorm<B>>,
    /// Output projection head.
    head: Linear<B>,
    /// Model dimension.
    #[module(skip)]
    d_model: usize,
    /// Sequence length.
    #[module(skip)]
    seq_len: usize,
    /// Whether to use positional encoding.
    #[module(skip)]
    use_pos_encoding: bool,
    /// Aggregation type (0=Mean, 1=First, 2=Last, 3=Flatten).
    #[module(skip)]
    aggregation: u8,
}

impl<B: Backend> TransformerModel<B> {
    /// Create a new TransformerModel.
    pub fn new(config: TransformerModelConfig, device: &B::Device) -> Self {
        // Input embedding: project n_vars to d_model
        let input_embedding = LinearConfig::new(config.n_vars, config.d_model).init(device);

        // Positional embedding (learnable)
        let pos_embedding = if config.use_pos_encoding
            && matches!(config.pos_encoding_type, PositionalEncodingType::Learnable)
        {
            let pe = Tensor::random(
                [config.seq_len, config.d_model],
                burn::tensor::Distribution::Normal(0.0, 0.02),
                device,
            );
            Some(Param::from_tensor(pe))
        } else {
            None
        };

        // Input dropout
        let input_dropout = DropoutConfig::new(config.dropout).init();

        // Encoder layers
        let encoder_layers: Vec<_> = (0..config.n_layers)
            .map(|_| TransformerEncoderLayer::new(&config, device))
            .collect();

        // Final norm for pre-norm models
        let final_norm = if config.pre_norm {
            Some(LayerNormConfig::new(config.d_model).init(device))
        } else {
            None
        };

        // Output head
        let head_input_size = match config.aggregation {
            AggregationType::Flatten => config.d_model * config.seq_len,
            _ => config.d_model,
        };
        let head = LinearConfig::new(head_input_size, config.n_outputs).init(device);

        Self {
            input_embedding,
            pos_embedding,
            input_dropout,
            encoder_layers,
            final_norm,
            head,
            d_model: config.d_model,
            seq_len: config.seq_len,
            use_pos_encoding: config.use_pos_encoding,
            aggregation: match config.aggregation {
                AggregationType::Mean => 0,
                AggregationType::First => 1,
                AggregationType::Last => 2,
                AggregationType::Flatten => 3,
            },
        }
    }

    /// Create sinusoidal positional encoding.
    fn sinusoidal_encoding(&self, seq_len: usize, d_model: usize, device: &B::Device) -> Tensor<B, 2> {
        let mut pe = vec![0.0f32; seq_len * d_model];

        for pos in 0..seq_len {
            for i in 0..d_model {
                let angle = pos as f32 / (10000.0f32).powf((2 * (i / 2)) as f32 / d_model as f32);
                pe[pos * d_model + i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }

        Tensor::<B, 1>::from_floats(pe.as_slice(), device).reshape([seq_len, d_model])
    }

    /// Aggregate sequence outputs.
    fn aggregate(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch, seq_len, d_model] = x.dims();

        match self.aggregation {
            0 => x.mean_dim(1).reshape([batch, d_model]), // Mean
            1 => x.slice([0..batch, 0..1, 0..d_model]).reshape([batch, d_model]), // First
            2 => x.slice([0..batch, seq_len - 1..seq_len, 0..d_model]).reshape([batch, d_model]), // Last
            _ => x.reshape([batch, seq_len * d_model]), // Flatten
        }
    }

    /// Forward pass returning logits.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [_batch, _n_vars, seq_len] = x.dims();
        let device = x.device();

        // Transpose to (batch, seq_len, n_vars)
        let x = x.swap_dims(1, 2);

        // Project to d_model: (batch, seq_len, d_model)
        let mut x = self.input_embedding.forward(x);
        let [_, _, d_model] = x.dims();

        // Add positional encoding
        if self.use_pos_encoding {
            let pos_enc = match &self.pos_embedding {
                Some(pe) => pe.val().clone(),
                None => self.sinusoidal_encoding(seq_len, d_model, &device),
            };
            x = x + pos_enc.unsqueeze::<3>();
        }

        // Input dropout
        x = self.input_dropout.forward(x);

        // Transformer encoder
        for layer in &self.encoder_layers {
            x = layer.forward(x);
        }

        // Final normalization (for pre-norm)
        if let Some(ref norm) = self.final_norm {
            x = norm.forward(x);
        }

        // Aggregate and project
        let x = self.aggregate(x);
        self.head.forward(x)
    }

    /// Forward pass returning probabilities (for classification).
    pub fn forward_probs(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let logits = self.forward(x);
        softmax(logits, 1)
    }

    /// Get encoder output without classification head.
    /// Returns: (batch, seq_len, d_model)
    pub fn encode(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_batch, _n_vars, seq_len] = x.dims();
        let device = x.device();

        // Transpose to (batch, seq_len, n_vars)
        let x = x.swap_dims(1, 2);

        // Project to d_model
        let mut x = self.input_embedding.forward(x);
        let [_, _, d_model] = x.dims();

        // Add positional encoding
        if self.use_pos_encoding {
            let pos_enc = match &self.pos_embedding {
                Some(pe) => pe.val().clone(),
                None => self.sinusoidal_encoding(seq_len, d_model, &device),
            };
            x = x + pos_enc.unsqueeze::<3>();
        }

        // Input dropout
        x = self.input_dropout.forward(x);

        // Transformer encoder
        for layer in &self.encoder_layers {
            x = layer.forward(x);
        }

        // Final normalization
        if let Some(ref norm) = self.final_norm {
            x = norm.forward(x);
        }

        x
    }

    /// Get the model dimension.
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Get the sequence length.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_transformer_model_default() {
        let device = Default::default();
        let config = TransformerModelConfig::new(3, 50, 5);
        let model: TransformerModel<TestBackend> = config.init(&device);

        let input = Tensor::random([4, 3, 50], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let output = model.forward(input);

        assert_eq!(output.dims(), [4, 5]);
    }

    #[test]
    fn test_transformer_model_with_pre_norm() {
        let device = Default::default();
        let config = TransformerModelConfig::new(2, 100, 10)
            .with_d_model(64)
            .with_n_layers(4)
            .with_n_heads(4)
            .with_pre_norm(true);
        let model: TransformerModel<TestBackend> = config.init(&device);

        let input = Tensor::random([2, 2, 100], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let output = model.forward(input);

        assert_eq!(output.dims(), [2, 10]);
    }

    #[test]
    fn test_transformer_model_learnable_pos() {
        let device = Default::default();
        let config = TransformerModelConfig::new(1, 32, 3)
            .with_pos_encoding(PositionalEncodingType::Learnable);
        let model: TransformerModel<TestBackend> = config.init(&device);

        let input = Tensor::random([8, 1, 32], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let output = model.forward(input);

        assert_eq!(output.dims(), [8, 3]);
    }

    #[test]
    fn test_transformer_encode() {
        let device = Default::default();
        let config = TransformerModelConfig::new(3, 50, 5)
            .with_d_model(32);
        let model: TransformerModel<TestBackend> = config.init(&device);

        let input = Tensor::random([4, 3, 50], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let encoded = model.encode(input);

        // Encoded shape: (batch, seq_len, d_model)
        assert_eq!(encoded.dims(), [4, 50, 32]);
    }

    #[test]
    fn test_aggregation_types() {
        let device = Default::default();

        // Test flatten aggregation
        let config = TransformerModelConfig::new(2, 10, 4)
            .with_d_model(16)
            .with_aggregation(AggregationType::Flatten);
        let model: TransformerModel<TestBackend> = config.init(&device);

        let input = Tensor::random([2, 2, 10], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let output = model.forward(input);

        assert_eq!(output.dims(), [2, 4]);
    }

    #[test]
    fn test_forward_probs() {
        let device = Default::default();
        let config = TransformerModelConfig::new(3, 50, 5);
        let model: TransformerModel<TestBackend> = config.init(&device);

        let input = Tensor::random([4, 3, 50], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let probs = model.forward_probs(input);

        assert_eq!(probs.dims(), [4, 5]);

        // Probabilities should sum to 1
        let sum = probs.clone().sum_dim(1);
        let expected = Tensor::ones([4, 1], &device);
        let diff = (sum - expected).abs().max().into_scalar();
        assert!(diff < 1e-5, "Probabilities should sum to 1");
    }
}
