//! TSSequencerPlus: Sequencer model adapted for time series.
//!
//! Uses bidirectional LSTM-like recurrence for sequence modeling instead
//! of self-attention, offering a more parameter-efficient alternative.
//!
//! Reference: "Sequencer: Deep LSTM for Image Classification" by Tatsunami & Taki (2022)

use burn::nn::{
    pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig},
    Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
    Lstm, LstmConfig,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use serde::{Deserialize, Serialize};

/// Configuration for TSSequencerPlus model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TSSequencerPlusConfig {
    /// Number of input variables/channels.
    pub n_vars: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// Model dimension (hidden size).
    pub d_model: usize,
    /// Number of Sequencer blocks.
    pub n_layers: usize,
    /// LSTM hidden size in Sequencer blocks.
    pub lstm_hidden: usize,
    /// Feedforward expansion factor.
    pub ff_mult: usize,
    /// Dropout rate.
    pub dropout: f64,
    /// Whether to use bidirectional LSTM in Sequencer blocks.
    pub bidirectional: bool,
}

impl Default for TSSequencerPlusConfig {
    fn default() -> Self {
        Self {
            n_vars: 1,
            seq_len: 100,
            n_classes: 2,
            d_model: 128,
            n_layers: 4,
            lstm_hidden: 64,
            ff_mult: 4,
            dropout: 0.1,
            bidirectional: true,
        }
    }
}

impl TSSequencerPlusConfig {
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

    /// Set number of layers.
    #[must_use]
    pub fn with_n_layers(mut self, n_layers: usize) -> Self {
        self.n_layers = n_layers;
        self
    }

    /// Set LSTM hidden size.
    #[must_use]
    pub fn with_lstm_hidden(mut self, lstm_hidden: usize) -> Self {
        self.lstm_hidden = lstm_hidden;
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

    /// Set whether to use bidirectional LSTM.
    #[must_use]
    pub fn with_bidirectional(mut self, bidirectional: bool) -> Self {
        self.bidirectional = bidirectional;
        self
    }

    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> TSSequencerPlus<B> {
        TSSequencerPlus::new(self.clone(), device)
    }
}

/// Sequencer block with LSTM-based sequence mixing.
///
/// Uses LSTM to mix information across the sequence dimension,
/// similar to how attention mixes tokens in transformers.
#[derive(Module, Debug)]
struct SequencerBlock<B: Backend> {
    /// Layer norm before LSTM.
    norm1: LayerNorm<B>,
    /// Forward LSTM for sequence mixing.
    lstm_fwd: Lstm<B>,
    /// Backward LSTM for bidirectional (optional).
    lstm_bwd: Option<Lstm<B>>,
    /// Projection after LSTM.
    lstm_proj: Linear<B>,
    /// Layer norm before FFN.
    norm2: LayerNorm<B>,
    /// Feedforward linear 1.
    ff_linear1: Linear<B>,
    /// Feedforward linear 2.
    ff_linear2: Linear<B>,
    /// Dropout.
    dropout: Dropout,
    /// Whether bidirectional.
    #[module(skip)]
    bidirectional: bool,
    /// LSTM hidden size.
    #[module(skip)]
    lstm_hidden: usize,
}

impl<B: Backend> SequencerBlock<B> {
    fn new(
        d_model: usize,
        lstm_hidden: usize,
        ff_mult: usize,
        dropout: f64,
        bidirectional: bool,
        device: &B::Device,
    ) -> Self {
        let norm1 = LayerNormConfig::new(d_model).init(device);

        // LSTM for sequence mixing
        let lstm_fwd = LstmConfig::new(d_model, lstm_hidden, true).init(device);
        let lstm_bwd = if bidirectional {
            Some(LstmConfig::new(d_model, lstm_hidden, true).init(device))
        } else {
            None
        };

        // Project LSTM output back to d_model
        let lstm_out_size = if bidirectional {
            lstm_hidden * 2
        } else {
            lstm_hidden
        };
        let lstm_proj = LinearConfig::new(lstm_out_size, d_model).init(device);

        let norm2 = LayerNormConfig::new(d_model).init(device);
        let d_ff = d_model * ff_mult;
        let ff_linear1 = LinearConfig::new(d_model, d_ff).init(device);
        let ff_linear2 = LinearConfig::new(d_ff, d_model).init(device);
        let dropout_layer = DropoutConfig::new(dropout).init();

        Self {
            norm1,
            lstm_fwd,
            lstm_bwd,
            lstm_proj,
            norm2,
            ff_linear1,
            ff_linear2,
            dropout: dropout_layer,
            bidirectional,
            lstm_hidden,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_batch, _seq_len, _d_model] = x.dims();

        // LSTM branch with residual
        let residual = x.clone();
        let normed = self.norm1.forward(x);

        // Forward LSTM: expects (batch, seq, features)
        let (fwd_out, _) = self.lstm_fwd.forward(normed.clone(), None);

        // Backward LSTM (reverse sequence, run LSTM, reverse back)
        let lstm_out = if let Some(ref lstm_bwd) = self.lstm_bwd {
            // Reverse sequence along time dimension
            let reversed = normed.flip([1]);
            let (bwd_out, _) = lstm_bwd.forward(reversed, None);
            // Reverse back
            let bwd_out = bwd_out.flip([1]);
            // Concatenate forward and backward
            Tensor::cat(vec![fwd_out, bwd_out], 2)
        } else {
            fwd_out
        };

        // Project back to d_model and add residual
        let lstm_out = self.lstm_proj.forward(lstm_out);
        let x = residual + self.dropout.forward(lstm_out);

        // Feedforward branch with residual
        let residual = x.clone();
        let normed = self.norm2.forward(x);
        let ff_out = self.ff_linear1.forward(normed);
        let ff_out = burn::tensor::activation::gelu(ff_out);
        let ff_out = self.dropout.forward(ff_out);
        let ff_out = self.ff_linear2.forward(ff_out);

        residual + self.dropout.forward(ff_out)
    }
}

/// TSSequencerPlus for time series classification.
///
/// Uses LSTM-based Sequencer blocks for sequence modeling,
/// offering an alternative to attention-based transformers.
///
/// # Architecture
///
/// ```text
/// Input (B, V, L)
///       |
///       +---> [Permute to (B, L, V)] -> [Proj to d_model]
///       |
///       +---> [Sequencer Block 1] -> [Sequencer Block 2] -> ... -> [Sequencer Block N]
///       |
///       +---> [Global Average Pool] -> (B, D)
///       |
///       +---> [Linear] -> Output (B, n_classes)
/// ```
///
/// Each Sequencer Block:
/// ```text
/// x -> [LayerNorm] -> [BiLSTM] -> [Linear] -> + x
///                                              |
/// -> [LayerNorm] -> [Linear] -> [GELU] -> [Linear] -> + -> out
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use tsai_models::transformer::{TSSequencerPlus, TSSequencerPlusConfig};
///
/// let config = TSSequencerPlusConfig::new(3, 100, 5)
///     .with_d_model(128)
///     .with_n_layers(4)
///     .with_bidirectional(true);
/// let model = config.init::<NdArray>(&device);
///
/// let x = Tensor::random([32, 3, 100], Distribution::Normal(0.0, 1.0), &device);
/// let output = model.forward(x);
/// // output shape: [32, 5]
/// ```
#[derive(Module, Debug)]
pub struct TSSequencerPlus<B: Backend> {
    /// Input projection.
    input_proj: Linear<B>,
    /// Sequencer blocks.
    blocks: Vec<SequencerBlock<B>>,
    /// Final layer norm.
    final_norm: LayerNorm<B>,
    /// Global average pooling.
    gap: AdaptiveAvgPool1d,
    /// Classification head.
    head: Linear<B>,
    /// Model dimension.
    #[module(skip)]
    d_model: usize,
}

impl<B: Backend> TSSequencerPlus<B> {
    /// Create a new TSSequencerPlus model.
    pub fn new(config: TSSequencerPlusConfig, device: &B::Device) -> Self {
        // Input projection
        let input_proj = LinearConfig::new(config.n_vars, config.d_model).init(device);

        // Sequencer blocks
        let blocks: Vec<_> = (0..config.n_layers)
            .map(|_| {
                SequencerBlock::new(
                    config.d_model,
                    config.lstm_hidden,
                    config.ff_mult,
                    config.dropout,
                    config.bidirectional,
                    device,
                )
            })
            .collect();

        let final_norm = LayerNormConfig::new(config.d_model).init(device);
        let gap = AdaptiveAvgPool1dConfig::new(1).init();
        let head = LinearConfig::new(config.d_model, config.n_classes).init(device);

        Self {
            input_proj,
            blocks,
            final_norm,
            gap,
            head,
            d_model: config.d_model,
        }
    }

    /// Forward pass.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, _n_vars, _seq_len] = x.dims();

        // Transpose: (B, V, L) -> (B, L, V)
        let out = x.swap_dims(1, 2);

        // Project to d_model: (B, L, V) -> (B, L, D)
        let out = self.input_proj.forward(out);

        // Apply Sequencer blocks
        let mut out = out;
        for block in &self.blocks {
            out = block.forward(out);
        }

        // Final normalization
        let out = self.final_norm.forward(out);

        // Transpose for pooling: (B, L, D) -> (B, D, L)
        let out = out.swap_dims(1, 2);

        // Global average pooling
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
    fn test_ts_sequencer_config_default() {
        let config = TSSequencerPlusConfig::default();
        assert_eq!(config.d_model, 128);
        assert_eq!(config.n_layers, 4);
        assert_eq!(config.lstm_hidden, 64);
        assert!(config.bidirectional);
    }

    #[test]
    fn test_ts_sequencer_config_new() {
        let config = TSSequencerPlusConfig::new(3, 200, 10);
        assert_eq!(config.n_vars, 3);
        assert_eq!(config.seq_len, 200);
        assert_eq!(config.n_classes, 10);
    }

    #[test]
    fn test_ts_sequencer_config_builder() {
        let config = TSSequencerPlusConfig::new(3, 100, 5)
            .with_d_model(256)
            .with_n_layers(6)
            .with_lstm_hidden(128)
            .with_ff_mult(2)
            .with_dropout(0.2)
            .with_bidirectional(false);

        assert_eq!(config.d_model, 256);
        assert_eq!(config.n_layers, 6);
        assert_eq!(config.lstm_hidden, 128);
        assert_eq!(config.ff_mult, 2);
        assert_eq!(config.dropout, 0.2);
        assert!(!config.bidirectional);
    }
}
