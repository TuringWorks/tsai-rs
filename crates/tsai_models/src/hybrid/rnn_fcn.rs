//! RNN-FCN: Hybrid Recurrent and Fully Convolutional Network.
//!
//! Combines the strengths of both architectures:
//! - RNN branch (LSTM/GRU) captures temporal dependencies
//! - FCN branch captures local patterns with convolutions
//! - Features are concatenated for final classification
//!
//! Reference: "LSTM Fully Convolutional Networks for Time Series Classification"
//! by Karim et al. (2019)

use burn::nn::{
    conv::{Conv1d, Conv1dConfig},
    gru::{Gru, GruConfig},
    lstm::{Lstm, LstmConfig},
    pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig},
    BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, Relu,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use serde::{Deserialize, Serialize};

/// RNN type for the hybrid model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RNNFCNType {
    /// LSTM-FCN variant.
    #[default]
    LSTM,
    /// GRU-FCN variant.
    GRU,
}

/// Configuration for RNN-FCN model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RNNFCNConfig {
    /// Number of input variables/channels.
    pub n_vars: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// RNN type (LSTM or GRU).
    pub rnn_type: RNNFCNType,
    /// RNN hidden size.
    pub rnn_hidden_size: usize,
    /// Number of RNN layers.
    pub rnn_layers: usize,
    /// FCN filter sizes.
    pub fcn_filters: Vec<usize>,
    /// FCN kernel sizes.
    pub fcn_kernels: Vec<usize>,
    /// Dropout rate.
    pub dropout: f64,
    /// Whether to use bidirectional RNN.
    pub bidirectional: bool,
}

impl Default for RNNFCNConfig {
    fn default() -> Self {
        Self {
            n_vars: 1,
            seq_len: 100,
            n_classes: 2,
            rnn_type: RNNFCNType::LSTM,
            rnn_hidden_size: 128,
            rnn_layers: 1,
            fcn_filters: vec![128, 256, 128],
            fcn_kernels: vec![8, 5, 3],
            dropout: 0.8, // High dropout as in original paper
            bidirectional: false,
        }
    }
}

impl RNNFCNConfig {
    /// Create a new config.
    pub fn new(n_vars: usize, seq_len: usize, n_classes: usize) -> Self {
        Self {
            n_vars,
            seq_len,
            n_classes,
            ..Default::default()
        }
    }

    /// Set RNN type.
    #[must_use]
    pub fn with_rnn_type(mut self, rnn_type: RNNFCNType) -> Self {
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
    pub fn with_rnn_layers(mut self, layers: usize) -> Self {
        self.rnn_layers = layers;
        self
    }

    /// Set FCN filters.
    #[must_use]
    pub fn with_fcn_filters(mut self, filters: Vec<usize>) -> Self {
        self.fcn_filters = filters;
        self
    }

    /// Set FCN kernel sizes.
    #[must_use]
    pub fn with_fcn_kernels(mut self, kernels: Vec<usize>) -> Self {
        self.fcn_kernels = kernels;
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
    pub fn init<B: Backend>(&self, device: &B::Device) -> RNNFCN<B> {
        RNNFCN::new(self.clone(), device)
    }
}

/// FCN block with convolution, batch norm, and ReLU.
#[derive(Module, Debug)]
struct FCNBlock<B: Backend> {
    conv: Conv1d<B>,
    bn: BatchNorm<B, 1>,
}

impl<B: Backend> FCNBlock<B> {
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

/// RNN-FCN model for time series classification.
///
/// Combines RNN (LSTM/GRU) and FCN branches for robust feature extraction.
///
/// # Architecture
///
/// ```text
/// Input (B, V, L)
///       |
///       +---> [Permute to (B, L, V)] -> [RNN] -> [Last hidden] -> Dropout
///       |                                                            |
///       +---> [Conv1->BN->ReLU] -> [Conv2->BN->ReLU] -> [Conv3->BN->ReLU] -> GAP
///                                                                            |
///                                                         Concat <-----------+
///                                                           |
///                                                        Linear -> Output
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use tsai_models::hybrid::{RNNFCN, RNNFCNConfig, RNNFCNType};
///
/// let config = RNNFCNConfig::new(3, 100, 5)
///     .with_rnn_type(RNNFCNType::LSTM)
///     .with_rnn_hidden_size(128);
/// let model = config.init::<NdArray>(&device);
///
/// let x = Tensor::random([32, 3, 100], Distribution::Normal(0.0, 1.0), &device);
/// let output = model.forward(x);
/// // output shape: [32, 5]
/// ```
#[derive(Module, Debug)]
pub struct RNNFCN<B: Backend> {
    /// LSTM layers (if using LSTM).
    lstm: Option<Lstm<B>>,
    /// GRU layers (if using GRU).
    gru: Option<Gru<B>>,
    /// FCN blocks.
    fcn_blocks: Vec<FCNBlock<B>>,
    /// Global average pooling.
    gap: AdaptiveAvgPool1d,
    /// Dropout for RNN output.
    dropout: Dropout,
    /// Final classifier.
    classifier: Linear<B>,
    /// RNN hidden size (for feature concatenation).
    #[module(skip)]
    rnn_hidden_size: usize,
    /// Last FCN filter size.
    #[module(skip)]
    fcn_out_size: usize,
    /// Whether bidirectional.
    #[module(skip)]
    bidirectional: bool,
}

impl<B: Backend> RNNFCN<B> {
    /// Create a new RNN-FCN model.
    pub fn new(config: RNNFCNConfig, device: &B::Device) -> Self {
        // RNN branch
        let (lstm, gru) = match config.rnn_type {
            RNNFCNType::LSTM => {
                let lstm = LstmConfig::new(config.n_vars, config.rnn_hidden_size, config.bidirectional)
                    .init(device);
                (Some(lstm), None)
            }
            RNNFCNType::GRU => {
                let gru = GruConfig::new(config.n_vars, config.rnn_hidden_size, config.bidirectional)
                    .init(device);
                (None, Some(gru))
            }
        };

        // FCN branch
        let mut fcn_blocks = Vec::new();
        let mut in_channels = config.n_vars;

        for (&filters, &kernel) in config.fcn_filters.iter().zip(&config.fcn_kernels) {
            fcn_blocks.push(FCNBlock::new(in_channels, filters, kernel, device));
            in_channels = filters;
        }

        let gap = AdaptiveAvgPool1dConfig::new(1).init();
        let dropout = DropoutConfig::new(config.dropout).init();

        // Classifier input size = RNN hidden (possibly *2 for bidirectional) + FCN output
        let rnn_out_size = if config.bidirectional {
            config.rnn_hidden_size * 2
        } else {
            config.rnn_hidden_size
        };
        let fcn_out_size = *config.fcn_filters.last().unwrap_or(&128);
        let classifier_in = rnn_out_size + fcn_out_size;
        let classifier = LinearConfig::new(classifier_in, config.n_classes).init(device);

        Self {
            lstm,
            gru,
            fcn_blocks,
            gap,
            dropout,
            classifier,
            rnn_hidden_size: config.rnn_hidden_size,
            fcn_out_size,
            bidirectional: config.bidirectional,
        }
    }

    /// Forward pass.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, _n_vars, seq_len] = x.dims();

        // RNN branch: expects (batch, seq_len, features)
        // Input is (batch, vars, seq_len), need to transpose
        let x_rnn = x.clone().swap_dims(1, 2); // (B, L, V)

        let rnn_out = if let Some(ref lstm) = self.lstm {
            let (output, _) = lstm.forward(x_rnn, None);
            // Take last time step: (B, L, H) -> (B, H)
            let last_idx = seq_len - 1;
            let hidden_size = output.dims()[2];
            output.slice([0..batch_size, last_idx..last_idx + 1, 0..hidden_size])
                .reshape([batch_size, hidden_size])
        } else if let Some(ref gru) = self.gru {
            // GRU forward returns just the output tensor
            let output = gru.forward(x_rnn, None);
            let last_idx = seq_len - 1;
            let hidden_size = output.dims()[2];
            output.slice([0..batch_size, last_idx..last_idx + 1, 0..hidden_size])
                .reshape([batch_size, hidden_size])
        } else {
            panic!("No RNN layer configured");
        };

        let rnn_out = self.dropout.forward(rnn_out);

        // FCN branch
        let mut fcn_out = x;
        for block in &self.fcn_blocks {
            fcn_out = block.forward(fcn_out);
        }

        // Global average pooling: (B, C, L) -> (B, C, 1) -> (B, C)
        let fcn_out = self.gap.forward(fcn_out);
        let fcn_out = fcn_out.reshape([batch_size, self.fcn_out_size]);

        // Concatenate RNN and FCN features
        let combined = Tensor::cat(vec![rnn_out, fcn_out], 1);

        // Classifier
        self.classifier.forward(combined)
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
    fn test_rnnfcn_config_default() {
        let config = RNNFCNConfig::default();
        assert_eq!(config.rnn_type, RNNFCNType::LSTM);
        assert_eq!(config.rnn_hidden_size, 128);
        assert_eq!(config.fcn_filters, vec![128, 256, 128]);
    }

    #[test]
    fn test_rnnfcn_config_new() {
        let config = RNNFCNConfig::new(3, 200, 10);
        assert_eq!(config.n_vars, 3);
        assert_eq!(config.seq_len, 200);
        assert_eq!(config.n_classes, 10);
    }

    #[test]
    fn test_rnnfcn_config_builder() {
        let config = RNNFCNConfig::new(3, 100, 5)
            .with_rnn_type(RNNFCNType::GRU)
            .with_rnn_hidden_size(64)
            .with_fcn_filters(vec![64, 128, 64])
            .with_dropout(0.5);

        assert_eq!(config.rnn_type, RNNFCNType::GRU);
        assert_eq!(config.rnn_hidden_size, 64);
        assert_eq!(config.fcn_filters, vec![64, 128, 64]);
        assert_eq!(config.dropout, 0.5);
    }
}
