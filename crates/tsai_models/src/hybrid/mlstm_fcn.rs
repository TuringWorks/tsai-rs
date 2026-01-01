//! MLSTM-FCN: Multivariate LSTM Fully Convolutional Network.
//!
//! An enhanced version of LSTM-FCN that uses squeeze-and-excitation
//! style attention on the LSTM outputs for multivariate time series.
//!
//! Reference: "Multivariate LSTM-FCNs for Time Series Classification"
//! by Karim et al. (2019)

use burn::nn::{
    conv::{Conv1d, Conv1dConfig},
    lstm::{Lstm, LstmConfig},
    pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig},
    BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, Relu,
};
use burn::prelude::*;
use burn::tensor::activation::{sigmoid, softmax};
use serde::{Deserialize, Serialize};

/// Configuration for MLSTM-FCN model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLSTMFCNConfig {
    /// Number of input variables/channels.
    pub n_vars: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// LSTM hidden sizes for each cell.
    pub lstm_hidden_sizes: Vec<usize>,
    /// FCN filter sizes.
    pub fcn_filters: Vec<usize>,
    /// FCN kernel sizes.
    pub fcn_kernels: Vec<usize>,
    /// Dropout rate.
    pub dropout: f64,
    /// Squeeze-and-excitation reduction ratio.
    pub se_reduction: usize,
    /// Whether to use attention on LSTM outputs.
    pub use_attention: bool,
}

impl Default for MLSTMFCNConfig {
    fn default() -> Self {
        Self {
            n_vars: 1,
            seq_len: 100,
            n_classes: 2,
            lstm_hidden_sizes: vec![128],
            fcn_filters: vec![128, 256, 128],
            fcn_kernels: vec![8, 5, 3],
            dropout: 0.8,
            se_reduction: 16,
            use_attention: true,
        }
    }
}

impl MLSTMFCNConfig {
    /// Create a new config.
    pub fn new(n_vars: usize, seq_len: usize, n_classes: usize) -> Self {
        Self {
            n_vars,
            seq_len,
            n_classes,
            ..Default::default()
        }
    }

    /// Set LSTM hidden sizes.
    #[must_use]
    pub fn with_lstm_hidden_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.lstm_hidden_sizes = sizes;
        self
    }

    /// Set FCN filter sizes.
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

    /// Set squeeze-and-excitation reduction ratio.
    #[must_use]
    pub fn with_se_reduction(mut self, reduction: usize) -> Self {
        self.se_reduction = reduction;
        self
    }

    /// Set whether to use attention.
    #[must_use]
    pub fn with_attention(mut self, use_attention: bool) -> Self {
        self.use_attention = use_attention;
        self
    }

    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> MLSTMFCN<B> {
        MLSTMFCN::new(self.clone(), device)
    }
}

/// Squeeze-and-Excitation block for channel attention.
#[derive(Module, Debug)]
struct SqueezeExcitation<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> SqueezeExcitation<B> {
    fn new(channels: usize, reduction: usize, device: &B::Device) -> Self {
        let reduced = (channels / reduction).max(1);
        let fc1 = LinearConfig::new(channels, reduced).init(device);
        let fc2 = LinearConfig::new(reduced, channels).init(device);
        Self { fc1, fc2 }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        // x: (batch, channels)
        let out = self.fc1.forward(x.clone());
        let out = Relu::new().forward(out);
        let out = self.fc2.forward(out);
        let scale = sigmoid(out);
        x * scale
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

/// MLSTM-FCN: Multivariate LSTM Fully Convolutional Network.
///
/// This model enhances LSTM-FCN with:
/// - Multiple LSTM cells for richer temporal representations
/// - Squeeze-and-excitation attention on LSTM outputs
/// - Per-variable processing for multivariate time series
///
/// # Architecture
///
/// ```text
/// Input (B, V, L)
///       |
///       +---> [Permute to (B, L, V)] -> [LSTM] -> [Last hidden] -> SE-Attention
///       |                                                              |
///       +---> [Conv1->BN->ReLU] -> [Conv2->BN->ReLU] -> [Conv3->BN->ReLU] -> GAP
///                                                                             |
///                                                          Concat <-----------+
///                                                            |
///                                                         Linear -> Output
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use tsai_models::hybrid::{MLSTMFCN, MLSTMFCNConfig};
///
/// let config = MLSTMFCNConfig::new(3, 100, 5)
///     .with_lstm_hidden_sizes(vec![128, 128])
///     .with_use_attention(true);
/// let model = config.init::<NdArray>(&device);
///
/// let x = Tensor::random([32, 3, 100], Distribution::Normal(0.0, 1.0), &device);
/// let output = model.forward(x);
/// // output shape: [32, 5]
/// ```
#[derive(Module, Debug)]
pub struct MLSTMFCN<B: Backend> {
    /// LSTM layers.
    lstms: Vec<Lstm<B>>,
    /// Squeeze-and-excitation attention.
    se_attention: Option<SqueezeExcitation<B>>,
    /// FCN blocks.
    fcn_blocks: Vec<FCNBlock<B>>,
    /// Global average pooling.
    gap: AdaptiveAvgPool1d,
    /// Dropout for LSTM output.
    dropout: Dropout,
    /// Final classifier.
    classifier: Linear<B>,
    /// Total LSTM hidden size.
    #[module(skip)]
    lstm_total_hidden: usize,
    /// FCN output size.
    #[module(skip)]
    fcn_out_size: usize,
}

impl<B: Backend> MLSTMFCN<B> {
    /// Create a new MLSTM-FCN model.
    pub fn new(config: MLSTMFCNConfig, device: &B::Device) -> Self {
        // Create LSTM layers
        let mut lstms = Vec::new();
        let mut lstm_total_hidden = 0;

        for &hidden_size in &config.lstm_hidden_sizes {
            let lstm = LstmConfig::new(config.n_vars, hidden_size, false).init(device);
            lstms.push(lstm);
            lstm_total_hidden += hidden_size;
        }

        // Squeeze-and-excitation attention on concatenated LSTM outputs
        let se_attention = if config.use_attention {
            Some(SqueezeExcitation::new(
                lstm_total_hidden,
                config.se_reduction,
                device,
            ))
        } else {
            None
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

        // Classifier input = LSTM outputs + FCN output
        let fcn_out_size = *config.fcn_filters.last().unwrap_or(&128);
        let classifier_in = lstm_total_hidden + fcn_out_size;
        let classifier = LinearConfig::new(classifier_in, config.n_classes).init(device);

        Self {
            lstms,
            se_attention,
            fcn_blocks,
            gap,
            dropout,
            classifier,
            lstm_total_hidden,
            fcn_out_size,
        }
    }

    /// Forward pass.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, _n_vars, seq_len] = x.dims();

        // LSTM branch: expects (batch, seq_len, features)
        let x_lstm = x.clone().swap_dims(1, 2); // (B, L, V)

        // Process through all LSTM layers and concatenate outputs
        let mut lstm_outputs = Vec::new();

        for lstm in &self.lstms {
            let (output, _) = lstm.forward(x_lstm.clone(), None);
            // Take last time step: (B, L, H) -> (B, H)
            let last_idx = seq_len - 1;
            let hidden_size = output.dims()[2];
            let last_hidden = output
                .slice([0..batch_size, last_idx..last_idx + 1, 0..hidden_size])
                .reshape([batch_size, hidden_size]);
            lstm_outputs.push(last_hidden);
        }

        // Concatenate all LSTM outputs
        let lstm_concat = Tensor::cat(lstm_outputs, 1); // (B, total_hidden)

        // Apply squeeze-and-excitation attention
        let lstm_out = if let Some(ref se) = self.se_attention {
            se.forward(lstm_concat)
        } else {
            lstm_concat
        };

        let lstm_out = self.dropout.forward(lstm_out);

        // FCN branch
        let mut fcn_out = x;
        for block in &self.fcn_blocks {
            fcn_out = block.forward(fcn_out);
        }

        // Global average pooling: (B, C, L) -> (B, C, 1) -> (B, C)
        let fcn_out = self.gap.forward(fcn_out);
        let fcn_out = fcn_out.reshape([batch_size, self.fcn_out_size]);

        // Concatenate LSTM and FCN features
        let combined = Tensor::cat(vec![lstm_out, fcn_out], 1);

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
    fn test_mlstmfcn_config_default() {
        let config = MLSTMFCNConfig::default();
        assert_eq!(config.lstm_hidden_sizes, vec![128]);
        assert_eq!(config.fcn_filters, vec![128, 256, 128]);
        assert!(config.use_attention);
    }

    #[test]
    fn test_mlstmfcn_config_new() {
        let config = MLSTMFCNConfig::new(3, 200, 10);
        assert_eq!(config.n_vars, 3);
        assert_eq!(config.seq_len, 200);
        assert_eq!(config.n_classes, 10);
    }

    #[test]
    fn test_mlstmfcn_config_builder() {
        let config = MLSTMFCNConfig::new(3, 100, 5)
            .with_lstm_hidden_sizes(vec![64, 64, 64])
            .with_fcn_filters(vec![64, 128, 64])
            .with_dropout(0.5)
            .with_se_reduction(8);

        assert_eq!(config.lstm_hidden_sizes, vec![64, 64, 64]);
        assert_eq!(config.fcn_filters, vec![64, 128, 64]);
        assert_eq!(config.dropout, 0.5);
        assert_eq!(config.se_reduction, 8);
    }
}
