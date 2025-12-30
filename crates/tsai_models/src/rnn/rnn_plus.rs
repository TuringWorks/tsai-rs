//! RNNPlus model for time series.

use burn::nn::{
    Dropout, DropoutConfig, Linear, LinearConfig, Lstm, LstmConfig,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use serde::{Deserialize, Serialize};

/// Type of RNN cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RNNType {
    /// Long Short-Term Memory.
    LSTM,
    /// Gated Recurrent Unit.
    GRU,
}

impl Default for RNNType {
    fn default() -> Self {
        Self::LSTM
    }
}

/// Configuration for RNNPlus model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RNNPlusConfig {
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
    /// Dropout rate.
    pub dropout: f64,
}

impl Default for RNNPlusConfig {
    fn default() -> Self {
        Self {
            n_vars: 1,
            seq_len: 100,
            n_classes: 2,
            hidden_size: 128,
            n_layers: 2,
            rnn_type: RNNType::LSTM,
            bidirectional: false,
            dropout: 0.1,
        }
    }
}

impl RNNPlusConfig {
    /// Create a new config.
    pub fn new(n_vars: usize, seq_len: usize, n_classes: usize) -> Self {
        Self {
            n_vars,
            seq_len,
            n_classes,
            ..Default::default()
        }
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
    pub fn init<B: Backend>(&self, device: &B::Device) -> RNNPlus<B> {
        RNNPlus::new(self.clone(), device)
    }
}

/// RNNPlus model for time series classification.
#[derive(Module, Debug)]
pub struct RNNPlus<B: Backend> {
    /// LSTM layers.
    lstm: Lstm<B>,
    /// Dropout layer.
    dropout: Dropout,
    /// Final classifier.
    fc: Linear<B>,
}

impl<B: Backend> RNNPlus<B> {
    /// Create a new RNNPlus model.
    pub fn new(config: RNNPlusConfig, device: &B::Device) -> Self {
        // Note: Currently only LSTM is supported. GRU would require additional Burn support.
        let lstm = LstmConfig::new(config.n_vars, config.hidden_size, config.bidirectional)
            .init(device);

        let output_dim = config.output_dim();
        let dropout = DropoutConfig::new(config.dropout).init();
        let fc = LinearConfig::new(output_dim, config.n_classes).init(device);

        Self { lstm, dropout, fc }
    }

    /// Forward pass.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch, _n_vars, seq_len] = x.dims();

        // Transpose to (batch, seq_len, n_vars) for RNN
        let x = x.swap_dims(1, 2);

        // Apply LSTM
        let (output, _) = self.lstm.forward(x, None);

        // Get output dimension from tensor
        let [_, _, hidden_dim] = output.dims();

        // Take last timestep output
        let last_output = output.slice([0..batch, (seq_len - 1)..seq_len, 0..hidden_dim]);
        let last_output = last_output.reshape([batch, hidden_dim]);

        // Apply dropout and classify
        let output = self.dropout.forward(last_output);
        self.fc.forward(output)
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
    fn test_rnn_config() {
        let config = RNNPlusConfig::default();
        assert_eq!(config.hidden_size, 128);
        assert_eq!(config.rnn_type, RNNType::LSTM);
    }

    #[test]
    fn test_output_dim() {
        let config = RNNPlusConfig::default();
        assert_eq!(config.output_dim(), 128);

        let config_bi = RNNPlusConfig {
            bidirectional: true,
            ..Default::default()
        };
        assert_eq!(config_bi.output_dim(), 256);
    }
}
