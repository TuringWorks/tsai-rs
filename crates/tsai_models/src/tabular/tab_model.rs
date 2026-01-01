//! TabModel: Simple tabular model for classification.
//!
//! A straightforward MLP-based model for tabular data with support
//! for both continuous and categorical features.

use burn::nn::{
    BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Embedding, EmbeddingConfig,
    Linear, LinearConfig, Relu,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use serde::{Deserialize, Serialize};

/// Configuration for TabModel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TabModelConfig {
    /// Number of continuous features.
    pub n_continuous: usize,
    /// Number of categorical features.
    pub n_categorical: usize,
    /// Cardinalities for each categorical feature.
    pub cat_cardinalities: Vec<usize>,
    /// Embedding dimension for categorical features.
    pub embed_dim: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// Hidden layer sizes.
    pub hidden_sizes: Vec<usize>,
    /// Dropout rate.
    pub dropout: f64,
    /// Whether to use batch normalization.
    pub use_bn: bool,
}

impl Default for TabModelConfig {
    fn default() -> Self {
        Self {
            n_continuous: 10,
            n_categorical: 5,
            cat_cardinalities: vec![10, 20, 30, 40, 50],
            embed_dim: 8,
            n_classes: 2,
            hidden_sizes: vec![200, 100],
            dropout: 0.1,
            use_bn: true,
        }
    }
}

impl TabModelConfig {
    /// Create a new config.
    pub fn new(n_continuous: usize, n_categorical: usize, n_classes: usize) -> Self {
        Self {
            n_continuous,
            n_categorical,
            n_classes,
            ..Default::default()
        }
    }

    /// Set categorical cardinalities.
    #[must_use]
    pub fn with_cardinalities(mut self, cardinalities: Vec<usize>) -> Self {
        self.cat_cardinalities = cardinalities;
        self
    }

    /// Set embedding dimension.
    #[must_use]
    pub fn with_embed_dim(mut self, embed_dim: usize) -> Self {
        self.embed_dim = embed_dim;
        self
    }

    /// Set hidden layer sizes.
    #[must_use]
    pub fn with_hidden_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.hidden_sizes = sizes;
        self
    }

    /// Set dropout rate.
    #[must_use]
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Set whether to use batch normalization.
    #[must_use]
    pub fn with_bn(mut self, use_bn: bool) -> Self {
        self.use_bn = use_bn;
        self
    }

    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> TabModel<B> {
        TabModel::new(self.clone(), device)
    }
}

/// MLP block with optional batch normalization.
#[derive(Module, Debug)]
struct TabMLPBlock<B: Backend> {
    linear: Linear<B>,
    bn: Option<BatchNorm<B, 1>>,
    dropout: Dropout,
}

impl<B: Backend> TabMLPBlock<B> {
    fn new(in_features: usize, out_features: usize, dropout: f64, use_bn: bool, device: &B::Device) -> Self {
        let linear = LinearConfig::new(in_features, out_features).init(device);
        let bn = if use_bn {
            Some(BatchNormConfig::new(out_features).init(device))
        } else {
            None
        };
        let dropout_layer = DropoutConfig::new(dropout).init();

        Self { linear, bn, dropout: dropout_layer }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let out = self.linear.forward(x);
        let out = Relu::new().forward(out);
        let out = if let Some(ref bn) = self.bn {
            let [batch, features] = out.dims();
            let out = out.reshape([batch, features, 1]);
            let out = bn.forward(out);
            out.reshape([batch, features])
        } else {
            out
        };
        self.dropout.forward(out)
    }
}

/// TabModel: Simple MLP-based tabular model.
///
/// Combines categorical embeddings with continuous features and
/// passes through an MLP for classification.
///
/// # Architecture
///
/// ```text
/// Continuous (B, N_cont)     Categorical (B, N_cat)
///       |                          |
///       |                          +---> [Embeddings] -> Flatten
///       |                                   |
///       +-------------> Concatenate <-------+
///                            |
///                            v
///                  [MLP Block 1] -> [MLP Block 2] -> ...
///                            |
///                            v
///                       [Linear]
///                            |
///                      Output (B, n_classes)
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use tsai_models::tabular::{TabModel, TabModelConfig};
///
/// let config = TabModelConfig::new(10, 5, 3)
///     .with_cardinalities(vec![10, 20, 30, 40, 50])
///     .with_hidden_sizes(vec![256, 128, 64])
///     .with_embed_dim(16);
/// let model = config.init::<NdArray>(&device);
/// ```
#[derive(Module, Debug)]
pub struct TabModel<B: Backend> {
    /// Embeddings for categorical features.
    cat_embeddings: Vec<Embedding<B>>,
    /// Continuous feature normalization.
    cont_bn: Option<BatchNorm<B, 1>>,
    /// MLP blocks.
    blocks: Vec<TabMLPBlock<B>>,
    /// Output layer.
    head: Linear<B>,
    /// Embedding dimension.
    #[module(skip)]
    embed_dim: usize,
    /// Number of categorical features.
    #[module(skip)]
    n_categorical: usize,
    /// Number of continuous features.
    #[module(skip)]
    n_continuous: usize,
}

impl<B: Backend> TabModel<B> {
    /// Create a new TabModel.
    pub fn new(config: TabModelConfig, device: &B::Device) -> Self {
        // Create embeddings for each categorical feature
        let cat_embeddings: Vec<_> = config
            .cat_cardinalities
            .iter()
            .take(config.n_categorical)
            .map(|&card| EmbeddingConfig::new(card, config.embed_dim).init(device))
            .collect();

        // Optional BN for continuous features
        let cont_bn = if config.use_bn && config.n_continuous > 0 {
            Some(BatchNormConfig::new(config.n_continuous).init(device))
        } else {
            None
        };

        // Calculate input size: continuous + (n_categorical * embed_dim)
        let input_size = config.n_continuous + config.n_categorical * config.embed_dim;

        // Build MLP blocks
        let mut blocks = Vec::new();
        let mut prev_size = input_size;
        for &hidden_size in &config.hidden_sizes {
            blocks.push(TabMLPBlock::new(prev_size, hidden_size, config.dropout, config.use_bn, device));
            prev_size = hidden_size;
        }

        // Output layer
        let head = LinearConfig::new(prev_size, config.n_classes).init(device);

        Self {
            cat_embeddings,
            cont_bn,
            blocks,
            head,
            embed_dim: config.embed_dim,
            n_categorical: config.n_categorical,
            n_continuous: config.n_continuous,
        }
    }

    /// Forward pass.
    ///
    /// # Arguments
    ///
    /// * `x_continuous` - Continuous features (batch, n_continuous)
    /// * `x_categorical` - Categorical features (batch, n_categorical) as indices
    pub fn forward(
        &self,
        x_continuous: Tensor<B, 2>,
        x_categorical: Tensor<B, 2, Int>,
    ) -> Tensor<B, 2> {
        let [batch, _] = x_continuous.dims();

        // Normalize continuous features
        let cont = if let Some(ref bn) = self.cont_bn {
            let reshaped = x_continuous.reshape([batch, self.n_continuous, 1]);
            let normed = bn.forward(reshaped);
            normed.reshape([batch, self.n_continuous])
        } else {
            x_continuous
        };

        // Embed categorical features
        let mut cat_embeds = Vec::new();
        for (i, embedding) in self.cat_embeddings.iter().enumerate() {
            if i < self.n_categorical {
                let cat_col = x_categorical.clone().slice([0..batch, i..(i + 1)]);
                let embedded = embedding.forward(cat_col); // (batch, 1, embed_dim)
                let embedded = embedded.reshape([batch, self.embed_dim]);
                cat_embeds.push(embedded);
            }
        }

        // Concatenate all features
        let mut features = vec![cont];
        features.extend(cat_embeds);
        let combined = Tensor::cat(features, 1);

        // Apply MLP blocks
        let mut out = combined;
        for block in &self.blocks {
            out = block.forward(out);
        }

        // Output
        self.head.forward(out)
    }

    /// Forward pass returning probabilities.
    pub fn forward_probs(
        &self,
        x_continuous: Tensor<B, 2>,
        x_categorical: Tensor<B, 2, Int>,
    ) -> Tensor<B, 2> {
        let logits = self.forward(x_continuous, x_categorical);
        softmax(logits, 1)
    }

    /// Forward pass for continuous-only data.
    pub fn forward_continuous(&self, x_continuous: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch, _] = x_continuous.dims();

        // Normalize continuous features
        let cont = if let Some(ref bn) = self.cont_bn {
            let reshaped = x_continuous.reshape([batch, self.n_continuous, 1]);
            let normed = bn.forward(reshaped);
            normed.reshape([batch, self.n_continuous])
        } else {
            x_continuous
        };

        // Apply MLP blocks (no categorical features)
        let mut out = cont;
        for block in &self.blocks {
            out = block.forward(out);
        }

        // Output
        self.head.forward(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tab_model_config_default() {
        let config = TabModelConfig::default();
        assert_eq!(config.n_continuous, 10);
        assert_eq!(config.n_categorical, 5);
        assert_eq!(config.embed_dim, 8);
        assert!(config.use_bn);
    }

    #[test]
    fn test_tab_model_config_new() {
        let config = TabModelConfig::new(20, 8, 10);
        assert_eq!(config.n_continuous, 20);
        assert_eq!(config.n_categorical, 8);
        assert_eq!(config.n_classes, 10);
    }

    #[test]
    fn test_tab_model_config_builder() {
        let config = TabModelConfig::new(10, 5, 3)
            .with_cardinalities(vec![5, 10, 15, 20, 25])
            .with_embed_dim(16)
            .with_hidden_sizes(vec![128, 64])
            .with_dropout(0.2)
            .with_bn(false);

        assert_eq!(config.cat_cardinalities, vec![5, 10, 15, 20, 25]);
        assert_eq!(config.embed_dim, 16);
        assert_eq!(config.hidden_sizes, vec![128, 64]);
        assert_eq!(config.dropout, 0.2);
        assert!(!config.use_bn);
    }
}
