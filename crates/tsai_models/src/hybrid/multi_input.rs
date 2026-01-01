//! MultiInputNet: Multi-modal model combining time series with tabular data.
//!
//! Combines a time series backbone with tabular features (continuous + categorical)
//! using a flexible fusion strategy.

use burn::nn::{
    conv::{Conv1d, Conv1dConfig},
    pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig},
    BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear,
    LinearConfig, Relu,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use serde::{Deserialize, Serialize};

/// Backbone type for time series feature extraction.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum BackboneType {
    /// Simple CNN backbone.
    CNN,
    /// ResNet-style backbone with residual connections.
    ResNet,
    /// FCN (Fully Convolutional Network) backbone.
    FCN,
}

impl Default for BackboneType {
    fn default() -> Self {
        Self::ResNet
    }
}

/// Fusion strategy for combining modalities.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum FusionType {
    /// Concatenate features from all modalities.
    Concat,
    /// Add features (requires same dimensions).
    Add,
    /// Gated fusion with learnable weights.
    Gated,
}

impl Default for FusionType {
    fn default() -> Self {
        Self::Concat
    }
}

/// Configuration for MultiInputNet model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiInputNetConfig {
    /// Number of time series input variables.
    pub n_ts_vars: usize,
    /// Time series sequence length.
    pub ts_seq_len: usize,
    /// Number of continuous tabular features.
    pub n_continuous: usize,
    /// Number of categorical tabular features.
    pub n_categorical: usize,
    /// Cardinalities for each categorical feature.
    pub cat_cardinalities: Vec<usize>,
    /// Embedding dimension for categorical features.
    pub cat_embed_dim: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// Time series backbone type.
    pub backbone: BackboneType,
    /// Fusion type for combining modalities.
    pub fusion: FusionType,
    /// Number of filters in backbone layers.
    pub backbone_filters: Vec<usize>,
    /// Hidden dimension for tabular MLP.
    pub tab_hidden_dim: usize,
    /// Final hidden dimension before classification.
    pub final_hidden_dim: usize,
    /// Dropout rate.
    pub dropout: f64,
}

impl Default for MultiInputNetConfig {
    fn default() -> Self {
        Self {
            n_ts_vars: 1,
            ts_seq_len: 100,
            n_continuous: 10,
            n_categorical: 5,
            cat_cardinalities: vec![10, 20, 30, 40, 50],
            cat_embed_dim: 8,
            n_classes: 2,
            backbone: BackboneType::default(),
            fusion: FusionType::default(),
            backbone_filters: vec![64, 128, 256],
            tab_hidden_dim: 128,
            final_hidden_dim: 256,
            dropout: 0.1,
        }
    }
}

impl MultiInputNetConfig {
    /// Create a new config.
    pub fn new(n_ts_vars: usize, ts_seq_len: usize, n_classes: usize) -> Self {
        Self {
            n_ts_vars,
            ts_seq_len,
            n_classes,
            ..Default::default()
        }
    }

    /// Set number of continuous features.
    #[must_use]
    pub fn with_n_continuous(mut self, n: usize) -> Self {
        self.n_continuous = n;
        self
    }

    /// Set categorical features.
    #[must_use]
    pub fn with_categorical(mut self, n_categorical: usize, cardinalities: Vec<usize>) -> Self {
        self.n_categorical = n_categorical;
        self.cat_cardinalities = cardinalities;
        self
    }

    /// Set categorical embedding dimension.
    #[must_use]
    pub fn with_cat_embed_dim(mut self, dim: usize) -> Self {
        self.cat_embed_dim = dim;
        self
    }

    /// Set backbone type.
    #[must_use]
    pub fn with_backbone(mut self, backbone: BackboneType) -> Self {
        self.backbone = backbone;
        self
    }

    /// Set fusion type.
    #[must_use]
    pub fn with_fusion(mut self, fusion: FusionType) -> Self {
        self.fusion = fusion;
        self
    }

    /// Set backbone filters.
    #[must_use]
    pub fn with_backbone_filters(mut self, filters: Vec<usize>) -> Self {
        self.backbone_filters = filters;
        self
    }

    /// Set tabular hidden dimension.
    #[must_use]
    pub fn with_tab_hidden_dim(mut self, dim: usize) -> Self {
        self.tab_hidden_dim = dim;
        self
    }

    /// Set final hidden dimension.
    #[must_use]
    pub fn with_final_hidden_dim(mut self, dim: usize) -> Self {
        self.final_hidden_dim = dim;
        self
    }

    /// Set dropout rate.
    #[must_use]
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiInputNet<B> {
        MultiInputNet::new(self.clone(), device)
    }
}

/// Residual block for time series.
#[derive(Module, Debug)]
struct ResBlock<B: Backend> {
    conv1: Conv1d<B>,
    bn1: BatchNorm<B, 1>,
    conv2: Conv1d<B>,
    bn2: BatchNorm<B, 1>,
    shortcut: Option<Conv1d<B>>,
}

impl<B: Backend> ResBlock<B> {
    fn new(in_channels: usize, out_channels: usize, device: &B::Device) -> Self {
        let conv1 = Conv1dConfig::new(in_channels, out_channels, 3)
            .with_padding(burn::nn::PaddingConfig1d::Same)
            .with_bias(false)
            .init(device);
        let bn1 = BatchNormConfig::new(out_channels).init(device);

        let conv2 = Conv1dConfig::new(out_channels, out_channels, 3)
            .with_padding(burn::nn::PaddingConfig1d::Same)
            .with_bias(false)
            .init(device);
        let bn2 = BatchNormConfig::new(out_channels).init(device);

        let shortcut = if in_channels != out_channels {
            Some(
                Conv1dConfig::new(in_channels, out_channels, 1)
                    .with_bias(false)
                    .init(device),
            )
        } else {
            None
        };

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            shortcut,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let relu = Relu::new();
        let out = relu.forward(self.bn1.forward(self.conv1.forward(x.clone())));
        let out = self.bn2.forward(self.conv2.forward(out));

        let shortcut = match &self.shortcut {
            Some(sc) => sc.forward(x),
            None => x,
        };

        relu.forward(out + shortcut)
    }
}

/// Simple conv block.
#[derive(Module, Debug)]
struct ConvBlock<B: Backend> {
    conv: Conv1d<B>,
    bn: BatchNorm<B, 1>,
}

impl<B: Backend> ConvBlock<B> {
    fn new(in_channels: usize, out_channels: usize, kernel_size: usize, device: &B::Device) -> Self {
        let conv = Conv1dConfig::new(in_channels, out_channels, kernel_size)
            .with_padding(burn::nn::PaddingConfig1d::Same)
            .init(device);
        let bn = BatchNormConfig::new(out_channels).init(device);
        Self { conv, bn }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        Relu::new().forward(self.bn.forward(self.conv.forward(x)))
    }
}

/// MultiInputNet model for multi-modal time series classification.
///
/// Combines time series features with tabular data (continuous + categorical)
/// using a flexible backbone and fusion strategy.
#[derive(Module, Debug)]
pub struct MultiInputNet<B: Backend> {
    // Time series backbone (ResNet-style)
    ts_res_blocks: Vec<ResBlock<B>>,
    // Time series backbone (CNN-style)
    ts_conv_blocks: Vec<ConvBlock<B>>,
    // Global average pooling
    gap: AdaptiveAvgPool1d,
    // Categorical embeddings
    cat_embeddings: Vec<Embedding<B>>,
    // Tabular MLP
    tab_fc1: Linear<B>,
    tab_fc2: Linear<B>,
    // Fusion layers (for gated fusion)
    fusion_gate: Option<Linear<B>>,
    // Projection to align dimensions (for Add fusion)
    ts_proj: Option<Linear<B>>,
    tab_proj: Option<Linear<B>>,
    // Final layers
    final_fc: Linear<B>,
    head: Linear<B>,
    dropout: Dropout,
    // Config flags (not module parameters)
    #[module(skip)]
    use_resnet: bool,
    #[module(skip)]
    use_gated_fusion: bool,
    #[module(skip)]
    use_add_fusion: bool,
}

impl<B: Backend> MultiInputNet<B> {
    /// Create a new MultiInputNet model.
    pub fn new(config: MultiInputNetConfig, device: &B::Device) -> Self {
        // Build time series backbone
        let mut ts_res_blocks = Vec::new();
        let mut ts_conv_blocks = Vec::new();

        let mut in_channels = config.n_ts_vars;
        for &out_channels in &config.backbone_filters {
            match config.backbone {
                BackboneType::ResNet => {
                    ts_res_blocks.push(ResBlock::new(in_channels, out_channels, device));
                }
                BackboneType::CNN | BackboneType::FCN => {
                    let kernel = if config.backbone == BackboneType::FCN { 8 } else { 3 };
                    ts_conv_blocks.push(ConvBlock::new(in_channels, out_channels, kernel, device));
                }
            }
            in_channels = out_channels;
        }

        let ts_out_dim = *config.backbone_filters.last().unwrap_or(&64);
        let gap = AdaptiveAvgPool1dConfig::new(1).init();

        // Build categorical embeddings
        let cat_embeddings: Vec<_> = config
            .cat_cardinalities
            .iter()
            .map(|&card| EmbeddingConfig::new(card, config.cat_embed_dim).init(device))
            .collect();

        // Tabular input dimension
        let tab_in_dim =
            config.n_continuous + config.n_categorical * config.cat_embed_dim;
        let tab_out_dim = config.tab_hidden_dim;

        let tab_fc1 = LinearConfig::new(tab_in_dim, config.tab_hidden_dim).init(device);
        let tab_fc2 = LinearConfig::new(config.tab_hidden_dim, tab_out_dim).init(device);

        // Fusion layers
        let (fusion_gate, ts_proj, tab_proj, final_in_dim) = match config.fusion {
            FusionType::Concat => (None, None, None, ts_out_dim + tab_out_dim),
            FusionType::Add => {
                // Project both to same dimension
                let proj_dim = config.final_hidden_dim;
                let ts_proj = Some(LinearConfig::new(ts_out_dim, proj_dim).init(device));
                let tab_proj = Some(LinearConfig::new(tab_out_dim, proj_dim).init(device));
                (None, ts_proj, tab_proj, proj_dim)
            }
            FusionType::Gated => {
                let combined_dim = ts_out_dim + tab_out_dim;
                let gate = Some(LinearConfig::new(combined_dim, combined_dim).init(device));
                (gate, None, None, combined_dim)
            }
        };

        let final_fc = LinearConfig::new(final_in_dim, config.final_hidden_dim).init(device);
        let head = LinearConfig::new(config.final_hidden_dim, config.n_classes).init(device);
        let dropout = DropoutConfig::new(config.dropout).init();

        let use_resnet = config.backbone == BackboneType::ResNet;
        let use_gated_fusion = config.fusion == FusionType::Gated;
        let use_add_fusion = config.fusion == FusionType::Add;

        Self {
            ts_res_blocks,
            ts_conv_blocks,
            gap,
            cat_embeddings,
            tab_fc1,
            tab_fc2,
            fusion_gate,
            ts_proj,
            tab_proj,
            final_fc,
            head,
            dropout,
            use_resnet,
            use_gated_fusion,
            use_add_fusion,
        }
    }

    /// Forward pass for time series only (no tabular data).
    pub fn forward_ts_only(&self, ts: Tensor<B, 3>) -> Tensor<B, 2> {
        let ts_features = self.extract_ts_features(ts);
        let out = self.dropout.forward(Relu::new().forward(self.final_fc.forward(ts_features)));
        self.head.forward(out)
    }

    /// Extract time series features.
    fn extract_ts_features(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let mut out = x;

        if self.use_resnet {
            for block in &self.ts_res_blocks {
                out = block.forward(out);
            }
        } else {
            for block in &self.ts_conv_blocks {
                out = block.forward(out);
            }
        }

        // Global average pooling
        let out = self.gap.forward(out);
        let [batch, channels, _] = out.dims();
        out.reshape([batch, channels])
    }

    /// Extract tabular features.
    fn extract_tab_features(
        &self,
        continuous: Tensor<B, 2>,
        categorical: Vec<Tensor<B, 2, burn::tensor::Int>>,
    ) -> Tensor<B, 2> {
        let batch_size = continuous.dims()[0];

        // Embed categorical features
        let mut tab_features = vec![continuous];
        for (i, cat_tensor) in categorical.iter().enumerate() {
            if i < self.cat_embeddings.len() {
                // Embedding expects [batch, seq_len], outputs [batch, seq_len, embed_dim]
                // Since we have single values, seq_len=1, so output is [batch, 1, embed_dim]
                let embedded = self.cat_embeddings[i].forward(cat_tensor.clone());
                // Reshape [batch, 1, embed_dim] -> [batch, embed_dim]
                let [_b, seq, embed_dim] = embedded.dims();
                let embedded = embedded.reshape([batch_size, seq * embed_dim]);
                tab_features.push(embedded);
            }
        }

        // Concatenate all tabular features
        let combined = if tab_features.len() > 1 {
            Tensor::cat(tab_features, 1)
        } else {
            tab_features.into_iter().next().unwrap()
        };

        // MLP
        let out = Relu::new().forward(self.tab_fc1.forward(combined));
        self.dropout.forward(Relu::new().forward(self.tab_fc2.forward(out)))
    }

    /// Forward pass with both time series and tabular data.
    pub fn forward(
        &self,
        ts: Tensor<B, 3>,
        continuous: Tensor<B, 2>,
        categorical: Vec<Tensor<B, 2, burn::tensor::Int>>,
    ) -> Tensor<B, 2> {
        let ts_features = self.extract_ts_features(ts);
        let tab_features = self.extract_tab_features(continuous, categorical);

        // Fuse features
        let fused = if self.use_add_fusion {
            let ts_proj = self.ts_proj.as_ref().unwrap();
            let tab_proj = self.tab_proj.as_ref().unwrap();
            ts_proj.forward(ts_features) + tab_proj.forward(tab_features)
        } else if self.use_gated_fusion {
            let gate = self.fusion_gate.as_ref().unwrap();
            let combined = Tensor::cat(vec![ts_features.clone(), tab_features.clone()], 1);
            let gate_values = burn::tensor::activation::sigmoid(gate.forward(combined.clone()));
            combined * gate_values
        } else {
            // Default: Concat
            Tensor::cat(vec![ts_features, tab_features], 1)
        };

        // Final classification
        let out = self.dropout.forward(Relu::new().forward(self.final_fc.forward(fused)));
        self.head.forward(out)
    }

    /// Forward pass returning probabilities.
    pub fn forward_probs(
        &self,
        ts: Tensor<B, 3>,
        continuous: Tensor<B, 2>,
        categorical: Vec<Tensor<B, 2, burn::tensor::Int>>,
    ) -> Tensor<B, 2> {
        let logits = self.forward(ts, continuous, categorical);
        softmax(logits, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_multi_input_config() {
        let config = MultiInputNetConfig::default();
        assert_eq!(config.backbone, BackboneType::ResNet);
        assert_eq!(config.fusion, FusionType::Concat);
        assert_eq!(config.n_ts_vars, 1);
    }

    #[test]
    fn test_multi_input_forward_ts_only() {
        let device = Default::default();
        let config = MultiInputNetConfig::new(3, 100, 5)
            .with_n_continuous(0)
            .with_categorical(0, vec![]);
        let model: MultiInputNet<TestBackend> = config.init(&device);

        let ts = Tensor::<TestBackend, 3>::zeros([4, 3, 100], &device);
        let out = model.forward_ts_only(ts);
        assert_eq!(out.dims(), [4, 5]);
    }

    #[test]
    fn test_multi_input_fusion_types() {
        let device = Default::default();

        // Test Concat fusion
        let config = MultiInputNetConfig::new(3, 100, 5)
            .with_n_continuous(10)
            .with_categorical(2, vec![5, 10])
            .with_fusion(FusionType::Concat);
        let model: MultiInputNet<TestBackend> = config.init(&device);

        let ts = Tensor::<TestBackend, 3>::zeros([4, 3, 100], &device);
        let cont = Tensor::<TestBackend, 2>::zeros([4, 10], &device);
        // Categorical tensors: [batch, 1] shape with Int type
        let cat = vec![
            Tensor::<TestBackend, 2, burn::tensor::Int>::zeros([4, 1], &device),
            Tensor::<TestBackend, 2, burn::tensor::Int>::zeros([4, 1], &device),
        ];
        let out = model.forward(ts.clone(), cont.clone(), cat.clone());
        assert_eq!(out.dims(), [4, 5]);

        // Test Add fusion
        let config = MultiInputNetConfig::new(3, 100, 5)
            .with_n_continuous(10)
            .with_categorical(2, vec![5, 10])
            .with_fusion(FusionType::Add);
        let model: MultiInputNet<TestBackend> = config.init(&device);
        let out = model.forward(ts.clone(), cont.clone(), cat.clone());
        assert_eq!(out.dims(), [4, 5]);

        // Test Gated fusion
        let config = MultiInputNetConfig::new(3, 100, 5)
            .with_n_continuous(10)
            .with_categorical(2, vec![5, 10])
            .with_fusion(FusionType::Gated);
        let model: MultiInputNet<TestBackend> = config.init(&device);
        let out = model.forward(ts, cont, cat);
        assert_eq!(out.dims(), [4, 5]);
    }
}
