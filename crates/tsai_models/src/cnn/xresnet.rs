//! XResNet1d: XResNet adapted for 1D time series.
//!
//! Implements the XResNet architecture with Bag of Tricks improvements
//! adapted for time series classification.
//!
//! Reference: "Bag of Tricks for Image Classification with CNNs" (He et al., 2019)

use burn::nn::{
    conv::{Conv1d, Conv1dConfig},
    pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig, MaxPool1d, MaxPool1dConfig},
    BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, Relu,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use serde::{Deserialize, Serialize};

/// Configuration for XResNet1d model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XResNet1dConfig {
    /// Number of input variables/channels.
    pub n_vars: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// Base number of filters.
    pub base_filters: usize,
    /// Expansion factor for bottleneck blocks.
    pub expansion: usize,
    /// Number of blocks in each stage.
    pub layers: Vec<usize>,
    /// Dropout rate.
    pub dropout: f64,
    /// Whether to use squeeze-and-excitation.
    pub use_se: bool,
    /// SE reduction ratio.
    pub se_reduction: usize,
}

impl Default for XResNet1dConfig {
    fn default() -> Self {
        Self {
            n_vars: 1,
            seq_len: 100,
            n_classes: 2,
            base_filters: 64,
            expansion: 1,
            layers: vec![2, 2, 2, 2], // ResNet18-like
            dropout: 0.0,
            use_se: false,
            se_reduction: 16,
        }
    }
}

impl XResNet1dConfig {
    /// Create a new config.
    pub fn new(n_vars: usize, seq_len: usize, n_classes: usize) -> Self {
        Self {
            n_vars,
            seq_len,
            n_classes,
            ..Default::default()
        }
    }

    /// Create XResNet18 config.
    pub fn xresnet18(n_vars: usize, seq_len: usize, n_classes: usize) -> Self {
        Self::new(n_vars, seq_len, n_classes)
            .with_layers(vec![2, 2, 2, 2])
            .with_expansion(1)
    }

    /// Create XResNet34 config.
    pub fn xresnet34(n_vars: usize, seq_len: usize, n_classes: usize) -> Self {
        Self::new(n_vars, seq_len, n_classes)
            .with_layers(vec![3, 4, 6, 3])
            .with_expansion(1)
    }

    /// Create XResNet50 config.
    pub fn xresnet50(n_vars: usize, seq_len: usize, n_classes: usize) -> Self {
        Self::new(n_vars, seq_len, n_classes)
            .with_layers(vec![3, 4, 6, 3])
            .with_expansion(4)
    }

    /// Set number of blocks in each stage.
    #[must_use]
    pub fn with_layers(mut self, layers: Vec<usize>) -> Self {
        self.layers = layers;
        self
    }

    /// Set expansion factor.
    #[must_use]
    pub fn with_expansion(mut self, expansion: usize) -> Self {
        self.expansion = expansion;
        self
    }

    /// Set base filters.
    #[must_use]
    pub fn with_base_filters(mut self, filters: usize) -> Self {
        self.base_filters = filters;
        self
    }

    /// Set dropout rate.
    #[must_use]
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Enable squeeze-and-excitation.
    #[must_use]
    pub fn with_se(mut self, use_se: bool, reduction: usize) -> Self {
        self.use_se = use_se;
        self.se_reduction = reduction;
        self
    }

    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> XResNet1d<B> {
        XResNet1d::new(self.clone(), device)
    }
}

/// Squeeze-and-Excitation block for channel attention.
#[derive(Module, Debug)]
struct SEBlock<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    #[module(skip)]
    channels: usize,
}

impl<B: Backend> SEBlock<B> {
    fn new(channels: usize, reduction: usize, device: &B::Device) -> Self {
        let reduced = (channels / reduction).max(1);
        let fc1 = LinearConfig::new(channels, reduced).init(device);
        let fc2 = LinearConfig::new(reduced, channels).init(device);
        Self { fc1, fc2, channels }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, channels, _seq_len] = x.dims();

        // Global average pooling
        let squeezed = x.clone().mean_dim(2).reshape([batch, channels]);

        // FC layers with ReLU and Sigmoid
        let out = self.fc1.forward(squeezed);
        let out = Relu::new().forward(out);
        let out = self.fc2.forward(out);
        let out = burn::tensor::activation::sigmoid(out);

        // Reshape and apply attention
        let attention = out.reshape([batch, channels, 1]);
        x * attention
    }
}

/// XResNet basic block (for expansion=1).
#[derive(Module, Debug)]
struct XResNetBasicBlock<B: Backend> {
    conv1: Conv1d<B>,
    bn1: BatchNorm<B, 1>,
    conv2: Conv1d<B>,
    bn2: BatchNorm<B, 1>,
    se: Option<SEBlock<B>>,
    downsample_conv: Option<Conv1d<B>>,
    downsample_bn: Option<BatchNorm<B, 1>>,
    #[module(skip)]
    stride: usize,
}

impl<B: Backend> XResNetBasicBlock<B> {
    fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        use_se: bool,
        se_reduction: usize,
        device: &B::Device,
    ) -> Self {
        let conv1 = Conv1dConfig::new(in_channels, out_channels, 3)
            .with_stride(stride)
            .with_padding(burn::nn::PaddingConfig1d::Same)
            .init(device);
        let bn1 = BatchNormConfig::new(out_channels).init(device);

        let conv2 = Conv1dConfig::new(out_channels, out_channels, 3)
            .with_padding(burn::nn::PaddingConfig1d::Same)
            .init(device);
        let bn2 = BatchNormConfig::new(out_channels).init(device);

        let se = if use_se {
            Some(SEBlock::new(out_channels, se_reduction, device))
        } else {
            None
        };

        // Downsample path if needed
        let (downsample_conv, downsample_bn) = if stride != 1 || in_channels != out_channels {
            let conv = Conv1dConfig::new(in_channels, out_channels, 1)
                .with_stride(stride)
                .init(device);
            let bn = BatchNormConfig::new(out_channels).init(device);
            (Some(conv), Some(bn))
        } else {
            (None, None)
        };

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            se,
            downsample_conv,
            downsample_bn,
            stride,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let identity = x.clone();

        // Main path
        let out = self.conv1.forward(x);
        let out = self.bn1.forward(out);
        let out = Relu::new().forward(out);

        let out = self.conv2.forward(out);
        let out = self.bn2.forward(out);

        // SE attention
        let out = if let Some(ref se) = self.se {
            se.forward(out)
        } else {
            out
        };

        // Downsample identity if needed
        let identity = if let (Some(ref conv), Some(ref bn)) = (&self.downsample_conv, &self.downsample_bn) {
            bn.forward(conv.forward(identity))
        } else {
            identity
        };

        // Residual connection
        let out = out + identity;
        Relu::new().forward(out)
    }
}

/// XResNet bottleneck block (for expansion>1).
#[derive(Module, Debug)]
struct XResNetBottleneck<B: Backend> {
    conv1: Conv1d<B>,
    bn1: BatchNorm<B, 1>,
    conv2: Conv1d<B>,
    bn2: BatchNorm<B, 1>,
    conv3: Conv1d<B>,
    bn3: BatchNorm<B, 1>,
    se: Option<SEBlock<B>>,
    downsample_conv: Option<Conv1d<B>>,
    downsample_bn: Option<BatchNorm<B, 1>>,
    #[module(skip)]
    stride: usize,
    #[module(skip)]
    expansion: usize,
}

impl<B: Backend> XResNetBottleneck<B> {
    fn new(
        in_channels: usize,
        base_channels: usize,
        stride: usize,
        expansion: usize,
        use_se: bool,
        se_reduction: usize,
        device: &B::Device,
    ) -> Self {
        let out_channels = base_channels * expansion;

        // 1x1 reduce
        let conv1 = Conv1dConfig::new(in_channels, base_channels, 1).init(device);
        let bn1 = BatchNormConfig::new(base_channels).init(device);

        // 3x3 conv
        let conv2 = Conv1dConfig::new(base_channels, base_channels, 3)
            .with_stride(stride)
            .with_padding(burn::nn::PaddingConfig1d::Same)
            .init(device);
        let bn2 = BatchNormConfig::new(base_channels).init(device);

        // 1x1 expand
        let conv3 = Conv1dConfig::new(base_channels, out_channels, 1).init(device);
        let bn3 = BatchNormConfig::new(out_channels).init(device);

        let se = if use_se {
            Some(SEBlock::new(out_channels, se_reduction, device))
        } else {
            None
        };

        // Downsample path if needed
        let (downsample_conv, downsample_bn) = if stride != 1 || in_channels != out_channels {
            let conv = Conv1dConfig::new(in_channels, out_channels, 1)
                .with_stride(stride)
                .init(device);
            let bn = BatchNormConfig::new(out_channels).init(device);
            (Some(conv), Some(bn))
        } else {
            (None, None)
        };

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            se,
            downsample_conv,
            downsample_bn,
            stride,
            expansion,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let identity = x.clone();

        // Main path
        let out = self.conv1.forward(x);
        let out = self.bn1.forward(out);
        let out = Relu::new().forward(out);

        let out = self.conv2.forward(out);
        let out = self.bn2.forward(out);
        let out = Relu::new().forward(out);

        let out = self.conv3.forward(out);
        let out = self.bn3.forward(out);

        // SE attention
        let out = if let Some(ref se) = self.se {
            se.forward(out)
        } else {
            out
        };

        // Downsample identity if needed
        let identity = if let (Some(ref conv), Some(ref bn)) = (&self.downsample_conv, &self.downsample_bn) {
            bn.forward(conv.forward(identity))
        } else {
            identity
        };

        // Residual connection
        let out = out + identity;
        Relu::new().forward(out)
    }
}

/// XResNet1d for time series classification.
///
/// Implements XResNet with Bag of Tricks improvements:
/// - ResNet-C: 3x3 stem instead of 7x7
/// - ResNet-D: Average pooling in downsample path
/// - Optional Squeeze-and-Excitation attention
///
/// # Architecture
///
/// ```text
/// Input (B, V, L)
///       |
///       +---> [Stem: 3x3 Conv x3] -> [MaxPool]
///       |
///       +---> [Stage 1: BasicBlock/Bottleneck x N]
///       +---> [Stage 2: BasicBlock/Bottleneck x N]
///       +---> [Stage 3: BasicBlock/Bottleneck x N]
///       +---> [Stage 4: BasicBlock/Bottleneck x N]
///       |
///       +---> [Global Average Pool] -> (B, C)
///       |
///       +---> [Dropout] -> [Linear] -> Output (B, n_classes)
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use tsai_models::cnn::{XResNet1d, XResNet1dConfig};
///
/// // XResNet18 for time series
/// let config = XResNet1dConfig::xresnet18(3, 100, 5);
/// let model = config.init::<NdArray>(&device);
///
/// // XResNet50 with SE attention
/// let config = XResNet1dConfig::xresnet50(3, 100, 5)
///     .with_se(true, 16);
/// let model = config.init::<NdArray>(&device);
/// ```
#[derive(Module, Debug)]
pub struct XResNet1d<B: Backend> {
    /// Stem convolutions.
    stem_conv1: Conv1d<B>,
    stem_bn1: BatchNorm<B, 1>,
    stem_conv2: Conv1d<B>,
    stem_bn2: BatchNorm<B, 1>,
    stem_conv3: Conv1d<B>,
    stem_bn3: BatchNorm<B, 1>,
    /// Max pooling after stem.
    maxpool: MaxPool1d,
    /// Stage 1 blocks (basic or bottleneck).
    stage1_basic: Vec<XResNetBasicBlock<B>>,
    stage1_bottleneck: Vec<XResNetBottleneck<B>>,
    /// Stage 2 blocks.
    stage2_basic: Vec<XResNetBasicBlock<B>>,
    stage2_bottleneck: Vec<XResNetBottleneck<B>>,
    /// Stage 3 blocks.
    stage3_basic: Vec<XResNetBasicBlock<B>>,
    stage3_bottleneck: Vec<XResNetBottleneck<B>>,
    /// Stage 4 blocks.
    stage4_basic: Vec<XResNetBasicBlock<B>>,
    stage4_bottleneck: Vec<XResNetBottleneck<B>>,
    /// Global average pooling.
    gap: AdaptiveAvgPool1d,
    /// Dropout.
    dropout: Dropout,
    /// Classification head.
    head: Linear<B>,
    /// Whether using bottleneck blocks.
    #[module(skip)]
    use_bottleneck: bool,
    /// Final number of channels.
    #[module(skip)]
    final_channels: usize,
}

impl<B: Backend> XResNet1d<B> {
    /// Create a new XResNet1d model.
    pub fn new(config: XResNet1dConfig, device: &B::Device) -> Self {
        let use_bottleneck = config.expansion > 1;
        let base = config.base_filters;

        // Stem: 3x3 convolutions (ResNet-C style)
        let stem_conv1 = Conv1dConfig::new(config.n_vars, base / 2, 3)
            .with_stride(2)
            .with_padding(burn::nn::PaddingConfig1d::Same)
            .init(device);
        let stem_bn1 = BatchNormConfig::new(base / 2).init(device);

        let stem_conv2 = Conv1dConfig::new(base / 2, base / 2, 3)
            .with_padding(burn::nn::PaddingConfig1d::Same)
            .init(device);
        let stem_bn2 = BatchNormConfig::new(base / 2).init(device);

        let stem_conv3 = Conv1dConfig::new(base / 2, base, 3)
            .with_padding(burn::nn::PaddingConfig1d::Same)
            .init(device);
        let stem_bn3 = BatchNormConfig::new(base).init(device);

        let maxpool = MaxPool1dConfig::new(3).with_stride(2).with_padding(burn::nn::PaddingConfig1d::Same).init();

        // Build stages
        let filters = [base, base * 2, base * 4, base * 8];
        let strides = [1, 2, 2, 2];

        let mut stage1_basic = Vec::new();
        let mut stage1_bottleneck = Vec::new();
        let mut stage2_basic = Vec::new();
        let mut stage2_bottleneck = Vec::new();
        let mut stage3_basic = Vec::new();
        let mut stage3_bottleneck = Vec::new();
        let mut stage4_basic = Vec::new();
        let mut stage4_bottleneck = Vec::new();

        let mut in_channels = base;

        for (stage_idx, &n_blocks) in config.layers.iter().enumerate() {
            let out_channels = filters[stage_idx] * config.expansion;
            let stride = strides[stage_idx];

            for block_idx in 0..n_blocks {
                let block_stride = if block_idx == 0 { stride } else { 1 };
                let block_in = if block_idx == 0 { in_channels } else { out_channels };

                if use_bottleneck {
                    let block = XResNetBottleneck::new(
                        block_in,
                        filters[stage_idx],
                        block_stride,
                        config.expansion,
                        config.use_se,
                        config.se_reduction,
                        device,
                    );
                    match stage_idx {
                        0 => stage1_bottleneck.push(block),
                        1 => stage2_bottleneck.push(block),
                        2 => stage3_bottleneck.push(block),
                        3 => stage4_bottleneck.push(block),
                        _ => {}
                    }
                } else {
                    let block = XResNetBasicBlock::new(
                        block_in,
                        filters[stage_idx],
                        block_stride,
                        config.use_se,
                        config.se_reduction,
                        device,
                    );
                    match stage_idx {
                        0 => stage1_basic.push(block),
                        1 => stage2_basic.push(block),
                        2 => stage3_basic.push(block),
                        3 => stage4_basic.push(block),
                        _ => {}
                    }
                }
            }
            in_channels = out_channels;
        }

        let final_channels = filters[config.layers.len() - 1] * config.expansion;

        let gap = AdaptiveAvgPool1dConfig::new(1).init();
        let dropout = DropoutConfig::new(config.dropout).init();
        let head = LinearConfig::new(final_channels, config.n_classes).init(device);

        Self {
            stem_conv1,
            stem_bn1,
            stem_conv2,
            stem_bn2,
            stem_conv3,
            stem_bn3,
            maxpool,
            stage1_basic,
            stage1_bottleneck,
            stage2_basic,
            stage2_bottleneck,
            stage3_basic,
            stage3_bottleneck,
            stage4_basic,
            stage4_bottleneck,
            gap,
            dropout,
            head,
            use_bottleneck,
            final_channels,
        }
    }

    /// Forward pass.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, _, _] = x.dims();

        // Stem
        let out = self.stem_conv1.forward(x);
        let out = self.stem_bn1.forward(out);
        let out = Relu::new().forward(out);

        let out = self.stem_conv2.forward(out);
        let out = self.stem_bn2.forward(out);
        let out = Relu::new().forward(out);

        let out = self.stem_conv3.forward(out);
        let out = self.stem_bn3.forward(out);
        let out = Relu::new().forward(out);

        let out = self.maxpool.forward(out);

        // Stages
        let out = if self.use_bottleneck {
            let mut out = out;
            for block in &self.stage1_bottleneck {
                out = block.forward(out);
            }
            for block in &self.stage2_bottleneck {
                out = block.forward(out);
            }
            for block in &self.stage3_bottleneck {
                out = block.forward(out);
            }
            for block in &self.stage4_bottleneck {
                out = block.forward(out);
            }
            out
        } else {
            let mut out = out;
            for block in &self.stage1_basic {
                out = block.forward(out);
            }
            for block in &self.stage2_basic {
                out = block.forward(out);
            }
            for block in &self.stage3_basic {
                out = block.forward(out);
            }
            for block in &self.stage4_basic {
                out = block.forward(out);
            }
            out
        };

        // Global average pooling
        let out = self.gap.forward(out);
        let out = out.reshape([batch_size, self.final_channels]);

        // Classification
        let out = self.dropout.forward(out);
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
    fn test_xresnet_config_default() {
        let config = XResNet1dConfig::default();
        assert_eq!(config.base_filters, 64);
        assert_eq!(config.expansion, 1);
        assert_eq!(config.layers, vec![2, 2, 2, 2]);
    }

    #[test]
    fn test_xresnet18_config() {
        let config = XResNet1dConfig::xresnet18(3, 100, 5);
        assert_eq!(config.n_vars, 3);
        assert_eq!(config.seq_len, 100);
        assert_eq!(config.n_classes, 5);
        assert_eq!(config.layers, vec![2, 2, 2, 2]);
        assert_eq!(config.expansion, 1);
    }

    #[test]
    fn test_xresnet50_config() {
        let config = XResNet1dConfig::xresnet50(3, 100, 10);
        assert_eq!(config.layers, vec![3, 4, 6, 3]);
        assert_eq!(config.expansion, 4);
    }

    #[test]
    fn test_xresnet_config_builder() {
        let config = XResNet1dConfig::new(3, 100, 5)
            .with_base_filters(32)
            .with_layers(vec![1, 1, 1, 1])
            .with_dropout(0.2)
            .with_se(true, 8);

        assert_eq!(config.base_filters, 32);
        assert_eq!(config.layers, vec![1, 1, 1, 1]);
        assert_eq!(config.dropout, 0.2);
        assert!(config.use_se);
        assert_eq!(config.se_reduction, 8);
    }
}
