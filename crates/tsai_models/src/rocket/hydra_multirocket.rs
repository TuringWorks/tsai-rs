//! HydraMultiRocketPlus: Combined Hydra and MultiRocket architecture.
//!
//! Combines the best of both approaches:
//! - Hydra's dictionary-based random kernels with multiple dilations
//! - MultiRocket's multiple feature types (PPV, MPV, MIPV, LSPV)
//! - Learnable classification head with dropout
//!
//! Reference: Based on concepts from both HYDRA and MultiRocket papers.

use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig, Relu};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use super::multirocket::FeatureType;

/// Configuration for HydraMultiRocketPlus model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HydraMultiRocketPlusConfig {
    /// Number of input variables/channels.
    pub n_vars: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// Number of kernel groups.
    pub n_groups: usize,
    /// Number of kernels per group.
    pub kernels_per_group: usize,
    /// Kernel length.
    pub kernel_length: usize,
    /// Maximum dilation exponent (dilation = 2^i for i in 0..max_dilation_exp).
    pub max_dilation_exp: usize,
    /// Feature types to extract.
    pub feature_types: Vec<FeatureType>,
    /// Hidden dimension for classifier.
    pub hidden_dim: usize,
    /// Dropout rate.
    pub dropout: f64,
    /// Random seed.
    pub seed: u64,
}

impl Default for HydraMultiRocketPlusConfig {
    fn default() -> Self {
        Self {
            n_vars: 1,
            seq_len: 100,
            n_classes: 2,
            n_groups: 8,
            kernels_per_group: 8,
            kernel_length: 9,
            max_dilation_exp: 5, // dilations: 1, 2, 4, 8, 16, 32
            feature_types: vec![
                FeatureType::PPV,
                FeatureType::MPV,
                FeatureType::MIPV,
                FeatureType::LSPV,
            ],
            hidden_dim: 128,
            dropout: 0.1,
            seed: 42,
        }
    }
}

impl HydraMultiRocketPlusConfig {
    /// Create a new config.
    pub fn new(n_vars: usize, seq_len: usize, n_classes: usize) -> Self {
        Self {
            n_vars,
            seq_len,
            n_classes,
            ..Default::default()
        }
    }

    /// Set the number of kernel groups.
    #[must_use]
    pub fn with_n_groups(mut self, n_groups: usize) -> Self {
        self.n_groups = n_groups;
        self
    }

    /// Set kernels per group.
    #[must_use]
    pub fn with_kernels_per_group(mut self, kernels_per_group: usize) -> Self {
        self.kernels_per_group = kernels_per_group;
        self
    }

    /// Set kernel length.
    #[must_use]
    pub fn with_kernel_length(mut self, kernel_length: usize) -> Self {
        self.kernel_length = kernel_length;
        self
    }

    /// Set feature types.
    #[must_use]
    pub fn with_feature_types(mut self, feature_types: Vec<FeatureType>) -> Self {
        self.feature_types = feature_types;
        self
    }

    /// Set hidden dimension.
    #[must_use]
    pub fn with_hidden_dim(mut self, hidden_dim: usize) -> Self {
        self.hidden_dim = hidden_dim;
        self
    }

    /// Set dropout rate.
    #[must_use]
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Calculate total number of features.
    pub fn n_features(&self) -> usize {
        self.n_groups * self.kernels_per_group * self.feature_types.len() * self.n_vars
    }

    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> HydraMultiRocketPlus<B> {
        HydraMultiRocketPlus::new(self.clone(), device)
    }
}

/// HydraMultiRocketPlus model for time series classification.
///
/// Combines Hydra's efficient dictionary-based kernels with MultiRocket's
/// rich feature extraction (PPV, MPV, MIPV, LSPV).
///
/// # Architecture
///
/// ```text
/// Input (B, V, L) -> [Dictionary-based Random Kernels with Dilations]
///                 -> [Extract PPV, MPV, MIPV, LSPV per kernel]
///                 -> Concat Features
///                 -> Linear -> ReLU -> Dropout -> Linear -> Output
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use tsai_models::rocket::{HydraMultiRocketPlus, HydraMultiRocketPlusConfig};
///
/// let config = HydraMultiRocketPlusConfig::new(3, 100, 5)
///     .with_n_groups(8)
///     .with_kernels_per_group(8);
/// let model = config.init::<NdArray>(&device);
///
/// let x = Tensor::random([32, 3, 100], Distribution::Normal(0.0, 1.0), &device);
/// let output = model.forward(x);
/// // output shape: [32, 5]
/// ```
#[derive(Module, Debug)]
pub struct HydraMultiRocketPlus<B: Backend> {
    /// First linear layer.
    fc1: Linear<B>,
    /// Second linear layer (classifier).
    fc2: Linear<B>,
    /// Dropout.
    dropout: Dropout,
    /// Flattened kernel weights.
    #[module(skip)]
    kernel_weights: Vec<f32>,
    /// Flattened kernel biases.
    #[module(skip)]
    kernel_biases: Vec<f32>,
    /// Dilations per group.
    #[module(skip)]
    dilations: Vec<usize>,
    /// Number of kernels per group.
    #[module(skip)]
    kernels_per_group: usize,
    /// Feature types (as indices: 0=PPV, 1=MPV, 2=MIPV, 3=LSPV).
    #[module(skip)]
    feature_type_ids: Vec<usize>,
    /// Number of input variables.
    #[module(skip)]
    n_vars: usize,
    /// Kernel length.
    #[module(skip)]
    kernel_length: usize,
    /// Number of groups.
    #[module(skip)]
    n_groups: usize,
}

impl<B: Backend> HydraMultiRocketPlus<B> {
    /// Create a new HydraMultiRocketPlus model.
    pub fn new(config: HydraMultiRocketPlusConfig, device: &B::Device) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(config.seed);

        // Generate kernels
        let mut kernel_weights = Vec::new();
        let mut kernel_biases = Vec::new();
        let mut dilations = Vec::new();

        for group_idx in 0..config.n_groups {
            // Dilation increases exponentially per group
            let dilation = 1 << (group_idx % (config.max_dilation_exp + 1));
            dilations.push(dilation);

            for _ in 0..config.kernels_per_group {
                // Random kernel weights
                let mut kernel: Vec<f32> = (0..config.kernel_length)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect();

                // Normalize kernel (zero mean)
                let mean: f32 = kernel.iter().sum::<f32>() / config.kernel_length as f32;
                for w in &mut kernel {
                    *w -= mean;
                }

                kernel_weights.extend(kernel);
                kernel_biases.push(rng.gen_range(-1.0..1.0));
            }
        }

        // Convert feature types to IDs
        let feature_type_ids: Vec<usize> = config
            .feature_types
            .iter()
            .map(|f| match f {
                FeatureType::PPV => 0,
                FeatureType::MPV => 1,
                FeatureType::MIPV => 2,
                FeatureType::LSPV => 3,
            })
            .collect();

        // Classifier layers
        let n_features = config.n_features();
        let fc1 = LinearConfig::new(n_features, config.hidden_dim).init(device);
        let fc2 = LinearConfig::new(config.hidden_dim, config.n_classes).init(device);
        let dropout = DropoutConfig::new(config.dropout).init();

        Self {
            fc1,
            fc2,
            dropout,
            kernel_weights,
            kernel_biases,
            dilations,
            kernels_per_group: config.kernels_per_group,
            feature_type_ids,
            n_vars: config.n_vars,
            kernel_length: config.kernel_length,
            n_groups: config.n_groups,
        }
    }

    /// Compute convolution and extract MultiRocket features.
    fn extract_features(&self, x: &Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, n_vars, seq_len] = x.dims();
        let device = x.device();

        // Get input data
        let input_data = x.clone().into_data();
        let input_flat: Vec<f32> = input_data
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_default();

        let n_feature_types = self.feature_type_ids.len();
        let total_kernels = self.n_groups * self.kernels_per_group;
        let features_per_sample = total_kernels * n_feature_types * n_vars;

        let mut all_features = Vec::with_capacity(batch_size * features_per_sample);

        for b in 0..batch_size {
            for v in 0..n_vars {
                // Get time series for this sample and variable
                let ts_start = b * n_vars * seq_len + v * seq_len;
                let ts: &[f32] = &input_flat[ts_start..ts_start + seq_len];

                for group_idx in 0..self.n_groups {
                    let dilation = self.dilations[group_idx];

                    for kernel_idx in 0..self.kernels_per_group {
                        // Get kernel
                        let k_start =
                            (group_idx * self.kernels_per_group + kernel_idx) * self.kernel_length;
                        let kernel = &self.kernel_weights[k_start..k_start + self.kernel_length];
                        let bias = self.kernel_biases[group_idx * self.kernels_per_group + kernel_idx];

                        // Compute convolution and track statistics
                        let effective_len = (self.kernel_length - 1) * dilation + 1;

                        if effective_len > seq_len {
                            // Can't apply this kernel, output zeros
                            for _ in &self.feature_type_ids {
                                all_features.push(0.0);
                            }
                            continue;
                        }

                        let output_len = seq_len - effective_len + 1;

                        // Track positive values for MultiRocket features
                        let mut positive_values: Vec<f32> = Vec::new();
                        let mut positive_indices: Vec<usize> = Vec::new();
                        let mut current_stretch = 0;
                        let mut longest_stretch = 0;
                        let mut total_count = 0;

                        for i in 0..output_len {
                            let mut conv_val = 0.0f32;
                            for (k, &w) in kernel.iter().enumerate() {
                                let idx = i + k * dilation;
                                if idx < seq_len {
                                    conv_val += ts[idx] * w;
                                }
                            }
                            conv_val += bias;

                            if conv_val > 0.0 {
                                positive_values.push(conv_val);
                                positive_indices.push(total_count);
                                current_stretch += 1;
                                longest_stretch = longest_stretch.max(current_stretch);
                            } else {
                                current_stretch = 0;
                            }
                            total_count += 1;
                        }

                        // Extract requested features
                        for &feat_type in &self.feature_type_ids {
                            let feature_value = match feat_type {
                                0 => {
                                    // PPV: Proportion of Positive Values
                                    if total_count > 0 {
                                        positive_values.len() as f32 / total_count as f32
                                    } else {
                                        0.0
                                    }
                                }
                                1 => {
                                    // MPV: Mean of Positive Values
                                    if !positive_values.is_empty() {
                                        positive_values.iter().sum::<f32>() / positive_values.len() as f32
                                    } else {
                                        0.0
                                    }
                                }
                                2 => {
                                    // MIPV: Mean of Indices of Positive Values
                                    if !positive_indices.is_empty() && total_count > 0 {
                                        let mean_idx: f32 = positive_indices.iter().sum::<usize>() as f32
                                            / positive_indices.len() as f32;
                                        mean_idx / total_count as f32
                                    } else {
                                        0.5
                                    }
                                }
                                3 => {
                                    // LSPV: Longest Stretch of Positive Values
                                    if total_count > 0 {
                                        longest_stretch as f32 / total_count as f32
                                    } else {
                                        0.0
                                    }
                                }
                                _ => 0.0,
                            };
                            all_features.push(feature_value);
                        }
                    }
                }
            }
        }

        Tensor::<B, 1>::from_floats(all_features.as_slice(), &device)
            .reshape([batch_size, features_per_sample])
    }

    /// Forward pass.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let features = self.extract_features(&x);
        let out = self.fc1.forward(features);
        let out = Relu::new().forward(out);
        let out = self.dropout.forward(out);
        self.fc2.forward(out)
    }

    /// Forward pass returning probabilities.
    pub fn forward_probs(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let logits = self.forward(x);
        softmax(logits, 1)
    }

    /// Get the number of kernel groups.
    pub fn num_groups(&self) -> usize {
        self.n_groups
    }

    /// Get the total number of features.
    pub fn num_features(&self) -> usize {
        self.n_groups * self.kernels_per_group * self.feature_type_ids.len() * self.n_vars
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hydra_multirocket_config_default() {
        let config = HydraMultiRocketPlusConfig::default();
        assert_eq!(config.n_groups, 8);
        assert_eq!(config.kernels_per_group, 8);
        assert_eq!(config.kernel_length, 9);
        assert_eq!(config.feature_types.len(), 4);
        // 8 groups * 8 kernels * 4 features * 1 var = 256
        assert_eq!(config.n_features(), 256);
    }

    #[test]
    fn test_hydra_multirocket_config_new() {
        let config = HydraMultiRocketPlusConfig::new(3, 200, 10);
        assert_eq!(config.n_vars, 3);
        assert_eq!(config.seq_len, 200);
        assert_eq!(config.n_classes, 10);
        // 8 groups * 8 kernels * 4 features * 3 vars = 768
        assert_eq!(config.n_features(), 768);
    }

    #[test]
    fn test_hydra_multirocket_config_builder() {
        let config = HydraMultiRocketPlusConfig::new(2, 100, 5)
            .with_n_groups(4)
            .with_kernels_per_group(16)
            .with_kernel_length(7)
            .with_feature_types(vec![FeatureType::PPV, FeatureType::MPV])
            .with_hidden_dim(64)
            .with_dropout(0.2);

        assert_eq!(config.n_groups, 4);
        assert_eq!(config.kernels_per_group, 16);
        assert_eq!(config.kernel_length, 7);
        assert_eq!(config.feature_types.len(), 2);
        assert_eq!(config.hidden_dim, 64);
        assert_eq!(config.dropout, 0.2);
        // 4 groups * 16 kernels * 2 features * 2 vars = 256
        assert_eq!(config.n_features(), 256);
    }
}
