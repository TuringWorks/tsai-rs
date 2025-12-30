//! MultiRocket model for time series classification.
//!
//! MultiRocket extends MiniRocket by extracting multiple feature types:
//! - Proportion of Positive Values (PPV)
//! - Mean of Positive Values (MPV)
//! - Mean of Indices of Positive Values (MIPV)
//! - Longest Stretch of Positive Values (LSPV)
//!
//! Reference: "MultiRocket: Effective summary statistics for convolutional outputs
//! in time series classification" by Tan et al. (2021)

use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::prelude::*;
use burn::tensor::activation::softmax;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use super::minirocket::KERNEL_PATTERNS;

/// Feature types extracted by MultiRocket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureType {
    /// Proportion of Positive Values (original MiniRocket feature).
    PPV,
    /// Mean of Positive Values.
    MPV,
    /// Mean of Indices of Positive Values.
    MIPV,
    /// Longest Stretch of Positive Values.
    LSPV,
}

impl Default for FeatureType {
    fn default() -> Self {
        Self::PPV
    }
}

/// Configuration for MultiRocket model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiRocketConfig {
    /// Number of input variables/channels.
    pub n_vars: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// Number of kernels (features = n_kernels * 4 for all feature types).
    pub n_kernels: usize,
    /// Feature types to extract.
    pub feature_types: Vec<FeatureType>,
    /// Random seed for deterministic kernel generation.
    pub seed: u64,
}

impl Default for MultiRocketConfig {
    fn default() -> Self {
        Self {
            n_vars: 1,
            seq_len: 100,
            n_classes: 2,
            n_kernels: 10000,
            feature_types: vec![
                FeatureType::PPV,
                FeatureType::MPV,
                FeatureType::MIPV,
                FeatureType::LSPV,
            ],
            seed: 42,
        }
    }
}

impl MultiRocketConfig {
    /// Create a new config.
    pub fn new(n_vars: usize, seq_len: usize, n_classes: usize) -> Self {
        Self {
            n_vars,
            seq_len,
            n_classes,
            ..Default::default()
        }
    }

    /// Set the number of kernels.
    #[must_use]
    pub fn with_n_kernels(mut self, n_kernels: usize) -> Self {
        self.n_kernels = n_kernels;
        self
    }

    /// Set which feature types to extract.
    #[must_use]
    pub fn with_feature_types(mut self, feature_types: Vec<FeatureType>) -> Self {
        self.feature_types = feature_types;
        self
    }

    /// Get the total number of output features.
    pub fn n_features(&self) -> usize {
        self.n_kernels * self.feature_types.len()
    }

    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiRocket<B> {
        MultiRocket::new(self.clone(), device)
    }
}

/// MultiRocket feature extraction state.
#[derive(Debug, Clone)]
pub struct MultiRocketFeatures {
    /// Kernel weights.
    pub kernels: Vec<Vec<f32>>,
    /// Kernel dilations.
    pub dilations: Vec<usize>,
    /// Kernel biases/thresholds.
    pub biases: Vec<f32>,
    /// Feature types to extract.
    pub feature_types: Vec<FeatureType>,
    /// Number of kernels.
    pub n_kernels: usize,
    /// Total number of features.
    pub n_features: usize,
}

impl MultiRocketFeatures {
    /// Create feature extraction state from config.
    pub fn new(config: &MultiRocketConfig) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(config.seed);

        let n_pattern_sets = config.n_kernels / 84 + 1;
        let mut kernels = Vec::new();
        let mut dilations = Vec::new();
        let mut biases = Vec::new();

        // Generate kernels with different dilations
        for _ in 0..n_pattern_sets {
            // Random dilation
            let max_dilation = (config.seq_len - 1) / 8;
            let dilation = rng.gen_range(1..=max_dilation.max(1));

            for pattern in &KERNEL_PATTERNS {
                if kernels.len() >= config.n_kernels {
                    break;
                }

                // Convert pattern to f32 weights
                let kernel: Vec<f32> = pattern.iter().map(|&x| x as f32).collect();
                kernels.push(kernel);
                dilations.push(dilation);

                // Random bias
                biases.push(rng.gen_range(-1.0..1.0));
            }
        }

        // Truncate to exact number
        kernels.truncate(config.n_kernels);
        dilations.truncate(config.n_kernels);
        biases.truncate(config.n_kernels);

        let n_features = config.n_kernels * config.feature_types.len();

        Self {
            kernels,
            dilations,
            biases,
            feature_types: config.feature_types.clone(),
            n_kernels: config.n_kernels,
            n_features,
        }
    }

    /// Fit biases using training data.
    ///
    /// Adjusts biases to achieve approximately 50% PPV for balanced features.
    pub fn fit_biases<B: Backend>(&mut self, x: &Tensor<B, 3>) {
        let [n_samples, n_vars, seq_len] = x.dims();
        let x_data: Vec<f32> = x.to_data().to_vec().unwrap();

        for (_k_idx, ((kernel, &dilation), bias)) in self
            .kernels
            .iter()
            .zip(&self.dilations)
            .zip(&mut self.biases)
            .enumerate()
        {
            let kernel_len = kernel.len();
            let effective_len = (kernel_len - 1) * dilation + 1;

            if effective_len > seq_len {
                continue;
            }

            // Collect convolution outputs
            let mut conv_outputs: Vec<f32> = Vec::new();

            for b in 0..n_samples.min(100) {
                for v in 0..n_vars {
                    for t in 0..=(seq_len - effective_len) {
                        let mut conv_val = 0.0f32;
                        for (i, &w) in kernel.iter().enumerate() {
                            let idx = b * n_vars * seq_len + v * seq_len + t + i * dilation;
                            conv_val += x_data[idx] * w;
                        }
                        conv_outputs.push(conv_val);
                    }
                }
            }

            // Set bias to median for ~50% PPV
            if !conv_outputs.is_empty() {
                conv_outputs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                *bias = conv_outputs[conv_outputs.len() / 2];
            }
        }
    }

    /// Extract features from input data.
    ///
    /// Returns tensor of shape (batch, n_features) where n_features = n_kernels * len(feature_types).
    pub fn extract<B: Backend>(&self, x: &Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch, n_vars, seq_len] = x.dims();
        let device = x.device();

        let x_data: Vec<f32> = x.to_data().to_vec().unwrap();

        let n_feature_types = self.feature_types.len();
        let mut features = vec![0.0f32; batch * self.n_features];

        for b in 0..batch {
            for (k_idx, ((kernel, &dilation), &bias)) in self
                .kernels
                .iter()
                .zip(&self.dilations)
                .zip(&self.biases)
                .enumerate()
            {
                let kernel_len = kernel.len();
                let effective_len = (kernel_len - 1) * dilation + 1;

                if effective_len > seq_len {
                    continue;
                }

                // Compute convolution outputs and track positive values
                let mut positive_values: Vec<f32> = Vec::new();
                let mut positive_indices: Vec<usize> = Vec::new();
                let mut total_count = 0;
                let mut current_stretch = 0;
                let mut longest_stretch = 0;

                for v in 0..n_vars {
                    for t in 0..=(seq_len - effective_len) {
                        let mut conv_val = 0.0f32;
                        for (i, &w) in kernel.iter().enumerate() {
                            let idx = b * n_vars * seq_len + v * seq_len + t + i * dilation;
                            conv_val += x_data[idx] * w;
                        }

                        if conv_val > bias {
                            positive_values.push(conv_val);
                            positive_indices.push(total_count);
                            current_stretch += 1;
                            if current_stretch > longest_stretch {
                                longest_stretch = current_stretch;
                            }
                        } else {
                            current_stretch = 0;
                        }
                        total_count += 1;
                    }
                }

                // Compute requested features
                for (f_idx, &feature_type) in self.feature_types.iter().enumerate() {
                    let feature_value = match feature_type {
                        FeatureType::PPV => {
                            // Proportion of Positive Values
                            if total_count > 0 {
                                positive_values.len() as f32 / total_count as f32
                            } else {
                                0.0
                            }
                        }
                        FeatureType::MPV => {
                            // Mean of Positive Values
                            if !positive_values.is_empty() {
                                positive_values.iter().sum::<f32>() / positive_values.len() as f32
                            } else {
                                0.0
                            }
                        }
                        FeatureType::MIPV => {
                            // Mean of Indices of Positive Values (normalized)
                            if !positive_indices.is_empty() && total_count > 0 {
                                let mean_idx: f32 = positive_indices.iter().sum::<usize>() as f32
                                    / positive_indices.len() as f32;
                                mean_idx / total_count as f32
                            } else {
                                0.5 // Default to middle
                            }
                        }
                        FeatureType::LSPV => {
                            // Longest Stretch of Positive Values (normalized)
                            if total_count > 0 {
                                longest_stretch as f32 / total_count as f32
                            } else {
                                0.0
                            }
                        }
                    };

                    let feat_idx = b * self.n_features + k_idx * n_feature_types + f_idx;
                    features[feat_idx] = feature_value;
                }
            }
        }

        Tensor::<B, 1>::from_floats(features.as_slice(), &device).reshape([batch, self.n_features])
    }
}

/// MultiRocket model for time series classification.
///
/// Extends MiniRocket with multiple feature types for improved accuracy.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_models::rocket::{MultiRocket, MultiRocketConfig, MultiRocketFeatures, FeatureType};
///
/// let config = MultiRocketConfig::new(3, 100, 5)
///     .with_n_kernels(10000)
///     .with_feature_types(vec![FeatureType::PPV, FeatureType::MPV]);
///
/// let features = MultiRocketFeatures::new(&config);
/// let model = config.init::<NdArray>(&device);
///
/// let x_features = features.extract(&x);
/// let logits = model.forward(x_features);
/// ```
#[derive(Module, Debug)]
pub struct MultiRocket<B: Backend> {
    /// Linear classifier.
    classifier: Linear<B>,
    /// Number of expected features.
    #[module(skip)]
    n_features: usize,
}

impl<B: Backend> MultiRocket<B> {
    /// Create a new MultiRocket classifier.
    pub fn new(config: MultiRocketConfig, device: &B::Device) -> Self {
        let n_features = config.n_features();
        let classifier = LinearConfig::new(n_features, config.n_classes).init(device);
        Self {
            classifier,
            n_features,
        }
    }

    /// Get the expected number of input features.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Forward pass on pre-extracted features.
    pub fn forward(&self, features: Tensor<B, 2>) -> Tensor<B, 2> {
        self.classifier.forward(features)
    }

    /// Forward pass returning probabilities.
    pub fn forward_probs(&self, features: Tensor<B, 2>) -> Tensor<B, 2> {
        let logits = self.forward(features);
        softmax(logits, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multirocket_config_default() {
        let config = MultiRocketConfig::default();
        assert_eq!(config.n_kernels, 10000);
        assert_eq!(config.feature_types.len(), 4);
        assert_eq!(config.n_features(), 40000); // 10000 * 4
    }

    #[test]
    fn test_multirocket_config_builder() {
        let config = MultiRocketConfig::new(3, 200, 10)
            .with_n_kernels(5000)
            .with_feature_types(vec![FeatureType::PPV, FeatureType::MPV]);

        assert_eq!(config.n_vars, 3);
        assert_eq!(config.seq_len, 200);
        assert_eq!(config.n_classes, 10);
        assert_eq!(config.n_kernels, 5000);
        assert_eq!(config.feature_types.len(), 2);
        assert_eq!(config.n_features(), 10000); // 5000 * 2
    }

    #[test]
    fn test_multirocket_features_creation() {
        let config = MultiRocketConfig::new(1, 100, 2).with_n_kernels(100);
        let features = MultiRocketFeatures::new(&config);

        assert_eq!(features.n_kernels, 100);
        assert_eq!(features.kernels.len(), 100);
        assert_eq!(features.feature_types.len(), 4);
        assert_eq!(features.n_features, 400); // 100 * 4
    }
}
