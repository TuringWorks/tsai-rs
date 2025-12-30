//! ROCKET (Random Convolutional Kernel Transform) for time series classification.
//!
//! Based on "ROCKET: Exceptionally fast and accurate time series classification
//! using random convolutional kernels" by Dempster et al. (2020).
//!
//! ROCKET extracts features using a large number of random convolutional kernels
//! with random lengths, weights, biases, dilations, and paddings. Each kernel
//! produces two features: the max value and the proportion of positive values (PPV).

use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::prelude::*;
use burn::tensor::activation::softmax;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// Configuration for ROCKET model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RocketConfig {
    /// Number of input variables/channels.
    pub n_vars: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// Number of random kernels.
    pub n_kernels: usize,
    /// Possible kernel lengths.
    pub kernel_lengths: Vec<usize>,
    /// Random seed for deterministic kernel generation.
    pub seed: u64,
}

impl Default for RocketConfig {
    fn default() -> Self {
        Self {
            n_vars: 1,
            seq_len: 100,
            n_classes: 2,
            n_kernels: 10000,
            kernel_lengths: vec![7, 9, 11],
            seed: 42,
        }
    }
}

impl RocketConfig {
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

    /// Set the kernel lengths.
    #[must_use]
    pub fn with_kernel_lengths(mut self, lengths: Vec<usize>) -> Self {
        self.kernel_lengths = lengths;
        self
    }

    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Rocket<B> {
        Rocket::new(self.clone(), device)
    }
}

/// A random convolutional kernel for ROCKET.
#[derive(Debug, Clone)]
pub struct RandomKernel {
    /// Kernel weights.
    pub weights: Vec<f32>,
    /// Kernel length.
    pub length: usize,
    /// Dilation factor.
    pub dilation: usize,
    /// Padding amount.
    pub padding: usize,
    /// Bias value.
    pub bias: f32,
}

/// ROCKET feature extraction state.
#[derive(Debug, Clone)]
pub struct RocketFeatures {
    /// Random kernels.
    pub kernels: Vec<RandomKernel>,
    /// Number of kernels.
    pub n_kernels: usize,
    /// Number of output features (2 per kernel: max and PPV).
    pub n_features: usize,
}

impl RocketFeatures {
    /// Create feature extraction state from config.
    pub fn new(config: &RocketConfig) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
        let mut kernels = Vec::with_capacity(config.n_kernels);

        for _ in 0..config.n_kernels {
            // Random kernel length from available options
            let length = config.kernel_lengths[rng.gen_range(0..config.kernel_lengths.len())];

            // Random weights from normal distribution, then normalized
            let mut weights: Vec<f32> = (0..length)
                .map(|_| {
                    let u1: f32 = rng.gen();
                    let u2: f32 = rng.gen();
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
                })
                .collect();

            // Normalize weights to have zero mean
            let mean: f32 = weights.iter().sum::<f32>() / length as f32;
            for w in &mut weights {
                *w -= mean;
            }

            // Random dilation
            let max_dilation = ((config.seq_len - 1) / (length - 1)).max(1);
            let dilation = rng.gen_range(1..=max_dilation);

            // Random padding (0 or enough to maintain output size)
            let padding = if rng.gen::<bool>() {
                (length - 1) * dilation / 2
            } else {
                0
            };

            // Random bias from uniform distribution
            let bias: f32 = rng.gen_range(-1.0..1.0);

            kernels.push(RandomKernel {
                weights,
                length,
                dilation,
                padding,
                bias,
            });
        }

        Self {
            kernels,
            n_kernels: config.n_kernels,
            n_features: config.n_kernels * 2, // 2 features per kernel
        }
    }

    /// Fit biases using quantiles of the convolution outputs.
    ///
    /// This step is important for good PPV feature extraction.
    /// It adjusts biases so that ~50% of values are positive.
    pub fn fit_biases<B: Backend>(&mut self, x: &Tensor<B, 3>) {
        let [n_samples, n_vars, seq_len] = x.dims();
        let x_data: Vec<f32> = x.to_data().to_vec().unwrap();

        for kernel in &mut self.kernels {
            let effective_len = (kernel.length - 1) * kernel.dilation + 1;
            let output_len = if kernel.padding > 0 {
                seq_len
            } else {
                seq_len.saturating_sub(effective_len - 1)
            };

            if output_len == 0 {
                continue;
            }

            // Collect all convolution outputs for this kernel
            let mut all_outputs: Vec<f32> = Vec::new();

            for b in 0..n_samples.min(100) {
                // Use subset for efficiency
                for v in 0..n_vars {
                    for t in 0..output_len {
                        let mut conv_val = 0.0f32;

                        for (i, &w) in kernel.weights.iter().enumerate() {
                            let pos = t as i32 - kernel.padding as i32 + (i * kernel.dilation) as i32;
                            if pos >= 0 && (pos as usize) < seq_len {
                                let idx = b * n_vars * seq_len + v * seq_len + pos as usize;
                                conv_val += x_data[idx] * w;
                            }
                        }

                        all_outputs.push(conv_val);
                    }
                }
            }

            // Set bias to median to get ~50% PPV
            if !all_outputs.is_empty() {
                all_outputs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                kernel.bias = all_outputs[all_outputs.len() / 2];
            }
        }
    }

    /// Extract features from input data.
    ///
    /// Returns a tensor of shape (batch, n_features) where n_features = n_kernels * 2.
    /// For each kernel, we compute:
    /// - Max value of the convolution output
    /// - Proportion of positive values (PPV) after subtracting bias
    pub fn extract<B: Backend>(&self, x: &Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch, n_vars, seq_len] = x.dims();
        let device = x.device();

        let x_data: Vec<f32> = x.to_data().to_vec().unwrap();
        let mut features = vec![0.0f32; batch * self.n_features];

        for b in 0..batch {
            for (k_idx, kernel) in self.kernels.iter().enumerate() {
                let effective_len = (kernel.length - 1) * kernel.dilation + 1;
                let output_len = if kernel.padding > 0 {
                    seq_len
                } else {
                    seq_len.saturating_sub(effective_len - 1)
                };

                if output_len == 0 {
                    continue;
                }

                let mut max_val = f32::NEG_INFINITY;
                let mut ppv_sum = 0.0f32;
                let mut count = 0;

                // Sum across all variables
                for v in 0..n_vars {
                    for t in 0..output_len {
                        let mut conv_val = 0.0f32;

                        for (i, &w) in kernel.weights.iter().enumerate() {
                            let pos = t as i32 - kernel.padding as i32 + (i * kernel.dilation) as i32;
                            if pos >= 0 && (pos as usize) < seq_len {
                                let idx = b * n_vars * seq_len + v * seq_len + pos as usize;
                                conv_val += x_data[idx] * w;
                            }
                        }

                        // Update max
                        if conv_val > max_val {
                            max_val = conv_val;
                        }

                        // PPV: proportion of positive values after bias
                        if conv_val > kernel.bias {
                            ppv_sum += 1.0;
                        }
                        count += 1;
                    }
                }

                // Store features: max and PPV
                let feat_idx = b * self.n_features + k_idx * 2;
                features[feat_idx] = if max_val.is_finite() { max_val } else { 0.0 };
                features[feat_idx + 1] = if count > 0 { ppv_sum / count as f32 } else { 0.0 };
            }
        }

        Tensor::<B, 1>::from_floats(features.as_slice(), &device).reshape([batch, self.n_features])
    }
}

/// ROCKET model for time series classification.
///
/// Uses random convolutional kernels to extract features, then applies
/// a linear classifier. This is a simple but highly effective approach.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_models::rocket::{RocketConfig, Rocket, RocketFeatures};
///
/// let config = RocketConfig::new(3, 100, 5)
///     .with_n_kernels(10000);
///
/// // Create feature extractor and model
/// let features = RocketFeatures::new(&config);
/// let model = config.init::<NdArray>(&device);
///
/// // Extract features and classify
/// let x_features = features.extract(&x);
/// let logits = model.forward(x_features);
/// ```
#[derive(Module, Debug)]
pub struct Rocket<B: Backend> {
    /// Linear classifier on top of features.
    classifier: Linear<B>,
    /// Number of expected features.
    n_features: usize,
}

impl<B: Backend> Rocket<B> {
    /// Create a new ROCKET classifier.
    pub fn new(config: RocketConfig, device: &B::Device) -> Self {
        let n_features = config.n_kernels * 2;
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
    fn test_rocket_config_default() {
        let config = RocketConfig::default();
        assert_eq!(config.n_kernels, 10000);
        assert_eq!(config.kernel_lengths, vec![7, 9, 11]);
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn test_rocket_config_builder() {
        let config = RocketConfig::new(3, 200, 10)
            .with_n_kernels(5000)
            .with_kernel_lengths(vec![5, 7, 9, 11]);

        assert_eq!(config.n_vars, 3);
        assert_eq!(config.seq_len, 200);
        assert_eq!(config.n_classes, 10);
        assert_eq!(config.n_kernels, 5000);
        assert_eq!(config.kernel_lengths.len(), 4);
    }

    #[test]
    fn test_rocket_features_creation() {
        let config = RocketConfig::new(1, 100, 2).with_n_kernels(100);

        let features = RocketFeatures::new(&config);
        assert_eq!(features.n_kernels, 100);
        assert_eq!(features.n_features, 200); // 2 features per kernel
        assert_eq!(features.kernels.len(), 100);

        // Check that kernels have valid properties
        for kernel in &features.kernels {
            assert!(config.kernel_lengths.contains(&kernel.length));
            assert!(kernel.dilation >= 1);
            assert!(kernel.weights.len() == kernel.length);

            // Check weights are normalized (zero mean)
            let mean: f32 = kernel.weights.iter().sum::<f32>() / kernel.length as f32;
            assert!(mean.abs() < 1e-5, "Kernel weights should have zero mean");
        }
    }

    #[test]
    fn test_rocket_deterministic() {
        let config = RocketConfig::new(1, 100, 2).with_n_kernels(100);

        let features1 = RocketFeatures::new(&config);
        let features2 = RocketFeatures::new(&config);

        // Same seed should produce same kernels
        for (k1, k2) in features1.kernels.iter().zip(&features2.kernels) {
            assert_eq!(k1.length, k2.length);
            assert_eq!(k1.dilation, k2.dilation);
            assert_eq!(k1.padding, k2.padding);
            for (w1, w2) in k1.weights.iter().zip(&k2.weights) {
                assert!((w1 - w2).abs() < 1e-6);
            }
        }
    }
}
