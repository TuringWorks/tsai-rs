//! HydraPlus: Hybrid Dictionary-based Random Architecture.
//!
//! HydraPlus combines dictionary-based random convolutions with a learnable
//! classification head, offering competitive accuracy with fast training.
//!
//! Key features:
//! - Multiple groups of random kernels with different dilations
//! - Various pooling strategies (max, mean, PPV)
//! - Learnable classification head
//! - Efficient feature extraction
//!
//! Reference: "HYDRA: Competing convolutional kernels for fast and accurate
//! time series classification" by Dempster et al. (2023)

use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig, Relu};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// Pooling type for Hydra features.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HydraPooling {
    /// Max pooling over time.
    Max,
    /// Mean pooling over time.
    Mean,
    /// Proportion of positive values (like ROCKET).
    PPV,
}

impl Default for HydraPooling {
    fn default() -> Self {
        Self::Max
    }
}

/// Configuration for HydraPlus model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HydraPlusConfig {
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
    /// Maximum dilation.
    pub max_dilation: usize,
    /// Pooling types to use.
    pub pooling_types: Vec<HydraPooling>,
    /// Hidden dimension for classifier.
    pub hidden_dim: usize,
    /// Dropout rate.
    pub dropout: f64,
    /// Random seed.
    pub seed: u64,
}

impl Default for HydraPlusConfig {
    fn default() -> Self {
        Self {
            n_vars: 1,
            seq_len: 100,
            n_classes: 2,
            n_groups: 8,
            kernels_per_group: 8,
            kernel_length: 9,
            max_dilation: 32,
            pooling_types: vec![HydraPooling::Max, HydraPooling::PPV],
            hidden_dim: 128,
            dropout: 0.1,
            seed: 42,
        }
    }
}

impl HydraPlusConfig {
    /// Create a new config.
    pub fn new(n_vars: usize, seq_len: usize, n_classes: usize) -> Self {
        Self {
            n_vars,
            seq_len,
            n_classes,
            ..Default::default()
        }
    }

    /// Set the number of groups.
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

    /// Set pooling types.
    #[must_use]
    pub fn with_pooling_types(mut self, pooling_types: Vec<HydraPooling>) -> Self {
        self.pooling_types = pooling_types;
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
        self.n_groups * self.kernels_per_group * self.pooling_types.len() * self.n_vars
    }

    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> HydraPlus<B> {
        HydraPlus::new(self.clone(), device)
    }
}

/// HydraPlus model for time series classification.
///
/// Uses dictionary-based random convolutions with learnable classification head.
///
/// # Architecture
///
/// ```text
/// Input (B, V, L) -> [Random Convolutions with Dilations]
///                 -> [Multiple Pooling Types]
///                 -> Concat Features
///                 -> Linear -> ReLU -> Dropout -> Linear -> Output
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use tsai_models::rocket::HydraPlus;
///
/// let config = HydraPlusConfig::new(3, 100, 5)
///     .with_n_groups(8)
///     .with_kernels_per_group(8);
/// let model = config.init::<NdArray>(&device);
///
/// let x = Tensor::random([32, 3, 100], Distribution::Normal(0.0, 1.0), &device);
/// let output = model.forward(x);
/// // output shape: [32, 5]
/// ```
#[derive(Module, Debug)]
pub struct HydraPlus<B: Backend> {
    /// First linear layer.
    fc1: Linear<B>,
    /// Second linear layer (classifier).
    fc2: Linear<B>,
    /// Dropout.
    dropout: Dropout,
    /// Flattened kernel weights (n_groups * kernels_per_group * kernel_length).
    kernel_weights: Vec<f32>,
    /// Flattened kernel biases (n_groups * kernels_per_group).
    kernel_biases: Vec<f32>,
    /// Dilations per group.
    dilations: Vec<usize>,
    /// Kernels per group.
    kernels_per_group: usize,
    /// Pooling types.
    pooling_types_ids: Vec<usize>,
    /// Number of input variables.
    n_vars: usize,
    /// Kernel length.
    kernel_length: usize,
    /// Number of groups.
    n_groups: usize,
}

impl<B: Backend> HydraPlus<B> {
    /// Create a new HydraPlus model.
    pub fn new(config: HydraPlusConfig, device: &B::Device) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(config.seed);

        // Flattened storage for kernels
        let mut kernel_weights = Vec::new();
        let mut kernel_biases = Vec::new();
        let mut dilations = Vec::new();

        for group_idx in 0..config.n_groups {
            // Dilation increases exponentially per group
            let dilation = 1 << (group_idx % (config.max_dilation.ilog2() as usize + 1));
            dilations.push(dilation);

            // Generate random kernels for this group
            for _ in 0..config.kernels_per_group {
                // Random kernel weights (normalized)
                let mut kernel: Vec<f32> = (0..config.kernel_length)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect();

                // Normalize kernel (zero mean)
                let sum: f32 = kernel.iter().sum();
                let mean = sum / config.kernel_length as f32;
                for w in &mut kernel {
                    *w -= mean;
                }

                kernel_weights.extend(kernel);

                // Random bias for PPV (quantile-like)
                kernel_biases.push(rng.gen_range(-1.0..1.0));
            }
        }

        // Convert pooling types to IDs
        let pooling_types_ids: Vec<usize> = config
            .pooling_types
            .iter()
            .map(|p| match p {
                HydraPooling::Max => 0,
                HydraPooling::Mean => 1,
                HydraPooling::PPV => 2,
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
            pooling_types_ids,
            n_vars: config.n_vars,
            kernel_length: config.kernel_length,
            n_groups: config.n_groups,
        }
    }

    /// Apply convolution with dilation (CPU computation for random kernels).
    fn apply_convolution(
        &self,
        input: &[f32],
        kernel: &[f32],
        bias: f32,
        dilation: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        let kernel_len = kernel.len();
        let effective_len = (kernel_len - 1) * dilation + 1;

        if seq_len < effective_len {
            return vec![0.0; seq_len];
        }

        let output_len = seq_len - effective_len + 1;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let mut sum = 0.0f32;
            for (k, &w) in kernel.iter().enumerate() {
                let idx = i + k * dilation;
                if idx < seq_len {
                    sum += input[idx] * w;
                }
            }
            output.push(sum + bias);
        }

        output
    }

    /// Apply pooling to convolution output.
    fn apply_pooling(&self, output: &[f32], pooling_type: usize) -> f32 {
        if output.is_empty() {
            return 0.0;
        }

        match pooling_type {
            0 => {
                // Max pooling
                output.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            }
            1 => {
                // Mean pooling
                output.iter().sum::<f32>() / output.len() as f32
            }
            2 => {
                // PPV (proportion of positive values)
                let positive = output.iter().filter(|&&x| x > 0.0).count();
                positive as f32 / output.len() as f32
            }
            _ => 0.0,
        }
    }

    /// Extract features from input.
    fn extract_features(&self, x: &Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, n_vars, seq_len] = x.dims();
        let device = x.device();

        // Get input data
        let input_data = x.clone().into_data();
        let input_flat: Vec<f32> = input_data
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_default();

        // Calculate number of features per sample
        let n_pooling = self.pooling_types_ids.len();
        let total_kernels = self.n_groups * self.kernels_per_group;
        let features_per_sample = total_kernels * n_pooling * n_vars;

        let mut all_features = Vec::with_capacity(batch_size * features_per_sample);

        for b in 0..batch_size {
            for v in 0..n_vars {
                // Get time series for this sample and variable
                let ts_start = b * n_vars * seq_len + v * seq_len;
                let ts: Vec<f32> = input_flat[ts_start..ts_start + seq_len].to_vec();

                // Apply each kernel group
                for group_idx in 0..self.n_groups {
                    let dilation = self.dilations[group_idx];

                    for kernel_idx in 0..self.kernels_per_group {
                        // Get kernel weights from flattened storage
                        let kernel_start =
                            (group_idx * self.kernels_per_group + kernel_idx) * self.kernel_length;
                        let kernel_end = kernel_start + self.kernel_length;
                        let kernel = &self.kernel_weights[kernel_start..kernel_end];

                        // Get bias
                        let bias_idx = group_idx * self.kernels_per_group + kernel_idx;
                        let bias = self.kernel_biases[bias_idx];

                        // Apply convolution
                        let conv_output =
                            self.apply_convolution(&ts, kernel, bias, dilation, seq_len);

                        // Apply each pooling type
                        for &pooling_type in &self.pooling_types_ids {
                            let feature = self.apply_pooling(&conv_output, pooling_type);
                            all_features.push(feature);
                        }
                    }
                }
            }
        }

        // Convert to tensor
        Tensor::<B, 1>::from_floats(all_features.as_slice(), &device)
            .reshape([batch_size, features_per_sample])
    }

    /// Forward pass.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        // Extract features
        let features = self.extract_features(&x);

        // Classifier
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
        self.n_groups * self.kernels_per_group * self.pooling_types_ids.len() * self.n_vars
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hydra_config_default() {
        let config = HydraPlusConfig::default();
        assert_eq!(config.n_groups, 8);
        assert_eq!(config.kernels_per_group, 8);
        assert_eq!(config.kernel_length, 9);
        assert_eq!(config.pooling_types.len(), 2);
    }

    #[test]
    fn test_hydra_config_new() {
        let config = HydraPlusConfig::new(3, 200, 10);
        assert_eq!(config.n_vars, 3);
        assert_eq!(config.seq_len, 200);
        assert_eq!(config.n_classes, 10);
    }

    #[test]
    fn test_hydra_config_builder() {
        let config = HydraPlusConfig::new(3, 100, 5)
            .with_n_groups(4)
            .with_kernels_per_group(16)
            .with_kernel_length(7)
            .with_pooling_types(vec![HydraPooling::Max, HydraPooling::Mean, HydraPooling::PPV])
            .with_hidden_dim(64)
            .with_dropout(0.2);

        assert_eq!(config.n_groups, 4);
        assert_eq!(config.kernels_per_group, 16);
        assert_eq!(config.kernel_length, 7);
        assert_eq!(config.pooling_types.len(), 3);
        assert_eq!(config.hidden_dim, 64);
        assert_eq!(config.dropout, 0.2);
    }

    #[test]
    fn test_n_features() {
        let config = HydraPlusConfig::new(2, 100, 5)
            .with_n_groups(4)
            .with_kernels_per_group(8)
            .with_pooling_types(vec![HydraPooling::Max, HydraPooling::PPV]);

        // 4 groups * 8 kernels * 2 pooling types * 2 variables = 128 features
        assert_eq!(config.n_features(), 128);
    }
}
