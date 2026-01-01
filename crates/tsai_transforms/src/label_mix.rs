//! Label mixing transforms (MixUp, CutMix).
//!
//! These transforms mix samples and their labels during training
//! to improve generalization.
//!
//! # Base Handler
//!
//! The [`MixHandler1d`] trait provides a common interface for all mixing transforms.
//! Implementations can define custom mixing logic while benefiting from shared
//! utilities for batch processing and label mixing.

use burn::prelude::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use tsai_core::{CoreError, Result, Seed, Split, TSBatch, TSTensor, Transform};

/// Base handler trait for label mixing transforms.
///
/// This trait provides a common interface for transforms that mix samples
/// and their labels during training. Implementations define how to:
/// - Sample mixing coefficients (e.g., from Beta distribution)
/// - Mix input data (X values)
/// - Mix labels (Y values)
///
/// # Example
///
/// ```rust,ignore
/// use tsai_transforms::label_mix::MixHandler1d;
///
/// struct MyMixTransform {
///     alpha: f32,
/// }
///
/// impl MixHandler1d for MyMixTransform {
///     fn sample_lambda(&self, rng: &mut ChaCha8Rng) -> f32 {
///         // Custom lambda sampling
///     }
///
///     fn mix_samples(&self, x1: &[f32], x2: &[f32], lambda: f32) -> Vec<f32> {
///         // Custom sample mixing
///     }
/// }
/// ```
pub trait MixHandler1d {
    /// Sample a mixing coefficient lambda.
    ///
    /// Typically drawn from Beta(alpha, alpha) distribution.
    /// Returns value in [0, 1] where:
    /// - 1.0 means use only the original sample
    /// - 0.0 means use only the mixed sample
    fn sample_lambda(&self, rng: &mut ChaCha8Rng) -> f32;

    /// Mix two samples with given lambda.
    ///
    /// Default implementation: linear interpolation
    /// x' = lambda * x1 + (1 - lambda) * x2
    fn mix_samples(&self, x1: &[f32], x2: &[f32], lambda: f32) -> Vec<f32> {
        x1.iter()
            .zip(x2.iter())
            .map(|(&a, &b)| lambda * a + (1.0 - lambda) * b)
            .collect()
    }

    /// Mix two label vectors with given lambda.
    ///
    /// Default implementation: linear interpolation
    /// y' = lambda * y1 + (1 - lambda) * y2
    fn mix_labels(&self, y1: &[f32], y2: &[f32], lambda: f32) -> Vec<f32> {
        y1.iter()
            .zip(y2.iter())
            .map(|(&a, &b)| lambda * a + (1.0 - lambda) * b)
            .collect()
    }

    /// Get the probability of applying this transform.
    fn probability(&self) -> f32 {
        1.0
    }

    /// Get the name of this transform.
    fn name(&self) -> &str {
        "MixHandler1d"
    }

    /// Check if the transform should be applied based on probability.
    fn should_apply(&self, rng: &mut ChaCha8Rng) -> bool {
        rng.gen::<f32>() <= self.probability()
    }

    /// Sample lambda from Beta(alpha, alpha) using simple approximation.
    ///
    /// This helper can be used by implementations.
    fn sample_beta_lambda(alpha: f32, rng: &mut ChaCha8Rng) -> f32 {
        let u: f32 = rng.gen();
        let v: f32 = rng.gen();
        let x = u.powf(1.0 / alpha);
        let y = v.powf(1.0 / alpha);
        x / (x + y)
    }

    /// Create shuffled indices for pairing samples.
    fn create_shuffle_indices(batch_size: usize, rng: &mut ChaCha8Rng) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..batch_size).collect();
        indices.shuffle(rng);
        indices
    }
}

/// Mix strategy enumeration for selecting mixing behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MixStrategy {
    /// Standard MixUp: linear interpolation of entire samples.
    MixUp,
    /// CutMix: paste a cut region from another sample.
    CutMix,
    /// IntraClass CutMix: only mix within same class.
    IntraClassCutMix,
    /// Random choice between strategies.
    Random,
}

impl Default for MixStrategy {
    fn default() -> Self {
        Self::MixUp
    }
}

/// Configuration for a generic mix handler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixHandlerConfig {
    /// Alpha parameter for Beta distribution.
    pub alpha: f32,
    /// Probability of applying the transform.
    pub p: f32,
    /// Mixing strategy to use.
    pub strategy: MixStrategy,
}

impl Default for MixHandlerConfig {
    fn default() -> Self {
        Self {
            alpha: 0.4,
            p: 1.0,
            strategy: MixStrategy::MixUp,
        }
    }
}

/// Generic mix handler that can apply different mixing strategies.
pub struct GenericMixHandler {
    config: MixHandlerConfig,
    seed: Seed,
}

impl GenericMixHandler {
    /// Create a new generic mix handler.
    #[must_use]
    pub fn new(config: MixHandlerConfig) -> Self {
        Self {
            config,
            seed: Seed::from_entropy(),
        }
    }

    /// Create with default MixUp configuration.
    #[must_use]
    pub fn mixup(alpha: f32) -> Self {
        Self::new(MixHandlerConfig {
            alpha,
            strategy: MixStrategy::MixUp,
            ..Default::default()
        })
    }

    /// Create with CutMix configuration.
    #[must_use]
    pub fn cutmix(alpha: f32) -> Self {
        Self::new(MixHandlerConfig {
            alpha,
            strategy: MixStrategy::CutMix,
            ..Default::default()
        })
    }

    /// Set the random seed.
    #[must_use]
    pub fn with_seed(mut self, seed: Seed) -> Self {
        self.seed = seed;
        self
    }

    /// Set the probability.
    #[must_use]
    pub fn with_probability(mut self, p: f32) -> Self {
        self.config.p = p.clamp(0.0, 1.0);
        self
    }

    /// Get the current strategy.
    pub fn strategy(&self) -> MixStrategy {
        self.config.strategy
    }
}

impl MixHandler1d for GenericMixHandler {
    fn sample_lambda(&self, rng: &mut ChaCha8Rng) -> f32 {
        Self::sample_beta_lambda(self.config.alpha, rng)
    }

    fn probability(&self) -> f32 {
        self.config.p
    }

    fn name(&self) -> &str {
        match self.config.strategy {
            MixStrategy::MixUp => "GenericMixHandler(MixUp)",
            MixStrategy::CutMix => "GenericMixHandler(CutMix)",
            MixStrategy::IntraClassCutMix => "GenericMixHandler(IntraClassCutMix)",
            MixStrategy::Random => "GenericMixHandler(Random)",
        }
    }
}

/// Configuration for MixUp transform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixUpConfig {
    /// Alpha parameter for Beta distribution.
    pub alpha: f32,
    /// Probability of applying MixUp.
    pub p: f32,
}

impl Default for MixUpConfig {
    fn default() -> Self {
        Self { alpha: 0.4, p: 1.0 }
    }
}

/// MixUp augmentation for 1D time series.
///
/// Mixes pairs of samples and their labels according to:
/// x' = lambda * x1 + (1 - lambda) * x2
/// y' = lambda * y1 + (1 - lambda) * y2
///
/// where lambda ~ Beta(alpha, alpha)
pub struct MixUp1d {
    config: MixUpConfig,
    seed: Seed,
}

impl MixUp1d {
    /// Create a new MixUp transform.
    #[must_use]
    pub fn new(alpha: f32) -> Self {
        Self {
            config: MixUpConfig {
                alpha,
                ..Default::default()
            },
            seed: Seed::from_entropy(),
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: MixUpConfig) -> Self {
        Self {
            config,
            seed: Seed::from_entropy(),
        }
    }

    /// Set the random seed.
    #[must_use]
    pub fn with_seed(mut self, seed: Seed) -> Self {
        self.seed = seed;
        self
    }

    /// Sample lambda from Beta(alpha, alpha) using the inverse CDF method.
    fn sample_lambda(&self, rng: &mut ChaCha8Rng) -> f32 {
        // Simple approximation for Beta distribution
        // For alpha close to 0.4, this works reasonably well
        let u: f32 = rng.gen();
        let v: f32 = rng.gen();

        // Box-Muller approximation for Beta
        let alpha = self.config.alpha;
        let x = u.powf(1.0 / alpha);
        let y = v.powf(1.0 / alpha);

        x / (x + y)
    }
}

impl<B: Backend> Transform<B> for MixUp1d {
    fn apply(&self, batch: TSBatch<B>, split: Split) -> Result<TSBatch<B>> {
        if split.is_eval() {
            return Ok(batch);
        }

        let Some(y) = batch.y.clone() else {
            // No labels to mix
            return Ok(batch);
        };

        let mut rng = self.seed.to_rng();

        if rng.gen::<f32>() > self.config.p {
            return Ok(batch);
        }

        let shape = batch.x.shape();
        let batch_size = shape.batch();
        let n_vars = shape.vars();
        let seq_len = shape.len();
        let device = batch.device();

        // Sample lambda from Beta distribution
        let lambda = self.sample_lambda(&mut rng);

        // Create shuffled indices for mixing pairs
        let mut indices: Vec<usize> = (0..batch_size).collect();
        indices.shuffle(&mut rng);

        // Get data as raw values
        let x_data = batch.x.into_inner().into_data();
        let x_values: Vec<f32> = x_data
            .as_slice()
            .map_err(|e| CoreError::ShapeMismatch(format!("Failed to get X data: {e:?}")))?
            .to_vec();

        let y_data = y.into_data();
        let y_values: Vec<f32> = y_data
            .as_slice()
            .map_err(|e| CoreError::ShapeMismatch(format!("Failed to get Y data: {e:?}")))?
            .to_vec();

        let y_dims: Vec<usize> = y_data.shape.clone();
        let y_cols = if y_dims.len() > 1 { y_dims[1] } else { 1 };

        // Mix X values: x' = lambda * x + (1 - lambda) * x_shuffled
        let mut mixed_x: Vec<f32> = vec![0.0; batch_size * n_vars * seq_len];
        for b in 0..batch_size {
            let shuffled_b = indices[b];
            for v in 0..n_vars {
                for t in 0..seq_len {
                    let idx = b * n_vars * seq_len + v * seq_len + t;
                    let shuffled_idx = shuffled_b * n_vars * seq_len + v * seq_len + t;
                    mixed_x[idx] = lambda * x_values[idx] + (1.0 - lambda) * x_values[shuffled_idx];
                }
            }
        }

        // Mix Y values: y' = lambda * y + (1 - lambda) * y_shuffled
        let mut mixed_y: Vec<f32> = vec![0.0; batch_size * y_cols];
        for b in 0..batch_size {
            let shuffled_b = indices[b];
            for c in 0..y_cols {
                let idx = b * y_cols + c;
                let shuffled_idx = shuffled_b * y_cols + c;
                mixed_y[idx] = lambda * y_values[idx] + (1.0 - lambda) * y_values[shuffled_idx];
            }
        }

        // Reconstruct tensors
        let mixed_x_tensor: Tensor<B, 3> = Tensor::<B, 1>::from_floats(mixed_x.as_slice(), &device)
            .reshape([batch_size, n_vars, seq_len]);

        let mixed_y_tensor: Tensor<B, 2> = Tensor::<B, 1>::from_floats(mixed_y.as_slice(), &device)
            .reshape([batch_size, y_cols]);

        Ok(TSBatch {
            x: TSTensor::new(mixed_x_tensor)?,
            y: Some(mixed_y_tensor),
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "MixUp1d"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

/// Configuration for CutMix transform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CutMixConfig {
    /// Alpha parameter for Beta distribution.
    pub alpha: f32,
    /// Probability of applying CutMix.
    pub p: f32,
}

impl Default for CutMixConfig {
    fn default() -> Self {
        Self { alpha: 1.0, p: 1.0 }
    }
}

/// CutMix augmentation for 1D time series.
///
/// Cuts a segment from one sample and pastes it onto another,
/// mixing the labels proportionally to the segment lengths.
pub struct CutMix1d {
    config: CutMixConfig,
    seed: Seed,
}

impl CutMix1d {
    /// Create a new CutMix transform.
    #[must_use]
    pub fn new(alpha: f32) -> Self {
        Self {
            config: CutMixConfig {
                alpha,
                ..Default::default()
            },
            seed: Seed::from_entropy(),
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: CutMixConfig) -> Self {
        Self {
            config,
            seed: Seed::from_entropy(),
        }
    }

    /// Set the random seed.
    #[must_use]
    pub fn with_seed(mut self, seed: Seed) -> Self {
        self.seed = seed;
        self
    }
}

impl CutMix1d {
    /// Sample lambda from Beta(alpha, alpha) using the inverse CDF method.
    fn sample_lambda(&self, rng: &mut ChaCha8Rng) -> f32 {
        let u: f32 = rng.gen();
        let v: f32 = rng.gen();

        let alpha = self.config.alpha;
        let x = u.powf(1.0 / alpha);
        let y = v.powf(1.0 / alpha);

        x / (x + y)
    }
}

impl<B: Backend> Transform<B> for CutMix1d {
    fn apply(&self, batch: TSBatch<B>, split: Split) -> Result<TSBatch<B>> {
        if split.is_eval() {
            return Ok(batch);
        }

        let Some(y) = batch.y.clone() else {
            return Ok(batch);
        };

        let mut rng = self.seed.to_rng();

        if rng.gen::<f32>() > self.config.p {
            return Ok(batch);
        }

        let shape = batch.x.shape();
        let batch_size = shape.batch();
        let n_vars = shape.vars();
        let seq_len = shape.len();
        let device = batch.device();

        // Sample lambda (determines the cut ratio)
        let lambda = self.sample_lambda(&mut rng);

        // Calculate cut length based on lambda
        let cut_len = ((1.0 - lambda) * seq_len as f32).round() as usize;
        let cut_len = cut_len.max(1).min(seq_len - 1);

        // Random cut start position
        let cut_start = rng.gen_range(0..seq_len.saturating_sub(cut_len).max(1));
        let cut_end = (cut_start + cut_len).min(seq_len);

        // Actual lambda based on cut region
        let actual_lambda = 1.0 - (cut_end - cut_start) as f32 / seq_len as f32;

        // Create shuffled indices for mixing pairs
        let mut indices: Vec<usize> = (0..batch_size).collect();
        indices.shuffle(&mut rng);

        // Get data as raw values
        let x_data = batch.x.into_inner().into_data();
        let x_values: Vec<f32> = x_data
            .as_slice()
            .map_err(|e| CoreError::ShapeMismatch(format!("Failed to get X data: {e:?}")))?
            .to_vec();

        let y_data = y.into_data();
        let y_values: Vec<f32> = y_data
            .as_slice()
            .map_err(|e| CoreError::ShapeMismatch(format!("Failed to get Y data: {e:?}")))?
            .to_vec();

        let y_dims: Vec<usize> = y_data.shape.clone();
        let y_cols = if y_dims.len() > 1 { y_dims[1] } else { 1 };

        // Apply CutMix: paste cut region from shuffled sample
        let mut mixed_x: Vec<f32> = x_values.clone();
        for b in 0..batch_size {
            let shuffled_b = indices[b];
            for v in 0..n_vars {
                for t in cut_start..cut_end {
                    let dst_idx = b * n_vars * seq_len + v * seq_len + t;
                    let src_idx = shuffled_b * n_vars * seq_len + v * seq_len + t;
                    mixed_x[dst_idx] = x_values[src_idx];
                }
            }
        }

        // Mix Y values based on actual cut ratio
        let mut mixed_y: Vec<f32> = vec![0.0; batch_size * y_cols];
        for b in 0..batch_size {
            let shuffled_b = indices[b];
            for c in 0..y_cols {
                let idx = b * y_cols + c;
                let shuffled_idx = shuffled_b * y_cols + c;
                mixed_y[idx] =
                    actual_lambda * y_values[idx] + (1.0 - actual_lambda) * y_values[shuffled_idx];
            }
        }

        // Reconstruct tensors
        let mixed_x_tensor: Tensor<B, 3> = Tensor::<B, 1>::from_floats(mixed_x.as_slice(), &device)
            .reshape([batch_size, n_vars, seq_len]);

        let mixed_y_tensor: Tensor<B, 2> = Tensor::<B, 1>::from_floats(mixed_y.as_slice(), &device)
            .reshape([batch_size, y_cols]);

        Ok(TSBatch {
            x: TSTensor::new(mixed_x_tensor)?,
            y: Some(mixed_y_tensor),
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "CutMix1d"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

/// Configuration for IntraClassCutMix transform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntraClassCutMixConfig {
    /// Alpha parameter for Beta distribution.
    pub alpha: f32,
    /// Probability of applying the transform.
    pub p: f32,
}

impl Default for IntraClassCutMixConfig {
    fn default() -> Self {
        Self { alpha: 1.0, p: 1.0 }
    }
}

/// IntraClass CutMix augmentation.
///
/// Like CutMix, but only mixes samples from the same class.
/// This helps preserve class-discriminative features.
pub struct IntraClassCutMix1d {
    config: IntraClassCutMixConfig,
    seed: Seed,
}

impl IntraClassCutMix1d {
    /// Create a new IntraClassCutMix transform.
    #[must_use]
    pub fn new(alpha: f32) -> Self {
        Self {
            config: IntraClassCutMixConfig {
                alpha,
                ..Default::default()
            },
            seed: Seed::from_entropy(),
        }
    }

    /// Set the random seed.
    #[must_use]
    pub fn with_seed(mut self, seed: Seed) -> Self {
        self.seed = seed;
        self
    }
}

impl IntraClassCutMix1d {
    /// Sample lambda from Beta(alpha, alpha) using the inverse CDF method.
    fn sample_lambda(&self, rng: &mut ChaCha8Rng) -> f32 {
        let u: f32 = rng.gen();
        let v: f32 = rng.gen();

        let alpha = self.config.alpha;
        let x = u.powf(1.0 / alpha);
        let y = v.powf(1.0 / alpha);

        x / (x + y)
    }
}

impl<B: Backend> Transform<B> for IntraClassCutMix1d {
    fn apply(&self, batch: TSBatch<B>, split: Split) -> Result<TSBatch<B>> {
        if split.is_eval() {
            return Ok(batch);
        }

        let Some(y) = batch.y.clone() else {
            return Ok(batch);
        };

        let mut rng = self.seed.to_rng();

        if rng.gen::<f32>() > self.config.p {
            return Ok(batch);
        }

        let shape = batch.x.shape();
        let batch_size = shape.batch();
        let n_vars = shape.vars();
        let seq_len = shape.len();
        let device = batch.device();

        // Get data as raw values
        let x_data = batch.x.into_inner().into_data();
        let x_values: Vec<f32> = x_data
            .as_slice()
            .map_err(|e| CoreError::ShapeMismatch(format!("Failed to get X data: {e:?}")))?
            .to_vec();

        let y_data = y.into_data();
        let y_values: Vec<f32> = y_data
            .as_slice()
            .map_err(|e| CoreError::ShapeMismatch(format!("Failed to get Y data: {e:?}")))?
            .to_vec();

        let y_dims: Vec<usize> = y_data.shape.clone();
        let y_cols = if y_dims.len() > 1 { y_dims[1] } else { 1 };

        // Group samples by class (argmax of y for one-hot, or y value for integer labels)
        let mut class_indices: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();

        for b in 0..batch_size {
            let class_id = if y_cols > 1 {
                // One-hot encoding: find argmax
                let mut max_idx = 0;
                let mut max_val = y_values[b * y_cols];
                for c in 1..y_cols {
                    if y_values[b * y_cols + c] > max_val {
                        max_val = y_values[b * y_cols + c];
                        max_idx = c;
                    }
                }
                max_idx
            } else {
                // Integer label
                y_values[b] as usize
            };

            class_indices.entry(class_id).or_default().push(b);
        }

        // Sample lambda (determines the cut ratio)
        let lambda = self.sample_lambda(&mut rng);

        // Calculate cut length based on lambda
        let cut_len = ((1.0 - lambda) * seq_len as f32).round() as usize;
        let cut_len = cut_len.max(1).min(seq_len - 1);

        // Random cut start position
        let cut_start = rng.gen_range(0..seq_len.saturating_sub(cut_len).max(1));
        let cut_end = (cut_start + cut_len).min(seq_len);

        // Actual lambda based on cut region
        let actual_lambda = 1.0 - (cut_end - cut_start) as f32 / seq_len as f32;

        // Create mapping: for each sample, find another sample from the same class
        let mut mix_partner: Vec<usize> = (0..batch_size).collect();
        for (_class_id, indices) in &class_indices {
            if indices.len() < 2 {
                // Can't mix if only one sample of this class
                continue;
            }

            // Shuffle within class
            let mut shuffled: Vec<usize> = indices.clone();
            shuffled.shuffle(&mut rng);

            // Pair samples: each sample mixes with the next in the shuffled list
            for i in 0..indices.len() {
                let partner_idx = (i + 1) % indices.len();
                mix_partner[indices[i]] = shuffled[partner_idx];
            }
        }

        // Apply IntraClass CutMix: paste cut region from same-class sample
        let mut mixed_x: Vec<f32> = x_values.clone();
        for b in 0..batch_size {
            let partner_b = mix_partner[b];
            if partner_b == b {
                // No partner found (single sample of this class), skip
                continue;
            }

            for v in 0..n_vars {
                for t in cut_start..cut_end {
                    let dst_idx = b * n_vars * seq_len + v * seq_len + t;
                    let src_idx = partner_b * n_vars * seq_len + v * seq_len + t;
                    mixed_x[dst_idx] = x_values[src_idx];
                }
            }
        }

        // Mix Y values based on actual cut ratio (only for samples that were mixed)
        let mut mixed_y: Vec<f32> = vec![0.0; batch_size * y_cols];
        for b in 0..batch_size {
            let partner_b = mix_partner[b];
            for c in 0..y_cols {
                let idx = b * y_cols + c;
                let partner_idx = partner_b * y_cols + c;
                if partner_b == b {
                    // No mixing occurred
                    mixed_y[idx] = y_values[idx];
                } else {
                    mixed_y[idx] =
                        actual_lambda * y_values[idx] + (1.0 - actual_lambda) * y_values[partner_idx];
                }
            }
        }

        // Reconstruct tensors
        let mixed_x_tensor: Tensor<B, 3> = Tensor::<B, 1>::from_floats(mixed_x.as_slice(), &device)
            .reshape([batch_size, n_vars, seq_len]);

        let mixed_y_tensor: Tensor<B, 2> = Tensor::<B, 1>::from_floats(mixed_y.as_slice(), &device)
            .reshape([batch_size, y_cols]);

        Ok(TSBatch {
            x: TSTensor::new(mixed_x_tensor)?,
            y: Some(mixed_y_tensor),
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "IntraClassCutMix1d"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixup_config() {
        let config = MixUpConfig::default();
        assert_eq!(config.alpha, 0.4);
        assert_eq!(config.p, 1.0);
    }

    #[test]
    fn test_cutmix_config() {
        let config = CutMixConfig::default();
        assert_eq!(config.alpha, 1.0);
    }

    #[test]
    fn test_mixup_lambda_sampling() {
        let mixup = MixUp1d::new(0.4).with_seed(Seed::new(42));
        let mut rng = mixup.seed.to_rng();

        // Sample multiple lambdas and verify they're in [0, 1]
        for _ in 0..100 {
            let lambda = mixup.sample_lambda(&mut rng);
            assert!(lambda >= 0.0 && lambda <= 1.0);
        }
    }
}
