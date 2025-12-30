//! Time series augmentation transforms.
//!
//! This module provides various augmentation techniques for time series data,
//! matching the transforms available in Python tsai.

use burn::prelude::*;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

use tsai_core::{CoreError, Result, Seed, Split, TSBatch, TSTensor, Transform};

/// Configuration for Gaussian noise augmentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianNoiseConfig {
    /// Standard deviation of the noise.
    pub std: f32,
    /// Probability of applying the transform.
    pub p: f32,
}

impl Default for GaussianNoiseConfig {
    fn default() -> Self {
        Self { std: 0.1, p: 0.5 }
    }
}

/// Adds Gaussian noise to time series data.
///
/// Only applied during training.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_transforms::augment::GaussianNoise;
///
/// let transform = GaussianNoise::new(0.1);
/// ```
pub struct GaussianNoise {
    config: GaussianNoiseConfig,
    seed: Seed,
}

impl GaussianNoise {
    /// Create a new Gaussian noise transform.
    #[must_use]
    pub fn new(std: f32) -> Self {
        Self {
            config: GaussianNoiseConfig {
                std,
                ..Default::default()
            },
            seed: Seed::from_entropy(),
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: GaussianNoiseConfig) -> Self {
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

impl<B: Backend> Transform<B> for GaussianNoise {
    fn apply(&self, batch: TSBatch<B>, split: Split) -> Result<TSBatch<B>> {
        // Only apply during training
        if split.is_eval() {
            return Ok(batch);
        }

        let mut rng = self.seed.to_rng();

        // Check probability
        if rng.gen::<f32>() > self.config.p {
            return Ok(batch);
        }

        // Generate noise with same shape as input
        let shape = batch.x.shape();
        let noise_shape = [shape.batch(), shape.vars(), shape.len()];

        // Create random noise tensor
        let noise: Tensor<B, 3> = Tensor::random(
            noise_shape,
            burn::tensor::Distribution::Normal(0.0, self.config.std as f64),
            &batch.device(),
        );

        let noisy_x = batch.x.into_inner() + noise;

        Ok(TSBatch {
            x: tsai_core::TSTensor::new(noisy_x)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "GaussianNoise"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

/// Configuration for time warping augmentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWarpConfig {
    /// Maximum warping magnitude.
    pub magnitude: f32,
    /// Number of knots for the warping spline.
    pub n_knots: usize,
    /// Probability of applying the transform.
    pub p: f32,
}

impl Default for TimeWarpConfig {
    fn default() -> Self {
        Self {
            magnitude: 0.2,
            n_knots: 4,
            p: 0.5,
        }
    }
}

/// Warps the time axis of time series data.
///
/// Uses a smooth warping function to distort the temporal structure.
pub struct TimeWarp {
    config: TimeWarpConfig,
    seed: Seed,
}

impl TimeWarp {
    /// Create a new time warp transform.
    #[must_use]
    pub fn new(magnitude: f32) -> Self {
        Self {
            config: TimeWarpConfig {
                magnitude,
                ..Default::default()
            },
            seed: Seed::from_entropy(),
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: TimeWarpConfig) -> Self {
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

impl<B: Backend> Transform<B> for TimeWarp {
    fn apply(&self, batch: TSBatch<B>, split: Split) -> Result<TSBatch<B>> {
        if split.is_eval() {
            return Ok(batch);
        }

        let mut rng = self.seed.to_rng();

        if rng.gen::<f32>() > self.config.p {
            return Ok(batch);
        }

        let shape = batch.x.shape();
        let batch_size = shape.batch();
        let n_vars = shape.vars();
        let seq_len = shape.len();
        let device = batch.device();

        // Generate random warping path using piecewise linear interpolation
        // Create n_knots random perturbations and interpolate
        let n_knots = self.config.n_knots.max(2);

        // Knot positions (including start and end)
        let knot_positions: Vec<f32> = (0..n_knots)
            .map(|i| i as f32 / (n_knots - 1) as f32 * (seq_len - 1) as f32)
            .collect();

        // Generate random warping values at knots
        // Values represent how much to stretch/compress around each knot
        let mut warp_values: Vec<f32> = Vec::with_capacity(n_knots);
        warp_values.push(0.0); // Start at 0
        for _ in 1..n_knots - 1 {
            let perturbation: f32 = rng.gen_range(-self.config.magnitude..self.config.magnitude);
            warp_values.push(perturbation * seq_len as f32);
        }
        warp_values.push(0.0); // End at 0

        // Create warped indices through linear interpolation
        let mut warped_indices: Vec<f32> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let t_f = t as f32;

            // Find which knot segment we're in
            let mut segment = 0;
            for i in 0..n_knots - 1 {
                if t_f >= knot_positions[i] && t_f <= knot_positions[i + 1] {
                    segment = i;
                    break;
                }
            }

            // Linear interpolation of warp value
            let t0 = knot_positions[segment];
            let t1 = knot_positions[segment + 1];
            let w0 = warp_values[segment];
            let w1 = warp_values[segment + 1];

            let alpha = if (t1 - t0).abs() > 1e-6 {
                (t_f - t0) / (t1 - t0)
            } else {
                0.0
            };
            let warp_offset = w0 + alpha * (w1 - w0);

            // New index = original + warp, clamped to valid range
            let new_idx = (t_f + warp_offset).clamp(0.0, (seq_len - 1) as f32);
            warped_indices.push(new_idx);
        }

        // Apply warping using linear interpolation for sampling
        // Get data as raw values for manipulation
        let x_data = batch.x.into_inner().into_data();
        let x_values: Vec<f32> = x_data.as_slice().map_err(|e| {
            CoreError::ShapeMismatch(format!("Failed to get tensor data: {e:?}"))
        })?.to_vec();

        let mut warped_values: Vec<f32> = vec![0.0; batch_size * n_vars * seq_len];

        for b in 0..batch_size {
            for v in 0..n_vars {
                for t in 0..seq_len {
                    let new_idx = warped_indices[t];
                    let idx_low = new_idx.floor() as usize;
                    let idx_high = (idx_low + 1).min(seq_len - 1);
                    let frac = new_idx - new_idx.floor();

                    // Linear interpolation between adjacent values
                    let src_low = b * n_vars * seq_len + v * seq_len + idx_low;
                    let src_high = b * n_vars * seq_len + v * seq_len + idx_high;
                    let dst = b * n_vars * seq_len + v * seq_len + t;

                    let val_low = x_values[src_low];
                    let val_high = x_values[src_high];
                    warped_values[dst] = val_low + frac * (val_high - val_low);
                }
            }
        }

        let warped_tensor: Tensor<B, 3> = Tensor::<B, 1>::from_floats(
            warped_values.as_slice(),
            &device,
        )
        .reshape([batch_size, n_vars, seq_len]);

        Ok(TSBatch {
            x: tsai_core::TSTensor::new(warped_tensor)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "TimeWarp"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

/// Configuration for magnitude scaling augmentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MagScaleConfig {
    /// Maximum scale factor (values will be scaled between 1/factor and factor).
    pub factor: f32,
    /// Probability of applying the transform.
    pub p: f32,
}

impl Default for MagScaleConfig {
    fn default() -> Self {
        Self { factor: 1.2, p: 0.5 }
    }
}

/// Scales the magnitude of time series data.
pub struct MagScale {
    config: MagScaleConfig,
    seed: Seed,
}

impl MagScale {
    /// Create a new magnitude scaling transform.
    #[must_use]
    pub fn new(factor: f32) -> Self {
        Self {
            config: MagScaleConfig {
                factor,
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

impl<B: Backend> Transform<B> for MagScale {
    fn apply(&self, batch: TSBatch<B>, split: Split) -> Result<TSBatch<B>> {
        if split.is_eval() {
            return Ok(batch);
        }

        let mut rng = self.seed.to_rng();

        if rng.gen::<f32>() > self.config.p {
            return Ok(batch);
        }

        // Generate random scale factor
        let min_scale = 1.0 / self.config.factor;
        let max_scale = self.config.factor;
        let scale: f32 = rng.gen_range(min_scale..max_scale);

        let scaled_x = batch.x.into_inner() * scale;

        Ok(TSBatch {
            x: tsai_core::TSTensor::new(scaled_x)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "MagScale"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

/// Configuration for cutout augmentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CutOutConfig {
    /// Maximum length of the cutout region as a fraction of sequence length.
    pub max_cut_ratio: f32,
    /// Probability of applying the transform.
    pub p: f32,
    /// Value to fill the cutout region with.
    pub fill_value: f32,
}

impl Default for CutOutConfig {
    fn default() -> Self {
        Self {
            max_cut_ratio: 0.2,
            p: 0.5,
            fill_value: 0.0,
        }
    }
}

/// Cuts out a random contiguous region of the time series.
pub struct CutOut {
    config: CutOutConfig,
    seed: Seed,
}

impl CutOut {
    /// Create a new cutout transform.
    #[must_use]
    pub fn new(max_cut_ratio: f32) -> Self {
        Self {
            config: CutOutConfig {
                max_cut_ratio,
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

impl<B: Backend> Transform<B> for CutOut {
    fn apply(&self, batch: TSBatch<B>, split: Split) -> Result<TSBatch<B>> {
        if split.is_eval() {
            return Ok(batch);
        }

        let mut rng = self.seed.to_rng();

        if rng.gen::<f32>() > self.config.p {
            return Ok(batch);
        }

        let shape = batch.x.shape();
        let batch_size = shape.batch();
        let n_vars = shape.vars();
        let seq_len = shape.len();
        let device = batch.device();

        // Calculate cutout length
        let cut_len = ((seq_len as f32 * self.config.max_cut_ratio) * rng.gen::<f32>()) as usize;
        let cut_len = cut_len.max(1).min(seq_len);

        // Random start position for each sample in batch
        let start = rng.gen_range(0..seq_len.saturating_sub(cut_len).max(1));
        let end = (start + cut_len).min(seq_len);

        // Get data as raw values for manipulation
        let x_data = batch.x.into_inner().into_data();
        let mut x_values: Vec<f32> = x_data.as_slice().map_err(|e| {
            CoreError::ShapeMismatch(format!("Failed to get tensor data: {e:?}"))
        })?.to_vec();

        // Apply cutout by setting values in the cut region to fill_value
        for b in 0..batch_size {
            for v in 0..n_vars {
                for t in start..end {
                    let idx = b * n_vars * seq_len + v * seq_len + t;
                    x_values[idx] = self.config.fill_value;
                }
            }
        }

        let cutout_tensor: Tensor<B, 3> = Tensor::<B, 1>::from_floats(
            x_values.as_slice(),
            &device,
        )
        .reshape([batch_size, n_vars, seq_len]);

        Ok(TSBatch {
            x: tsai_core::TSTensor::new(cutout_tensor)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "CutOut"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

/// Configuration for magnitude warping augmentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MagWarpConfig {
    /// Standard deviation of the warping curve.
    pub sigma: f32,
    /// Number of knots for the warping spline.
    pub n_knots: usize,
    /// Probability of applying the transform.
    pub p: f32,
}

impl Default for MagWarpConfig {
    fn default() -> Self {
        Self {
            sigma: 0.2,
            n_knots: 4,
            p: 0.5,
        }
    }
}

/// Warps the magnitude of time series data using smooth curves.
///
/// Multiplies the time series by a smooth random curve, creating
/// amplitude variations while preserving the overall pattern.
pub struct MagWarp {
    config: MagWarpConfig,
    seed: Seed,
}

impl MagWarp {
    /// Create a new magnitude warp transform.
    #[must_use]
    pub fn new(sigma: f32) -> Self {
        Self {
            config: MagWarpConfig {
                sigma,
                ..Default::default()
            },
            seed: Seed::from_entropy(),
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: MagWarpConfig) -> Self {
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

impl<B: Backend> Transform<B> for MagWarp {
    fn apply(&self, batch: TSBatch<B>, split: Split) -> Result<TSBatch<B>> {
        if split.is_eval() {
            return Ok(batch);
        }

        let mut rng = self.seed.to_rng();

        if rng.gen::<f32>() > self.config.p {
            return Ok(batch);
        }

        let shape = batch.x.shape();
        let batch_size = shape.batch();
        let n_vars = shape.vars();
        let seq_len = shape.len();
        let device = batch.device();

        // Generate smooth warping curve using cubic interpolation
        let n_knots = self.config.n_knots.max(2);

        // Knot positions (evenly spaced)
        let knot_positions: Vec<f32> = (0..n_knots)
            .map(|i| i as f32 / (n_knots - 1) as f32 * (seq_len - 1) as f32)
            .collect();

        // Generate random values at knots (centered around 1.0)
        let mut knot_values: Vec<f32> = Vec::with_capacity(n_knots);
        for _ in 0..n_knots {
            // Sample from normal distribution centered at 1.0
            let u1: f32 = rng.gen();
            let u2: f32 = rng.gen();
            // Box-Muller transform for normal distribution
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            knot_values.push(1.0 + z * self.config.sigma);
        }

        // Create warping curve through linear interpolation
        let mut warp_curve: Vec<f32> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let t_f = t as f32;

            // Find segment
            let mut segment = 0;
            for i in 0..n_knots - 1 {
                if t_f >= knot_positions[i] && t_f <= knot_positions[i + 1] {
                    segment = i;
                    break;
                }
            }

            // Linear interpolation
            let t0 = knot_positions[segment];
            let t1 = knot_positions[segment + 1];
            let v0 = knot_values[segment];
            let v1 = knot_values[segment + 1];

            let alpha = if (t1 - t0).abs() > 1e-6 {
                (t_f - t0) / (t1 - t0)
            } else {
                0.0
            };
            warp_curve.push(v0 + alpha * (v1 - v0));
        }

        // Apply warping to data
        let x_data = batch.x.into_inner().into_data();
        let x_values: Vec<f32> = x_data
            .as_slice()
            .map_err(|e| CoreError::ShapeMismatch(format!("Failed to get tensor data: {e:?}")))?
            .to_vec();

        let mut warped_values: Vec<f32> = vec![0.0; batch_size * n_vars * seq_len];

        for b in 0..batch_size {
            for v in 0..n_vars {
                for t in 0..seq_len {
                    let idx = b * n_vars * seq_len + v * seq_len + t;
                    warped_values[idx] = x_values[idx] * warp_curve[t];
                }
            }
        }

        let warped_tensor: Tensor<B, 3> = Tensor::<B, 1>::from_floats(warped_values.as_slice(), &device)
            .reshape([batch_size, n_vars, seq_len]);

        Ok(TSBatch {
            x: tsai_core::TSTensor::new(warped_tensor)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "MagWarp"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

/// Configuration for window warping augmentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowWarpConfig {
    /// Window ratio (fraction of sequence to warp).
    pub window_ratio: f32,
    /// Warp scales (compression/expansion factors to choose from).
    pub scales: Vec<f32>,
    /// Probability of applying the transform.
    pub p: f32,
}

impl Default for WindowWarpConfig {
    fn default() -> Self {
        Self {
            window_ratio: 0.1,
            scales: vec![0.5, 2.0],
            p: 0.5,
        }
    }
}

/// Warps a random window of the time series by compressing or expanding it.
///
/// Selects a random segment and either compresses or expands it in time,
/// while padding/cropping to maintain the original sequence length.
pub struct WindowWarp {
    config: WindowWarpConfig,
    seed: Seed,
}

impl WindowWarp {
    /// Create a new window warp transform.
    #[must_use]
    pub fn new(window_ratio: f32) -> Self {
        Self {
            config: WindowWarpConfig {
                window_ratio,
                ..Default::default()
            },
            seed: Seed::from_entropy(),
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: WindowWarpConfig) -> Self {
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

impl<B: Backend> Transform<B> for WindowWarp {
    fn apply(&self, batch: TSBatch<B>, split: Split) -> Result<TSBatch<B>> {
        if split.is_eval() {
            return Ok(batch);
        }

        let mut rng = self.seed.to_rng();

        if rng.gen::<f32>() > self.config.p {
            return Ok(batch);
        }

        if self.config.scales.is_empty() {
            return Ok(batch);
        }

        let shape = batch.x.shape();
        let batch_size = shape.batch();
        let n_vars = shape.vars();
        let seq_len = shape.len();
        let device = batch.device();

        // Select random scale
        let scale_idx = rng.gen_range(0..self.config.scales.len());
        let scale = self.config.scales[scale_idx];

        // Calculate window size
        let window_len = ((seq_len as f32 * self.config.window_ratio).round() as usize).max(2);
        let window_start = rng.gen_range(0..seq_len.saturating_sub(window_len).max(1));
        let window_end = (window_start + window_len).min(seq_len);

        // Get data
        let x_data = batch.x.into_inner().into_data();
        let x_values: Vec<f32> = x_data
            .as_slice()
            .map_err(|e| CoreError::ShapeMismatch(format!("Failed to get tensor data: {e:?}")))?
            .to_vec();

        let mut warped_values: Vec<f32> = vec![0.0; batch_size * n_vars * seq_len];

        for b in 0..batch_size {
            for v in 0..n_vars {
                // Extract the window
                let mut window: Vec<f32> = Vec::with_capacity(window_len);
                for t in window_start..window_end {
                    let idx = b * n_vars * seq_len + v * seq_len + t;
                    window.push(x_values[idx]);
                }

                // Resample the window (linear interpolation)
                let new_window_len = ((window_len as f32 * scale).round() as usize).max(1);
                let mut resampled: Vec<f32> = Vec::with_capacity(new_window_len);

                for i in 0..new_window_len {
                    let src_idx = (i as f32 / new_window_len as f32) * (window.len() - 1) as f32;
                    let idx_low = src_idx.floor() as usize;
                    let idx_high = (idx_low + 1).min(window.len() - 1);
                    let frac = src_idx - src_idx.floor();
                    resampled.push(window[idx_low] * (1.0 - frac) + window[idx_high] * frac);
                }

                // Reconstruct the sequence
                let mut new_seq: Vec<f32> = Vec::with_capacity(seq_len);

                // Before window
                for t in 0..window_start {
                    let idx = b * n_vars * seq_len + v * seq_len + t;
                    new_seq.push(x_values[idx]);
                }

                // Warped window
                for val in &resampled {
                    new_seq.push(*val);
                }

                // After window
                for t in window_end..seq_len {
                    let idx = b * n_vars * seq_len + v * seq_len + t;
                    new_seq.push(x_values[idx]);
                }

                // Resample to original length if needed
                let actual_len = new_seq.len();
                for t in 0..seq_len {
                    let src_idx = (t as f32 / seq_len as f32) * (actual_len - 1) as f32;
                    let idx_low = src_idx.floor() as usize;
                    let idx_high = (idx_low + 1).min(actual_len - 1);
                    let frac = src_idx - src_idx.floor();
                    let val = new_seq[idx_low] * (1.0 - frac) + new_seq[idx_high] * frac;

                    let dst = b * n_vars * seq_len + v * seq_len + t;
                    warped_values[dst] = val;
                }
            }
        }

        let warped_tensor: Tensor<B, 3> =
            Tensor::<B, 1>::from_floats(warped_values.as_slice(), &device)
                .reshape([batch_size, n_vars, seq_len]);

        Ok(TSBatch {
            x: tsai_core::TSTensor::new(warped_tensor)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "WindowWarp"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

/// Configuration for horizontal flip augmentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HorizontalFlipConfig {
    /// Probability of applying the transform.
    pub p: f32,
}

impl Default for HorizontalFlipConfig {
    fn default() -> Self {
        Self { p: 0.5 }
    }
}

/// Flips the time series horizontally (reverses time axis).
///
/// This is equivalent to viewing the time series backwards in time.
/// Can help the model learn time-invariant features.
pub struct HorizontalFlip {
    config: HorizontalFlipConfig,
    seed: Seed,
}

impl HorizontalFlip {
    /// Create a new horizontal flip transform.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: HorizontalFlipConfig::default(),
            seed: Seed::from_entropy(),
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: HorizontalFlipConfig) -> Self {
        Self {
            config,
            seed: Seed::from_entropy(),
        }
    }

    /// Set probability.
    #[must_use]
    pub fn with_p(mut self, p: f32) -> Self {
        self.config.p = p;
        self
    }

    /// Set the random seed.
    #[must_use]
    pub fn with_seed(mut self, seed: Seed) -> Self {
        self.seed = seed;
        self
    }
}

impl Default for HorizontalFlip {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Transform<B> for HorizontalFlip {
    fn apply(&self, batch: TSBatch<B>, split: Split) -> Result<TSBatch<B>> {
        if split.is_eval() {
            return Ok(batch);
        }

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
            .map_err(|e| CoreError::ShapeMismatch(format!("Failed to get tensor data: {e:?}")))?
            .to_vec();

        // Flip each sequence
        let mut flipped_values: Vec<f32> = vec![0.0; batch_size * n_vars * seq_len];

        for b in 0..batch_size {
            for v in 0..n_vars {
                for t in 0..seq_len {
                    let src = b * n_vars * seq_len + v * seq_len + (seq_len - 1 - t);
                    let dst = b * n_vars * seq_len + v * seq_len + t;
                    flipped_values[dst] = x_values[src];
                }
            }
        }

        let flipped_tensor: Tensor<B, 3> =
            Tensor::<B, 1>::from_floats(flipped_values.as_slice(), &device)
                .reshape([batch_size, n_vars, seq_len]);

        Ok(TSBatch {
            x: tsai_core::TSTensor::new(flipped_tensor)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "HorizontalFlip"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

/// Configuration for random shift augmentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomShiftConfig {
    /// Maximum shift as a fraction of sequence length.
    pub max_shift: f32,
    /// Whether to shift forward, backward, or both.
    pub direction: ShiftDirection,
    /// Value to fill empty positions after shift.
    pub fill_value: f32,
    /// Probability of applying the transform.
    pub p: f32,
}

/// Direction of shift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShiftDirection {
    /// Only shift forward (positive direction).
    Forward,
    /// Only shift backward (negative direction).
    Backward,
    /// Randomly shift in either direction.
    Both,
}

impl Default for RandomShiftConfig {
    fn default() -> Self {
        Self {
            max_shift: 0.2,
            direction: ShiftDirection::Both,
            fill_value: 0.0,
            p: 0.5,
        }
    }
}

/// Randomly shifts the time series along the time axis.
///
/// Shifts the time series left or right, filling empty positions
/// with a specified value (default: 0).
pub struct RandomShift {
    config: RandomShiftConfig,
    seed: Seed,
}

impl RandomShift {
    /// Create a new random shift transform.
    #[must_use]
    pub fn new(max_shift: f32) -> Self {
        Self {
            config: RandomShiftConfig {
                max_shift,
                ..Default::default()
            },
            seed: Seed::from_entropy(),
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: RandomShiftConfig) -> Self {
        Self {
            config,
            seed: Seed::from_entropy(),
        }
    }

    /// Set the shift direction.
    #[must_use]
    pub fn with_direction(mut self, direction: ShiftDirection) -> Self {
        self.config.direction = direction;
        self
    }

    /// Set the fill value.
    #[must_use]
    pub fn with_fill_value(mut self, fill_value: f32) -> Self {
        self.config.fill_value = fill_value;
        self
    }

    /// Set the random seed.
    #[must_use]
    pub fn with_seed(mut self, seed: Seed) -> Self {
        self.seed = seed;
        self
    }
}

impl<B: Backend> Transform<B> for RandomShift {
    fn apply(&self, batch: TSBatch<B>, split: Split) -> Result<TSBatch<B>> {
        if split.is_eval() {
            return Ok(batch);
        }

        let mut rng = self.seed.to_rng();

        if rng.gen::<f32>() > self.config.p {
            return Ok(batch);
        }

        let shape = batch.x.shape();
        let batch_size = shape.batch();
        let n_vars = shape.vars();
        let seq_len = shape.len();
        let device = batch.device();

        // Calculate shift amount
        let max_shift_steps = (seq_len as f32 * self.config.max_shift).round() as i32;
        let shift_amount: i32 = match self.config.direction {
            ShiftDirection::Forward => rng.gen_range(0..=max_shift_steps),
            ShiftDirection::Backward => -rng.gen_range(0..=max_shift_steps),
            ShiftDirection::Both => rng.gen_range(-max_shift_steps..=max_shift_steps),
        };

        if shift_amount == 0 {
            return Ok(batch);
        }

        // Get data as raw values
        let x_data = batch.x.into_inner().into_data();
        let x_values: Vec<f32> = x_data
            .as_slice()
            .map_err(|e| CoreError::ShapeMismatch(format!("Failed to get tensor data: {e:?}")))?
            .to_vec();

        // Apply shift
        let mut shifted_values: Vec<f32> = vec![self.config.fill_value; batch_size * n_vars * seq_len];

        for b in 0..batch_size {
            for v in 0..n_vars {
                for t in 0..seq_len {
                    let src_t = t as i32 - shift_amount;
                    if src_t >= 0 && src_t < seq_len as i32 {
                        let src = b * n_vars * seq_len + v * seq_len + src_t as usize;
                        let dst = b * n_vars * seq_len + v * seq_len + t;
                        shifted_values[dst] = x_values[src];
                    }
                }
            }
        }

        let shifted_tensor: Tensor<B, 3> =
            Tensor::<B, 1>::from_floats(shifted_values.as_slice(), &device)
                .reshape([batch_size, n_vars, seq_len]);

        Ok(TSBatch {
            x: tsai_core::TSTensor::new(shifted_tensor)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "RandomShift"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

/// Configuration for permutation augmentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermutationConfig {
    /// Number of segments to permute.
    pub n_segments: usize,
    /// Probability of applying the transform.
    pub p: f32,
}

impl Default for PermutationConfig {
    fn default() -> Self {
        Self {
            n_segments: 5,
            p: 0.5,
        }
    }
}

/// Randomly permutes segments of the time series.
///
/// Divides the time series into n_segments and randomly shuffles their order.
/// This helps the model learn features that are invariant to temporal ordering
/// of local patterns.
pub struct Permutation {
    config: PermutationConfig,
    seed: Seed,
}

impl Permutation {
    /// Create a new permutation transform.
    #[must_use]
    pub fn new(n_segments: usize) -> Self {
        Self {
            config: PermutationConfig {
                n_segments,
                ..Default::default()
            },
            seed: Seed::from_entropy(),
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: PermutationConfig) -> Self {
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

impl<B: Backend> Transform<B> for Permutation {
    fn apply(&self, batch: TSBatch<B>, split: Split) -> Result<TSBatch<B>> {
        if split.is_eval() {
            return Ok(batch);
        }

        let mut rng = self.seed.to_rng();

        if rng.gen::<f32>() > self.config.p {
            return Ok(batch);
        }

        let shape = batch.x.shape();
        let batch_size = shape.batch();
        let n_vars = shape.vars();
        let seq_len = shape.len();
        let device = batch.device();

        let n_segments = self.config.n_segments.min(seq_len).max(1);
        let segment_len = seq_len / n_segments;

        if segment_len == 0 {
            return Ok(batch);
        }

        // Get data
        let x_data = batch.x.into_inner().into_data();
        let x_values: Vec<f32> = x_data
            .as_slice()
            .map_err(|e| CoreError::ShapeMismatch(format!("Failed to get tensor data: {e:?}")))?
            .to_vec();

        // Create permutation order
        let mut perm_order: Vec<usize> = (0..n_segments).collect();

        // Fisher-Yates shuffle
        for i in (1..n_segments).rev() {
            let j = rng.gen_range(0..=i);
            perm_order.swap(i, j);
        }

        let mut permuted_values: Vec<f32> = vec![0.0; batch_size * n_vars * seq_len];

        for b in 0..batch_size {
            for v in 0..n_vars {
                let mut dest_t = 0;

                // Copy segments in permuted order
                for &seg_idx in &perm_order {
                    let seg_start = seg_idx * segment_len;
                    let seg_end = if seg_idx == n_segments - 1 {
                        seq_len // Last segment gets remainder
                    } else {
                        seg_start + segment_len
                    };

                    for src_t in seg_start..seg_end {
                        if dest_t < seq_len {
                            let src = b * n_vars * seq_len + v * seq_len + src_t;
                            let dst = b * n_vars * seq_len + v * seq_len + dest_t;
                            permuted_values[dst] = x_values[src];
                            dest_t += 1;
                        }
                    }
                }
            }
        }

        let permuted_tensor: Tensor<B, 3> =
            Tensor::<B, 1>::from_floats(permuted_values.as_slice(), &device)
                .reshape([batch_size, n_vars, seq_len]);

        Ok(TSBatch {
            x: tsai_core::TSTensor::new(permuted_tensor)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "Permutation"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

/// Configuration for rotation (circular shift) augmentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotationConfig {
    /// Maximum rotation as a fraction of sequence length.
    pub max_rotation: f32,
    /// Probability of applying the transform.
    pub p: f32,
}

impl Default for RotationConfig {
    fn default() -> Self {
        Self {
            max_rotation: 0.5,
            p: 0.5,
        }
    }
}

/// Rotates (circularly shifts) the time series.
///
/// Unlike RandomShift which pads with zeros, rotation wraps values around.
pub struct Rotation {
    config: RotationConfig,
    seed: Seed,
}

impl Rotation {
    /// Create a new rotation transform.
    #[must_use]
    pub fn new(max_rotation: f32) -> Self {
        Self {
            config: RotationConfig {
                max_rotation,
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

impl<B: Backend> Transform<B> for Rotation {
    fn apply(&self, batch: TSBatch<B>, split: Split) -> Result<TSBatch<B>> {
        if split.is_eval() {
            return Ok(batch);
        }

        let mut rng = self.seed.to_rng();

        if rng.gen::<f32>() > self.config.p {
            return Ok(batch);
        }

        let shape = batch.x.shape();
        let batch_size = shape.batch();
        let n_vars = shape.vars();
        let seq_len = shape.len();
        let device = batch.device();

        // Calculate rotation amount
        let max_shift = (seq_len as f32 * self.config.max_rotation).round() as i32;
        let shift: i32 = rng.gen_range(-max_shift..=max_shift);

        if shift == 0 {
            return Ok(batch);
        }

        // Get data
        let x_data = batch.x.into_inner().into_data();
        let x_values: Vec<f32> = x_data
            .as_slice()
            .map_err(|e| CoreError::ShapeMismatch(format!("Failed to get tensor data: {e:?}")))?
            .to_vec();

        // Apply circular rotation
        let mut rotated_values: Vec<f32> = vec![0.0; batch_size * n_vars * seq_len];

        for b in 0..batch_size {
            for v in 0..n_vars {
                for t in 0..seq_len {
                    // Calculate source index with wraparound
                    let src_t = ((t as i32 - shift) % seq_len as i32 + seq_len as i32) as usize % seq_len;
                    let src = b * n_vars * seq_len + v * seq_len + src_t;
                    let dst = b * n_vars * seq_len + v * seq_len + t;
                    rotated_values[dst] = x_values[src];
                }
            }
        }

        let rotated_tensor: Tensor<B, 3> =
            Tensor::<B, 1>::from_floats(rotated_values.as_slice(), &device)
                .reshape([batch_size, n_vars, seq_len]);

        Ok(TSBatch {
            x: tsai_core::TSTensor::new(rotated_tensor)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "Rotation"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

/// Identity transform that passes through data unchanged.
#[derive(Debug, Clone, Default)]
pub struct Identity;

impl<B: Backend> Transform<B> for Identity {
    fn apply(&self, batch: TSBatch<B>, _split: Split) -> Result<TSBatch<B>> {
        Ok(batch)
    }

    fn name(&self) -> &str {
        "Identity"
    }
}

// ============================================================================
// SpecAugment - Audio/Spectrogram Augmentation
// ============================================================================

/// Configuration for frequency masking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyMaskConfig {
    /// Maximum number of frequency channels to mask.
    pub freq_mask_param: usize,
    /// Number of frequency masks to apply.
    pub num_masks: usize,
    /// Probability of applying the transform.
    pub p: f32,
    /// Value to use for masking (default: 0.0).
    pub mask_value: f32,
}

impl Default for FrequencyMaskConfig {
    fn default() -> Self {
        Self {
            freq_mask_param: 27,
            num_masks: 1,
            p: 0.5,
            mask_value: 0.0,
        }
    }
}

/// Frequency masking for SpecAugment.
///
/// Masks a random band of consecutive frequency channels (variable dimension).
/// Commonly used for audio spectrogram augmentation.
///
/// Reference: Park et al., "SpecAugment: A Simple Data Augmentation Method for ASR", 2019.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_transforms::augment::FrequencyMask;
///
/// // Mask up to 15 frequency channels, apply twice
/// let transform = FrequencyMask::new(15).with_num_masks(2);
/// ```
#[derive(Debug, Clone)]
pub struct FrequencyMask {
    /// Configuration for the transform.
    pub config: FrequencyMaskConfig,
    seed: Seed,
}

impl FrequencyMask {
    /// Create a new frequency mask transform.
    #[must_use]
    pub fn new(freq_mask_param: usize) -> Self {
        Self {
            config: FrequencyMaskConfig {
                freq_mask_param,
                ..Default::default()
            },
            seed: Seed::default(),
        }
    }

    /// Set the number of masks to apply.
    #[must_use]
    pub fn with_num_masks(mut self, num_masks: usize) -> Self {
        self.config.num_masks = num_masks;
        self
    }

    /// Set the probability of applying the transform.
    #[must_use]
    pub fn with_probability(mut self, p: f32) -> Self {
        self.config.p = p;
        self
    }

    /// Set the mask value.
    #[must_use]
    pub fn with_mask_value(mut self, value: f32) -> Self {
        self.config.mask_value = value;
        self
    }

    /// Set the random seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Seed::new(seed);
        self
    }
}

impl<B: Backend> Transform<B> for FrequencyMask {
    fn apply(&self, batch: TSBatch<B>, split: Split) -> Result<TSBatch<B>> {
        if !split.is_train() {
            return Ok(batch);
        }

        let mut rng = self.seed.to_rng();

        // Check if we should apply
        if rng.gen::<f32>() > self.config.p {
            return Ok(batch);
        }

        let shape = batch.x.shape();
        let batch_size = shape.batch();
        let n_vars = shape.vars();
        let seq_len = shape.len();
        let device = batch.device();

        // We'll mask along the variable (frequency) dimension
        if n_vars == 0 || self.config.freq_mask_param == 0 {
            return Ok(batch);
        }

        // Start with zeros tensor and add ones where we don't mask
        let mut mask_data = vec![1.0f32; batch_size * n_vars * seq_len];

        for b in 0..batch_size {
            for _ in 0..self.config.num_masks {
                // Random mask width (0 to freq_mask_param)
                let mask_width = rng.gen_range(0..=self.config.freq_mask_param.min(n_vars));
                if mask_width == 0 {
                    continue;
                }

                // Random start position
                let max_start = n_vars.saturating_sub(mask_width);
                let start = if max_start > 0 {
                    rng.gen_range(0..=max_start)
                } else {
                    0
                };

                // Apply mask to this batch sample
                for v in start..(start + mask_width).min(n_vars) {
                    for t in 0..seq_len {
                        mask_data[b * n_vars * seq_len + v * seq_len + t] = 0.0;
                    }
                }
            }
        }

        let mask_tensor = Tensor::<B, 1>::from_floats(mask_data.as_slice(), &device)
            .reshape([batch_size, n_vars, seq_len]);

        // Apply mask: x * mask + mask_value * (1 - mask)
        let mask_value = Tensor::full([batch_size, n_vars, seq_len], self.config.mask_value, &device);
        let ones = Tensor::ones([batch_size, n_vars, seq_len], &device);
        let inv_mask = ones - mask_tensor.clone();
        let x_inner = batch.x.into_inner();
        let x = x_inner * mask_tensor + mask_value * inv_mask;

        Ok(TSBatch {
            x: TSTensor::new(x)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "FrequencyMask"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

/// Configuration for time masking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeMaskConfig {
    /// Maximum number of time steps to mask.
    pub time_mask_param: usize,
    /// Number of time masks to apply.
    pub num_masks: usize,
    /// Probability of applying the transform.
    pub p: f32,
    /// Value to use for masking (default: 0.0).
    pub mask_value: f32,
}

impl Default for TimeMaskConfig {
    fn default() -> Self {
        Self {
            time_mask_param: 100,
            num_masks: 1,
            p: 0.5,
            mask_value: 0.0,
        }
    }
}

/// Time masking for SpecAugment.
///
/// Masks a random band of consecutive time steps (sequence dimension).
/// Commonly used for audio spectrogram augmentation.
///
/// Reference: Park et al., "SpecAugment: A Simple Data Augmentation Method for ASR", 2019.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_transforms::augment::TimeMask;
///
/// // Mask up to 50 time steps, apply twice
/// let transform = TimeMask::new(50).with_num_masks(2);
/// ```
#[derive(Debug, Clone)]
pub struct TimeMask {
    /// Configuration for the transform.
    pub config: TimeMaskConfig,
    seed: Seed,
}

impl TimeMask {
    /// Create a new time mask transform.
    #[must_use]
    pub fn new(time_mask_param: usize) -> Self {
        Self {
            config: TimeMaskConfig {
                time_mask_param,
                ..Default::default()
            },
            seed: Seed::default(),
        }
    }

    /// Set the number of masks to apply.
    #[must_use]
    pub fn with_num_masks(mut self, num_masks: usize) -> Self {
        self.config.num_masks = num_masks;
        self
    }

    /// Set the probability of applying the transform.
    #[must_use]
    pub fn with_probability(mut self, p: f32) -> Self {
        self.config.p = p;
        self
    }

    /// Set the mask value.
    #[must_use]
    pub fn with_mask_value(mut self, value: f32) -> Self {
        self.config.mask_value = value;
        self
    }

    /// Set the random seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Seed::new(seed);
        self
    }
}

impl<B: Backend> Transform<B> for TimeMask {
    fn apply(&self, batch: TSBatch<B>, split: Split) -> Result<TSBatch<B>> {
        if !split.is_train() {
            return Ok(batch);
        }

        let mut rng = self.seed.to_rng();

        // Check if we should apply
        if rng.gen::<f32>() > self.config.p {
            return Ok(batch);
        }

        let shape = batch.x.shape();
        let batch_size = shape.batch();
        let n_vars = shape.vars();
        let seq_len = shape.len();
        let device = batch.device();

        // We'll mask along the time (sequence) dimension
        if seq_len == 0 || self.config.time_mask_param == 0 {
            return Ok(batch);
        }

        // Start with ones tensor
        let mut mask_data = vec![1.0f32; batch_size * n_vars * seq_len];

        for b in 0..batch_size {
            for _ in 0..self.config.num_masks {
                // Random mask width (0 to time_mask_param)
                let mask_width = rng.gen_range(0..=self.config.time_mask_param.min(seq_len));
                if mask_width == 0 {
                    continue;
                }

                // Random start position
                let max_start = seq_len.saturating_sub(mask_width);
                let start = if max_start > 0 {
                    rng.gen_range(0..=max_start)
                } else {
                    0
                };

                // Apply mask to all variables at this time range
                for v in 0..n_vars {
                    for t in start..(start + mask_width).min(seq_len) {
                        mask_data[b * n_vars * seq_len + v * seq_len + t] = 0.0;
                    }
                }
            }
        }

        let mask_tensor = Tensor::<B, 1>::from_floats(mask_data.as_slice(), &device)
            .reshape([batch_size, n_vars, seq_len]);

        // Apply mask: x * mask + mask_value * (1 - mask)
        let mask_value = Tensor::full([batch_size, n_vars, seq_len], self.config.mask_value, &device);
        let ones = Tensor::ones([batch_size, n_vars, seq_len], &device);
        let inv_mask = ones - mask_tensor.clone();
        let x_inner = batch.x.into_inner();
        let x = x_inner * mask_tensor + mask_value * inv_mask;

        Ok(TSBatch {
            x: TSTensor::new(x)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "TimeMask"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

/// Configuration for SpecAugment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecAugmentConfig {
    /// Frequency mask configuration.
    pub freq_mask: FrequencyMaskConfig,
    /// Time mask configuration.
    pub time_mask: TimeMaskConfig,
}

impl Default for SpecAugmentConfig {
    fn default() -> Self {
        Self {
            freq_mask: FrequencyMaskConfig::default(),
            time_mask: TimeMaskConfig::default(),
        }
    }
}

/// SpecAugment: Combined frequency and time masking.
///
/// Applies both frequency masking and time masking as described in the SpecAugment paper.
/// This is a convenience wrapper that combines FrequencyMask and TimeMask.
///
/// Reference: Park et al., "SpecAugment: A Simple Data Augmentation Method for ASR", 2019.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_transforms::augment::SpecAugment;
///
/// // Create SpecAugment with default settings
/// let transform = SpecAugment::new(27, 100)
///     .with_freq_masks(2)
///     .with_time_masks(2);
/// ```
#[derive(Debug, Clone)]
pub struct SpecAugment {
    /// Configuration for the transform.
    pub config: SpecAugmentConfig,
    seed: Seed,
}

impl SpecAugment {
    /// Create a new SpecAugment transform.
    ///
    /// # Arguments
    /// * `freq_mask_param` - Maximum frequency mask width
    /// * `time_mask_param` - Maximum time mask width
    #[must_use]
    pub fn new(freq_mask_param: usize, time_mask_param: usize) -> Self {
        Self {
            config: SpecAugmentConfig {
                freq_mask: FrequencyMaskConfig {
                    freq_mask_param,
                    ..Default::default()
                },
                time_mask: TimeMaskConfig {
                    time_mask_param,
                    ..Default::default()
                },
            },
            seed: Seed::default(),
        }
    }

    /// Set the number of frequency masks to apply.
    #[must_use]
    pub fn with_freq_masks(mut self, num: usize) -> Self {
        self.config.freq_mask.num_masks = num;
        self
    }

    /// Set the number of time masks to apply.
    #[must_use]
    pub fn with_time_masks(mut self, num: usize) -> Self {
        self.config.time_mask.num_masks = num;
        self
    }

    /// Set the mask value for both frequency and time masks.
    #[must_use]
    pub fn with_mask_value(mut self, value: f32) -> Self {
        self.config.freq_mask.mask_value = value;
        self.config.time_mask.mask_value = value;
        self
    }

    /// Set the probability for frequency masking.
    #[must_use]
    pub fn with_freq_probability(mut self, p: f32) -> Self {
        self.config.freq_mask.p = p;
        self
    }

    /// Set the probability for time masking.
    #[must_use]
    pub fn with_time_probability(mut self, p: f32) -> Self {
        self.config.time_mask.p = p;
        self
    }

    /// Set the random seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Seed::new(seed);
        self
    }
}

impl<B: Backend> Transform<B> for SpecAugment {
    fn apply(&self, batch: TSBatch<B>, split: Split) -> Result<TSBatch<B>> {
        if !split.is_train() {
            return Ok(batch);
        }

        // Apply frequency masking
        let freq_mask = FrequencyMask {
            config: self.config.freq_mask.clone(),
            seed: self.seed.clone(),
        };
        let batch = freq_mask.apply(batch, split)?;

        // Apply time masking (with different seed offset)
        let time_mask = TimeMask {
            config: self.config.time_mask.clone(),
            seed: Seed::new(self.seed.value().wrapping_add(1)),
        };
        time_mask.apply(batch, split)
    }

    fn name(&self) -> &str {
        "SpecAugment"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

/// Compose multiple transforms together.
#[derive(Default)]
pub struct Compose<B: Backend> {
    transforms: Vec<Box<dyn Transform<B>>>,
}

impl<B: Backend> Compose<B> {
    /// Create a new empty composition.
    #[must_use]
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    /// Add a transform to the composition.
    pub fn add<T: Transform<B> + 'static>(mut self, transform: T) -> Self {
        self.transforms.push(Box::new(transform));
        self
    }
}

impl<B: Backend> Transform<B> for Compose<B> {
    fn apply(&self, mut batch: TSBatch<B>, split: Split) -> Result<TSBatch<B>> {
        for transform in &self.transforms {
            if transform.should_apply(split) {
                batch = transform.apply(batch, split)?;
            }
        }
        Ok(batch)
    }

    fn name(&self) -> &str {
        "Compose"
    }
}

// ============================================================================
// Temporal Transformations
// ============================================================================

/// Configuration for random circular shift transform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TSRandomShiftConfig {
    /// Maximum shift as fraction of sequence length (e.g., 0.5 = up to 50% shift).
    pub max_shift_frac: f32,
    /// Probability of applying the transform.
    pub p: f32,
}

impl Default for TSRandomShiftConfig {
    fn default() -> Self {
        Self {
            max_shift_frac: 0.5,
            p: 0.5,
        }
    }
}

/// Random circular shift augmentation.
///
/// Applies a random circular shift to the time axis, wrapping values
/// from the end to the beginning (or vice versa). This simulates
/// different starting points in the time series.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_transforms::TSRandomShift;
///
/// let transform = TSRandomShift::new(0.3)  // Up to 30% shift
///     .with_probability(0.5);
/// ```
pub struct TSRandomShift {
    config: TSRandomShiftConfig,
    seed: Seed,
}

impl TSRandomShift {
    /// Create a new random shift transform.
    ///
    /// # Arguments
    ///
    /// * `max_shift_frac` - Maximum shift as fraction of sequence length
    #[must_use]
    pub fn new(max_shift_frac: f32) -> Self {
        Self {
            config: TSRandomShiftConfig {
                max_shift_frac,
                ..Default::default()
            },
            seed: Seed::from_entropy(),
        }
    }

    /// Set the probability of applying the transform.
    #[must_use]
    pub fn with_probability(mut self, p: f32) -> Self {
        self.config.p = p;
        self
    }

    /// Set the random seed.
    #[must_use]
    pub fn with_seed(mut self, seed: Seed) -> Self {
        self.seed = seed;
        self
    }
}

impl<B: Backend> Transform<B> for TSRandomShift {
    fn apply(&self, batch: TSBatch<B>, split: Split) -> Result<TSBatch<B>> {
        if split.is_eval() {
            return Ok(batch);
        }

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

        // Maximum shift in samples
        let max_shift = (self.config.max_shift_frac * seq_len as f32).round() as i32;
        if max_shift == 0 {
            return Ok(TSBatch {
                x: TSTensor::new(
                    Tensor::<B, 1>::from_floats(x_values.as_slice(), &device)
                        .reshape([batch_size, n_vars, seq_len]),
                )?,
                y: batch.y,
                mask: batch.mask,
            });
        }

        let mut shifted_x = vec![0.0f32; batch_size * n_vars * seq_len];

        for b in 0..batch_size {
            // Random shift amount for this sample
            let shift = rng.gen_range(-max_shift..=max_shift);

            for v in 0..n_vars {
                for t in 0..seq_len {
                    // Circular shift: wrap around
                    let src_t = ((t as i32 - shift).rem_euclid(seq_len as i32)) as usize;
                    let dst_idx = b * n_vars * seq_len + v * seq_len + t;
                    let src_idx = b * n_vars * seq_len + v * seq_len + src_t;
                    shifted_x[dst_idx] = x_values[src_idx];
                }
            }
        }

        let shifted_tensor: Tensor<B, 3> =
            Tensor::<B, 1>::from_floats(shifted_x.as_slice(), &device)
                .reshape([batch_size, n_vars, seq_len]);

        Ok(TSBatch {
            x: TSTensor::new(shifted_tensor)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "TSRandomShift"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

/// Configuration for horizontal flip (time reversal) transform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TSHorizontalFlipConfig {
    /// Probability of applying the transform.
    pub p: f32,
}

impl Default for TSHorizontalFlipConfig {
    fn default() -> Self {
        Self { p: 0.5 }
    }
}

/// Horizontal flip (time reversal) augmentation.
///
/// Reverses the time series along the time axis. This can help
/// models learn features that are invariant to time direction.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_transforms::TSHorizontalFlip;
///
/// let transform = TSHorizontalFlip::new()
///     .with_probability(0.3);
/// ```
pub struct TSHorizontalFlip {
    config: TSHorizontalFlipConfig,
    seed: Seed,
}

impl TSHorizontalFlip {
    /// Create a new horizontal flip transform.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: TSHorizontalFlipConfig::default(),
            seed: Seed::from_entropy(),
        }
    }

    /// Set the probability of applying the transform.
    #[must_use]
    pub fn with_probability(mut self, p: f32) -> Self {
        self.config.p = p;
        self
    }

    /// Set the random seed.
    #[must_use]
    pub fn with_seed(mut self, seed: Seed) -> Self {
        self.seed = seed;
        self
    }
}

impl Default for TSHorizontalFlip {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Transform<B> for TSHorizontalFlip {
    fn apply(&self, batch: TSBatch<B>, split: Split) -> Result<TSBatch<B>> {
        if split.is_eval() {
            return Ok(batch);
        }

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

        let mut flipped_x = vec![0.0f32; batch_size * n_vars * seq_len];

        for b in 0..batch_size {
            // Decide whether to flip this sample
            let do_flip = rng.gen::<f32>() < 0.5;

            for v in 0..n_vars {
                for t in 0..seq_len {
                    let dst_idx = b * n_vars * seq_len + v * seq_len + t;
                    let src_t = if do_flip { seq_len - 1 - t } else { t };
                    let src_idx = b * n_vars * seq_len + v * seq_len + src_t;
                    flipped_x[dst_idx] = x_values[src_idx];
                }
            }
        }

        let flipped_tensor: Tensor<B, 3> =
            Tensor::<B, 1>::from_floats(flipped_x.as_slice(), &device)
                .reshape([batch_size, n_vars, seq_len]);

        Ok(TSBatch {
            x: TSTensor::new(flipped_tensor)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "TSHorizontalFlip"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

/// Configuration for vertical flip (value negation) transform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TSVerticalFlipConfig {
    /// Probability of applying the transform.
    pub p: f32,
}

impl Default for TSVerticalFlipConfig {
    fn default() -> Self {
        Self { p: 0.5 }
    }
}

/// Vertical flip (value negation) augmentation.
///
/// Negates the values of the time series. This can help models
/// learn features that are invariant to sign changes.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_transforms::TSVerticalFlip;
///
/// let transform = TSVerticalFlip::new()
///     .with_probability(0.3);
/// ```
pub struct TSVerticalFlip {
    config: TSVerticalFlipConfig,
    seed: Seed,
}

impl TSVerticalFlip {
    /// Create a new vertical flip transform.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: TSVerticalFlipConfig::default(),
            seed: Seed::from_entropy(),
        }
    }

    /// Set the probability of applying the transform.
    #[must_use]
    pub fn with_probability(mut self, p: f32) -> Self {
        self.config.p = p;
        self
    }

    /// Set the random seed.
    #[must_use]
    pub fn with_seed(mut self, seed: Seed) -> Self {
        self.seed = seed;
        self
    }
}

impl Default for TSVerticalFlip {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Transform<B> for TSVerticalFlip {
    fn apply(&self, batch: TSBatch<B>, split: Split) -> Result<TSBatch<B>> {
        if split.is_eval() {
            return Ok(batch);
        }

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

        let mut flipped_x = vec![0.0f32; batch_size * n_vars * seq_len];

        for b in 0..batch_size {
            // Decide whether to flip this sample
            let do_flip = rng.gen::<f32>() < 0.5;

            for v in 0..n_vars {
                for t in 0..seq_len {
                    let idx = b * n_vars * seq_len + v * seq_len + t;
                    flipped_x[idx] = if do_flip { -x_values[idx] } else { x_values[idx] };
                }
            }
        }

        let flipped_tensor: Tensor<B, 3> =
            Tensor::<B, 1>::from_floats(flipped_x.as_slice(), &device)
                .reshape([batch_size, n_vars, seq_len]);

        Ok(TSBatch {
            x: TSTensor::new(flipped_tensor)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "TSVerticalFlip"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tsai_core::backend::NdArray;
    type TestBackend = NdArray;

    #[test]
    fn test_gaussian_noise_config() {
        let config = GaussianNoiseConfig::default();
        assert_eq!(config.std, 0.1);
        assert_eq!(config.p, 0.5);
    }

    #[test]
    fn test_mag_scale_config() {
        let config = MagScaleConfig::default();
        assert_eq!(config.factor, 1.2);
    }

    #[test]
    fn test_horizontal_flip_config() {
        let config = HorizontalFlipConfig::default();
        assert_eq!(config.p, 0.5);

        let flip = HorizontalFlip::new();
        assert_eq!(<HorizontalFlip as Transform<TestBackend>>::name(&flip), "HorizontalFlip");
    }

    #[test]
    fn test_random_shift_config() {
        let config = RandomShiftConfig::default();
        assert_eq!(config.max_shift, 0.2);
        assert_eq!(config.direction, ShiftDirection::Both);
        assert_eq!(config.fill_value, 0.0);

        let shift = RandomShift::new(0.1)
            .with_direction(ShiftDirection::Forward)
            .with_fill_value(-1.0);
        assert_eq!(shift.config.max_shift, 0.1);
        assert_eq!(shift.config.direction, ShiftDirection::Forward);
        assert_eq!(shift.config.fill_value, -1.0);
    }

    #[test]
    fn test_permutation_config() {
        let config = PermutationConfig::default();
        assert_eq!(config.n_segments, 5);
        assert_eq!(config.p, 0.5);

        let perm = Permutation::new(4);
        assert_eq!(perm.config.n_segments, 4);
    }

    #[test]
    fn test_rotation_config() {
        let config = RotationConfig::default();
        assert_eq!(config.max_rotation, 0.5);
        assert_eq!(config.p, 0.5);

        let rot = Rotation::new(0.3);
        assert_eq!(rot.config.max_rotation, 0.3);
    }

    #[test]
    fn test_identity_name() {
        let identity = Identity;
        assert_eq!(<Identity as Transform<TestBackend>>::name(&identity), "Identity");
    }

    #[test]
    fn test_frequency_mask_config() {
        let config = FrequencyMaskConfig::default();
        assert_eq!(config.freq_mask_param, 27);
        assert_eq!(config.num_masks, 1);
        assert_eq!(config.p, 0.5);
        assert_eq!(config.mask_value, 0.0);

        let mask = FrequencyMask::new(15)
            .with_num_masks(2)
            .with_probability(0.8)
            .with_mask_value(-1.0);
        assert_eq!(mask.config.freq_mask_param, 15);
        assert_eq!(mask.config.num_masks, 2);
        assert_eq!(mask.config.p, 0.8);
        assert_eq!(mask.config.mask_value, -1.0);
    }

    #[test]
    fn test_time_mask_config() {
        let config = TimeMaskConfig::default();
        assert_eq!(config.time_mask_param, 100);
        assert_eq!(config.num_masks, 1);
        assert_eq!(config.p, 0.5);
        assert_eq!(config.mask_value, 0.0);

        let mask = TimeMask::new(50)
            .with_num_masks(3)
            .with_probability(0.7)
            .with_mask_value(0.5);
        assert_eq!(mask.config.time_mask_param, 50);
        assert_eq!(mask.config.num_masks, 3);
        assert_eq!(mask.config.p, 0.7);
        assert_eq!(mask.config.mask_value, 0.5);
    }

    #[test]
    fn test_specaugment_config() {
        let config = SpecAugmentConfig::default();
        assert_eq!(config.freq_mask.freq_mask_param, 27);
        assert_eq!(config.time_mask.time_mask_param, 100);

        let augment = SpecAugment::new(20, 80)
            .with_freq_masks(2)
            .with_time_masks(3)
            .with_mask_value(-1.0)
            .with_freq_probability(0.9)
            .with_time_probability(0.8);

        assert_eq!(augment.config.freq_mask.freq_mask_param, 20);
        assert_eq!(augment.config.freq_mask.num_masks, 2);
        assert_eq!(augment.config.freq_mask.p, 0.9);
        assert_eq!(augment.config.freq_mask.mask_value, -1.0);
        assert_eq!(augment.config.time_mask.time_mask_param, 80);
        assert_eq!(augment.config.time_mask.num_masks, 3);
        assert_eq!(augment.config.time_mask.p, 0.8);
        assert_eq!(augment.config.time_mask.mask_value, -1.0);
    }

    #[test]
    fn test_specaugment_name() {
        let augment = SpecAugment::new(27, 100);
        assert_eq!(<SpecAugment as Transform<TestBackend>>::name(&augment), "SpecAugment");
    }

    #[test]
    fn test_frequency_mask_name() {
        let mask = FrequencyMask::new(15);
        assert_eq!(<FrequencyMask as Transform<TestBackend>>::name(&mask), "FrequencyMask");
    }

    #[test]
    fn test_time_mask_name() {
        let mask = TimeMask::new(50);
        assert_eq!(<TimeMask as Transform<TestBackend>>::name(&mask), "TimeMask");
    }
}
