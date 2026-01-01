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

// ============================================================================
// Magnitude-based Noise Transforms
// ============================================================================

/// Configuration for magnitude-scaled additive noise.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MagAddNoiseConfig {
    /// Noise magnitude as a fraction of the signal magnitude.
    pub magnitude: f32,
    /// Probability of applying the transform.
    pub p: f32,
}

impl Default for MagAddNoiseConfig {
    fn default() -> Self {
        Self {
            magnitude: 0.1,
            p: 0.5,
        }
    }
}

/// Adds noise scaled by the magnitude of the time series.
///
/// The noise at each time step is scaled by the absolute value of the signal,
/// making the noise proportional to the signal strength.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_transforms::augment::MagAddNoise;
///
/// let transform = MagAddNoise::new(0.1);  // 10% magnitude noise
/// ```
pub struct MagAddNoise {
    config: MagAddNoiseConfig,
    seed: Seed,
}

impl MagAddNoise {
    /// Create a new magnitude-scaled additive noise transform.
    #[must_use]
    pub fn new(magnitude: f32) -> Self {
        Self {
            config: MagAddNoiseConfig {
                magnitude,
                ..Default::default()
            },
            seed: Seed::from_entropy(),
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: MagAddNoiseConfig) -> Self {
        Self {
            config,
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

impl<B: Backend> Transform<B> for MagAddNoise {
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

        // Add noise scaled by magnitude
        let mut noisy_values = vec![0.0f32; batch_size * n_vars * seq_len];

        for i in 0..x_values.len() {
            let mag = x_values[i].abs();
            // Sample from normal distribution
            let u1: f32 = rng.gen();
            let u2: f32 = rng.gen();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            let noise = z * self.config.magnitude * mag;
            noisy_values[i] = x_values[i] + noise;
        }

        let noisy_tensor: Tensor<B, 3> =
            Tensor::<B, 1>::from_floats(noisy_values.as_slice(), &device)
                .reshape([batch_size, n_vars, seq_len]);

        Ok(TSBatch {
            x: TSTensor::new(noisy_tensor)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "MagAddNoise"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

/// Configuration for magnitude-scaled multiplicative noise.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MagMulNoiseConfig {
    /// Noise magnitude (standard deviation of the multiplier).
    pub magnitude: f32,
    /// Probability of applying the transform.
    pub p: f32,
}

impl Default for MagMulNoiseConfig {
    fn default() -> Self {
        Self {
            magnitude: 0.1,
            p: 0.5,
        }
    }
}

/// Multiplies the time series by magnitude-scaled noise.
///
/// Each value is multiplied by (1 + noise), where noise is sampled
/// from a normal distribution with the specified magnitude.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_transforms::augment::MagMulNoise;
///
/// let transform = MagMulNoise::new(0.1);  // 10% multiplicative noise
/// ```
pub struct MagMulNoise {
    config: MagMulNoiseConfig,
    seed: Seed,
}

impl MagMulNoise {
    /// Create a new magnitude-scaled multiplicative noise transform.
    #[must_use]
    pub fn new(magnitude: f32) -> Self {
        Self {
            config: MagMulNoiseConfig {
                magnitude,
                ..Default::default()
            },
            seed: Seed::from_entropy(),
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: MagMulNoiseConfig) -> Self {
        Self {
            config,
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

impl<B: Backend> Transform<B> for MagMulNoise {
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

        // Multiply by (1 + noise)
        let mut noisy_values = vec![0.0f32; batch_size * n_vars * seq_len];

        for i in 0..x_values.len() {
            // Sample from normal distribution
            let u1: f32 = rng.gen();
            let u2: f32 = rng.gen();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            let multiplier = 1.0 + z * self.config.magnitude;
            noisy_values[i] = x_values[i] * multiplier;
        }

        let noisy_tensor: Tensor<B, 3> =
            Tensor::<B, 1>::from_floats(noisy_values.as_slice(), &device)
                .reshape([batch_size, n_vars, seq_len]);

        Ok(TSBatch {
            x: TSTensor::new(noisy_tensor)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "MagMulNoise"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

// ============================================================================
// Variable and Mask Dropout Transforms
// ============================================================================

/// Configuration for mask-based dropout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskOutConfig {
    /// Maximum fraction of time steps to mask.
    pub max_mask_ratio: f32,
    /// Value to fill masked regions.
    pub fill_value: f32,
    /// Probability of applying the transform.
    pub p: f32,
}

impl Default for MaskOutConfig {
    fn default() -> Self {
        Self {
            max_mask_ratio: 0.2,
            fill_value: 0.0,
            p: 0.5,
        }
    }
}

/// Randomly masks out portions of the time series with a specified value.
///
/// Unlike CutOut which masks contiguous regions, MaskOut can mask
/// random scattered time steps.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_transforms::augment::MaskOut;
///
/// let transform = MaskOut::new(0.15);  // Mask up to 15% of time steps
/// ```
pub struct MaskOut {
    config: MaskOutConfig,
    seed: Seed,
}

impl MaskOut {
    /// Create a new mask-out transform.
    #[must_use]
    pub fn new(max_mask_ratio: f32) -> Self {
        Self {
            config: MaskOutConfig {
                max_mask_ratio,
                ..Default::default()
            },
            seed: Seed::from_entropy(),
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: MaskOutConfig) -> Self {
        Self {
            config,
            seed: Seed::from_entropy(),
        }
    }

    /// Set the fill value for masked regions.
    #[must_use]
    pub fn with_fill_value(mut self, fill_value: f32) -> Self {
        self.config.fill_value = fill_value;
        self
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

impl<B: Backend> Transform<B> for MaskOut {
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
        let mut x_values: Vec<f32> = x_data
            .as_slice()
            .map_err(|e| CoreError::ShapeMismatch(format!("Failed to get X data: {e:?}")))?
            .to_vec();

        // Calculate number of time steps to mask
        let mask_ratio: f32 = rng.gen_range(0.0..self.config.max_mask_ratio);
        let n_mask = ((seq_len as f32 * mask_ratio).round() as usize).max(1);

        for b in 0..batch_size {
            // Select random time steps to mask
            let mut indices: Vec<usize> = (0..seq_len).collect();
            // Fisher-Yates shuffle to select first n_mask indices
            for i in 0..n_mask.min(seq_len) {
                let j = rng.gen_range(i..seq_len);
                indices.swap(i, j);
            }

            // Mask selected time steps across all variables
            for &t in indices.iter().take(n_mask) {
                for v in 0..n_vars {
                    let idx = b * n_vars * seq_len + v * seq_len + t;
                    x_values[idx] = self.config.fill_value;
                }
            }
        }

        let masked_tensor: Tensor<B, 3> =
            Tensor::<B, 1>::from_floats(x_values.as_slice(), &device)
                .reshape([batch_size, n_vars, seq_len]);

        Ok(TSBatch {
            x: TSTensor::new(masked_tensor)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "MaskOut"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

/// Configuration for variable dropout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarOutConfig {
    /// Maximum fraction of variables to drop.
    pub max_drop_ratio: f32,
    /// Value to fill dropped variables.
    pub fill_value: f32,
    /// Probability of applying the transform.
    pub p: f32,
}

impl Default for VarOutConfig {
    fn default() -> Self {
        Self {
            max_drop_ratio: 0.2,
            fill_value: 0.0,
            p: 0.5,
        }
    }
}

/// Randomly drops out entire variables (channels) from the time series.
///
/// This is useful for multivariate time series to encourage the model
/// to learn robust features across different variable combinations.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_transforms::augment::VarOut;
///
/// let transform = VarOut::new(0.2);  // Drop up to 20% of variables
/// ```
pub struct VarOut {
    config: VarOutConfig,
    seed: Seed,
}

impl VarOut {
    /// Create a new variable dropout transform.
    #[must_use]
    pub fn new(max_drop_ratio: f32) -> Self {
        Self {
            config: VarOutConfig {
                max_drop_ratio,
                ..Default::default()
            },
            seed: Seed::from_entropy(),
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: VarOutConfig) -> Self {
        Self {
            config,
            seed: Seed::from_entropy(),
        }
    }

    /// Set the fill value for dropped variables.
    #[must_use]
    pub fn with_fill_value(mut self, fill_value: f32) -> Self {
        self.config.fill_value = fill_value;
        self
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

impl<B: Backend> Transform<B> for VarOut {
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

        if n_vars <= 1 {
            // Cannot drop variables if only one exists
            return Ok(batch);
        }

        // Get data as raw values
        let x_data = batch.x.into_inner().into_data();
        let mut x_values: Vec<f32> = x_data
            .as_slice()
            .map_err(|e| CoreError::ShapeMismatch(format!("Failed to get X data: {e:?}")))?
            .to_vec();

        // Calculate number of variables to drop
        let drop_ratio: f32 = rng.gen_range(0.0..self.config.max_drop_ratio);
        let n_drop = ((n_vars as f32 * drop_ratio).round() as usize).min(n_vars - 1);

        if n_drop == 0 {
            let tensor: Tensor<B, 3> =
                Tensor::<B, 1>::from_floats(x_values.as_slice(), &device)
                    .reshape([batch_size, n_vars, seq_len]);
            return Ok(TSBatch {
                x: TSTensor::new(tensor)?,
                y: batch.y,
                mask: batch.mask,
            });
        }

        for b in 0..batch_size {
            // Select random variables to drop
            let mut var_indices: Vec<usize> = (0..n_vars).collect();
            for i in 0..n_drop {
                let j = rng.gen_range(i..n_vars);
                var_indices.swap(i, j);
            }

            // Drop selected variables
            for &v in var_indices.iter().take(n_drop) {
                for t in 0..seq_len {
                    let idx = b * n_vars * seq_len + v * seq_len + t;
                    x_values[idx] = self.config.fill_value;
                }
            }
        }

        let dropped_tensor: Tensor<B, 3> =
            Tensor::<B, 1>::from_floats(x_values.as_slice(), &device)
                .reshape([batch_size, n_vars, seq_len]);

        Ok(TSBatch {
            x: TSTensor::new(dropped_tensor)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "VarOut"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

// ============================================================================
// RandAugment - Automatic Augmentation
// ============================================================================

/// Available transforms for RandAugment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RandAugmentOp {
    /// Gaussian noise
    GaussianNoise,
    /// Magnitude-scaled additive noise
    MagAddNoise,
    /// Magnitude-scaled multiplicative noise
    MagMulNoise,
    /// Time warping
    TimeWarp,
    /// Magnitude warping
    MagWarp,
    /// Window warping
    WindowWarp,
    /// Magnitude scaling
    MagScale,
    /// Cutout/masking
    CutOut,
    /// Random mask-out
    MaskOut,
    /// Horizontal flip (time reversal)
    HorizontalFlip,
    /// Random shift
    RandomShift,
    /// Permutation
    Permutation,
    /// Rotation
    Rotation,
}

impl RandAugmentOp {
    /// Get all available operations.
    pub fn all() -> Vec<Self> {
        vec![
            Self::GaussianNoise,
            Self::MagAddNoise,
            Self::MagMulNoise,
            Self::TimeWarp,
            Self::MagWarp,
            Self::WindowWarp,
            Self::MagScale,
            Self::CutOut,
            Self::MaskOut,
            Self::HorizontalFlip,
            Self::RandomShift,
            Self::Permutation,
            Self::Rotation,
        ]
    }
}

/// Configuration for RandAugment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandAugmentConfig {
    /// Number of transforms to apply.
    pub n: usize,
    /// Magnitude of transforms (0.0 to 1.0).
    pub magnitude: f32,
    /// Available operations to sample from.
    pub ops: Vec<RandAugmentOp>,
    /// Probability of applying any augmentation.
    pub p: f32,
}

impl Default for RandAugmentConfig {
    fn default() -> Self {
        Self {
            n: 2,
            magnitude: 0.5,
            ops: RandAugmentOp::all(),
            p: 1.0,
        }
    }
}

/// RandAugment: Practical automated data augmentation.
///
/// Randomly samples N transforms from a pool and applies them with
/// a specified magnitude. Based on the paper "RandAugment: Practical
/// automated data augmentation with a reduced search space".
///
/// # Example
///
/// ```rust,ignore
/// use tsai_transforms::augment::RandAugment;
///
/// let transform = RandAugment::new(2, 0.5);  // Apply 2 transforms at 50% magnitude
/// ```
pub struct RandAugment {
    config: RandAugmentConfig,
    seed: Seed,
}

impl RandAugment {
    /// Create a new RandAugment transform.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of transforms to apply
    /// * `magnitude` - Magnitude of transforms (0.0 to 1.0)
    #[must_use]
    pub fn new(n: usize, magnitude: f32) -> Self {
        Self {
            config: RandAugmentConfig {
                n,
                magnitude: magnitude.clamp(0.0, 1.0),
                ..Default::default()
            },
            seed: Seed::from_entropy(),
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: RandAugmentConfig) -> Self {
        Self {
            config,
            seed: Seed::from_entropy(),
        }
    }

    /// Set the available operations.
    #[must_use]
    pub fn with_ops(mut self, ops: Vec<RandAugmentOp>) -> Self {
        self.config.ops = ops;
        self
    }

    /// Set the probability of applying augmentation.
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

    /// Apply a single operation to the batch.
    fn apply_op<B: Backend>(
        &self,
        batch: TSBatch<B>,
        op: RandAugmentOp,
        rng: &mut impl Rng,
    ) -> Result<TSBatch<B>> {
        let m = self.config.magnitude;

        match op {
            RandAugmentOp::GaussianNoise => {
                let transform = GaussianNoise::new(m * 0.3)
                    .with_seed(Seed::new(rng.gen()));
                transform.apply(batch, Split::Train)
            }
            RandAugmentOp::MagAddNoise => {
                let transform = MagAddNoise::new(m * 0.2)
                    .with_probability(1.0)
                    .with_seed(Seed::new(rng.gen()));
                transform.apply(batch, Split::Train)
            }
            RandAugmentOp::MagMulNoise => {
                let transform = MagMulNoise::new(m * 0.2)
                    .with_probability(1.0)
                    .with_seed(Seed::new(rng.gen()));
                transform.apply(batch, Split::Train)
            }
            RandAugmentOp::TimeWarp => {
                let transform = TimeWarp::new(m * 0.4)
                    .with_seed(Seed::new(rng.gen()));
                transform.apply(batch, Split::Train)
            }
            RandAugmentOp::MagWarp => {
                let transform = MagWarp::new(m * 0.3)
                    .with_seed(Seed::new(rng.gen()));
                transform.apply(batch, Split::Train)
            }
            RandAugmentOp::WindowWarp => {
                let transform = WindowWarp::new(m * 0.3)
                    .with_seed(Seed::new(rng.gen()));
                transform.apply(batch, Split::Train)
            }
            RandAugmentOp::MagScale => {
                let transform = MagScale::new(1.0 + m * 0.5)
                    .with_seed(Seed::new(rng.gen()));
                transform.apply(batch, Split::Train)
            }
            RandAugmentOp::CutOut => {
                let transform = CutOut::new(m * 0.3)
                    .with_seed(Seed::new(rng.gen()));
                transform.apply(batch, Split::Train)
            }
            RandAugmentOp::MaskOut => {
                let transform = MaskOut::new(m * 0.2)
                    .with_probability(1.0)
                    .with_seed(Seed::new(rng.gen()));
                transform.apply(batch, Split::Train)
            }
            RandAugmentOp::HorizontalFlip => {
                let config = HorizontalFlipConfig { p: 1.0 };
                let transform = HorizontalFlip::from_config(config)
                    .with_seed(Seed::new(rng.gen()));
                transform.apply(batch, Split::Train)
            }
            RandAugmentOp::RandomShift => {
                let config = RandomShiftConfig {
                    max_shift: m * 0.3,
                    p: 1.0,
                    ..Default::default()
                };
                let transform = RandomShift::from_config(config)
                    .with_seed(Seed::new(rng.gen()));
                transform.apply(batch, Split::Train)
            }
            RandAugmentOp::Permutation => {
                let n_segments = ((m * 10.0).round() as usize).max(2);
                let config = PermutationConfig {
                    n_segments,
                    p: 1.0,
                };
                let transform = Permutation::from_config(config)
                    .with_seed(Seed::new(rng.gen()));
                transform.apply(batch, Split::Train)
            }
            RandAugmentOp::Rotation => {
                // Use new() - default p=0.5 is fine since we already decided to apply
                let transform = Rotation::new(m * 0.5)
                    .with_seed(Seed::new(rng.gen()));
                transform.apply(batch, Split::Train)
            }
        }
    }
}

impl<B: Backend> Transform<B> for RandAugment {
    fn apply(&self, mut batch: TSBatch<B>, split: Split) -> Result<TSBatch<B>> {
        if split.is_eval() {
            return Ok(batch);
        }

        let mut rng = self.seed.to_rng();

        if rng.gen::<f32>() > self.config.p {
            return Ok(batch);
        }

        if self.config.ops.is_empty() {
            return Ok(batch);
        }

        // Randomly select N operations
        for _ in 0..self.config.n {
            let op_idx = rng.gen_range(0..self.config.ops.len());
            let op = self.config.ops[op_idx];
            batch = self.apply_op(batch, op, &mut rng)?;
        }

        Ok(batch)
    }

    fn name(&self) -> &str {
        "RandAugment"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

// ============================================================================
// Random Resized Crop Transform
// ============================================================================

/// Configuration for random resized crop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomResizedCropConfig {
    /// Minimum crop ratio (fraction of original length).
    pub min_ratio: f32,
    /// Maximum crop ratio (fraction of original length).
    pub max_ratio: f32,
    /// Probability of applying the transform.
    pub p: f32,
}

impl Default for RandomResizedCropConfig {
    fn default() -> Self {
        Self {
            min_ratio: 0.5,
            max_ratio: 1.0,
            p: 0.5,
        }
    }
}

/// Randomly crops and resizes the time series to the original length.
///
/// Crops a random portion of the time series and then resizes it back
/// to the original length using linear interpolation.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_transforms::augment::RandomResizedCrop;
///
/// let transform = RandomResizedCrop::new(0.5, 1.0);  // Crop 50-100% of length
/// ```
pub struct RandomResizedCrop {
    config: RandomResizedCropConfig,
    seed: Seed,
}

impl RandomResizedCrop {
    /// Create a new random resized crop transform.
    #[must_use]
    pub fn new(min_ratio: f32, max_ratio: f32) -> Self {
        Self {
            config: RandomResizedCropConfig {
                min_ratio,
                max_ratio,
                ..Default::default()
            },
            seed: Seed::from_entropy(),
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: RandomResizedCropConfig) -> Self {
        Self {
            config,
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

impl<B: Backend> Transform<B> for RandomResizedCrop {
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

        if seq_len <= 1 {
            return Ok(batch);
        }

        // Get data as raw values
        let x_data = batch.x.into_inner().into_data();
        let x_values: Vec<f32> = x_data
            .as_slice()
            .map_err(|e| CoreError::ShapeMismatch(format!("Failed to get X data: {e:?}")))?
            .to_vec();

        let mut cropped_values = vec![0.0f32; batch_size * n_vars * seq_len];

        for b in 0..batch_size {
            // Random crop ratio for this sample
            let crop_ratio = rng.gen_range(self.config.min_ratio..=self.config.max_ratio);
            let crop_len = ((seq_len as f32 * crop_ratio).round() as usize).max(2);

            // Random start position
            let max_start = seq_len.saturating_sub(crop_len);
            let start = if max_start > 0 {
                rng.gen_range(0..=max_start)
            } else {
                0
            };

            // Resize cropped region back to original length using linear interpolation
            for v in 0..n_vars {
                for t in 0..seq_len {
                    // Map target position to source position
                    let src_pos = start as f32 + (t as f32 / seq_len as f32) * (crop_len - 1) as f32;
                    let idx_low = src_pos.floor() as usize;
                    let idx_high = (idx_low + 1).min(start + crop_len - 1);
                    let frac = src_pos - src_pos.floor();

                    let src_low = b * n_vars * seq_len + v * seq_len + idx_low;
                    let src_high = b * n_vars * seq_len + v * seq_len + idx_high;
                    let dst = b * n_vars * seq_len + v * seq_len + t;

                    cropped_values[dst] =
                        x_values[src_low] * (1.0 - frac) + x_values[src_high] * frac;
                }
            }
        }

        let cropped_tensor: Tensor<B, 3> =
            Tensor::<B, 1>::from_floats(cropped_values.as_slice(), &device)
                .reshape([batch_size, n_vars, seq_len]);

        Ok(TSBatch {
            x: TSTensor::new(cropped_tensor)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "RandomResizedCrop"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

// ============================================================================
// Time Noise Transform
// ============================================================================

/// Configuration for time noise transform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeNoiseConfig {
    /// Maximum jitter as fraction of sequence length.
    pub max_jitter: f32,
    /// Probability of applying the transform.
    pub p: f32,
}

impl Default for TimeNoiseConfig {
    fn default() -> Self {
        Self {
            max_jitter: 0.1,
            p: 0.5,
        }
    }
}

/// Adds noise to the time dimension by jittering time indices.
///
/// This transform perturbs the time axis by adding random offsets to each
/// time step's position, then interpolating to get the new values.
/// Useful for making models robust to timing variations.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_transforms::augment::TimeNoise;
///
/// let transform = TimeNoise::new(0.1);  // Up to 10% jitter
/// ```
pub struct TimeNoise {
    config: TimeNoiseConfig,
    seed: Seed,
}

impl TimeNoise {
    /// Create a new time noise transform.
    ///
    /// # Arguments
    ///
    /// * `max_jitter` - Maximum jitter as fraction of sequence length
    #[must_use]
    pub fn new(max_jitter: f32) -> Self {
        Self {
            config: TimeNoiseConfig {
                max_jitter,
                ..Default::default()
            },
            seed: Seed::from_entropy(),
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: TimeNoiseConfig) -> Self {
        Self {
            config,
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

impl<B: Backend> Transform<B> for TimeNoise {
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

        if seq_len <= 2 {
            return Ok(batch);
        }

        // Get data as raw values
        let x_data = batch.x.into_inner().into_data();
        let x_values: Vec<f32> = x_data
            .as_slice()
            .map_err(|e| CoreError::ShapeMismatch(format!("Failed to get X data: {e:?}")))?
            .to_vec();

        let max_jitter_samples = (self.config.max_jitter * seq_len as f32) as i32;
        let mut jittered_values = vec![0.0f32; batch_size * n_vars * seq_len];

        for b in 0..batch_size {
            // Generate random jitter for each time step
            let jitter: Vec<f32> = (0..seq_len)
                .map(|_| {
                    if max_jitter_samples > 0 {
                        rng.gen_range(-max_jitter_samples..=max_jitter_samples) as f32
                    } else {
                        0.0
                    }
                })
                .collect();

            for v in 0..n_vars {
                for t in 0..seq_len {
                    // Calculate source position with jitter
                    let src_pos = (t as f32 + jitter[t]).clamp(0.0, (seq_len - 1) as f32);
                    let idx_low = src_pos.floor() as usize;
                    let idx_high = (idx_low + 1).min(seq_len - 1);
                    let frac = src_pos - src_pos.floor();

                    let src_low = b * n_vars * seq_len + v * seq_len + idx_low;
                    let src_high = b * n_vars * seq_len + v * seq_len + idx_high;
                    let dst = b * n_vars * seq_len + v * seq_len + t;

                    jittered_values[dst] =
                        x_values[src_low] * (1.0 - frac) + x_values[src_high] * frac;
                }
            }
        }

        let jittered_tensor: Tensor<B, 3> =
            Tensor::<B, 1>::from_floats(jittered_values.as_slice(), &device)
                .reshape([batch_size, n_vars, seq_len]);

        Ok(TSBatch {
            x: TSTensor::new(jittered_tensor)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "TimeNoise"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

// ============================================================================
// Blur/Smoothing Transforms
// ============================================================================

/// Configuration for Gaussian blur transform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlurConfig {
    /// Kernel size (odd number, e.g., 3, 5, 7).
    pub kernel_size: usize,
    /// Standard deviation of Gaussian kernel.
    pub sigma: f32,
    /// Probability of applying the transform.
    pub p: f32,
}

impl Default for BlurConfig {
    fn default() -> Self {
        Self {
            kernel_size: 5,
            sigma: 1.0,
            p: 0.5,
        }
    }
}

/// Applies Gaussian blur to the time series.
///
/// Convolves the time series with a 1D Gaussian kernel to smooth out
/// high-frequency noise. Useful for data augmentation and denoising.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_transforms::augment::Blur;
///
/// let transform = Blur::new(5, 1.0);  // 5-point kernel with sigma=1.0
/// ```
pub struct Blur {
    config: BlurConfig,
    seed: Seed,
}

impl Blur {
    /// Create a new blur transform.
    ///
    /// # Arguments
    ///
    /// * `kernel_size` - Size of the Gaussian kernel (should be odd)
    /// * `sigma` - Standard deviation of the Gaussian
    #[must_use]
    pub fn new(kernel_size: usize, sigma: f32) -> Self {
        // Ensure kernel size is odd
        let kernel_size = if kernel_size % 2 == 0 {
            kernel_size + 1
        } else {
            kernel_size
        };

        Self {
            config: BlurConfig {
                kernel_size,
                sigma,
                ..Default::default()
            },
            seed: Seed::from_entropy(),
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: BlurConfig) -> Self {
        Self {
            config,
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

    /// Generate Gaussian kernel.
    fn gaussian_kernel(&self) -> Vec<f32> {
        let half_size = self.config.kernel_size / 2;
        let sigma = self.config.sigma;

        let mut kernel: Vec<f32> = (0..self.config.kernel_size)
            .map(|i| {
                let x = i as f32 - half_size as f32;
                (-x * x / (2.0 * sigma * sigma)).exp()
            })
            .collect();

        // Normalize
        let sum: f32 = kernel.iter().sum();
        for k in &mut kernel {
            *k /= sum;
        }

        kernel
    }
}

impl<B: Backend> Transform<B> for Blur {
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

        if seq_len < self.config.kernel_size {
            return Ok(batch);
        }

        // Get data as raw values
        let x_data = batch.x.into_inner().into_data();
        let x_values: Vec<f32> = x_data
            .as_slice()
            .map_err(|e| CoreError::ShapeMismatch(format!("Failed to get X data: {e:?}")))?
            .to_vec();

        let kernel = self.gaussian_kernel();
        let half_size = self.config.kernel_size / 2;
        let mut blurred_values = vec![0.0f32; batch_size * n_vars * seq_len];

        for b in 0..batch_size {
            for v in 0..n_vars {
                for t in 0..seq_len {
                    let mut sum = 0.0f32;
                    let mut weight_sum = 0.0f32;

                    for (k, &kernel_val) in kernel.iter().enumerate() {
                        let src_t = t as i32 - half_size as i32 + k as i32;
                        if src_t >= 0 && src_t < seq_len as i32 {
                            let src_idx = b * n_vars * seq_len + v * seq_len + src_t as usize;
                            sum += x_values[src_idx] * kernel_val;
                            weight_sum += kernel_val;
                        }
                    }

                    let dst = b * n_vars * seq_len + v * seq_len + t;
                    blurred_values[dst] = if weight_sum > 0.0 {
                        sum / weight_sum
                    } else {
                        x_values[dst]
                    };
                }
            }
        }

        let blurred_tensor: Tensor<B, 3> =
            Tensor::<B, 1>::from_floats(blurred_values.as_slice(), &device)
                .reshape([batch_size, n_vars, seq_len]);

        Ok(TSBatch {
            x: TSTensor::new(blurred_tensor)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "Blur"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

/// Smoothing method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SmoothingMethod {
    /// Simple moving average.
    MovingAverage,
    /// Exponential moving average.
    Exponential,
    /// Savitzky-Golay filter (polynomial smoothing).
    SavitzkyGolay,
}

impl Default for SmoothingMethod {
    fn default() -> Self {
        Self::MovingAverage
    }
}

/// Configuration for smoothing transform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmoothConfig {
    /// Smoothing method.
    pub method: SmoothingMethod,
    /// Window size for moving average.
    pub window_size: usize,
    /// Alpha for exponential smoothing (0-1).
    pub alpha: f32,
    /// Probability of applying the transform.
    pub p: f32,
}

impl Default for SmoothConfig {
    fn default() -> Self {
        Self {
            method: SmoothingMethod::MovingAverage,
            window_size: 5,
            alpha: 0.3,
            p: 0.5,
        }
    }
}

/// Applies various smoothing filters to the time series.
///
/// Supports multiple smoothing methods:
/// - Moving average: Simple window-based averaging
/// - Exponential: Exponential moving average with decay
/// - Savitzky-Golay: Polynomial smoothing for better edge preservation
///
/// # Example
///
/// ```rust,ignore
/// use tsai_transforms::augment::{Smooth, SmoothingMethod};
///
/// let transform = Smooth::new(SmoothingMethod::MovingAverage)
///     .with_window_size(5);
/// ```
pub struct Smooth {
    config: SmoothConfig,
    seed: Seed,
}

impl Smooth {
    /// Create a new smoothing transform.
    #[must_use]
    pub fn new(method: SmoothingMethod) -> Self {
        Self {
            config: SmoothConfig {
                method,
                ..Default::default()
            },
            seed: Seed::from_entropy(),
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: SmoothConfig) -> Self {
        Self {
            config,
            seed: Seed::from_entropy(),
        }
    }

    /// Set the window size for moving average.
    #[must_use]
    pub fn with_window_size(mut self, window_size: usize) -> Self {
        self.config.window_size = window_size;
        self
    }

    /// Set the alpha for exponential smoothing.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.config.alpha = alpha.clamp(0.0, 1.0);
        self
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

    /// Apply moving average smoothing.
    fn moving_average(&self, values: &[f32], n_vars: usize, seq_len: usize) -> Vec<f32> {
        let batch_size = values.len() / (n_vars * seq_len);
        let half_window = self.config.window_size / 2;
        let mut smoothed = values.to_vec();

        for b in 0..batch_size {
            for v in 0..n_vars {
                for t in 0..seq_len {
                    let mut sum = 0.0f32;
                    let mut count = 0;

                    for offset in 0..self.config.window_size {
                        let src_t = t as i32 - half_window as i32 + offset as i32;
                        if src_t >= 0 && src_t < seq_len as i32 {
                            let idx = b * n_vars * seq_len + v * seq_len + src_t as usize;
                            sum += values[idx];
                            count += 1;
                        }
                    }

                    let dst = b * n_vars * seq_len + v * seq_len + t;
                    smoothed[dst] = if count > 0 { sum / count as f32 } else { values[dst] };
                }
            }
        }

        smoothed
    }

    /// Apply exponential smoothing.
    fn exponential_smooth(&self, values: &[f32], n_vars: usize, seq_len: usize) -> Vec<f32> {
        let batch_size = values.len() / (n_vars * seq_len);
        let alpha = self.config.alpha;
        let mut smoothed = values.to_vec();

        for b in 0..batch_size {
            for v in 0..n_vars {
                // Forward pass
                let base = b * n_vars * seq_len + v * seq_len;
                let mut ema = values[base];

                for t in 0..seq_len {
                    let idx = base + t;
                    ema = alpha * values[idx] + (1.0 - alpha) * ema;
                    smoothed[idx] = ema;
                }
            }
        }

        smoothed
    }

    /// Apply Savitzky-Golay filter (simplified quadratic fit).
    fn savitzky_golay(&self, values: &[f32], n_vars: usize, seq_len: usize) -> Vec<f32> {
        let batch_size = values.len() / (n_vars * seq_len);
        let half_window = self.config.window_size / 2;

        // Precompute Savitzky-Golay coefficients for quadratic fit
        // Using simplified weights for window_size=5: [-3, 12, 17, 12, -3] / 35
        let coeffs: Vec<f32> = match self.config.window_size {
            3 => vec![0.25, 0.5, 0.25],
            5 => vec![-3.0 / 35.0, 12.0 / 35.0, 17.0 / 35.0, 12.0 / 35.0, -3.0 / 35.0],
            7 => vec![
                -2.0 / 21.0,
                3.0 / 21.0,
                6.0 / 21.0,
                7.0 / 21.0,
                6.0 / 21.0,
                3.0 / 21.0,
                -2.0 / 21.0,
            ],
            _ => {
                // Default to moving average weights
                vec![1.0 / self.config.window_size as f32; self.config.window_size]
            }
        };

        let mut smoothed = values.to_vec();

        for b in 0..batch_size {
            for v in 0..n_vars {
                for t in 0..seq_len {
                    let mut sum = 0.0f32;
                    let mut weight_sum = 0.0f32;

                    for (k, &coeff) in coeffs.iter().enumerate() {
                        let src_t = t as i32 - half_window as i32 + k as i32;
                        if src_t >= 0 && src_t < seq_len as i32 {
                            let idx = b * n_vars * seq_len + v * seq_len + src_t as usize;
                            sum += values[idx] * coeff;
                            weight_sum += coeff.abs();
                        }
                    }

                    let dst = b * n_vars * seq_len + v * seq_len + t;
                    smoothed[dst] = if weight_sum > 0.0 {
                        sum / weight_sum * coeffs.iter().map(|c| c.abs()).sum::<f32>()
                    } else {
                        values[dst]
                    };
                }
            }
        }

        smoothed
    }
}

impl<B: Backend> Transform<B> for Smooth {
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

        if seq_len < self.config.window_size {
            return Ok(batch);
        }

        // Get data as raw values
        let x_data = batch.x.into_inner().into_data();
        let x_values: Vec<f32> = x_data
            .as_slice()
            .map_err(|e| CoreError::ShapeMismatch(format!("Failed to get X data: {e:?}")))?
            .to_vec();

        let smoothed_values = match self.config.method {
            SmoothingMethod::MovingAverage => self.moving_average(&x_values, n_vars, seq_len),
            SmoothingMethod::Exponential => self.exponential_smooth(&x_values, n_vars, seq_len),
            SmoothingMethod::SavitzkyGolay => self.savitzky_golay(&x_values, n_vars, seq_len),
        };

        let smoothed_tensor: Tensor<B, 3> =
            Tensor::<B, 1>::from_floats(smoothed_values.as_slice(), &device)
                .reshape([batch_size, n_vars, seq_len]);

        Ok(TSBatch {
            x: TSTensor::new(smoothed_tensor)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "Smooth"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

// ============================================================================
// Frequency-Based Transforms
// ============================================================================

/// Configuration for random frequency noise transform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomFreqNoiseConfig {
    /// Noise magnitude (relative to signal std).
    pub magnitude: f32,
    /// Frequency band: "low", "mid", "high", or "all".
    pub band: String,
    /// Probability of applying the transform.
    pub p: f32,
}

impl Default for RandomFreqNoiseConfig {
    fn default() -> Self {
        Self {
            magnitude: 0.1,
            band: "high".to_string(),
            p: 0.5,
        }
    }
}

/// Adds noise to specific frequency bands of the time series.
///
/// This transform adds noise that targets specific frequency components:
/// - "low": Adds slowly varying noise (affects trend)
/// - "mid": Adds medium-frequency noise
/// - "high": Adds high-frequency noise (affects fine details)
/// - "all": Adds noise across all frequencies
///
/// Uses convolution-based approximation without requiring FFT.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_transforms::augment::RandomFreqNoise;
///
/// let transform = RandomFreqNoise::new(0.1, "high");
/// ```
pub struct RandomFreqNoise {
    config: RandomFreqNoiseConfig,
    seed: Seed,
}

impl RandomFreqNoise {
    /// Create a new random frequency noise transform.
    ///
    /// # Arguments
    ///
    /// * `magnitude` - Noise magnitude relative to signal std
    /// * `band` - Frequency band: "low", "mid", "high", or "all"
    #[must_use]
    pub fn new(magnitude: f32, band: &str) -> Self {
        Self {
            config: RandomFreqNoiseConfig {
                magnitude,
                band: band.to_string(),
                ..Default::default()
            },
            seed: Seed::from_entropy(),
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: RandomFreqNoiseConfig) -> Self {
        Self {
            config,
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

    /// Generate band-limited noise.
    fn generate_band_noise(&self, rng: &mut impl Rng, seq_len: usize, std: f32) -> Vec<f32> {
        let noise_std = self.config.magnitude * std;

        match self.config.band.as_str() {
            "low" => {
                // Low-frequency: smooth random walk
                let mut noise = vec![0.0f32; seq_len];
                let step_size = noise_std * 0.5;
                let mut cumsum = 0.0f32;
                for n in &mut noise {
                    cumsum += rng.gen_range(-step_size..step_size);
                    *n = cumsum;
                }
                // Remove mean
                let mean: f32 = noise.iter().sum::<f32>() / seq_len as f32;
                for n in &mut noise {
                    *n -= mean;
                }
                noise
            }
            "mid" => {
                // Mid-frequency: sinusoidal with random phase
                let mut noise = vec![0.0f32; seq_len];
                let num_components = 3;
                for _ in 0..num_components {
                    let freq = rng.gen_range(2..10) as f32;
                    let phase = rng.gen_range(0.0..std::f32::consts::TAU);
                    let amp = noise_std / (num_components as f32).sqrt();
                    for (i, n) in noise.iter_mut().enumerate() {
                        let t = i as f32 / seq_len as f32;
                        *n += amp * (std::f32::consts::TAU * freq * t + phase).sin();
                    }
                }
                noise
            }
            "high" => {
                // High-frequency: white noise with high-pass filtering
                let mut noise: Vec<f32> = (0..seq_len)
                    .map(|_| rng.gen_range(-noise_std..noise_std))
                    .collect();
                // Simple high-pass: subtract smoothed version
                let window = 5.min(seq_len);
                let mut smoothed = noise.clone();
                for i in window / 2..seq_len - window / 2 {
                    let sum: f32 = (0..window).map(|j| noise[i - window / 2 + j]).sum();
                    smoothed[i] = sum / window as f32;
                }
                for (n, s) in noise.iter_mut().zip(&smoothed) {
                    *n -= s;
                }
                noise
            }
            _ => {
                // "all": standard white noise
                (0..seq_len)
                    .map(|_| rng.gen_range(-noise_std..noise_std))
                    .collect()
            }
        }
    }
}

impl<B: Backend> Transform<B> for RandomFreqNoise {
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

        let x_data = batch.x.into_inner().into_data();
        let mut x_values: Vec<f32> = x_data
            .as_slice()
            .map_err(|e| CoreError::ShapeMismatch(format!("Failed to get X data: {e:?}")))?
            .to_vec();

        for b in 0..batch_size {
            for v in 0..n_vars {
                let start = b * n_vars * seq_len + v * seq_len;
                let end = start + seq_len;
                let ts = &x_values[start..end];

                // Compute std of this channel
                let mean: f32 = ts.iter().sum::<f32>() / seq_len as f32;
                let var: f32 = ts.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / seq_len as f32;
                let std = var.sqrt().max(1e-8);

                // Generate band-limited noise
                let noise = self.generate_band_noise(&mut rng, seq_len, std);

                // Add noise
                for (i, n) in noise.iter().enumerate() {
                    x_values[start + i] += n;
                }
            }
        }

        let noisy_tensor: Tensor<B, 3> =
            Tensor::<B, 1>::from_floats(x_values.as_slice(), &device)
                .reshape([batch_size, n_vars, seq_len]);

        Ok(TSBatch {
            x: TSTensor::new(noisy_tensor)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "RandomFreqNoise"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

/// Configuration for frequency denoising transform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreqDenoiseConfig {
    /// Cutoff frequency as fraction of Nyquist (0.0 to 1.0).
    pub cutoff: f32,
    /// Filter order (higher = sharper cutoff).
    pub order: usize,
    /// Probability of applying the transform.
    pub p: f32,
}

impl Default for FreqDenoiseConfig {
    fn default() -> Self {
        Self {
            cutoff: 0.3,
            order: 3,
            p: 0.5,
        }
    }
}

/// Denoises time series by filtering high frequencies.
///
/// Applies a low-pass filter to remove high-frequency noise.
/// Uses cascaded moving averages to approximate a low-pass filter.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_transforms::augment::FreqDenoise;
///
/// let transform = FreqDenoise::new(0.3);  // Keep frequencies below 30% of Nyquist
/// ```
pub struct FreqDenoise {
    config: FreqDenoiseConfig,
    seed: Seed,
}

impl FreqDenoise {
    /// Create a new frequency denoising transform.
    ///
    /// # Arguments
    ///
    /// * `cutoff` - Cutoff frequency as fraction of Nyquist (0.0 to 1.0)
    #[must_use]
    pub fn new(cutoff: f32) -> Self {
        Self {
            config: FreqDenoiseConfig {
                cutoff: cutoff.clamp(0.01, 0.99),
                ..Default::default()
            },
            seed: Seed::from_entropy(),
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: FreqDenoiseConfig) -> Self {
        Self {
            config,
            seed: Seed::from_entropy(),
        }
    }

    /// Set the filter order.
    #[must_use]
    pub fn with_order(mut self, order: usize) -> Self {
        self.config.order = order.max(1);
        self
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

    /// Apply low-pass filter using cascaded moving averages.
    fn apply_lowpass(&self, signal: &[f32]) -> Vec<f32> {
        let seq_len = signal.len();

        // Window size based on cutoff (inverse relationship)
        let window_size = ((1.0 / self.config.cutoff).round() as usize)
            .max(3)
            .min(seq_len / 2);

        let mut result = signal.to_vec();

        // Apply cascaded moving averages (approximates Gaussian)
        for _ in 0..self.config.order {
            let mut smoothed = result.clone();
            let half_window = window_size / 2;

            for i in 0..seq_len {
                let start = i.saturating_sub(half_window);
                let end = (i + half_window + 1).min(seq_len);
                let sum: f32 = result[start..end].iter().sum();
                smoothed[i] = sum / (end - start) as f32;
            }

            result = smoothed;
        }

        result
    }
}

impl<B: Backend> Transform<B> for FreqDenoise {
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

        let x_data = batch.x.into_inner().into_data();
        let x_values: Vec<f32> = x_data
            .as_slice()
            .map_err(|e| CoreError::ShapeMismatch(format!("Failed to get X data: {e:?}")))?
            .to_vec();

        let mut denoised_values = vec![0.0f32; batch_size * n_vars * seq_len];

        for b in 0..batch_size {
            for v in 0..n_vars {
                let start = b * n_vars * seq_len + v * seq_len;
                let end = start + seq_len;
                let ts = &x_values[start..end];

                let filtered = self.apply_lowpass(ts);

                for (i, &val) in filtered.iter().enumerate() {
                    denoised_values[start + i] = val;
                }
            }
        }

        let denoised_tensor: Tensor<B, 3> =
            Tensor::<B, 1>::from_floats(denoised_values.as_slice(), &device)
                .reshape([batch_size, n_vars, seq_len]);

        Ok(TSBatch {
            x: TSTensor::new(denoised_tensor)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "FreqDenoise"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

// ============================================================================
// Random Convolution Transform
// ============================================================================

/// Configuration for random convolution augmentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomConvConfig {
    /// Kernel size range (min, max).
    pub kernel_size_range: (usize, usize),
    /// Number of random convolutions to apply.
    pub n_convs: usize,
    /// Probability of applying the transform.
    pub p: f32,
    /// Whether to preserve the input mean.
    pub preserve_mean: bool,
}

impl Default for RandomConvConfig {
    fn default() -> Self {
        Self {
            kernel_size_range: (3, 7),
            n_convs: 1,
            p: 0.5,
            preserve_mean: true,
        }
    }
}

/// Random convolution augmentation (TSRandomConv).
///
/// Applies random convolutional kernels to the time series, creating
/// smoothed/filtered versions of the data as augmentation.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_transforms::augment::RandomConv;
///
/// // Create with default kernel size range (3-7)
/// let transform = RandomConv::new();
///
/// // Or customize
/// let transform = RandomConv::new()
///     .with_kernel_range(5, 15)
///     .with_n_convs(2)
///     .with_probability(0.8);
/// ```
pub struct RandomConv {
    config: RandomConvConfig,
    seed: Seed,
}

impl Default for RandomConv {
    fn default() -> Self {
        Self::new()
    }
}

impl RandomConv {
    /// Create a new random convolution transform with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: RandomConvConfig::default(),
            seed: Seed::from_entropy(),
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: RandomConvConfig) -> Self {
        Self {
            config,
            seed: Seed::from_entropy(),
        }
    }

    /// Set the kernel size range.
    #[must_use]
    pub fn with_kernel_range(mut self, min: usize, max: usize) -> Self {
        self.config.kernel_size_range = (min.max(1), max.max(min + 1));
        self
    }

    /// Set the number of convolutions to apply.
    #[must_use]
    pub fn with_n_convs(mut self, n: usize) -> Self {
        self.config.n_convs = n.max(1);
        self
    }

    /// Set the probability of applying the transform.
    #[must_use]
    pub fn with_probability(mut self, p: f32) -> Self {
        self.config.p = p.clamp(0.0, 1.0);
        self
    }

    /// Set whether to preserve the mean.
    #[must_use]
    pub fn with_preserve_mean(mut self, preserve: bool) -> Self {
        self.config.preserve_mean = preserve;
        self
    }

    /// Set the random seed.
    #[must_use]
    pub fn with_seed(mut self, seed: Seed) -> Self {
        self.seed = seed;
        self
    }

    /// Generate a random convolution kernel.
    fn random_kernel(&self, rng: &mut impl Rng, size: usize) -> Vec<f32> {
        // Generate random weights
        let mut kernel: Vec<f32> = (0..size).map(|_| rng.gen::<f32>()).collect();

        // Normalize to sum to 1 (low-pass filter effect)
        let sum: f32 = kernel.iter().sum();
        if sum > 0.0 {
            for k in &mut kernel {
                *k /= sum;
            }
        }

        kernel
    }

    /// Apply convolution to a 1D signal.
    fn convolve(&self, signal: &[f32], kernel: &[f32]) -> Vec<f32> {
        let n = signal.len();
        let k = kernel.len();
        let half_k = k / 2;

        let mut output = vec![0.0f32; n];

        for i in 0..n {
            let mut sum = 0.0f32;
            for (j, &kval) in kernel.iter().enumerate() {
                let idx = i as isize + j as isize - half_k as isize;
                let sample = if idx < 0 {
                    signal[0]
                } else if idx >= n as isize {
                    signal[n - 1]
                } else {
                    signal[idx as usize]
                };
                sum += sample * kval;
            }
            output[i] = sum;
        }

        output
    }
}

impl<B: Backend> Transform<B> for RandomConv {
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

        let x_data = batch.x.into_inner().into_data();
        let x_values: Vec<f32> = x_data
            .as_slice()
            .map_err(|e| CoreError::ShapeMismatch(format!("Failed to get X data: {e:?}")))?
            .to_vec();

        let mut conv_values = vec![0.0f32; batch_size * n_vars * seq_len];

        for b in 0..batch_size {
            for v in 0..n_vars {
                let start = b * n_vars * seq_len + v * seq_len;
                let end = start + seq_len;
                let ts = &x_values[start..end];

                // Calculate original mean if needed
                let orig_mean: f32 = if self.config.preserve_mean {
                    ts.iter().sum::<f32>() / ts.len() as f32
                } else {
                    0.0
                };

                // Apply multiple random convolutions
                let mut result = ts.to_vec();
                for _ in 0..self.config.n_convs {
                    let kernel_size = rng.gen_range(
                        self.config.kernel_size_range.0..=self.config.kernel_size_range.1,
                    );
                    // Ensure odd kernel size for symmetric convolution
                    let kernel_size = if kernel_size % 2 == 0 {
                        kernel_size + 1
                    } else {
                        kernel_size
                    };
                    let kernel = self.random_kernel(&mut rng, kernel_size);
                    result = self.convolve(&result, &kernel);
                }

                // Restore mean if needed
                if self.config.preserve_mean {
                    let new_mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
                    let diff = orig_mean - new_mean;
                    for val in &mut result {
                        *val += diff;
                    }
                }

                for (i, &val) in result.iter().enumerate() {
                    conv_values[start + i] = val;
                }
            }
        }

        let conv_tensor: Tensor<B, 3> =
            Tensor::<B, 1>::from_floats(conv_values.as_slice(), &device)
                .reshape([batch_size, n_vars, seq_len]);

        Ok(TSBatch {
            x: TSTensor::new(conv_tensor)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "RandomConv"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

// =============================================================================
// TSRandomCropPad: Random crop with padding
// =============================================================================

/// Configuration for random crop and pad augmentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomCropPadConfig {
    /// Target length after crop/pad (None = use original length).
    pub target_len: Option<usize>,
    /// Minimum crop ratio (0.0-1.0).
    pub min_crop_ratio: f32,
    /// Maximum crop ratio (0.0-1.0).
    pub max_crop_ratio: f32,
    /// Padding value.
    pub pad_value: f32,
    /// Probability of applying the transform.
    pub p: f32,
}

impl Default for RandomCropPadConfig {
    fn default() -> Self {
        Self {
            target_len: None,
            min_crop_ratio: 0.5,
            max_crop_ratio: 1.0,
            pad_value: 0.0,
            p: 0.5,
        }
    }
}

/// Random crop and pad transform.
///
/// Randomly crops a portion of the time series and pads back
/// to the original (or target) length.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_transforms::augment::RandomCropPad;
///
/// // Random crop between 50% and 100% of sequence, pad back
/// let transform = RandomCropPad::new(0.5, 1.0);
/// ```
pub struct RandomCropPad {
    config: RandomCropPadConfig,
    seed: Seed,
}

impl RandomCropPad {
    /// Create a new random crop and pad transform.
    #[must_use]
    pub fn new(min_crop_ratio: f32, max_crop_ratio: f32) -> Self {
        Self {
            config: RandomCropPadConfig {
                min_crop_ratio,
                max_crop_ratio,
                ..Default::default()
            },
            seed: Seed::from_entropy(),
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: RandomCropPadConfig) -> Self {
        Self {
            config,
            seed: Seed::from_entropy(),
        }
    }

    /// Set target length.
    #[must_use]
    pub fn with_target_len(mut self, len: usize) -> Self {
        self.config.target_len = Some(len);
        self
    }

    /// Set padding value.
    #[must_use]
    pub fn with_pad_value(mut self, value: f32) -> Self {
        self.config.pad_value = value;
        self
    }

    /// Set probability.
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

impl<B: Backend> Transform<B> for RandomCropPad {
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

        let target_len = self.config.target_len.unwrap_or(seq_len);

        // Random crop ratio
        let crop_ratio = rng.gen_range(self.config.min_crop_ratio..=self.config.max_crop_ratio);
        let crop_len = ((seq_len as f32 * crop_ratio) as usize).max(1);

        // Random crop start position
        let max_start = seq_len.saturating_sub(crop_len);
        let start = if max_start > 0 {
            rng.gen_range(0..=max_start)
        } else {
            0
        };

        // Extract cropped portion
        let x = batch.x.into_inner();
        let cropped = x.slice([0..batch_size, 0..n_vars, start..start + crop_len]);

        // Pad or resize to target length
        let result = if crop_len < target_len {
            // Need to pad
            let pad_len = target_len - crop_len;
            let pad_before = pad_len / 2;
            let _pad_after = pad_len - pad_before;

            // Create padded tensor
            let pad_value = self.config.pad_value;
            let _full_tensor: Tensor<B, 3> = Tensor::full([batch_size, n_vars, target_len], pad_value, &device);

            // Copy cropped data into the middle
            // Using slice assignment would be ideal but we'll reconstruct
            let mut values: Vec<f32> = vec![pad_value; batch_size * n_vars * target_len];
            let cropped_data = cropped.to_data();
            let cropped_values: Vec<f32> = cropped_data.to_vec().map_err(|_| CoreError::TransformError("Failed to convert tensor data".to_string()))?;

            for b in 0..batch_size {
                for v in 0..n_vars {
                    for t in 0..crop_len {
                        let src_idx = b * n_vars * crop_len + v * crop_len + t;
                        let dst_idx = b * n_vars * target_len + v * target_len + (pad_before + t);
                        if src_idx < cropped_values.len() && dst_idx < values.len() {
                            values[dst_idx] = cropped_values[src_idx];
                        }
                    }
                }
            }

            Tensor::<B, 1>::from_floats(values.as_slice(), &device)
                .reshape([batch_size, n_vars, target_len])
        } else if crop_len > target_len {
            // Need to subsample/resize (simple linear interpolation)
            let mut values: Vec<f32> = vec![0.0; batch_size * n_vars * target_len];
            let cropped_data = cropped.to_data();
            let cropped_values: Vec<f32> = cropped_data.to_vec().map_err(|_| CoreError::TransformError("Failed to convert tensor data".to_string()))?;

            for b in 0..batch_size {
                for v in 0..n_vars {
                    for t in 0..target_len {
                        let src_t = t as f32 * (crop_len - 1) as f32 / (target_len - 1).max(1) as f32;
                        let t0 = src_t.floor() as usize;
                        let t1 = (t0 + 1).min(crop_len - 1);
                        let frac = src_t - t0 as f32;

                        let src_idx0 = b * n_vars * crop_len + v * crop_len + t0;
                        let src_idx1 = b * n_vars * crop_len + v * crop_len + t1;
                        let dst_idx = b * n_vars * target_len + v * target_len + t;

                        if src_idx0 < cropped_values.len() && src_idx1 < cropped_values.len() && dst_idx < values.len() {
                            values[dst_idx] = cropped_values[src_idx0] * (1.0 - frac) + cropped_values[src_idx1] * frac;
                        }
                    }
                }
            }

            Tensor::<B, 1>::from_floats(values.as_slice(), &device)
                .reshape([batch_size, n_vars, target_len])
        } else {
            cropped
        };

        Ok(TSBatch {
            x: TSTensor::new(result)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "RandomCropPad"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

// =============================================================================
// TSRandomZoomOut: Random zoom out (shrink and pad)
// =============================================================================

/// Configuration for random zoom out augmentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomZoomOutConfig {
    /// Minimum zoom ratio (0.0-1.0, where 0.5 means shrink to half).
    pub min_zoom: f32,
    /// Maximum zoom ratio (0.0-1.0).
    pub max_zoom: f32,
    /// Padding value for the zoomed-out regions.
    pub pad_value: f32,
    /// Probability of applying the transform.
    pub p: f32,
}

impl Default for RandomZoomOutConfig {
    fn default() -> Self {
        Self {
            min_zoom: 0.5,
            max_zoom: 1.0,
            pad_value: 0.0,
            p: 0.5,
        }
    }
}

/// Random zoom out transform.
///
/// Shrinks the time series and pads to maintain original length.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_transforms::augment::RandomZoomOut;
///
/// // Random zoom between 50% and 100%
/// let transform = RandomZoomOut::new(0.5, 1.0);
/// ```
pub struct RandomZoomOut {
    config: RandomZoomOutConfig,
    seed: Seed,
}

impl RandomZoomOut {
    /// Create a new random zoom out transform.
    #[must_use]
    pub fn new(min_zoom: f32, max_zoom: f32) -> Self {
        Self {
            config: RandomZoomOutConfig {
                min_zoom,
                max_zoom,
                ..Default::default()
            },
            seed: Seed::from_entropy(),
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: RandomZoomOutConfig) -> Self {
        Self {
            config,
            seed: Seed::from_entropy(),
        }
    }

    /// Set padding value.
    #[must_use]
    pub fn with_pad_value(mut self, value: f32) -> Self {
        self.config.pad_value = value;
        self
    }

    /// Set probability.
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

impl<B: Backend> Transform<B> for RandomZoomOut {
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

        // Random zoom ratio
        let zoom = rng.gen_range(self.config.min_zoom..=self.config.max_zoom);
        if zoom >= 0.99 {
            return Ok(batch);
        }

        let zoomed_len = ((seq_len as f32 * zoom) as usize).max(1);

        // Subsample original to zoomed_len using linear interpolation
        let x = batch.x.into_inner();
        let x_data = x.to_data();
        let x_values: Vec<f32> = x_data.to_vec().map_err(|_| CoreError::TransformError("Failed to convert tensor data".to_string()))?;

        let mut result_values: Vec<f32> = vec![self.config.pad_value; batch_size * n_vars * seq_len];

        // Center the zoomed content
        let pad_before = (seq_len - zoomed_len) / 2;

        for b in 0..batch_size {
            for v in 0..n_vars {
                for t in 0..zoomed_len {
                    // Map from zoomed position to original position
                    let src_t = t as f32 * (seq_len - 1) as f32 / (zoomed_len - 1).max(1) as f32;
                    let t0 = src_t.floor() as usize;
                    let t1 = (t0 + 1).min(seq_len - 1);
                    let frac = src_t - t0 as f32;

                    let src_idx0 = b * n_vars * seq_len + v * seq_len + t0;
                    let src_idx1 = b * n_vars * seq_len + v * seq_len + t1;
                    let dst_idx = b * n_vars * seq_len + v * seq_len + (pad_before + t);

                    if src_idx0 < x_values.len() && src_idx1 < x_values.len() && dst_idx < result_values.len() {
                        result_values[dst_idx] = x_values[src_idx0] * (1.0 - frac) + x_values[src_idx1] * frac;
                    }
                }
            }
        }

        let result: Tensor<B, 3> = Tensor::<B, 1>::from_floats(result_values.as_slice(), &device)
            .reshape([batch_size, n_vars, seq_len]);

        Ok(TSBatch {
            x: TSTensor::new(result)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "RandomZoomOut"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

// =============================================================================
// TSMagScalePerVar: Per-variable magnitude scaling
// =============================================================================

/// Configuration for per-variable magnitude scaling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MagScalePerVarConfig {
    /// Minimum scaling factor.
    pub min_scale: f32,
    /// Maximum scaling factor.
    pub max_scale: f32,
    /// Probability of applying the transform.
    pub p: f32,
}

impl Default for MagScalePerVarConfig {
    fn default() -> Self {
        Self {
            min_scale: 0.8,
            max_scale: 1.2,
            p: 0.5,
        }
    }
}

/// Per-variable magnitude scaling transform.
///
/// Applies independent random scaling to each variable/channel.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_transforms::augment::MagScalePerVar;
///
/// // Scale each variable independently by 0.8x to 1.2x
/// let transform = MagScalePerVar::new(0.8, 1.2);
/// ```
pub struct MagScalePerVar {
    config: MagScalePerVarConfig,
    seed: Seed,
}

impl MagScalePerVar {
    /// Create a new per-variable magnitude scale transform.
    #[must_use]
    pub fn new(min_scale: f32, max_scale: f32) -> Self {
        Self {
            config: MagScalePerVarConfig {
                min_scale,
                max_scale,
                ..Default::default()
            },
            seed: Seed::from_entropy(),
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: MagScalePerVarConfig) -> Self {
        Self {
            config,
            seed: Seed::from_entropy(),
        }
    }

    /// Set probability.
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

impl<B: Backend> Transform<B> for MagScalePerVar {
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
        let _seq_len = shape.len();
        let device = batch.device();

        // Generate random scale for each variable in each batch sample
        let mut scales: Vec<f32> = Vec::with_capacity(batch_size * n_vars);
        for _ in 0..batch_size {
            for _ in 0..n_vars {
                scales.push(rng.gen_range(self.config.min_scale..=self.config.max_scale));
            }
        }

        // Create scale tensor: (batch, vars, 1) for broadcasting
        let scale_tensor: Tensor<B, 3> = Tensor::<B, 1>::from_floats(scales.as_slice(), &device)
            .reshape([batch_size, n_vars, 1]);

        // Apply scaling
        let x = batch.x.into_inner();
        let scaled = x * scale_tensor;

        Ok(TSBatch {
            x: TSTensor::new(scaled)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "MagScalePerVar"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

// =============================================================================
// TSRandomTrend: Add random trend to time series
// =============================================================================

/// Configuration for random trend augmentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomTrendConfig {
    /// Maximum trend magnitude (slope).
    pub max_trend: f32,
    /// Whether to also add quadratic trend.
    pub add_quadratic: bool,
    /// Maximum quadratic coefficient.
    pub max_quadratic: f32,
    /// Probability of applying the transform.
    pub p: f32,
}

impl Default for RandomTrendConfig {
    fn default() -> Self {
        Self {
            max_trend: 0.1,
            add_quadratic: false,
            max_quadratic: 0.01,
            p: 0.5,
        }
    }
}

/// Random trend augmentation.
///
/// Adds a random linear (and optionally quadratic) trend to the time series.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_transforms::augment::RandomTrend;
///
/// // Add random linear trend with max slope 0.1
/// let transform = RandomTrend::new(0.1);
///
/// // Add both linear and quadratic trends
/// let transform = RandomTrend::new(0.1).with_quadratic(0.01);
/// ```
pub struct RandomTrend {
    config: RandomTrendConfig,
    seed: Seed,
}

impl RandomTrend {
    /// Create a new random trend transform.
    #[must_use]
    pub fn new(max_trend: f32) -> Self {
        Self {
            config: RandomTrendConfig {
                max_trend,
                ..Default::default()
            },
            seed: Seed::from_entropy(),
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: RandomTrendConfig) -> Self {
        Self {
            config,
            seed: Seed::from_entropy(),
        }
    }

    /// Enable quadratic trend.
    #[must_use]
    pub fn with_quadratic(mut self, max_quadratic: f32) -> Self {
        self.config.add_quadratic = true;
        self.config.max_quadratic = max_quadratic;
        self
    }

    /// Set probability.
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

impl<B: Backend> Transform<B> for RandomTrend {
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

        // Generate random trend for each sample and variable
        let mut trend_values: Vec<f32> = Vec::with_capacity(batch_size * n_vars * seq_len);

        for _b in 0..batch_size {
            for _v in 0..n_vars {
                // Random linear slope
                let slope = rng.gen_range(-self.config.max_trend..=self.config.max_trend);

                // Optional quadratic coefficient
                let quad = if self.config.add_quadratic {
                    rng.gen_range(-self.config.max_quadratic..=self.config.max_quadratic)
                } else {
                    0.0
                };

                for t in 0..seq_len {
                    let t_norm = t as f32 / seq_len.max(1) as f32;
                    let trend = slope * t_norm + quad * t_norm * t_norm;
                    trend_values.push(trend);
                }
            }
        }

        let trend_tensor: Tensor<B, 3> = Tensor::<B, 1>::from_floats(trend_values.as_slice(), &device)
            .reshape([batch_size, n_vars, seq_len]);

        let x = batch.x.into_inner();
        let result = x + trend_tensor;

        Ok(TSBatch {
            x: TSTensor::new(result)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "RandomTrend"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

// =============================================================================
// TSTimeStepOut: Random time step dropout
// =============================================================================

/// Configuration for time step dropout augmentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeStepOutConfig {
    /// Probability of dropping each time step.
    pub drop_prob: f32,
    /// Value to replace dropped steps with (None = use previous value).
    pub fill_value: Option<f32>,
    /// Probability of applying the transform.
    pub p: f32,
}

impl Default for TimeStepOutConfig {
    fn default() -> Self {
        Self {
            drop_prob: 0.1,
            fill_value: None,
            p: 0.5,
        }
    }
}

/// Time step dropout augmentation.
///
/// Randomly drops time steps and fills with a constant or previous value.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_transforms::augment::TimeStepOut;
///
/// // Drop 10% of time steps, fill with 0
/// let transform = TimeStepOut::new(0.1).with_fill_value(0.0);
///
/// // Drop 15% of time steps, fill with previous value
/// let transform = TimeStepOut::new(0.15);
/// ```
pub struct TimeStepOut {
    config: TimeStepOutConfig,
    seed: Seed,
}

impl TimeStepOut {
    /// Create a new time step dropout transform.
    #[must_use]
    pub fn new(drop_prob: f32) -> Self {
        Self {
            config: TimeStepOutConfig {
                drop_prob,
                ..Default::default()
            },
            seed: Seed::from_entropy(),
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: TimeStepOutConfig) -> Self {
        Self {
            config,
            seed: Seed::from_entropy(),
        }
    }

    /// Set fill value for dropped steps.
    #[must_use]
    pub fn with_fill_value(mut self, value: f32) -> Self {
        self.config.fill_value = Some(value);
        self
    }

    /// Set probability.
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

impl<B: Backend> Transform<B> for TimeStepOut {
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

        let x = batch.x.into_inner();
        let x_data = x.to_data();
        let x_values: Vec<f32> = x_data.to_vec().map_err(|_| CoreError::TransformError("Failed to convert tensor data".to_string()))?;

        let mut result_values = x_values.clone();

        for b in 0..batch_size {
            // Generate dropout mask for this sample (same mask for all vars)
            let mut drop_mask: Vec<bool> = Vec::with_capacity(seq_len);
            for _ in 0..seq_len {
                drop_mask.push(rng.gen::<f32>() < self.config.drop_prob);
            }

            for v in 0..n_vars {
                for t in 0..seq_len {
                    if drop_mask[t] {
                        let idx = b * n_vars * seq_len + v * seq_len + t;
                        let fill = if let Some(val) = self.config.fill_value {
                            val
                        } else {
                            // Use previous value (or first value if t=0)
                            let prev_t = if t > 0 { t - 1 } else { 0 };
                            let prev_idx = b * n_vars * seq_len + v * seq_len + prev_t;
                            x_values[prev_idx]
                        };
                        result_values[idx] = fill;
                    }
                }
            }
        }

        let result: Tensor<B, 3> = Tensor::<B, 1>::from_floats(result_values.as_slice(), &device)
            .reshape([batch_size, n_vars, seq_len]);

        Ok(TSBatch {
            x: TSTensor::new(result)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "TimeStepOut"
    }

    fn should_apply(&self, split: Split) -> bool {
        split.is_train()
    }
}

// =============================================================================
// TSShuffleSteps: Shuffle time steps within segments
// =============================================================================

/// Configuration for shuffle steps augmentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShuffleStepsConfig {
    /// Number of segments to divide the sequence into.
    pub n_segments: usize,
    /// Probability of applying the transform.
    pub p: f32,
}

impl Default for ShuffleStepsConfig {
    fn default() -> Self {
        Self {
            n_segments: 5,
            p: 0.5,
        }
    }
}

/// Shuffle steps within segments augmentation.
///
/// Divides the time series into segments and shuffles steps within each segment.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_transforms::augment::ShuffleSteps;
///
/// // Shuffle steps within 5 segments
/// let transform = ShuffleSteps::new(5);
/// ```
pub struct ShuffleSteps {
    config: ShuffleStepsConfig,
    seed: Seed,
}

impl ShuffleSteps {
    /// Create a new shuffle steps transform.
    #[must_use]
    pub fn new(n_segments: usize) -> Self {
        Self {
            config: ShuffleStepsConfig {
                n_segments,
                ..Default::default()
            },
            seed: Seed::from_entropy(),
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: ShuffleStepsConfig) -> Self {
        Self {
            config,
            seed: Seed::from_entropy(),
        }
    }

    /// Set probability.
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

impl<B: Backend> Transform<B> for ShuffleSteps {
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

        let x = batch.x.into_inner();
        let x_data = x.to_data();
        let x_values: Vec<f32> = x_data.to_vec().map_err(|_| CoreError::TransformError("Failed to convert tensor data".to_string()))?;

        let mut result_values = vec![0.0f32; batch_size * n_vars * seq_len];

        let n_segments = self.config.n_segments.max(1);
        let segment_len = seq_len / n_segments;

        for b in 0..batch_size {
            // Create shuffled indices for each segment
            let mut shuffled_indices: Vec<usize> = Vec::with_capacity(seq_len);

            for seg in 0..n_segments {
                let start = seg * segment_len;
                let end = if seg == n_segments - 1 { seq_len } else { start + segment_len };

                let mut segment_indices: Vec<usize> = (start..end).collect();
                segment_indices.shuffle(&mut rng);
                shuffled_indices.extend(segment_indices);
            }

            // Apply shuffled indices to all variables
            for v in 0..n_vars {
                for (new_t, &old_t) in shuffled_indices.iter().enumerate() {
                    let src_idx = b * n_vars * seq_len + v * seq_len + old_t;
                    let dst_idx = b * n_vars * seq_len + v * seq_len + new_t;
                    result_values[dst_idx] = x_values[src_idx];
                }
            }
        }

        let result: Tensor<B, 3> = Tensor::<B, 1>::from_floats(result_values.as_slice(), &device)
            .reshape([batch_size, n_vars, seq_len]);

        Ok(TSBatch {
            x: TSTensor::new(result)?,
            y: batch.y,
            mask: batch.mask,
        })
    }

    fn name(&self) -> &str {
        "ShuffleSteps"
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
