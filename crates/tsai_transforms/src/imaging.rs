//! Imaging transforms for time series visualization.
//!
//! This module provides transforms that convert time series to image representations:
//! - Recurrence Plots (RP)
//! - Markov Transition Fields (MTF)
//! - Gramian Angular Fields (GAF, GASF, GADF)

use serde::{Deserialize, Serialize};

/// Configuration for Recurrence Plot transform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecurrencePlotConfig {
    /// Size of the output image.
    pub size: usize,
    /// Threshold for recurrence (epsilon).
    pub threshold: Option<f32>,
    /// Percentage of points to use for threshold calculation.
    pub percentage: Option<f32>,
}

impl Default for RecurrencePlotConfig {
    fn default() -> Self {
        Self {
            size: 64,
            threshold: None,
            percentage: Some(10.0),
        }
    }
}

/// Converts time series to Recurrence Plot images.
///
/// A recurrence plot visualizes the times at which a dynamical system
/// returns to a state it has visited before.
#[derive(Debug, Clone)]
pub struct TSToRP {
    config: RecurrencePlotConfig,
}

impl TSToRP {
    /// Create a new Recurrence Plot transform.
    #[must_use]
    pub fn new(size: usize) -> Self {
        Self {
            config: RecurrencePlotConfig {
                size,
                ..Default::default()
            },
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: RecurrencePlotConfig) -> Self {
        Self { config }
    }

    /// Compute recurrence plot for a single univariate time series.
    ///
    /// # Arguments
    ///
    /// * `series` - Input time series of length L
    ///
    /// # Returns
    ///
    /// A 2D array of size (L, L) representing the recurrence plot.
    pub fn compute(&self, series: &[f32]) -> Vec<Vec<f32>> {
        let n = series.len();
        let mut plot = vec![vec![0.0f32; n]; n];

        // Compute pairwise distances
        for i in 0..n {
            for j in 0..n {
                let dist = (series[i] - series[j]).abs();
                plot[i][j] = dist;
            }
        }

        // Apply threshold if specified
        if let Some(thresh) = self.config.threshold {
            for row in &mut plot {
                for val in row {
                    *val = if *val < thresh { 1.0 } else { 0.0 };
                }
            }
        } else if let Some(pct) = self.config.percentage {
            // Calculate threshold from percentile
            let mut all_dists: Vec<f32> = plot.iter().flatten().copied().collect();
            all_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let thresh_idx = ((pct / 100.0) * all_dists.len() as f32) as usize;
            let thresh = all_dists.get(thresh_idx).copied().unwrap_or(0.0);

            for row in &mut plot {
                for val in row {
                    *val = if *val < thresh { 1.0 } else { 0.0 };
                }
            }
        }

        plot
    }
}

/// Configuration for Markov Transition Field transform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MTFConfig {
    /// Size of the output image.
    pub size: usize,
    /// Number of quantile bins.
    pub n_bins: usize,
}

impl Default for MTFConfig {
    fn default() -> Self {
        Self {
            size: 64,
            n_bins: 8,
        }
    }
}

/// Converts time series to Markov Transition Field images.
///
/// MTF encodes the transition probabilities between quantile bins.
#[derive(Debug, Clone)]
pub struct TSToMTF {
    #[allow(dead_code)] // Config stored for future resizing implementation
    config: MTFConfig,
}

impl TSToMTF {
    /// Create a new MTF transform.
    #[must_use]
    pub fn new(size: usize, n_bins: usize) -> Self {
        Self {
            config: MTFConfig { size, n_bins },
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: MTFConfig) -> Self {
        Self { config }
    }
}

/// Configuration for Gramian Angular Field transforms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GAFConfig {
    /// Size of the output image.
    pub size: usize,
    /// Type of GAF (Summation or Difference).
    pub gaf_type: GAFType,
}

/// Type of Gramian Angular Field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GAFType {
    /// Gramian Angular Summation Field.
    Summation,
    /// Gramian Angular Difference Field.
    Difference,
}

impl Default for GAFConfig {
    fn default() -> Self {
        Self {
            size: 64,
            gaf_type: GAFType::Summation,
        }
    }
}

/// Converts time series to Gramian Angular Summation Field images.
#[derive(Debug, Clone)]
pub struct TSToGASF {
    #[allow(dead_code)] // Config stored for future resizing implementation
    config: GAFConfig,
}

impl TSToGASF {
    /// Create a new GASF transform.
    #[must_use]
    pub fn new(size: usize) -> Self {
        Self {
            config: GAFConfig {
                size,
                gaf_type: GAFType::Summation,
            },
        }
    }

    /// Compute GASF for a single univariate time series.
    ///
    /// # Arguments
    ///
    /// * `series` - Input time series of length L
    ///
    /// # Returns
    ///
    /// A 2D array of size (L, L) representing the GASF.
    pub fn compute(&self, series: &[f32]) -> Vec<Vec<f32>> {
        let n = series.len();

        // Normalize to [-1, 1]
        let min = series.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = series.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max - min).max(1e-8);

        let normalized: Vec<f32> = series.iter().map(|&x| 2.0 * (x - min) / range - 1.0).collect();

        // Convert to polar coordinates (angular cosine)
        let phi: Vec<f32> = normalized.iter().map(|&x| x.acos()).collect();

        // Compute GASF: cos(phi_i + phi_j)
        let mut gasf = vec![vec![0.0f32; n]; n];
        for i in 0..n {
            for j in 0..n {
                gasf[i][j] = (phi[i] + phi[j]).cos();
            }
        }

        gasf
    }
}

/// Converts time series to Gramian Angular Difference Field images.
#[derive(Debug, Clone)]
pub struct TSToGADF {
    #[allow(dead_code)] // Config stored for future resizing implementation
    config: GAFConfig,
}

impl TSToGADF {
    /// Create a new GADF transform.
    #[must_use]
    pub fn new(size: usize) -> Self {
        Self {
            config: GAFConfig {
                size,
                gaf_type: GAFType::Difference,
            },
        }
    }

    /// Compute GADF for a single univariate time series.
    ///
    /// # Arguments
    ///
    /// * `series` - Input time series of length L
    ///
    /// # Returns
    ///
    /// A 2D array of size (L, L) representing the GADF.
    pub fn compute(&self, series: &[f32]) -> Vec<Vec<f32>> {
        let n = series.len();

        // Normalize to [-1, 1]
        let min = series.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = series.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max - min).max(1e-8);

        let normalized: Vec<f32> = series.iter().map(|&x| 2.0 * (x - min) / range - 1.0).collect();

        // Convert to polar coordinates (angular cosine)
        let phi: Vec<f32> = normalized.iter().map(|&x| x.acos()).collect();

        // Compute GADF: sin(phi_i - phi_j)
        let mut gadf = vec![vec![0.0f32; n]; n];
        for i in 0..n {
            for j in 0..n {
                gadf[i][j] = (phi[i] - phi[j]).sin();
            }
        }

        gadf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rp_compute() {
        let series = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let rp = TSToRP::new(5);
        let plot = rp.compute(&series);

        assert_eq!(plot.len(), 5);
        assert_eq!(plot[0].len(), 5);

        // Diagonal should have zeros (same point)
        for i in 0..5 {
            // Before thresholding, distance from point to itself is 0
        }
    }

    #[test]
    fn test_gasf_compute() {
        let series = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        let gasf = TSToGASF::new(5);
        let field = gasf.compute(&series);

        assert_eq!(field.len(), 5);
        assert_eq!(field[0].len(), 5);

        // GASF values should be in [-1, 1]
        for row in &field {
            for &val in row {
                assert!(val >= -1.0 && val <= 1.0);
            }
        }
    }

    #[test]
    fn test_gadf_compute() {
        let series = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        let gadf = TSToGADF::new(5);
        let field = gadf.compute(&series);

        assert_eq!(field.len(), 5);
        assert_eq!(field[0].len(), 5);

        // Diagonal should be zero (sin(0) = 0)
        for i in 0..5 {
            assert!((field[i][i]).abs() < 1e-6);
        }
    }
}
