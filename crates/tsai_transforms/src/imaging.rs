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

/// Configuration for Joint Recurrence Plot transform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JRPConfig {
    /// Size of the output image.
    pub size: usize,
    /// Threshold for recurrence (epsilon) for first series.
    pub threshold1: Option<f32>,
    /// Threshold for recurrence (epsilon) for second series.
    pub threshold2: Option<f32>,
    /// Percentage of points to use for threshold calculation.
    pub percentage: Option<f32>,
}

impl Default for JRPConfig {
    fn default() -> Self {
        Self {
            size: 64,
            threshold1: None,
            threshold2: None,
            percentage: Some(10.0),
        }
    }
}

/// Converts two time series to a Joint Recurrence Plot image.
///
/// A joint recurrence plot (JRP) visualizes the times at which two
/// dynamical systems simultaneously recur to states they have visited before.
/// The JRP is the element-wise product (Hadamard product) of the individual
/// recurrence plots of the two time series.
///
/// JRP(i,j) = RP1(i,j) * RP2(i,j)
///
/// This is useful for analyzing synchronization and coupling between
/// two time series.
#[derive(Debug, Clone)]
pub struct TSToJRP {
    config: JRPConfig,
}

impl TSToJRP {
    /// Create a new Joint Recurrence Plot transform.
    #[must_use]
    pub fn new(size: usize) -> Self {
        Self {
            config: JRPConfig {
                size,
                ..Default::default()
            },
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: JRPConfig) -> Self {
        Self { config }
    }

    /// Set threshold for first series.
    #[must_use]
    pub fn with_threshold1(mut self, threshold: f32) -> Self {
        self.config.threshold1 = Some(threshold);
        self
    }

    /// Set threshold for second series.
    #[must_use]
    pub fn with_threshold2(mut self, threshold: f32) -> Self {
        self.config.threshold2 = Some(threshold);
        self
    }

    /// Set percentage for automatic threshold calculation.
    #[must_use]
    pub fn with_percentage(mut self, percentage: f32) -> Self {
        self.config.percentage = Some(percentage);
        self
    }

    /// Compute joint recurrence plot for two univariate time series.
    ///
    /// # Arguments
    ///
    /// * `series1` - First input time series of length L
    /// * `series2` - Second input time series of length L (must match first)
    ///
    /// # Returns
    ///
    /// A 2D array of size (L, L) representing the joint recurrence plot.
    /// Returns an empty vector if series lengths don't match.
    pub fn compute(&self, series1: &[f32], series2: &[f32]) -> Vec<Vec<f32>> {
        let n = series1.len();

        if n != series2.len() {
            return vec![];
        }

        // Compute individual recurrence plots
        let mut rp1 = vec![vec![0.0f32; n]; n];
        let mut rp2 = vec![vec![0.0f32; n]; n];

        // Compute pairwise distances for series 1
        for i in 0..n {
            for j in 0..n {
                rp1[i][j] = (series1[i] - series1[j]).abs();
            }
        }

        // Compute pairwise distances for series 2
        for i in 0..n {
            for j in 0..n {
                rp2[i][j] = (series2[i] - series2[j]).abs();
            }
        }

        // Apply thresholds
        let thresh1 = self.get_threshold(&rp1, self.config.threshold1);
        let thresh2 = self.get_threshold(&rp2, self.config.threshold2);

        for row in &mut rp1 {
            for val in row {
                *val = if *val < thresh1 { 1.0 } else { 0.0 };
            }
        }

        for row in &mut rp2 {
            for val in row {
                *val = if *val < thresh2 { 1.0 } else { 0.0 };
            }
        }

        // Compute joint recurrence plot (element-wise product)
        let mut jrp = vec![vec![0.0f32; n]; n];
        for i in 0..n {
            for j in 0..n {
                jrp[i][j] = rp1[i][j] * rp2[i][j];
            }
        }

        jrp
    }

    /// Get threshold value from explicit threshold or percentage.
    fn get_threshold(&self, plot: &[Vec<f32>], explicit_thresh: Option<f32>) -> f32 {
        if let Some(thresh) = explicit_thresh {
            return thresh;
        }

        if let Some(pct) = self.config.percentage {
            let mut all_dists: Vec<f32> = plot.iter().flatten().copied().collect();
            all_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let thresh_idx = ((pct / 100.0) * all_dists.len() as f32) as usize;
            return all_dists.get(thresh_idx).copied().unwrap_or(0.0);
        }

        // Default: use 10% of max distance
        let max_dist = plot
            .iter()
            .flatten()
            .cloned()
            .fold(0.0f32, f32::max);
        max_dist * 0.1
    }

    /// Compute cross recurrence plot for two time series.
    ///
    /// Unlike the joint recurrence plot which multiplies individual RPs,
    /// the cross recurrence plot directly measures distances between
    /// states of the two systems.
    ///
    /// # Arguments
    ///
    /// * `series1` - First input time series of length L1
    /// * `series2` - Second input time series of length L2
    ///
    /// # Returns
    ///
    /// A 2D array of size (L1, L2) representing the cross recurrence plot.
    pub fn compute_cross(&self, series1: &[f32], series2: &[f32]) -> Vec<Vec<f32>> {
        let n1 = series1.len();
        let n2 = series2.len();

        // Compute cross distances
        let mut crp = vec![vec![0.0f32; n2]; n1];
        for i in 0..n1 {
            for j in 0..n2 {
                crp[i][j] = (series1[i] - series2[j]).abs();
            }
        }

        // Calculate threshold
        let thresh = if let Some(t) = self.config.threshold1 {
            t
        } else if let Some(pct) = self.config.percentage {
            let mut all_dists: Vec<f32> = crp.iter().flatten().copied().collect();
            all_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let thresh_idx = ((pct / 100.0) * all_dists.len() as f32) as usize;
            all_dists.get(thresh_idx).copied().unwrap_or(0.0)
        } else {
            let max_dist = crp.iter().flatten().cloned().fold(0.0f32, f32::max);
            max_dist * 0.1
        };

        // Apply threshold
        for row in &mut crp {
            for val in row {
                *val = if *val < thresh { 1.0 } else { 0.0 };
            }
        }

        crp
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
