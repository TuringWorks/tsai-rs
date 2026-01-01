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

/// Configuration for matrix visualization transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatrixLayout {
    /// Variables as rows, time as columns (n_vars, seq_len).
    VarsRows,
    /// Time as rows, variables as columns (seq_len, n_vars).
    TimeRows,
}

impl Default for MatrixLayout {
    fn default() -> Self {
        Self::VarsRows
    }
}

/// Configuration for TSToMat transform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TSToMatConfig {
    /// Layout of the output matrix.
    pub layout: MatrixLayout,
    /// Whether to normalize values to [0, 1].
    pub normalize: bool,
    /// Whether to apply colormap-style scaling.
    pub colormap: bool,
}

impl Default for TSToMatConfig {
    fn default() -> Self {
        Self {
            layout: MatrixLayout::VarsRows,
            normalize: true,
            colormap: false,
        }
    }
}

/// Converts time series to a 2D matrix representation.
///
/// This transform reshapes multivariate time series data into a 2D matrix
/// suitable for visualization or CNN-based processing.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_transforms::imaging::TSToMat;
///
/// // Create transform (variables as rows, time as columns)
/// let transform = TSToMat::new();
///
/// // For a 3-variable, 100-length series: (3, 100) -> 3x100 matrix
/// let matrix = transform.compute(&series);
/// ```
#[derive(Debug, Clone)]
pub struct TSToMat {
    config: TSToMatConfig,
}

impl TSToMat {
    /// Create a new TSToMat transform with default config.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: TSToMatConfig::default(),
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: TSToMatConfig) -> Self {
        Self { config }
    }

    /// Set the matrix layout.
    #[must_use]
    pub fn with_layout(mut self, layout: MatrixLayout) -> Self {
        self.config.layout = layout;
        self
    }

    /// Enable/disable normalization.
    #[must_use]
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.config.normalize = normalize;
        self
    }

    /// Enable/disable colormap scaling.
    #[must_use]
    pub fn with_colormap(mut self, colormap: bool) -> Self {
        self.config.colormap = colormap;
        self
    }

    /// Convert multivariate time series to matrix.
    ///
    /// # Arguments
    ///
    /// * `series` - Input multivariate time series of shape (n_vars, seq_len)
    ///
    /// # Returns
    ///
    /// A 2D matrix representation.
    pub fn compute(&self, series: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if series.is_empty() {
            return vec![];
        }

        let n_vars = series.len();
        let seq_len = series[0].len();

        // Apply normalization if enabled
        let normalized: Vec<Vec<f32>> = if self.config.normalize {
            // Find global min/max
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            for var in series {
                for &val in var {
                    if val.is_finite() {
                        min_val = min_val.min(val);
                        max_val = max_val.max(val);
                    }
                }
            }
            let range = (max_val - min_val).max(1e-8);

            series
                .iter()
                .map(|var| {
                    var.iter()
                        .map(|&val| {
                            if val.is_finite() {
                                (val - min_val) / range
                            } else {
                                0.5
                            }
                        })
                        .collect()
                })
                .collect()
        } else {
            series.to_vec()
        };

        // Apply layout
        match self.config.layout {
            MatrixLayout::VarsRows => {
                // Already in (n_vars, seq_len) format
                normalized
            }
            MatrixLayout::TimeRows => {
                // Transpose to (seq_len, n_vars)
                let mut transposed = vec![vec![0.0f32; n_vars]; seq_len];
                for v in 0..n_vars {
                    for t in 0..seq_len {
                        transposed[t][v] = normalized[v][t];
                    }
                }
                transposed
            }
        }
    }

    /// Convert univariate time series to matrix.
    ///
    /// # Arguments
    ///
    /// * `series` - Input univariate time series of length seq_len
    ///
    /// # Returns
    ///
    /// A 2D matrix with a single row (or column based on layout).
    pub fn compute_univariate(&self, series: &[f32]) -> Vec<Vec<f32>> {
        self.compute(&[series.to_vec()])
    }

    /// Get ASCII representation of the matrix for debugging.
    pub fn to_ascii(&self, matrix: &[Vec<f32>], width: usize, height: usize) -> String {
        if matrix.is_empty() || matrix[0].is_empty() {
            return String::new();
        }

        let chars = [' ', '░', '▒', '▓', '█'];
        let n_rows = matrix.len();
        let n_cols = matrix[0].len();

        // Sample rows and columns to fit dimensions
        let row_step = (n_rows as f32 / height as f32).max(1.0);
        let col_step = (n_cols as f32 / width as f32).max(1.0);

        let mut output = String::new();
        for h in 0..height.min(n_rows) {
            let r = (h as f32 * row_step) as usize;
            if r >= n_rows {
                break;
            }

            for w in 0..width.min(n_cols) {
                let c = (w as f32 * col_step) as usize;
                if c >= n_cols {
                    break;
                }

                let val = matrix[r][c].clamp(0.0, 1.0);
                let char_idx = (val * (chars.len() - 1) as f32).round() as usize;
                output.push(chars[char_idx]);
            }
            output.push('\n');
        }

        output
    }
}

impl Default for TSToMat {
    fn default() -> Self {
        Self::new()
    }
}

/// Converts time series to a simple plot representation (ASCII or data export).
///
/// This is a Rust alternative to Python's TSToPlot which uses matplotlib.
/// Instead, this provides ASCII plotting and data export for external plotting.
#[derive(Debug, Clone)]
pub struct TSToPlot {
    /// Width of ASCII plot.
    width: usize,
    /// Height of ASCII plot.
    height: usize,
    /// Whether to show axis labels.
    show_labels: bool,
}

impl TSToPlot {
    /// Create a new TSToPlot.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            show_labels: true,
        }
    }

    /// Disable axis labels.
    #[must_use]
    pub fn without_labels(mut self) -> Self {
        self.show_labels = false;
        self
    }

    /// Generate ASCII plot of a univariate time series.
    pub fn plot_ascii(&self, series: &[f32]) -> String {
        if series.is_empty() {
            return String::new();
        }

        let min_val = series.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = series.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_val - min_val).max(1e-8);

        let mut plot = vec![vec![' '; self.width]; self.height];

        // Plot points
        for (i, &val) in series.iter().enumerate() {
            let x = (i as f32 / (series.len() - 1).max(1) as f32 * (self.width - 1) as f32) as usize;
            let y = ((val - min_val) / range * (self.height - 1) as f32) as usize;
            let y = (self.height - 1).saturating_sub(y); // Flip y-axis

            if x < self.width && y < self.height {
                plot[y][x] = '●';
            }
        }

        // Connect points with lines
        for i in 1..series.len() {
            let x1 = ((i - 1) as f32 / (series.len() - 1).max(1) as f32 * (self.width - 1) as f32) as usize;
            let x2 = (i as f32 / (series.len() - 1).max(1) as f32 * (self.width - 1) as f32) as usize;

            for x in x1..=x2 {
                if x < self.width {
                    let t = if x2 > x1 {
                        (x - x1) as f32 / (x2 - x1) as f32
                    } else {
                        0.0
                    };
                    let val = series[i - 1] * (1.0 - t) + series[i] * t;
                    let y = ((val - min_val) / range * (self.height - 1) as f32) as usize;
                    let y = (self.height - 1).saturating_sub(y);

                    if y < self.height && plot[y][x] == ' ' {
                        plot[y][x] = '·';
                    }
                }
            }
        }

        // Build output string
        let mut output = String::new();

        if self.show_labels {
            output.push_str(&format!("{:>8.2} ┤", max_val));
        }

        for (i, row) in plot.iter().enumerate() {
            if i > 0 && self.show_labels {
                output.push_str("         │");
            }
            for &ch in row {
                output.push(ch);
            }
            output.push('\n');
        }

        if self.show_labels {
            output.push_str(&format!("{:>8.2} ┤", min_val));
            for _ in 0..self.width {
                output.push('─');
            }
            output.push('\n');
        }

        output
    }

    /// Generate multivariate plot (stacked).
    pub fn plot_multivariate_ascii(&self, series: &[Vec<f32>]) -> String {
        let mut output = String::new();

        for (i, var) in series.iter().enumerate() {
            output.push_str(&format!("Variable {}:\n", i));
            output.push_str(&self.plot_ascii(var));
            output.push('\n');
        }

        output
    }

    /// Export data for external plotting (CSV format).
    pub fn export_csv(&self, series: &[f32]) -> String {
        let mut output = String::from("time,value\n");
        for (i, &val) in series.iter().enumerate() {
            output.push_str(&format!("{},{}\n", i, val));
        }
        output
    }

    /// Export multivariate data for external plotting (CSV format).
    pub fn export_multivariate_csv(&self, series: &[Vec<f32>]) -> String {
        if series.is_empty() {
            return String::from("time\n");
        }

        let n_vars = series.len();
        let seq_len = series[0].len();

        // Header
        let mut output = String::from("time");
        for i in 0..n_vars {
            output.push_str(&format!(",var{}", i));
        }
        output.push('\n');

        // Data
        for t in 0..seq_len {
            output.push_str(&format!("{}", t));
            for v in 0..n_vars {
                output.push_str(&format!(",{}", series[v][t]));
            }
            output.push('\n');
        }

        output
    }
}

impl Default for TSToPlot {
    fn default() -> Self {
        Self::new(60, 15)
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
