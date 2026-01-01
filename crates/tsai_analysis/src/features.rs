//! Time series feature extraction similar to tsfresh.
//!
//! Provides statistical and spectral features for traditional ML pipelines.
//!
//! # Example
//!
//! ```rust,ignore
//! use tsai_analysis::features::{extract_features, FeatureSet, FeatureExtractor};
//!
//! // Extract all features from a time series
//! let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let features = extract_features(&series, FeatureSet::Minimal);
//!
//! // Or use the extractor for batch processing
//! let extractor = FeatureExtractor::new(FeatureSet::Comprehensive);
//! let feature_matrix = extractor.transform(&batch_data)?;
//! ```

use std::collections::HashMap;

/// Feature set presets for different use cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureSet {
    /// Minimal set: mean, std, min, max, length (5 features)
    Minimal,
    /// Efficient set: adds median, skewness, kurtosis, quantiles (15 features)
    Efficient,
    /// Comprehensive set: adds autocorrelation, entropy, crossing rates (30+ features)
    Comprehensive,
    /// All available features (50+ features)
    All,
}

impl Default for FeatureSet {
    fn default() -> Self {
        Self::Efficient
    }
}

/// Extract features from a single time series.
///
/// Returns a HashMap of feature name to value.
pub fn extract_features(series: &[f32], feature_set: FeatureSet) -> HashMap<String, f32> {
    let mut features = HashMap::new();

    if series.is_empty() {
        return features;
    }

    // Basic statistics (always included)
    let n = series.len() as f32;
    let mean = series.iter().sum::<f32>() / n;
    let min = series.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = series.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    features.insert("length".to_string(), n);
    features.insert("mean".to_string(), mean);
    features.insert("min".to_string(), min);
    features.insert("max".to_string(), max);

    // Variance and standard deviation
    let variance = series.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = variance.sqrt();
    features.insert("std".to_string(), std);
    features.insert("variance".to_string(), variance);

    if matches!(feature_set, FeatureSet::Minimal) {
        return features;
    }

    // Efficient features
    let mut sorted = series.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    features.insert("median".to_string(), median(&sorted));
    features.insert("q1".to_string(), quantile(&sorted, 0.25));
    features.insert("q3".to_string(), quantile(&sorted, 0.75));
    features.insert("iqr".to_string(), quantile(&sorted, 0.75) - quantile(&sorted, 0.25));
    features.insert("range".to_string(), max - min);

    // Skewness and kurtosis
    if std > 0.0 {
        let skewness = series.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f32>() / n;
        let kurtosis = series.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f32>() / n - 3.0;
        features.insert("skewness".to_string(), skewness);
        features.insert("kurtosis".to_string(), kurtosis);
    }

    // Sum of absolute values
    features.insert("abs_sum".to_string(), series.iter().map(|x| x.abs()).sum());

    // Root mean square
    let rms = (series.iter().map(|x| x.powi(2)).sum::<f32>() / n).sqrt();
    features.insert("rms".to_string(), rms);

    if matches!(feature_set, FeatureSet::Efficient) {
        return features;
    }

    // Comprehensive features
    // First derivative statistics
    if series.len() > 1 {
        let diff: Vec<f32> = series.windows(2).map(|w| w[1] - w[0]).collect();
        let diff_mean = diff.iter().sum::<f32>() / diff.len() as f32;
        let diff_std = (diff.iter().map(|x| (x - diff_mean).powi(2)).sum::<f32>() / diff.len() as f32).sqrt();
        let diff_abs_sum: f32 = diff.iter().map(|x| x.abs()).sum();

        features.insert("diff_mean".to_string(), diff_mean);
        features.insert("diff_std".to_string(), diff_std);
        features.insert("diff_abs_sum".to_string(), diff_abs_sum);
    }

    // Zero crossing rate
    let zero_crossings = series
        .windows(2)
        .filter(|w| (w[0] >= 0.0 && w[1] < 0.0) || (w[0] < 0.0 && w[1] >= 0.0))
        .count() as f32;
    features.insert("zero_crossing_rate".to_string(), zero_crossings / (n - 1.0).max(1.0));

    // Mean crossing rate
    let mean_crossings = series
        .windows(2)
        .filter(|w| (w[0] >= mean && w[1] < mean) || (w[0] < mean && w[1] >= mean))
        .count() as f32;
    features.insert("mean_crossing_rate".to_string(), mean_crossings / (n - 1.0).max(1.0));

    // Autocorrelation at lag 1
    if series.len() > 1 {
        let ac1 = autocorrelation(series, 1);
        features.insert("autocorr_lag1".to_string(), ac1);
    }
    if series.len() > 5 {
        features.insert("autocorr_lag5".to_string(), autocorrelation(series, 5));
    }
    if series.len() > 10 {
        features.insert("autocorr_lag10".to_string(), autocorrelation(series, 10));
    }

    // Approximate entropy
    if series.len() >= 10 {
        let apen = approximate_entropy(series, 2, 0.2 * std);
        features.insert("approx_entropy".to_string(), apen);
    }

    // Count above/below mean
    let above_mean = series.iter().filter(|&&x| x > mean).count() as f32;
    let below_mean = series.iter().filter(|&&x| x < mean).count() as f32;
    features.insert("count_above_mean".to_string(), above_mean);
    features.insert("count_below_mean".to_string(), below_mean);
    features.insert("ratio_above_mean".to_string(), above_mean / n);

    // First/last location of min/max
    if let Some((first_max_idx, _)) = series.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
        features.insert("first_max_location".to_string(), first_max_idx as f32 / n);
    }
    if let Some((first_min_idx, _)) = series.iter().enumerate().min_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
        features.insert("first_min_location".to_string(), first_min_idx as f32 / n);
    }

    // Longest strike above/below mean
    features.insert("longest_strike_above_mean".to_string(), longest_strike(series, mean, true) as f32);
    features.insert("longest_strike_below_mean".to_string(), longest_strike(series, mean, false) as f32);

    if matches!(feature_set, FeatureSet::Comprehensive) {
        return features;
    }

    // All features (additional)
    // Second derivative
    if series.len() > 2 {
        let diff2: Vec<f32> = series.windows(3).map(|w| w[2] - 2.0 * w[1] + w[0]).collect();
        let diff2_mean = diff2.iter().sum::<f32>() / diff2.len() as f32;
        let diff2_std = (diff2.iter().map(|x| (x - diff2_mean).powi(2)).sum::<f32>() / diff2.len() as f32).sqrt();
        features.insert("diff2_mean".to_string(), diff2_mean);
        features.insert("diff2_std".to_string(), diff2_std);
    }

    // Energy and peak count
    let energy: f32 = series.iter().map(|x| x.powi(2)).sum();
    features.insert("energy".to_string(), energy);

    let peak_count = count_peaks(series, 0.1 * std);
    features.insert("peak_count".to_string(), peak_count as f32);

    // Quantile features
    for q in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9] {
        features.insert(format!("q{:.0}", q * 100.0), quantile(&sorted, q));
    }

    // Binned entropy
    let binned_ent = binned_entropy(series, 10);
    features.insert("binned_entropy".to_string(), binned_ent);

    // Absolute energy
    features.insert("abs_energy".to_string(), series.iter().map(|x| x.abs().powi(2)).sum());

    features
}

/// Feature extractor for batch processing.
#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    feature_set: FeatureSet,
    feature_names: Option<Vec<String>>,
}

impl FeatureExtractor {
    /// Create a new feature extractor.
    pub fn new(feature_set: FeatureSet) -> Self {
        Self {
            feature_set,
            feature_names: None,
        }
    }

    /// Extract features from a batch of time series.
    ///
    /// Input: Vec of time series, each as Vec<f32>
    /// Output: 2D array (n_samples, n_features) and feature names
    pub fn transform(&mut self, batch: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<String>) {
        if batch.is_empty() {
            return (vec![], vec![]);
        }

        // Extract features from first sample to get feature names
        let first_features = extract_features(&batch[0], self.feature_set);
        let mut feature_names: Vec<String> = first_features.keys().cloned().collect();
        feature_names.sort();

        self.feature_names = Some(feature_names.clone());

        // Extract features for all samples
        let feature_matrix: Vec<Vec<f32>> = batch
            .iter()
            .map(|series| {
                let features = extract_features(series, self.feature_set);
                feature_names
                    .iter()
                    .map(|name| *features.get(name).unwrap_or(&f32::NAN))
                    .collect()
            })
            .collect();

        (feature_matrix, feature_names)
    }

    /// Get feature names after transformation.
    pub fn feature_names(&self) -> Option<&[String]> {
        self.feature_names.as_deref()
    }

    /// Get number of features for a given feature set.
    pub fn n_features(feature_set: FeatureSet) -> usize {
        match feature_set {
            FeatureSet::Minimal => 6,
            FeatureSet::Efficient => 15,
            FeatureSet::Comprehensive => 30,
            FeatureSet::All => 50,
        }
    }
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new(FeatureSet::Efficient)
    }
}

// Helper functions

fn median(sorted: &[f32]) -> f32 {
    let n = sorted.len();
    if n == 0 {
        return f32::NAN;
    }
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

fn quantile(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return f32::NAN;
    }
    let idx = (q * (sorted.len() - 1) as f32).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn autocorrelation(series: &[f32], lag: usize) -> f32 {
    if series.len() <= lag {
        return f32::NAN;
    }

    let n = series.len();
    let mean = series.iter().sum::<f32>() / n as f32;

    let mut num = 0.0;
    let mut den = 0.0;

    for i in 0..n {
        den += (series[i] - mean).powi(2);
        if i + lag < n {
            num += (series[i] - mean) * (series[i + lag] - mean);
        }
    }

    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}

fn approximate_entropy(series: &[f32], m: usize, r: f32) -> f32 {
    if series.len() < m + 1 {
        return f32::NAN;
    }

    let n = series.len();

    let phi = |m: usize| -> f32 {
        let patterns: Vec<Vec<f32>> = (0..=n - m)
            .map(|i| series[i..i + m].to_vec())
            .collect();

        let mut c_sum = 0.0;
        for i in 0..patterns.len() {
            let mut count = 0;
            for j in 0..patterns.len() {
                let max_diff = patterns[i]
                    .iter()
                    .zip(&patterns[j])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                if max_diff <= r {
                    count += 1;
                }
            }
            c_sum += (count as f32 / patterns.len() as f32).ln();
        }
        c_sum / patterns.len() as f32
    };

    phi(m) - phi(m + 1)
}

fn longest_strike(series: &[f32], threshold: f32, above: bool) -> usize {
    let mut max_strike = 0;
    let mut current_strike = 0;

    for &val in series {
        let condition = if above { val > threshold } else { val < threshold };
        if condition {
            current_strike += 1;
            max_strike = max_strike.max(current_strike);
        } else {
            current_strike = 0;
        }
    }

    max_strike
}

fn count_peaks(series: &[f32], min_height: f32) -> usize {
    if series.len() < 3 {
        return 0;
    }

    series
        .windows(3)
        .filter(|w| w[1] > w[0] && w[1] > w[2] && w[1] > min_height)
        .count()
}

fn binned_entropy(series: &[f32], n_bins: usize) -> f32 {
    if series.is_empty() || n_bins == 0 {
        return 0.0;
    }

    let min = series.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = series.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    if (max - min).abs() < f32::EPSILON {
        return 0.0;
    }

    let bin_width = (max - min) / n_bins as f32;
    let mut bins = vec![0usize; n_bins];

    for &val in series {
        let bin_idx = ((val - min) / bin_width).floor() as usize;
        let bin_idx = bin_idx.min(n_bins - 1);
        bins[bin_idx] += 1;
    }

    let n = series.len() as f32;
    bins.iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f32 / n;
            -p * p.ln()
        })
        .sum()
}

/// Extract features from multivariate time series.
///
/// Returns features for each variable concatenated.
pub fn extract_multivariate_features(
    series: &[Vec<f32>],
    feature_set: FeatureSet,
) -> HashMap<String, f32> {
    let mut all_features = HashMap::new();

    for (var_idx, var_series) in series.iter().enumerate() {
        let var_features = extract_features(var_series, feature_set);
        for (name, value) in var_features {
            all_features.insert(format!("var{}__{}", var_idx, name), value);
        }
    }

    // Add cross-variable features
    if series.len() >= 2 {
        for i in 0..series.len() {
            for j in (i + 1)..series.len() {
                if series[i].len() == series[j].len() && !series[i].is_empty() {
                    let corr = pearson_correlation(&series[i], &series[j]);
                    all_features.insert(format!("corr_var{}_{}", i, j), corr);
                }
            }
        }
    }

    all_features
}

fn pearson_correlation(x: &[f32], y: &[f32]) -> f32 {
    if x.len() != y.len() || x.is_empty() {
        return f32::NAN;
    }

    let n = x.len() as f32;
    let mean_x = x.iter().sum::<f32>() / n;
    let mean_y = y.iter().sum::<f32>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x == 0.0 || var_y == 0.0 {
        0.0
    } else {
        cov / (var_x.sqrt() * var_y.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_minimal_features() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let features = extract_features(&series, FeatureSet::Minimal);

        assert_eq!(features.get("length"), Some(&5.0));
        assert_eq!(features.get("mean"), Some(&3.0));
        assert_eq!(features.get("min"), Some(&1.0));
        assert_eq!(features.get("max"), Some(&5.0));
        assert!(features.contains_key("std"));
        assert!(features.contains_key("variance"));
    }

    #[test]
    fn test_extract_efficient_features() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let features = extract_features(&series, FeatureSet::Efficient);

        assert!(features.contains_key("median"));
        assert!(features.contains_key("q1"));
        assert!(features.contains_key("q3"));
        assert!(features.contains_key("iqr"));
        assert!(features.contains_key("skewness"));
        assert!(features.contains_key("kurtosis"));
    }

    #[test]
    fn test_extract_comprehensive_features() {
        let series: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let features = extract_features(&series, FeatureSet::Comprehensive);

        assert!(features.contains_key("zero_crossing_rate"));
        assert!(features.contains_key("mean_crossing_rate"));
        assert!(features.contains_key("autocorr_lag1"));
        assert!(features.contains_key("approx_entropy"));
        assert!(features.contains_key("longest_strike_above_mean"));
    }

    #[test]
    fn test_feature_extractor_batch() {
        let batch = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![5.0, 4.0, 3.0, 2.0, 1.0],
            vec![1.0, 3.0, 5.0, 3.0, 1.0],
        ];

        let mut extractor = FeatureExtractor::new(FeatureSet::Minimal);
        let (features, names) = extractor.transform(&batch);

        assert_eq!(features.len(), 3);
        assert_eq!(names.len(), 6); // Minimal has 6 features
        assert!(names.contains(&"mean".to_string()));
        assert!(names.contains(&"std".to_string()));
    }

    #[test]
    fn test_autocorrelation() {
        // Linearly increasing sequence has high positive autocorrelation
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ac = autocorrelation(&series, 1);
        assert!(ac >= 0.7, "Expected high positive autocorrelation, got {}", ac);

        // Random-like alternating sequence should have negative autocorrelation at lag 1
        let alternating = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let ac_alt = autocorrelation(&alternating, 1);
        assert!(ac_alt < 0.0, "Expected negative autocorrelation for alternating, got {}", ac_alt);
    }

    #[test]
    fn test_multivariate_features() {
        let series = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![5.0, 4.0, 3.0, 2.0, 1.0],
        ];
        let features = extract_multivariate_features(&series, FeatureSet::Minimal);

        assert!(features.contains_key("var0__mean"));
        assert!(features.contains_key("var1__mean"));
        assert!(features.contains_key("corr_var0_1"));
    }

    #[test]
    fn test_pearson_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr = pearson_correlation(&x, &y);
        assert!((corr - (-1.0)).abs() < 0.001); // Perfect negative correlation
    }

    #[test]
    fn test_empty_series() {
        let series: Vec<f32> = vec![];
        let features = extract_features(&series, FeatureSet::All);
        assert!(features.is_empty());
    }

    #[test]
    fn test_binned_entropy() {
        // Uniform distribution should have high entropy
        let uniform: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let ent = binned_entropy(&uniform, 10);
        assert!(ent > 2.0);

        // Constant series should have zero entropy
        let constant = vec![1.0; 100];
        let ent = binned_entropy(&constant, 10);
        assert_eq!(ent, 0.0);
    }
}
