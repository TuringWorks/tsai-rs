//! Calibration analysis for classification models.
//!
//! Calibration measures how well the predicted probabilities match
//! the actual frequencies of outcomes. A well-calibrated model's
//! predictions of 80% confidence should be correct ~80% of the time.

use serde::{Deserialize, Serialize};

/// Calibration analysis results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResult {
    /// Expected Calibration Error (ECE).
    pub ece: f32,
    /// Maximum Calibration Error (MCE).
    pub mce: f32,
    /// Number of bins used.
    pub n_bins: usize,
    /// Bin edges (n_bins + 1 values from 0 to 1).
    pub bin_edges: Vec<f32>,
    /// Average confidence in each bin.
    pub bin_confidences: Vec<f32>,
    /// Actual accuracy in each bin.
    pub bin_accuracies: Vec<f32>,
    /// Number of samples in each bin.
    pub bin_counts: Vec<usize>,
    /// Total number of samples.
    pub total_samples: usize,
}

impl CalibrationResult {
    /// Check if the model is well-calibrated (ECE < threshold).
    pub fn is_well_calibrated(&self, threshold: f32) -> bool {
        self.ece < threshold
    }

    /// Get reliability diagram data as (confidence, accuracy, count) tuples.
    pub fn reliability_diagram_data(&self) -> Vec<(f32, f32, usize)> {
        self.bin_confidences
            .iter()
            .zip(self.bin_accuracies.iter())
            .zip(self.bin_counts.iter())
            .map(|((&conf, &acc), &count)| (conf, acc, count))
            .collect()
    }

    /// Get bins that are overconfident (confidence > accuracy).
    pub fn overconfident_bins(&self) -> Vec<usize> {
        self.bin_confidences
            .iter()
            .zip(self.bin_accuracies.iter())
            .enumerate()
            .filter(|(_, (&conf, &acc))| conf > acc + 0.01)
            .map(|(i, _)| i)
            .collect()
    }

    /// Get bins that are underconfident (confidence < accuracy).
    pub fn underconfident_bins(&self) -> Vec<usize> {
        self.bin_confidences
            .iter()
            .zip(self.bin_accuracies.iter())
            .enumerate()
            .filter(|(_, (&conf, &acc))| conf < acc - 0.01)
            .map(|(i, _)| i)
            .collect()
    }

    /// Display calibration summary as text.
    pub fn summary(&self) -> String {
        let mut output = String::new();
        output.push_str("=== Calibration Analysis ===\n\n");
        output.push_str(&format!("Expected Calibration Error (ECE): {:.4}\n", self.ece));
        output.push_str(&format!("Maximum Calibration Error (MCE): {:.4}\n", self.mce));
        output.push_str(&format!("Number of bins: {}\n", self.n_bins));
        output.push_str(&format!("Total samples: {}\n\n", self.total_samples));

        output.push_str("Reliability Diagram:\n");
        output.push_str("  Bin  | Confidence | Accuracy | Count | Gap\n");
        output.push_str("-------+------------+----------+-------+------\n");

        for i in 0..self.n_bins {
            if self.bin_counts[i] > 0 {
                let gap = self.bin_confidences[i] - self.bin_accuracies[i];
                let gap_str = if gap > 0.01 {
                    format!("+{:.2}", gap)
                } else if gap < -0.01 {
                    format!("{:.2}", gap)
                } else {
                    "~0.00".to_string()
                };

                output.push_str(&format!(
                    "  {:3}  |   {:.3}    |  {:.3}   | {:5} | {}\n",
                    i + 1,
                    self.bin_confidences[i],
                    self.bin_accuracies[i],
                    self.bin_counts[i],
                    gap_str
                ));
            }
        }

        let overconf = self.overconfident_bins();
        let underconf = self.underconfident_bins();

        if !overconf.is_empty() {
            output.push_str(&format!(
                "\nOverconfident bins: {:?}\n",
                overconf.iter().map(|&i| i + 1).collect::<Vec<_>>()
            ));
        }
        if !underconf.is_empty() {
            output.push_str(&format!(
                "Underconfident bins: {:?}\n",
                underconf.iter().map(|&i| i + 1).collect::<Vec<_>>()
            ));
        }

        output
    }
}

/// Compute calibration metrics for classification predictions.
///
/// # Arguments
///
/// * `confidences` - Predicted confidence scores (max probability per sample)
/// * `predictions` - Predicted class labels
/// * `targets` - True class labels
/// * `n_bins` - Number of bins for the reliability diagram (default: 10)
///
/// # Returns
///
/// CalibrationResult containing ECE, MCE, and per-bin statistics.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_analysis::calibration::compute_calibration;
///
/// let confidences = vec![0.9, 0.8, 0.7, 0.6];
/// let predictions = vec![1, 0, 1, 0];
/// let targets = vec![1, 0, 0, 0];  // 3/4 correct
///
/// let result = compute_calibration(&confidences, &predictions, &targets, 10);
/// println!("ECE: {:.4}", result.ece);
/// ```
pub fn compute_calibration(
    confidences: &[f32],
    predictions: &[i64],
    targets: &[i64],
    n_bins: usize,
) -> CalibrationResult {
    let n_bins = n_bins.max(1);
    let n = confidences.len();

    assert_eq!(
        predictions.len(),
        n,
        "predictions length must match confidences"
    );
    assert_eq!(targets.len(), n, "targets length must match confidences");

    // Create bin edges
    let bin_edges: Vec<f32> = (0..=n_bins).map(|i| i as f32 / n_bins as f32).collect();

    // Initialize bin accumulators
    let mut bin_sums = vec![0.0f32; n_bins];
    let mut bin_correct = vec![0usize; n_bins];
    let mut bin_counts = vec![0usize; n_bins];

    // Assign samples to bins
    for i in 0..n {
        let conf = confidences[i].clamp(0.0, 1.0);
        let correct = predictions[i] == targets[i];

        // Find bin (handle edge case of conf == 1.0)
        let bin_idx = if conf >= 1.0 {
            n_bins - 1
        } else {
            ((conf * n_bins as f32) as usize).min(n_bins - 1)
        };

        bin_sums[bin_idx] += conf;
        bin_counts[bin_idx] += 1;
        if correct {
            bin_correct[bin_idx] += 1;
        }
    }

    // Compute per-bin statistics
    let mut bin_confidences = vec![0.0f32; n_bins];
    let mut bin_accuracies = vec![0.0f32; n_bins];

    for i in 0..n_bins {
        if bin_counts[i] > 0 {
            bin_confidences[i] = bin_sums[i] / bin_counts[i] as f32;
            bin_accuracies[i] = bin_correct[i] as f32 / bin_counts[i] as f32;
        } else {
            // Empty bins get midpoint confidence
            bin_confidences[i] = (bin_edges[i] + bin_edges[i + 1]) / 2.0;
            bin_accuracies[i] = bin_confidences[i];
        }
    }

    // Compute ECE and MCE
    let mut ece = 0.0f32;
    let mut mce = 0.0f32;

    for i in 0..n_bins {
        let gap = (bin_confidences[i] - bin_accuracies[i]).abs();
        let weight = bin_counts[i] as f32 / n.max(1) as f32;
        ece += weight * gap;
        mce = mce.max(gap);
    }

    CalibrationResult {
        ece,
        mce,
        n_bins,
        bin_edges,
        bin_confidences,
        bin_accuracies,
        bin_counts,
        total_samples: n,
    }
}

/// Compute calibration from probability matrix.
///
/// Convenience function that extracts confidences and predictions
/// from a probability matrix.
///
/// # Arguments
///
/// * `probabilities` - Probability matrix of shape (n_samples, n_classes)
/// * `targets` - True class labels
/// * `n_bins` - Number of bins
pub fn calibration_from_probs(
    probabilities: &[Vec<f32>],
    targets: &[i64],
    n_bins: usize,
) -> CalibrationResult {
    let confidences: Vec<f32> = probabilities
        .iter()
        .map(|probs| {
            probs
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .collect();

    let predictions: Vec<i64> = probabilities
        .iter()
        .map(|probs| {
            probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as i64)
                .unwrap_or(0)
        })
        .collect();

    compute_calibration(&confidences, &predictions, targets, n_bins)
}

/// Temperature scaling for calibration.
///
/// Applies temperature scaling to logits to improve calibration.
/// Temperature > 1 softens probabilities (reduces overconfidence).
/// Temperature < 1 sharpens probabilities.
///
/// # Arguments
///
/// * `logits` - Raw model outputs
/// * `temperature` - Scaling temperature
///
/// # Returns
///
/// Calibrated probabilities (via softmax with temperature).
pub fn temperature_scale(logits: &[Vec<f32>], temperature: f32) -> Vec<Vec<f32>> {
    let t = temperature.max(1e-6);

    logits
        .iter()
        .map(|sample_logits| {
            // Scale logits
            let scaled: Vec<f32> = sample_logits.iter().map(|&l| l / t).collect();

            // Softmax
            let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_vals: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
            let sum: f32 = exp_vals.iter().sum();

            exp_vals.iter().map(|&e| e / sum).collect()
        })
        .collect()
}

/// Find optimal temperature for calibration.
///
/// Uses grid search to find temperature that minimizes ECE.
///
/// # Arguments
///
/// * `logits` - Raw model outputs
/// * `targets` - True class labels
/// * `n_bins` - Number of bins for ECE computation
/// * `temp_range` - (min_temp, max_temp) range to search
/// * `n_steps` - Number of temperature values to try
///
/// # Returns
///
/// (optimal_temperature, calibration_result)
pub fn find_optimal_temperature(
    logits: &[Vec<f32>],
    targets: &[i64],
    n_bins: usize,
    temp_range: (f32, f32),
    n_steps: usize,
) -> (f32, CalibrationResult) {
    let (min_t, max_t) = temp_range;
    let step_size = (max_t - min_t) / (n_steps.max(1) - 1) as f32;

    let mut best_temp = 1.0f32;
    let mut best_ece = f32::INFINITY;
    let mut best_result = None;

    for i in 0..n_steps {
        let temp = min_t + i as f32 * step_size;
        let probs = temperature_scale(logits, temp);
        let result = calibration_from_probs(&probs, targets, n_bins);

        if result.ece < best_ece {
            best_ece = result.ece;
            best_temp = temp;
            best_result = Some(result);
        }
    }

    (best_temp, best_result.unwrap_or_else(|| {
        let probs = temperature_scale(logits, 1.0);
        calibration_from_probs(&probs, targets, n_bins)
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_calibration() {
        // Perfectly calibrated: 100% confident and correct
        let confidences = vec![1.0, 1.0, 1.0, 1.0];
        let predictions = vec![0, 1, 2, 0];
        let targets = vec![0, 1, 2, 0];

        let result = compute_calibration(&confidences, &predictions, &targets, 10);

        assert!(result.ece < 0.01);
        assert_eq!(result.total_samples, 4);
    }

    #[test]
    fn test_calibration_overconfident() {
        // Overconfident: 90% confident but only 50% correct
        let confidences = vec![0.9, 0.9, 0.9, 0.9];
        let predictions = vec![1, 1, 1, 1];
        let targets = vec![1, 1, 0, 0]; // 2/4 correct

        let result = compute_calibration(&confidences, &predictions, &targets, 10);

        // ECE should be high (0.9 - 0.5 = 0.4)
        assert!(result.ece > 0.3);
        assert!(!result.overconfident_bins().is_empty());
    }

    #[test]
    fn test_calibration_from_probs() {
        let probabilities = vec![
            vec![0.9, 0.1],
            vec![0.3, 0.7],
            vec![0.6, 0.4],
        ];
        let targets = vec![0, 1, 0];

        let result = calibration_from_probs(&probabilities, &targets, 5);

        assert_eq!(result.total_samples, 3);
        assert_eq!(result.n_bins, 5);
    }

    #[test]
    fn test_temperature_scale() {
        let logits = vec![vec![2.0, 1.0, 0.0]];

        // Temperature 1.0 should give normal softmax
        let probs_t1 = temperature_scale(&logits, 1.0);
        assert!((probs_t1[0].iter().sum::<f32>() - 1.0).abs() < 1e-6);

        // Higher temperature should soften probabilities
        let probs_t2 = temperature_scale(&logits, 2.0);
        assert!(probs_t2[0][0] < probs_t1[0][0]); // Max prob should decrease

        // Lower temperature should sharpen
        let probs_t05 = temperature_scale(&logits, 0.5);
        assert!(probs_t05[0][0] > probs_t1[0][0]); // Max prob should increase
    }

    #[test]
    fn test_find_optimal_temperature() {
        let logits = vec![
            vec![2.0, 0.0],
            vec![1.5, 0.0],
            vec![0.0, 1.0],
        ];
        let targets = vec![0, 0, 1];

        let (temp, result) = find_optimal_temperature(&logits, &targets, 5, (0.5, 3.0), 10);

        assert!(temp >= 0.5 && temp <= 3.0);
        assert!(result.ece >= 0.0);
    }

    #[test]
    fn test_calibration_result_methods() {
        let confidences = vec![0.8, 0.7, 0.6, 0.5];
        let predictions = vec![0, 0, 1, 1];
        let targets = vec![0, 1, 1, 0];

        let result = compute_calibration(&confidences, &predictions, &targets, 4);

        assert!(result.is_well_calibrated(0.5));
        assert!(!result.reliability_diagram_data().is_empty());
        assert!(!result.summary().is_empty());
    }
}
