//! Training metrics.

use burn::prelude::*;

/// Trait for training metrics.
pub trait Metric<B: Backend>: Send + Sync {
    /// Compute the metric from predictions and targets.
    fn compute(&self, preds: &Tensor<B, 2>, targets: &Tensor<B, 2>) -> f32;

    /// Get the metric name.
    fn name(&self) -> &str;

    /// Whether higher is better.
    fn higher_is_better(&self) -> bool {
        true
    }
}

/// Classification accuracy metric.
#[derive(Debug, Clone, Default)]
pub struct Accuracy;

impl<B: Backend> Metric<B> for Accuracy {
    fn compute(&self, preds: &Tensor<B, 2>, targets: &Tensor<B, 2>) -> f32 {
        let pred_classes = preds.clone().argmax(1);
        let target_classes = if targets.dims()[1] > 1 {
            // One-hot encoded
            targets.clone().argmax(1)
        } else {
            // Class indices - simplified for now
            targets.clone().argmax(1)
        };

        let correct = pred_classes.equal(target_classes);
        let correct_sum: f32 = correct.int().sum().into_scalar().elem();
        let total = preds.dims()[0] as f32;

        correct_sum / total
    }

    fn name(&self) -> &str {
        "accuracy"
    }
}

/// Mean Squared Error metric.
#[derive(Debug, Clone, Default)]
pub struct MSE;

impl<B: Backend> Metric<B> for MSE {
    fn compute(&self, preds: &Tensor<B, 2>, targets: &Tensor<B, 2>) -> f32 {
        let diff = preds.clone() - targets.clone();
        let squared = diff.clone() * diff;
        squared.mean().into_scalar().elem()
    }

    fn name(&self) -> &str {
        "mse"
    }

    fn higher_is_better(&self) -> bool {
        false
    }
}

/// Mean Absolute Error metric.
#[derive(Debug, Clone, Default)]
pub struct MAE;

impl<B: Backend> Metric<B> for MAE {
    fn compute(&self, preds: &Tensor<B, 2>, targets: &Tensor<B, 2>) -> f32 {
        let diff = preds.clone() - targets.clone();
        diff.abs().mean().into_scalar().elem()
    }

    fn name(&self) -> &str {
        "mae"
    }

    fn higher_is_better(&self) -> bool {
        false
    }
}

/// F1 Score metric (macro average).
///
/// Computes the macro-averaged F1 score across all classes.
/// F1 = 2 * (precision * recall) / (precision + recall)
#[derive(Debug, Clone)]
pub struct F1Score {
    n_classes: usize,
}

impl F1Score {
    /// Create a new F1 score metric.
    pub fn new(n_classes: usize) -> Self {
        Self { n_classes }
    }
}

impl<B: Backend> Metric<B> for F1Score {
    fn compute(&self, preds: &Tensor<B, 2>, targets: &Tensor<B, 2>) -> f32 {
        let pred_classes = preds.clone().argmax(1);
        let target_classes = targets.clone().argmax(1);

        // Get predictions as data for CPU computation
        let pred_data = pred_classes.into_data();
        let target_data = target_classes.into_data();

        let pred_vec: Vec<i64> = pred_data
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_default();
        let target_vec: Vec<i64> = target_data
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_default();

        let n_samples = pred_vec.len();
        if n_samples == 0 {
            return 0.0;
        }

        // Compute per-class metrics
        let mut total_f1 = 0.0;
        let mut valid_classes = 0;

        for class in 0..self.n_classes {
            let class_i64 = class as i64;

            // True positives: predicted class and actual class both equal to current class
            let mut tp = 0;
            // False positives: predicted class equals current but actual doesn't
            let mut fp = 0;
            // False negatives: actual class equals current but predicted doesn't
            let mut fn_ = 0;

            for i in 0..n_samples {
                let pred = pred_vec[i];
                let target = target_vec[i];

                if pred == class_i64 && target == class_i64 {
                    tp += 1;
                } else if pred == class_i64 && target != class_i64 {
                    fp += 1;
                } else if pred != class_i64 && target == class_i64 {
                    fn_ += 1;
                }
            }

            // Compute precision and recall for this class
            let precision = if tp + fp > 0 {
                tp as f32 / (tp + fp) as f32
            } else {
                0.0
            };

            let recall = if tp + fn_ > 0 {
                tp as f32 / (tp + fn_) as f32
            } else {
                0.0
            };

            // Compute F1 for this class
            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };

            // Only count classes that appear in the targets
            if tp + fn_ > 0 {
                total_f1 += f1;
                valid_classes += 1;
            }
        }

        // Return macro-averaged F1
        if valid_classes > 0 {
            total_f1 / valid_classes as f32
        } else {
            0.0
        }
    }

    fn name(&self) -> &str {
        "f1_score"
    }
}

/// Precision metric (macro average).
#[derive(Debug, Clone)]
pub struct Precision {
    n_classes: usize,
}

impl Precision {
    /// Create a new precision metric.
    pub fn new(n_classes: usize) -> Self {
        Self { n_classes }
    }
}

impl<B: Backend> Metric<B> for Precision {
    fn compute(&self, preds: &Tensor<B, 2>, targets: &Tensor<B, 2>) -> f32 {
        let pred_classes = preds.clone().argmax(1);
        let target_classes = targets.clone().argmax(1);

        let pred_data = pred_classes.into_data();
        let target_data = target_classes.into_data();

        let pred_vec: Vec<i64> = pred_data
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_default();
        let target_vec: Vec<i64> = target_data
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_default();

        let n_samples = pred_vec.len();
        if n_samples == 0 {
            return 0.0;
        }

        let mut total_precision = 0.0;
        let mut valid_classes = 0;

        for class in 0..self.n_classes {
            let class_i64 = class as i64;
            let mut tp = 0;
            let mut fp = 0;

            for i in 0..n_samples {
                if pred_vec[i] == class_i64 {
                    if target_vec[i] == class_i64 {
                        tp += 1;
                    } else {
                        fp += 1;
                    }
                }
            }

            if tp + fp > 0 {
                total_precision += tp as f32 / (tp + fp) as f32;
                valid_classes += 1;
            }
        }

        if valid_classes > 0 {
            total_precision / valid_classes as f32
        } else {
            0.0
        }
    }

    fn name(&self) -> &str {
        "precision"
    }
}

/// Recall metric (macro average).
#[derive(Debug, Clone)]
pub struct Recall {
    n_classes: usize,
}

impl Recall {
    /// Create a new recall metric.
    pub fn new(n_classes: usize) -> Self {
        Self { n_classes }
    }
}

impl<B: Backend> Metric<B> for Recall {
    fn compute(&self, preds: &Tensor<B, 2>, targets: &Tensor<B, 2>) -> f32 {
        let pred_classes = preds.clone().argmax(1);
        let target_classes = targets.clone().argmax(1);

        let pred_data = pred_classes.into_data();
        let target_data = target_classes.into_data();

        let pred_vec: Vec<i64> = pred_data
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_default();
        let target_vec: Vec<i64> = target_data
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_default();

        let n_samples = pred_vec.len();
        if n_samples == 0 {
            return 0.0;
        }

        let mut total_recall = 0.0;
        let mut valid_classes = 0;

        for class in 0..self.n_classes {
            let class_i64 = class as i64;
            let mut tp = 0;
            let mut fn_ = 0;

            for i in 0..n_samples {
                if target_vec[i] == class_i64 {
                    if pred_vec[i] == class_i64 {
                        tp += 1;
                    } else {
                        fn_ += 1;
                    }
                }
            }

            if tp + fn_ > 0 {
                total_recall += tp as f32 / (tp + fn_) as f32;
                valid_classes += 1;
            }
        }

        if valid_classes > 0 {
            total_recall / valid_classes as f32
        } else {
            0.0
        }
    }

    fn name(&self) -> &str {
        "recall"
    }
}

/// Root Mean Squared Error metric.
#[derive(Debug, Clone, Default)]
pub struct RMSE;

impl<B: Backend> Metric<B> for RMSE {
    fn compute(&self, preds: &Tensor<B, 2>, targets: &Tensor<B, 2>) -> f32 {
        let diff = preds.clone() - targets.clone();
        let squared = diff.clone() * diff;
        let mse: f32 = squared.mean().into_scalar().elem();
        mse.sqrt()
    }

    fn name(&self) -> &str {
        "rmse"
    }

    fn higher_is_better(&self) -> bool {
        false
    }
}

/// Area Under ROC Curve (AUC) metric.
///
/// For binary classification, computes the standard AUC.
/// For multi-class classification, computes macro-averaged one-vs-rest AUC.
///
/// Predictions should be probabilities (after softmax).
#[derive(Debug, Clone)]
pub struct AUC {
    n_classes: usize,
}

impl AUC {
    /// Create a new AUC metric.
    ///
    /// # Arguments
    /// * `n_classes` - Number of classes (2 for binary classification)
    pub fn new(n_classes: usize) -> Self {
        Self { n_classes }
    }

    /// Compute binary AUC using the trapezoidal rule.
    ///
    /// This implements the standard AUC computation by sorting predictions
    /// and computing the area under the ROC curve.
    fn compute_binary_auc(probs: &[f32], labels: &[bool]) -> f32 {
        if probs.is_empty() || labels.is_empty() {
            return 0.5;
        }

        let n = probs.len();
        let n_pos: usize = labels.iter().filter(|&&x| x).count();
        let n_neg = n - n_pos;

        if n_pos == 0 || n_neg == 0 {
            // All same class - AUC is undefined, return 0.5
            return 0.5;
        }

        // Sort by prediction score (ascending) - higher rank = higher score
        // This is required for the Wilcoxon-Mann-Whitney formula
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            probs[a]
                .partial_cmp(&probs[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Compute AUC using the Wilcoxon-Mann-Whitney statistic
        // AUC = (sum of ranks of positives - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
        // With ascending sort, rank 1 = lowest score, rank n = highest score
        // Perfect AUC: all positives have highest ranks (n_neg+1 to n)
        let mut rank_sum = 0.0;
        let mut current_rank = 0.0;
        let mut i = 0;

        while i < n {
            // Find tied group
            let current_prob = probs[indices[i]];
            let mut j = i;
            while j < n && (probs[indices[j]] - current_prob).abs() < 1e-10 {
                j += 1;
            }

            // Average rank for tied group (1-based ranks)
            // Ranks in this group: (current_rank + 1) to (current_rank + group_size)
            let group_size = (j - i) as f32;
            let avg_rank = current_rank + (group_size + 1.0) / 2.0;

            // Sum ranks for positive samples in this group
            for k in i..j {
                if labels[indices[k]] {
                    rank_sum += avg_rank;
                }
            }

            current_rank += group_size;
            i = j;
        }

        // Compute AUC
        let auc = (rank_sum - (n_pos as f32 * (n_pos as f32 + 1.0) / 2.0))
            / (n_pos as f32 * n_neg as f32);

        auc.clamp(0.0, 1.0)
    }
}

impl Default for AUC {
    fn default() -> Self {
        Self { n_classes: 2 }
    }
}

impl<B: Backend> Metric<B> for AUC {
    fn compute(&self, preds: &Tensor<B, 2>, targets: &Tensor<B, 2>) -> f32 {
        let [n_samples, n_classes] = preds.dims();

        if n_samples == 0 {
            return 0.5;
        }

        // Get predictions as data for CPU computation
        let pred_data = preds.clone().into_data();
        let target_data = targets.clone().into_data();

        let pred_flat: Vec<f32> = pred_data
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_default();
        let target_flat: Vec<f32> = target_data
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_default();

        if self.n_classes == 2 {
            // Binary classification: use probability of positive class
            let probs: Vec<f32> = (0..n_samples)
                .map(|i| pred_flat[i * n_classes + 1])
                .collect();
            let labels: Vec<bool> = (0..n_samples)
                .map(|i| target_flat[i * n_classes + 1] > 0.5)
                .collect();

            Self::compute_binary_auc(&probs, &labels)
        } else {
            // Multi-class: compute macro-averaged one-vs-rest AUC
            let mut total_auc = 0.0;
            let mut valid_classes = 0;

            for class in 0..self.n_classes {
                // Probability for this class
                let probs: Vec<f32> = (0..n_samples)
                    .map(|i| pred_flat[i * n_classes + class])
                    .collect();

                // Binary label: is this the true class?
                let labels: Vec<bool> = (0..n_samples)
                    .map(|i| target_flat[i * n_classes + class] > 0.5)
                    .collect();

                // Check if this class has both positive and negative samples
                let n_pos = labels.iter().filter(|&&x| x).count();
                if n_pos > 0 && n_pos < n_samples {
                    let class_auc = Self::compute_binary_auc(&probs, &labels);
                    total_auc += class_auc;
                    valid_classes += 1;
                }
            }

            if valid_classes > 0 {
                total_auc / valid_classes as f32
            } else {
                0.5
            }
        }
    }

    fn name(&self) -> &str {
        "auc"
    }
}

/// Matthews Correlation Coefficient (MCC) metric.
///
/// A balanced metric that takes into account all four cells of the confusion matrix.
/// Values range from -1 (total disagreement) to +1 (perfect prediction).
#[derive(Debug, Clone)]
pub struct MCC {
    n_classes: usize,
}

impl MCC {
    /// Create a new MCC metric.
    pub fn new(n_classes: usize) -> Self {
        Self { n_classes }
    }
}

impl<B: Backend> Metric<B> for MCC {
    fn compute(&self, preds: &Tensor<B, 2>, targets: &Tensor<B, 2>) -> f32 {
        let pred_classes = preds.clone().argmax(1);
        let target_classes = targets.clone().argmax(1);

        let pred_data = pred_classes.into_data();
        let target_data = target_classes.into_data();

        let pred_vec: Vec<i64> = pred_data
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_default();
        let target_vec: Vec<i64> = target_data
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_default();

        let n_samples = pred_vec.len();
        if n_samples == 0 {
            return 0.0;
        }

        if self.n_classes == 2 {
            // Binary MCC
            let mut tp = 0;
            let mut tn = 0;
            let mut fp = 0;
            let mut fn_ = 0;

            for i in 0..n_samples {
                let p = pred_vec[i] == 1;
                let t = target_vec[i] == 1;

                if p && t {
                    tp += 1;
                } else if !p && !t {
                    tn += 1;
                } else if p && !t {
                    fp += 1;
                } else {
                    fn_ += 1;
                }
            }

            let numerator = (tp * tn) as f32 - (fp * fn_) as f32;
            let denominator = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)) as f32;

            if denominator > 0.0 {
                numerator / denominator.sqrt()
            } else {
                0.0
            }
        } else {
            // Multi-class MCC using the general formula
            // Build confusion matrix
            let mut conf_matrix = vec![vec![0i64; self.n_classes]; self.n_classes];
            for i in 0..n_samples {
                let p = pred_vec[i] as usize;
                let t = target_vec[i] as usize;
                if p < self.n_classes && t < self.n_classes {
                    conf_matrix[t][p] += 1;
                }
            }

            // Compute components
            let n = n_samples as f64;
            let mut c = 0.0; // sum of diagonal (correct predictions)
            let s = n * n; // total squared

            // Row and column sums
            let mut row_sums = vec![0i64; self.n_classes];
            let mut col_sums = vec![0i64; self.n_classes];

            for k in 0..self.n_classes {
                c += conf_matrix[k][k] as f64;
                for l in 0..self.n_classes {
                    row_sums[k] += conf_matrix[k][l];
                    col_sums[k] += conf_matrix[l][k];
                }
            }

            // Compute sums of products
            let mut pk_tk = 0.0;
            let mut pk_sq = 0.0;
            let mut tk_sq = 0.0;

            for k in 0..self.n_classes {
                pk_tk += (col_sums[k] * row_sums[k]) as f64;
                pk_sq += (col_sums[k] * col_sums[k]) as f64;
                tk_sq += (row_sums[k] * row_sums[k]) as f64;
            }

            let numerator = c * n - pk_tk;
            let denominator = ((s - pk_sq) * (s - tk_sq)).sqrt();

            if denominator > 0.0 {
                (numerator / denominator) as f32
            } else {
                0.0
            }
        }
    }

    fn name(&self) -> &str {
        "mcc"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tsai_core::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_metric_names() {
        assert_eq!(<Accuracy as Metric<TestBackend>>::name(&Accuracy), "accuracy");
        assert_eq!(<MSE as Metric<TestBackend>>::name(&MSE), "mse");
        assert_eq!(<MAE as Metric<TestBackend>>::name(&MAE), "mae");
    }

    #[test]
    fn test_higher_is_better() {
        assert!(<Accuracy as Metric<TestBackend>>::higher_is_better(&Accuracy));
        assert!(!<MSE as Metric<TestBackend>>::higher_is_better(&MSE));
        assert!(!<MAE as Metric<TestBackend>>::higher_is_better(&MAE));
    }

    #[test]
    fn test_auc_metric_names() {
        let auc = AUC::new(2);
        assert_eq!(<AUC as Metric<TestBackend>>::name(&auc), "auc");
        assert!(<AUC as Metric<TestBackend>>::higher_is_better(&auc));
    }

    #[test]
    fn test_mcc_metric_names() {
        let mcc = MCC::new(2);
        assert_eq!(<MCC as Metric<TestBackend>>::name(&mcc), "mcc");
        assert!(<MCC as Metric<TestBackend>>::higher_is_better(&mcc));
    }

    #[test]
    fn test_auc_binary_perfect() {
        // Perfect predictions should give AUC = 1.0
        let probs = vec![0.9, 0.8, 0.7, 0.2, 0.1, 0.0];
        let labels = vec![true, true, true, false, false, false];
        let auc = AUC::compute_binary_auc(&probs, &labels);
        assert!((auc - 1.0).abs() < 1e-5, "Expected AUC=1.0, got {}", auc);
    }

    #[test]
    fn test_auc_binary_random() {
        // All same probabilities should give AUC = 0.5
        let probs = vec![0.5, 0.5, 0.5, 0.5];
        let labels = vec![true, true, false, false];
        let auc = AUC::compute_binary_auc(&probs, &labels);
        assert!((auc - 0.5).abs() < 1e-5, "Expected AUC=0.5, got {}", auc);
    }

    #[test]
    fn test_auc_all_same_class() {
        // All same class should return 0.5
        let probs = vec![0.9, 0.8, 0.7];
        let labels = vec![true, true, true];
        let auc = AUC::compute_binary_auc(&probs, &labels);
        assert!((auc - 0.5).abs() < 1e-5, "Expected AUC=0.5 for single class, got {}", auc);
    }

    #[test]
    fn test_auc_empty() {
        let probs: Vec<f32> = vec![];
        let labels: Vec<bool> = vec![];
        let auc = AUC::compute_binary_auc(&probs, &labels);
        assert!((auc - 0.5).abs() < 1e-5);
    }
}
