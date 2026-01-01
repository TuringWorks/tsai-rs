//! Classification report with per-class metrics.
//!
//! Provides detailed classification metrics including precision, recall,
//! and F1-score for each class, along with macro and weighted averages.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Per-class classification metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassMetrics {
    /// Class label (index).
    pub class: i64,
    /// Class name (if provided).
    pub name: Option<String>,
    /// Precision: TP / (TP + FP)
    pub precision: f32,
    /// Recall: TP / (TP + FN)
    pub recall: f32,
    /// F1-Score: 2 * (precision * recall) / (precision + recall)
    pub f1_score: f32,
    /// Support: number of true instances of this class
    pub support: usize,
}

/// Classification report with per-class and aggregate metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationReport {
    /// Per-class metrics.
    pub classes: Vec<ClassMetrics>,
    /// Overall accuracy.
    pub accuracy: f32,
    /// Macro-averaged precision (unweighted mean of per-class precision).
    pub macro_precision: f32,
    /// Macro-averaged recall.
    pub macro_recall: f32,
    /// Macro-averaged F1.
    pub macro_f1: f32,
    /// Weighted-averaged precision (weighted by support).
    pub weighted_precision: f32,
    /// Weighted-averaged recall.
    pub weighted_recall: f32,
    /// Weighted-averaged F1.
    pub weighted_f1: f32,
    /// Total number of samples.
    pub total_samples: usize,
}

impl ClassificationReport {
    /// Display the report as a formatted string.
    pub fn to_string_table(&self) -> String {
        let mut output = String::new();

        output.push_str("              precision    recall  f1-score   support\n\n");

        for class in &self.classes {
            let name = class
                .name
                .clone()
                .unwrap_or_else(|| format!("Class {}", class.class));
            output.push_str(&format!(
                "{:>12}      {:.2}      {:.2}      {:.2}     {:5}\n",
                name, class.precision, class.recall, class.f1_score, class.support
            ));
        }

        output.push_str("\n");
        output.push_str(&format!(
            "{:>12}      {:.2}      {:.2}      {:.2}     {:5}\n",
            "accuracy", "", "", self.accuracy, self.total_samples
        ));
        output.push_str(&format!(
            "{:>12}      {:.2}      {:.2}      {:.2}     {:5}\n",
            "macro avg", self.macro_precision, self.macro_recall, self.macro_f1, self.total_samples
        ));
        output.push_str(&format!(
            "{:>12}      {:.2}      {:.2}      {:.2}     {:5}\n",
            "weighted avg",
            self.weighted_precision,
            self.weighted_recall,
            self.weighted_f1,
            self.total_samples
        ));

        output
    }

    /// Get the class with lowest F1-score (worst performing).
    pub fn worst_class(&self) -> Option<&ClassMetrics> {
        self.classes
            .iter()
            .filter(|c| c.support > 0)
            .min_by(|a, b| a.f1_score.partial_cmp(&b.f1_score).unwrap())
    }

    /// Get the class with highest F1-score (best performing).
    pub fn best_class(&self) -> Option<&ClassMetrics> {
        self.classes
            .iter()
            .filter(|c| c.support > 0)
            .max_by(|a, b| a.f1_score.partial_cmp(&b.f1_score).unwrap())
    }

    /// Get classes with F1-score below threshold.
    pub fn low_performing_classes(&self, threshold: f32) -> Vec<&ClassMetrics> {
        self.classes
            .iter()
            .filter(|c| c.support > 0 && c.f1_score < threshold)
            .collect()
    }
}

/// Compute a classification report from predictions and targets.
///
/// # Arguments
///
/// * `predictions` - Predicted class labels
/// * `targets` - True class labels
/// * `class_names` - Optional names for each class
///
/// # Returns
///
/// ClassificationReport with per-class and aggregate metrics.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_analysis::report::classification_report;
///
/// let predictions = vec![0, 1, 2, 0, 1, 2];
/// let targets = vec![0, 1, 1, 0, 2, 2];
///
/// let report = classification_report(&predictions, &targets, None);
/// println!("{}", report.to_string_table());
/// ```
pub fn classification_report(
    predictions: &[i64],
    targets: &[i64],
    class_names: Option<&[String]>,
) -> ClassificationReport {
    let n = predictions.len();
    assert_eq!(targets.len(), n, "predictions and targets must have same length");

    // Find all classes
    let mut all_classes: Vec<i64> = predictions.iter().chain(targets.iter()).copied().collect();
    all_classes.sort_unstable();
    all_classes.dedup();

    // Count TP, FP, FN for each class
    let mut tp: HashMap<i64, usize> = HashMap::new();
    let mut fp: HashMap<i64, usize> = HashMap::new();
    let mut fn_: HashMap<i64, usize> = HashMap::new();
    let mut support: HashMap<i64, usize> = HashMap::new();

    for class in &all_classes {
        tp.insert(*class, 0);
        fp.insert(*class, 0);
        fn_.insert(*class, 0);
        support.insert(*class, 0);
    }

    let mut correct = 0;
    for i in 0..n {
        let pred = predictions[i];
        let target = targets[i];

        *support.get_mut(&target).unwrap() += 1;

        if pred == target {
            *tp.get_mut(&pred).unwrap() += 1;
            correct += 1;
        } else {
            *fp.get_mut(&pred).unwrap() += 1;
            *fn_.get_mut(&target).unwrap() += 1;
        }
    }

    // Compute per-class metrics
    let mut classes = Vec::new();
    for (idx, class) in all_classes.iter().enumerate() {
        let tp_c = tp[class] as f32;
        let fp_c = fp[class] as f32;
        let fn_c = fn_[class] as f32;
        let sup = support[class];

        let precision = if tp_c + fp_c > 0.0 {
            tp_c / (tp_c + fp_c)
        } else {
            0.0
        };

        let recall = if tp_c + fn_c > 0.0 {
            tp_c / (tp_c + fn_c)
        } else {
            0.0
        };

        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        let name = class_names
            .and_then(|names| names.get(idx))
            .cloned();

        classes.push(ClassMetrics {
            class: *class,
            name,
            precision,
            recall,
            f1_score: f1,
            support: sup,
        });
    }

    // Compute macro averages
    let n_classes = classes.iter().filter(|c| c.support > 0).count() as f32;
    let macro_precision = if n_classes > 0.0 {
        classes.iter().filter(|c| c.support > 0).map(|c| c.precision).sum::<f32>() / n_classes
    } else {
        0.0
    };
    let macro_recall = if n_classes > 0.0 {
        classes.iter().filter(|c| c.support > 0).map(|c| c.recall).sum::<f32>() / n_classes
    } else {
        0.0
    };
    let macro_f1 = if n_classes > 0.0 {
        classes.iter().filter(|c| c.support > 0).map(|c| c.f1_score).sum::<f32>() / n_classes
    } else {
        0.0
    };

    // Compute weighted averages
    let total_support: usize = classes.iter().map(|c| c.support).sum();
    let weighted_precision = if total_support > 0 {
        classes
            .iter()
            .map(|c| c.precision * c.support as f32)
            .sum::<f32>()
            / total_support as f32
    } else {
        0.0
    };
    let weighted_recall = if total_support > 0 {
        classes
            .iter()
            .map(|c| c.recall * c.support as f32)
            .sum::<f32>()
            / total_support as f32
    } else {
        0.0
    };
    let weighted_f1 = if total_support > 0 {
        classes
            .iter()
            .map(|c| c.f1_score * c.support as f32)
            .sum::<f32>()
            / total_support as f32
    } else {
        0.0
    };

    let accuracy = if n > 0 { correct as f32 / n as f32 } else { 0.0 };

    ClassificationReport {
        classes,
        accuracy,
        macro_precision,
        macro_recall,
        macro_f1,
        weighted_precision,
        weighted_recall,
        weighted_f1,
        total_samples: n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classification_report_perfect() {
        let predictions = vec![0, 1, 2, 0, 1, 2];
        let targets = vec![0, 1, 2, 0, 1, 2];

        let report = classification_report(&predictions, &targets, None);

        assert!((report.accuracy - 1.0).abs() < 1e-6);
        assert!((report.macro_f1 - 1.0).abs() < 1e-6);
        assert_eq!(report.total_samples, 6);
        assert_eq!(report.classes.len(), 3);
    }

    #[test]
    fn test_classification_report_binary() {
        let predictions = vec![0, 0, 1, 1, 1, 0];
        let targets = vec![0, 1, 1, 1, 0, 0];

        let report = classification_report(&predictions, &targets, None);

        // 4/6 correct
        assert!((report.accuracy - 4.0 / 6.0).abs() < 1e-6);
        assert_eq!(report.classes.len(), 2);
    }

    #[test]
    fn test_classification_report_with_names() {
        let predictions = vec![0, 1, 0];
        let targets = vec![0, 1, 1];
        let names = vec!["Cat".to_string(), "Dog".to_string()];

        let report = classification_report(&predictions, &targets, Some(&names));

        assert_eq!(report.classes[0].name, Some("Cat".to_string()));
        assert_eq!(report.classes[1].name, Some("Dog".to_string()));
    }

    #[test]
    fn test_classification_report_display() {
        let predictions = vec![0, 1, 2];
        let targets = vec![0, 1, 2];

        let report = classification_report(&predictions, &targets, None);
        let table = report.to_string_table();

        assert!(table.contains("precision"));
        assert!(table.contains("recall"));
        assert!(table.contains("f1-score"));
        assert!(table.contains("macro avg"));
        assert!(table.contains("weighted avg"));
    }

    #[test]
    fn test_worst_best_class() {
        let predictions = vec![0, 0, 0, 1, 1, 2];
        let targets = vec![0, 0, 0, 1, 0, 2]; // Class 1 has issues

        let report = classification_report(&predictions, &targets, None);

        let worst = report.worst_class().unwrap();
        let best = report.best_class().unwrap();

        assert!(best.f1_score >= worst.f1_score);
    }

    #[test]
    fn test_low_performing_classes() {
        let predictions = vec![0, 0, 1, 1, 2, 2];
        let targets = vec![0, 0, 1, 0, 2, 2];

        let report = classification_report(&predictions, &targets, None);
        let low = report.low_performing_classes(0.9);

        // Some classes should be below 0.9 F1
        assert!(!low.is_empty() || report.macro_f1 >= 0.9);
    }
}
