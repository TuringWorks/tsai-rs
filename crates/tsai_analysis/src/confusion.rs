//! Confusion matrix computation and visualization.


use serde::{Deserialize, Serialize};

/// Confusion matrix for classification evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    /// The matrix values (row = true, col = pred).
    pub matrix: Vec<Vec<usize>>,
    /// Number of classes.
    pub n_classes: usize,
    /// Class labels.
    pub labels: Option<Vec<String>>,
}

impl ConfusionMatrix {
    /// Create a new confusion matrix.
    pub fn new(n_classes: usize) -> Self {
        Self {
            matrix: vec![vec![0; n_classes]; n_classes],
            n_classes,
            labels: None,
        }
    }

    /// Set class labels.
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.labels = Some(labels);
        self
    }

    /// Add a prediction.
    pub fn add(&mut self, true_class: usize, pred_class: usize) {
        if true_class < self.n_classes && pred_class < self.n_classes {
            self.matrix[true_class][pred_class] += 1;
        }
    }

    /// Get accuracy.
    pub fn accuracy(&self) -> f32 {
        let correct: usize = (0..self.n_classes).map(|i| self.matrix[i][i]).sum();
        let total: usize = self.matrix.iter().flatten().sum();
        if total == 0 {
            0.0
        } else {
            correct as f32 / total as f32
        }
    }

    /// Get precision for a class.
    pub fn precision(&self, class: usize) -> f32 {
        let tp = self.matrix[class][class];
        let fp: usize = (0..self.n_classes)
            .filter(|&i| i != class)
            .map(|i| self.matrix[i][class])
            .sum();
        if tp + fp == 0 {
            0.0
        } else {
            tp as f32 / (tp + fp) as f32
        }
    }

    /// Get recall for a class.
    pub fn recall(&self, class: usize) -> f32 {
        let tp = self.matrix[class][class];
        let fn_val: usize = (0..self.n_classes)
            .filter(|&i| i != class)
            .map(|i| self.matrix[class][i])
            .sum();
        if tp + fn_val == 0 {
            0.0
        } else {
            tp as f32 / (tp + fn_val) as f32
        }
    }

    /// Get F1 score for a class.
    pub fn f1(&self, class: usize) -> f32 {
        let p = self.precision(class);
        let r = self.recall(class);
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }

    /// Get macro-averaged F1 score.
    pub fn macro_f1(&self) -> f32 {
        let sum: f32 = (0..self.n_classes).map(|i| self.f1(i)).sum();
        sum / self.n_classes as f32
    }

    /// Normalize the matrix (row-wise, shows recall).
    pub fn normalize(&self) -> Vec<Vec<f32>> {
        self.matrix
            .iter()
            .map(|row| {
                let sum: usize = row.iter().sum();
                if sum == 0 {
                    vec![0.0; self.n_classes]
                } else {
                    row.iter().map(|&v| v as f32 / sum as f32).collect()
                }
            })
            .collect()
    }

    /// Get a text representation.
    pub fn to_string_table(&self) -> String {
        let mut s = String::new();
        let labels = self.labels.as_ref();

        // Header
        s.push_str("       ");
        for j in 0..self.n_classes {
            let label = labels.map(|l| l[j].as_str()).unwrap_or("");
            s.push_str(&format!("{:>8}", label.chars().take(7).collect::<String>()));
        }
        s.push('\n');

        // Rows
        for i in 0..self.n_classes {
            let label = labels.map(|l| l[i].as_str()).unwrap_or("");
            s.push_str(&format!("{:>6} ", label.chars().take(6).collect::<String>()));
            for j in 0..self.n_classes {
                s.push_str(&format!("{:>8}", self.matrix[i][j]));
            }
            s.push('\n');
        }

        s
    }
}

/// Compute confusion matrix from predictions and targets.
///
/// # Arguments
///
/// * `preds` - Predicted class indices
/// * `targets` - True class indices
/// * `n_classes` - Number of classes
///
/// # Returns
///
/// A confusion matrix.
pub fn confusion_matrix(preds: &[usize], targets: &[usize], n_classes: usize) -> ConfusionMatrix {
    let mut cm = ConfusionMatrix::new(n_classes);
    for (&pred, &target) in preds.iter().zip(targets) {
        cm.add(target, pred);
    }
    cm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confusion_matrix() {
        let preds = vec![0, 0, 1, 1, 2, 2];
        let targets = vec![0, 1, 1, 1, 2, 0];

        let cm = confusion_matrix(&preds, &targets, 3);

        assert_eq!(cm.matrix[0][0], 1); // TP for class 0
        assert_eq!(cm.matrix[1][0], 1); // FN for class 0 (was 1, pred 0)
        assert_eq!(cm.matrix[1][1], 2); // TP for class 1
    }

    #[test]
    fn test_accuracy() {
        let preds = vec![0, 1, 2];
        let targets = vec![0, 1, 2];
        let cm = confusion_matrix(&preds, &targets, 3);
        assert!((cm.accuracy() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_metrics() {
        let preds = vec![0, 0, 1, 1];
        let targets = vec![0, 1, 0, 1];
        let cm = confusion_matrix(&preds, &targets, 2);

        assert!((cm.precision(0) - 0.5).abs() < 1e-6);
        assert!((cm.recall(0) - 0.5).abs() < 1e-6);
    }
}
