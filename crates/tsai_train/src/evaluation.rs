//! Model evaluation utilities.
//!
//! Provides functions to evaluate trained models and compute metrics.

use burn::module::AutodiffModule;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use tsai_data::TSDataLoaders;

/// Simple confusion matrix for evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    /// The matrix values (row = true, col = pred).
    pub matrix: Vec<Vec<usize>>,
    /// Number of classes.
    pub n_classes: usize,
}

impl ConfusionMatrix {
    /// Create a new confusion matrix from predictions and targets.
    pub fn from_predictions(predictions: &[usize], targets: &[usize], n_classes: usize) -> Self {
        let mut matrix = vec![vec![0; n_classes]; n_classes];
        for (&pred, &target) in predictions.iter().zip(targets) {
            if target < n_classes && pred < n_classes {
                matrix[target][pred] += 1;
            }
        }
        Self { matrix, n_classes }
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

    /// Get a formatted string representation.
    pub fn to_table(&self) -> String {
        let mut s = String::new();

        // Header
        s.push_str("         ");
        for j in 0..self.n_classes {
            s.push_str(&format!("{:>7}", format!("P{}", j)));
        }
        s.push('\n');

        // Rows
        for i in 0..self.n_classes {
            s.push_str(&format!("   T{:<4} ", i));
            for j in 0..self.n_classes {
                s.push_str(&format!("{:>7}", self.matrix[i][j]));
            }
            s.push('\n');
        }

        s
    }
}

/// Evaluation results with predictions and metrics.
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// Predicted class indices.
    pub predictions: Vec<usize>,
    /// True class indices.
    pub targets: Vec<usize>,
    /// Prediction probabilities (n_samples, n_classes).
    pub probabilities: Vec<Vec<f32>>,
    /// Accuracy.
    pub accuracy: f32,
    /// Number of correct predictions.
    pub correct: usize,
    /// Total number of samples.
    pub total: usize,
}

impl EvaluationResult {
    /// Compute confusion matrix from results.
    pub fn confusion_matrix(&self, n_classes: usize) -> ConfusionMatrix {
        ConfusionMatrix::from_predictions(&self.predictions, &self.targets, n_classes)
    }

    /// Print a summary of the evaluation.
    pub fn print_summary(&self) {
        println!("Evaluation Results:");
        println!("  Accuracy: {:.2}%", self.accuracy * 100.0);
        println!("  Correct: {} / {}", self.correct, self.total);
    }

    /// Print confusion matrix and per-class metrics.
    pub fn print_confusion_matrix(&self, n_classes: usize) {
        let cm = self.confusion_matrix(n_classes);
        println!("\nConfusion Matrix (T=True, P=Predicted):");
        println!("{}", cm.to_table());
        println!("Per-class metrics:");
        println!("{:<10} {:>10} {:>10} {:>10}", "Class", "Precision", "Recall", "F1");
        println!("{}", "-".repeat(42));
        for i in 0..n_classes {
            println!(
                "{:<10} {:>9.2}% {:>9.2}% {:>9.2}%",
                i,
                cm.precision(i) * 100.0,
                cm.recall(i) * 100.0,
                cm.f1(i) * 100.0
            );
        }
        println!("{}", "-".repeat(42));
        println!(
            "{:<10} {:>10} {:>10} {:>9.2}%",
            "Macro F1",
            "",
            "",
            cm.macro_f1() * 100.0
        );
    }
}

/// Evaluate a classification model on a dataset.
///
/// # Arguments
///
/// * `model` - The trained model
/// * `dls` - DataLoaders containing the validation/test set
/// * `forward_fn` - Forward function for the model
///
/// # Returns
///
/// Evaluation results with predictions and metrics.
pub fn evaluate_classification<B, M, G>(
    model: &M,
    dls: &TSDataLoaders,
    forward_fn: G,
) -> Result<EvaluationResult>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    G: Fn(&M::InnerModule, Tensor<B::InnerBackend, 3>) -> Tensor<B::InnerBackend, 2>,
{
    let inner_model = model.clone().valid();
    let inner_device: <B::InnerBackend as Backend>::Device = Default::default();

    let mut all_predictions = Vec::new();
    let mut all_targets = Vec::new();
    let mut all_probabilities = Vec::new();

    for batch_result in dls.valid().iter::<B::InnerBackend>(&inner_device) {
        let batch = batch_result?;

        // Get input tensor
        let x = batch.x.inner().clone();

        // Get targets
        let y = batch.y.expect("Evaluation requires targets");
        let [batch_size, _] = y.dims();

        // Forward pass
        let logits = forward_fn(&inner_model, x);

        // Get predictions and probabilities
        let probs = burn::tensor::activation::softmax(logits.clone(), 1);
        let preds = logits.argmax(1).squeeze::<1>(1);

        // Convert to CPU values
        // Note: argmax returns different int types on different backends (I32 on NdArray, I64 on WGPU)
        // We convert to i32 first, falling back to i64 if that fails
        let preds_data: Vec<usize> = {
            let data = preds.into_data();
            if let Ok(vec) = data.clone().to_vec::<i32>() {
                vec.into_iter().map(|x| x as usize).collect()
            } else if let Ok(vec) = data.to_vec::<i64>() {
                vec.into_iter().map(|x| x as usize).collect()
            } else {
                panic!("Unsupported prediction data type");
            }
        };
        let targets_data: Vec<f32> = y.reshape([batch_size]).into_data().to_vec().unwrap();
        let probs_data: Vec<f32> = probs.into_data().to_vec().unwrap();

        let n_classes = probs_data.len() / batch_size;

        for i in 0..batch_size {
            all_predictions.push(preds_data[i]);
            all_targets.push(targets_data[i] as usize);

            let prob_start = i * n_classes;
            let prob_end = prob_start + n_classes;
            all_probabilities.push(probs_data[prob_start..prob_end].to_vec());
        }
    }

    // Compute accuracy
    let correct = all_predictions
        .iter()
        .zip(&all_targets)
        .filter(|(p, t)| *p == *t)
        .count();
    let total = all_predictions.len();
    let accuracy = if total > 0 {
        correct as f32 / total as f32
    } else {
        0.0
    };

    Ok(EvaluationResult {
        predictions: all_predictions,
        targets: all_targets,
        probabilities: all_probabilities,
        accuracy,
        correct,
        total,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confusion_matrix() {
        let predictions = vec![0, 1, 0, 1];
        let targets = vec![0, 1, 1, 1];
        let cm = ConfusionMatrix::from_predictions(&predictions, &targets, 2);

        assert_eq!(cm.matrix[0][0], 1); // TP class 0
        assert_eq!(cm.matrix[1][1], 2); // TP class 1
        assert_eq!(cm.matrix[1][0], 1); // FN class 1 (pred 0)
        assert!((cm.accuracy() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_evaluation_result() {
        let result = EvaluationResult {
            predictions: vec![0, 1, 0, 1],
            targets: vec![0, 1, 1, 1],
            probabilities: vec![
                vec![0.8, 0.2],
                vec![0.3, 0.7],
                vec![0.6, 0.4],
                vec![0.2, 0.8],
            ],
            accuracy: 0.75,
            correct: 3,
            total: 4,
        };

        let cm = result.confusion_matrix(2);
        assert_eq!(cm.accuracy(), 0.75);
    }
}
