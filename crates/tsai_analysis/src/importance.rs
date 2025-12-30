//! Permutation importance for features and time steps.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Importance score for a feature or time step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermutationImportance {
    /// Feature/step index.
    pub index: usize,
    /// Name (if available).
    pub name: Option<String>,
    /// Mean importance score.
    pub importance: f32,
    /// Standard deviation of importance.
    pub std: f32,
}

/// Compute feature (variable) importance via permutation.
///
/// For each variable, permute its values across samples and measure
/// the decrease in model performance.
///
/// # Arguments
///
/// * `x` - Input data of shape (n_samples, n_vars, seq_len)
/// * `y` - True labels
/// * `predict_fn` - Function that takes x and returns predictions
/// * `metric_fn` - Function that computes metric from (preds, targets)
/// * `n_repeats` - Number of permutation repeats
///
/// # Returns
///
/// Vector of importance scores, one per variable.
pub fn feature_importance<F, M>(
    x: &ndarray::Array3<f32>,
    y: &[usize],
    predict_fn: F,
    metric_fn: M,
    n_repeats: usize,
) -> Vec<PermutationImportance>
where
    F: Fn(&ndarray::Array3<f32>) -> Vec<usize> + Sync,
    M: Fn(&[usize], &[usize]) -> f32 + Sync,
{
    let n_vars = x.shape()[1];

    // Baseline score
    let baseline_preds = predict_fn(x);
    let baseline_score = metric_fn(&baseline_preds, y);

    (0..n_vars)
        .into_par_iter()
        .map(|var_idx| {
            let mut scores = Vec::with_capacity(n_repeats);

            for _ in 0..n_repeats {
                // Permute the variable
                let mut x_perm = x.clone();
                let n_samples = x.shape()[0];

                // Create permutation indices
                use rand::prelude::*;
                let mut rng = rand::thread_rng();
                let mut indices: Vec<usize> = (0..n_samples).collect();
                indices.shuffle(&mut rng);

                // Apply permutation to this variable
                for (i, &perm_i) in indices.iter().enumerate() {
                    for t in 0..x.shape()[2] {
                        x_perm[[i, var_idx, t]] = x[[perm_i, var_idx, t]];
                    }
                }

                let perm_preds = predict_fn(&x_perm);
                let perm_score = metric_fn(&perm_preds, y);
                scores.push(baseline_score - perm_score);
            }

            let mean = scores.iter().sum::<f32>() / scores.len() as f32;
            let variance =
                scores.iter().map(|&s| (s - mean).powi(2)).sum::<f32>() / scores.len() as f32;
            let std = variance.sqrt();

            PermutationImportance {
                index: var_idx,
                name: None,
                importance: mean,
                std,
            }
        })
        .collect()
}

/// Compute time step importance via permutation.
///
/// For each time step, permute its values and measure performance decrease.
pub fn step_importance<F, M>(
    x: &ndarray::Array3<f32>,
    y: &[usize],
    predict_fn: F,
    metric_fn: M,
    n_repeats: usize,
) -> Vec<PermutationImportance>
where
    F: Fn(&ndarray::Array3<f32>) -> Vec<usize> + Sync,
    M: Fn(&[usize], &[usize]) -> f32 + Sync,
{
    let seq_len = x.shape()[2];

    // Baseline score
    let baseline_preds = predict_fn(x);
    let baseline_score = metric_fn(&baseline_preds, y);

    (0..seq_len)
        .into_par_iter()
        .map(|step_idx| {
            let mut scores = Vec::with_capacity(n_repeats);

            for _ in 0..n_repeats {
                let mut x_perm = x.clone();
                let n_samples = x.shape()[0];

                use rand::prelude::*;
                let mut rng = rand::thread_rng();
                let mut indices: Vec<usize> = (0..n_samples).collect();
                indices.shuffle(&mut rng);

                // Permute this time step across all variables
                for (i, &perm_i) in indices.iter().enumerate() {
                    for v in 0..x.shape()[1] {
                        x_perm[[i, v, step_idx]] = x[[perm_i, v, step_idx]];
                    }
                }

                let perm_preds = predict_fn(&x_perm);
                let perm_score = metric_fn(&perm_preds, y);
                scores.push(baseline_score - perm_score);
            }

            let mean = scores.iter().sum::<f32>() / scores.len() as f32;
            let variance =
                scores.iter().map(|&s| (s - mean).powi(2)).sum::<f32>() / scores.len() as f32;
            let std = variance.sqrt();

            PermutationImportance {
                index: step_idx,
                name: None,
                importance: mean,
                std,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permutation_importance_struct() {
        let pi = PermutationImportance {
            index: 0,
            name: Some("feature_0".to_string()),
            importance: 0.05,
            std: 0.01,
        };
        assert_eq!(pi.index, 0);
        assert!((pi.importance - 0.05).abs() < 1e-6);
    }
}
