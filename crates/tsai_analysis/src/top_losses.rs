//! Top losses analysis.

use serde::{Deserialize, Serialize};

/// A sample with its loss value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopLoss {
    /// Sample index.
    pub index: usize,
    /// Loss value.
    pub loss: f32,
    /// True label.
    pub target: usize,
    /// Predicted label.
    pub pred: usize,
    /// Prediction probability.
    pub prob: f32,
}

/// Get the top losses (samples with highest loss).
///
/// # Arguments
///
/// * `losses` - Per-sample losses
/// * `targets` - True class indices
/// * `preds` - Predicted class indices
/// * `probs` - Prediction probabilities for the predicted class
/// * `k` - Number of top losses to return
///
/// # Returns
///
/// Vector of top losses, sorted by loss descending.
pub fn top_losses(
    losses: &[f32],
    targets: &[usize],
    preds: &[usize],
    probs: &[f32],
    k: usize,
) -> Vec<TopLoss> {
    let mut indexed: Vec<_> = losses
        .iter()
        .enumerate()
        .map(|(i, &loss)| TopLoss {
            index: i,
            loss,
            target: targets.get(i).copied().unwrap_or(0),
            pred: preds.get(i).copied().unwrap_or(0),
            prob: probs.get(i).copied().unwrap_or(0.0),
        })
        .collect();

    // Sort by loss descending
    indexed.sort_by(|a, b| b.loss.partial_cmp(&a.loss).unwrap_or(std::cmp::Ordering::Equal));

    indexed.truncate(k);
    indexed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_losses() {
        let losses = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let targets = vec![0, 1, 0, 1, 0];
        let preds = vec![0, 0, 0, 0, 0];
        let probs = vec![0.9, 0.6, 0.8, 0.55, 0.85];

        let top = top_losses(&losses, &targets, &preds, &probs, 3);

        assert_eq!(top.len(), 3);
        assert_eq!(top[0].index, 3); // Highest loss
        assert_eq!(top[0].loss, 0.9);
        assert_eq!(top[1].index, 1);
        assert_eq!(top[2].index, 2);
    }
}
