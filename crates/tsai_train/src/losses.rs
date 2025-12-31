//! Loss functions.
//!
//! Provides loss functions for classification and regression tasks.

use burn::nn::loss::{CrossEntropyLossConfig, MseLoss};
use burn::prelude::*;
use burn::tensor::activation::{log_softmax, softmax};

/// Cross-entropy loss for classification.
#[derive(Debug, Default)]
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    /// Create a new cross-entropy loss.
    pub fn new() -> Self {
        Self
    }

    /// Compute the loss.
    pub fn forward<B: Backend>(
        &self,
        logits: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
    ) -> Tensor<B, 1> {
        let loss = CrossEntropyLossConfig::new().init(&logits.device());
        loss.forward(logits, targets)
    }
}

/// Mean Squared Error loss for regression.
#[derive(Debug, Default)]
pub struct MSELoss;

impl MSELoss {
    /// Create a new MSE loss.
    pub fn new() -> Self {
        Self
    }

    /// Compute the loss.
    pub fn forward<B: Backend>(&self, preds: Tensor<B, 2>, targets: Tensor<B, 2>) -> Tensor<B, 1> {
        let loss = MseLoss::new();
        loss.forward(preds, targets, burn::nn::loss::Reduction::Mean)
    }
}

/// Huber loss (smooth L1).
///
/// Combines MSE for small errors with L1 for large errors, making it
/// robust to outliers while maintaining smooth gradients near zero.
///
/// L = 0.5 * (y - pred)^2           if |y - pred| <= delta
/// L = delta * |y - pred| - 0.5 * delta^2   otherwise
#[derive(Debug)]
pub struct HuberLoss {
    /// Threshold between L2 and L1 behavior.
    pub delta: f32,
}

impl HuberLoss {
    /// Create a new Huber loss.
    pub fn new(delta: f32) -> Self {
        Self { delta }
    }

    /// Compute the loss.
    pub fn forward<B: Backend>(&self, preds: Tensor<B, 2>, targets: Tensor<B, 2>) -> Tensor<B, 1> {
        let diff = preds - targets;
        let abs_diff = diff.clone().abs();
        let device = abs_diff.device();

        // Get tensor data for conditional computation
        let abs_data: Vec<f32> = abs_diff.clone().into_data().to_vec().unwrap();
        let diff_data: Vec<f32> = diff.into_data().to_vec().unwrap();

        let delta = self.delta;
        let half_delta_sq = 0.5 * delta * delta;

        // Compute Huber loss element-wise
        let huber_values: Vec<f32> = abs_data
            .iter()
            .zip(&diff_data)
            .map(|(&abs_val, &diff_val)| {
                if abs_val <= delta {
                    // Quadratic region
                    0.5 * diff_val * diff_val
                } else {
                    // Linear region
                    delta * abs_val - half_delta_sq
                }
            })
            .collect();

        // Compute mean
        let mean: f32 = huber_values.iter().sum::<f32>() / huber_values.len() as f32;
        Tensor::<B, 1>::from_floats([mean], &device)
    }
}

impl Default for HuberLoss {
    fn default() -> Self {
        Self::new(1.0)
    }
}

/// Focal Loss for handling class imbalance.
///
/// Down-weights easy examples and focuses training on hard examples.
/// Particularly useful for imbalanced datasets.
///
/// FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
///
/// Reference: "Focal Loss for Dense Object Detection" by Lin et al. (2017)
#[derive(Debug)]
pub struct FocalLoss {
    /// Focusing parameter. Higher values increase focus on hard examples.
    pub gamma: f32,
    /// Class weights for handling imbalance. None means equal weights.
    pub alpha: Option<Vec<f32>>,
    /// Small epsilon for numerical stability.
    epsilon: f32,
}

impl FocalLoss {
    /// Create a new Focal Loss.
    ///
    /// # Arguments
    ///
    /// * `gamma` - Focusing parameter (typically 2.0). Higher values focus more on hard examples.
    pub fn new(gamma: f32) -> Self {
        Self {
            gamma,
            alpha: None,
            epsilon: 1e-8,
        }
    }

    /// Set class weights.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Per-class weights. For binary classification with imbalance,
    ///   use higher weight for minority class.
    #[must_use]
    pub fn with_alpha(mut self, alpha: Vec<f32>) -> Self {
        self.alpha = Some(alpha);
        self
    }

    /// Compute focal loss.
    ///
    /// # Arguments
    ///
    /// * `logits` - Raw model outputs of shape (batch, n_classes)
    /// * `targets` - Integer class labels of shape (batch,)
    ///
    /// # Returns
    ///
    /// Scalar loss tensor of shape (1,)
    pub fn forward<B: Backend>(
        &self,
        logits: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
    ) -> Tensor<B, 1> {
        let [batch_size, n_classes] = logits.dims();
        let device = logits.device();

        // Convert logits to probabilities
        let probs = softmax(logits.clone(), 1);
        let log_probs = log_softmax(logits, 1);

        // Get probability data for gathering
        let probs_data: Vec<f32> = probs.into_data().to_vec().unwrap();
        let log_probs_data: Vec<f32> = log_probs.into_data().to_vec().unwrap();
        let targets_data: Vec<i32> = targets.into_data().to_vec().unwrap();

        // Compute focal loss for each sample
        let mut focal_losses: Vec<f32> = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let target_class = targets_data[i] as usize;
            let p_t = probs_data[i * n_classes + target_class].max(self.epsilon);
            let log_p_t = log_probs_data[i * n_classes + target_class];

            // Focal weight: (1 - p_t)^gamma
            let focal_weight = (1.0 - p_t).powf(self.gamma);

            // Alpha weight if provided
            let alpha_weight = self
                .alpha
                .as_ref()
                .map(|a| a.get(target_class).copied().unwrap_or(1.0))
                .unwrap_or(1.0);

            // FL = -alpha * (1 - p_t)^gamma * log(p_t)
            let loss = -alpha_weight * focal_weight * log_p_t;
            focal_losses.push(loss);
        }

        // Return mean loss
        let mean_loss: f32 = focal_losses.iter().sum::<f32>() / batch_size as f32;
        Tensor::<B, 1>::from_floats([mean_loss], &device)
    }
}

impl Default for FocalLoss {
    fn default() -> Self {
        Self::new(2.0)
    }
}

/// Label Smoothing Cross Entropy Loss.
///
/// Prevents overconfidence by smoothing the target distribution.
/// Instead of one-hot targets, uses soft targets with small probability
/// on non-target classes.
#[derive(Debug)]
pub struct LabelSmoothingLoss {
    /// Smoothing factor (0 = no smoothing, 1 = uniform distribution).
    pub smoothing: f32,
}

impl LabelSmoothingLoss {
    /// Create a new Label Smoothing Loss.
    ///
    /// # Arguments
    ///
    /// * `smoothing` - Smoothing factor, typically 0.1
    pub fn new(smoothing: f32) -> Self {
        Self { smoothing }
    }

    /// Compute label smoothing loss.
    ///
    /// # Arguments
    ///
    /// * `logits` - Raw model outputs of shape (batch, n_classes)
    /// * `targets` - Integer class labels of shape (batch,)
    ///
    /// # Returns
    ///
    /// Scalar loss tensor of shape (1,)
    pub fn forward<B: Backend>(
        &self,
        logits: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
    ) -> Tensor<B, 1> {
        let [batch_size, n_classes] = logits.dims();
        let device = logits.device();

        // Log softmax for numerical stability
        let log_probs = log_softmax(logits, 1);
        let log_probs_data: Vec<f32> = log_probs.into_data().to_vec().unwrap();
        let targets_data: Vec<i32> = targets.into_data().to_vec().unwrap();

        // Smooth target distribution
        let smooth_positive = 1.0 - self.smoothing;
        let smooth_negative = self.smoothing / (n_classes - 1) as f32;

        let mut total_loss = 0.0f32;

        for i in 0..batch_size {
            let target_class = targets_data[i] as usize;
            let mut sample_loss = 0.0f32;

            for c in 0..n_classes {
                let log_p = log_probs_data[i * n_classes + c];
                let target_prob = if c == target_class {
                    smooth_positive
                } else {
                    smooth_negative
                };
                sample_loss -= target_prob * log_p;
            }

            total_loss += sample_loss;
        }

        let mean_loss = total_loss / batch_size as f32;
        Tensor::<B, 1>::from_floats([mean_loss], &device)
    }
}

impl Default for LabelSmoothingLoss {
    fn default() -> Self {
        Self::new(0.1)
    }
}

/// Log-Cosh Loss for regression.
///
/// Log-Cosh is the logarithm of the hyperbolic cosine of the prediction error.
/// It's similar to MSE for small errors but is less sensitive to outliers.
///
/// L = log(cosh(pred - target))
///   ≈ (pred - target)^2 / 2   for small errors
///   ≈ |pred - target| - log(2)   for large errors
///
/// This makes it robust to outliers while maintaining smooth gradients.
#[derive(Debug, Default)]
pub struct LogCoshLoss;

impl LogCoshLoss {
    /// Create a new Log-Cosh loss.
    pub fn new() -> Self {
        Self
    }

    /// Compute the loss.
    ///
    /// # Arguments
    ///
    /// * `preds` - Predictions of shape (batch, features)
    /// * `targets` - Targets of shape (batch, features)
    ///
    /// # Returns
    ///
    /// Scalar loss tensor of shape (1,)
    pub fn forward<B: Backend>(&self, preds: Tensor<B, 2>, targets: Tensor<B, 2>) -> Tensor<B, 1> {
        let diff = preds - targets;
        let device = diff.device();

        // Get tensor data
        let diff_data: Vec<f32> = diff.into_data().to_vec().unwrap();

        // Compute log(cosh(x)) = log((exp(x) + exp(-x)) / 2)
        // For numerical stability: log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2)
        let log_cosh_values: Vec<f32> = diff_data
            .iter()
            .map(|&x| {
                let abs_x = x.abs();
                // Numerically stable computation
                abs_x + (1.0 + (-2.0 * abs_x).exp()).ln() - std::f32::consts::LN_2
            })
            .collect();

        // Compute mean
        let mean: f32 = log_cosh_values.iter().sum::<f32>() / log_cosh_values.len() as f32;
        Tensor::<B, 1>::from_floats([mean], &device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_entropy_loss_creation() {
        let _loss = CrossEntropyLoss::new();
        // Just verify it can be created
    }

    #[test]
    fn test_huber_loss_creation() {
        let loss = HuberLoss::new(0.5);
        assert_eq!(loss.delta, 0.5);

        let default_loss = HuberLoss::default();
        assert_eq!(default_loss.delta, 1.0);
    }

    #[test]
    fn test_focal_loss_creation() {
        let loss = FocalLoss::new(2.0);
        assert_eq!(loss.gamma, 2.0);
        assert!(loss.alpha.is_none());

        // With class weights
        let weighted_loss = FocalLoss::new(2.0).with_alpha(vec![0.25, 0.75]);
        assert!(weighted_loss.alpha.is_some());
        assert_eq!(weighted_loss.alpha.unwrap(), vec![0.25, 0.75]);
    }

    #[test]
    fn test_focal_loss_default() {
        let loss = FocalLoss::default();
        assert_eq!(loss.gamma, 2.0);
    }

    #[test]
    fn test_label_smoothing_loss_creation() {
        let loss = LabelSmoothingLoss::new(0.1);
        assert_eq!(loss.smoothing, 0.1);

        let default_loss = LabelSmoothingLoss::default();
        assert_eq!(default_loss.smoothing, 0.1);
    }

    #[test]
    fn test_log_cosh_loss_creation() {
        let _loss = LogCoshLoss::new();
        let _default_loss = LogCoshLoss::default();
        // Just verify it can be created
    }
}
