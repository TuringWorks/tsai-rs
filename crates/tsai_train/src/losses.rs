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

/// Center Loss for learning discriminative features.
///
/// Minimizes intra-class variance by penalizing the distance between
/// features and their class centers. Often used together with cross-entropy
/// for improved feature discrimination.
///
/// L_center = (1/2) * sum(||f_i - c_{y_i}||^2)
///
/// Reference: "A Discriminative Feature Learning Approach for Deep Face Recognition"
/// by Wen et al. (2016)
#[derive(Debug)]
pub struct CenterLoss {
    /// Number of classes.
    num_classes: usize,
    /// Feature dimension.
    feature_dim: usize,
    /// Learning rate for center updates.
    alpha: f32,
    /// Class centers (num_classes, feature_dim).
    centers: Vec<Vec<f32>>,
}

impl CenterLoss {
    /// Create a new Center Loss.
    ///
    /// # Arguments
    ///
    /// * `num_classes` - Number of classes
    /// * `feature_dim` - Dimension of feature embeddings
    /// * `alpha` - Learning rate for center updates (default: 0.5)
    pub fn new(num_classes: usize, feature_dim: usize) -> Self {
        // Initialize centers to zeros
        let centers = vec![vec![0.0f32; feature_dim]; num_classes];
        Self {
            num_classes,
            feature_dim,
            alpha: 0.5,
            centers,
        }
    }

    /// Set the center update learning rate.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Compute center loss.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature embeddings of shape (batch, feature_dim)
    /// * `targets` - Class labels of shape (batch,)
    ///
    /// # Returns
    ///
    /// Scalar loss tensor of shape (1,)
    pub fn forward<B: Backend>(
        &mut self,
        features: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
    ) -> Tensor<B, 1> {
        let device = features.device();
        let [batch_size, feat_dim] = features.dims();

        assert_eq!(
            feat_dim, self.feature_dim,
            "Feature dimension mismatch: expected {}, got {}",
            self.feature_dim, feat_dim
        );

        let features_data: Vec<f32> = features.clone().into_data().to_vec().unwrap();
        let targets_data: Vec<i64> = targets.into_data().to_vec().unwrap();

        let mut total_loss = 0.0f32;
        let mut center_updates: Vec<Vec<f32>> = vec![vec![0.0f32; self.feature_dim]; self.num_classes];
        let mut class_counts: Vec<usize> = vec![0; self.num_classes];

        // Compute loss and accumulate center updates
        for i in 0..batch_size {
            let class_idx = targets_data[i] as usize;
            if class_idx >= self.num_classes {
                continue;
            }

            class_counts[class_idx] += 1;

            // Compute distance to center
            let mut dist_sq = 0.0f32;
            for j in 0..self.feature_dim {
                let diff = features_data[i * self.feature_dim + j] - self.centers[class_idx][j];
                dist_sq += diff * diff;
                // Accumulate updates
                center_updates[class_idx][j] += diff;
            }

            total_loss += 0.5 * dist_sq;
        }

        // Update centers
        for c in 0..self.num_classes {
            if class_counts[c] > 0 {
                let count = class_counts[c] as f32;
                for j in 0..self.feature_dim {
                    self.centers[c][j] += self.alpha * center_updates[c][j] / count;
                }
            }
        }

        let mean_loss = total_loss / batch_size as f32;
        Tensor::<B, 1>::from_floats([mean_loss], &device)
    }

    /// Get current class centers.
    pub fn get_centers(&self) -> &Vec<Vec<f32>> {
        &self.centers
    }

    /// Reset centers to zeros.
    pub fn reset_centers(&mut self) {
        self.centers = vec![vec![0.0f32; self.feature_dim]; self.num_classes];
    }
}

/// Loss type for MaskedLossWrapper.
#[derive(Debug, Clone, Copy, Default)]
pub enum BaseLossType {
    /// Mean Squared Error loss.
    #[default]
    MSE,
    /// Huber loss.
    Huber,
    /// Log-Cosh loss.
    LogCosh,
}

/// Masked Loss Wrapper for handling NaN values.
///
/// Wraps a base loss function and masks out NaN values in predictions
/// or targets before computing the loss. Useful for time series with
/// missing values or partial supervision.
#[derive(Debug)]
pub struct MaskedLossWrapper {
    /// Base loss type.
    loss_type: BaseLossType,
    /// Huber delta (only used if loss_type is Huber).
    huber_delta: f32,
}

impl MaskedLossWrapper {
    /// Create a new masked loss wrapper with MSE as base loss.
    pub fn new() -> Self {
        Self {
            loss_type: BaseLossType::MSE,
            huber_delta: 1.0,
        }
    }

    /// Create with a specific base loss type.
    #[must_use]
    pub fn with_loss_type(mut self, loss_type: BaseLossType) -> Self {
        self.loss_type = loss_type;
        self
    }

    /// Set Huber delta (only applies if using Huber loss).
    #[must_use]
    pub fn with_huber_delta(mut self, delta: f32) -> Self {
        self.huber_delta = delta;
        self
    }

    /// Compute masked loss.
    ///
    /// # Arguments
    ///
    /// * `preds` - Predictions of shape (batch, features)
    /// * `targets` - Targets of shape (batch, features)
    ///
    /// # Returns
    ///
    /// Scalar loss tensor. Returns 0 if all values are masked.
    pub fn forward<B: Backend>(&self, preds: Tensor<B, 2>, targets: Tensor<B, 2>) -> Tensor<B, 1> {
        let device = preds.device();

        let preds_data: Vec<f32> = preds.into_data().to_vec().unwrap();
        let targets_data: Vec<f32> = targets.into_data().to_vec().unwrap();

        // Filter out NaN values
        let valid_pairs: Vec<(f32, f32)> = preds_data
            .iter()
            .zip(&targets_data)
            .filter(|(&p, &t)| !p.is_nan() && !t.is_nan())
            .map(|(&p, &t)| (p, t))
            .collect();

        if valid_pairs.is_empty() {
            return Tensor::<B, 1>::from_floats([0.0], &device);
        }

        let n = valid_pairs.len() as f32;

        let loss = match self.loss_type {
            BaseLossType::MSE => {
                let sum: f32 = valid_pairs.iter().map(|(p, t)| (p - t).powi(2)).sum();
                sum / n
            }
            BaseLossType::Huber => {
                let delta = self.huber_delta;
                let half_delta_sq = 0.5 * delta * delta;
                let sum: f32 = valid_pairs
                    .iter()
                    .map(|(p, t)| {
                        let diff = (p - t).abs();
                        if diff <= delta {
                            0.5 * diff * diff
                        } else {
                            delta * diff - half_delta_sq
                        }
                    })
                    .sum();
                sum / n
            }
            BaseLossType::LogCosh => {
                let sum: f32 = valid_pairs
                    .iter()
                    .map(|(p, t)| {
                        let x = p - t;
                        let abs_x = x.abs();
                        abs_x + (1.0 + (-2.0 * abs_x).exp()).ln() - std::f32::consts::LN_2
                    })
                    .sum();
                sum / n
            }
        };

        Tensor::<B, 1>::from_floats([loss], &device)
    }

    /// Get the fraction of valid (non-NaN) values.
    pub fn get_valid_fraction(preds: &[f32], targets: &[f32]) -> f32 {
        let total = preds.len().min(targets.len());
        if total == 0 {
            return 0.0;
        }

        let valid_count = preds
            .iter()
            .zip(targets)
            .filter(|(&p, &t)| !p.is_nan() && !t.is_nan())
            .count();

        valid_count as f32 / total as f32
    }
}

impl Default for MaskedLossWrapper {
    fn default() -> Self {
        Self::new()
    }
}

/// Center Plus Loss: Combined Center Loss and Cross-Entropy Loss.
///
/// This loss function combines:
/// 1. Cross-entropy loss for classification
/// 2. Center loss for learning discriminative features
///
/// L = L_softmax + lambda * L_center
///
/// The lambda parameter controls the trade-off between classification
/// accuracy (softmax) and feature discrimination (center).
///
/// Reference: "A Discriminative Feature Learning Approach for Deep Face Recognition"
/// by Wen et al. (2016)
#[derive(Debug)]
pub struct CenterPlusLoss {
    /// Center loss component.
    center_loss: CenterLoss,
    /// Weight for center loss relative to cross-entropy.
    lambda: f32,
}

impl CenterPlusLoss {
    /// Create a new Center Plus Loss.
    ///
    /// # Arguments
    ///
    /// * `num_classes` - Number of classes
    /// * `feature_dim` - Dimension of feature embeddings
    /// * `lambda` - Weight for center loss (default: 0.003)
    pub fn new(num_classes: usize, feature_dim: usize) -> Self {
        Self {
            center_loss: CenterLoss::new(num_classes, feature_dim),
            lambda: 0.003,
        }
    }

    /// Set the center loss weight.
    #[must_use]
    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.lambda = lambda;
        self
    }

    /// Set the center update learning rate.
    #[must_use]
    pub fn with_center_alpha(mut self, alpha: f32) -> Self {
        self.center_loss = self.center_loss.with_alpha(alpha);
        self
    }

    /// Compute combined center plus softmax loss.
    ///
    /// # Arguments
    ///
    /// * `logits` - Raw model outputs for classification of shape (batch, n_classes)
    /// * `features` - Feature embeddings of shape (batch, feature_dim)
    /// * `targets` - Class labels of shape (batch,)
    ///
    /// # Returns
    ///
    /// Scalar loss tensor of shape (1,)
    pub fn forward<B: Backend>(
        &mut self,
        logits: Tensor<B, 2>,
        features: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
    ) -> Tensor<B, 1> {
        let device = logits.device();

        // Compute cross-entropy loss
        let ce_loss = CrossEntropyLossConfig::new()
            .init(&device)
            .forward(logits, targets.clone());
        let ce_data: Vec<f32> = ce_loss.into_data().to_vec().unwrap();

        // Compute center loss
        let center_loss = self.center_loss.forward(features, targets);
        let center_data: Vec<f32> = center_loss.into_data().to_vec().unwrap();

        // Combined loss
        let total_loss = ce_data[0] + self.lambda * center_data[0];
        Tensor::<B, 1>::from_floats([total_loss], &device)
    }

    /// Get center loss component for separate center loss value.
    pub fn get_center_loss(&self) -> &CenterLoss {
        &self.center_loss
    }

    /// Get current class centers.
    pub fn get_centers(&self) -> &Vec<Vec<f32>> {
        self.center_loss.get_centers()
    }

    /// Reset centers to zeros.
    pub fn reset_centers(&mut self) {
        self.center_loss.reset_centers();
    }
}

/// Tweedie Loss for compound Poisson distributions.
///
/// The Tweedie distribution is a family of distributions that includes
/// Gaussian (p=0), Poisson (p=1), and Gamma (p=2) as special cases.
/// It's particularly useful for modeling zero-inflated continuous data.
///
/// L = -y * exp((1-p)*log(mu)) / (1-p) + exp((2-p)*log(mu)) / (2-p)
///
/// Where:
/// - y is the target
/// - mu is the predicted mean (exp(log_mu) for numerical stability)
/// - p is the Tweedie power parameter (typically 1 < p < 2)
///
/// Common applications:
/// - Insurance claim amounts (zero-inflated)
/// - Rainfall data
/// - Revenue prediction with many zeros
#[derive(Debug)]
pub struct TweedieLoss {
    /// Tweedie power parameter. 1 < p < 2 for compound Poisson.
    pub power: f32,
    /// Small epsilon for numerical stability.
    epsilon: f32,
}

impl TweedieLoss {
    /// Create a new Tweedie Loss.
    ///
    /// # Arguments
    ///
    /// * `power` - Tweedie power parameter (typically 1.5)
    ///   - 0: Gaussian
    ///   - 1: Poisson
    ///   - 1 < p < 2: Compound Poisson-Gamma
    ///   - 2: Gamma
    pub fn new(power: f32) -> Self {
        Self {
            power,
            epsilon: 1e-8,
        }
    }

    /// Compute Tweedie loss.
    ///
    /// # Arguments
    ///
    /// * `preds` - Log predictions (log(mu)) of shape (batch, features)
    /// * `targets` - Targets of shape (batch, features)
    ///
    /// # Returns
    ///
    /// Scalar loss tensor of shape (1,)
    pub fn forward<B: Backend>(&self, preds: Tensor<B, 2>, targets: Tensor<B, 2>) -> Tensor<B, 1> {
        let device = preds.device();

        let preds_data: Vec<f32> = preds.into_data().to_vec().unwrap();
        let targets_data: Vec<f32> = targets.into_data().to_vec().unwrap();

        let p = self.power;
        let one_minus_p = 1.0 - p;
        let two_minus_p = 2.0 - p;

        let mut total_loss = 0.0f32;
        let n = preds_data.len();

        for i in 0..n {
            let log_mu = preds_data[i];
            let y = targets_data[i].max(0.0); // Ensure non-negative

            // mu = exp(log_mu) with clipping for stability
            let mu = log_mu.exp().max(self.epsilon);

            // Tweedie deviance:
            // d = 2 * (y^(2-p)/((1-p)*(2-p)) - y*mu^(1-p)/(1-p) + mu^(2-p)/(2-p))
            // Simplified negative log-likelihood form:
            // L = -y * mu^(1-p) / (1-p) + mu^(2-p) / (2-p)

            let term1 = if y > self.epsilon {
                -y * mu.powf(one_minus_p) / one_minus_p
            } else {
                0.0
            };

            let term2 = mu.powf(two_minus_p) / two_minus_p;

            total_loss += term1 + term2;
        }

        let mean_loss = total_loss / n as f32;
        Tensor::<B, 1>::from_floats([mean_loss], &device)
    }

    /// Compute Tweedie deviance (alternative formulation).
    ///
    /// The deviance is a measure of goodness of fit:
    /// D = 2 * sum(w * (y*g(y) - y*g(mu) - h(y) + h(mu)))
    ///
    /// where g and h depend on the power parameter.
    pub fn deviance<B: Backend>(&self, preds: Tensor<B, 2>, targets: Tensor<B, 2>) -> Tensor<B, 1> {
        let device = preds.device();

        let preds_data: Vec<f32> = preds.into_data().to_vec().unwrap();
        let targets_data: Vec<f32> = targets.into_data().to_vec().unwrap();

        let p = self.power;

        let mut total_dev = 0.0f32;
        let n = preds_data.len();

        for i in 0..n {
            let log_mu = preds_data[i];
            let y = targets_data[i].max(0.0);
            let mu = log_mu.exp().max(self.epsilon);

            // Unit deviance for Tweedie distribution
            let dev = if y > self.epsilon {
                let term1 = y.powf(2.0 - p) / ((1.0 - p) * (2.0 - p));
                let term2 = y * mu.powf(1.0 - p) / (1.0 - p);
                let term3 = mu.powf(2.0 - p) / (2.0 - p);
                2.0 * (term1 - term2 + term3)
            } else {
                2.0 * mu.powf(2.0 - p) / (2.0 - p)
            };

            total_dev += dev;
        }

        let mean_dev = total_dev / n as f32;
        Tensor::<B, 1>::from_floats([mean_dev], &device)
    }
}

impl Default for TweedieLoss {
    fn default() -> Self {
        Self::new(1.5) // Compound Poisson-Gamma
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
