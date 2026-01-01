//! Custom optimizers for time series deep learning.
//!
//! Provides advanced optimizers beyond those in Burn:
//! - [`RAdam`] - Rectified Adam with variance rectification
//! - [`Ranger`] - RAdam + Lookahead for improved training stability

use burn::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// RAdam (Rectified Adam) optimizer configuration.
///
/// RAdam addresses the variance issue in the early stages of Adam training
/// by using a variance rectification term that prevents the need for warmup.
///
/// Reference: "On the Variance of the Adaptive Learning Rate and Beyond"
/// by Liu et al. (2019)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RAdamConfig {
    /// Learning rate.
    pub lr: f64,
    /// First moment decay (beta1).
    pub beta1: f64,
    /// Second moment decay (beta2).
    pub beta2: f64,
    /// Small epsilon for numerical stability.
    pub epsilon: f64,
    /// Weight decay coefficient.
    pub weight_decay: f64,
}

impl Default for RAdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
        }
    }
}

impl RAdamConfig {
    /// Create a new RAdam configuration with the given learning rate.
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            ..Default::default()
        }
    }

    /// Set beta1 (first moment decay).
    #[must_use]
    pub fn with_beta1(mut self, beta1: f64) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set beta2 (second moment decay).
    #[must_use]
    pub fn with_beta2(mut self, beta2: f64) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Set epsilon for numerical stability.
    #[must_use]
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set weight decay.
    #[must_use]
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

/// RAdam optimizer state for a single parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RAdamParamState {
    /// First moment (mean of gradients).
    pub m: Vec<f32>,
    /// Second moment (mean of squared gradients).
    pub v: Vec<f32>,
}

/// RAdam optimizer.
///
/// Rectified Adam that uses variance rectification to address the
/// large variance of adaptive learning rate during early training.
/// This often eliminates the need for learning rate warmup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RAdam {
    config: RAdamConfig,
    /// Current training step.
    step: usize,
    /// Per-parameter states indexed by parameter id.
    states: HashMap<String, RAdamParamState>,
}

impl RAdam {
    /// Create a new RAdam optimizer.
    pub fn new(config: RAdamConfig) -> Self {
        Self {
            config,
            step: 0,
            states: HashMap::new(),
        }
    }

    /// Get current learning rate.
    pub fn lr(&self) -> f64 {
        self.config.lr
    }

    /// Set learning rate.
    pub fn set_lr(&mut self, lr: f64) {
        self.config.lr = lr;
    }

    /// Get current step.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Compute the maximum length of approximated SMA.
    fn rho_inf(&self) -> f64 {
        2.0 / (1.0 - self.config.beta2) - 1.0
    }

    /// Update a single parameter tensor.
    pub fn step_param<B: Backend>(
        &mut self,
        param_id: &str,
        param: Tensor<B, 1>,
        grad: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let device = param.device();
        self.step += 1;
        let t = self.step as f64;

        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let lr = self.config.lr;
        let eps = self.config.epsilon;
        let wd = self.config.weight_decay;

        // Compute rho_inf before borrowing states
        let rho_inf = self.rho_inf();

        // Get parameter data
        let param_data: Vec<f32> = param.clone().into_data().to_vec().unwrap();
        let grad_data: Vec<f32> = grad.into_data().to_vec().unwrap();
        let n = param_data.len();

        // Initialize or get state
        let state = self.states.entry(param_id.to_string()).or_insert_with(|| {
            RAdamParamState {
                m: vec![0.0; n],
                v: vec![0.0; n],
            }
        });

        // Update biased first and second moment estimates
        for i in 0..n {
            state.m[i] = (beta1 as f32) * state.m[i] + (1.0 - beta1 as f32) * grad_data[i];
            state.v[i] =
                (beta2 as f32) * state.v[i] + (1.0 - beta2 as f32) * grad_data[i] * grad_data[i];
        }

        // Compute bias-corrected first moment
        let beta1_t = beta1.powi(t as i32);
        let beta2_t = beta2.powi(t as i32);
        let m_hat: Vec<f32> = state.m.iter().map(|&m| m / (1.0 - beta1_t as f32)).collect();

        // Compute SMA (simple moving average length)
        let rho_t = rho_inf - 2.0 * t * beta2_t / (1.0 - beta2_t);

        // Compute updated parameters
        let updated: Vec<f32> = if rho_t > 5.0 {
            // Variance is tractable - use adaptive learning rate
            let v_hat: Vec<f32> = state.v.iter().map(|&v| v / (1.0 - beta2_t as f32)).collect();

            // Compute variance rectification term
            let r = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf
                / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))
                .sqrt();

            param_data
                .iter()
                .zip(m_hat.iter())
                .zip(v_hat.iter())
                .map(|((&p, &m), &v)| {
                    let update = (r as f32) * m / (v.sqrt() + eps as f32);
                    p - (lr as f32) * update - (wd as f32) * p
                })
                .collect()
        } else {
            // Variance is not tractable - use unadapted learning rate
            param_data
                .iter()
                .zip(m_hat.iter())
                .map(|(&p, &m)| p - (lr as f32) * m - (wd as f32) * p)
                .collect()
        };

        Tensor::from_floats(updated.as_slice(), &device)
    }
}

/// Ranger optimizer configuration.
///
/// Ranger combines RAdam with Lookahead for improved training stability
/// and generalization. It uses RAdam as the inner optimizer and applies
/// Lookahead's slow weights mechanism.
///
/// Reference: "Lookahead Optimizer: k steps forward, 1 step back"
/// by Zhang et al. (2019)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangerConfig {
    /// RAdam configuration.
    pub radam: RAdamConfig,
    /// Lookahead alpha (interpolation coefficient).
    pub alpha: f64,
    /// Lookahead k (synchronization period).
    pub k: usize,
}

impl Default for RangerConfig {
    fn default() -> Self {
        Self {
            radam: RAdamConfig::default(),
            alpha: 0.5,
            k: 6,
        }
    }
}

impl RangerConfig {
    /// Create a new Ranger configuration with the given learning rate.
    pub fn new(lr: f64) -> Self {
        Self {
            radam: RAdamConfig::new(lr),
            ..Default::default()
        }
    }

    /// Set lookahead alpha.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set lookahead k (synchronization period).
    #[must_use]
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set weight decay.
    #[must_use]
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.radam.weight_decay = weight_decay;
        self
    }

    /// Set beta1.
    #[must_use]
    pub fn with_beta1(mut self, beta1: f64) -> Self {
        self.radam.beta1 = beta1;
        self
    }

    /// Set beta2.
    #[must_use]
    pub fn with_beta2(mut self, beta2: f64) -> Self {
        self.radam.beta2 = beta2;
        self
    }
}

/// Ranger optimizer state for a single parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangerParamState {
    /// RAdam state.
    pub radam_state: RAdamParamState,
    /// Slow weights for lookahead.
    pub slow_weights: Vec<f32>,
}

/// Ranger optimizer.
///
/// Combines RAdam (Rectified Adam) with Lookahead for:
/// - Variance rectification (no warmup needed)
/// - Improved generalization through slow weights
/// - More stable training dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ranger {
    config: RangerConfig,
    /// Inner RAdam optimizer.
    radam: RAdam,
    /// Slow weights for lookahead indexed by parameter id.
    slow_weights: HashMap<String, Vec<f32>>,
    /// Counter for lookahead synchronization.
    sync_counter: usize,
}

impl Ranger {
    /// Create a new Ranger optimizer.
    pub fn new(config: RangerConfig) -> Self {
        let radam = RAdam::new(config.radam.clone());
        Self {
            config,
            radam,
            slow_weights: HashMap::new(),
            sync_counter: 0,
        }
    }

    /// Get current learning rate.
    pub fn lr(&self) -> f64 {
        self.radam.lr()
    }

    /// Set learning rate.
    pub fn set_lr(&mut self, lr: f64) {
        self.radam.set_lr(lr);
    }

    /// Get current step.
    pub fn step(&self) -> usize {
        self.radam.step()
    }

    /// Update a single parameter tensor.
    pub fn step_param<B: Backend>(
        &mut self,
        param_id: &str,
        param: Tensor<B, 1>,
        grad: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let device = param.device();

        // RAdam step
        let fast_weights = self.radam.step_param(param_id, param, grad);
        let fast_data: Vec<f32> = fast_weights.clone().into_data().to_vec().unwrap();

        // Initialize slow weights if needed
        if !self.slow_weights.contains_key(param_id) {
            self.slow_weights
                .insert(param_id.to_string(), fast_data.clone());
        }

        self.sync_counter += 1;

        // Lookahead synchronization
        if self.sync_counter >= self.config.k {
            self.sync_counter = 0;

            let slow = self.slow_weights.get_mut(param_id).unwrap();
            let alpha = self.config.alpha as f32;

            // Update slow weights: slow = slow + alpha * (fast - slow)
            for i in 0..slow.len() {
                slow[i] += alpha * (fast_data[i] - slow[i]);
            }

            // Return slow weights (they become the new fast weights)
            Tensor::from_floats(slow.as_slice(), &device)
        } else {
            fast_weights
        }
    }

    /// Force lookahead synchronization.
    pub fn sync_lookahead(&mut self) {
        self.sync_counter = self.config.k;
    }
}

/// Wrapper to make RAdam/Ranger work with Burn's optimizer interface.
///
/// This provides a facade that can be used with Burn's training infrastructure
/// while using our custom optimizer implementations internally.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    /// Current learning rate.
    pub lr: f64,
    /// Current step.
    pub step: usize,
}

impl Default for OptimizerState {
    fn default() -> Self {
        Self { lr: 1e-3, step: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_radam_config() {
        let config = RAdamConfig::new(1e-3)
            .with_beta1(0.9)
            .with_beta2(0.999)
            .with_weight_decay(0.01);

        assert!((config.lr - 1e-3).abs() < 1e-10);
        assert!((config.beta1 - 0.9).abs() < 1e-10);
        assert!((config.beta2 - 0.999).abs() < 1e-10);
        assert!((config.weight_decay - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_radam_step() {
        let config = RAdamConfig::new(0.1);
        let mut optimizer = RAdam::new(config);

        let device = Default::default();
        let param = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 3.0], &device);
        let grad = Tensor::<TestBackend, 1>::from_floats([0.1, 0.1, 0.1], &device);

        let updated = optimizer.step_param("test", param.clone(), grad);
        let updated_data: Vec<f32> = updated.into_data().to_vec().unwrap();

        // Parameters should have decreased (gradient descent)
        assert!(updated_data[0] < 1.0);
        assert!(updated_data[1] < 2.0);
        assert!(updated_data[2] < 3.0);
    }

    #[test]
    fn test_ranger_config() {
        let config = RangerConfig::new(1e-3).with_alpha(0.5).with_k(6);

        assert!((config.radam.lr - 1e-3).abs() < 1e-10);
        assert!((config.alpha - 0.5).abs() < 1e-10);
        assert_eq!(config.k, 6);
    }

    #[test]
    fn test_ranger_step() {
        let config = RangerConfig::new(0.1);
        let mut optimizer = Ranger::new(config);

        let device = Default::default();

        // Run multiple steps
        for _ in 0..10 {
            let param = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 3.0], &device);
            let grad = Tensor::<TestBackend, 1>::from_floats([0.1, 0.1, 0.1], &device);
            let _ = optimizer.step_param("test", param, grad);
        }

        assert_eq!(optimizer.step(), 10);
    }
}
