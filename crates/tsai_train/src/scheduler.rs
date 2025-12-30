//! Learning rate schedulers.

use serde::{Deserialize, Serialize};

/// Trait for learning rate schedulers.
pub trait Scheduler: Send + Sync {
    /// Get the learning rate for the current step.
    fn get_lr(&self, step: usize) -> f64;

    /// Get the scheduler name.
    fn name(&self) -> &str;
}

/// Configuration for OneCycleLR scheduler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OneCycleLRConfig {
    /// Maximum learning rate.
    pub max_lr: f64,
    /// Total number of steps.
    pub total_steps: usize,
    /// Percentage of steps for warmup.
    pub pct_start: f64,
    /// Division factor for initial LR.
    pub div_factor: f64,
    /// Final division factor.
    pub final_div_factor: f64,
}

impl Default for OneCycleLRConfig {
    fn default() -> Self {
        Self {
            max_lr: 1e-3,
            total_steps: 1000,
            pct_start: 0.3,
            div_factor: 25.0,
            final_div_factor: 10000.0,
        }
    }
}

/// One-cycle learning rate scheduler.
///
/// Implements the 1cycle policy from Leslie Smith's paper.
/// The LR starts low, increases to max_lr, then decreases to a very low value.
#[derive(Debug, Clone)]
pub struct OneCycleLR {
    config: OneCycleLRConfig,
    initial_lr: f64,
    final_lr: f64,
    warmup_steps: usize,
}

impl OneCycleLR {
    /// Create a new OneCycleLR scheduler.
    pub fn new(config: OneCycleLRConfig) -> Self {
        let initial_lr = config.max_lr / config.div_factor;
        let final_lr = config.max_lr / config.final_div_factor;
        let warmup_steps = (config.total_steps as f64 * config.pct_start) as usize;

        Self {
            config,
            initial_lr,
            final_lr,
            warmup_steps,
        }
    }

    /// Create with just max_lr and total_steps.
    pub fn simple(max_lr: f64, total_steps: usize) -> Self {
        Self::new(OneCycleLRConfig {
            max_lr,
            total_steps,
            ..Default::default()
        })
    }
}

impl Scheduler for OneCycleLR {
    fn get_lr(&self, step: usize) -> f64 {
        let step = step.min(self.config.total_steps.saturating_sub(1));

        if step < self.warmup_steps {
            // Warmup phase: linear increase from initial_lr to max_lr
            let progress = step as f64 / self.warmup_steps as f64;
            self.initial_lr + (self.config.max_lr - self.initial_lr) * progress
        } else {
            // Annealing phase: cosine decay from max_lr to final_lr
            let annealing_steps = self.config.total_steps - self.warmup_steps;
            let progress = (step - self.warmup_steps) as f64 / annealing_steps as f64;
            let cosine = (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
            self.final_lr + (self.config.max_lr - self.final_lr) * cosine
        }
    }

    fn name(&self) -> &str {
        "OneCycleLR"
    }
}

/// Cosine annealing scheduler.
#[derive(Debug, Clone)]
pub struct CosineAnnealingLR {
    initial_lr: f64,
    min_lr: f64,
    total_steps: usize,
}

impl CosineAnnealingLR {
    /// Create a new cosine annealing scheduler.
    pub fn new(initial_lr: f64, min_lr: f64, total_steps: usize) -> Self {
        Self {
            initial_lr,
            min_lr,
            total_steps,
        }
    }
}

impl Scheduler for CosineAnnealingLR {
    fn get_lr(&self, step: usize) -> f64 {
        let step = step.min(self.total_steps.saturating_sub(1));
        let progress = step as f64 / self.total_steps as f64;
        let cosine = (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
        self.min_lr + (self.initial_lr - self.min_lr) * cosine
    }

    fn name(&self) -> &str {
        "CosineAnnealingLR"
    }
}

/// Step decay scheduler.
#[derive(Debug, Clone)]
pub struct StepLR {
    initial_lr: f64,
    step_size: usize,
    gamma: f64,
}

impl StepLR {
    /// Create a new step decay scheduler.
    pub fn new(initial_lr: f64, step_size: usize, gamma: f64) -> Self {
        Self {
            initial_lr,
            step_size,
            gamma,
        }
    }
}

impl Scheduler for StepLR {
    fn get_lr(&self, step: usize) -> f64 {
        let n_decays = step / self.step_size;
        self.initial_lr * self.gamma.powi(n_decays as i32)
    }

    fn name(&self) -> &str {
        "StepLR"
    }
}

/// Constant learning rate (no scheduling).
#[derive(Debug, Clone)]
pub struct ConstantLR {
    lr: f64,
}

impl ConstantLR {
    /// Create a new constant LR scheduler.
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }
}

impl Scheduler for ConstantLR {
    fn get_lr(&self, _step: usize) -> f64 {
        self.lr
    }

    fn name(&self) -> &str {
        "ConstantLR"
    }
}

/// Exponential decay scheduler.
#[derive(Debug, Clone)]
pub struct ExponentialLR {
    initial_lr: f64,
    gamma: f64,
}

impl ExponentialLR {
    /// Create a new exponential decay scheduler.
    ///
    /// # Arguments
    /// * `initial_lr` - Starting learning rate
    /// * `gamma` - Decay factor applied each step (typically 0.95-0.99)
    pub fn new(initial_lr: f64, gamma: f64) -> Self {
        Self { initial_lr, gamma }
    }
}

impl Scheduler for ExponentialLR {
    fn get_lr(&self, step: usize) -> f64 {
        self.initial_lr * self.gamma.powi(step as i32)
    }

    fn name(&self) -> &str {
        "ExponentialLR"
    }
}

/// Polynomial decay scheduler.
#[derive(Debug, Clone)]
pub struct PolynomialLR {
    initial_lr: f64,
    end_lr: f64,
    total_steps: usize,
    power: f64,
}

impl PolynomialLR {
    /// Create a new polynomial decay scheduler.
    ///
    /// # Arguments
    /// * `initial_lr` - Starting learning rate
    /// * `end_lr` - Final learning rate
    /// * `total_steps` - Total number of steps
    /// * `power` - Polynomial power (1.0 = linear, 2.0 = quadratic, etc.)
    pub fn new(initial_lr: f64, end_lr: f64, total_steps: usize, power: f64) -> Self {
        Self {
            initial_lr,
            end_lr,
            total_steps,
            power,
        }
    }

    /// Create a linear decay scheduler.
    pub fn linear(initial_lr: f64, end_lr: f64, total_steps: usize) -> Self {
        Self::new(initial_lr, end_lr, total_steps, 1.0)
    }
}

impl Scheduler for PolynomialLR {
    fn get_lr(&self, step: usize) -> f64 {
        let step = step.min(self.total_steps);
        let progress = step as f64 / self.total_steps as f64;
        let decay = (1.0 - progress).powf(self.power);
        self.end_lr + (self.initial_lr - self.end_lr) * decay
    }

    fn name(&self) -> &str {
        "PolynomialLR"
    }
}

/// Cosine annealing with warm restarts.
///
/// Implements SGDR: Stochastic Gradient Descent with Warm Restarts.
/// The learning rate follows a cosine curve, then restarts at max_lr.
#[derive(Debug, Clone)]
pub struct CosineAnnealingWarmRestarts {
    initial_lr: f64,
    min_lr: f64,
    t_0: usize,
    t_mult: usize,
}

impl CosineAnnealingWarmRestarts {
    /// Create a new cosine annealing with warm restarts scheduler.
    ///
    /// # Arguments
    /// * `initial_lr` - Maximum learning rate
    /// * `min_lr` - Minimum learning rate
    /// * `t_0` - Number of steps for first restart
    /// * `t_mult` - Factor to increase restart period after each restart
    pub fn new(initial_lr: f64, min_lr: f64, t_0: usize, t_mult: usize) -> Self {
        Self {
            initial_lr,
            min_lr,
            t_0,
            t_mult: t_mult.max(1),
        }
    }
}

impl Scheduler for CosineAnnealingWarmRestarts {
    fn get_lr(&self, step: usize) -> f64 {
        // Find which restart cycle we're in
        let mut cycle_start = 0;
        let mut cycle_len = self.t_0;
        let mut cycle = 0;

        while step >= cycle_start + cycle_len {
            cycle_start += cycle_len;
            if self.t_mult > 1 {
                cycle_len *= self.t_mult;
            }
            cycle += 1;
            // Safety check to prevent infinite loop
            if cycle > 100 {
                break;
            }
        }

        // Position within current cycle
        let t_cur = step - cycle_start;
        let progress = t_cur as f64 / cycle_len as f64;
        let cosine = (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
        self.min_lr + (self.initial_lr - self.min_lr) * cosine
    }

    fn name(&self) -> &str {
        "CosineAnnealingWarmRestarts"
    }
}

/// Linear warmup followed by another scheduler.
#[derive(Debug, Clone)]
pub struct LinearWarmup<S: Scheduler> {
    warmup_steps: usize,
    warmup_start_lr: f64,
    inner: S,
}

impl<S: Scheduler> LinearWarmup<S> {
    /// Create a new warmup scheduler.
    ///
    /// # Arguments
    /// * `warmup_steps` - Number of warmup steps
    /// * `warmup_start_lr` - Starting learning rate for warmup
    /// * `inner` - Scheduler to use after warmup
    pub fn new(warmup_steps: usize, warmup_start_lr: f64, inner: S) -> Self {
        Self {
            warmup_steps,
            warmup_start_lr,
            inner,
        }
    }
}

impl<S: Scheduler> Scheduler for LinearWarmup<S> {
    fn get_lr(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            // Linear warmup
            let target_lr = self.inner.get_lr(self.warmup_steps);
            let progress = step as f64 / self.warmup_steps as f64;
            self.warmup_start_lr + (target_lr - self.warmup_start_lr) * progress
        } else {
            self.inner.get_lr(step)
        }
    }

    fn name(&self) -> &str {
        "LinearWarmup"
    }
}

/// ReduceLROnPlateau - reduces LR when metric stops improving.
///
/// This is stateful and must be updated with metric values.
#[derive(Debug, Clone)]
pub struct ReduceLROnPlateau {
    current_lr: f64,
    factor: f64,
    patience: usize,
    min_lr: f64,
    mode: ReduceMode,
    best_value: f64,
    num_bad_epochs: usize,
    cooldown: usize,
    cooldown_counter: usize,
}

/// Mode for ReduceLROnPlateau.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceMode {
    /// Reduce LR when metric stops decreasing (for losses).
    Min,
    /// Reduce LR when metric stops increasing (for accuracy).
    Max,
}

impl ReduceLROnPlateau {
    /// Create a new ReduceLROnPlateau scheduler.
    ///
    /// # Arguments
    /// * `initial_lr` - Starting learning rate
    /// * `factor` - Factor to reduce LR by (new_lr = old_lr * factor)
    /// * `patience` - Number of epochs with no improvement before reducing
    /// * `min_lr` - Minimum learning rate
    /// * `mode` - Whether to minimize or maximize the metric
    pub fn new(
        initial_lr: f64,
        factor: f64,
        patience: usize,
        min_lr: f64,
        mode: ReduceMode,
    ) -> Self {
        let best_value = match mode {
            ReduceMode::Min => f64::INFINITY,
            ReduceMode::Max => f64::NEG_INFINITY,
        };

        Self {
            current_lr: initial_lr,
            factor,
            patience,
            min_lr,
            mode,
            best_value,
            num_bad_epochs: 0,
            cooldown: 0,
            cooldown_counter: 0,
        }
    }

    /// Set cooldown period after LR reduction.
    #[must_use]
    pub fn with_cooldown(mut self, cooldown: usize) -> Self {
        self.cooldown = cooldown;
        self
    }

    /// Update with a new metric value.
    ///
    /// Returns true if learning rate was reduced.
    pub fn step(&mut self, metric: f64) -> bool {
        // Handle cooldown
        if self.cooldown_counter > 0 {
            self.cooldown_counter -= 1;
            return false;
        }

        // Check if metric improved
        let improved = match self.mode {
            ReduceMode::Min => metric < self.best_value,
            ReduceMode::Max => metric > self.best_value,
        };

        if improved {
            self.best_value = metric;
            self.num_bad_epochs = 0;
            false
        } else {
            self.num_bad_epochs += 1;
            if self.num_bad_epochs > self.patience {
                let new_lr = (self.current_lr * self.factor).max(self.min_lr);
                if new_lr < self.current_lr {
                    self.current_lr = new_lr;
                    self.num_bad_epochs = 0;
                    self.cooldown_counter = self.cooldown;
                    return true;
                }
            }
            false
        }
    }

    /// Get the current learning rate.
    pub fn current_lr(&self) -> f64 {
        self.current_lr
    }
}

impl Scheduler for ReduceLROnPlateau {
    fn get_lr(&self, _step: usize) -> f64 {
        // This scheduler is metric-based, not step-based
        // Returns the current LR which is updated via step()
        self.current_lr
    }

    fn name(&self) -> &str {
        "ReduceLROnPlateau"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_cycle_lr() {
        let scheduler = OneCycleLR::simple(1e-3, 1000);

        // Start should be low
        let start_lr = scheduler.get_lr(0);
        assert!(start_lr < 1e-3);

        // Peak should be at max
        let peak_lr = scheduler.get_lr(300); // 30% of steps
        assert!((peak_lr - 1e-3).abs() < 1e-6);

        // End should be very low
        let end_lr = scheduler.get_lr(999);
        assert!(end_lr < start_lr);
    }

    #[test]
    fn test_cosine_annealing() {
        let scheduler = CosineAnnealingLR::new(1e-3, 1e-5, 100);

        assert!((scheduler.get_lr(0) - 1e-3).abs() < 1e-8);
        assert!(scheduler.get_lr(50) < 1e-3);
        // At step 99 (clamped), we get very close to min_lr but not exactly
        assert!(scheduler.get_lr(100) < 1e-4); // Should be close to min_lr
    }

    #[test]
    fn test_step_lr() {
        let scheduler = StepLR::new(1e-3, 10, 0.1);

        assert!((scheduler.get_lr(0) - 1e-3).abs() < 1e-8);
        assert!((scheduler.get_lr(10) - 1e-4).abs() < 1e-9);
        assert!((scheduler.get_lr(20) - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_lr() {
        let scheduler = ExponentialLR::new(1e-3, 0.9);

        assert!((scheduler.get_lr(0) - 1e-3).abs() < 1e-8);
        // After 1 step: 1e-3 * 0.9 = 9e-4
        assert!((scheduler.get_lr(1) - 9e-4).abs() < 1e-8);
        // After 10 steps: 1e-3 * 0.9^10 ≈ 3.486e-4
        assert!(scheduler.get_lr(10) < 4e-4);
    }

    #[test]
    fn test_polynomial_lr() {
        let scheduler = PolynomialLR::linear(1e-3, 1e-5, 100);

        assert!((scheduler.get_lr(0) - 1e-3).abs() < 1e-8);
        // At step 50, should be halfway: (1e-3 + 1e-5) / 2 ≈ 5.05e-4
        let mid_lr = scheduler.get_lr(50);
        assert!(mid_lr < 1e-3 && mid_lr > 1e-5);
        // At end should be close to 1e-5
        assert!((scheduler.get_lr(100) - 1e-5).abs() < 1e-7);
    }

    #[test]
    fn test_cosine_warm_restarts() {
        let scheduler = CosineAnnealingWarmRestarts::new(1e-3, 1e-5, 10, 2);

        // Start should be at max
        assert!((scheduler.get_lr(0) - 1e-3).abs() < 1e-8);
        // End of first cycle (step 9) should be near min
        assert!(scheduler.get_lr(9) < 2e-4);
        // After restart (step 10) should be back at max
        assert!(scheduler.get_lr(10) > 9e-4);
    }

    #[test]
    fn test_reduce_lr_on_plateau() {
        let mut scheduler = ReduceLROnPlateau::new(1e-3, 0.1, 2, 1e-6, ReduceMode::Min);

        assert!((scheduler.current_lr() - 1e-3).abs() < 1e-8);

        // Improving: no reduction
        scheduler.step(1.0);
        scheduler.step(0.9);
        assert!((scheduler.current_lr() - 1e-3).abs() < 1e-8);

        // Not improving for 3 epochs (patience=2, so need 3)
        scheduler.step(0.95); // bad epoch 1
        scheduler.step(0.95); // bad epoch 2
        let reduced = scheduler.step(0.95); // bad epoch 3 -> reduce
        assert!(reduced);
        assert!((scheduler.current_lr() - 1e-4).abs() < 1e-9);
    }

    #[test]
    fn test_linear_warmup() {
        let inner = ConstantLR::new(1e-3);
        let scheduler = LinearWarmup::new(10, 0.0, inner);

        // Start at 0
        assert!(scheduler.get_lr(0) < 1e-5);
        // Midway should be half
        assert!((scheduler.get_lr(5) - 5e-4).abs() < 1e-6);
        // After warmup should be at target
        assert!((scheduler.get_lr(10) - 1e-3).abs() < 1e-8);
    }
}
