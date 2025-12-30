//! Training loop implementation.
//!
//! Provides the actual training loop for classification, regression, and forecasting.

use std::time::Instant;

use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use crate::error::Result;
use crate::scheduler::{OneCycleLR, Scheduler};
use tsai_data::TSDataLoaders;

/// Training output with metrics and final model.
#[derive(Debug)]
pub struct TrainingOutput<M> {
    /// Trained model.
    pub model: M,
    /// Training losses per epoch.
    pub train_losses: Vec<f32>,
    /// Validation losses per epoch.
    pub valid_losses: Vec<f32>,
    /// Validation accuracies per epoch.
    pub valid_accs: Vec<f32>,
    /// Best validation accuracy.
    pub best_valid_acc: f32,
    /// Best epoch.
    pub best_epoch: usize,
    /// Total training time in seconds.
    pub training_time_secs: f64,
}

/// Configuration for classification training.
#[derive(Debug, Clone)]
pub struct ClassificationTrainerConfig {
    /// Number of epochs.
    pub n_epochs: usize,
    /// Learning rate.
    pub lr: f64,
    /// Weight decay.
    pub weight_decay: f32,
    /// Gradient clipping value (0 = disabled).
    pub grad_clip: f32,
    /// Whether to print progress.
    pub verbose: bool,
    /// Early stopping patience (0 = disabled).
    pub early_stopping_patience: usize,
    /// Minimum delta for early stopping improvement.
    pub early_stopping_min_delta: f32,
}

impl Default for ClassificationTrainerConfig {
    fn default() -> Self {
        Self {
            n_epochs: 25,
            lr: 1e-3,
            weight_decay: 0.01,
            grad_clip: 1.0,
            verbose: true,
            early_stopping_patience: 0,
            early_stopping_min_delta: 0.001,
        }
    }
}

/// Trainer for classification models.
pub struct ClassificationTrainer<B: AutodiffBackend> {
    config: ClassificationTrainerConfig,
    device: B::Device,
}

impl<B: AutodiffBackend> ClassificationTrainer<B>
where
    B::FloatElem: From<f32>,
{
    /// Create a new trainer.
    pub fn new(config: ClassificationTrainerConfig, device: B::Device) -> Self {
        Self { config, device }
    }

    /// Train a classification model using a forward function.
    ///
    /// This is a generic training method that takes a forward function closure
    /// to handle model-specific forward passes.
    pub fn fit_with_forward<M, F, G>(
        &self,
        model: M,
        dls: &TSDataLoaders,
        forward_fn: F,
        valid_forward_fn: G,
    ) -> Result<TrainingOutput<M>>
    where
        M: AutodiffModule<B> + Clone,
        F: Fn(&M, Tensor<B, 3>) -> Tensor<B, 2>,
        G: Fn(&M::InnerModule, Tensor<B::InnerBackend, 3>) -> Tensor<B::InnerBackend, 2>,
    {
        let start_time = Instant::now();

        // Setup optimizer
        let optimizer_config = AdamConfig::new().with_weight_decay(Some(
            burn::optim::decay::WeightDecayConfig::new(self.config.weight_decay),
        ));
        let mut optim = optimizer_config.init::<B, M>();

        // Setup scheduler
        let steps_per_epoch = dls.train().n_batches();
        let total_steps = self.config.n_epochs * steps_per_epoch;
        let scheduler = OneCycleLR::simple(self.config.lr, total_steps);

        // Track best model
        let mut best_model = model.clone();
        let mut best_valid_acc = 0.0f32;
        let mut best_epoch = 0;

        // History
        let mut train_losses = Vec::with_capacity(self.config.n_epochs);
        let mut valid_losses = Vec::with_capacity(self.config.n_epochs);
        let mut valid_accs = Vec::with_capacity(self.config.n_epochs);

        let mut current_model = model;
        let mut global_step = 0;

        // Early stopping state
        let mut epochs_without_improvement = 0;
        let early_stopping_enabled = self.config.early_stopping_patience > 0;

        for epoch in 0..self.config.n_epochs {
            // Training phase
            let train_loss = self.train_epoch(
                &mut current_model,
                &mut optim,
                dls,
                &scheduler,
                &mut global_step,
                &forward_fn,
            )?;
            train_losses.push(train_loss);

            // Validation phase
            let (valid_loss, valid_acc) =
                self.valid_epoch(&current_model, dls, &valid_forward_fn)?;
            valid_losses.push(valid_loss);
            valid_accs.push(valid_acc);

            // Track best and early stopping
            let improved = valid_acc > best_valid_acc + self.config.early_stopping_min_delta;
            if improved {
                best_valid_acc = valid_acc;
                best_epoch = epoch;
                best_model = current_model.clone();
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement += 1;
            }

            if self.config.verbose {
                let marker = if improved { " *" } else { "" };
                println!(
                    "Epoch {:3}/{}: train_loss={:.4}, valid_loss={:.4}, valid_acc={:.2}%{}",
                    epoch + 1,
                    self.config.n_epochs,
                    train_loss,
                    valid_loss,
                    valid_acc * 100.0,
                    marker
                );
            }

            // Check early stopping
            if early_stopping_enabled && epochs_without_improvement >= self.config.early_stopping_patience {
                if self.config.verbose {
                    println!(
                        "\nEarly stopping triggered after {} epochs without improvement",
                        self.config.early_stopping_patience
                    );
                }
                break;
            }
        }

        let training_time_secs = start_time.elapsed().as_secs_f64();

        if self.config.verbose {
            println!("\nTraining complete in {:.1}s", training_time_secs);
            println!(
                "Best validation accuracy: {:.2}% at epoch {}",
                best_valid_acc * 100.0,
                best_epoch + 1
            );
        }

        Ok(TrainingOutput {
            model: best_model,
            train_losses,
            valid_losses,
            valid_accs,
            best_valid_acc,
            best_epoch,
            training_time_secs,
        })
    }

    fn train_epoch<M, O, F>(
        &self,
        model: &mut M,
        optim: &mut O,
        dls: &TSDataLoaders,
        scheduler: &OneCycleLR,
        global_step: &mut usize,
        forward_fn: &F,
    ) -> Result<f32>
    where
        M: AutodiffModule<B> + Clone,
        O: Optimizer<M, B>,
        F: Fn(&M, Tensor<B, 3>) -> Tensor<B, 2>,
    {
        let mut total_loss = 0.0f32;
        let mut n_batches = 0;

        // Create loss function
        let loss_fn = CrossEntropyLossConfig::new().init(&self.device);

        for batch_result in dls.train().iter::<B>(&self.device) {
            let batch = batch_result?;

            // Get learning rate from scheduler
            let lr = scheduler.get_lr(*global_step);

            // Get input tensor
            let x = batch.x.inner().clone();

            // Get targets as class indices
            let y = batch.y.expect("Training requires targets");
            let [batch_size, _] = y.dims();
            let targets: Tensor<B, 1, Int> = y.reshape([batch_size]).int();

            // Forward pass
            let logits = forward_fn(model, x);

            // Cross entropy loss
            let loss = loss_fn.forward(logits, targets).mean();
            let loss_value = loss.clone().into_scalar().elem::<f32>();
            total_loss += loss_value;

            // Backward pass
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, model);

            // Optimizer step with scheduled learning rate
            *model = optim.step(lr, model.clone(), grads);

            n_batches += 1;
            *global_step += 1;
        }

        Ok(total_loss / n_batches as f32)
    }

    fn valid_epoch<M, G>(
        &self,
        model: &M,
        dls: &TSDataLoaders,
        valid_forward_fn: &G,
    ) -> Result<(f32, f32)>
    where
        M: AutodiffModule<B>,
        G: Fn(&M::InnerModule, Tensor<B::InnerBackend, 3>) -> Tensor<B::InnerBackend, 2>,
    {
        let inner_model = model.clone().valid();
        let inner_device: <B::InnerBackend as Backend>::Device = self.device.clone().into();

        // Create loss function for validation
        let loss_fn = CrossEntropyLossConfig::new().init(&inner_device);

        let mut total_loss = 0.0f32;
        let mut correct = 0usize;
        let mut total = 0usize;

        for batch_result in dls.valid().iter::<B::InnerBackend>(&inner_device) {
            let batch = batch_result?;

            // Get input tensor
            let x = batch.x.inner().clone();

            // Get targets
            let y = batch.y.expect("Validation requires targets");
            let [batch_size, _] = y.dims();
            let targets: Tensor<B::InnerBackend, 1, Int> = y.reshape([batch_size]).int();

            // Forward pass (no gradients)
            let logits = valid_forward_fn(&inner_model, x);

            // Cross entropy loss
            let loss = loss_fn.forward(logits.clone(), targets.clone()).mean();
            total_loss += loss.into_scalar().elem::<f32>();

            // Accuracy
            let preds = logits.argmax(1).squeeze(1);
            let correct_batch: i32 = preds.equal(targets).int().sum().into_scalar().elem();
            correct += correct_batch as usize;
            total += batch_size;
        }

        let n_batches = dls.valid().n_batches();
        let avg_loss = if n_batches > 0 {
            total_loss / n_batches as f32
        } else {
            0.0
        };
        let accuracy = if total > 0 {
            correct as f32 / total as f32
        } else {
            0.0
        };

        Ok((avg_loss, accuracy))
    }
}

/// Convenience function to train a classification model.
pub fn train_classification<B, M, F, G>(
    model: M,
    dls: &TSDataLoaders,
    n_epochs: usize,
    lr: f64,
    device: &B::Device,
    forward_fn: F,
    valid_forward_fn: G,
) -> Result<TrainingOutput<M>>
where
    B: AutodiffBackend,
    B::FloatElem: From<f32>,
    M: AutodiffModule<B> + Clone,
    F: Fn(&M, Tensor<B, 3>) -> Tensor<B, 2>,
    G: Fn(&M::InnerModule, Tensor<B::InnerBackend, 3>) -> Tensor<B::InnerBackend, 2>,
{
    let config = ClassificationTrainerConfig {
        n_epochs,
        lr,
        ..Default::default()
    };
    let trainer = ClassificationTrainer::<B>::new(config, device.clone());
    trainer.fit_with_forward(model, dls, forward_fn, valid_forward_fn)
}

/// Configuration for regression training.
#[derive(Debug, Clone)]
pub struct RegressionTrainerConfig {
    /// Number of epochs.
    pub n_epochs: usize,
    /// Learning rate.
    pub lr: f64,
    /// Weight decay.
    pub weight_decay: f32,
    /// Gradient clipping value (0 = disabled).
    pub grad_clip: f32,
    /// Whether to print progress.
    pub verbose: bool,
    /// Early stopping patience (0 = disabled).
    pub early_stopping_patience: usize,
    /// Minimum delta for early stopping improvement.
    pub early_stopping_min_delta: f32,
}

impl Default for RegressionTrainerConfig {
    fn default() -> Self {
        Self {
            n_epochs: 25,
            lr: 1e-3,
            weight_decay: 0.01,
            grad_clip: 1.0,
            verbose: true,
            early_stopping_patience: 0,
            early_stopping_min_delta: 0.0001,
        }
    }
}

/// Regression training output with metrics and final model.
#[derive(Debug)]
pub struct RegressionOutput<M> {
    /// Trained model.
    pub model: M,
    /// Training losses per epoch.
    pub train_losses: Vec<f32>,
    /// Validation losses per epoch.
    pub valid_losses: Vec<f32>,
    /// Best validation loss.
    pub best_valid_loss: f32,
    /// Best epoch.
    pub best_epoch: usize,
    /// Total training time in seconds.
    pub training_time_secs: f64,
}

/// Trainer for regression models.
pub struct RegressionTrainer<B: AutodiffBackend> {
    config: RegressionTrainerConfig,
    device: B::Device,
}

impl<B: AutodiffBackend> RegressionTrainer<B>
where
    B::FloatElem: From<f32>,
{
    /// Create a new trainer.
    pub fn new(config: RegressionTrainerConfig, device: B::Device) -> Self {
        Self { config, device }
    }

    /// Train a regression model using a forward function.
    ///
    /// This is a generic training method that takes a forward function closure
    /// to handle model-specific forward passes.
    pub fn fit_with_forward<M, F, G>(
        &self,
        model: M,
        dls: &TSDataLoaders,
        forward_fn: F,
        valid_forward_fn: G,
    ) -> Result<RegressionOutput<M>>
    where
        M: AutodiffModule<B> + Clone,
        F: Fn(&M, Tensor<B, 3>) -> Tensor<B, 2>,
        G: Fn(&M::InnerModule, Tensor<B::InnerBackend, 3>) -> Tensor<B::InnerBackend, 2>,
    {
        let start_time = Instant::now();

        // Setup optimizer
        let optimizer_config = AdamConfig::new().with_weight_decay(Some(
            burn::optim::decay::WeightDecayConfig::new(self.config.weight_decay),
        ));
        let mut optim = optimizer_config.init::<B, M>();

        // Setup scheduler
        let steps_per_epoch = dls.train().n_batches();
        let total_steps = self.config.n_epochs * steps_per_epoch;
        let scheduler = OneCycleLR::simple(self.config.lr, total_steps);

        // Track best model
        let mut best_model = model.clone();
        let mut best_valid_loss = f32::INFINITY;
        let mut best_epoch = 0;

        // History
        let mut train_losses = Vec::with_capacity(self.config.n_epochs);
        let mut valid_losses = Vec::with_capacity(self.config.n_epochs);

        let mut current_model = model;
        let mut global_step = 0;

        // Early stopping state
        let mut epochs_without_improvement = 0;
        let early_stopping_enabled = self.config.early_stopping_patience > 0;

        for epoch in 0..self.config.n_epochs {
            // Training phase
            let train_loss = self.train_epoch(
                &mut current_model,
                &mut optim,
                dls,
                &scheduler,
                &mut global_step,
                &forward_fn,
            )?;
            train_losses.push(train_loss);

            // Validation phase
            let valid_loss = self.valid_epoch(&current_model, dls, &valid_forward_fn)?;
            valid_losses.push(valid_loss);

            // Track best and early stopping
            let improved = valid_loss < best_valid_loss - self.config.early_stopping_min_delta;
            if improved {
                best_valid_loss = valid_loss;
                best_epoch = epoch;
                best_model = current_model.clone();
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement += 1;
            }

            if self.config.verbose {
                let marker = if improved { " *" } else { "" };
                println!(
                    "Epoch {:3}/{}: train_loss={:.6}, valid_loss={:.6}{}",
                    epoch + 1,
                    self.config.n_epochs,
                    train_loss,
                    valid_loss,
                    marker
                );
            }

            // Check early stopping
            if early_stopping_enabled && epochs_without_improvement >= self.config.early_stopping_patience {
                if self.config.verbose {
                    println!(
                        "\nEarly stopping triggered after {} epochs without improvement",
                        self.config.early_stopping_patience
                    );
                }
                break;
            }
        }

        let training_time_secs = start_time.elapsed().as_secs_f64();

        if self.config.verbose {
            println!("\nTraining complete in {:.1}s", training_time_secs);
            println!(
                "Best validation loss: {:.6} at epoch {}",
                best_valid_loss,
                best_epoch + 1
            );
        }

        Ok(RegressionOutput {
            model: best_model,
            train_losses,
            valid_losses,
            best_valid_loss,
            best_epoch,
            training_time_secs,
        })
    }

    fn train_epoch<M, O, F>(
        &self,
        model: &mut M,
        optim: &mut O,
        dls: &TSDataLoaders,
        scheduler: &OneCycleLR,
        global_step: &mut usize,
        forward_fn: &F,
    ) -> Result<f32>
    where
        M: AutodiffModule<B> + Clone,
        O: Optimizer<M, B>,
        F: Fn(&M, Tensor<B, 3>) -> Tensor<B, 2>,
    {
        let mut total_loss = 0.0f32;
        let mut n_batches = 0;

        for batch_result in dls.train().iter::<B>(&self.device) {
            let batch = batch_result?;

            // Get learning rate from scheduler
            let lr = scheduler.get_lr(*global_step);

            // Get input tensor
            let x = batch.x.inner().clone();

            // Get targets
            let y = batch.y.expect("Training requires targets");

            // Forward pass
            let preds = forward_fn(model, x);

            // MSE loss
            let diff = preds - y;
            let loss = (diff.clone() * diff).mean();
            let loss_value = loss.clone().into_scalar().elem::<f32>();
            total_loss += loss_value;

            // Backward pass
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, model);

            // Optimizer step with scheduled learning rate
            *model = optim.step(lr, model.clone(), grads);

            n_batches += 1;
            *global_step += 1;
        }

        Ok(total_loss / n_batches as f32)
    }

    fn valid_epoch<M, G>(
        &self,
        model: &M,
        dls: &TSDataLoaders,
        valid_forward_fn: &G,
    ) -> Result<f32>
    where
        M: AutodiffModule<B>,
        G: Fn(&M::InnerModule, Tensor<B::InnerBackend, 3>) -> Tensor<B::InnerBackend, 2>,
    {
        let inner_model = model.clone().valid();
        let inner_device: <B::InnerBackend as Backend>::Device = self.device.clone().into();

        let mut total_loss = 0.0f32;
        let mut n_batches = 0;

        for batch_result in dls.valid().iter::<B::InnerBackend>(&inner_device) {
            let batch = batch_result?;

            // Get input tensor
            let x = batch.x.inner().clone();

            // Get targets
            let y = batch.y.expect("Validation requires targets");

            // Forward pass (no gradients)
            let preds = valid_forward_fn(&inner_model, x);

            // MSE loss
            let diff = preds - y;
            let loss = (diff.clone() * diff).mean();
            total_loss += loss.into_scalar().elem::<f32>();
            n_batches += 1;
        }

        Ok(if n_batches > 0 {
            total_loss / n_batches as f32
        } else {
            0.0
        })
    }
}

/// Convenience function to train a regression model.
pub fn train_regression<B, M, F, G>(
    model: M,
    dls: &TSDataLoaders,
    n_epochs: usize,
    lr: f64,
    device: &B::Device,
    forward_fn: F,
    valid_forward_fn: G,
) -> Result<RegressionOutput<M>>
where
    B: AutodiffBackend,
    B::FloatElem: From<f32>,
    M: AutodiffModule<B> + Clone,
    F: Fn(&M, Tensor<B, 3>) -> Tensor<B, 2>,
    G: Fn(&M::InnerModule, Tensor<B::InnerBackend, 3>) -> Tensor<B::InnerBackend, 2>,
{
    let config = RegressionTrainerConfig {
        n_epochs,
        lr,
        ..Default::default()
    };
    let trainer = RegressionTrainer::<B>::new(config, device.clone());
    trainer.fit_with_forward(model, dls, forward_fn, valid_forward_fn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_config_default() {
        let config = ClassificationTrainerConfig::default();
        assert_eq!(config.n_epochs, 25);
        assert_eq!(config.lr, 1e-3);
    }

    #[test]
    fn test_regression_trainer_config_default() {
        let config = RegressionTrainerConfig::default();
        assert_eq!(config.n_epochs, 25);
        assert_eq!(config.lr, 1e-3);
    }
}
