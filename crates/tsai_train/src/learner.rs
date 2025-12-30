//! Learner for managing training.

use std::marker::PhantomData;

use burn::module::AutodiffModule;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use serde::{Deserialize, Serialize};

use crate::callback::{CallbackContext, CallbackList, ProgressCallback};
use crate::error::Result;
use crate::metrics::{Accuracy, Metric};
use crate::training::TrainingOutput;
use tsai_data::TSDataLoaders;

/// Configuration for the Learner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnerConfig {
    /// Learning rate.
    pub lr: f64,
    /// Weight decay.
    pub weight_decay: f64,
    /// Gradient clipping value (if > 0).
    pub grad_clip: f64,
    /// Whether to use mixed precision (if supported).
    pub mixed_precision: bool,
}

impl Default for LearnerConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            weight_decay: 0.01,
            grad_clip: 0.0,
            mixed_precision: false,
        }
    }
}

/// Training state for checkpointing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingState {
    /// Current epoch.
    pub epoch: usize,
    /// Current step.
    pub step: usize,
    /// Best validation loss.
    pub best_valid_loss: f32,
    /// Training history.
    pub history: TrainingHistory,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            epoch: 0,
            step: 0,
            best_valid_loss: f32::INFINITY,
            history: TrainingHistory::default(),
        }
    }
}

/// Training history.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingHistory {
    /// Training losses per epoch.
    pub train_losses: Vec<f32>,
    /// Validation losses per epoch.
    pub valid_losses: Vec<f32>,
    /// Metrics per epoch.
    pub metrics: Vec<std::collections::HashMap<String, f32>>,
    /// Learning rates per epoch.
    pub lrs: Vec<f64>,
}

/// Learner manages model training.
///
/// # Example
///
/// ```rust,ignore
/// let learner = Learner::new(model, dls, config, &device);
/// learner.fit(10)?;
/// ```
pub struct Learner<B, M>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    /// The model being trained.
    model: M,
    /// Dataloaders.
    dls: TSDataLoaders,
    /// Configuration.
    config: LearnerConfig,
    /// Training state.
    state: TrainingState,
    /// Callbacks.
    callbacks: CallbackList,
    /// Metrics.
    metrics: Vec<Box<dyn Metric<B>>>,
    /// Device.
    device: B::Device,
    /// Phantom data for backend.
    _backend: PhantomData<B>,
}

impl<B, M> Learner<B, M>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    /// Create a new Learner.
    pub fn new(model: M, dls: TSDataLoaders, config: LearnerConfig, device: &B::Device) -> Self {
        let mut callbacks = CallbackList::new();
        callbacks.add(ProgressCallback::new(false));

        let metrics: Vec<Box<dyn Metric<B>>> = vec![Box::new(Accuracy)];

        Self {
            model,
            dls,
            config,
            state: TrainingState::default(),
            callbacks,
            metrics,
            device: device.clone(),
            _backend: PhantomData,
        }
    }

    /// Add a callback.
    pub fn add_callback<C: crate::callback::Callback + 'static>(mut self, callback: C) -> Self {
        self.callbacks.add(callback);
        self
    }

    /// Add a metric.
    pub fn add_metric<M2: Metric<B> + 'static>(mut self, metric: M2) -> Self {
        self.metrics.push(Box::new(metric));
        self
    }

    /// Get the model.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get mutable model reference.
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Get the training state.
    pub fn state(&self) -> &TrainingState {
        &self.state
    }

    /// Get the training history.
    pub fn history(&self) -> &TrainingHistory {
        &self.state.history
    }

    /// Get the device.
    pub fn device(&self) -> &B::Device {
        &self.device
    }

    /// Fit the model using one-cycle learning rate policy.
    ///
    /// This method requires forward function closures since different models
    /// have different forward signatures.
    ///
    /// # Arguments
    ///
    /// * `n_epochs` - Number of training epochs
    /// * `forward_fn` - Forward function for training (with gradients)
    /// * `valid_forward_fn` - Forward function for validation (without gradients)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let output = learner.fit_one_cycle(
    ///     25,
    ///     |model, x| model.forward(x),
    ///     |model, x| model.forward(x),
    /// )?;
    /// ```
    pub fn fit_one_cycle<F, G>(
        mut self,
        n_epochs: usize,
        forward_fn: F,
        valid_forward_fn: G,
    ) -> Result<TrainingOutput<M>>
    where
        M: Clone,
        B::FloatElem: From<f32>,
        F: Fn(&M, Tensor<B, 3>) -> Tensor<B, 2>,
        G: Fn(&M::InnerModule, Tensor<B::InnerBackend, 3>) -> Tensor<B::InnerBackend, 2>,
    {
        use crate::training::{ClassificationTrainer, ClassificationTrainerConfig};

        // Create callback context
        let mut ctx = CallbackContext::new(n_epochs, self.dls.train().n_batches());

        // Notify callbacks of training start
        let _ = self.callbacks.before_fit(&mut ctx);

        // Configure trainer
        let trainer_config = ClassificationTrainerConfig {
            n_epochs,
            lr: self.config.lr,
            weight_decay: self.config.weight_decay as f32,
            grad_clip: self.config.grad_clip as f32,
            verbose: true,
            early_stopping_patience: 0,
            early_stopping_min_delta: 0.001,
        };

        let trainer = ClassificationTrainer::<B>::new(trainer_config, self.device.clone());

        // Run training
        let output = trainer.fit_with_forward(
            self.model,
            &self.dls,
            forward_fn,
            valid_forward_fn,
        )?;

        // Update state
        self.state.history.train_losses = output.train_losses.clone();
        self.state.history.valid_losses = output.valid_losses.clone();
        self.state.best_valid_loss = output.valid_losses.last().copied().unwrap_or(f32::INFINITY);

        // Notify callbacks of training end
        let _ = self.callbacks.after_fit(&mut ctx);

        Ok(output)
    }

    /// Fit the model with early stopping.
    ///
    /// # Arguments
    ///
    /// * `n_epochs` - Maximum number of training epochs
    /// * `patience` - Number of epochs without improvement before stopping
    /// * `forward_fn` - Forward function for training
    /// * `valid_forward_fn` - Forward function for validation
    pub fn fit_with_early_stopping<F, G>(
        self,
        n_epochs: usize,
        patience: usize,
        forward_fn: F,
        valid_forward_fn: G,
    ) -> Result<TrainingOutput<M>>
    where
        M: Clone,
        B::FloatElem: From<f32>,
        F: Fn(&M, Tensor<B, 3>) -> Tensor<B, 2>,
        G: Fn(&M::InnerModule, Tensor<B::InnerBackend, 3>) -> Tensor<B::InnerBackend, 2>,
    {
        use crate::training::{ClassificationTrainer, ClassificationTrainerConfig};

        let trainer_config = ClassificationTrainerConfig {
            n_epochs,
            lr: self.config.lr,
            weight_decay: self.config.weight_decay as f32,
            grad_clip: self.config.grad_clip as f32,
            verbose: true,
            early_stopping_patience: patience,
            early_stopping_min_delta: 0.001,
        };

        let trainer = ClassificationTrainer::<B>::new(trainer_config, self.device.clone());

        trainer.fit_with_forward(
            self.model,
            &self.dls,
            forward_fn,
            valid_forward_fn,
        )
    }

    /// Get predictions on the validation set.
    ///
    /// # Arguments
    ///
    /// * `forward_fn` - Forward function for inference
    pub fn get_preds<G>(&self, forward_fn: G) -> Result<Predictions<B::InnerBackend>>
    where
        G: Fn(&M::InnerModule, Tensor<B::InnerBackend, 3>) -> Tensor<B::InnerBackend, 2>,
    {
        let inner_model = self.model.clone().valid();
        let inner_device: <B::InnerBackend as Backend>::Device = self.device.clone().into();

        let mut all_preds: Vec<Tensor<B::InnerBackend, 2>> = Vec::new();
        let mut all_targets: Vec<Tensor<B::InnerBackend, 2>> = Vec::new();

        for batch_result in self.dls.valid().iter::<B::InnerBackend>(&inner_device) {
            let batch = batch_result?;
            let x = batch.x.inner().clone();
            let logits = forward_fn(&inner_model, x);
            all_preds.push(logits);

            if let Some(y) = batch.y {
                all_targets.push(y);
            }
        }

        // Concatenate all predictions
        let preds = Tensor::cat(all_preds, 0);
        let mut predictions = Predictions::new(preds);

        if !all_targets.is_empty() {
            let targets = Tensor::cat(all_targets, 0);
            predictions = predictions.with_targets(targets);
        }

        Ok(predictions)
    }
}

/// Predictions output from get_X_preds.
#[derive(Debug)]
pub struct Predictions<B: Backend> {
    /// Optional input data.
    pub x: Option<Tensor<B, 3>>,
    /// Predictions/probabilities.
    pub preds: Tensor<B, 2>,
    /// Optional targets.
    pub targets: Option<Tensor<B, 2>>,
    /// Decoded predictions (class indices for classification).
    pub decoded: Tensor<B, 1, Int>,
    /// Optional per-sample loss.
    pub losses: Option<Tensor<B, 1>>,
}

impl<B: Backend> Predictions<B> {
    /// Create new predictions.
    pub fn new(preds: Tensor<B, 2>) -> Self {
        let decoded = preds.clone().argmax(1).squeeze(1);
        Self {
            x: None,
            preds,
            targets: None,
            decoded,
            losses: None,
        }
    }

    /// Add inputs.
    pub fn with_x(mut self, x: Tensor<B, 3>) -> Self {
        self.x = Some(x);
        self
    }

    /// Add targets.
    pub fn with_targets(mut self, targets: Tensor<B, 2>) -> Self {
        self.targets = Some(targets);
        self
    }

    /// Add per-sample losses.
    pub fn with_losses(mut self, losses: Tensor<B, 1>) -> Self {
        self.losses = Some(losses);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learner_config_default() {
        let config = LearnerConfig::default();
        assert_eq!(config.lr, 1e-3);
        assert_eq!(config.weight_decay, 0.01);
    }

    #[test]
    fn test_training_state_default() {
        let state = TrainingState::default();
        assert_eq!(state.epoch, 0);
        assert_eq!(state.step, 0);
    }
}
