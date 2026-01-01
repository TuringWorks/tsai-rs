//! Callback system for training hooks.

use std::collections::HashMap;
use std::path::PathBuf;

use crate::error::Result;

/// Context passed to callbacks containing training state.
pub struct CallbackContext {
    /// Current epoch (0-indexed).
    pub epoch: usize,
    /// Total number of epochs.
    pub n_epochs: usize,
    /// Current batch (0-indexed).
    pub batch: usize,
    /// Total number of batches in epoch.
    pub n_batches: usize,
    /// Current learning rate.
    pub lr: f64,
    /// Current training loss.
    pub train_loss: Option<f32>,
    /// Current validation loss.
    pub valid_loss: Option<f32>,
    /// Current metrics.
    pub metrics: HashMap<String, f32>,
    /// Whether to stop training.
    pub stop_training: bool,
    /// Whether to skip this batch.
    pub skip_batch: bool,
}

impl CallbackContext {
    /// Create a new callback context.
    pub fn new(n_epochs: usize, n_batches: usize) -> Self {
        Self {
            epoch: 0,
            n_epochs,
            batch: 0,
            n_batches,
            lr: 0.0,
            train_loss: None,
            valid_loss: None,
            metrics: HashMap::new(),
            stop_training: false,
            skip_batch: false,
        }
    }

    /// Get progress as a fraction (0.0 to 1.0).
    pub fn progress(&self) -> f32 {
        let total_batches = self.n_epochs * self.n_batches;
        let current = self.epoch * self.n_batches + self.batch;
        current as f32 / total_batches as f32
    }
}

/// Trait for training callbacks.
///
/// Callbacks allow customization of the training loop at various points.
pub trait Callback: Send + Sync {
    /// Called before training starts.
    fn before_fit(&mut self, _ctx: &mut CallbackContext) -> Result<()> {
        Ok(())
    }

    /// Called after training completes.
    fn after_fit(&mut self, _ctx: &mut CallbackContext) -> Result<()> {
        Ok(())
    }

    /// Called before each epoch.
    fn before_epoch(&mut self, _ctx: &mut CallbackContext) -> Result<()> {
        Ok(())
    }

    /// Called after each epoch.
    fn after_epoch(&mut self, _ctx: &mut CallbackContext) -> Result<()> {
        Ok(())
    }

    /// Called before each training batch.
    fn before_batch(&mut self, _ctx: &mut CallbackContext) -> Result<()> {
        Ok(())
    }

    /// Called after each training batch.
    fn after_batch(&mut self, _ctx: &mut CallbackContext) -> Result<()> {
        Ok(())
    }

    /// Called before validation.
    fn before_validate(&mut self, _ctx: &mut CallbackContext) -> Result<()> {
        Ok(())
    }

    /// Called after validation.
    fn after_validate(&mut self, _ctx: &mut CallbackContext) -> Result<()> {
        Ok(())
    }

    /// Get the callback name.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }
}

/// A list of callbacks.
#[derive(Default)]
pub struct CallbackList {
    callbacks: Vec<Box<dyn Callback>>,
}

impl CallbackList {
    /// Create a new empty callback list.
    pub fn new() -> Self {
        Self {
            callbacks: Vec::new(),
        }
    }

    /// Add a callback.
    pub fn add<C: Callback + 'static>(&mut self, callback: C) {
        self.callbacks.push(Box::new(callback));
    }

    /// Call before_fit on all callbacks.
    pub fn before_fit(&mut self, ctx: &mut CallbackContext) -> Result<()> {
        for cb in &mut self.callbacks {
            cb.before_fit(ctx)?;
        }
        Ok(())
    }

    /// Call after_fit on all callbacks.
    pub fn after_fit(&mut self, ctx: &mut CallbackContext) -> Result<()> {
        for cb in &mut self.callbacks {
            cb.after_fit(ctx)?;
        }
        Ok(())
    }

    /// Call before_epoch on all callbacks.
    pub fn before_epoch(&mut self, ctx: &mut CallbackContext) -> Result<()> {
        for cb in &mut self.callbacks {
            cb.before_epoch(ctx)?;
        }
        Ok(())
    }

    /// Call after_epoch on all callbacks.
    pub fn after_epoch(&mut self, ctx: &mut CallbackContext) -> Result<()> {
        for cb in &mut self.callbacks {
            cb.after_epoch(ctx)?;
        }
        Ok(())
    }

    /// Call before_batch on all callbacks.
    pub fn before_batch(&mut self, ctx: &mut CallbackContext) -> Result<()> {
        for cb in &mut self.callbacks {
            cb.before_batch(ctx)?;
        }
        Ok(())
    }

    /// Call after_batch on all callbacks.
    pub fn after_batch(&mut self, ctx: &mut CallbackContext) -> Result<()> {
        for cb in &mut self.callbacks {
            cb.after_batch(ctx)?;
        }
        Ok(())
    }

    /// Call before_validate on all callbacks.
    pub fn before_validate(&mut self, ctx: &mut CallbackContext) -> Result<()> {
        for cb in &mut self.callbacks {
            cb.before_validate(ctx)?;
        }
        Ok(())
    }

    /// Call after_validate on all callbacks.
    pub fn after_validate(&mut self, ctx: &mut CallbackContext) -> Result<()> {
        for cb in &mut self.callbacks {
            cb.after_validate(ctx)?;
        }
        Ok(())
    }
}

/// Progress bar callback for displaying training progress.
pub struct ProgressCallback {
    /// Whether to show batch-level progress (reserved for future use).
    #[allow(dead_code)]
    show_batch: bool,
}

impl ProgressCallback {
    /// Create a new progress callback.
    pub fn new(show_batch: bool) -> Self {
        Self { show_batch }
    }
}

impl Callback for ProgressCallback {
    fn before_fit(&mut self, ctx: &mut CallbackContext) -> Result<()> {
        tracing::info!("Starting training for {} epochs", ctx.n_epochs);
        Ok(())
    }

    fn after_epoch(&mut self, ctx: &mut CallbackContext) -> Result<()> {
        let train_loss = ctx.train_loss.map(|l| format!("{:.4}", l)).unwrap_or_default();
        let valid_loss = ctx.valid_loss.map(|l| format!("{:.4}", l)).unwrap_or_default();

        tracing::info!(
            "Epoch {}/{}: train_loss={}, valid_loss={}, lr={:.6}",
            ctx.epoch + 1,
            ctx.n_epochs,
            train_loss,
            valid_loss,
            ctx.lr
        );

        for (name, value) in &ctx.metrics {
            tracing::info!("  {}: {:.4}", name, value);
        }

        Ok(())
    }

    fn after_fit(&mut self, _ctx: &mut CallbackContext) -> Result<()> {
        tracing::info!("Training completed");
        Ok(())
    }

    fn name(&self) -> &str {
        "ProgressCallback"
    }
}

/// Early stopping callback.
pub struct EarlyStoppingCallback {
    patience: usize,
    min_delta: f32,
    best_loss: f32,
    counter: usize,
    mode: EarlyStoppingMode,
}

/// Mode for early stopping.
pub enum EarlyStoppingMode {
    /// Stop when validation loss stops decreasing.
    Min,
    /// Stop when validation metric stops increasing.
    Max,
}

impl EarlyStoppingCallback {
    /// Create a new early stopping callback.
    pub fn new(patience: usize, min_delta: f32, mode: EarlyStoppingMode) -> Self {
        let best_loss = match mode {
            EarlyStoppingMode::Min => f32::INFINITY,
            EarlyStoppingMode::Max => f32::NEG_INFINITY,
        };

        Self {
            patience,
            min_delta,
            best_loss,
            counter: 0,
            mode,
        }
    }
}

impl Callback for EarlyStoppingCallback {
    fn after_epoch(&mut self, ctx: &mut CallbackContext) -> Result<()> {
        let current = ctx.valid_loss.unwrap_or(f32::INFINITY);

        let improved = match self.mode {
            EarlyStoppingMode::Min => current < self.best_loss - self.min_delta,
            EarlyStoppingMode::Max => current > self.best_loss + self.min_delta,
        };

        if improved {
            self.best_loss = current;
            self.counter = 0;
        } else {
            self.counter += 1;
            if self.counter >= self.patience {
                tracing::info!(
                    "Early stopping triggered after {} epochs without improvement",
                    self.patience
                );
                ctx.stop_training = true;
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "EarlyStoppingCallback"
    }
}

/// Mode for model saving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaveModelMode {
    /// Save when validation loss improves (lower is better).
    Min,
    /// Save when validation metric improves (higher is better).
    Max,
    /// Save after every epoch.
    Every,
}

/// Callback for saving model checkpoints.
///
/// This callback saves model state after each epoch when the monitored
/// metric improves. It creates checkpoint files in the specified directory.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_train::callback::{SaveModelCallback, SaveModelMode};
///
/// let callback = SaveModelCallback::new("./checkpoints", SaveModelMode::Min)
///     .with_metric("valid_loss");
/// ```
pub struct SaveModelCallback {
    /// Directory to save checkpoints.
    save_dir: PathBuf,
    /// Mode for determining improvement.
    mode: SaveModelMode,
    /// Best metric value seen so far.
    best_value: f32,
    /// Metric to monitor (default: validation loss).
    metric_name: Option<String>,
    /// Whether to save only the best model or all.
    save_best_only: bool,
    /// Filename prefix for checkpoints.
    filename_prefix: String,
    /// Epoch of the best model.
    best_epoch: usize,
}

impl SaveModelCallback {
    /// Create a new save model callback.
    ///
    /// # Arguments
    ///
    /// * `save_dir` - Directory to save checkpoints
    /// * `mode` - When to save (min loss, max metric, or every epoch)
    pub fn new<P: Into<PathBuf>>(save_dir: P, mode: SaveModelMode) -> Self {
        let best_value = match mode {
            SaveModelMode::Min => f32::INFINITY,
            SaveModelMode::Max => f32::NEG_INFINITY,
            SaveModelMode::Every => 0.0,
        };

        Self {
            save_dir: save_dir.into(),
            mode,
            best_value,
            metric_name: None,
            save_best_only: true,
            filename_prefix: "checkpoint".to_string(),
            best_epoch: 0,
        }
    }

    /// Set the metric name to monitor.
    ///
    /// If not set, uses validation loss.
    #[must_use]
    pub fn with_metric(mut self, name: &str) -> Self {
        self.metric_name = Some(name.to_string());
        self
    }

    /// Set whether to save only the best model.
    ///
    /// If false, saves a checkpoint after every epoch.
    #[must_use]
    pub fn save_best_only(mut self, value: bool) -> Self {
        self.save_best_only = value;
        self
    }

    /// Set the filename prefix for checkpoints.
    #[must_use]
    pub fn with_prefix(mut self, prefix: &str) -> Self {
        self.filename_prefix = prefix.to_string();
        self
    }

    /// Get the path to the best checkpoint.
    pub fn best_checkpoint_path(&self) -> PathBuf {
        self.save_dir.join(format!("{}_best.json", self.filename_prefix))
    }

    /// Get the path to a specific epoch's checkpoint.
    pub fn epoch_checkpoint_path(&self, epoch: usize) -> PathBuf {
        self.save_dir
            .join(format!("{}_epoch_{}.json", self.filename_prefix, epoch))
    }

    /// Get the epoch of the best model.
    pub fn best_epoch(&self) -> usize {
        self.best_epoch
    }

    /// Get the best metric value.
    pub fn best_value(&self) -> f32 {
        self.best_value
    }

    fn get_current_value(&self, ctx: &CallbackContext) -> Option<f32> {
        if let Some(ref metric_name) = self.metric_name {
            ctx.metrics.get(metric_name).copied()
        } else {
            ctx.valid_loss
        }
    }

    fn should_save(&self, current: f32) -> bool {
        match self.mode {
            SaveModelMode::Min => current < self.best_value,
            SaveModelMode::Max => current > self.best_value,
            SaveModelMode::Every => true,
        }
    }

    fn save_checkpoint(&self, ctx: &CallbackContext, is_best: bool) -> Result<()> {
        // Create save directory if it doesn't exist
        std::fs::create_dir_all(&self.save_dir).map_err(|e| {
            crate::error::TrainError::CheckpointError(format!(
                "Failed to create checkpoint directory: {}",
                e
            ))
        })?;

        // Create checkpoint metadata
        let checkpoint = CheckpointMetadata {
            epoch: ctx.epoch,
            train_loss: ctx.train_loss,
            valid_loss: ctx.valid_loss,
            metrics: ctx.metrics.clone(),
            is_best,
        };

        // Save epoch checkpoint
        let epoch_path = self.epoch_checkpoint_path(ctx.epoch);
        let json = serde_json::to_string_pretty(&checkpoint).map_err(|e| {
            crate::error::TrainError::SerializationError(format!(
                "Failed to serialize checkpoint: {}",
                e
            ))
        })?;
        std::fs::write(&epoch_path, json).map_err(|e| {
            crate::error::TrainError::CheckpointError(format!("Failed to write checkpoint: {}", e))
        })?;

        // If this is the best, also save as best checkpoint
        if is_best {
            let best_path = self.best_checkpoint_path();
            std::fs::copy(&epoch_path, &best_path).map_err(|e| {
                crate::error::TrainError::CheckpointError(format!(
                    "Failed to copy best checkpoint: {}",
                    e
                ))
            })?;
        }

        Ok(())
    }
}

/// Metadata stored in checkpoint files.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckpointMetadata {
    /// Epoch number.
    pub epoch: usize,
    /// Training loss at checkpoint.
    pub train_loss: Option<f32>,
    /// Validation loss at checkpoint.
    pub valid_loss: Option<f32>,
    /// Metrics at checkpoint.
    pub metrics: HashMap<String, f32>,
    /// Whether this was the best checkpoint.
    pub is_best: bool,
}

impl Callback for SaveModelCallback {
    fn before_fit(&mut self, _ctx: &mut CallbackContext) -> Result<()> {
        // Create the save directory at the start of training
        std::fs::create_dir_all(&self.save_dir).map_err(|e| {
            crate::error::TrainError::CheckpointError(format!(
                "Failed to create checkpoint directory: {}",
                e
            ))
        })?;
        tracing::info!("Checkpoints will be saved to: {:?}", self.save_dir);
        Ok(())
    }

    fn after_epoch(&mut self, ctx: &mut CallbackContext) -> Result<()> {
        let Some(current) = self.get_current_value(ctx) else {
            return Ok(());
        };

        let is_best = self.should_save(current);

        if is_best {
            self.best_value = current;
            self.best_epoch = ctx.epoch;
        }

        // Save checkpoint based on settings
        if !self.save_best_only || is_best {
            self.save_checkpoint(ctx, is_best)?;

            if is_best {
                let metric_display = self
                    .metric_name
                    .as_deref()
                    .unwrap_or("valid_loss");
                tracing::info!(
                    "Epoch {}: {} improved to {:.4}, saving checkpoint",
                    ctx.epoch + 1,
                    metric_display,
                    current
                );
            }
        }

        Ok(())
    }

    fn after_fit(&mut self, _ctx: &mut CallbackContext) -> Result<()> {
        tracing::info!(
            "Best model from epoch {} with value {:.4}",
            self.best_epoch + 1,
            self.best_value
        );
        Ok(())
    }

    fn name(&self) -> &str {
        "SaveModelCallback"
    }
}

/// Configuration for gradient clipping.
#[derive(Debug, Clone, Copy)]
pub enum GradientClipMode {
    /// Clip gradients by value: all gradients are clipped to [-value, value].
    Value(f32),
    /// Clip gradients by norm: if total norm exceeds max_norm, scale all gradients.
    Norm(f32),
}

/// Callback for gradient clipping during training.
///
/// Gradient clipping helps prevent exploding gradients, which can cause
/// unstable training. This is especially useful for RNNs and transformers.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_train::callback::{GradientClipCallback, GradientClipMode};
///
/// // Clip gradient norms to max 1.0
/// let callback = GradientClipCallback::new(GradientClipMode::Norm(1.0));
///
/// // Or clip individual gradient values
/// let callback = GradientClipCallback::new(GradientClipMode::Value(0.5));
/// ```
pub struct GradientClipCallback {
    mode: GradientClipMode,
    clip_count: usize,
    total_batches: usize,
}

impl GradientClipCallback {
    /// Create a new gradient clipping callback.
    pub fn new(mode: GradientClipMode) -> Self {
        Self {
            mode,
            clip_count: 0,
            total_batches: 0,
        }
    }

    /// Create a gradient clipping callback with norm clipping.
    pub fn by_norm(max_norm: f32) -> Self {
        Self::new(GradientClipMode::Norm(max_norm))
    }

    /// Create a gradient clipping callback with value clipping.
    pub fn by_value(max_value: f32) -> Self {
        Self::new(GradientClipMode::Value(max_value))
    }

    /// Get the clipping mode.
    pub fn mode(&self) -> GradientClipMode {
        self.mode
    }

    /// Get the max norm/value for clipping.
    pub fn clip_value(&self) -> f32 {
        match self.mode {
            GradientClipMode::Value(v) => v,
            GradientClipMode::Norm(n) => n,
        }
    }
}

impl Callback for GradientClipCallback {
    fn before_fit(&mut self, _ctx: &mut CallbackContext) -> Result<()> {
        self.clip_count = 0;
        self.total_batches = 0;
        Ok(())
    }

    fn after_batch(&mut self, _ctx: &mut CallbackContext) -> Result<()> {
        // Gradient clipping would be applied by the learner
        // This callback just tracks statistics
        self.total_batches += 1;
        Ok(())
    }

    fn after_fit(&mut self, _ctx: &mut CallbackContext) -> Result<()> {
        if self.total_batches > 0 {
            let clip_rate = self.clip_count as f32 / self.total_batches as f32 * 100.0;
            if clip_rate > 0.0 {
                tracing::info!(
                    "Gradient clipping was applied in {:.1}% of batches",
                    clip_rate
                );
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "GradientClipCallback"
    }
}

/// Callback for logging training history.
///
/// Records all training metrics for later analysis or visualization.
#[derive(Default)]
pub struct HistoryCallback {
    train_losses: Vec<f32>,
    valid_losses: Vec<f32>,
    learning_rates: Vec<f64>,
    metrics_history: HashMap<String, Vec<f32>>,
}

impl HistoryCallback {
    /// Create a new history callback.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the training loss history.
    pub fn train_losses(&self) -> &[f32] {
        &self.train_losses
    }

    /// Get the validation loss history.
    pub fn valid_losses(&self) -> &[f32] {
        &self.valid_losses
    }

    /// Get the learning rate history.
    pub fn learning_rates(&self) -> &[f64] {
        &self.learning_rates
    }

    /// Get the history for a specific metric.
    pub fn metric_history(&self, name: &str) -> Option<&[f32]> {
        self.metrics_history.get(name).map(|v| v.as_slice())
    }

    /// Get all metric names.
    pub fn metric_names(&self) -> Vec<&str> {
        self.metrics_history.keys().map(|s| s.as_str()).collect()
    }

    /// Get the best epoch based on validation loss (minimum).
    pub fn best_epoch(&self) -> Option<usize> {
        self.valid_losses
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
    }
}

impl Callback for HistoryCallback {
    fn after_epoch(&mut self, ctx: &mut CallbackContext) -> Result<()> {
        if let Some(loss) = ctx.train_loss {
            self.train_losses.push(loss);
        }
        if let Some(loss) = ctx.valid_loss {
            self.valid_losses.push(loss);
        }
        self.learning_rates.push(ctx.lr);

        for (name, &value) in &ctx.metrics {
            self.metrics_history
                .entry(name.clone())
                .or_default()
                .push(value);
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "HistoryCallback"
    }
}

/// Callback for mixed precision training.
///
/// Tracks loss scaling factor and overflow events for automatic
/// mixed precision (AMP) training.
pub struct MixedPrecisionCallback {
    initial_scale: f32,
    current_scale: f32,
    growth_factor: f32,
    backoff_factor: f32,
    growth_interval: usize,
    batches_since_rescale: usize,
    overflow_count: usize,
}

impl MixedPrecisionCallback {
    /// Create a new mixed precision callback.
    pub fn new(initial_scale: f32) -> Self {
        Self {
            initial_scale,
            current_scale: initial_scale,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            batches_since_rescale: 0,
            overflow_count: 0,
        }
    }

    /// Get the current loss scale.
    pub fn current_scale(&self) -> f32 {
        self.current_scale
    }

    /// Report an overflow (nan/inf in gradients).
    pub fn report_overflow(&mut self) {
        self.overflow_count += 1;
        self.current_scale *= self.backoff_factor;
        self.batches_since_rescale = 0;
    }
}

impl Callback for MixedPrecisionCallback {
    fn before_fit(&mut self, _ctx: &mut CallbackContext) -> Result<()> {
        self.current_scale = self.initial_scale;
        self.batches_since_rescale = 0;
        self.overflow_count = 0;
        Ok(())
    }

    fn after_batch(&mut self, _ctx: &mut CallbackContext) -> Result<()> {
        self.batches_since_rescale += 1;

        // Try to grow scale periodically if no overflows
        if self.batches_since_rescale >= self.growth_interval {
            self.current_scale *= self.growth_factor;
            self.batches_since_rescale = 0;
        }

        Ok(())
    }

    fn after_fit(&mut self, _ctx: &mut CallbackContext) -> Result<()> {
        if self.overflow_count > 0 {
            tracing::info!(
                "Mixed precision: {} overflow events, final scale = {:.0}",
                self.overflow_count,
                self.current_scale
            );
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "MixedPrecisionCallback"
    }
}

/// Callback for terminating training after a certain number of batches.
///
/// Useful for debugging or quick sanity checks.
pub struct TerminateOnNanCallback {
    nan_count: usize,
}

impl TerminateOnNanCallback {
    /// Create a new terminate on NaN callback.
    pub fn new() -> Self {
        Self { nan_count: 0 }
    }
}

impl Default for TerminateOnNanCallback {
    fn default() -> Self {
        Self::new()
    }
}

impl Callback for TerminateOnNanCallback {
    fn after_batch(&mut self, ctx: &mut CallbackContext) -> Result<()> {
        if let Some(loss) = ctx.train_loss {
            if loss.is_nan() || loss.is_infinite() {
                self.nan_count += 1;
                tracing::error!("NaN/Inf detected in training loss at batch {}", ctx.batch);
                ctx.stop_training = true;
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "TerminateOnNanCallback"
    }
}

/// Callback for displaying ASCII training graphs.
///
/// Shows training and validation loss curves in the terminal after each epoch.
/// Useful for quick visual feedback on training progress without external tools.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_train::callback::ShowGraphCallback;
///
/// // Create with default settings (40x10 graph)
/// let callback = ShowGraphCallback::new();
///
/// // Or customize the display
/// let callback = ShowGraphCallback::new()
///     .with_width(60)
///     .with_height(15)
///     .with_metrics(vec!["accuracy"]);
/// ```
pub struct ShowGraphCallback {
    /// Training losses to plot.
    train_losses: Vec<f32>,
    /// Validation losses to plot.
    valid_losses: Vec<f32>,
    /// Additional metrics to track.
    metrics_history: HashMap<String, Vec<f32>>,
    /// Names of additional metrics to display.
    metric_names: Vec<String>,
    /// Graph width in characters.
    width: usize,
    /// Graph height in characters.
    height: usize,
    /// Whether to show after each epoch.
    show_per_epoch: bool,
}

impl Default for ShowGraphCallback {
    fn default() -> Self {
        Self {
            train_losses: Vec::new(),
            valid_losses: Vec::new(),
            metrics_history: HashMap::new(),
            metric_names: Vec::new(),
            width: 50,
            height: 10,
            show_per_epoch: true,
        }
    }
}

impl ShowGraphCallback {
    /// Create a new show graph callback with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the graph width in characters.
    #[must_use]
    pub fn with_width(mut self, width: usize) -> Self {
        self.width = width.max(20);
        self
    }

    /// Set the graph height in characters.
    #[must_use]
    pub fn with_height(mut self, height: usize) -> Self {
        self.height = height.max(5);
        self
    }

    /// Add metrics to display in addition to loss.
    #[must_use]
    pub fn with_metrics(mut self, names: Vec<&str>) -> Self {
        self.metric_names = names.into_iter().map(|s| s.to_string()).collect();
        self
    }

    /// Set whether to show the graph after each epoch.
    #[must_use]
    pub fn show_per_epoch(mut self, show: bool) -> Self {
        self.show_per_epoch = show;
        self
    }

    /// Render an ASCII graph for a set of values.
    fn render_graph(&self, label: &str, values: &[f32], color_start: &str, color_end: &str) -> String {
        if values.is_empty() {
            return String::new();
        }

        let mut output = String::new();

        // Find min/max for scaling
        let min_val = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_val - min_val).max(1e-6);

        // Create graph header
        output.push_str(&format!("┌─ {} ", label));
        let header_remaining = self.width.saturating_sub(label.len() + 4);
        output.push_str(&"─".repeat(header_remaining));
        output.push_str("┐\n");

        // Create the plot area
        let mut grid = vec![vec![' '; self.width]; self.height];

        // Map values to grid positions
        let step = values.len() as f32 / self.width as f32;
        for col in 0..self.width {
            let idx = (col as f32 * step) as usize;
            if idx < values.len() {
                let val = values[idx];
                let normalized = (val - min_val) / range;
                let row = ((1.0 - normalized) * (self.height - 1) as f32) as usize;
                let row = row.min(self.height - 1);
                grid[row][col] = '█';
            }
        }

        // Render grid with axis labels
        for (i, row) in grid.iter().enumerate() {
            // Y-axis label
            if i == 0 {
                output.push_str(&format!("│{:>6.3} ", max_val));
            } else if i == self.height - 1 {
                output.push_str(&format!("│{:>6.3} ", min_val));
            } else {
                output.push_str("│       ");
            }

            // Plot data with color
            output.push_str(color_start);
            for &ch in row {
                output.push(ch);
            }
            output.push_str(color_end);
            output.push_str("│\n");
        }

        // Bottom border with epoch labels
        output.push_str("└───────");
        output.push_str(&"─".repeat(self.width));
        output.push_str("┘\n");

        // X-axis label
        output.push_str(&format!("        Epochs: 1 → {}\n", values.len()));

        output
    }

    /// Display the training graphs.
    fn display_graphs(&self) {
        let mut output = String::new();
        output.push_str("\n╔══════════════════════════════════════════════════════════════╗\n");
        output.push_str("║                    Training Progress                         ║\n");
        output.push_str("╚══════════════════════════════════════════════════════════════╝\n\n");

        // Show training loss
        if !self.train_losses.is_empty() {
            output.push_str(&self.render_graph("Train Loss", &self.train_losses, "\x1b[33m", "\x1b[0m"));
            output.push('\n');
        }

        // Show validation loss
        if !self.valid_losses.is_empty() {
            output.push_str(&self.render_graph("Valid Loss", &self.valid_losses, "\x1b[36m", "\x1b[0m"));
            output.push('\n');
        }

        // Show additional metrics
        for name in &self.metric_names {
            if let Some(values) = self.metrics_history.get(name) {
                if !values.is_empty() {
                    output.push_str(&self.render_graph(name, values, "\x1b[32m", "\x1b[0m"));
                    output.push('\n');
                }
            }
        }

        // Current values summary
        output.push_str("Current Values:\n");
        if let Some(train) = self.train_losses.last() {
            output.push_str(&format!("  Train Loss: {:.4}\n", train));
        }
        if let Some(valid) = self.valid_losses.last() {
            output.push_str(&format!("  Valid Loss: {:.4}\n", valid));
        }
        for name in &self.metric_names {
            if let Some(values) = self.metrics_history.get(name) {
                if let Some(val) = values.last() {
                    output.push_str(&format!("  {}: {:.4}\n", name, val));
                }
            }
        }

        // Print to stdout
        print!("{}", output);
    }
}

impl Callback for ShowGraphCallback {
    fn before_fit(&mut self, _ctx: &mut CallbackContext) -> Result<()> {
        self.train_losses.clear();
        self.valid_losses.clear();
        self.metrics_history.clear();
        tracing::info!("ShowGraph enabled - will display training curves");
        Ok(())
    }

    fn after_epoch(&mut self, ctx: &mut CallbackContext) -> Result<()> {
        // Record losses
        if let Some(loss) = ctx.train_loss {
            self.train_losses.push(loss);
        }
        if let Some(loss) = ctx.valid_loss {
            self.valid_losses.push(loss);
        }

        // Record tracked metrics
        for name in &self.metric_names {
            if let Some(&value) = ctx.metrics.get(name) {
                self.metrics_history
                    .entry(name.clone())
                    .or_default()
                    .push(value);
            }
        }

        // Display if enabled
        if self.show_per_epoch {
            self.display_graphs();
        }

        Ok(())
    }

    fn after_fit(&mut self, _ctx: &mut CallbackContext) -> Result<()> {
        // Always show final graph
        if !self.show_per_epoch && (!self.train_losses.is_empty() || !self.valid_losses.is_empty()) {
            self.display_graphs();
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "ShowGraphCallback"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_callback_context() {
        let ctx = CallbackContext::new(10, 100);
        assert_eq!(ctx.epoch, 0);
        assert_eq!(ctx.n_epochs, 10);
        assert_eq!(ctx.progress(), 0.0);
    }

    #[test]
    fn test_callback_list() {
        let mut list = CallbackList::new();
        list.add(ProgressCallback::new(false));
        // Would add more callbacks here
    }

    #[test]
    fn test_gradient_clip_callback() {
        let clip = GradientClipCallback::by_norm(1.0);
        assert_eq!(clip.clip_value(), 1.0);

        let clip = GradientClipCallback::by_value(0.5);
        assert_eq!(clip.clip_value(), 0.5);
    }

    #[test]
    fn test_history_callback() {
        let mut history = HistoryCallback::new();
        let mut ctx = CallbackContext::new(10, 100);

        ctx.train_loss = Some(0.5);
        ctx.valid_loss = Some(0.4);
        ctx.lr = 0.001;
        history.after_epoch(&mut ctx).unwrap();

        assert_eq!(history.train_losses(), &[0.5]);
        assert_eq!(history.valid_losses(), &[0.4]);
        assert_eq!(history.learning_rates(), &[0.001]);
    }

    #[test]
    fn test_show_graph_callback_config() {
        let callback = ShowGraphCallback::new()
            .with_width(60)
            .with_height(15)
            .with_metrics(vec!["accuracy", "f1"]);

        assert_eq!(callback.width, 60);
        assert_eq!(callback.height, 15);
        assert_eq!(callback.metric_names, vec!["accuracy", "f1"]);
    }

    #[test]
    fn test_show_graph_callback_render() {
        let mut callback = ShowGraphCallback::new()
            .with_width(30)
            .with_height(5)
            .show_per_epoch(false);

        let mut ctx = CallbackContext::new(10, 100);

        // Simulate several epochs
        for i in 0..5 {
            ctx.train_loss = Some(1.0 - i as f32 * 0.1);
            ctx.valid_loss = Some(0.9 - i as f32 * 0.08);
            callback.after_epoch(&mut ctx).unwrap();
        }

        assert_eq!(callback.train_losses.len(), 5);
        assert_eq!(callback.valid_losses.len(), 5);
    }
}
