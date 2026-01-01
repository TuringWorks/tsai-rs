//! # tsai_train
//!
//! Training loop, callbacks, metrics, and checkpointing for tsai-rs.
//!
//! This crate provides:
//! - [`Learner`] for managing the training process
//! - Callback system with lifecycle hooks
//! - Learning rate schedulers (one-cycle, etc.)
//! - Metrics (accuracy, MSE, MAE)
//! - Checkpointing and model saving
//! - Compatibility facades (TSClassifier, TSRegressor, TSForecaster)
//!
//! ## Example
//!
//! ```rust,ignore
//! use tsai_train::{Learner, LearnerConfig};
//! use tsai_data::TSDataLoaders;
//! use tsai_models::InceptionTimePlus;
//!
//! let model = InceptionTimePlus::new(config, &device);
//! let learner = Learner::new(model, dls)
//!     .with_optimizer(Adam::new(1e-3))
//!     .with_loss(CrossEntropyLoss::new());
//!
//! learner.fit_one_cycle(10, 1e-3)?;
//! ```

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod callback;
pub mod compat;
pub mod error;
pub mod evaluation;
pub mod learner;
pub mod losses;
pub mod metrics;
pub mod optimizer;
pub mod scheduler;
pub mod training;

pub use callback::{
    BatchSubsamplerCallback, Callback, CallbackContext, CallbackList, CheckpointMetadata,
    EarlyStoppingCallback, GradientClipCallback, GradientClipMode, HistoryCallback,
    MixedPrecisionCallback, NoiseInjection, NoisyStudentCallback, NoisyStudentStats,
    PredictionDynamicsCallback, PredictionDynamicsSummary, PredictionTrackingMode,
    ProgressCallback, PseudoLabel, PseudoLabelFilter, SamplePredictionHistory, SaveModelCallback,
    SaveModelMode, ShowGraphCallback, SubsampleStrategy, TerminateOnNanCallback,
    TransformSchedule, TransformSchedulerCallback, WeightedPerSampleLossCallback, WeightStrategy,
};
pub use error::{Result, TrainError};
pub use learner::{Learner, LearnerConfig, TrainingState};
pub use losses::{CrossEntropyLoss, FocalLoss, HuberLoss, LabelSmoothingLoss, LogCoshLoss, MSELoss};
pub use optimizer::{RAdam, RAdamConfig, Ranger, RangerConfig};
pub use metrics::{Accuracy, F1Score, Metric, Precision, Recall, AUC, MCC, MAE, MAPE, MSE, RMSE};
pub use scheduler::{
    ConstantLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, ExponentialLR, LinearWarmup,
    OneCycleLR, PolynomialLR, ReduceLROnPlateau, ReduceMode, Scheduler, StepLR,
};
// Re-export model traits from tsai_core for convenience
pub use tsai_core::{TSClassificationModel, TSForecastingModel, TSRegressionModel};
pub use training::{
    train_classification, train_regression,
    ClassificationTrainer, ClassificationTrainerConfig, TrainingOutput,
    RegressionTrainer, RegressionTrainerConfig, RegressionOutput,
};
pub use evaluation::{evaluate_classification, ConfusionMatrix, EvaluationResult};

#[cfg(feature = "wandb")]
pub mod wandb;
