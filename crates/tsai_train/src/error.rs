//! Error types for training.

use thiserror::Error;

/// Result type alias for training operations.
pub type Result<T> = std::result::Result<T, TrainError>;

/// Errors that can occur during training.
#[derive(Error, Debug)]
pub enum TrainError {
    /// Model forward pass failed.
    #[error("Forward pass failed: {0}")]
    ForwardError(String),

    /// Backward pass failed.
    #[error("Backward pass failed: {0}")]
    BackwardError(String),

    /// Optimizer step failed.
    #[error("Optimizer step failed: {0}")]
    OptimizerError(String),

    /// Invalid learning rate.
    #[error("Invalid learning rate: {0}")]
    InvalidLearningRate(String),

    /// Checkpoint error.
    #[error("Checkpoint error: {0}")]
    CheckpointError(String),

    /// Callback error.
    #[error("Callback error: {0}")]
    CallbackError(String),

    /// Data error.
    #[error("Data error: {0}")]
    DataError(#[from] tsai_data::DataError),

    /// Core error.
    #[error("Core error: {0}")]
    CoreError(#[from] tsai_core::CoreError),

    /// I/O error.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Training was interrupted.
    #[error("Training interrupted: {0}")]
    Interrupted(String),

    /// Other error.
    #[error("{0}")]
    Other(String),
}
