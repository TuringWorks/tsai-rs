//! Error types for tsai_core.

use thiserror::Error;

/// Result type alias using [`CoreError`].
pub type Result<T> = std::result::Result<T, CoreError>;

/// Core errors that can occur in tsai_core operations.
#[derive(Error, Debug)]
pub enum CoreError {
    /// Invalid tensor shape provided.
    #[error("Invalid shape: expected {expected}, got {got}")]
    InvalidShape {
        /// Expected shape description.
        expected: String,
        /// Actual shape description.
        got: String,
    },

    /// Shape mismatch between tensors.
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),

    /// Dimension error.
    #[error("Dimension error: expected {expected} dimensions, got {got}")]
    DimensionError {
        /// Expected number of dimensions.
        expected: usize,
        /// Actual number of dimensions.
        got: usize,
    },

    /// Transform error.
    #[error("Transform error: {0}")]
    TransformError(String),

    /// Serialization/deserialization error.
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Generic error.
    #[error("{0}")]
    Other(String),
}
