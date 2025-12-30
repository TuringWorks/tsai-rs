//! Error types for tsai_data.

use thiserror::Error;

/// Result type alias using [`DataError`].
pub type Result<T> = std::result::Result<T, DataError>;

/// Errors that can occur in data operations.
#[derive(Error, Debug)]
pub enum DataError {
    /// Invalid data shape.
    #[error("Invalid shape: {0}")]
    InvalidShape(String),

    /// Empty dataset.
    #[error("Dataset is empty")]
    EmptyDataset,

    /// Index out of bounds.
    #[error("Index {index} out of bounds for length {length}")]
    IndexOutOfBounds {
        /// The requested index.
        index: usize,
        /// The length of the collection.
        length: usize,
    },

    /// Batch size error.
    #[error("Invalid batch size: {0}")]
    InvalidBatchSize(String),

    /// Split error.
    #[error("Split error: {0}")]
    SplitError(String),

    /// File format error.
    #[error("File format error: {0}")]
    FormatError(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// I/O error (string variant for custom messages).
    #[error("I/O error: {0}")]
    Io(String),

    /// Download error.
    #[error("Download error: {0}")]
    Download(String),

    /// Parse error.
    #[error("Parse error: {0}")]
    Parse(String),

    /// Invalid input.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Core error.
    #[error("Core error: {0}")]
    CoreError(#[from] tsai_core::CoreError),

    /// Other error.
    #[error("{0}")]
    Other(String),
}
