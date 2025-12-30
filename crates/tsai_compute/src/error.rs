//! Unified error types for tsai_compute.

use std::fmt;

/// Result type for compute operations.
pub type ComputeResult<T> = Result<T, ComputeError>;

/// Unified error type for all compute operations.
#[derive(Debug, thiserror::Error)]
pub enum ComputeError {
    /// Device not found or unavailable.
    #[error("Device not found: {0}")]
    DeviceNotFound(String),

    /// Device is not available for computation.
    #[error("Device unavailable: {0}")]
    DeviceUnavailable(String),

    /// Backend initialization failed.
    #[error("Backend initialization failed: {0}")]
    BackendInitFailed(String),

    /// Memory allocation failed.
    #[error("Memory allocation failed: size={size}, reason={reason}")]
    AllocationFailed {
        /// Requested allocation size in bytes.
        size: usize,
        /// Reason for allocation failure.
        reason: String,
    },

    /// Out of memory on device.
    #[error("Out of memory: requested {requested} bytes, available {available} bytes")]
    OutOfMemory {
        /// Requested memory in bytes.
        requested: u64,
        /// Available memory in bytes.
        available: u64,
    },

    /// Buffer operation error.
    #[error("Buffer error: {0}")]
    BufferError(String),

    /// Invalid buffer mapping.
    #[error("Buffer mapping failed: {0}")]
    MappingFailed(String),

    /// Command encoding error.
    #[error("Command encoding error: {0}")]
    EncodingError(String),

    /// Command submission failed.
    #[error("Command submission failed: {0}")]
    SubmissionFailed(String),

    /// Synchronization error.
    #[error("Synchronization error: {0}")]
    SyncError(String),

    /// Kernel compilation or execution error.
    #[error("Kernel error: {0}")]
    KernelError(String),

    /// Data transfer error.
    #[error("Transfer error: {0}")]
    TransferError(String),

    /// Feature not supported on this device.
    #[error("Feature not supported: {feature} on device {device}")]
    FeatureNotSupported {
        /// Name of the unsupported feature.
        feature: String,
        /// Device identifier.
        device: String,
    },

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Invalid shape or dimensions.
    #[error("Invalid shape: {0}")]
    InvalidShape(String),

    /// Backend-specific error.
    #[error("{backend} error: {message}")]
    BackendError {
        /// Which backend produced the error.
        backend: BackendKind,
        /// Error message from the backend.
        message: String,
    },

    /// Discovery failed.
    #[error("Device discovery failed: {0}")]
    DiscoveryFailed(String),

    /// Scheduler error.
    #[error("Scheduler error: {0}")]
    SchedulerError(String),

    /// Internal error (should not happen).
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Backend identifier for error context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// CPU backend.
    Cpu,
    /// NVIDIA CUDA backend.
    Cuda,
    /// Apple Metal backend.
    Metal,
    /// Vulkan backend.
    Vulkan,
    /// OpenCL backend.
    OpenCl,
    /// AMD ROCm backend.
    Rocm,
    /// Apple MLX backend.
    Mlx,
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendKind::Cpu => write!(f, "CPU"),
            BackendKind::Cuda => write!(f, "CUDA"),
            BackendKind::Metal => write!(f, "Metal"),
            BackendKind::Vulkan => write!(f, "Vulkan"),
            BackendKind::OpenCl => write!(f, "OpenCL"),
            BackendKind::Rocm => write!(f, "ROCm"),
            BackendKind::Mlx => write!(f, "MLX"),
        }
    }
}

impl ComputeError {
    /// Create a backend-specific error.
    pub fn backend(backend: BackendKind, message: impl Into<String>) -> Self {
        ComputeError::BackendError {
            backend,
            message: message.into(),
        }
    }

    /// Create an allocation error.
    pub fn allocation(size: usize, reason: impl Into<String>) -> Self {
        ComputeError::AllocationFailed {
            size,
            reason: reason.into(),
        }
    }

    /// Create an out of memory error.
    pub fn oom(requested: u64, available: u64) -> Self {
        ComputeError::OutOfMemory {
            requested,
            available,
        }
    }

    /// Create a feature not supported error.
    pub fn unsupported(feature: impl Into<String>, device: impl Into<String>) -> Self {
        ComputeError::FeatureNotSupported {
            feature: feature.into(),
            device: device.into(),
        }
    }
}

// Conversions from backend-specific errors

#[cfg(feature = "cuda")]
impl From<cudarc::driver::DriverError> for ComputeError {
    fn from(err: cudarc::driver::DriverError) -> Self {
        ComputeError::backend(BackendKind::Cuda, err.to_string())
    }
}

#[cfg(feature = "vulkan")]
impl From<ash::vk::Result> for ComputeError {
    fn from(err: ash::vk::Result) -> Self {
        ComputeError::backend(BackendKind::Vulkan, format!("{:?}", err))
    }
}

#[cfg(feature = "opencl")]
impl From<opencl3::error_codes::ClError> for ComputeError {
    fn from(err: opencl3::error_codes::ClError) -> Self {
        ComputeError::backend(BackendKind::OpenCl, format!("{:?}", err))
    }
}

#[cfg(feature = "mlx")]
impl From<mlx_rs::error::Exception> for ComputeError {
    fn from(err: mlx_rs::error::Exception) -> Self {
        ComputeError::backend(BackendKind::Mlx, err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = ComputeError::DeviceNotFound("GPU:0".to_string());
        assert!(err.to_string().contains("GPU:0"));

        let err = ComputeError::oom(1024 * 1024, 512 * 1024);
        assert!(err.to_string().contains("1048576"));

        let err = ComputeError::backend(BackendKind::Cuda, "driver error");
        assert!(err.to_string().contains("CUDA"));
    }

    #[test]
    fn test_error_constructors() {
        let err = ComputeError::allocation(1024, "insufficient memory");
        match err {
            ComputeError::AllocationFailed { size, reason } => {
                assert_eq!(size, 1024);
                assert!(reason.contains("insufficient"));
            }
            _ => panic!("Wrong error type"),
        }
    }
}
