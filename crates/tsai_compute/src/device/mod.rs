//! Device abstractions for heterogeneous compute.
//!
//! This module provides the core traits and types for representing
//! compute devices across different backends (CPU, GPU, accelerators).

mod capability;
mod pool;
mod selector;

pub use capability::*;
pub use pool::*;
pub use selector::*;

use std::fmt::{self, Debug};
use std::hash::Hash;

/// Represents a compute device (CPU, GPU, accelerator).
///
/// This trait provides a unified interface for querying device properties
/// and capabilities across all supported backends.
pub trait ComputeDevice: Clone + Debug + Send + Sync + Hash + Eq + 'static {
    /// Get the device type.
    fn device_type(&self) -> DeviceType;

    /// Get the unique device identifier.
    fn device_id(&self) -> DeviceId;

    /// Get the human-readable device name.
    fn name(&self) -> &str;

    /// Get the device capabilities.
    fn capabilities(&self) -> &DeviceCapabilities;

    /// Check if the device supports a specific feature.
    fn supports(&self, feature: DeviceFeature) -> bool {
        self.capabilities().features.contains(&feature)
    }

    /// Get total memory available on device (bytes).
    fn total_memory(&self) -> u64 {
        self.capabilities().total_memory
    }

    /// Get currently available memory (bytes).
    /// Returns total memory if real-time query is not supported.
    fn available_memory(&self) -> u64 {
        // Default implementation returns total memory
        // Backends can override for real-time queries
        self.total_memory()
    }

    /// Check if device is currently available for computation.
    fn is_available(&self) -> bool {
        true
    }

    /// Get the NUMA node this device is associated with (if applicable).
    fn numa_node(&self) -> Option<u32> {
        self.capabilities().numa_node
    }
}

/// Device type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum DeviceType {
    /// CPU device (with optional SIMD/NUMA).
    Cpu = 0,
    /// NVIDIA GPU via CUDA.
    CudaGpu = 1,
    /// AMD GPU via ROCm/HIP.
    RocmGpu = 2,
    /// Apple GPU via Metal.
    MetalGpu = 3,
    /// Cross-platform GPU via Vulkan.
    VulkanGpu = 4,
    /// OpenCL-compatible device.
    OpenClDevice = 5,
    /// Apple Silicon via MLX framework.
    MlxGpu = 6,
}

impl DeviceType {
    /// Check if this is a GPU device type.
    pub fn is_gpu(&self) -> bool {
        !matches!(self, DeviceType::Cpu)
    }

    /// Check if this is a discrete GPU (not integrated).
    pub fn is_discrete(&self) -> bool {
        matches!(
            self,
            DeviceType::CudaGpu | DeviceType::RocmGpu | DeviceType::VulkanGpu
        )
    }

    /// Get the priority score for device selection (higher is better).
    /// MLX gets highest priority on macOS due to unified memory and Apple Silicon optimization.
    pub fn priority_score(&self) -> i32 {
        match self {
            DeviceType::MlxGpu => 11000,   // Highest on macOS - unified memory, optimized for Apple Silicon
            DeviceType::CudaGpu => 10000,
            DeviceType::RocmGpu => 9000,
            DeviceType::MetalGpu => 8000,
            DeviceType::VulkanGpu => 7000,
            DeviceType::OpenClDevice => 6000,
            DeviceType::Cpu => 1000,
        }
    }
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceType::Cpu => write!(f, "CPU"),
            DeviceType::CudaGpu => write!(f, "CUDA"),
            DeviceType::RocmGpu => write!(f, "ROCm"),
            DeviceType::MetalGpu => write!(f, "Metal"),
            DeviceType::VulkanGpu => write!(f, "Vulkan"),
            DeviceType::OpenClDevice => write!(f, "OpenCL"),
            DeviceType::MlxGpu => write!(f, "MLX"),
        }
    }
}

/// Unique device identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId {
    /// The device type.
    pub device_type: DeviceType,
    /// Device index within its type (e.g., GPU 0, GPU 1).
    pub index: u32,
    /// NUMA node affinity (for CPU devices).
    pub numa_node: Option<u32>,
}

impl DeviceId {
    /// Create a new device ID.
    pub fn new(device_type: DeviceType, index: u32) -> Self {
        Self {
            device_type,
            index,
            numa_node: None,
        }
    }

    /// Create a device ID with NUMA node affinity.
    pub fn with_numa(device_type: DeviceType, index: u32, numa_node: u32) -> Self {
        Self {
            device_type,
            index,
            numa_node: Some(numa_node),
        }
    }

    /// Create a CPU device ID.
    pub fn cpu(index: u32) -> Self {
        Self::new(DeviceType::Cpu, index)
    }

    /// Create a CUDA GPU device ID.
    pub fn cuda(index: u32) -> Self {
        Self::new(DeviceType::CudaGpu, index)
    }

    /// Create a Metal GPU device ID.
    pub fn metal(index: u32) -> Self {
        Self::new(DeviceType::MetalGpu, index)
    }

    /// Create a Vulkan GPU device ID.
    pub fn vulkan(index: u32) -> Self {
        Self::new(DeviceType::VulkanGpu, index)
    }

    /// Create an OpenCL device ID.
    pub fn opencl(index: u32) -> Self {
        Self::new(DeviceType::OpenClDevice, index)
    }

    /// Create a ROCm GPU device ID.
    pub fn rocm(index: u32) -> Self {
        Self::new(DeviceType::RocmGpu, index)
    }

    /// Create an MLX CPU device ID.
    pub fn mlx_cpu(index: u32) -> Self {
        Self::new(DeviceType::Cpu, index)
    }

    /// Create an MLX GPU device ID.
    pub fn mlx_gpu(index: u32) -> Self {
        Self::new(DeviceType::MlxGpu, index)
    }
}

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.numa_node {
            Some(node) => write!(f, "{}:{}:numa{}", self.device_type, self.index, node),
            None => write!(f, "{}:{}", self.device_type, self.index),
        }
    }
}

impl Default for DeviceId {
    fn default() -> Self {
        Self::cpu(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_type_priority() {
        assert!(DeviceType::CudaGpu.priority_score() > DeviceType::Cpu.priority_score());
        assert!(DeviceType::MetalGpu.priority_score() > DeviceType::OpenClDevice.priority_score());
    }

    #[test]
    fn test_device_id_display() {
        let id = DeviceId::cuda(0);
        assert_eq!(id.to_string(), "CUDA:0");

        let id = DeviceId::with_numa(DeviceType::Cpu, 0, 1);
        assert_eq!(id.to_string(), "CPU:0:numa1");
    }

    #[test]
    fn test_device_type_is_gpu() {
        assert!(!DeviceType::Cpu.is_gpu());
        assert!(DeviceType::CudaGpu.is_gpu());
        assert!(DeviceType::MetalGpu.is_gpu());
    }
}
