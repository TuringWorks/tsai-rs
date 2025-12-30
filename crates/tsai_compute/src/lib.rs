//! # tsai_compute
//!
//! A heterogeneous compute abstraction layer for tsai-rs.
//!
//! This crate provides a unified interface for compute operations across
//! different hardware backends:
//!
//! - **CPU**: SIMD-optimized (AVX2, AVX-512, NEON) with optional NUMA awareness
//! - **Metal**: Apple GPU via Metal framework (macOS/iOS)
//! - **CUDA**: NVIDIA GPU via CUDA toolkit
//! - **Vulkan**: Cross-platform GPU via Vulkan
//! - **OpenCL**: Cross-platform GPU/CPU via OpenCL
//! - **ROCm**: AMD GPU via ROCm/HIP
//!
//! ## Features
//!
//! - Runtime device discovery and selection
//! - Unified buffer and memory management
//! - Command encoding and synchronization
//! - Workload-aware scheduling
//! - Burn framework integration (optional)
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use tsai_compute::{HardwareDiscovery, DevicePool};
//!
//! // Discover all available devices
//! let pool = HardwareDiscovery::discover_all()?;
//! pool.print_summary();
//!
//! // Get the best device
//! let device = pool.best_device().expect("No devices available");
//! println!("Using: {}", device.name());
//! ```
//!
//! ## Feature Flags
//!
//! - `cpu` (default): CPU backend with SIMD
//! - `numa`: NUMA-aware memory allocation
//! - `cuda`: NVIDIA CUDA support
//! - `metal`: Apple Metal support (macOS only)
//! - `vulkan`: Vulkan compute support
//! - `opencl`: OpenCL support
//! - `rocm`: AMD ROCm support
//! - `burn-bridge`: Burn framework integration

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod backend;
pub mod bridge;
pub mod device;
pub mod discovery;
pub mod error;
pub mod memory;
pub mod scheduler;

// Re-export commonly used types
pub use backend::{CommandEncoder, ComputeBackend, Fence};
pub use device::{
    ComputeDevice, DeviceCapabilities, DeviceFeature, DeviceId, DevicePool, DeviceType,
    SelectionStrategy, SimdLevel,
};
pub use discovery::{get_device_pool, get_discovery_time_us, refresh_device_pool, HardwareDiscovery};
pub use error::{ComputeError, ComputeResult};
pub use memory::{Buffer, BufferUsage, MemoryPool};
pub use scheduler::{Priority, RoundRobinScheduler, Scheduler, SimpleScheduler, Workload, WorkloadScheduler};

// Backend-specific re-exports
pub use backend::cpu::{CpuBackend, CpuBuffer, CpuDevice};

#[cfg(target_os = "macos")]
pub use backend::metal::{MetalBackend, MetalBuffer, MetalDevice};

#[cfg(feature = "cuda")]
pub use backend::cuda::{CudaBackend, CudaBuffer, CudaDevice};

#[cfg(feature = "vulkan")]
pub use backend::vulkan::{VulkanBackend, VulkanBuffer, VulkanDevice};

#[cfg(feature = "opencl")]
pub use backend::opencl::{OpenClBackend, OpenClBuffer, OpenClDevice};

#[cfg(feature = "rocm")]
pub use backend::rocm::{RocmBackend, RocmBuffer, RocmDevice};

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::backend::{CommandEncoder, ComputeBackend, Fence};
    pub use crate::device::{
        ComputeDevice, DeviceCapabilities, DeviceFeature, DeviceId, DevicePool, DeviceType,
    };
    pub use crate::discovery::{get_device_pool, HardwareDiscovery};
    pub use crate::error::{ComputeError, ComputeResult};
    pub use crate::memory::{Buffer, BufferUsage};
    pub use crate::scheduler::{Scheduler, Workload};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_discovery() {
        let pool = HardwareDiscovery::discover_all().unwrap();
        assert!(pool.has_devices());

        println!("\n=== tsai_compute Device Discovery ===\n");
        pool.print_summary();
    }

    #[test]
    fn test_cpu_backend() {
        let devices = CpuBackend::enumerate_devices().unwrap();
        assert!(!devices.is_empty());

        let backend = CpuBackend::new(&devices[0]).unwrap();
        let buffer = backend.allocate_buffer(1024, BufferUsage::HOST_VISIBLE).unwrap();
        assert_eq!(buffer.size(), 1024);
    }

    #[test]
    fn test_device_selection() {
        let pool = HardwareDiscovery::discover_all().unwrap();

        // Test best device selection
        let best = pool.best_device();
        assert!(best.is_some());

        // Test device filtering
        let cpus = pool.cpu_devices();
        assert!(!cpus.is_empty());
    }

    #[test]
    fn test_scheduler() {
        let pool = HardwareDiscovery::discover_all().unwrap();
        let scheduler = WorkloadScheduler::new();

        let workload = Workload::new()
            .with_flops(1_000_000)
            .with_memory(1024 * 1024);

        let device = scheduler.select_device(&pool, &workload).unwrap();
        println!("Selected device: {:?}", device);
    }
}
