//! Hardware discovery orchestration.
//!
//! This module provides unified device discovery across all backends.
//!
//! ## Memoization
//!
//! Hardware discovery is automatically memoized via [`get_device_pool()`].
//! The first call performs full discovery (~50µs), subsequent calls return
//! instantly from cache (<50ns).
//!
//! ```rust,ignore
//! use tsai_compute::get_device_pool;
//!
//! // First call: performs discovery
//! let pool = get_device_pool()?;
//!
//! // Subsequent calls: returns cached result instantly
//! let pool2 = get_device_pool()?;
//! ```

mod cpu;
mod gpu;

pub use cpu::*;
pub use gpu::*;

use std::sync::Arc;
use std::time::Instant;

use crate::backend::cpu::CpuBackend;
use crate::backend::ComputeBackend;
use crate::device::{DevicePool, DeviceType};
use crate::error::ComputeResult;

/// Hardware discovery orchestrator.
///
/// Discovers all available compute devices across all backends
/// and registers them in a device pool.
pub struct HardwareDiscovery;

impl HardwareDiscovery {
    /// Perform full system discovery.
    ///
    /// This discovers all available compute devices and returns
    /// a pool containing them.
    pub fn discover_all() -> ComputeResult<DevicePool> {
        let pool = DevicePool::new();

        // Phase 1: CPU discovery (always available)
        Self::discover_cpu(&pool)?;

        // Phase 2: GPU discovery (feature-gated)
        #[cfg(target_os = "macos")]
        Self::discover_metal(&pool)?;

        #[cfg(feature = "cuda")]
        Self::discover_cuda(&pool)?;

        #[cfg(feature = "vulkan")]
        Self::discover_vulkan(&pool)?;

        #[cfg(feature = "opencl")]
        Self::discover_opencl(&pool)?;

        #[cfg(feature = "rocm")]
        Self::discover_rocm(&pool)?;

        // Phase 3: Select default device
        pool.auto_select_default()?;

        Ok(pool)
    }

    /// Discover CPU devices.
    pub fn discover_cpu(pool: &DevicePool) -> ComputeResult<()> {
        let devices = CpuBackend::enumerate_devices()?;
        pool.register_all(devices)?;
        tracing::info!("Discovered {} CPU device(s)", pool.devices_of_type(DeviceType::Cpu).len());
        Ok(())
    }

    /// Discover Metal devices (macOS).
    #[cfg(target_os = "macos")]
    pub fn discover_metal(pool: &DevicePool) -> ComputeResult<()> {
        use crate::backend::metal::MetalBackend;

        match MetalBackend::enumerate_devices() {
            Ok(devices) => {
                let count = devices.len();
                pool.register_all(devices)?;
                tracing::info!("Discovered {} Metal device(s)", count);
            }
            Err(e) => {
                tracing::warn!("Metal discovery failed: {}", e);
            }
        }
        Ok(())
    }

    /// Discover CUDA devices.
    #[cfg(feature = "cuda")]
    pub fn discover_cuda(pool: &DevicePool) -> ComputeResult<()> {
        use crate::backend::cuda::CudaBackend;

        match CudaBackend::enumerate_devices() {
            Ok(devices) => {
                let count = devices.len();
                pool.register_all(devices)?;
                tracing::info!("Discovered {} CUDA device(s)", count);
            }
            Err(e) => {
                tracing::warn!("CUDA discovery failed: {}", e);
            }
        }
        Ok(())
    }

    /// Discover Vulkan devices.
    #[cfg(feature = "vulkan")]
    pub fn discover_vulkan(pool: &DevicePool) -> ComputeResult<()> {
        use crate::backend::vulkan::VulkanBackend;

        match VulkanBackend::enumerate_devices() {
            Ok(devices) => {
                let count = devices.len();
                pool.register_all(devices)?;
                tracing::info!("Discovered {} Vulkan device(s)", count);
            }
            Err(e) => {
                tracing::debug!("Vulkan discovery failed: {}", e);
            }
        }
        Ok(())
    }

    /// Discover OpenCL devices.
    #[cfg(feature = "opencl")]
    pub fn discover_opencl(pool: &DevicePool) -> ComputeResult<()> {
        use crate::backend::opencl::OpenClBackend;

        match OpenClBackend::enumerate_devices() {
            Ok(devices) => {
                let count = devices.len();
                pool.register_all(devices)?;
                tracing::info!("Discovered {} OpenCL device(s)", count);
            }
            Err(e) => {
                tracing::debug!("OpenCL discovery failed: {}", e);
            }
        }
        Ok(())
    }

    /// Discover ROCm devices.
    #[cfg(feature = "rocm")]
    pub fn discover_rocm(pool: &DevicePool) -> ComputeResult<()> {
        use crate::backend::rocm::RocmBackend;

        match RocmBackend::enumerate_devices() {
            Ok(devices) => {
                let count = devices.len();
                pool.register_all(devices)?;
                tracing::info!("Discovered {} ROCm device(s)", count);
            }
            Err(e) => {
                tracing::debug!("ROCm discovery failed: {}", e);
            }
        }
        Ok(())
    }

    /// Discover only CPU devices.
    pub fn discover_cpu_only() -> ComputeResult<DevicePool> {
        let pool = DevicePool::new();
        Self::discover_cpu(&pool)?;
        pool.auto_select_default()?;
        Ok(pool)
    }

    /// Discover only GPU devices.
    pub fn discover_gpu_only() -> ComputeResult<DevicePool> {
        let pool = DevicePool::new();

        #[cfg(target_os = "macos")]
        Self::discover_metal(&pool)?;

        #[cfg(feature = "cuda")]
        Self::discover_cuda(&pool)?;

        #[cfg(feature = "vulkan")]
        Self::discover_vulkan(&pool)?;

        #[cfg(feature = "opencl")]
        Self::discover_opencl(&pool)?;

        #[cfg(feature = "rocm")]
        Self::discover_rocm(&pool)?;

        pool.auto_select_default()?;
        Ok(pool)
    }
}

/// Cached discovery result with timing information.
struct CachedDiscovery {
    pool: Arc<DevicePool>,
    discovery_time_us: u64,
}

/// Get a shared device pool with automatic discovery.
///
/// This is a convenience function that performs discovery once
/// and caches the result. Subsequent calls return instantly from cache.
///
/// # Performance
///
/// - First call: ~50µs (full hardware discovery)
/// - Subsequent calls: <50ns (returns cached result)
///
/// # Example
///
/// ```rust,ignore
/// use tsai_compute::get_device_pool;
///
/// let pool = get_device_pool()?;
/// println!("Found {} devices", pool.device_count());
/// ```
pub fn get_device_pool() -> ComputeResult<Arc<DevicePool>> {
    use std::sync::OnceLock;

    static CACHE: OnceLock<CachedDiscovery> = OnceLock::new();

    // If already initialized, return it
    if let Some(cached) = CACHE.get() {
        return Ok(cached.pool.clone());
    }

    // Perform timed discovery
    let start = Instant::now();
    let pool = Arc::new(HardwareDiscovery::discover_all()?);
    let discovery_time_us = start.elapsed().as_micros() as u64;

    tracing::debug!("Hardware discovery completed in {}µs", discovery_time_us);

    // Cache the result (handles race conditions)
    let cached = CACHE.get_or_init(|| CachedDiscovery {
        pool: pool.clone(),
        discovery_time_us,
    });

    Ok(cached.pool.clone())
}

/// Get the time taken for the initial hardware discovery in microseconds.
///
/// Returns `None` if discovery hasn't been performed yet.
pub fn get_discovery_time_us() -> Option<u64> {
    use std::sync::OnceLock;

    static CACHE: OnceLock<CachedDiscovery> = OnceLock::new();
    CACHE.get().map(|c| c.discovery_time_us)
}

/// Force a refresh of the device pool cache.
///
/// This performs a new hardware discovery and updates the cache.
/// Useful if hardware has been hot-plugged.
///
/// Note: This creates a new pool but doesn't invalidate existing
/// references to the old pool.
pub fn refresh_device_pool() -> ComputeResult<Arc<DevicePool>> {
    let start = Instant::now();
    let pool = Arc::new(HardwareDiscovery::discover_all()?);
    let discovery_time_us = start.elapsed().as_micros() as u64;

    tracing::info!("Hardware discovery refreshed in {}µs", discovery_time_us);

    Ok(pool)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discovery_all() {
        let pool = HardwareDiscovery::discover_all().unwrap();
        assert!(pool.has_devices());
        pool.print_summary();
    }

    #[test]
    fn test_discovery_cpu() {
        let pool = HardwareDiscovery::discover_cpu_only().unwrap();
        assert!(!pool.cpu_devices().is_empty());
    }
}
