//! Burn framework integration.
//!
//! This module provides integration between tsai_compute and the Burn
//! deep learning framework, allowing Burn backends to leverage
//! tsai_compute's device management and selection.

use std::sync::Arc;

use crate::device::{DeviceCapabilities, DevicePool, DeviceType};
use crate::discovery::HardwareDiscovery;
use crate::error::ComputeResult;

/// Burn backend bridge for compute-aware device selection.
///
/// This bridge allows Burn backends to use tsai_compute's device
/// discovery and selection logic.
pub struct ComputeBridge {
    pool: Arc<DevicePool>,
}

impl ComputeBridge {
    /// Create a new bridge with automatic device discovery.
    pub fn new() -> ComputeResult<Self> {
        let pool = Arc::new(HardwareDiscovery::discover_all()?);
        Ok(Self { pool })
    }

    /// Create a bridge from an existing device pool.
    pub fn from_pool(pool: Arc<DevicePool>) -> Self {
        Self { pool }
    }

    /// Get the device pool.
    pub fn pool(&self) -> &DevicePool {
        &self.pool
    }

    /// Get the best device type for Burn backend selection.
    pub fn best_backend_type(&self) -> Option<DeviceType> {
        self.pool.best_device().map(|d| d.device_type())
    }

    /// Check if GPU compute is available.
    pub fn has_gpu(&self) -> bool {
        !self.pool.gpu_devices().is_empty()
    }

    /// Get device capabilities for the best device.
    pub fn best_capabilities(&self) -> Option<DeviceCapabilities> {
        self.pool.best_device().map(|d| d.capabilities().clone())
    }

    /// Print device summary.
    pub fn print_devices(&self) {
        self.pool.print_summary();
    }
}

impl Default for ComputeBridge {
    fn default() -> Self {
        Self::new().expect("Failed to initialize compute bridge")
    }
}

/// Extension trait for compute-aware Burn backends.
pub trait ComputeAware {
    /// Get the recommended device type from tsai_compute.
    fn recommended_device_type() -> DeviceType;

    /// Get device capabilities from tsai_compute.
    fn device_capabilities() -> Option<DeviceCapabilities>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_bridge() {
        let bridge = ComputeBridge::new().unwrap();
        bridge.print_devices();

        println!("Best backend type: {:?}", bridge.best_backend_type());
        println!("Has GPU: {}", bridge.has_gpu());
    }
}
