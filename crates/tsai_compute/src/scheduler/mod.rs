//! Workload scheduling for heterogeneous compute.
//!
//! This module provides schedulers for dispatching workloads
//! to optimal devices based on workload characteristics.

mod workload;

pub use workload::*;

use crate::device::{DeviceId, DevicePool};
use crate::error::{ComputeError, ComputeResult};

/// Workload descriptor for scheduling decisions.
#[derive(Debug, Clone)]
pub struct Workload {
    /// Estimated FLOPS required.
    pub flops: u64,
    /// Memory requirements (bytes).
    pub memory_bytes: u64,
    /// Whether this is memory-bound.
    pub memory_bound: bool,
    /// Preferred precision.
    pub precision: crate::device::Precision,
    /// Whether parallelization across devices is possible.
    pub parallelizable: bool,
    /// Priority level.
    pub priority: Priority,
}

impl Default for Workload {
    fn default() -> Self {
        Self {
            flops: 0,
            memory_bytes: 0,
            memory_bound: false,
            precision: crate::device::Precision::Float32,
            parallelizable: true,
            priority: Priority::Normal,
        }
    }
}

impl Workload {
    /// Create a new workload descriptor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set FLOPS requirement.
    pub fn with_flops(mut self, flops: u64) -> Self {
        self.flops = flops;
        self
    }

    /// Set memory requirement.
    pub fn with_memory(mut self, bytes: u64) -> Self {
        self.memory_bytes = bytes;
        self
    }

    /// Mark as memory-bound.
    pub fn memory_bound(mut self) -> Self {
        self.memory_bound = true;
        self
    }

    /// Set priority.
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }
}

/// Priority level for workloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    /// Background task, can be preempted.
    Background,
    /// Normal priority.
    Normal,
    /// High priority.
    High,
    /// Real-time, should not be delayed.
    RealTime,
}

/// Scheduler trait for workload dispatch.
pub trait Scheduler: Send + Sync {
    /// Select the best device for a given workload.
    fn select_device(
        &self,
        pool: &DevicePool,
        workload: &Workload,
    ) -> ComputeResult<DeviceId>;

    /// Select multiple devices for parallel execution.
    fn select_devices(
        &self,
        pool: &DevicePool,
        workload: &Workload,
        max_devices: usize,
    ) -> ComputeResult<Vec<DeviceId>>;

    /// Report completion of a workload (for adaptive scheduling).
    fn report_completion(
        &self,
        device: &DeviceId,
        workload: &Workload,
        duration_ns: u64,
    );
}

/// Simple scheduler that always selects the best device.
pub struct SimpleScheduler;

impl Scheduler for SimpleScheduler {
    fn select_device(
        &self,
        pool: &DevicePool,
        _workload: &Workload,
    ) -> ComputeResult<DeviceId> {
        pool.best_device()
            .map(|d| d.device_id())
            .ok_or_else(|| ComputeError::SchedulerError("No devices available".to_string()))
    }

    fn select_devices(
        &self,
        pool: &DevicePool,
        _workload: &Workload,
        max_devices: usize,
    ) -> ComputeResult<Vec<DeviceId>> {
        let devices: Vec<_> = pool
            .all_devices()
            .iter()
            .take(max_devices)
            .map(|d| d.device_id())
            .collect();

        if devices.is_empty() {
            return Err(ComputeError::SchedulerError("No devices available".to_string()));
        }

        Ok(devices)
    }

    fn report_completion(
        &self,
        _device: &DeviceId,
        _workload: &Workload,
        _duration_ns: u64,
    ) {
        // Simple scheduler doesn't track history
    }
}

/// Round-robin scheduler for load balancing.
pub struct RoundRobinScheduler {
    counter: std::sync::atomic::AtomicUsize,
}

impl RoundRobinScheduler {
    /// Create a new round-robin scheduler.
    pub fn new() -> Self {
        Self {
            counter: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl Default for RoundRobinScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl Scheduler for RoundRobinScheduler {
    fn select_device(
        &self,
        pool: &DevicePool,
        _workload: &Workload,
    ) -> ComputeResult<DeviceId> {
        let devices = pool.all_devices();
        if devices.is_empty() {
            return Err(ComputeError::SchedulerError("No devices available".to_string()));
        }

        let idx = self.counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(devices[idx % devices.len()].device_id())
    }

    fn select_devices(
        &self,
        pool: &DevicePool,
        _workload: &Workload,
        max_devices: usize,
    ) -> ComputeResult<Vec<DeviceId>> {
        let devices: Vec<_> = pool
            .all_devices()
            .iter()
            .take(max_devices)
            .map(|d| d.device_id())
            .collect();

        if devices.is_empty() {
            return Err(ComputeError::SchedulerError("No devices available".to_string()));
        }

        Ok(devices)
    }

    fn report_completion(
        &self,
        _device: &DeviceId,
        _workload: &Workload,
        _duration_ns: u64,
    ) {
        // Round-robin doesn't adapt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workload_builder() {
        let workload = Workload::new()
            .with_flops(1_000_000)
            .with_memory(1024 * 1024)
            .with_priority(Priority::High);

        assert_eq!(workload.flops, 1_000_000);
        assert_eq!(workload.memory_bytes, 1024 * 1024);
        assert_eq!(workload.priority, Priority::High);
    }
}
