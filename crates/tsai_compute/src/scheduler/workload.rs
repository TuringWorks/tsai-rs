//! Workload-aware scheduling.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

use super::{Priority, Scheduler, Workload};
use crate::device::{DeviceCapabilities, DeviceId, DevicePool, DeviceType};
use crate::error::{ComputeError, ComputeResult};

/// Workload-aware scheduler that considers device capabilities.
pub struct WorkloadScheduler {
    /// Historical execution times per device.
    history: RwLock<HashMap<DeviceId, DeviceHistory>>,
    /// Prefer GPU for large workloads.
    gpu_threshold_flops: u64,
    /// Minimum memory for GPU selection.
    gpu_threshold_memory: u64,
}

#[derive(Default)]
struct DeviceHistory {
    total_time_ns: AtomicU64,
    execution_count: AtomicU64,
}

impl DeviceHistory {
    fn record(&self, duration_ns: u64) {
        self.total_time_ns.fetch_add(duration_ns, Ordering::Relaxed);
        self.execution_count.fetch_add(1, Ordering::Relaxed);
    }

    #[allow(dead_code)] // Reserved for adaptive scheduling implementation
    fn average_ns(&self) -> Option<u64> {
        let count = self.execution_count.load(Ordering::Relaxed);
        if count == 0 {
            return None;
        }
        Some(self.total_time_ns.load(Ordering::Relaxed) / count)
    }
}

impl WorkloadScheduler {
    /// Create a new workload scheduler.
    pub fn new() -> Self {
        Self {
            history: RwLock::new(HashMap::new()),
            gpu_threshold_flops: 1_000_000_000, // 1 GFLOP
            gpu_threshold_memory: 100 * 1024 * 1024, // 100 MB
        }
    }

    /// Set the FLOPS threshold for GPU selection.
    pub fn with_gpu_flops_threshold(mut self, flops: u64) -> Self {
        self.gpu_threshold_flops = flops;
        self
    }

    /// Set the memory threshold for GPU selection.
    pub fn with_gpu_memory_threshold(mut self, bytes: u64) -> Self {
        self.gpu_threshold_memory = bytes;
        self
    }

    /// Score a device for a workload.
    fn score_device(&self, caps: &DeviceCapabilities, workload: &Workload) -> i64 {
        let mut score: i64 = 0;

        // Check memory fit
        if caps.total_memory < workload.memory_bytes {
            return i64::MIN; // Can't fit
        }

        // Compute capability score
        if workload.flops > 0 {
            let tflops = caps.peak_tflops_fp32.max(0.001);
            let estimated_time_ms = (workload.flops as f64) / (tflops as f64 * 1e9);
            score += (1000.0 / estimated_time_ms.max(0.001)) as i64;
        }

        // Memory bandwidth score for memory-bound workloads
        if workload.memory_bound {
            score += (caps.memory_bandwidth_gbps * 10.0) as i64;
        }

        // Precision support
        if caps.supports_precision(workload.precision) {
            score += 100;
        }

        // Priority boost for real-time workloads on faster devices
        if workload.priority == Priority::RealTime {
            score += caps.compute_units as i64 * 10;
        }

        score
    }

    /// Determine if workload should prefer GPU.
    fn should_prefer_gpu(&self, workload: &Workload) -> bool {
        workload.flops >= self.gpu_threshold_flops
            || workload.memory_bytes >= self.gpu_threshold_memory
    }
}

impl Default for WorkloadScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl Scheduler for WorkloadScheduler {
    fn select_device(
        &self,
        pool: &DevicePool,
        workload: &Workload,
    ) -> ComputeResult<DeviceId> {
        let devices = pool.all_devices();
        if devices.is_empty() {
            return Err(ComputeError::SchedulerError("No devices available".to_string()));
        }

        let prefer_gpu = self.should_prefer_gpu(workload);

        // Score all devices
        let best = devices
            .iter()
            .filter(|d| {
                // Filter by GPU preference
                if prefer_gpu && d.device_type() == DeviceType::Cpu {
                    // Only use CPU if no GPUs available
                    !devices.iter().any(|d2| d2.device_type().is_gpu())
                } else {
                    true
                }
            })
            .max_by_key(|d| self.score_device(d.capabilities(), workload))
            .ok_or_else(|| ComputeError::SchedulerError("No suitable device found".to_string()))?;

        Ok(best.device_id())
    }

    fn select_devices(
        &self,
        pool: &DevicePool,
        workload: &Workload,
        max_devices: usize,
    ) -> ComputeResult<Vec<DeviceId>> {
        if !workload.parallelizable {
            // For non-parallelizable workloads, just select one
            return self.select_device(pool, workload).map(|d| vec![d]);
        }

        let mut devices = pool.all_devices();
        if devices.is_empty() {
            return Err(ComputeError::SchedulerError("No devices available".to_string()));
        }

        // Sort by score
        devices.sort_by_key(|d| std::cmp::Reverse(self.score_device(d.capabilities(), workload)));

        // Take top devices
        let selected: Vec<_> = devices
            .iter()
            .take(max_devices)
            .map(|d| d.device_id())
            .collect();

        Ok(selected)
    }

    fn report_completion(
        &self,
        device: &DeviceId,
        _workload: &Workload,
        duration_ns: u64,
    ) {
        let mut history = self.history.write();
        history
            .entry(*device)
            .or_insert_with(DeviceHistory::default)
            .record(duration_ns);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workload_scheduler() {
        let scheduler = WorkloadScheduler::new()
            .with_gpu_flops_threshold(1_000_000)
            .with_gpu_memory_threshold(10 * 1024 * 1024);

        let small_workload = Workload::new().with_flops(1000);
        assert!(!scheduler.should_prefer_gpu(&small_workload));

        let large_workload = Workload::new().with_flops(1_000_000_000);
        assert!(scheduler.should_prefer_gpu(&large_workload));
    }
}
