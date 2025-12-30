//! Device selection strategies.

use super::{AnyDevice, DeviceCapabilities, DeviceId, DevicePool, DeviceType};
use crate::error::{ComputeError, ComputeResult};
use std::fmt;
use std::sync::Arc;

/// Device selection strategy.
#[derive(Clone)]
pub enum SelectionStrategy {
    /// Select device with most memory.
    MaxMemory,
    /// Select device with highest compute capability.
    MaxCompute,
    /// Select device with lowest expected latency (usually integrated/CPU).
    LowLatency,
    /// Balance between compute power and availability.
    Balanced,
    /// Use CPU only.
    CpuOnly,
    /// Prefer GPU, fall back to CPU.
    PreferGpu,
    /// Explicit device selection.
    Explicit(DeviceId),
    /// Custom scoring function.
    Custom(Arc<dyn Fn(&DeviceCapabilities) -> i64 + Send + Sync>),
}

impl fmt::Debug for SelectionStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SelectionStrategy::MaxMemory => write!(f, "MaxMemory"),
            SelectionStrategy::MaxCompute => write!(f, "MaxCompute"),
            SelectionStrategy::LowLatency => write!(f, "LowLatency"),
            SelectionStrategy::Balanced => write!(f, "Balanced"),
            SelectionStrategy::CpuOnly => write!(f, "CpuOnly"),
            SelectionStrategy::PreferGpu => write!(f, "PreferGpu"),
            SelectionStrategy::Explicit(id) => write!(f, "Explicit({:?})", id),
            SelectionStrategy::Custom(_) => write!(f, "Custom(<function>)"),
        }
    }
}

impl Default for SelectionStrategy {
    fn default() -> Self {
        SelectionStrategy::Balanced
    }
}

/// Device selector for choosing optimal devices.
pub struct DeviceSelector {
    pool: Arc<DevicePool>,
    strategy: SelectionStrategy,
    min_memory_bytes: Option<u64>,
    required_features: Vec<super::DeviceFeature>,
    preferred_types: Vec<DeviceType>,
}

impl DeviceSelector {
    /// Create a new device selector.
    pub fn new(pool: Arc<DevicePool>) -> Self {
        Self {
            pool,
            strategy: SelectionStrategy::default(),
            min_memory_bytes: None,
            required_features: Vec::new(),
            preferred_types: Vec::new(),
        }
    }

    /// Set the selection strategy.
    pub fn with_strategy(mut self, strategy: SelectionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Require minimum memory.
    pub fn with_min_memory_gb(mut self, gb: f64) -> Self {
        self.min_memory_bytes = Some((gb * 1024.0 * 1024.0 * 1024.0) as u64);
        self
    }

    /// Require minimum memory in bytes.
    pub fn with_min_memory_bytes(mut self, bytes: u64) -> Self {
        self.min_memory_bytes = Some(bytes);
        self
    }

    /// Require specific features.
    pub fn with_required_features(mut self, features: Vec<super::DeviceFeature>) -> Self {
        self.required_features = features;
        self
    }

    /// Add a required feature.
    pub fn require_feature(mut self, feature: super::DeviceFeature) -> Self {
        self.required_features.push(feature);
        self
    }

    /// Set preferred device types (in order).
    pub fn with_preferred_types(mut self, types: Vec<DeviceType>) -> Self {
        self.preferred_types = types;
        self
    }

    /// Prefer GPU over CPU.
    pub fn prefer_gpu(mut self) -> Self {
        self.preferred_types = vec![
            DeviceType::CudaGpu,
            DeviceType::RocmGpu,
            DeviceType::MetalGpu,
            DeviceType::VulkanGpu,
            DeviceType::OpenClDevice,
            DeviceType::Cpu,
        ];
        self
    }

    /// Prefer CPU.
    pub fn prefer_cpu(mut self) -> Self {
        self.preferred_types = vec![DeviceType::Cpu];
        self.strategy = SelectionStrategy::CpuOnly;
        self
    }

    /// Select the best device matching criteria.
    pub fn select(&self) -> ComputeResult<Arc<dyn AnyDevice>> {
        let candidates = self.filter_candidates()?;

        if candidates.is_empty() {
            return Err(ComputeError::DeviceNotFound(
                "No devices match the selection criteria".to_string(),
            ));
        }

        let best = self.rank_candidates(&candidates);
        Ok(best)
    }

    /// Select multiple devices (for parallel execution).
    pub fn select_multiple(&self, max_count: usize) -> ComputeResult<Vec<Arc<dyn AnyDevice>>> {
        let candidates = self.filter_candidates()?;

        if candidates.is_empty() {
            return Err(ComputeError::DeviceNotFound(
                "No devices match the selection criteria".to_string(),
            ));
        }

        let mut ranked = self.rank_all_candidates(&candidates);
        ranked.truncate(max_count);
        Ok(ranked)
    }

    /// Filter devices that match the criteria.
    fn filter_candidates(&self) -> ComputeResult<Vec<Arc<dyn AnyDevice>>> {
        let all_devices = self.pool.all_devices();

        let candidates: Vec<_> = all_devices
            .into_iter()
            .filter(|d| {
                // Check availability
                if !d.is_available() {
                    return false;
                }

                let caps = d.capabilities();

                // Check minimum memory
                if let Some(min_mem) = self.min_memory_bytes {
                    if caps.total_memory < min_mem {
                        return false;
                    }
                }

                // Check required features
                for feature in &self.required_features {
                    if !caps.has_feature(*feature) {
                        return false;
                    }
                }

                // Check preferred types (if specified)
                if !self.preferred_types.is_empty()
                    && !self.preferred_types.contains(&d.device_type())
                {
                    return false;
                }

                // Check strategy-specific filters
                match &self.strategy {
                    SelectionStrategy::CpuOnly => d.device_type() == DeviceType::Cpu,
                    SelectionStrategy::Explicit(id) => d.device_id() == *id,
                    _ => true,
                }
            })
            .collect();

        Ok(candidates)
    }

    /// Rank candidates and return the best one.
    fn rank_candidates(&self, candidates: &[Arc<dyn AnyDevice>]) -> Arc<dyn AnyDevice> {
        candidates
            .iter()
            .max_by_key(|d| self.score_device(d.as_ref()))
            .cloned()
            .expect("candidates should not be empty")
    }

    /// Rank all candidates and return sorted list.
    fn rank_all_candidates(&self, candidates: &[Arc<dyn AnyDevice>]) -> Vec<Arc<dyn AnyDevice>> {
        let mut ranked: Vec<_> = candidates.to_vec();
        ranked.sort_by_key(|d| std::cmp::Reverse(self.score_device(d.as_ref())));
        ranked
    }

    /// Score a device based on strategy.
    fn score_device(&self, device: &dyn AnyDevice) -> i64 {
        let caps = device.capabilities();

        match &self.strategy {
            SelectionStrategy::MaxMemory => caps.total_memory as i64,

            SelectionStrategy::MaxCompute => {
                let mut score = caps.compute_units as i64 * 1000;
                score += (caps.peak_tflops_fp32 * 1000.0) as i64;
                score
            }

            SelectionStrategy::LowLatency => {
                // Prefer integrated devices and CPUs
                let mut score: i64 = 10000;
                if caps.is_integrated {
                    score += 5000;
                }
                if device.device_type() == DeviceType::Cpu {
                    score += 3000;
                }
                // Penalize based on memory (larger = slower access typically)
                score -= (caps.total_memory / (1024 * 1024 * 1024)) as i64;
                score
            }

            SelectionStrategy::Balanced => {
                let mut score: i64 = 0;

                // Type priority
                score += device.device_type().priority_score() as i64;

                // Memory (capped at 32GB for scoring)
                let mem_gb = (caps.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)).min(32.0);
                score += (mem_gb * 100.0) as i64;

                // Compute
                score += caps.compute_units as i64 * 10;
                score += (caps.peak_tflops_fp32 * 100.0) as i64;

                // Preference order bonus
                for (i, pref_type) in self.preferred_types.iter().enumerate() {
                    if device.device_type() == *pref_type {
                        score += ((self.preferred_types.len() - i) * 500) as i64;
                        break;
                    }
                }

                score
            }

            SelectionStrategy::CpuOnly => {
                if device.device_type() == DeviceType::Cpu {
                    caps.compute_units as i64 * 100 + (caps.total_memory / (1024 * 1024)) as i64
                } else {
                    0
                }
            }

            SelectionStrategy::PreferGpu => {
                let mut score = self.score_balanced(device, caps);
                if device.device_type().is_gpu() {
                    score += 10000;
                }
                score
            }

            SelectionStrategy::Explicit(id) => {
                if device.device_id() == *id {
                    i64::MAX
                } else {
                    0
                }
            }

            SelectionStrategy::Custom(scorer) => scorer(caps),
        }
    }

    /// Balanced scoring helper.
    fn score_balanced(&self, device: &dyn AnyDevice, caps: &DeviceCapabilities) -> i64 {
        let mut score: i64 = 0;
        score += device.device_type().priority_score() as i64;
        let mem_gb = (caps.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)).min(32.0);
        score += (mem_gb * 100.0) as i64;
        score += caps.compute_units as i64 * 10;
        score += (caps.peak_tflops_fp32 * 100.0) as i64;
        score
    }
}

/// Builder for creating device selectors.
pub struct DeviceSelectorBuilder {
    pool: Arc<DevicePool>,
    strategy: SelectionStrategy,
    min_memory_bytes: Option<u64>,
    required_features: Vec<super::DeviceFeature>,
    preferred_types: Vec<DeviceType>,
}

impl DeviceSelectorBuilder {
    /// Create a new builder.
    pub fn new(pool: Arc<DevicePool>) -> Self {
        Self {
            pool,
            strategy: SelectionStrategy::default(),
            min_memory_bytes: None,
            required_features: Vec::new(),
            preferred_types: Vec::new(),
        }
    }

    /// Set strategy.
    pub fn strategy(mut self, strategy: SelectionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set minimum memory in GB.
    pub fn min_memory_gb(mut self, gb: f64) -> Self {
        self.min_memory_bytes = Some((gb * 1024.0 * 1024.0 * 1024.0) as u64);
        self
    }

    /// Require a feature.
    pub fn require(mut self, feature: super::DeviceFeature) -> Self {
        self.required_features.push(feature);
        self
    }

    /// Prefer GPU.
    pub fn prefer_gpu(mut self) -> Self {
        self.preferred_types = vec![
            DeviceType::CudaGpu,
            DeviceType::RocmGpu,
            DeviceType::MetalGpu,
            DeviceType::VulkanGpu,
            DeviceType::OpenClDevice,
            DeviceType::Cpu,
        ];
        self
    }

    /// Build the selector.
    pub fn build(self) -> DeviceSelector {
        DeviceSelector {
            pool: self.pool,
            strategy: self.strategy,
            min_memory_bytes: self.min_memory_bytes,
            required_features: self.required_features,
            preferred_types: self.preferred_types,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{Hash, Hasher};

    // Use the mock from pool tests
    #[derive(Debug, Clone)]
    struct MockDevice {
        id: DeviceId,
        name: String,
        caps: DeviceCapabilities,
    }

    impl Hash for MockDevice {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.id.hash(state);
        }
    }

    impl PartialEq for MockDevice {
        fn eq(&self, other: &Self) -> bool {
            self.id == other.id
        }
    }

    impl Eq for MockDevice {}

    impl super::super::ComputeDevice for MockDevice {
        fn device_type(&self) -> DeviceType {
            self.id.device_type
        }
        fn device_id(&self) -> DeviceId {
            self.id
        }
        fn name(&self) -> &str {
            &self.name
        }
        fn capabilities(&self) -> &DeviceCapabilities {
            &self.caps
        }
    }

    fn create_test_pool() -> Arc<DevicePool> {
        let pool = Arc::new(DevicePool::new());

        // Add a CPU
        let cpu = MockDevice {
            id: DeviceId::cpu(0),
            name: "CPU".to_string(),
            caps: {
                let mut caps = DeviceCapabilities::default();
                caps.total_memory = 32 * 1024 * 1024 * 1024;
                caps.compute_units = 16;
                caps
            },
        };
        pool.register(cpu).unwrap();

        // Add a "GPU" with more compute
        let gpu = MockDevice {
            id: DeviceId::cuda(0),
            name: "GPU".to_string(),
            caps: {
                let mut caps = DeviceCapabilities::default();
                caps.total_memory = 8 * 1024 * 1024 * 1024;
                caps.compute_units = 128;
                caps.peak_tflops_fp32 = 10.0;
                caps
            },
        };
        pool.register(gpu).unwrap();

        pool
    }

    #[test]
    fn test_selector_max_memory() {
        let pool = create_test_pool();
        let selector = DeviceSelector::new(pool).with_strategy(SelectionStrategy::MaxMemory);

        let device = selector.select().unwrap();
        assert_eq!(device.device_type(), DeviceType::Cpu); // CPU has more memory
    }

    #[test]
    fn test_selector_max_compute() {
        let pool = create_test_pool();
        let selector = DeviceSelector::new(pool).with_strategy(SelectionStrategy::MaxCompute);

        let device = selector.select().unwrap();
        assert_eq!(device.device_type(), DeviceType::CudaGpu); // GPU has more compute
    }

    #[test]
    fn test_selector_cpu_only() {
        let pool = create_test_pool();
        let selector = DeviceSelector::new(pool).with_strategy(SelectionStrategy::CpuOnly);

        let device = selector.select().unwrap();
        assert_eq!(device.device_type(), DeviceType::Cpu);
    }

    #[test]
    fn test_selector_min_memory() {
        let pool = create_test_pool();
        let selector = DeviceSelector::new(pool).with_min_memory_gb(16.0);

        let device = selector.select().unwrap();
        assert_eq!(device.device_type(), DeviceType::Cpu); // Only CPU has 16+ GB
    }
}
