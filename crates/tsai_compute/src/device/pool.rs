//! Device pool for managing heterogeneous compute devices.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use super::{ComputeDevice, DeviceCapabilities, DeviceId, DeviceType};
use crate::error::{ComputeError, ComputeResult};

/// Type-erased device wrapper for heterogeneous device pools.
pub trait AnyDevice: Send + Sync + 'static {
    /// Get the device as `Any` for downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Get the device ID.
    fn device_id(&self) -> DeviceId;

    /// Get the device name.
    fn name(&self) -> &str;

    /// Get the device capabilities.
    fn capabilities(&self) -> &DeviceCapabilities;

    /// Check if the device is available.
    fn is_available(&self) -> bool;

    /// Get the device type.
    fn device_type(&self) -> DeviceType {
        self.device_id().device_type
    }

    /// Get total memory.
    fn total_memory(&self) -> u64 {
        self.capabilities().total_memory
    }
}

/// Implement AnyDevice for any ComputeDevice.
impl<D: ComputeDevice + 'static> AnyDevice for D {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn device_id(&self) -> DeviceId {
        ComputeDevice::device_id(self)
    }

    fn name(&self) -> &str {
        ComputeDevice::name(self)
    }

    fn capabilities(&self) -> &DeviceCapabilities {
        ComputeDevice::capabilities(self)
    }

    fn is_available(&self) -> bool {
        ComputeDevice::is_available(self)
    }
}

/// Pool of heterogeneous compute devices.
///
/// The device pool manages discovery, tracking, and selection of compute
/// devices across all supported backends.
pub struct DevicePool {
    /// All registered devices.
    devices: RwLock<HashMap<DeviceId, Arc<dyn AnyDevice>>>,
    /// The default device for computation.
    default_device: RwLock<Option<DeviceId>>,
    /// Device type preference order for selection.
    preference_order: RwLock<Vec<DeviceType>>,
}

impl DevicePool {
    /// Create a new empty device pool.
    pub fn new() -> Self {
        Self {
            devices: RwLock::new(HashMap::new()),
            default_device: RwLock::new(None),
            preference_order: RwLock::new(vec![
                DeviceType::CudaGpu,
                DeviceType::RocmGpu,
                DeviceType::MetalGpu,
                DeviceType::VulkanGpu,
                DeviceType::OpenClDevice,
                DeviceType::Cpu,
            ]),
        }
    }

    /// Register a device in the pool.
    pub fn register<D: ComputeDevice + 'static>(&self, device: D) -> ComputeResult<()> {
        let id = device.device_id();
        let mut devices = self.devices.write();
        devices.insert(id, Arc::new(device));

        // Set as default if this is the first device
        let mut default = self.default_device.write();
        if default.is_none() {
            *default = Some(id);
        }

        Ok(())
    }

    /// Register multiple devices.
    pub fn register_all<D: ComputeDevice + 'static>(
        &self,
        devices: impl IntoIterator<Item = D>,
    ) -> ComputeResult<()> {
        for device in devices {
            self.register(device)?;
        }
        Ok(())
    }

    /// Get a device by ID.
    pub fn get(&self, id: &DeviceId) -> Option<Arc<dyn AnyDevice>> {
        self.devices.read().get(id).cloned()
    }

    /// Get the default device.
    pub fn default_device(&self) -> Option<Arc<dyn AnyDevice>> {
        let default_id = self.default_device.read().clone()?;
        self.get(&default_id)
    }

    /// Set the default device.
    pub fn set_default(&self, id: DeviceId) -> ComputeResult<()> {
        if !self.devices.read().contains_key(&id) {
            return Err(ComputeError::DeviceNotFound(id.to_string()));
        }
        *self.default_device.write() = Some(id);
        Ok(())
    }

    /// Get the best available device based on preferences and capabilities.
    pub fn best_device(&self) -> Option<Arc<dyn AnyDevice>> {
        let devices = self.devices.read();
        let prefs = self.preference_order.read();

        // Score all devices
        let mut best: Option<(i64, Arc<dyn AnyDevice>)> = None;

        for device in devices.values() {
            if !device.is_available() {
                continue;
            }

            let score = self.score_device(device.as_ref(), &prefs);

            if best.is_none() || score > best.as_ref().unwrap().0 {
                best = Some((score, device.clone()));
            }
        }

        best.map(|(_, d)| d)
    }

    /// Score a device for selection.
    fn score_device(&self, device: &dyn AnyDevice, prefs: &[DeviceType]) -> i64 {
        let mut score: i64 = 0;

        // Base score from device type priority
        score += device.device_type().priority_score() as i64;

        // Bonus for being higher in preference order
        if let Some(pos) = prefs.iter().position(|t| *t == device.device_type()) {
            score += ((prefs.len() - pos) * 100) as i64;
        }

        let caps = device.capabilities();

        // Memory score (GB * 100)
        score += (caps.total_memory / (1024 * 1024 * 1024)) as i64 * 100;

        // Compute units score
        score += caps.compute_units as i64 * 10;

        // TFLOPS score
        score += (caps.peak_tflops_fp32 * 100.0) as i64;

        // Bonus for tensor cores
        if caps.has_feature(super::DeviceFeature::TensorCores) {
            score += 500;
        }

        score
    }

    /// Get all devices of a specific type.
    pub fn devices_of_type(&self, device_type: DeviceType) -> Vec<Arc<dyn AnyDevice>> {
        self.devices
            .read()
            .values()
            .filter(|d| d.device_type() == device_type)
            .cloned()
            .collect()
    }

    /// Get all GPU devices.
    pub fn gpu_devices(&self) -> Vec<Arc<dyn AnyDevice>> {
        self.devices
            .read()
            .values()
            .filter(|d| d.device_type().is_gpu())
            .cloned()
            .collect()
    }

    /// Get all CPU devices.
    pub fn cpu_devices(&self) -> Vec<Arc<dyn AnyDevice>> {
        self.devices_of_type(DeviceType::Cpu)
    }

    /// Get all available devices.
    pub fn all_devices(&self) -> Vec<Arc<dyn AnyDevice>> {
        self.devices.read().values().cloned().collect()
    }

    /// Get the total number of devices.
    pub fn device_count(&self) -> usize {
        self.devices.read().len()
    }

    /// Check if any devices are available.
    pub fn has_devices(&self) -> bool {
        !self.devices.read().is_empty()
    }

    /// Filter devices by a capability predicate.
    pub fn filter_by_capability<F>(&self, predicate: F) -> Vec<Arc<dyn AnyDevice>>
    where
        F: Fn(&DeviceCapabilities) -> bool,
    {
        self.devices
            .read()
            .values()
            .filter(|d| predicate(d.capabilities()))
            .cloned()
            .collect()
    }

    /// Set device preference order.
    pub fn set_preference_order(&self, order: Vec<DeviceType>) {
        *self.preference_order.write() = order;
    }

    /// Get device preference order.
    pub fn preference_order(&self) -> Vec<DeviceType> {
        self.preference_order.read().clone()
    }

    /// Remove a device from the pool.
    pub fn remove(&self, id: &DeviceId) -> Option<Arc<dyn AnyDevice>> {
        let device = self.devices.write().remove(id);

        // Clear default if it was removed
        let mut default = self.default_device.write();
        if default.as_ref() == Some(id) {
            *default = None;
        }

        device
    }

    /// Clear all devices from the pool.
    pub fn clear(&self) {
        self.devices.write().clear();
        *self.default_device.write() = None;
    }

    /// Auto-select the best device as default.
    pub fn auto_select_default(&self) -> ComputeResult<()> {
        if let Some(device) = self.best_device() {
            self.set_default(device.device_id())?;
        }
        Ok(())
    }

    /// Print device summary.
    pub fn print_summary(&self) {
        let devices = self.devices.read();
        let default_id = self.default_device.read();

        println!("Device Pool Summary ({} devices)", devices.len());
        println!("{}", "-".repeat(60));

        for device in devices.values() {
            let is_default = default_id.as_ref() == Some(&device.device_id());
            let marker = if is_default { " [DEFAULT]" } else { "" };

            let caps = device.capabilities();
            println!(
                "{}{}: {} ({:.1} GB, {} CUs)",
                device.device_id(),
                marker,
                device.name(),
                caps.memory_gb(),
                caps.compute_units
            );
        }
    }
}

impl Default for DevicePool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{Hash, Hasher};

    // Mock device for testing
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

    impl ComputeDevice for MockDevice {
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

    #[test]
    fn test_device_pool_register() {
        let pool = DevicePool::new();

        let device = MockDevice {
            id: DeviceId::cpu(0),
            name: "Test CPU".to_string(),
            caps: DeviceCapabilities::default(),
        };

        pool.register(device.clone()).unwrap();
        assert_eq!(pool.device_count(), 1);

        let retrieved = pool.get(&DeviceId::cpu(0)).unwrap();
        assert_eq!(retrieved.name(), "Test CPU");
    }

    #[test]
    fn test_device_pool_default() {
        let pool = DevicePool::new();

        let cpu = MockDevice {
            id: DeviceId::cpu(0),
            name: "CPU".to_string(),
            caps: DeviceCapabilities::default(),
        };

        pool.register(cpu).unwrap();

        // First device becomes default
        let default = pool.default_device().unwrap();
        assert_eq!(default.device_id(), DeviceId::cpu(0));
    }

    #[test]
    fn test_device_pool_filter() {
        let pool = DevicePool::new();

        let cpu = MockDevice {
            id: DeviceId::cpu(0),
            name: "CPU".to_string(),
            caps: {
                let mut caps = DeviceCapabilities::default();
                caps.total_memory = 16 * 1024 * 1024 * 1024; // 16 GB
                caps
            },
        };

        let small_cpu = MockDevice {
            id: DeviceId::cpu(1),
            name: "Small CPU".to_string(),
            caps: {
                let mut caps = DeviceCapabilities::default();
                caps.total_memory = 4 * 1024 * 1024 * 1024; // 4 GB
                caps
            },
        };

        pool.register(cpu).unwrap();
        pool.register(small_cpu).unwrap();

        // Filter by memory > 8 GB
        let large_devices =
            pool.filter_by_capability(|caps| caps.total_memory > 8 * 1024 * 1024 * 1024);
        assert_eq!(large_devices.len(), 1);
        assert_eq!(large_devices[0].name(), "CPU");
    }
}
