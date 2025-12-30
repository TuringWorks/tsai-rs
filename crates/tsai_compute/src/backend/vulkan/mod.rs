//! Vulkan compute backend.
//!
//! This backend provides cross-platform GPU computation via Vulkan,
//! supporting NVIDIA, AMD, Intel, and other Vulkan-capable GPUs.
//!
//! ## Feature Flag
//!
//! Enable the `vulkan` feature to use this backend:
//! ```toml
//! [dependencies]
//! tsai_compute = { version = "0.1", features = ["vulkan"] }
//! ```

use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::backend::{CommandEncoder, ComputeBackend, Fence};
use crate::device::{
    ComputeDevice, ComputeVersion, DeviceCapabilities, DeviceFeature, DeviceId, DeviceType,
};
use crate::error::{ComputeError, ComputeResult};
use crate::memory::{Buffer, BufferMapping, BufferUsage};

#[cfg(feature = "vulkan")]
use ash::vk;

/// Vulkan device representation.
#[derive(Debug, Clone)]
pub struct VulkanDevice {
    id: DeviceId,
    name: String,
    capabilities: DeviceCapabilities,
    #[cfg(feature = "vulkan")]
    physical_device_index: usize,
}

impl Hash for VulkanDevice {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl PartialEq for VulkanDevice {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for VulkanDevice {}

impl VulkanDevice {
    /// Create a Vulkan device from enumerated properties.
    #[cfg(feature = "vulkan")]
    fn from_properties(index: usize, props: &vk::PhysicalDeviceProperties) -> Self {
        let name = unsafe {
            std::ffi::CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .into_owned()
        };

        let api_version = props.api_version;
        let vendor_id = props.vendor_id;

        let mut capabilities = DeviceCapabilities::default();
        capabilities.compute_version = ComputeVersion::Vulkan { api_version };
        capabilities.vendor = match vendor_id {
            0x1002 => "AMD".to_string(),
            0x10DE => "NVIDIA".to_string(),
            0x8086 => "Intel".to_string(),
            0x13B5 => "ARM".to_string(),
            0x5143 => "Qualcomm".to_string(),
            0x106B => "Apple".to_string(),
            _ => format!("Unknown (0x{:04X})", vendor_id),
        };

        // Determine device type
        let is_discrete = props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
        let is_integrated = props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU;

        if is_discrete {
            capabilities.features.insert(DeviceFeature::DiscreteGpu);
        }

        // Estimate compute units from limits
        capabilities.compute_units = props.limits.max_compute_work_group_count[0].min(256) as u32;

        Self {
            id: DeviceId::vulkan(index as u32),
            name,
            capabilities,
            physical_device_index: index,
        }
    }

    /// Create a Vulkan device (non-vulkan feature fallback).
    #[cfg(not(feature = "vulkan"))]
    #[allow(dead_code)]
    fn new(index: u32, name: String, api_version: u32) -> Self {
        let mut capabilities = DeviceCapabilities::default();
        capabilities.compute_version = ComputeVersion::Vulkan { api_version };

        Self {
            id: DeviceId::vulkan(index),
            name,
            capabilities,
        }
    }
}

impl ComputeDevice for VulkanDevice {
    fn device_type(&self) -> DeviceType {
        DeviceType::VulkanGpu
    }

    fn device_id(&self) -> DeviceId {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }
}

/// Vulkan buffer wrapper.
pub struct VulkanBuffer {
    size: usize,
    usage: BufferUsage,
    device_id: DeviceId,
}

impl Buffer for VulkanBuffer {
    fn size(&self) -> usize {
        self.size
    }

    fn usage(&self) -> BufferUsage {
        self.usage
    }

    fn device_id(&self) -> DeviceId {
        self.device_id
    }

    fn map(&self) -> ComputeResult<BufferMapping<'_>> {
        Err(ComputeError::MappingFailed(
            "Vulkan buffer mapping not yet implemented".to_string(),
        ))
    }

    fn map_range(&self, _offset: usize, _size: usize) -> ComputeResult<BufferMapping<'_>> {
        Err(ComputeError::MappingFailed(
            "Vulkan buffer mapping not yet implemented".to_string(),
        ))
    }

    fn is_mapped(&self) -> bool {
        false
    }

    fn raw_ptr(&self) -> Option<*mut u8> {
        None
    }
}

/// Vulkan command encoder.
pub struct VulkanCommandEncoder {
    #[allow(dead_code)]
    commands: Vec<VulkanCommand>,
}

#[allow(dead_code)]
enum VulkanCommand {
    CopyHostToDevice { data: Vec<u8>, offset: usize },
    CopyDeviceToHost { offset: usize, size: usize },
    CopyBufferToBuffer { src_offset: usize, dst_offset: usize, size: usize },
    FillBuffer { offset: usize, size: usize, value: u8 },
    Barrier,
}

impl CommandEncoder for VulkanCommandEncoder {
    type Buffer = VulkanBuffer;

    fn copy_host_to_device(&mut self, src: &[u8], _dst: &Self::Buffer, offset: usize) {
        self.commands.push(VulkanCommand::CopyHostToDevice {
            data: src.to_vec(),
            offset,
        });
    }

    fn copy_device_to_host(&mut self, _src: &Self::Buffer, _dst: &mut [u8], offset: usize) {
        self.commands.push(VulkanCommand::CopyDeviceToHost {
            offset,
            size: 0,
        });
    }

    fn copy_buffer_to_buffer(
        &mut self,
        _src: &Self::Buffer,
        src_offset: usize,
        _dst: &Self::Buffer,
        dst_offset: usize,
        size: usize,
    ) {
        self.commands.push(VulkanCommand::CopyBufferToBuffer {
            src_offset,
            dst_offset,
            size,
        });
    }

    fn fill_buffer(&mut self, _buffer: &Self::Buffer, offset: usize, size: usize, value: u8) {
        self.commands.push(VulkanCommand::FillBuffer { offset, size, value });
    }

    fn barrier(&mut self) {
        self.commands.push(VulkanCommand::Barrier);
    }
}

/// Vulkan fence.
pub struct VulkanFence {
    completed: AtomicBool,
}

impl VulkanFence {
    fn new_signaled() -> Self {
        Self {
            completed: AtomicBool::new(true),
        }
    }
}

impl Fence for VulkanFence {
    fn is_signaled(&self) -> bool {
        self.completed.load(Ordering::SeqCst)
    }

    fn wait(&self) {
        while !self.is_signaled() {
            std::thread::yield_now();
        }
    }

    fn wait_timeout(&self, timeout_ms: u64) -> bool {
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_millis(timeout_ms);

        while !self.is_signaled() {
            if start.elapsed() >= timeout {
                return false;
            }
            std::thread::yield_now();
        }
        true
    }
}

/// Vulkan compute backend.
pub struct VulkanBackend {
    device: VulkanDevice,
    rng_seed: AtomicU64,
}

impl ComputeBackend for VulkanBackend {
    type Device = VulkanDevice;
    type Buffer = VulkanBuffer;
    type CommandEncoder = VulkanCommandEncoder;
    type Fence = VulkanFence;

    fn name() -> &'static str {
        "Vulkan"
    }

    fn enumerate_devices() -> ComputeResult<Vec<Self::Device>> {
        #[cfg(feature = "vulkan")]
        {
            use ash::Entry;

            // Load Vulkan library
            let entry = unsafe {
                Entry::load().map_err(|e| {
                    ComputeError::DiscoveryFailed(format!("Failed to load Vulkan: {}", e))
                })?
            };

            // Create a minimal instance for device enumeration
            let app_info = vk::ApplicationInfo::default()
                .application_name(c"tsai_compute")
                .application_version(vk::make_api_version(0, 0, 1, 0))
                .engine_name(c"tsai")
                .engine_version(vk::make_api_version(0, 0, 1, 0))
                .api_version(vk::API_VERSION_1_2);

            let create_info = vk::InstanceCreateInfo::default()
                .application_info(&app_info);

            let instance = unsafe {
                entry.create_instance(&create_info, None).map_err(|e| {
                    ComputeError::DiscoveryFailed(format!("Failed to create Vulkan instance: {:?}", e))
                })?
            };

            // Enumerate physical devices
            let physical_devices = unsafe {
                instance.enumerate_physical_devices().map_err(|e| {
                    ComputeError::DiscoveryFailed(format!("Failed to enumerate Vulkan devices: {:?}", e))
                })?
            };

            let mut devices = Vec::with_capacity(physical_devices.len());

            for (index, physical_device) in physical_devices.iter().enumerate() {
                let props = unsafe { instance.get_physical_device_properties(*physical_device) };

                // Only include GPU devices (discrete, integrated, or virtual)
                match props.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU
                    | vk::PhysicalDeviceType::INTEGRATED_GPU
                    | vk::PhysicalDeviceType::VIRTUAL_GPU => {
                        let device = VulkanDevice::from_properties(index, &props);

                        // Query memory properties
                        let mem_props = unsafe {
                            instance.get_physical_device_memory_properties(*physical_device)
                        };

                        // Sum device-local memory heaps
                        let mut total_memory: u64 = 0;
                        for i in 0..mem_props.memory_heap_count as usize {
                            let heap = mem_props.memory_heaps[i];
                            if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                                total_memory += heap.size;
                            }
                        }

                        // Update device with memory info
                        let mut device = device;
                        device.capabilities.total_memory = total_memory;
                        device.capabilities.available_memory = total_memory; // Approximate

                        devices.push(device);
                    }
                    _ => {
                        // Skip CPU and other device types
                    }
                }
            }

            // Clean up instance (we'll create a new one when actually using the device)
            unsafe { instance.destroy_instance(None) };

            Ok(devices)
        }

        #[cfg(not(feature = "vulkan"))]
        {
            Err(ComputeError::BackendInitFailed(
                "Vulkan support not compiled (enable 'vulkan' feature)".to_string(),
            ))
        }
    }

    fn new(device: &Self::Device) -> ComputeResult<Self> {
        #[cfg(feature = "vulkan")]
        {
            // Placeholder - would need to create logical device, queues, etc.
            Ok(Self {
                device: device.clone(),
                rng_seed: AtomicU64::new(0),
            })
        }

        #[cfg(not(feature = "vulkan"))]
        {
            let _ = device;
            Err(ComputeError::BackendInitFailed(
                "Vulkan support not compiled".to_string(),
            ))
        }
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn allocate_buffer(&self, size: usize, usage: BufferUsage) -> ComputeResult<Self::Buffer> {
        #[cfg(feature = "vulkan")]
        {
            // Placeholder - would use gpu-allocator for VMA-style allocation
            Ok(VulkanBuffer {
                size,
                usage,
                device_id: self.device.device_id(),
            })
        }

        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (size, usage);
            Err(ComputeError::BackendInitFailed(
                "Vulkan support not compiled".to_string(),
            ))
        }
    }

    fn create_encoder(&self) -> ComputeResult<Self::CommandEncoder> {
        Ok(VulkanCommandEncoder {
            commands: Vec::new(),
        })
    }

    fn submit(&self, _encoder: Self::CommandEncoder) -> ComputeResult<Self::Fence> {
        // Placeholder - would submit command buffer to queue
        Ok(VulkanFence::new_signaled())
    }

    fn wait(&self, fence: &Self::Fence) -> ComputeResult<()> {
        fence.wait();
        Ok(())
    }

    fn synchronize(&self) -> ComputeResult<()> {
        // Placeholder - would call vkQueueWaitIdle or similar
        Ok(())
    }

    fn seed(&self, seed: u64) {
        self.rng_seed.store(seed, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vulkan_not_available() {
        // Without vulkan feature, enumeration should fail gracefully
        let result = VulkanBackend::enumerate_devices();
        assert!(result.is_err());
    }
}
