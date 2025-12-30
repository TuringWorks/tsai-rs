//! OpenCL compute backend.
//!
//! This backend provides cross-platform GPU/CPU computation via OpenCL,
//! supporting a wide range of hardware from various vendors.
//!
//! ## Feature Flag
//!
//! Enable the `opencl` feature to use this backend:
//! ```toml
//! [dependencies]
//! tsai_compute = { version = "0.1", features = ["opencl"] }
//! ```

use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::backend::{CommandEncoder, ComputeBackend, Fence};
use crate::device::{ComputeDevice, ComputeVersion, DeviceCapabilities, DeviceFeature, DeviceId, DeviceType};
use crate::error::{ComputeError, ComputeResult};
use crate::memory::{Buffer, BufferMapping, BufferUsage};

/// OpenCL device representation.
#[derive(Debug, Clone)]
pub struct OpenClDevice {
    id: DeviceId,
    name: String,
    capabilities: DeviceCapabilities,
    #[cfg(feature = "opencl")]
    platform_index: usize,
    #[cfg(feature = "opencl")]
    device_index: usize,
}

impl Hash for OpenClDevice {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl PartialEq for OpenClDevice {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for OpenClDevice {}

impl OpenClDevice {
    /// Create an OpenCL device (non-opencl feature fallback).
    #[cfg(not(feature = "opencl"))]
    #[allow(dead_code)]
    fn new(platform_index: u32, device_index: u32, name: String, opencl_version: (u32, u32)) -> Self {
        let mut capabilities = DeviceCapabilities::default();
        capabilities.compute_version = ComputeVersion::OpenCl {
            major: opencl_version.0,
            minor: opencl_version.1,
        };

        Self {
            id: DeviceId::opencl(platform_index, device_index),
            name,
            capabilities,
        }
    }
}

/// Parse OpenCL version string like "OpenCL 3.0 ..." into (major, minor).
#[cfg(feature = "opencl")]
fn parse_opencl_version(version: &str) -> (u32, u32) {
    // Version string format: "OpenCL X.Y <vendor info>"
    let parts: Vec<&str> = version.split_whitespace().collect();
    if parts.len() >= 2 {
        let version_part = parts[1];
        let nums: Vec<&str> = version_part.split('.').collect();
        if nums.len() >= 2 {
            let major = nums[0].parse().unwrap_or(1);
            let minor = nums[1].parse().unwrap_or(0);
            return (major, minor);
        }
    }
    (1, 0) // Default to OpenCL 1.0
}

impl ComputeDevice for OpenClDevice {
    fn device_type(&self) -> DeviceType {
        DeviceType::OpenClDevice
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

/// OpenCL buffer wrapper.
pub struct OpenClBuffer {
    size: usize,
    usage: BufferUsage,
    device_id: DeviceId,
}

impl Buffer for OpenClBuffer {
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
            "OpenCL buffer mapping not yet implemented".to_string(),
        ))
    }

    fn map_range(&self, _offset: usize, _size: usize) -> ComputeResult<BufferMapping<'_>> {
        Err(ComputeError::MappingFailed(
            "OpenCL buffer mapping not yet implemented".to_string(),
        ))
    }

    fn is_mapped(&self) -> bool {
        false
    }

    fn raw_ptr(&self) -> Option<*mut u8> {
        None
    }
}

/// OpenCL command encoder.
#[allow(dead_code)]
pub struct OpenClCommandEncoder {
    commands: Vec<OpenClCommand>,
}

#[allow(dead_code)]
enum OpenClCommand {
    CopyHostToDevice { data: Vec<u8>, offset: usize },
    CopyDeviceToHost { offset: usize, size: usize },
    CopyBufferToBuffer { src_offset: usize, dst_offset: usize, size: usize },
    FillBuffer { offset: usize, size: usize, value: u8 },
    Barrier,
}

impl CommandEncoder for OpenClCommandEncoder {
    type Buffer = OpenClBuffer;

    fn copy_host_to_device(&mut self, src: &[u8], _dst: &Self::Buffer, offset: usize) {
        self.commands.push(OpenClCommand::CopyHostToDevice {
            data: src.to_vec(),
            offset,
        });
    }

    fn copy_device_to_host(&mut self, _src: &Self::Buffer, _dst: &mut [u8], offset: usize) {
        self.commands.push(OpenClCommand::CopyDeviceToHost {
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
        self.commands.push(OpenClCommand::CopyBufferToBuffer {
            src_offset,
            dst_offset,
            size,
        });
    }

    fn fill_buffer(&mut self, _buffer: &Self::Buffer, offset: usize, size: usize, value: u8) {
        self.commands.push(OpenClCommand::FillBuffer { offset, size, value });
    }

    fn barrier(&mut self) {
        self.commands.push(OpenClCommand::Barrier);
    }
}

/// OpenCL fence.
pub struct OpenClFence {
    completed: AtomicBool,
}

impl OpenClFence {
    fn new_signaled() -> Self {
        Self {
            completed: AtomicBool::new(true),
        }
    }
}

impl Fence for OpenClFence {
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

/// OpenCL compute backend.
pub struct OpenClBackend {
    device: OpenClDevice,
    rng_seed: AtomicU64,
}

impl ComputeBackend for OpenClBackend {
    type Device = OpenClDevice;
    type Buffer = OpenClBuffer;
    type CommandEncoder = OpenClCommandEncoder;
    type Fence = OpenClFence;

    fn name() -> &'static str {
        "OpenCL"
    }

    fn enumerate_devices() -> ComputeResult<Vec<Self::Device>> {
        #[cfg(feature = "opencl")]
        {
            use opencl3::platform::get_platforms;
            use opencl3::device::{Device, CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ACCELERATOR};

            // Get all OpenCL platforms
            let platforms = get_platforms().map_err(|e| {
                ComputeError::DiscoveryFailed(format!("Failed to get OpenCL platforms: {:?}", e))
            })?;

            let mut devices = Vec::new();
            let mut global_device_index = 0u32;

            for (platform_index, platform) in platforms.iter().enumerate() {
                // Get GPU and accelerator devices from this platform
                let device_ids = platform
                    .get_devices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR)
                    .unwrap_or_default();

                for (device_index, device_id) in device_ids.iter().enumerate() {
                    let device = Device::new(*device_id);

                    // Get device name
                    let name = device.name().unwrap_or_else(|_| "Unknown OpenCL Device".to_string());

                    // Get OpenCL version
                    let version_str = device.version().unwrap_or_default();
                    let (major, minor) = parse_opencl_version(&version_str);

                    // Get device capabilities
                    let mut capabilities = DeviceCapabilities::default();
                    capabilities.compute_version = ComputeVersion::OpenCl { major, minor };

                    // Get vendor
                    capabilities.vendor = device.vendor().unwrap_or_else(|_| "Unknown".to_string());

                    // Get compute units
                    capabilities.compute_units = device
                        .max_compute_units()
                        .unwrap_or(1) as u32;

                    // Get memory
                    capabilities.total_memory = device.global_mem_size().unwrap_or(0);
                    capabilities.available_memory = capabilities.total_memory;

                    // Check for features
                    if device.max_work_group_size().unwrap_or(0) >= 256 {
                        capabilities.features.insert(DeviceFeature::Compute);
                    }

                    devices.push(OpenClDevice {
                        id: DeviceId::opencl(platform_index as u32, device_index as u32),
                        name,
                        capabilities,
                        platform_index,
                        device_index,
                    });

                    global_device_index += 1;
                }
            }

            Ok(devices)
        }

        #[cfg(not(feature = "opencl"))]
        {
            Err(ComputeError::BackendInitFailed(
                "OpenCL support not compiled (enable 'opencl' feature)".to_string(),
            ))
        }
    }

    fn new(device: &Self::Device) -> ComputeResult<Self> {
        #[cfg(feature = "opencl")]
        {
            // Placeholder - would need to create context and command queue
            Ok(Self {
                device: device.clone(),
                rng_seed: AtomicU64::new(0),
            })
        }

        #[cfg(not(feature = "opencl"))]
        {
            let _ = device;
            Err(ComputeError::BackendInitFailed(
                "OpenCL support not compiled".to_string(),
            ))
        }
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn allocate_buffer(&self, size: usize, usage: BufferUsage) -> ComputeResult<Self::Buffer> {
        #[cfg(feature = "opencl")]
        {
            // Placeholder - would use clCreateBuffer
            Ok(OpenClBuffer {
                size,
                usage,
                device_id: self.device.device_id(),
            })
        }

        #[cfg(not(feature = "opencl"))]
        {
            let _ = (size, usage);
            Err(ComputeError::BackendInitFailed(
                "OpenCL support not compiled".to_string(),
            ))
        }
    }

    fn create_encoder(&self) -> ComputeResult<Self::CommandEncoder> {
        Ok(OpenClCommandEncoder {
            commands: Vec::new(),
        })
    }

    fn submit(&self, _encoder: Self::CommandEncoder) -> ComputeResult<Self::Fence> {
        // Placeholder - would submit commands to command queue
        Ok(OpenClFence::new_signaled())
    }

    fn wait(&self, fence: &Self::Fence) -> ComputeResult<()> {
        fence.wait();
        Ok(())
    }

    fn synchronize(&self) -> ComputeResult<()> {
        // Placeholder - would call clFinish or similar
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
    fn test_opencl_not_available() {
        // Without opencl feature, enumeration should fail gracefully
        let result = OpenClBackend::enumerate_devices();
        assert!(result.is_err());
    }
}
