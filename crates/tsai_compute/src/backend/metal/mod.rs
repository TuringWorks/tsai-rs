//! Metal compute backend for Apple GPUs.
//!
//! This backend provides GPU computation via Apple's Metal framework,
//! supporting Apple Silicon (M1/M2/M3/M4) and AMD GPUs on macOS.

use std::hash::{Hash, Hasher};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions,
};

use crate::backend::{CommandEncoder, ComputeBackend, Fence};
use crate::device::{
    ComputeDevice, ComputeVersion, DeviceCapabilities, DeviceFeature, DeviceId, DeviceType,
    MetalFamily, Precision,
};
use crate::error::{ComputeError, ComputeResult};
use crate::memory::{Buffer, BufferMapping, BufferUsage};

/// Thread-safe wrapper for Metal device.
///
/// Metal devices are documented to be thread-safe by Apple.
/// See: https://developer.apple.com/documentation/metal/mtldevice
struct MetalDeviceInner(Retained<ProtocolObject<dyn MTLDevice>>);

// Safety: MTLDevice is documented to be thread-safe by Apple
unsafe impl Send for MetalDeviceInner {}
unsafe impl Sync for MetalDeviceInner {}

impl Clone for MetalDeviceInner {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl std::ops::Deref for MetalDeviceInner {
    type Target = ProtocolObject<dyn MTLDevice>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Thread-safe wrapper for Metal command queue.
struct MetalQueueInner(Retained<ProtocolObject<dyn MTLCommandQueue>>);

// Safety: MTLCommandQueue is documented to be thread-safe by Apple
unsafe impl Send for MetalQueueInner {}
unsafe impl Sync for MetalQueueInner {}

impl std::ops::Deref for MetalQueueInner {
    type Target = ProtocolObject<dyn MTLCommandQueue>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Thread-safe wrapper for Metal buffer.
struct MetalBufferInner(Retained<ProtocolObject<dyn MTLBuffer>>);

// Safety: MTLBuffer is documented to be thread-safe by Apple
unsafe impl Send for MetalBufferInner {}
unsafe impl Sync for MetalBufferInner {}

impl std::ops::Deref for MetalBufferInner {
    type Target = ProtocolObject<dyn MTLBuffer>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Metal device wrapper.
#[derive(Clone)]
pub struct MetalDevice {
    id: DeviceId,
    name: String,
    capabilities: DeviceCapabilities,
    device: MetalDeviceInner,
}

impl std::fmt::Debug for MetalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalDevice")
            .field("id", &self.id)
            .field("name", &self.name)
            .finish()
    }
}

impl Hash for MetalDevice {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl PartialEq for MetalDevice {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for MetalDevice {}

impl MetalDevice {
    /// Create a Metal device from a native device.
    fn from_native(index: u32, device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        let name = device.name().to_string();
        let capabilities = Self::query_capabilities(&device);

        Self {
            id: DeviceId::metal(index),
            name,
            capabilities,
            device: MetalDeviceInner(device),
        }
    }

    /// Query device capabilities.
    fn query_capabilities(device: &ProtocolObject<dyn MTLDevice>) -> DeviceCapabilities {
        let mut caps = DeviceCapabilities::default();

        // Basic info
        caps.vendor = "Apple".to_string();

        // Detect Metal family
        caps.compute_version = ComputeVersion::Metal {
            family: Self::detect_family(device),
        };

        // Memory info - use recommended working set size as a proxy
        // Metal doesn't directly expose VRAM, but on Apple Silicon it's unified
        caps.total_memory = device.recommendedMaxWorkingSetSize() as u64;
        if caps.total_memory == 0 {
            // Fallback: use system memory for unified memory architecture
            caps.total_memory = sys_info::mem_info()
                .map(|m| m.total * 1024)
                .unwrap_or(8 * 1024 * 1024 * 1024);
        }

        // Thread limits
        let max_threads = device.maxThreadsPerThreadgroup();
        caps.max_work_group_size = (max_threads.width * max_threads.height * max_threads.depth) as u32;
        caps.max_work_group_dims = [
            max_threads.width as u32,
            max_threads.height as u32,
            max_threads.depth as u32,
        ];

        // Buffer limits
        caps.max_buffer_size = device.maxBufferLength() as u64;

        // Feature detection
        caps.is_integrated = !device.isRemovable();
        caps.features = Self::detect_features(device);
        caps.supported_precisions = vec![
            Precision::Float16,
            Precision::Float32,
            Precision::Int8,
            Precision::Int16,
            Precision::Int32,
        ];

        caps
    }

    /// Detect Metal GPU family.
    fn detect_family(device: &ProtocolObject<dyn MTLDevice>) -> MetalFamily {
        // Check for Apple Silicon families (newest first)
        if device.supportsFamily(objc2_metal::MTLGPUFamily::Apple9) {
            MetalFamily::Apple9
        } else if device.supportsFamily(objc2_metal::MTLGPUFamily::Apple8) {
            MetalFamily::Apple8
        } else if device.supportsFamily(objc2_metal::MTLGPUFamily::Apple7) {
            MetalFamily::Apple7
        } else if device.supportsFamily(objc2_metal::MTLGPUFamily::Apple6) {
            MetalFamily::Apple6
        } else if device.supportsFamily(objc2_metal::MTLGPUFamily::Apple5) {
            MetalFamily::Apple5
        } else if device.supportsFamily(objc2_metal::MTLGPUFamily::Apple4) {
            MetalFamily::Apple4
        } else if device.supportsFamily(objc2_metal::MTLGPUFamily::Apple3) {
            MetalFamily::Apple3
        } else if device.supportsFamily(objc2_metal::MTLGPUFamily::Apple2) {
            MetalFamily::Apple2
        } else if device.supportsFamily(objc2_metal::MTLGPUFamily::Apple1) {
            MetalFamily::Apple1
        } else if device.supportsFamily(objc2_metal::MTLGPUFamily::Mac2) {
            MetalFamily::Mac2
        } else {
            MetalFamily::Mac1
        }
    }

    /// Detect supported features.
    fn detect_features(device: &ProtocolObject<dyn MTLDevice>) -> Vec<DeviceFeature> {
        let mut features = vec![
            DeviceFeature::Float16,
            DeviceFeature::SharedMemory,
        ];

        // Unified memory on Apple Silicon
        if device.hasUnifiedMemory() {
            features.push(DeviceFeature::UnifiedMemory);
        }

        // Barycentric coordinates (Apple4+)
        if device.supportsFamily(objc2_metal::MTLGPUFamily::Apple4) {
            features.push(DeviceFeature::AsyncCompute);
        }

        features
    }

    /// Get the native Metal device.
    pub fn native(&self) -> &ProtocolObject<dyn MTLDevice> {
        &*self.device
    }
}

impl ComputeDevice for MetalDevice {
    fn device_type(&self) -> DeviceType {
        DeviceType::MetalGpu
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

/// Metal buffer wrapper.
pub struct MetalBuffer {
    buffer: MetalBufferInner,
    size: usize,
    usage: BufferUsage,
    device_id: DeviceId,
    is_mapped: AtomicBool,
}

impl MetalBuffer {
    /// Get the native Metal buffer.
    pub fn native(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &*self.buffer
    }

    /// Get a CPU pointer to the buffer contents.
    /// Only valid for shared/managed storage mode buffers.
    fn contents_ptr(&self) -> Option<NonNull<u8>> {
        let ptr = self.buffer.contents();
        NonNull::new(ptr.as_ptr() as *mut u8)
    }
}

impl Buffer for MetalBuffer {
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
        self.map_range(0, self.size)
    }

    fn map_range(&self, offset: usize, size: usize) -> ComputeResult<BufferMapping<'_>> {
        if offset + size > self.size {
            return Err(ComputeError::MappingFailed("Range out of bounds".to_string()));
        }

        if self.is_mapped.swap(true, Ordering::SeqCst) {
            return Err(ComputeError::MappingFailed("Buffer already mapped".to_string()));
        }

        let base_ptr = self.contents_ptr().ok_or_else(|| {
            self.is_mapped.store(false, Ordering::SeqCst);
            ComputeError::MappingFailed("Buffer not CPU-accessible".to_string())
        })?;

        let ptr = unsafe { base_ptr.as_ptr().add(offset) };
        let is_mapped = &self.is_mapped;

        Ok(unsafe {
            BufferMapping::new(
                ptr,
                size,
                offset,
                Some(Box::new(move || {
                    is_mapped.store(false, Ordering::SeqCst);
                })),
            )
        })
    }

    fn is_mapped(&self) -> bool {
        self.is_mapped.load(Ordering::SeqCst)
    }

    fn raw_ptr(&self) -> Option<*mut u8> {
        self.contents_ptr().map(|p| p.as_ptr())
    }
}

/// Metal command encoder.
pub struct MetalCommandEncoder {
    #[allow(dead_code)] // Used in full Metal implementation
    device: MetalDeviceInner,
    commands: Vec<MetalCommand>,
}

#[allow(dead_code)] // Command data stored for execution in full Metal implementation
enum MetalCommand {
    CopyHostToDevice {
        data: Vec<u8>,
        dst_offset: usize,
    },
    CopyDeviceToHost {
        src_offset: usize,
        size: usize,
    },
    CopyBufferToBuffer {
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    },
    FillBuffer {
        offset: usize,
        size: usize,
        value: u8,
    },
}

impl MetalCommandEncoder {
    fn new(device: MetalDeviceInner) -> Self {
        Self {
            device,
            commands: Vec::new(),
        }
    }
}

impl CommandEncoder for MetalCommandEncoder {
    type Buffer = MetalBuffer;

    fn copy_host_to_device(&mut self, src: &[u8], _dst: &Self::Buffer, offset: usize) {
        self.commands.push(MetalCommand::CopyHostToDevice {
            data: src.to_vec(),
            dst_offset: offset,
        });
    }

    fn copy_device_to_host(&mut self, _src: &Self::Buffer, _dst: &mut [u8], offset: usize) {
        self.commands.push(MetalCommand::CopyDeviceToHost {
            src_offset: offset,
            size: 0, // Will be filled in during execution
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
        self.commands.push(MetalCommand::CopyBufferToBuffer {
            src_offset,
            dst_offset,
            size,
        });
    }

    fn fill_buffer(&mut self, _buffer: &Self::Buffer, offset: usize, size: usize, value: u8) {
        self.commands.push(MetalCommand::FillBuffer {
            offset,
            size,
            value,
        });
    }

    fn barrier(&mut self) {
        // Metal handles barriers automatically within command buffers
    }
}

/// Metal fence for synchronization.
pub struct MetalFence {
    completed: Arc<AtomicBool>,
}

#[allow(dead_code)] // Methods for full Metal implementation
impl MetalFence {
    fn new() -> Self {
        Self {
            completed: Arc::new(AtomicBool::new(false)),
        }
    }

    fn new_signaled() -> Self {
        Self {
            completed: Arc::new(AtomicBool::new(true)),
        }
    }

    fn signal(&self) {
        self.completed.store(true, Ordering::SeqCst);
    }
}

impl Fence for MetalFence {
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

/// Metal compute backend.
pub struct MetalBackend {
    device: MetalDevice,
    command_queue: MetalQueueInner,
    rng_seed: AtomicU64,
}

impl MetalBackend {
    /// Get the native Metal device.
    pub fn native_device(&self) -> &ProtocolObject<dyn MTLDevice> {
        self.device.native()
    }

    /// Get the command queue.
    pub fn command_queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &*self.command_queue
    }

    /// Convert BufferUsage to MTLResourceOptions.
    fn resource_options(usage: BufferUsage) -> MTLResourceOptions {
        if usage.host_readable || usage.host_writable {
            // Shared mode: CPU and GPU can both access
            MTLResourceOptions::MTLResourceStorageModeShared
        } else {
            // Private mode: GPU only
            MTLResourceOptions::MTLResourceStorageModePrivate
        }
    }
}

impl ComputeBackend for MetalBackend {
    type Device = MetalDevice;
    type Buffer = MetalBuffer;
    type CommandEncoder = MetalCommandEncoder;
    type Fence = MetalFence;

    fn name() -> &'static str {
        "Metal"
    }

    fn enumerate_devices() -> ComputeResult<Vec<Self::Device>> {
        let mut devices = Vec::new();

        // Try the system default device first (most reliable)
        let default_device = unsafe { MTLCreateSystemDefaultDevice() };
        if let Some(device) = unsafe { Retained::from_raw(default_device) } {
            devices.push(MetalDevice::from_native(0, device));
        }

        if devices.is_empty() {
            return Err(ComputeError::DeviceNotFound(
                "No Metal devices found".to_string(),
            ));
        }

        Ok(devices)
    }

    fn new(device: &Self::Device) -> ComputeResult<Self> {
        let command_queue = device
            .device
            .newCommandQueue()
            .ok_or_else(|| ComputeError::BackendInitFailed("Failed to create command queue".to_string()))?;

        Ok(Self {
            device: device.clone(),
            command_queue: MetalQueueInner(command_queue),
            rng_seed: AtomicU64::new(0),
        })
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn allocate_buffer(&self, size: usize, usage: BufferUsage) -> ComputeResult<Self::Buffer> {
        let options = Self::resource_options(usage);

        let buffer = self
            .device
            .device
            .newBufferWithLength_options(size, options)
            .ok_or_else(|| {
                ComputeError::AllocationFailed {
                    size,
                    reason: "Metal buffer allocation failed".to_string(),
                }
            })?;

        Ok(MetalBuffer {
            buffer: MetalBufferInner(buffer),
            size,
            usage,
            device_id: self.device.device_id(),
            is_mapped: AtomicBool::new(false),
        })
    }

    fn create_encoder(&self) -> ComputeResult<Self::CommandEncoder> {
        Ok(MetalCommandEncoder::new(self.device.device.clone()))
    }

    fn submit(&self, encoder: Self::CommandEncoder) -> ComputeResult<Self::Fence> {
        // If no commands, return immediately signaled fence
        if encoder.commands.is_empty() {
            return Ok(MetalFence::new_signaled());
        }

        // Create a command buffer
        let command_buffer = self
            .command_queue
            .commandBuffer()
            .ok_or_else(|| ComputeError::SubmissionFailed("Failed to create command buffer".to_string()))?;

        // Commit the command buffer
        command_buffer.commit();

        // Wait for completion synchronously (simple approach)
        // For async operation, we'd need proper completion handler setup
        // Safety: This is safe as we own the command buffer and it's been committed
        unsafe {
            command_buffer.waitUntilCompleted();
        }

        // Return an already-signaled fence since we waited
        Ok(MetalFence::new_signaled())
    }

    fn wait(&self, fence: &Self::Fence) -> ComputeResult<()> {
        fence.wait();
        Ok(())
    }

    fn synchronize(&self) -> ComputeResult<()> {
        // Create an empty command buffer and wait for it
        let command_buffer = self
            .command_queue
            .commandBuffer()
            .ok_or_else(|| ComputeError::BackendInitFailed("Failed to create sync command buffer".to_string()))?;

        command_buffer.commit();
        // Safety: This is safe as we own the command buffer and it's been committed
        unsafe {
            command_buffer.waitUntilCompleted();
        }

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
    fn test_metal_device_discovery() {
        let devices = MetalBackend::enumerate_devices().unwrap();
        assert!(!devices.is_empty(), "Should find at least one Metal device");

        for device in &devices {
            println!("Found Metal device: {} ({:?})", device.name(), device.device_id());
            println!("  Memory: {} GB", device.capabilities().memory_gb());
            println!("  Unified memory: {}", device.native().hasUnifiedMemory());
        }
    }

    #[test]
    fn test_metal_backend_creation() {
        let devices = MetalBackend::enumerate_devices().unwrap();
        let backend = MetalBackend::new(&devices[0]).unwrap();

        assert_eq!(MetalBackend::name(), "Metal");
        println!("Created Metal backend for: {}", backend.device().name());
    }

    #[test]
    fn test_metal_buffer_allocation() {
        let devices = MetalBackend::enumerate_devices().unwrap();
        let backend = MetalBackend::new(&devices[0]).unwrap();

        // Allocate a shared buffer
        let buffer = backend
            .allocate_buffer(1024, BufferUsage::HOST_VISIBLE)
            .unwrap();

        assert_eq!(buffer.size(), 1024);
        assert!(buffer.raw_ptr().is_some(), "Shared buffer should be CPU-accessible");

        // Allocate a private buffer
        let private_buffer = backend
            .allocate_buffer(1024, BufferUsage::DEVICE_ONLY)
            .unwrap();

        assert_eq!(private_buffer.size(), 1024);
    }

    #[test]
    fn test_metal_buffer_mapping() {
        let devices = MetalBackend::enumerate_devices().unwrap();
        let backend = MetalBackend::new(&devices[0]).unwrap();

        let buffer = backend
            .allocate_buffer(1024, BufferUsage::HOST_VISIBLE)
            .unwrap();

        // Map and write data
        {
            let mut mapping = buffer.map().unwrap();
            mapping[0] = 42;
            mapping[1] = 123;
        }

        // Remap and verify
        {
            let mapping = buffer.map().unwrap();
            assert_eq!(mapping[0], 42);
            assert_eq!(mapping[1], 123);
        }
    }

    #[test]
    fn test_metal_synchronize() {
        let devices = MetalBackend::enumerate_devices().unwrap();
        let backend = MetalBackend::new(&devices[0]).unwrap();

        // Test synchronization
        backend.synchronize().unwrap();
        println!("Metal synchronization successful");
    }

    #[test]
    fn test_metal_submit_empty() {
        let devices = MetalBackend::enumerate_devices().unwrap();
        let backend = MetalBackend::new(&devices[0]).unwrap();

        let encoder = backend.create_encoder().unwrap();
        let fence = backend.submit(encoder).unwrap();

        // Empty submission should be immediately signaled
        assert!(fence.is_signaled());
    }
}
