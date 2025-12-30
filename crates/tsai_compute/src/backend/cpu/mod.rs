//! CPU compute backend with SIMD and NUMA support.
//!
//! This backend provides optimized CPU computation using:
//! - Runtime SIMD detection (AVX2, AVX-512, NEON)
//! - NUMA-aware memory allocation (optional)
//! - Parallel execution via Rayon

pub mod numa;
pub mod simd;

use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::backend::{CommandEncoder, ComputeBackend, Fence};
use crate::device::{ComputeDevice, DeviceCapabilities, DeviceId, DeviceType, SimdLevel};
use crate::error::{ComputeError, ComputeResult};
use crate::memory::{Buffer, BufferMapping, BufferUsage};

use self::numa::NumaAllocator;
use self::simd::SimdDispatch;

/// CPU device representation.
#[derive(Debug, Clone)]
pub struct CpuDevice {
    id: DeviceId,
    name: String,
    capabilities: DeviceCapabilities,
}

impl Hash for CpuDevice {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl PartialEq for CpuDevice {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for CpuDevice {}

impl CpuDevice {
    /// Create a new CPU device with automatic detection.
    pub fn new() -> Self {
        let simd = SimdLevel::detect();
        let numa_node = None; // Default, no specific NUMA affinity

        let mut capabilities = DeviceCapabilities::cpu(simd, numa_node);

        // Update with actual system info
        capabilities.compute_units = num_cpus::get() as u32;
        capabilities.vendor = Self::detect_vendor();

        Self {
            id: DeviceId::cpu(0),
            name: format!("CPU ({})", simd),
            capabilities,
        }
    }

    /// Create a CPU device for a specific NUMA node.
    pub fn numa_node(node: u32, simd: SimdLevel) -> Self {
        let capabilities = DeviceCapabilities::cpu(simd, Some(node));

        Self {
            id: DeviceId::with_numa(DeviceType::Cpu, 0, node),
            name: format!("CPU NUMA Node {} ({})", node, simd),
            capabilities,
        }
    }

    /// Detect CPU vendor.
    fn detect_vendor() -> String {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                // Could use CPUID to get vendor, simplified here
                "x86_64".to_string()
            } else {
                "x86_64".to_string()
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            "ARM".to_string()
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            "Unknown".to_string()
        }
    }
}

impl Default for CpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeDevice for CpuDevice {
    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
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

/// CPU buffer implementation.
pub struct CpuBuffer {
    data: Vec<u8>,
    usage: BufferUsage,
    device_id: DeviceId,
    is_mapped: AtomicBool,
}

impl CpuBuffer {
    /// Create a new CPU buffer.
    pub fn new(size: usize, usage: BufferUsage, device_id: DeviceId) -> Self {
        Self {
            data: vec![0u8; size],
            usage,
            device_id,
            is_mapped: AtomicBool::new(false),
        }
    }

    /// Create from existing data.
    pub fn from_data(data: Vec<u8>, usage: BufferUsage, device_id: DeviceId) -> Self {
        Self {
            data,
            usage,
            device_id,
            is_mapped: AtomicBool::new(false),
        }
    }

    /// Get the underlying data.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get mutable access to underlying data.
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

impl Buffer for CpuBuffer {
    fn size(&self) -> usize {
        self.data.len()
    }

    fn usage(&self) -> BufferUsage {
        self.usage
    }

    fn device_id(&self) -> DeviceId {
        self.device_id
    }

    fn map(&self) -> ComputeResult<BufferMapping<'_>> {
        self.map_range(0, self.data.len())
    }

    fn map_range(&self, offset: usize, size: usize) -> ComputeResult<BufferMapping<'_>> {
        if offset + size > self.data.len() {
            return Err(ComputeError::MappingFailed("Range out of bounds".to_string()));
        }

        if self.is_mapped.swap(true, Ordering::SeqCst) {
            return Err(ComputeError::MappingFailed("Buffer already mapped".to_string()));
        }

        let ptr = self.data.as_ptr().wrapping_add(offset) as *mut u8;
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
        Some(self.data.as_ptr() as *mut u8)
    }
}

/// CPU command encoder.
pub struct CpuCommandEncoder {
    commands: Vec<CpuCommand>,
}

enum CpuCommand {
    CopyH2D {
        src: Vec<u8>,
        dst_ptr: *mut u8,
        offset: usize,
    },
    CopyD2H {
        src_ptr: *const u8,
        dst: *mut u8,
        size: usize,
        offset: usize,
    },
    CopyD2D {
        src_ptr: *const u8,
        src_offset: usize,
        dst_ptr: *mut u8,
        dst_offset: usize,
        size: usize,
    },
    Fill {
        ptr: *mut u8,
        offset: usize,
        size: usize,
        value: u8,
    },
    Barrier,
}

// Safety: CpuCommand is only accessed from a single thread during execution
unsafe impl Send for CpuCommand {}

impl CpuCommandEncoder {
    fn new() -> Self {
        Self {
            commands: Vec::new(),
        }
    }

    fn execute(self) {
        for cmd in self.commands {
            match cmd {
                CpuCommand::CopyH2D { src, dst_ptr, offset } => {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src.as_ptr(),
                            dst_ptr.add(offset),
                            src.len(),
                        );
                    }
                }
                CpuCommand::CopyD2H { src_ptr, dst, size, offset } => {
                    unsafe {
                        std::ptr::copy_nonoverlapping(src_ptr.add(offset), dst, size);
                    }
                }
                CpuCommand::CopyD2D {
                    src_ptr,
                    src_offset,
                    dst_ptr,
                    dst_offset,
                    size,
                } => {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src_ptr.add(src_offset),
                            dst_ptr.add(dst_offset),
                            size,
                        );
                    }
                }
                CpuCommand::Fill { ptr, offset, size, value } => {
                    unsafe {
                        std::ptr::write_bytes(ptr.add(offset), value, size);
                    }
                }
                CpuCommand::Barrier => {
                    std::sync::atomic::fence(Ordering::SeqCst);
                }
            }
        }
    }
}

impl CommandEncoder for CpuCommandEncoder {
    type Buffer = CpuBuffer;

    fn copy_host_to_device(&mut self, src: &[u8], dst: &Self::Buffer, offset: usize) {
        self.commands.push(CpuCommand::CopyH2D {
            src: src.to_vec(),
            dst_ptr: dst.data.as_ptr() as *mut u8,
            offset,
        });
    }

    fn copy_device_to_host(&mut self, src: &Self::Buffer, dst: &mut [u8], offset: usize) {
        self.commands.push(CpuCommand::CopyD2H {
            src_ptr: src.data.as_ptr(),
            dst: dst.as_mut_ptr(),
            size: dst.len(),
            offset,
        });
    }

    fn copy_buffer_to_buffer(
        &mut self,
        src: &Self::Buffer,
        src_offset: usize,
        dst: &Self::Buffer,
        dst_offset: usize,
        size: usize,
    ) {
        self.commands.push(CpuCommand::CopyD2D {
            src_ptr: src.data.as_ptr(),
            src_offset,
            dst_ptr: dst.data.as_ptr() as *mut u8,
            dst_offset,
            size,
        });
    }

    fn fill_buffer(&mut self, buffer: &Self::Buffer, offset: usize, size: usize, value: u8) {
        self.commands.push(CpuCommand::Fill {
            ptr: buffer.data.as_ptr() as *mut u8,
            offset,
            size,
            value,
        });
    }

    fn barrier(&mut self) {
        self.commands.push(CpuCommand::Barrier);
    }
}

/// CPU fence (always signaled immediately for CPU).
pub struct CpuFence {
    signaled: AtomicBool,
}

impl CpuFence {
    fn new_signaled() -> Self {
        Self {
            signaled: AtomicBool::new(true),
        }
    }
}

impl Fence for CpuFence {
    fn is_signaled(&self) -> bool {
        self.signaled.load(Ordering::SeqCst)
    }

    fn wait(&self) {
        // CPU operations are synchronous, always signaled
    }

    fn wait_timeout(&self, _timeout_ms: u64) -> bool {
        true
    }
}

/// CPU compute backend.
pub struct CpuBackend {
    device: CpuDevice,
    simd: SimdDispatch,
    numa: NumaAllocator,
    rng_seed: AtomicU64,
}

impl CpuBackend {
    /// Get the SIMD dispatcher.
    pub fn simd(&self) -> &SimdDispatch {
        &self.simd
    }

    /// Get the NUMA allocator.
    pub fn numa(&self) -> &NumaAllocator {
        &self.numa
    }
}

impl ComputeBackend for CpuBackend {
    type Device = CpuDevice;
    type Buffer = CpuBuffer;
    type CommandEncoder = CpuCommandEncoder;
    type Fence = CpuFence;

    fn name() -> &'static str {
        "CPU"
    }

    fn enumerate_devices() -> ComputeResult<Vec<Self::Device>> {
        let numa = NumaAllocator::new();
        let simd = SimdLevel::detect();

        if numa.node_count() > 1 {
            // Create a device per NUMA node
            Ok((0..numa.node_count() as u32)
                .map(|node| CpuDevice::numa_node(node, simd))
                .collect())
        } else {
            Ok(vec![CpuDevice::new()])
        }
    }

    fn new(device: &Self::Device) -> ComputeResult<Self> {
        Ok(Self {
            device: device.clone(),
            simd: SimdDispatch::new(),
            numa: NumaAllocator::new(),
            rng_seed: AtomicU64::new(0),
        })
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn allocate_buffer(&self, size: usize, usage: BufferUsage) -> ComputeResult<Self::Buffer> {
        Ok(CpuBuffer::new(size, usage, self.device.device_id()))
    }

    fn create_encoder(&self) -> ComputeResult<Self::CommandEncoder> {
        Ok(CpuCommandEncoder::new())
    }

    fn submit(&self, encoder: Self::CommandEncoder) -> ComputeResult<Self::Fence> {
        // Execute immediately (CPU is synchronous)
        encoder.execute();
        Ok(CpuFence::new_signaled())
    }

    fn wait(&self, _fence: &Self::Fence) -> ComputeResult<()> {
        // CPU operations are synchronous
        Ok(())
    }

    fn synchronize(&self) -> ComputeResult<()> {
        // CPU is always synchronized
        std::sync::atomic::fence(Ordering::SeqCst);
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
    fn test_cpu_device() {
        let device = CpuDevice::new();
        println!("Device: {:?}", device);
        println!("Name: {}", device.name());
        println!("Capabilities: {:?}", device.capabilities());
    }

    #[test]
    fn test_cpu_backend() {
        let devices = CpuBackend::enumerate_devices().unwrap();
        assert!(!devices.is_empty());

        let backend = CpuBackend::new(&devices[0]).unwrap();
        assert_eq!(CpuBackend::name(), "CPU");

        // Test buffer allocation
        let buffer = backend
            .allocate_buffer(1024, BufferUsage::HOST_VISIBLE)
            .unwrap();
        assert_eq!(buffer.size(), 1024);
    }

    #[test]
    fn test_cpu_buffer_mapping() {
        let buffer = CpuBuffer::new(1024, BufferUsage::HOST_VISIBLE, DeviceId::cpu(0));

        {
            let mut mapping = buffer.map().unwrap();
            mapping[0] = 42;
            mapping[1] = 123;
        }

        let mapping = buffer.map().unwrap();
        assert_eq!(mapping[0], 42);
        assert_eq!(mapping[1], 123);
    }

    #[test]
    fn test_cpu_encoder() {
        let device = CpuDevice::new();
        let backend = CpuBackend::new(&device).unwrap();

        let buffer = backend
            .allocate_buffer(16, BufferUsage::HOST_VISIBLE)
            .unwrap();

        let src_data = vec![1u8, 2, 3, 4];
        let mut encoder = backend.create_encoder().unwrap();
        encoder.copy_host_to_device(&src_data, &buffer, 0);

        let fence = backend.submit(encoder).unwrap();
        backend.wait(&fence).unwrap();

        let mapping = buffer.map().unwrap();
        assert_eq!(&mapping[0..4], &[1, 2, 3, 4]);
    }
}
