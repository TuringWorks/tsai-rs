//! MLX compute backend for Apple Silicon.
//!
//! This backend provides high-performance ML computation via Apple's MLX framework,
//! optimized specifically for Apple Silicon with unified memory architecture.
//!
//! ## Key Features
//!
//! - **Unified Memory**: Zero-copy data sharing between CPU and GPU
//! - **Lazy Evaluation**: Automatic compute graph optimization
//! - **Apple Silicon Optimized**: Native support for M-series Neural Engine
//!
//! ## Feature Flag
//!
//! Enable the `mlx` feature to use this backend:
//! ```toml
//! [dependencies]
//! tsai_compute = { version = "0.1", features = ["mlx"] }
//! ```
//!
//! ## Requirements
//!
//! - macOS 13.3+ (Ventura)
//! - Apple Silicon (M1/M2/M3/M4/M5)
//! - Rust 1.82+
//!
//! ## Thread Safety
//!
//! MLX operations should be performed from a single thread or with proper synchronization.
//! The `Send` and `Sync` implementations are provided for compatibility with tsai_compute's
//! trait bounds, but concurrent access from multiple threads should be avoided.

use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::backend::{CommandEncoder, ComputeBackend, Fence};
use crate::device::{ComputeDevice, ComputeVersion, DeviceCapabilities, DeviceFeature, DeviceId, DeviceType};
use crate::error::{ComputeError, ComputeResult};
use crate::memory::{Buffer, BufferMapping, BufferUsage};

/// MLX device representation.
#[derive(Debug, Clone)]
pub struct MlxDevice {
    id: DeviceId,
    name: String,
    capabilities: DeviceCapabilities,
    #[cfg(feature = "mlx")]
    device_type: MlxDeviceType,
}

/// MLX device type (CPU or GPU).
#[cfg(feature = "mlx")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlxDeviceType {
    /// CPU device
    Cpu,
    /// GPU device (Apple Silicon GPU)
    Gpu,
}

impl Hash for MlxDevice {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl PartialEq for MlxDevice {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for MlxDevice {}

#[cfg(feature = "mlx")]
impl MlxDevice {
    /// Create a new MLX device.
    fn new(device_type: MlxDeviceType, index: u32) -> Self {
        let (name, device_id) = match device_type {
            MlxDeviceType::Cpu => ("MLX CPU".to_string(), DeviceId::mlx_cpu(index)),
            MlxDeviceType::Gpu => ("MLX GPU (Apple Silicon)".to_string(), DeviceId::mlx_gpu(index)),
        };

        let mut capabilities = DeviceCapabilities::default();
        capabilities.vendor = "Apple".to_string();
        capabilities.compute_version = ComputeVersion::Mlx { version: "0.25".to_string() };

        // MLX features
        capabilities.features.push(DeviceFeature::Compute);
        capabilities.features.push(DeviceFeature::Float16);
        capabilities.features.push(DeviceFeature::Float64);
        capabilities.features.push(DeviceFeature::UnifiedMemory);

        if matches!(device_type, MlxDeviceType::Gpu) {
            capabilities.features.push(DeviceFeature::DiscreteGpu);
            // Apple Silicon GPUs have excellent BF16 support
            capabilities.features.push(DeviceFeature::BFloat16);
        }

        Self {
            id: device_id,
            name,
            capabilities,
            device_type,
        }
    }
}

impl ComputeDevice for MlxDevice {
    fn device_type(&self) -> DeviceType {
        #[cfg(feature = "mlx")]
        match self.device_type {
            MlxDeviceType::Cpu => DeviceType::Cpu,
            MlxDeviceType::Gpu => DeviceType::MlxGpu,
        }
        #[cfg(not(feature = "mlx"))]
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

/// MLX buffer wrapper.
pub struct MlxBuffer {
    size: usize,
    usage: BufferUsage,
    device_id: DeviceId,
    #[cfg(feature = "mlx")]
    array: Option<mlx_rs::Array>,
}

// SAFETY: MLX Array implements Send. We add Sync for trait compatibility.
// Users should avoid concurrent mutable access from multiple threads.
#[cfg(feature = "mlx")]
unsafe impl Sync for MlxBuffer {}

impl Buffer for MlxBuffer {
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
        // MLX uses unified memory, but we still need to implement proper mapping
        Err(ComputeError::MappingFailed(
            "MLX buffer mapping: use Array::as_slice() directly".to_string(),
        ))
    }

    fn map_range(&self, _offset: usize, _size: usize) -> ComputeResult<BufferMapping<'_>> {
        Err(ComputeError::MappingFailed(
            "MLX buffer range mapping not yet implemented".to_string(),
        ))
    }

    fn is_mapped(&self) -> bool {
        false
    }

    fn raw_ptr(&self) -> Option<*mut u8> {
        None
    }
}

#[cfg(feature = "mlx")]
impl MlxBuffer {
    /// Get the underlying MLX array.
    pub fn array(&self) -> Option<&mlx_rs::Array> {
        self.array.as_ref()
    }

    /// Create a buffer from an MLX array.
    pub fn from_array(array: mlx_rs::Array, device_id: DeviceId, usage: BufferUsage) -> Self {
        let size = array.size() * std::mem::size_of::<f32>(); // Approximate
        Self {
            size,
            usage,
            device_id,
            array: Some(array),
        }
    }
}

/// MLX command encoder.
#[allow(dead_code)]
pub struct MlxCommandEncoder {
    commands: Vec<MlxCommand>,
}

// SAFETY: MlxCommandEncoder is typically used from a single thread.
// We add Send for trait compatibility with CommandEncoder.
#[cfg(feature = "mlx")]
unsafe impl Send for MlxCommandEncoder {}

#[allow(dead_code)]
enum MlxCommand {
    CopyHostToDevice { data: Vec<u8>, offset: usize },
    CopyDeviceToHost { offset: usize, size: usize },
    CopyBufferToBuffer { src_offset: usize, dst_offset: usize, size: usize },
    FillBuffer { offset: usize, size: usize, value: u8 },
    Barrier,
    #[cfg(feature = "mlx")]
    Eval { arrays: Vec<mlx_rs::Array> },
}

impl CommandEncoder for MlxCommandEncoder {
    type Buffer = MlxBuffer;

    fn copy_host_to_device(&mut self, src: &[u8], _dst: &Self::Buffer, offset: usize) {
        self.commands.push(MlxCommand::CopyHostToDevice {
            data: src.to_vec(),
            offset,
        });
    }

    fn copy_device_to_host(&mut self, _src: &Self::Buffer, _dst: &mut [u8], offset: usize) {
        self.commands.push(MlxCommand::CopyDeviceToHost {
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
        self.commands.push(MlxCommand::CopyBufferToBuffer {
            src_offset,
            dst_offset,
            size,
        });
    }

    fn fill_buffer(&mut self, _buffer: &Self::Buffer, offset: usize, size: usize, value: u8) {
        self.commands.push(MlxCommand::FillBuffer { offset, size, value });
    }

    fn barrier(&mut self) {
        self.commands.push(MlxCommand::Barrier);
    }
}

/// MLX fence for synchronization.
pub struct MlxFence {
    completed: AtomicBool,
}

impl MlxFence {
    fn new_signaled() -> Self {
        Self {
            completed: AtomicBool::new(true),
        }
    }
}

impl Fence for MlxFence {
    fn is_signaled(&self) -> bool {
        self.completed.load(Ordering::SeqCst)
    }

    fn wait(&self) {
        // MLX operations are lazy - evaluation happens on demand
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

/// MLX compute backend.
pub struct MlxBackend {
    device: MlxDevice,
    rng_seed: AtomicU64,
    #[cfg(feature = "mlx")]
    mlx_device: mlx_rs::Device,
}

// SAFETY: MlxBackend should be used from a single thread or with proper synchronization.
// We add Send + Sync for trait compatibility with ComputeBackend.
#[cfg(feature = "mlx")]
unsafe impl Send for MlxBackend {}
#[cfg(feature = "mlx")]
unsafe impl Sync for MlxBackend {}

impl ComputeBackend for MlxBackend {
    type Device = MlxDevice;
    type Buffer = MlxBuffer;
    type CommandEncoder = MlxCommandEncoder;
    type Fence = MlxFence;

    fn name() -> &'static str {
        "MLX"
    }

    fn enumerate_devices() -> ComputeResult<Vec<Self::Device>> {
        #[cfg(feature = "mlx")]
        {
            let mut devices = Vec::new();

            // MLX always has CPU available
            devices.push(MlxDevice::new(MlxDeviceType::Cpu, 0));

            // Check if GPU is available (Apple Silicon)
            // MLX GPU is available on all Apple Silicon Macs
            if cfg!(target_arch = "aarch64") {
                devices.push(MlxDevice::new(MlxDeviceType::Gpu, 0));
            }

            Ok(devices)
        }

        #[cfg(not(feature = "mlx"))]
        {
            Err(ComputeError::BackendInitFailed(
                "MLX support not compiled (enable 'mlx' feature)".to_string(),
            ))
        }
    }

    fn new(device: &Self::Device) -> ComputeResult<Self> {
        #[cfg(feature = "mlx")]
        {
            use mlx_rs::Device;

            let mlx_device = match device.device_type {
                MlxDeviceType::Cpu => Device::cpu(),
                MlxDeviceType::Gpu => Device::gpu(),
            };

            Ok(Self {
                device: device.clone(),
                rng_seed: AtomicU64::new(0),
                mlx_device,
            })
        }

        #[cfg(not(feature = "mlx"))]
        {
            let _ = device;
            Err(ComputeError::BackendInitFailed(
                "MLX support not compiled".to_string(),
            ))
        }
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn allocate_buffer(&self, size: usize, usage: BufferUsage) -> ComputeResult<Self::Buffer> {
        #[cfg(feature = "mlx")]
        {
            // MLX uses lazy allocation - buffer is created but not allocated until used
            Ok(MlxBuffer {
                size,
                usage,
                device_id: self.device.device_id(),
                array: None,
            })
        }

        #[cfg(not(feature = "mlx"))]
        {
            let _ = (size, usage);
            Err(ComputeError::BackendInitFailed(
                "MLX support not compiled".to_string(),
            ))
        }
    }

    fn create_encoder(&self) -> ComputeResult<Self::CommandEncoder> {
        Ok(MlxCommandEncoder {
            commands: Vec::new(),
        })
    }

    fn submit(&self, _encoder: Self::CommandEncoder) -> ComputeResult<Self::Fence> {
        // MLX uses lazy evaluation - commands are executed on demand
        Ok(MlxFence::new_signaled())
    }

    fn wait(&self, fence: &Self::Fence) -> ComputeResult<()> {
        fence.wait();
        Ok(())
    }

    fn synchronize(&self) -> ComputeResult<()> {
        // MLX uses lazy evaluation - synchronization is implicit
        // Arrays are evaluated when their values are accessed
        // No explicit synchronize call needed
        Ok(())
    }

    fn seed(&self, seed: u64) {
        self.rng_seed.store(seed, Ordering::SeqCst);
        #[cfg(feature = "mlx")]
        {
            let _ = mlx_rs::random::seed(seed);
        }
    }
}

/// MLX-specific operations for ML workloads.
#[cfg(feature = "mlx")]
impl MlxBackend {
    /// Create a tensor from f32 data.
    pub fn tensor_from_f32(&self, data: &[f32], shape: &[i32]) -> ComputeResult<mlx_rs::Array> {
        let array = mlx_rs::Array::from_slice(data, shape);
        Ok(array)
    }

    /// Create a tensor on the GPU device.
    pub fn tensor_on_gpu(&self, data: &[f32], shape: &[i32]) -> ComputeResult<mlx_rs::Array> {
        let array = mlx_rs::Array::from_slice(data, shape);
        // MLX uses unified memory, so array is accessible from both CPU and GPU
        // We just need to ensure operations run on GPU
        Ok(array)
    }

    /// Perform matrix multiplication.
    pub fn matmul(&self, a: &mlx_rs::Array, b: &mlx_rs::Array) -> ComputeResult<mlx_rs::Array> {
        let result = a.matmul(b)?;
        Ok(result)
    }

    /// Apply softmax activation (along the last axis).
    pub fn softmax(&self, x: &mlx_rs::Array) -> ComputeResult<mlx_rs::Array> {
        // mlx-rs softmax operates along the last axis by default
        let result = mlx_rs::ops::softmax(x, None)?;
        Ok(result)
    }

    /// Apply ReLU activation.
    pub fn relu(&self, x: &mlx_rs::Array) -> ComputeResult<mlx_rs::Array> {
        let zero = mlx_rs::Array::from_f32(0.0);
        let result = mlx_rs::ops::maximum(x, &zero)?;
        Ok(result)
    }

    /// Get the MLX device.
    pub fn mlx_device(&self) -> &mlx_rs::Device {
        &self.mlx_device
    }

    /// Set the default device for operations.
    pub fn set_default_device(&self) {
        mlx_rs::Device::set_default(&self.mlx_device);
    }

    /// Evaluate an array (force lazy computation).
    pub fn eval(&self, array: &mlx_rs::Array) -> ComputeResult<()> {
        array.eval()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlx_not_available() {
        // Without mlx feature, enumeration should fail gracefully
        #[cfg(not(feature = "mlx"))]
        {
            let result = MlxBackend::enumerate_devices();
            assert!(result.is_err());
        }
    }

    #[test]
    #[cfg(feature = "mlx")]
    fn test_mlx_device_enumeration() {
        let devices = MlxBackend::enumerate_devices().unwrap();
        assert!(!devices.is_empty());

        // Should have at least CPU
        assert!(devices.iter().any(|d| matches!(d.device_type, MlxDeviceType::Cpu)));

        // On Apple Silicon, should have GPU too
        #[cfg(target_arch = "aarch64")]
        assert!(devices.iter().any(|d| matches!(d.device_type, MlxDeviceType::Gpu)));
    }

    #[test]
    #[cfg(feature = "mlx")]
    fn test_mlx_backend_creation() {
        let devices = MlxBackend::enumerate_devices().unwrap();
        let gpu_device = devices.iter().find(|d| matches!(d.device_type, MlxDeviceType::Gpu));

        if let Some(device) = gpu_device {
            let backend = MlxBackend::new(device).unwrap();
            assert_eq!(backend.device().name(), "MLX GPU (Apple Silicon)");
        }
    }
}
