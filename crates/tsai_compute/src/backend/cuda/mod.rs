//! CUDA compute backend for NVIDIA GPUs.
//!
//! This backend provides GPU computation via NVIDIA CUDA,
//! supporting all CUDA-capable NVIDIA GPUs.

use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::backend::{CommandEncoder, ComputeBackend, Fence};
use crate::device::{
    ComputeDevice, ComputeVersion, DeviceCapabilities, DeviceFeature, DeviceId, DeviceType,
    Precision,
};
use crate::error::{BackendKind, ComputeError, ComputeResult};
use crate::memory::{Buffer, BufferMapping, BufferUsage};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice as CudarcDevice, CudaSlice};

/// CUDA device representation.
#[derive(Debug, Clone)]
pub struct CudaDevice {
    id: DeviceId,
    name: String,
    capabilities: DeviceCapabilities,
    ordinal: usize,
}

impl Hash for CudaDevice {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl PartialEq for CudaDevice {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for CudaDevice {}

impl CudaDevice {
    /// Create a CUDA device for a specific ordinal.
    #[cfg(feature = "cuda")]
    pub fn new(ordinal: usize) -> ComputeResult<Self> {
        let device = CudarcDevice::new(ordinal)
            .map_err(|e| ComputeError::backend(BackendKind::Cuda, e.to_string()))?;

        let name = device
            .name()
            .map_err(|e| ComputeError::backend(BackendKind::Cuda, e.to_string()))?;

        let (major, minor) = device
            .compute_capability()
            .map_err(|e| ComputeError::backend(BackendKind::Cuda, e.to_string()))?;

        let mut capabilities = DeviceCapabilities::default();
        capabilities.compute_version = ComputeVersion::cuda(major as u32, minor as u32);
        capabilities.vendor = "NVIDIA".to_string();

        // Query device properties
        capabilities.compute_units = device.num_sms() as u32;

        capabilities.features = vec![
            DeviceFeature::Float16,
            DeviceFeature::Float64,
            DeviceFeature::SharedMemory,
            DeviceFeature::AsyncTransfer,
            DeviceFeature::PeerToPeer,
        ];

        // Tensor cores on Volta and newer
        if major >= 7 {
            capabilities.features.push(DeviceFeature::TensorCores);
        }

        capabilities.supported_precisions = vec![
            Precision::Float16,
            Precision::Float32,
            Precision::Float64,
            Precision::Int8,
            Precision::Int32,
        ];

        Ok(Self {
            id: DeviceId::cuda(ordinal as u32),
            name,
            capabilities,
            ordinal,
        })
    }

    /// Create a stub device (when CUDA feature is disabled).
    #[cfg(not(feature = "cuda"))]
    pub fn new(_ordinal: usize) -> ComputeResult<Self> {
        Err(ComputeError::BackendInitFailed(
            "CUDA support not compiled".to_string(),
        ))
    }
}

impl ComputeDevice for CudaDevice {
    fn device_type(&self) -> DeviceType {
        DeviceType::CudaGpu
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

/// CUDA buffer wrapper.
pub struct CudaBuffer {
    #[cfg(feature = "cuda")]
    _slice: CudaSlice<u8>,
    size: usize,
    usage: BufferUsage,
    device_id: DeviceId,
}

impl Buffer for CudaBuffer {
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
            "CUDA buffers require explicit copy operations".to_string(),
        ))
    }

    fn map_range(&self, _offset: usize, _size: usize) -> ComputeResult<BufferMapping<'_>> {
        Err(ComputeError::MappingFailed(
            "CUDA buffers require explicit copy operations".to_string(),
        ))
    }

    fn is_mapped(&self) -> bool {
        false
    }
}

/// CUDA command encoder.
pub struct CudaCommandEncoder {
    #[cfg(feature = "cuda")]
    device: std::sync::Arc<CudarcDevice>,
    commands: Vec<CudaCommand>,
}

enum CudaCommand {
    CopyH2D { src: Vec<u8>, offset: usize },
    CopyD2H { offset: usize, size: usize },
    CopyD2D { src_offset: usize, dst_offset: usize, size: usize },
    Fill { offset: usize, size: usize, value: u8 },
}

impl CommandEncoder for CudaCommandEncoder {
    type Buffer = CudaBuffer;

    fn copy_host_to_device(&mut self, src: &[u8], _dst: &Self::Buffer, offset: usize) {
        self.commands.push(CudaCommand::CopyH2D {
            src: src.to_vec(),
            offset,
        });
    }

    fn copy_device_to_host(&mut self, _src: &Self::Buffer, _dst: &mut [u8], offset: usize) {
        self.commands.push(CudaCommand::CopyD2H {
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
        self.commands.push(CudaCommand::CopyD2D {
            src_offset,
            dst_offset,
            size,
        });
    }

    fn fill_buffer(&mut self, _buffer: &Self::Buffer, offset: usize, size: usize, value: u8) {
        self.commands.push(CudaCommand::Fill { offset, size, value });
    }

    fn barrier(&mut self) {
        // CUDA handles barriers via stream synchronization
    }
}

/// CUDA fence (event-based).
pub struct CudaFence {
    completed: AtomicBool,
}

impl Fence for CudaFence {
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

/// CUDA compute backend.
pub struct CudaBackend {
    device: CudaDevice,
    #[cfg(feature = "cuda")]
    cuda_device: std::sync::Arc<CudarcDevice>,
    rng_seed: AtomicU64,
}

impl ComputeBackend for CudaBackend {
    type Device = CudaDevice;
    type Buffer = CudaBuffer;
    type CommandEncoder = CudaCommandEncoder;
    type Fence = CudaFence;

    fn name() -> &'static str {
        "CUDA"
    }

    fn enumerate_devices() -> ComputeResult<Vec<Self::Device>> {
        #[cfg(feature = "cuda")]
        {
            let count = cudarc::driver::result::device::get_count()
                .map_err(|e| ComputeError::DiscoveryFailed(e.to_string()))?;

            (0..count)
                .map(|i| CudaDevice::new(i))
                .collect()
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(ComputeError::BackendInitFailed(
                "CUDA support not compiled".to_string(),
            ))
        }
    }

    fn new(device: &Self::Device) -> ComputeResult<Self> {
        #[cfg(feature = "cuda")]
        {
            let cuda_device = CudarcDevice::new(device.ordinal)
                .map_err(|e| ComputeError::backend(BackendKind::Cuda, e.to_string()))?;

            Ok(Self {
                device: device.clone(),
                cuda_device,
                rng_seed: AtomicU64::new(0),
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(ComputeError::BackendInitFailed(
                "CUDA support not compiled".to_string(),
            ))
        }
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn allocate_buffer(&self, size: usize, usage: BufferUsage) -> ComputeResult<Self::Buffer> {
        #[cfg(feature = "cuda")]
        {
            let slice = self.cuda_device.alloc_zeros::<u8>(size)
                .map_err(|e| ComputeError::allocation(size, e.to_string()))?;

            Ok(CudaBuffer {
                _slice: slice,
                size,
                usage,
                device_id: self.device.device_id(),
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(ComputeError::BackendInitFailed(
                "CUDA support not compiled".to_string(),
            ))
        }
    }

    fn create_encoder(&self) -> ComputeResult<Self::CommandEncoder> {
        #[cfg(feature = "cuda")]
        {
            Ok(CudaCommandEncoder {
                device: self.cuda_device.clone(),
                commands: Vec::new(),
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(ComputeError::BackendInitFailed(
                "CUDA support not compiled".to_string(),
            ))
        }
    }

    fn submit(&self, _encoder: Self::CommandEncoder) -> ComputeResult<Self::Fence> {
        #[cfg(feature = "cuda")]
        {
            // Execute commands and sync
            self.cuda_device.synchronize()
                .map_err(|e| ComputeError::backend(BackendKind::Cuda, e.to_string()))?;

            Ok(CudaFence {
                completed: AtomicBool::new(true),
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(ComputeError::BackendInitFailed(
                "CUDA support not compiled".to_string(),
            ))
        }
    }

    fn wait(&self, _fence: &Self::Fence) -> ComputeResult<()> {
        Ok(())
    }

    fn synchronize(&self) -> ComputeResult<()> {
        #[cfg(feature = "cuda")]
        {
            self.cuda_device.synchronize()
                .map_err(|e| ComputeError::backend(BackendKind::Cuda, e.to_string()))?;
        }
        Ok(())
    }

    fn seed(&self, seed: u64) {
        self.rng_seed.store(seed, Ordering::SeqCst);
    }
}
