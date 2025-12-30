//! Compute backend abstractions.
//!
//! This module provides the core traits for implementing compute backends
//! and dispatching operations across heterogeneous hardware.

pub mod cpu;

#[cfg(target_os = "macos")]
pub mod metal;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "vulkan")]
pub mod vulkan;

#[cfg(feature = "opencl")]
pub mod opencl;

#[cfg(feature = "rocm")]
pub mod rocm;

#[cfg(all(feature = "mlx", target_os = "macos"))]
pub mod mlx;

use crate::device::ComputeDevice;
use crate::error::ComputeResult;
use crate::memory::{Buffer, BufferUsage};

/// Core backend trait for compute operations.
///
/// Each backend implementation provides device enumeration, buffer management,
/// command encoding, and synchronization primitives.
pub trait ComputeBackend: Send + Sync + 'static {
    /// Associated device type for this backend.
    type Device: ComputeDevice;

    /// Buffer handle type.
    type Buffer: Buffer;

    /// Command encoder / stream type.
    type CommandEncoder: CommandEncoder<Buffer = Self::Buffer>;

    /// Synchronization primitive.
    type Fence: Fence;

    /// Get the backend name identifier.
    fn name() -> &'static str;

    /// Enumerate all available devices for this backend.
    fn enumerate_devices() -> ComputeResult<Vec<Self::Device>>;

    /// Create a backend instance bound to a specific device.
    fn new(device: &Self::Device) -> ComputeResult<Self>
    where
        Self: Sized;

    /// Get the device this backend is bound to.
    fn device(&self) -> &Self::Device;

    /// Allocate a buffer on the device.
    fn allocate_buffer(&self, size: usize, usage: BufferUsage) -> ComputeResult<Self::Buffer>;

    /// Create a command encoder for batching operations.
    fn create_encoder(&self) -> ComputeResult<Self::CommandEncoder>;

    /// Submit encoded commands and return a fence for synchronization.
    fn submit(&self, encoder: Self::CommandEncoder) -> ComputeResult<Self::Fence>;

    /// Wait for a specific fence to complete.
    fn wait(&self, fence: &Self::Fence) -> ComputeResult<()>;

    /// Synchronize all pending operations on this backend.
    fn synchronize(&self) -> ComputeResult<()>;

    /// Seed the random number generator for this backend.
    fn seed(&self, seed: u64);

    /// Check if this backend supports a specific operation.
    fn supports_operation(&self, op: &str) -> bool {
        // Default implementation - backends can override
        let _ = op;
        true
    }
}

/// Command encoder for batching operations.
///
/// Command encoders allow batching multiple operations together
/// for efficient submission to the device.
pub trait CommandEncoder: Send {
    /// Associated buffer type.
    type Buffer: Buffer;

    /// Copy data from host memory to a device buffer.
    fn copy_host_to_device(&mut self, src: &[u8], dst: &Self::Buffer, offset: usize);

    /// Copy data from a device buffer to host memory.
    fn copy_device_to_host(&mut self, src: &Self::Buffer, dst: &mut [u8], offset: usize);

    /// Copy data between device buffers.
    fn copy_buffer_to_buffer(
        &mut self,
        src: &Self::Buffer,
        src_offset: usize,
        dst: &Self::Buffer,
        dst_offset: usize,
        size: usize,
    );

    /// Fill a buffer with a byte pattern.
    fn fill_buffer(&mut self, buffer: &Self::Buffer, offset: usize, size: usize, value: u8);

    /// Insert a memory barrier.
    fn barrier(&mut self);
}

/// Synchronization fence.
///
/// Fences are used to synchronize between the host and device,
/// allowing the host to wait for device operations to complete.
pub trait Fence: Send + Sync {
    /// Check if the fence has been signaled (operation completed).
    fn is_signaled(&self) -> bool;

    /// Block until the fence is signaled.
    fn wait(&self);

    /// Wait with a timeout. Returns true if signaled, false if timeout.
    fn wait_timeout(&self, timeout_ms: u64) -> bool;
}

/// Kernel binding for compute dispatch.
#[derive(Debug, Clone)]
pub enum KernelBinding<'a, B: Buffer> {
    /// Buffer binding.
    Buffer(&'a B),
    /// Uniform/constant data.
    Uniform(&'a [u8]),
    /// Texture/sampler (placeholder).
    Texture,
}

/// Compute kernel interface.
pub trait ComputeKernel: Send + Sync {
    /// Get the kernel name.
    fn name(&self) -> &str;

    /// Get the recommended work group size.
    fn work_group_size(&self) -> [u32; 3];
}

/// Extended command encoder for compute kernels.
pub trait ComputeEncoder: CommandEncoder {
    /// Dispatch a compute kernel.
    fn dispatch<K: ComputeKernel>(
        &mut self,
        kernel: &K,
        grid: [u32; 3],
        block: [u32; 3],
        bindings: &[KernelBinding<'_, Self::Buffer>],
    );
}

/// Backend capabilities query.
pub trait BackendCapabilities {
    /// Check if async compute is supported.
    fn supports_async_compute(&self) -> bool {
        false
    }

    /// Check if multiple queues are supported.
    fn supports_multi_queue(&self) -> bool {
        false
    }

    /// Get the maximum number of concurrent operations.
    fn max_concurrent_ops(&self) -> usize {
        1
    }

    /// Check if peer-to-peer transfer is supported with another device.
    fn supports_peer_transfer(&self, _other: &dyn std::any::Any) -> bool {
        false
    }
}
