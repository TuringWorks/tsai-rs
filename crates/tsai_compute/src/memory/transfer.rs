//! Cross-device memory transfer utilities.

use crate::backend::{CommandEncoder, ComputeBackend};
use crate::device::DeviceId;
use crate::error::{ComputeError, ComputeResult};

use super::{Buffer, MemoryPool};

/// Transfer manager for cross-device data movement.
///
/// Handles transfers between devices, including staging through
/// host memory when peer-to-peer is not available.
pub struct TransferManager<B: ComputeBackend> {
    staging_pool: MemoryPool<B>,
}

impl<B: ComputeBackend> TransferManager<B> {
    /// Create a new transfer manager.
    pub fn new() -> Self {
        Self {
            staging_pool: MemoryPool::new(),
        }
    }

    /// Transfer data from host to device.
    pub fn host_to_device(
        &self,
        backend: &B,
        src: &[u8],
        dst: &B::Buffer,
        dst_offset: usize,
    ) -> ComputeResult<()> {
        if src.len() + dst_offset > dst.size() {
            return Err(ComputeError::BufferError(format!(
                "Transfer size {} exceeds buffer size {} at offset {}",
                src.len(),
                dst.size(),
                dst_offset
            )));
        }

        let mut encoder = backend.create_encoder()?;
        encoder.copy_host_to_device(src, dst, dst_offset);
        let fence = backend.submit(encoder)?;
        backend.wait(&fence)?;

        Ok(())
    }

    /// Transfer data from device to host.
    pub fn device_to_host(
        &self,
        backend: &B,
        src: &B::Buffer,
        src_offset: usize,
        dst: &mut [u8],
    ) -> ComputeResult<()> {
        if dst.len() + src_offset > src.size() {
            return Err(ComputeError::BufferError(format!(
                "Transfer size {} exceeds buffer size {} at offset {}",
                dst.len(),
                src.size(),
                src_offset
            )));
        }

        let mut encoder = backend.create_encoder()?;
        encoder.copy_device_to_host(src, dst, src_offset);
        let fence = backend.submit(encoder)?;
        backend.wait(&fence)?;

        Ok(())
    }

    /// Transfer data between buffers on the same device.
    pub fn device_to_device_same(
        &self,
        backend: &B,
        src: &B::Buffer,
        src_offset: usize,
        dst: &B::Buffer,
        dst_offset: usize,
        size: usize,
    ) -> ComputeResult<()> {
        if src_offset + size > src.size() {
            return Err(ComputeError::BufferError("Source overflow".to_string()));
        }
        if dst_offset + size > dst.size() {
            return Err(ComputeError::BufferError("Destination overflow".to_string()));
        }

        let mut encoder = backend.create_encoder()?;
        encoder.copy_buffer_to_buffer(src, src_offset, dst, dst_offset, size);
        let fence = backend.submit(encoder)?;
        backend.wait(&fence)?;

        Ok(())
    }

    /// Clear the staging pool.
    pub fn clear_staging(&self) {
        self.staging_pool.clear();
    }
}

impl<B: ComputeBackend> Default for TransferManager<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Async transfer handle.
pub struct TransferHandle<F> {
    fence: F,
    _src_device: DeviceId,
    _dst_device: DeviceId,
}

impl<F: crate::backend::Fence> TransferHandle<F> {
    /// Create a new transfer handle.
    pub fn new(fence: F, src_device: DeviceId, dst_device: DeviceId) -> Self {
        Self {
            fence,
            _src_device: src_device,
            _dst_device: dst_device,
        }
    }

    /// Check if the transfer is complete.
    pub fn is_complete(&self) -> bool {
        self.fence.is_signaled()
    }

    /// Wait for the transfer to complete.
    pub fn wait(&self) {
        self.fence.wait();
    }

    /// Wait with timeout. Returns true if completed.
    pub fn wait_timeout(&self, timeout_ms: u64) -> bool {
        self.fence.wait_timeout(timeout_ms)
    }
}

/// Transfer direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    /// Host to device.
    HostToDevice,
    /// Device to host.
    DeviceToHost,
    /// Device to device (same device).
    DeviceToDeviceSame,
    /// Device to device (different devices).
    DeviceToDevicePeer,
    /// Device to device via host staging.
    DeviceToDeviceStaged,
}

/// Transfer statistics.
#[derive(Debug, Clone, Default)]
pub struct TransferStats {
    /// Total bytes transferred.
    pub total_bytes: u64,
    /// Number of transfers.
    pub transfer_count: u64,
    /// Bytes transferred host to device.
    pub h2d_bytes: u64,
    /// Bytes transferred device to host.
    pub d2h_bytes: u64,
    /// Bytes transferred device to device.
    pub d2d_bytes: u64,
}
