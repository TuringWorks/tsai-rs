//! Memory pool for buffer reuse.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use parking_lot::Mutex;

use super::{Buffer, BufferUsage};
use crate::backend::ComputeBackend;
use crate::device::{ComputeDevice, DeviceId};
use crate::error::ComputeResult;

/// Cached buffer entry.
struct CachedBuffer<B: Buffer> {
    buffer: B,
    size: usize,
    usage: BufferUsage,
    last_used: Instant,
}

/// Per-device memory pool.
struct DeviceMemoryPool<B: Buffer> {
    free_buffers: Mutex<Vec<CachedBuffer<B>>>,
    allocated_bytes: AtomicU64,
    cached_bytes: AtomicU64,
}

impl<B: Buffer> DeviceMemoryPool<B> {
    fn new() -> Self {
        Self {
            free_buffers: Mutex::new(Vec::new()),
            allocated_bytes: AtomicU64::new(0),
            cached_bytes: AtomicU64::new(0),
        }
    }

    /// Try to reuse a buffer from the cache.
    fn try_reuse(&self, size: usize, usage: BufferUsage) -> Option<B> {
        let mut buffers = self.free_buffers.lock();

        // Find a buffer that matches size and usage
        // Allow slightly larger buffers (up to 50% overhead)
        let max_size = size + size / 2;

        let idx = buffers.iter().position(|b| {
            b.size >= size && b.size <= max_size && b.usage == usage
        })?;

        let cached = buffers.swap_remove(idx);
        self.cached_bytes.fetch_sub(cached.size as u64, Ordering::Relaxed);
        Some(cached.buffer)
    }

    /// Return a buffer to the cache.
    fn release(&self, buffer: B) {
        let size = buffer.size();
        let usage = buffer.usage();

        let cached = CachedBuffer {
            buffer,
            size,
            usage,
            last_used: Instant::now(),
        };

        self.cached_bytes.fetch_add(size as u64, Ordering::Relaxed);
        self.free_buffers.lock().push(cached);
    }

    /// Clear all cached buffers.
    fn clear(&self) {
        let mut buffers = self.free_buffers.lock();
        self.cached_bytes.store(0, Ordering::Relaxed);
        buffers.clear();
    }

    /// Trim old buffers.
    fn trim(&self, max_age: std::time::Duration) {
        let now = Instant::now();
        let mut buffers = self.free_buffers.lock();

        let mut trimmed_bytes: u64 = 0;
        buffers.retain(|b| {
            let keep = now.duration_since(b.last_used) < max_age;
            if !keep {
                trimmed_bytes += b.size as u64;
            }
            keep
        });

        self.cached_bytes.fetch_sub(trimmed_bytes, Ordering::Relaxed);
    }

    fn cached_bytes(&self) -> u64 {
        self.cached_bytes.load(Ordering::Relaxed)
    }

    fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes.load(Ordering::Relaxed)
    }
}

/// Memory pool for buffer reuse across devices.
///
/// The pool maintains a cache of freed buffers that can be reused
/// for new allocations, reducing allocation overhead.
pub struct MemoryPool<B: ComputeBackend> {
    pools: Mutex<HashMap<DeviceId, DeviceMemoryPool<B::Buffer>>>,
    max_cached_bytes: u64,
}

impl<B: ComputeBackend> MemoryPool<B> {
    /// Create a new memory pool.
    pub fn new() -> Self {
        Self {
            pools: Mutex::new(HashMap::new()),
            max_cached_bytes: 1024 * 1024 * 1024, // 1 GB default
        }
    }

    /// Create a memory pool with a maximum cache size.
    pub fn with_max_cache(max_bytes: u64) -> Self {
        Self {
            pools: Mutex::new(HashMap::new()),
            max_cached_bytes: max_bytes,
        }
    }

    /// Allocate or reuse a buffer.
    pub fn allocate(
        &self,
        backend: &B,
        size: usize,
        usage: BufferUsage,
    ) -> ComputeResult<B::Buffer> {
        let device_id = backend.device().device_id();

        // Try to reuse from cache
        {
            let pools = self.pools.lock();
            if let Some(pool) = pools.get(&device_id) {
                if let Some(buffer) = pool.try_reuse(size, usage) {
                    return Ok(buffer);
                }
            }
        }

        // Allocate new buffer
        backend.allocate_buffer(size, usage)
    }

    /// Release a buffer back to the pool for reuse.
    pub fn release(&self, device_id: DeviceId, buffer: B::Buffer) {
        let mut pools = self.pools.lock();

        // Check if we're over the cache limit
        let total_cached: u64 = pools.values().map(|p| p.cached_bytes()).sum();
        if total_cached >= self.max_cached_bytes {
            // Don't cache, just drop
            return;
        }

        let pool = pools
            .entry(device_id)
            .or_insert_with(DeviceMemoryPool::new);

        pool.release(buffer);
    }

    /// Clear all cached buffers.
    pub fn clear(&self) {
        let pools = self.pools.lock();
        for pool in pools.values() {
            pool.clear();
        }
    }

    /// Clear cached buffers for a specific device.
    pub fn clear_device(&self, device_id: &DeviceId) {
        let pools = self.pools.lock();
        if let Some(pool) = pools.get(device_id) {
            pool.clear();
        }
    }

    /// Trim old buffers from all devices.
    pub fn trim(&self, max_age: std::time::Duration) {
        let pools = self.pools.lock();
        for pool in pools.values() {
            pool.trim(max_age);
        }
    }

    /// Get total cached bytes across all devices.
    pub fn total_cached_bytes(&self) -> u64 {
        self.pools
            .lock()
            .values()
            .map(|p| p.cached_bytes())
            .sum()
    }

    /// Get cached bytes for a specific device.
    pub fn device_cached_bytes(&self, device_id: &DeviceId) -> u64 {
        self.pools
            .lock()
            .get(device_id)
            .map(|p| p.cached_bytes())
            .unwrap_or(0)
    }

    /// Get statistics.
    pub fn stats(&self) -> PoolStats {
        let pools = self.pools.lock();
        let mut stats = PoolStats::default();

        for (device_id, pool) in pools.iter() {
            stats.devices.push(DevicePoolStats {
                device_id: *device_id,
                cached_bytes: pool.cached_bytes(),
                allocated_bytes: pool.allocated_bytes(),
                buffer_count: pool.free_buffers.lock().len(),
            });
        }

        stats.total_cached_bytes = stats.devices.iter().map(|d| d.cached_bytes).sum();
        stats
    }
}

impl<B: ComputeBackend> Default for MemoryPool<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Pool statistics.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total cached bytes.
    pub total_cached_bytes: u64,
    /// Per-device statistics.
    pub devices: Vec<DevicePoolStats>,
}

/// Per-device pool statistics.
#[derive(Debug, Clone)]
pub struct DevicePoolStats {
    /// Device ID.
    pub device_id: DeviceId,
    /// Cached bytes.
    pub cached_bytes: u64,
    /// Total allocated bytes (including non-cached).
    pub allocated_bytes: u64,
    /// Number of cached buffers.
    pub buffer_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full tests require a concrete backend implementation
    #[test]
    fn test_buffer_usage_equality() {
        let u1 = BufferUsage::HOST_VISIBLE;
        let u2 = BufferUsage::HOST_VISIBLE;
        assert_eq!(u1, u2);

        let u3 = BufferUsage::DEVICE_ONLY;
        assert_ne!(u1, u3);
    }
}
