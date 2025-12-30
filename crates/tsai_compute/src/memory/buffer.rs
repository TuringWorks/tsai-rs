//! Buffer abstractions for device memory.

use crate::device::DeviceId;
use crate::error::ComputeResult;
use std::ops::{Deref, DerefMut};

/// Buffer usage flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferUsage {
    /// Host can read from this buffer.
    pub host_readable: bool,
    /// Host can write to this buffer.
    pub host_writable: bool,
    /// Device can read from this buffer.
    pub device_readable: bool,
    /// Device can write to this buffer.
    pub device_writable: bool,
    /// Buffer can be used as a transfer source.
    pub transfer_src: bool,
    /// Buffer can be used as a transfer destination.
    pub transfer_dst: bool,
}

impl BufferUsage {
    /// Device-only buffer (not accessible from host).
    pub const DEVICE_ONLY: Self = Self {
        host_readable: false,
        host_writable: false,
        device_readable: true,
        device_writable: true,
        transfer_src: false,
        transfer_dst: true,
    };

    /// Host-visible buffer (accessible from both host and device).
    pub const HOST_VISIBLE: Self = Self {
        host_readable: true,
        host_writable: true,
        device_readable: true,
        device_writable: true,
        transfer_src: true,
        transfer_dst: true,
    };

    /// Staging buffer for transfers.
    pub const STAGING: Self = Self {
        host_readable: true,
        host_writable: true,
        device_readable: false,
        device_writable: false,
        transfer_src: true,
        transfer_dst: true,
    };

    /// Read-only device buffer.
    pub const DEVICE_READ_ONLY: Self = Self {
        host_readable: false,
        host_writable: false,
        device_readable: true,
        device_writable: false,
        transfer_src: false,
        transfer_dst: true,
    };

    /// Uniform/constant buffer.
    pub const UNIFORM: Self = Self {
        host_readable: false,
        host_writable: true,
        device_readable: true,
        device_writable: false,
        transfer_src: false,
        transfer_dst: true,
    };

    /// Check if the buffer is host-accessible.
    pub fn is_host_accessible(&self) -> bool {
        self.host_readable || self.host_writable
    }

    /// Check if the buffer is device-accessible.
    pub fn is_device_accessible(&self) -> bool {
        self.device_readable || self.device_writable
    }
}

impl Default for BufferUsage {
    fn default() -> Self {
        Self::HOST_VISIBLE
    }
}

/// Abstract buffer trait for device memory.
pub trait Buffer: Send + Sync + 'static {
    /// Get the buffer size in bytes.
    fn size(&self) -> usize;

    /// Get the buffer usage flags.
    fn usage(&self) -> BufferUsage;

    /// Get the device this buffer is allocated on.
    fn device_id(&self) -> DeviceId;

    /// Map the buffer for host access.
    ///
    /// Returns a mapping that provides access to the buffer's memory.
    /// The buffer is automatically unmapped when the mapping is dropped.
    fn map(&self) -> ComputeResult<BufferMapping<'_>>;

    /// Map a portion of the buffer.
    fn map_range(&self, offset: usize, size: usize) -> ComputeResult<BufferMapping<'_>>;

    /// Check if the buffer is currently mapped.
    fn is_mapped(&self) -> bool;

    /// Get the raw pointer if the buffer is persistently mapped.
    /// Returns None if not mapped or not persistently mapped.
    fn raw_ptr(&self) -> Option<*mut u8> {
        None
    }
}

/// Buffer mapping for host access.
///
/// Provides a safe interface for reading and writing buffer memory
/// from the host. The buffer is automatically unmapped when dropped.
pub struct BufferMapping<'a> {
    ptr: *mut u8,
    size: usize,
    offset: usize,
    unmap_fn: Option<Box<dyn FnOnce() + 'a>>,
}

impl<'a> BufferMapping<'a> {
    /// Create a new buffer mapping.
    ///
    /// # Safety
    /// The pointer must be valid for the lifetime 'a and the specified size.
    pub unsafe fn new(
        ptr: *mut u8,
        size: usize,
        offset: usize,
        unmap_fn: Option<Box<dyn FnOnce() + 'a>>,
    ) -> Self {
        Self {
            ptr,
            size,
            offset,
            unmap_fn,
        }
    }

    /// Get the size of the mapped region.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the offset of the mapped region.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get the mapped memory as a byte slice.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    /// Get the mapped memory as a mutable byte slice.
    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    /// Read typed data from the mapping.
    pub fn read<T: Copy>(&self, offset: usize) -> Option<T> {
        if offset + std::mem::size_of::<T>() > self.size {
            return None;
        }
        unsafe {
            let ptr = self.ptr.add(offset) as *const T;
            Some(ptr.read_unaligned())
        }
    }

    /// Write typed data to the mapping.
    pub fn write<T: Copy>(&mut self, offset: usize, value: T) -> bool {
        if offset + std::mem::size_of::<T>() > self.size {
            return false;
        }
        unsafe {
            let ptr = self.ptr.add(offset) as *mut T;
            ptr.write_unaligned(value);
        }
        true
    }

    /// Copy data from a slice into the mapping.
    pub fn copy_from_slice(&mut self, offset: usize, data: &[u8]) -> bool {
        if offset + data.len() > self.size {
            return false;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.add(offset), data.len());
        }
        true
    }

    /// Copy data from the mapping into a slice.
    pub fn copy_to_slice(&self, offset: usize, dst: &mut [u8]) -> bool {
        if offset + dst.len() > self.size {
            return false;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr.add(offset), dst.as_mut_ptr(), dst.len());
        }
        true
    }
}

impl<'a> Drop for BufferMapping<'a> {
    fn drop(&mut self) {
        if let Some(unmap) = self.unmap_fn.take() {
            unmap();
        }
    }
}

impl<'a> Deref for BufferMapping<'a> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<'a> DerefMut for BufferMapping<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_slice_mut()
    }
}

// Safety: BufferMapping is Send if the underlying buffer is Send
unsafe impl<'a> Send for BufferMapping<'a> {}

/// A typed buffer view.
pub struct TypedBuffer<'a, T: Copy> {
    mapping: BufferMapping<'a>,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T: Copy> TypedBuffer<'a, T> {
    /// Create a typed view of a buffer mapping.
    pub fn new(mapping: BufferMapping<'a>) -> Self {
        Self {
            mapping,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the number of elements.
    pub fn len(&self) -> usize {
        self.mapping.size() / std::mem::size_of::<T>()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get an element by index.
    pub fn get(&self, index: usize) -> Option<T> {
        self.mapping.read(index * std::mem::size_of::<T>())
    }

    /// Set an element by index.
    pub fn set(&mut self, index: usize, value: T) -> bool {
        self.mapping.write(index * std::mem::size_of::<T>(), value)
    }

    /// Get as a slice of T.
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(
                self.mapping.as_slice().as_ptr() as *const T,
                self.len(),
            )
        }
    }

    /// Get as a mutable slice of T.
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        let len = self.len();
        unsafe {
            std::slice::from_raw_parts_mut(
                self.mapping.as_slice_mut().as_mut_ptr() as *mut T,
                len,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_usage() {
        let usage = BufferUsage::HOST_VISIBLE;
        assert!(usage.is_host_accessible());
        assert!(usage.is_device_accessible());

        let usage = BufferUsage::DEVICE_ONLY;
        assert!(!usage.is_host_accessible());
        assert!(usage.is_device_accessible());

        let usage = BufferUsage::STAGING;
        assert!(usage.is_host_accessible());
        assert!(!usage.is_device_accessible());
    }

    #[test]
    fn test_buffer_mapping() {
        let mut data = vec![0u8; 16];
        let mapping = unsafe {
            BufferMapping::new(data.as_mut_ptr(), data.len(), 0, None)
        };

        assert_eq!(mapping.size(), 16);
        assert_eq!(mapping.as_slice().len(), 16);
    }

    #[test]
    fn test_buffer_mapping_read_write() {
        let mut data = vec![0u8; 16];
        let mut mapping = unsafe {
            BufferMapping::new(data.as_mut_ptr(), data.len(), 0, None)
        };

        // Write some data
        mapping.write::<u32>(0, 42);
        mapping.write::<u32>(4, 123);

        // Read it back
        assert_eq!(mapping.read::<u32>(0), Some(42));
        assert_eq!(mapping.read::<u32>(4), Some(123));
    }
}
