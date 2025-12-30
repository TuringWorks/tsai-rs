//! ROCm compute backend for AMD GPUs.
//!
//! This backend provides GPU computation via AMD ROCm/HIP,
//! supporting AMD Radeon and Instinct GPUs.
//!
//! ## Detection Method
//!
//! On Linux, AMD GPUs are detected via sysfs at `/sys/class/drm/card*/device/`.
//! The backend checks for AMD vendor ID (0x1002) and extracts device information.
//!
//! ## Feature Flag
//!
//! Enable the `rocm` feature to use this backend:
//! ```toml
//! [dependencies]
//! tsai_compute = { version = "0.1", features = ["rocm"] }
//! ```

use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::backend::{CommandEncoder, ComputeBackend, Fence};
use crate::device::{ComputeDevice, ComputeVersion, DeviceCapabilities, DeviceFeature, DeviceId, DeviceType};
use crate::error::{ComputeError, ComputeResult};
use crate::memory::{Buffer, BufferMapping, BufferUsage};

/// ROCm device representation.
#[derive(Debug, Clone)]
pub struct RocmDevice {
    id: DeviceId,
    name: String,
    capabilities: DeviceCapabilities,
}

impl Hash for RocmDevice {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl PartialEq for RocmDevice {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for RocmDevice {}

/// Detect AMD GPUs by scanning sysfs on Linux.
#[cfg(all(feature = "rocm", target_os = "linux"))]
fn detect_amd_gpus_sysfs() -> ComputeResult<Vec<RocmDevice>> {
    use std::fs;
    use std::path::Path;

    const AMD_VENDOR_ID: &str = "0x1002";
    let drm_path = Path::new("/sys/class/drm");

    if !drm_path.exists() {
        return Err(ComputeError::DiscoveryFailed(
            "sysfs DRM path not found".to_string(),
        ));
    }

    let mut devices = Vec::new();
    let mut device_index = 0u32;

    // Iterate over card* directories
    let entries = fs::read_dir(drm_path).map_err(|e| {
        ComputeError::DiscoveryFailed(format!("Failed to read DRM directory: {}", e))
    })?;

    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        // Only process card* entries (not renderD*)
        if !name_str.starts_with("card") || name_str.contains('-') {
            continue;
        }

        let device_path = entry.path().join("device");

        // Check vendor ID
        let vendor_path = device_path.join("vendor");
        if let Ok(vendor) = fs::read_to_string(&vendor_path) {
            let vendor = vendor.trim();
            if vendor != AMD_VENDOR_ID {
                continue; // Not an AMD GPU
            }
        } else {
            continue;
        }

        // Get device ID
        let device_id = fs::read_to_string(device_path.join("device"))
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        // Try to get device name from uevent
        let device_name = fs::read_to_string(device_path.join("uevent"))
            .ok()
            .and_then(|content| {
                for line in content.lines() {
                    if line.starts_with("PCI_SLOT_NAME=") {
                        return Some(format!("AMD GPU [{}]", line.split('=').nth(1)?));
                    }
                }
                None
            })
            .unwrap_or_else(|| format!("AMD GPU {}", device_id));

        // Try to get GFX architecture from various sources
        let gfx_arch = detect_gfx_arch(&device_path, &device_id);

        // Try to get memory size from hwmon or mem_info
        let memory_bytes = detect_gpu_memory(&device_path);

        devices.push(RocmDevice::new(device_index, device_name, gfx_arch, memory_bytes));
        device_index += 1;
    }

    Ok(devices)
}

/// Detect GFX architecture from sysfs or device ID mapping.
#[cfg(all(feature = "rocm", target_os = "linux"))]
fn detect_gfx_arch(device_path: &std::path::Path, device_id: &str) -> String {
    use std::fs;

    // Try to read from amdgpu specific files
    if let Ok(arch) = fs::read_to_string(device_path.join("gpu_id")) {
        return arch.trim().to_string();
    }

    // Map known device IDs to GFX architectures (common AMD GPUs)
    // Format: 0xXXXX
    let arch = match device_id.to_lowercase().as_str() {
        // RDNA 3 (gfx1100 series)
        "0x744c" | "0x7480" => "gfx1100",  // RX 7900 series
        "0x7470" => "gfx1101",              // RX 7700/7800
        "0x7460" => "gfx1102",              // RX 7600

        // RDNA 2 (gfx1030 series)
        "0x73bf" | "0x73a5" => "gfx1030",  // RX 6900/6800
        "0x73df" | "0x73af" => "gfx1031",  // RX 6700
        "0x73ff" | "0x73ef" => "gfx1032",  // RX 6600

        // RDNA 1 (gfx1010 series)
        "0x731f" | "0x7340" => "gfx1010",  // RX 5700
        "0x7360" => "gfx1011",              // RX 5600
        "0x7310" => "gfx1012",              // RX 5500

        // Vega (gfx900 series)
        "0x687f" | "0x6867" => "gfx900",   // Vega 64/56
        "0x66af" => "gfx906",               // Radeon VII

        // MI series (data center)
        "0x7408" => "gfx90a",               // MI200 series
        "0x740f" => "gfx940",               // MI300 series

        _ => "unknown",
    };

    arch.to_string()
}

/// Detect GPU memory size from sysfs.
#[cfg(all(feature = "rocm", target_os = "linux"))]
fn detect_gpu_memory(device_path: &std::path::Path) -> u64 {
    use std::fs;

    // Try amdgpu specific memory info
    let mem_info_path = device_path.join("mem_info_vram_total");
    if let Ok(mem_str) = fs::read_to_string(&mem_info_path) {
        if let Ok(bytes) = mem_str.trim().parse::<u64>() {
            return bytes;
        }
    }

    // Try hwmon temperature sensor directory (often contains memory info)
    if let Ok(entries) = fs::read_dir(device_path.join("hwmon")) {
        for entry in entries.flatten() {
            let mem_path = entry.path().join("mem1_input");
            if let Ok(mem_str) = fs::read_to_string(&mem_path) {
                if let Ok(mb) = mem_str.trim().parse::<u64>() {
                    return mb * 1024 * 1024; // Convert MB to bytes
                }
            }
        }
    }

    // Default: unknown memory (return 0)
    0
}

impl RocmDevice {
    /// Create a ROCm device with detected information.
    fn new(index: u32, name: String, gfx_arch: String, memory_bytes: u64) -> Self {
        let mut capabilities = DeviceCapabilities::default();
        capabilities.compute_version = ComputeVersion::Rocm { gfx_arch: gfx_arch.clone() };
        capabilities.vendor = "AMD".to_string();
        capabilities.total_memory = memory_bytes;
        capabilities.available_memory = memory_bytes;

        // Determine compute capability based on GFX architecture
        if gfx_arch.starts_with("gfx9") || gfx_arch.starts_with("gfx10") || gfx_arch.starts_with("gfx11") {
            capabilities.features.insert(DeviceFeature::Compute);
            capabilities.features.insert(DeviceFeature::Float16);
            capabilities.features.insert(DeviceFeature::Float32);
            capabilities.features.insert(DeviceFeature::Float64);
        }

        // Mark as discrete GPU
        capabilities.features.insert(DeviceFeature::DiscreteGpu);

        Self {
            id: DeviceId::rocm(index),
            name,
            capabilities,
        }
    }
}

impl ComputeDevice for RocmDevice {
    fn device_type(&self) -> DeviceType {
        DeviceType::RocmGpu
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

/// ROCm buffer wrapper.
pub struct RocmBuffer {
    size: usize,
    usage: BufferUsage,
    device_id: DeviceId,
}

impl Buffer for RocmBuffer {
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
            "ROCm buffer mapping not yet implemented".to_string(),
        ))
    }

    fn map_range(&self, _offset: usize, _size: usize) -> ComputeResult<BufferMapping<'_>> {
        Err(ComputeError::MappingFailed(
            "ROCm buffer mapping not yet implemented".to_string(),
        ))
    }

    fn is_mapped(&self) -> bool {
        false
    }

    fn raw_ptr(&self) -> Option<*mut u8> {
        None
    }
}

/// ROCm command encoder.
#[allow(dead_code)]
pub struct RocmCommandEncoder {
    commands: Vec<RocmCommand>,
}

#[allow(dead_code)]
enum RocmCommand {
    CopyHostToDevice { data: Vec<u8>, offset: usize },
    CopyDeviceToHost { offset: usize, size: usize },
    CopyBufferToBuffer { src_offset: usize, dst_offset: usize, size: usize },
    FillBuffer { offset: usize, size: usize, value: u8 },
    Barrier,
}

impl CommandEncoder for RocmCommandEncoder {
    type Buffer = RocmBuffer;

    fn copy_host_to_device(&mut self, src: &[u8], _dst: &Self::Buffer, offset: usize) {
        self.commands.push(RocmCommand::CopyHostToDevice {
            data: src.to_vec(),
            offset,
        });
    }

    fn copy_device_to_host(&mut self, _src: &Self::Buffer, _dst: &mut [u8], offset: usize) {
        self.commands.push(RocmCommand::CopyDeviceToHost {
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
        self.commands.push(RocmCommand::CopyBufferToBuffer {
            src_offset,
            dst_offset,
            size,
        });
    }

    fn fill_buffer(&mut self, _buffer: &Self::Buffer, offset: usize, size: usize, value: u8) {
        self.commands.push(RocmCommand::FillBuffer { offset, size, value });
    }

    fn barrier(&mut self) {
        self.commands.push(RocmCommand::Barrier);
    }
}

/// ROCm fence.
pub struct RocmFence {
    completed: AtomicBool,
}

impl RocmFence {
    fn new_signaled() -> Self {
        Self {
            completed: AtomicBool::new(true),
        }
    }
}

impl Fence for RocmFence {
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

/// ROCm compute backend.
pub struct RocmBackend {
    device: RocmDevice,
    rng_seed: AtomicU64,
}

impl ComputeBackend for RocmBackend {
    type Device = RocmDevice;
    type Buffer = RocmBuffer;
    type CommandEncoder = RocmCommandEncoder;
    type Fence = RocmFence;

    fn name() -> &'static str {
        "ROCm"
    }

    fn enumerate_devices() -> ComputeResult<Vec<Self::Device>> {
        #[cfg(feature = "rocm")]
        {
            // Detect AMD GPUs via sysfs on Linux
            #[cfg(target_os = "linux")]
            {
                detect_amd_gpus_sysfs()
            }

            #[cfg(not(target_os = "linux"))]
            {
                // ROCm is primarily Linux-only
                Err(ComputeError::BackendInitFailed(
                    "ROCm is only supported on Linux".to_string(),
                ))
            }
        }

        #[cfg(not(feature = "rocm"))]
        {
            Err(ComputeError::BackendInitFailed(
                "ROCm support not compiled (enable 'rocm' feature)".to_string(),
            ))
        }
    }

    fn new(device: &Self::Device) -> ComputeResult<Self> {
        #[cfg(feature = "rocm")]
        {
            // Placeholder - would need to create HIP context
            Ok(Self {
                device: device.clone(),
                rng_seed: AtomicU64::new(0),
            })
        }

        #[cfg(not(feature = "rocm"))]
        {
            let _ = device;
            Err(ComputeError::BackendInitFailed(
                "ROCm support not compiled".to_string(),
            ))
        }
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn allocate_buffer(&self, size: usize, usage: BufferUsage) -> ComputeResult<Self::Buffer> {
        #[cfg(feature = "rocm")]
        {
            // Placeholder - would use hipMalloc
            Ok(RocmBuffer {
                size,
                usage,
                device_id: self.device.device_id(),
            })
        }

        #[cfg(not(feature = "rocm"))]
        {
            let _ = (size, usage);
            Err(ComputeError::BackendInitFailed(
                "ROCm support not compiled".to_string(),
            ))
        }
    }

    fn create_encoder(&self) -> ComputeResult<Self::CommandEncoder> {
        Ok(RocmCommandEncoder {
            commands: Vec::new(),
        })
    }

    fn submit(&self, _encoder: Self::CommandEncoder) -> ComputeResult<Self::Fence> {
        // Placeholder - would submit to HIP stream
        Ok(RocmFence::new_signaled())
    }

    fn wait(&self, fence: &Self::Fence) -> ComputeResult<()> {
        fence.wait();
        Ok(())
    }

    fn synchronize(&self) -> ComputeResult<()> {
        // Placeholder - would call hipDeviceSynchronize
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
    fn test_rocm_not_available() {
        // Without rocm feature, enumeration should fail gracefully
        let result = RocmBackend::enumerate_devices();
        assert!(result.is_err());
    }
}
