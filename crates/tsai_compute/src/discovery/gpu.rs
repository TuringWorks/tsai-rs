//! GPU device enumeration.

use crate::device::DeviceType;

/// Check if any GPU backend is available.
pub fn is_gpu_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        return true; // Metal always available on macOS
    }

    #[cfg(feature = "cuda")]
    {
        if cudarc::driver::result::device::get_count().map(|c| c > 0).unwrap_or(false) {
            return true;
        }
    }

    #[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
    {
        false
    }
}

/// Get the list of available GPU backend types.
pub fn available_gpu_backends() -> Vec<DeviceType> {
    let mut backends = Vec::new();

    #[cfg(target_os = "macos")]
    backends.push(DeviceType::MetalGpu);

    #[cfg(feature = "cuda")]
    backends.push(DeviceType::CudaGpu);

    #[cfg(feature = "vulkan")]
    backends.push(DeviceType::VulkanGpu);

    #[cfg(feature = "opencl")]
    backends.push(DeviceType::OpenClDevice);

    #[cfg(feature = "rocm")]
    backends.push(DeviceType::RocmGpu);

    backends
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_availability() {
        println!("GPU available: {}", is_gpu_available());
        println!("Available backends: {:?}", available_gpu_backends());
    }
}
