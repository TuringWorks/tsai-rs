//! CPU feature detection.

use crate::device::SimdLevel;

/// Detect CPU SIMD capabilities.
pub fn detect_simd_level() -> SimdLevel {
    SimdLevel::detect()
}

/// Get the number of physical CPU cores.
pub fn physical_cores() -> usize {
    num_cpus::get_physical()
}

/// Get the number of logical CPU cores (including hyperthreading).
pub fn logical_cores() -> usize {
    num_cpus::get()
}

/// Get total system memory in bytes.
pub fn total_memory() -> u64 {
    sys_info::mem_info()
        .map(|m| m.total * 1024)
        .unwrap_or(0)
}

/// Get available system memory in bytes.
pub fn available_memory() -> u64 {
    sys_info::mem_info()
        .map(|m| m.avail * 1024)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_detection() {
        let simd = detect_simd_level();
        println!("SIMD level: {:?}", simd);
        println!("Physical cores: {}", physical_cores());
        println!("Logical cores: {}", logical_cores());
        println!("Total memory: {} GB", total_memory() / (1024 * 1024 * 1024));
    }
}
