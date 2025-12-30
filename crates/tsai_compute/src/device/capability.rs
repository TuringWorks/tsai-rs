//! Device capability descriptors.
//!
//! This module provides types for describing device capabilities,
//! features, and compute characteristics.

use std::fmt;

/// Comprehensive device capability descriptor.
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Compute capability (e.g., CUDA compute capability, Metal family).
    pub compute_version: ComputeVersion,

    /// Number of compute units (cores, SMs, CUs).
    pub compute_units: u32,

    /// Maximum work group / block size.
    pub max_work_group_size: u32,

    /// Maximum dimensions for work groups.
    pub max_work_group_dims: [u32; 3],

    /// Maximum shared memory per work group (bytes).
    pub max_shared_memory: u64,

    /// Maximum allocatable buffer size (bytes).
    pub max_buffer_size: u64,

    /// Total device memory (bytes).
    pub total_memory: u64,

    /// Memory bandwidth (GB/s).
    pub memory_bandwidth_gbps: f32,

    /// Peak theoretical TFLOPS (FP32).
    pub peak_tflops_fp32: f32,

    /// Peak theoretical TFLOPS (FP16), if supported.
    pub peak_tflops_fp16: Option<f32>,

    /// Peak theoretical TFLOPS (INT8), if supported.
    pub peak_tflops_int8: Option<f32>,

    /// Supported precision types.
    pub supported_precisions: Vec<Precision>,

    /// Feature flags.
    pub features: Vec<DeviceFeature>,

    /// NUMA node affinity (for CPU).
    pub numa_node: Option<u32>,

    /// SIMD width (for CPU).
    pub simd_width: Option<u32>,

    /// Whether this is an integrated device.
    pub is_integrated: bool,

    /// Vendor name.
    pub vendor: String,

    /// Driver version.
    pub driver_version: String,
}

impl DeviceCapabilities {
    /// Create a new capability descriptor with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create CPU capabilities.
    pub fn cpu(simd_level: SimdLevel, numa_node: Option<u32>) -> Self {
        let simd_width = simd_level.vector_width();
        Self {
            compute_version: ComputeVersion::Cpu { simd: simd_level },
            compute_units: num_cpus::get() as u32,
            max_work_group_size: 1,
            max_work_group_dims: [1, 1, 1],
            max_shared_memory: 0,
            max_buffer_size: u64::MAX,
            total_memory: sys_info::mem_info().map(|m| m.total * 1024).unwrap_or(0),
            memory_bandwidth_gbps: 50.0, // Typical DDR4/DDR5
            peak_tflops_fp32: 0.0,       // Depends on CPU
            peak_tflops_fp16: None,
            peak_tflops_int8: None,
            supported_precisions: vec![
                Precision::Float32,
                Precision::Float64,
                Precision::Int32,
                Precision::Int64,
            ],
            features: Self::cpu_features(simd_level),
            numa_node,
            simd_width: Some(simd_width),
            is_integrated: true,
            vendor: "CPU".to_string(),
            driver_version: String::new(),
        }
    }

    fn cpu_features(simd: SimdLevel) -> Vec<DeviceFeature> {
        let mut features = vec![DeviceFeature::Float64];

        match simd {
            SimdLevel::Avx512 => {
                features.extend([
                    DeviceFeature::Avx512,
                    DeviceFeature::Avx2,
                    DeviceFeature::Fma,
                ]);
            }
            SimdLevel::Avx2 => {
                features.extend([DeviceFeature::Avx2, DeviceFeature::Fma]);
            }
            SimdLevel::Avx => {
                features.push(DeviceFeature::Avx2);
            }
            SimdLevel::Neon => {
                features.push(DeviceFeature::Neon);
            }
            SimdLevel::Sve => {
                features.extend([DeviceFeature::Sve, DeviceFeature::Neon]);
            }
            _ => {}
        }

        features
    }

    /// Check if a specific feature is supported.
    pub fn has_feature(&self, feature: DeviceFeature) -> bool {
        self.features.contains(&feature)
    }

    /// Check if a precision is supported.
    pub fn supports_precision(&self, precision: Precision) -> bool {
        self.supported_precisions.contains(&precision)
    }

    /// Get memory in gigabytes.
    pub fn memory_gb(&self) -> f64 {
        self.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self {
            compute_version: ComputeVersion::Unknown,
            compute_units: 1,
            max_work_group_size: 1,
            max_work_group_dims: [1, 1, 1],
            max_shared_memory: 0,
            max_buffer_size: u64::MAX,
            total_memory: 0,
            memory_bandwidth_gbps: 0.0,
            peak_tflops_fp32: 0.0,
            peak_tflops_fp16: None,
            peak_tflops_int8: None,
            supported_precisions: vec![Precision::Float32],
            features: vec![],
            numa_node: None,
            simd_width: None,
            is_integrated: false,
            vendor: String::new(),
            driver_version: String::new(),
        }
    }
}

/// Feature flags for capability querying.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceFeature {
    // CPU SIMD features
    /// SSE2 SIMD instructions.
    Sse2,
    /// SSE4 SIMD instructions.
    Sse4,
    /// AVX SIMD instructions.
    Avx,
    /// AVX2 SIMD instructions.
    Avx2,
    /// AVX-512 SIMD instructions.
    Avx512,
    /// Fused multiply-add instructions.
    Fma,
    /// ARM NEON SIMD instructions.
    Neon,
    /// ARM SVE SIMD instructions.
    Sve,

    // NUMA
    /// NUMA topology awareness.
    NumaAwareness,

    // Precision features
    /// 16-bit floating point support.
    Float16,
    /// Brain floating point (bfloat16) support.
    BFloat16,
    /// 64-bit floating point support.
    Float64,
    /// 8-bit integer compute support.
    Int8Compute,

    // GPU-specific features
    /// NVIDIA tensor cores or AMD matrix cores.
    TensorCores,
    /// Ray tracing acceleration.
    RayTracing,
    /// Mesh shader support.
    MeshShaders,

    // Memory features
    /// Shared memory between compute units.
    SharedMemory,
    /// Unified memory address space.
    UnifiedMemory,
    /// Pinned host memory for fast transfers.
    PinnedMemory,
    /// Managed memory with automatic migration.
    ManagedMemory,
    /// Asynchronous memory transfers.
    AsyncTransfer,
    /// Peer-to-peer device communication.
    PeerToPeer,

    // Execution features
    /// Asynchronous compute capability.
    AsyncCompute,
    /// Multiple command queue support.
    MultiQueue,
    /// Subgroup/warp operations.
    Subgroups,
    /// Cooperative group operations.
    CooperativeGroups,

    // General compute features
    /// General compute capability.
    Compute,
    /// Discrete GPU (not integrated).
    DiscreteGpu,
}

impl fmt::Display for DeviceFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceFeature::Sse2 => write!(f, "SSE2"),
            DeviceFeature::Sse4 => write!(f, "SSE4"),
            DeviceFeature::Avx => write!(f, "AVX"),
            DeviceFeature::Avx2 => write!(f, "AVX2"),
            DeviceFeature::Avx512 => write!(f, "AVX-512"),
            DeviceFeature::Fma => write!(f, "FMA"),
            DeviceFeature::Neon => write!(f, "NEON"),
            DeviceFeature::Sve => write!(f, "SVE"),
            DeviceFeature::NumaAwareness => write!(f, "NUMA"),
            DeviceFeature::Float16 => write!(f, "FP16"),
            DeviceFeature::BFloat16 => write!(f, "BF16"),
            DeviceFeature::Float64 => write!(f, "FP64"),
            DeviceFeature::Int8Compute => write!(f, "INT8"),
            DeviceFeature::TensorCores => write!(f, "TensorCores"),
            DeviceFeature::RayTracing => write!(f, "RayTracing"),
            DeviceFeature::MeshShaders => write!(f, "MeshShaders"),
            DeviceFeature::SharedMemory => write!(f, "SharedMem"),
            DeviceFeature::UnifiedMemory => write!(f, "UnifiedMem"),
            DeviceFeature::PinnedMemory => write!(f, "PinnedMem"),
            DeviceFeature::ManagedMemory => write!(f, "ManagedMem"),
            DeviceFeature::AsyncTransfer => write!(f, "AsyncTransfer"),
            DeviceFeature::PeerToPeer => write!(f, "P2P"),
            DeviceFeature::AsyncCompute => write!(f, "AsyncCompute"),
            DeviceFeature::MultiQueue => write!(f, "MultiQueue"),
            DeviceFeature::Subgroups => write!(f, "Subgroups"),
            DeviceFeature::CooperativeGroups => write!(f, "CoopGroups"),
            DeviceFeature::Compute => write!(f, "Compute"),
            DeviceFeature::DiscreteGpu => write!(f, "DiscreteGPU"),
        }
    }
}

/// Supported precision types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    /// 16-bit floating point (half precision).
    Float16,
    /// 16-bit brain floating point (for ML).
    BFloat16,
    /// 32-bit floating point (single precision).
    Float32,
    /// 64-bit floating point (double precision).
    Float64,
    /// 8-bit signed integer.
    Int8,
    /// 16-bit signed integer.
    Int16,
    /// 32-bit signed integer.
    Int32,
    /// 64-bit signed integer.
    Int64,
    /// 8-bit unsigned integer.
    UInt8,
    /// 16-bit unsigned integer.
    UInt16,
    /// 32-bit unsigned integer.
    UInt32,
    /// 64-bit unsigned integer.
    UInt64,
}

impl Precision {
    /// Get the size in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            Precision::Int8 | Precision::UInt8 => 1,
            Precision::Float16 | Precision::BFloat16 | Precision::Int16 | Precision::UInt16 => 2,
            Precision::Float32 | Precision::Int32 | Precision::UInt32 => 4,
            Precision::Float64 | Precision::Int64 | Precision::UInt64 => 8,
        }
    }
}

/// Compute version for different backends.
#[derive(Debug, Clone, PartialEq)]
pub enum ComputeVersion {
    /// CPU with SIMD level.
    Cpu {
        /// SIMD instruction set level.
        simd: SimdLevel,
    },

    /// CUDA compute capability (e.g., 8.6 for RTX 3080).
    Cuda {
        /// Major version of compute capability.
        major: u32,
        /// Minor version of compute capability.
        minor: u32,
    },

    /// Metal GPU family.
    Metal {
        /// Metal GPU family identifier.
        family: MetalFamily,
    },

    /// Vulkan API version.
    Vulkan {
        /// Packed Vulkan API version number.
        api_version: u32,
    },

    /// OpenCL version.
    OpenCl {
        /// Major version of OpenCL.
        major: u32,
        /// Minor version of OpenCL.
        minor: u32,
    },

    /// ROCm/HIP architecture.
    Rocm {
        /// GFX architecture string (e.g., "gfx1030").
        gfx_arch: String,
    },

    /// Apple MLX framework.
    Mlx {
        /// MLX version string (e.g., "0.25").
        version: String,
    },

    /// Unknown or unspecified.
    Unknown,
}

impl ComputeVersion {
    /// Create a CUDA compute capability.
    pub fn cuda(major: u32, minor: u32) -> Self {
        ComputeVersion::Cuda { major, minor }
    }

    /// Create a Metal family.
    pub fn metal(family: MetalFamily) -> Self {
        ComputeVersion::Metal { family }
    }

    /// Create a Vulkan version.
    pub fn vulkan(api_version: u32) -> Self {
        ComputeVersion::Vulkan { api_version }
    }

    /// Create an OpenCL version.
    pub fn opencl(major: u32, minor: u32) -> Self {
        ComputeVersion::OpenCl { major, minor }
    }

    /// Create an MLX version.
    pub fn mlx(version: impl Into<String>) -> Self {
        ComputeVersion::Mlx { version: version.into() }
    }
}

impl fmt::Display for ComputeVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComputeVersion::Cpu { simd } => write!(f, "CPU-{}", simd),
            ComputeVersion::Cuda { major, minor } => write!(f, "CUDA {}.{}", major, minor),
            ComputeVersion::Metal { family } => write!(f, "Metal {:?}", family),
            ComputeVersion::Vulkan { api_version } => {
                let major = api_version >> 22;
                let minor = (api_version >> 12) & 0x3FF;
                let patch = api_version & 0xFFF;
                write!(f, "Vulkan {}.{}.{}", major, minor, patch)
            }
            ComputeVersion::OpenCl { major, minor } => write!(f, "OpenCL {}.{}", major, minor),
            ComputeVersion::Rocm { gfx_arch } => write!(f, "ROCm {}", gfx_arch),
            ComputeVersion::Mlx { version } => write!(f, "MLX {}", version),
            ComputeVersion::Unknown => write!(f, "Unknown"),
        }
    }
}

/// SIMD instruction set level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SimdLevel {
    /// No SIMD.
    None,
    /// SSE2 (128-bit, x86).
    Sse2,
    /// SSE4.1/4.2 (128-bit, x86).
    Sse4,
    /// AVX (256-bit, x86).
    Avx,
    /// AVX2 (256-bit with FMA, x86).
    Avx2,
    /// AVX-512 (512-bit, x86).
    Avx512,
    /// NEON (128-bit, ARM).
    Neon,
    /// SVE (scalable, ARM).
    Sve,
}

impl SimdLevel {
    /// Get the vector width in floats (f32).
    pub fn vector_width(&self) -> u32 {
        match self {
            SimdLevel::None => 1,
            SimdLevel::Sse2 | SimdLevel::Sse4 | SimdLevel::Neon => 4,
            SimdLevel::Avx | SimdLevel::Avx2 => 8,
            SimdLevel::Avx512 => 16,
            SimdLevel::Sve => 8, // Minimum for SVE
        }
    }

    /// Get the vector width in bytes.
    pub fn vector_bytes(&self) -> u32 {
        self.vector_width() * 4
    }

    /// Detect the current CPU's SIMD level.
    #[cfg(target_arch = "x86_64")]
    pub fn detect() -> Self {
        if is_x86_feature_detected!("avx512f") {
            SimdLevel::Avx512
        } else if is_x86_feature_detected!("avx2") {
            SimdLevel::Avx2
        } else if is_x86_feature_detected!("avx") {
            SimdLevel::Avx
        } else if is_x86_feature_detected!("sse4.1") {
            SimdLevel::Sse4
        } else if is_x86_feature_detected!("sse2") {
            SimdLevel::Sse2
        } else {
            SimdLevel::None
        }
    }

    /// Detect SIMD level on AArch64.
    #[cfg(target_arch = "aarch64")]
    pub fn detect() -> Self {
        // NEON is mandatory on AArch64
        // Check for SVE if available
        #[cfg(target_feature = "sve")]
        {
            SimdLevel::Sve
        }
        #[cfg(not(target_feature = "sve"))]
        {
            SimdLevel::Neon
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub fn detect() -> Self {
        SimdLevel::None
    }
}

impl fmt::Display for SimdLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimdLevel::None => write!(f, "Scalar"),
            SimdLevel::Sse2 => write!(f, "SSE2"),
            SimdLevel::Sse4 => write!(f, "SSE4"),
            SimdLevel::Avx => write!(f, "AVX"),
            SimdLevel::Avx2 => write!(f, "AVX2"),
            SimdLevel::Avx512 => write!(f, "AVX-512"),
            SimdLevel::Neon => write!(f, "NEON"),
            SimdLevel::Sve => write!(f, "SVE"),
        }
    }
}

/// Metal GPU family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetalFamily {
    /// Apple GPU family 1 (A7, A8).
    Apple1,
    /// Apple GPU family 2 (A8X).
    Apple2,
    /// Apple GPU family 3 (A9, A10).
    Apple3,
    /// Apple GPU family 4 (A11).
    Apple4,
    /// Apple GPU family 5 (A12).
    Apple5,
    /// Apple GPU family 6 (A13).
    Apple6,
    /// Apple GPU family 7 (A14, M1).
    Apple7,
    /// Apple GPU family 8 (A15, A16, M2).
    Apple8,
    /// Apple GPU family 9 (A17, M3).
    Apple9,
    /// Mac GPU family 1 (Intel-based Macs).
    Mac1,
    /// Mac GPU family 2 (Apple Silicon Macs).
    Mac2,
}

impl MetalFamily {
    /// Get the family number for comparison.
    pub fn family_number(&self) -> u32 {
        match self {
            MetalFamily::Apple1 => 1,
            MetalFamily::Apple2 => 2,
            MetalFamily::Apple3 => 3,
            MetalFamily::Apple4 => 4,
            MetalFamily::Apple5 => 5,
            MetalFamily::Apple6 => 6,
            MetalFamily::Apple7 => 7,
            MetalFamily::Apple8 => 8,
            MetalFamily::Apple9 => 9,
            MetalFamily::Mac1 => 101,
            MetalFamily::Mac2 => 102,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_detection() {
        let simd = SimdLevel::detect();
        // Should detect something on most modern CPUs
        println!("Detected SIMD level: {}", simd);
    }

    #[test]
    fn test_simd_vector_width() {
        assert_eq!(SimdLevel::None.vector_width(), 1);
        assert_eq!(SimdLevel::Sse2.vector_width(), 4);
        assert_eq!(SimdLevel::Avx2.vector_width(), 8);
        assert_eq!(SimdLevel::Avx512.vector_width(), 16);
    }

    #[test]
    fn test_compute_version_display() {
        let version = ComputeVersion::cuda(8, 6);
        assert_eq!(version.to_string(), "CUDA 8.6");

        let version = ComputeVersion::Cpu {
            simd: SimdLevel::Avx2,
        };
        assert_eq!(version.to_string(), "CPU-AVX2");
    }

    #[test]
    fn test_precision_size() {
        assert_eq!(Precision::Float16.size_bytes(), 2);
        assert_eq!(Precision::Float32.size_bytes(), 4);
        assert_eq!(Precision::Float64.size_bytes(), 8);
    }
}
