//! SIMD dispatch for CPU backend.
//!
//! Provides runtime SIMD detection and dispatch for vectorized operations.

use crate::device::SimdLevel;

/// SIMD dispatch handler.
///
/// Detects CPU SIMD capabilities at runtime and dispatches
/// operations to the appropriate implementation.
#[derive(Debug, Clone)]
pub struct SimdDispatch {
    level: SimdLevel,
}

impl SimdDispatch {
    /// Create a new SIMD dispatcher with automatic detection.
    pub fn new() -> Self {
        Self {
            level: SimdLevel::detect(),
        }
    }

    /// Create a dispatcher with a specific SIMD level.
    pub fn with_level(level: SimdLevel) -> Self {
        Self { level }
    }

    /// Get the detected SIMD level.
    pub fn level(&self) -> SimdLevel {
        self.level
    }

    /// Get the vector width in floats.
    pub fn vector_width(&self) -> usize {
        self.level.vector_width() as usize
    }

    /// Dispatch a vectorized operation.
    ///
    /// Calls the appropriate function based on the detected SIMD level.
    #[inline]
    pub fn dispatch<T, FAvx512, FAvx2, FNeon, FScalar>(
        &self,
        avx512_fn: FAvx512,
        avx2_fn: FAvx2,
        neon_fn: FNeon,
        scalar_fn: FScalar,
    ) -> T
    where
        FAvx512: FnOnce() -> T,
        FAvx2: FnOnce() -> T,
        FNeon: FnOnce() -> T,
        FScalar: FnOnce() -> T,
    {
        match self.level {
            SimdLevel::Avx512 => avx512_fn(),
            SimdLevel::Avx2 | SimdLevel::Avx => avx2_fn(),
            SimdLevel::Neon | SimdLevel::Sve => neon_fn(),
            _ => scalar_fn(),
        }
    }

    /// Check if AVX2 or better is available.
    pub fn has_avx2(&self) -> bool {
        matches!(self.level, SimdLevel::Avx2 | SimdLevel::Avx512)
    }

    /// Check if AVX-512 is available.
    pub fn has_avx512(&self) -> bool {
        matches!(self.level, SimdLevel::Avx512)
    }

    /// Check if NEON is available.
    pub fn has_neon(&self) -> bool {
        matches!(self.level, SimdLevel::Neon | SimdLevel::Sve)
    }
}

impl Default for SimdDispatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Vectorized operations using the `wide` crate for portable SIMD.
#[cfg(feature = "cpu")]
pub mod ops {
    use wide::f32x8;

    /// Vectorized element-wise addition.
    #[inline]
    pub fn add_f32_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), out.len());

        let chunks = a.len() / 8;
        let remainder = a.len() % 8;

        // Process 8 elements at a time
        for i in 0..chunks {
            let va = f32x8::from(&a[i * 8..i * 8 + 8]);
            let vb = f32x8::from(&b[i * 8..i * 8 + 8]);
            let vr = va + vb;
            out[i * 8..i * 8 + 8].copy_from_slice(vr.as_array_ref());
        }

        // Handle remainder
        let start = chunks * 8;
        for i in 0..remainder {
            out[start + i] = a[start + i] + b[start + i];
        }
    }

    /// Vectorized element-wise multiplication.
    #[inline]
    pub fn mul_f32_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), out.len());

        let chunks = a.len() / 8;
        let remainder = a.len() % 8;

        for i in 0..chunks {
            let va = f32x8::from(&a[i * 8..i * 8 + 8]);
            let vb = f32x8::from(&b[i * 8..i * 8 + 8]);
            let vr = va * vb;
            out[i * 8..i * 8 + 8].copy_from_slice(vr.as_array_ref());
        }

        let start = chunks * 8;
        for i in 0..remainder {
            out[start + i] = a[start + i] * b[start + i];
        }
    }

    /// Vectorized sum reduction.
    #[inline]
    pub fn sum_f32_avx2(data: &[f32]) -> f32 {
        let chunks = data.len() / 8;
        let remainder = data.len() % 8;

        let mut acc = f32x8::ZERO;

        for i in 0..chunks {
            let v = f32x8::from(&data[i * 8..i * 8 + 8]);
            acc += v;
        }

        // Horizontal sum of the vector
        let mut sum: f32 = acc.as_array_ref().iter().sum();

        // Add remainder
        let start = chunks * 8;
        for i in 0..remainder {
            sum += data[start + i];
        }

        sum
    }

    /// Vectorized dot product.
    #[inline]
    pub fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let chunks = a.len() / 8;
        let remainder = a.len() % 8;

        let mut acc = f32x8::ZERO;

        for i in 0..chunks {
            let va = f32x8::from(&a[i * 8..i * 8 + 8]);
            let vb = f32x8::from(&b[i * 8..i * 8 + 8]);
            acc += va * vb;
        }

        let mut sum: f32 = acc.as_array_ref().iter().sum();

        let start = chunks * 8;
        for i in 0..remainder {
            sum += a[start + i] * b[start + i];
        }

        sum
    }

    /// Scalar fallback for addition.
    #[inline]
    pub fn add_f32_scalar(a: &[f32], b: &[f32], out: &mut [f32]) {
        for ((a, b), o) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
            *o = *a + *b;
        }
    }

    /// Scalar fallback for multiplication.
    #[inline]
    pub fn mul_f32_scalar(a: &[f32], b: &[f32], out: &mut [f32]) {
        for ((a, b), o) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
            *o = *a * *b;
        }
    }

    /// Scalar fallback for sum.
    #[inline]
    pub fn sum_f32_scalar(data: &[f32]) -> f32 {
        data.iter().sum()
    }

    /// Scalar fallback for dot product.
    #[inline]
    pub fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(a, b)| a * b).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_detection() {
        let dispatch = SimdDispatch::new();
        println!("Detected SIMD level: {:?}", dispatch.level());
        println!("Vector width: {}", dispatch.vector_width());
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_vectorized_add() {
        let a: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..100).map(|i| i as f32 * 2.0).collect();
        let mut out = vec![0.0f32; 100];

        ops::add_f32_avx2(&a, &b, &mut out);

        for i in 0..100 {
            assert!((out[i] - (a[i] + b[i])).abs() < 1e-6);
        }
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_vectorized_dot() {
        let a: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..100).map(|_| 1.0).collect();

        let result = ops::dot_f32_avx2(&a, &b);
        let expected: f32 = (0..100).map(|i| i as f32).sum();

        assert!((result - expected).abs() < 1e-3);
    }
}
