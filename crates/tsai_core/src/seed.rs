//! Deterministic random number generation utilities.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// A seed for deterministic random number generation.
///
/// Using the same seed will produce the same sequence of random numbers,
/// ensuring reproducibility across runs.
///
/// # Example
///
/// ```rust
/// use tsai_core::Seed;
/// use rand::Rng;
///
/// let seed = Seed::new(42);
/// let mut rng = seed.to_rng();
///
/// // Same seed produces same results
/// let seed2 = Seed::new(42);
/// let mut rng2 = seed2.to_rng();
///
/// let val1: f32 = rng.gen();
/// let val2: f32 = rng2.gen();
/// assert_eq!(val1, val2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Seed(u64);

impl Seed {
    /// Create a new seed with the given value.
    ///
    /// # Arguments
    ///
    /// * `value` - The seed value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tsai_core::Seed;
    /// let seed = Seed::new(42);
    /// ```
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Create a seed from the current system time.
    ///
    /// This is useful for non-reproducible random behavior.
    #[must_use]
    pub fn from_entropy() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let duration = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        Self(duration.as_nanos() as u64)
    }

    /// Get the underlying seed value.
    #[must_use]
    pub const fn value(&self) -> u64 {
        self.0
    }

    /// Create a new random number generator from this seed.
    ///
    /// Uses ChaCha8 for cryptographically secure, reproducible random numbers.
    #[must_use]
    pub fn to_rng(&self) -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(self.0)
    }

    /// Derive a new seed from this seed using a key.
    ///
    /// This is useful for creating independent random streams
    /// from a single master seed.
    ///
    /// # Arguments
    ///
    /// * `key` - A string key to derive the new seed.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tsai_core::Seed;
    ///
    /// let master = Seed::new(42);
    /// let shuffle_seed = master.derive("shuffle");
    /// let noise_seed = master.derive("noise");
    ///
    /// // Different keys produce different seeds
    /// assert_ne!(shuffle_seed.value(), noise_seed.value());
    /// ```
    #[must_use]
    pub fn derive(&self, key: &str) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.0.hash(&mut hasher);
        key.hash(&mut hasher);
        Self(hasher.finish())
    }
}

impl Default for Seed {
    fn default() -> Self {
        Self::new(0)
    }
}

impl From<u64> for Seed {
    fn from(value: u64) -> Self {
        Self::new(value)
    }
}

impl From<Seed> for u64 {
    fn from(seed: Seed) -> Self {
        seed.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_seed_reproducibility() {
        let seed1 = Seed::new(42);
        let seed2 = Seed::new(42);

        let mut rng1 = seed1.to_rng();
        let mut rng2 = seed2.to_rng();

        for _ in 0..100 {
            let val1: f64 = rng1.gen();
            let val2: f64 = rng2.gen();
            assert_eq!(val1, val2);
        }
    }

    #[test]
    fn test_seed_derive() {
        let master = Seed::new(42);
        let derived1 = master.derive("key1");
        let derived2 = master.derive("key2");
        let derived1_again = master.derive("key1");

        assert_ne!(derived1.value(), derived2.value());
        assert_eq!(derived1.value(), derived1_again.value());
    }

    #[test]
    fn test_seed_serialization() {
        let seed = Seed::new(12345);
        let json = serde_json::to_string(&seed).unwrap();
        let restored: Seed = serde_json::from_str(&json).unwrap();
        assert_eq!(seed, restored);
    }
}
