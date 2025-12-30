//! Sampling strategies for dataloaders.
//!
//! Note: Samplers are designed for future integration with dataloaders.
//! Currently dataloaders use a built-in shuffle mechanism.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use tsai_core::Seed;

/// A sampler that produces indices for iteration.
///
/// Note: Reserved for future integration with dataloaders.
#[allow(dead_code)]
pub trait Sampler: Send + Sync {
    /// Get the indices for the next epoch.
    fn sample(&mut self, n: usize) -> Vec<usize>;

    /// Get the total number of samples.
    fn len(&self, n: usize) -> usize;
}

/// Sequential sampler that iterates indices in order.
#[derive(Debug, Clone, Default)]
pub struct SequentialSampler;

impl Sampler for SequentialSampler {
    fn sample(&mut self, n: usize) -> Vec<usize> {
        (0..n).collect()
    }

    fn len(&self, n: usize) -> usize {
        n
    }
}

/// Random sampler that shuffles indices each epoch.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct RandomSampler {
    rng: ChaCha8Rng,
}

impl RandomSampler {
    /// Create a new random sampler with a seed.
    #[must_use]
    pub fn new(seed: Seed) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed.value()),
        }
    }

    /// Create a random sampler with entropy-based seed.
    #[must_use]
    pub fn from_entropy() -> Self {
        Self {
            rng: ChaCha8Rng::from_entropy(),
        }
    }
}

impl Sampler for RandomSampler {
    fn sample(&mut self, n: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut self.rng);
        indices
    }

    fn len(&self, n: usize) -> usize {
        n
    }
}

/// Stratified sampler that maintains class balance.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct StratifiedSampler {
    rng: ChaCha8Rng,
    labels: Vec<usize>,
}

impl StratifiedSampler {
    /// Create a new stratified sampler.
    ///
    /// # Arguments
    ///
    /// * `labels` - Class labels for each sample
    /// * `seed` - Random seed for shuffling
    #[must_use]
    pub fn new(labels: Vec<usize>, seed: Seed) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed.value()),
            labels,
        }
    }
}

impl Sampler for StratifiedSampler {
    fn sample(&mut self, _n: usize) -> Vec<usize> {
        use std::collections::HashMap;

        // Group indices by class
        let mut class_indices: HashMap<usize, Vec<usize>> = HashMap::new();
        for (idx, &label) in self.labels.iter().enumerate() {
            class_indices.entry(label).or_default().push(idx);
        }

        // Shuffle each class
        for indices in class_indices.values_mut() {
            indices.shuffle(&mut self.rng);
        }

        // Interleave classes (round-robin)
        let mut result = Vec::with_capacity(self.labels.len());
        let _n_classes = class_indices.len();
        let max_len = class_indices.values().map(|v| v.len()).max().unwrap_or(0);

        let mut class_keys: Vec<usize> = class_indices.keys().copied().collect();
        class_keys.sort();

        for i in 0..max_len {
            for &class in &class_keys {
                if let Some(indices) = class_indices.get(&class) {
                    if i < indices.len() {
                        result.push(indices[i]);
                    }
                }
            }
        }

        result
    }

    fn len(&self, _n: usize) -> usize {
        self.labels.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_sampler() {
        let mut sampler = SequentialSampler;
        let indices = sampler.sample(10);
        assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_random_sampler_determinism() {
        let seed = Seed::new(42);
        let mut sampler1 = RandomSampler::new(seed);
        let mut sampler2 = RandomSampler::new(seed);

        let indices1 = sampler1.sample(10);
        let indices2 = sampler2.sample(10);

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_random_sampler_shuffles() {
        let mut sampler = RandomSampler::new(Seed::new(42));
        let indices = sampler.sample(100);

        // Very unlikely to be in order
        let in_order: Vec<usize> = (0..100).collect();
        assert_ne!(indices, in_order);
    }

    #[test]
    fn test_stratified_sampler() {
        let labels = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let mut sampler = StratifiedSampler::new(labels, Seed::new(42));
        let indices = sampler.sample(9);

        assert_eq!(indices.len(), 9);
        // Check all indices are present
        let mut sorted = indices.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
    }
}
