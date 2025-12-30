//! Transform trait for data augmentation.

use burn::prelude::*;

use crate::error::Result;
use crate::split::Split;
use crate::tensor::TSBatch;

/// A transform that can be applied to time series batches.
///
/// Transforms are the primary mechanism for data augmentation in tsai-rs.
/// They can be composed and applied differently based on the data split
/// (train/valid/test).
///
/// # Implementation Notes
///
/// - Transforms should be deterministic given the same seed
/// - Transforms should return `Result` instead of panicking
/// - Transforms should document their effect on tensor shapes
///
/// # Example
///
/// ```rust,ignore
/// use tsai_core::{Transform, TSBatch, Split, Result};
/// use burn::prelude::*;
///
/// struct GaussianNoise {
///     std: f32,
/// }
///
/// impl<B: Backend> Transform<B> for GaussianNoise {
///     fn apply(&self, batch: TSBatch<B>, split: Split) -> Result<TSBatch<B>> {
///         // Only apply during training
///         if split.is_eval() {
///             return Ok(batch);
///         }
///         // Add noise...
///         Ok(batch)
///     }
/// }
/// ```
pub trait Transform<B: Backend>: Send + Sync {
    /// Apply the transform to a batch.
    ///
    /// # Arguments
    ///
    /// * `batch` - The input batch to transform
    /// * `split` - The data split (train/valid/test)
    ///
    /// # Returns
    ///
    /// The transformed batch, or an error if the transform fails.
    fn apply(&self, batch: TSBatch<B>, split: Split) -> Result<TSBatch<B>>;

    /// Get the name of this transform for logging/debugging.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// Check if this transform should be applied for the given split.
    ///
    /// By default, transforms are applied to all splits.
    /// Override this for augmentations that should only apply during training.
    fn should_apply(&self, _split: Split) -> bool {
        true
    }
}

/// Identity transform that passes through data unchanged.
///
/// Useful as a placeholder or for testing.
#[derive(Debug, Clone, Default)]
pub struct Identity;

impl<B: Backend> Transform<B> for Identity {
    fn apply(&self, batch: TSBatch<B>, _split: Split) -> Result<TSBatch<B>> {
        Ok(batch)
    }

    fn name(&self) -> &str {
        "Identity"
    }
}

/// A composed transform that applies multiple transforms in sequence.
#[derive(Default)]
pub struct Compose<B: Backend> {
    transforms: Vec<Box<dyn Transform<B>>>,
}

impl<B: Backend> Compose<B> {
    /// Create a new empty composition.
    #[must_use]
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    /// Add a transform to the composition.
    pub fn push<T: Transform<B> + 'static>(&mut self, transform: T) {
        self.transforms.push(Box::new(transform));
    }

    /// Create a composition from a vector of transforms.
    #[must_use]
    pub fn from_vec(transforms: Vec<Box<dyn Transform<B>>>) -> Self {
        Self { transforms }
    }
}

impl<B: Backend> Transform<B> for Compose<B> {
    fn apply(&self, mut batch: TSBatch<B>, split: Split) -> Result<TSBatch<B>> {
        for transform in &self.transforms {
            if transform.should_apply(split) {
                batch = transform.apply(batch, split)?;
            }
        }
        Ok(batch)
    }

    fn name(&self) -> &str {
        "Compose"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_identity_name() {
        let identity = Identity;
        assert_eq!(<Identity as Transform<TestBackend>>::name(&identity), "Identity");
    }

    #[test]
    fn test_split_should_apply() {
        let identity = Identity;
        assert!(<Identity as Transform<TestBackend>>::should_apply(&identity, Split::Train));
        assert!(<Identity as Transform<TestBackend>>::should_apply(&identity, Split::Valid));
        assert!(<Identity as Transform<TestBackend>>::should_apply(&identity, Split::Test));
    }
}
