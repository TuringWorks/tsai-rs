//! Time series shape metadata.

use serde::{Deserialize, Serialize};

use crate::error::{CoreError, Result};

/// Shape metadata for time series tensors.
///
/// Follows the convention `(B, V, L)`:
/// - `B`: Batch size (number of samples)
/// - `V`: Variables/channels/features
/// - `L`: Sequence length (time steps)
///
/// # Example
///
/// ```rust
/// use tsai_core::TSShape;
///
/// let shape = TSShape::new(32, 3, 100);
/// assert_eq!(shape.batch(), 32);
/// assert_eq!(shape.vars(), 3);
/// assert_eq!(shape.len(), 100);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TSShape {
    batch: usize,
    vars: usize,
    len: usize,
}

impl TSShape {
    /// Create a new TSShape with the specified dimensions.
    ///
    /// # Arguments
    ///
    /// * `batch` - Batch size (number of samples)
    /// * `vars` - Number of variables/channels
    /// * `len` - Sequence length (time steps)
    ///
    /// # Example
    ///
    /// ```rust
    /// use tsai_core::TSShape;
    /// let shape = TSShape::new(32, 3, 100);
    /// ```
    #[must_use]
    pub const fn new(batch: usize, vars: usize, len: usize) -> Self {
        Self { batch, vars, len }
    }

    /// Create a TSShape from a slice of dimensions.
    ///
    /// # Arguments
    ///
    /// * `dims` - A slice containing exactly 3 dimensions: [batch, vars, len]
    ///
    /// # Errors
    ///
    /// Returns an error if the slice doesn't contain exactly 3 elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tsai_core::TSShape;
    ///
    /// let shape = TSShape::from_dims(&[32, 3, 100]).unwrap();
    /// assert_eq!(shape.batch(), 32);
    /// ```
    pub fn from_dims(dims: &[usize]) -> Result<Self> {
        if dims.len() != 3 {
            return Err(CoreError::DimensionError {
                expected: 3,
                got: dims.len(),
            });
        }
        Ok(Self::new(dims[0], dims[1], dims[2]))
    }

    /// Get the batch size.
    #[must_use]
    pub const fn batch(&self) -> usize {
        self.batch
    }

    /// Get the number of variables/channels.
    #[must_use]
    pub const fn vars(&self) -> usize {
        self.vars
    }

    /// Get the sequence length.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Check if this is an empty shape (any dimension is zero).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.batch == 0 || self.vars == 0 || self.len == 0
    }

    /// Get the total number of elements.
    #[must_use]
    pub const fn numel(&self) -> usize {
        self.batch * self.vars * self.len
    }

    /// Convert to a tuple.
    #[must_use]
    pub const fn as_tuple(&self) -> (usize, usize, usize) {
        (self.batch, self.vars, self.len)
    }

    /// Convert to an array.
    #[must_use]
    pub const fn as_array(&self) -> [usize; 3] {
        [self.batch, self.vars, self.len]
    }

    /// Create a new shape with a different batch size.
    #[must_use]
    pub const fn with_batch(&self, batch: usize) -> Self {
        Self {
            batch,
            vars: self.vars,
            len: self.len,
        }
    }

    /// Create a new shape with a different number of variables.
    #[must_use]
    pub const fn with_vars(&self, vars: usize) -> Self {
        Self {
            batch: self.batch,
            vars,
            len: self.len,
        }
    }

    /// Create a new shape with a different sequence length.
    #[must_use]
    pub const fn with_len(&self, len: usize) -> Self {
        Self {
            batch: self.batch,
            vars: self.vars,
            len,
        }
    }

    /// Check if this shape is compatible with another shape for operations
    /// that require matching vars and len dimensions.
    #[must_use]
    pub const fn is_compatible(&self, other: &Self) -> bool {
        self.vars == other.vars && self.len == other.len
    }
}

impl std::fmt::Display for TSShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(B={}, V={}, L={})", self.batch, self.vars, self.len)
    }
}

impl From<(usize, usize, usize)> for TSShape {
    fn from((batch, vars, len): (usize, usize, usize)) -> Self {
        Self::new(batch, vars, len)
    }
}

impl From<[usize; 3]> for TSShape {
    fn from([batch, vars, len]: [usize; 3]) -> Self {
        Self::new(batch, vars, len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_creation() {
        let shape = TSShape::new(32, 3, 100);
        assert_eq!(shape.batch(), 32);
        assert_eq!(shape.vars(), 3);
        assert_eq!(shape.len(), 100);
    }

    #[test]
    fn test_shape_from_dims() {
        let shape = TSShape::from_dims(&[32, 3, 100]).unwrap();
        assert_eq!(shape.as_tuple(), (32, 3, 100));

        assert!(TSShape::from_dims(&[32, 3]).is_err());
        assert!(TSShape::from_dims(&[32, 3, 100, 1]).is_err());
    }

    #[test]
    fn test_shape_numel() {
        let shape = TSShape::new(32, 3, 100);
        assert_eq!(shape.numel(), 32 * 3 * 100);
    }

    #[test]
    fn test_shape_is_empty() {
        assert!(!TSShape::new(32, 3, 100).is_empty());
        assert!(TSShape::new(0, 3, 100).is_empty());
        assert!(TSShape::new(32, 0, 100).is_empty());
        assert!(TSShape::new(32, 3, 0).is_empty());
    }

    #[test]
    fn test_shape_with_methods() {
        let shape = TSShape::new(32, 3, 100);
        assert_eq!(shape.with_batch(64).batch(), 64);
        assert_eq!(shape.with_vars(6).vars(), 6);
        assert_eq!(shape.with_len(200).len(), 200);
    }

    #[test]
    fn test_shape_compatibility() {
        let shape1 = TSShape::new(32, 3, 100);
        let shape2 = TSShape::new(64, 3, 100);
        let shape3 = TSShape::new(32, 6, 100);

        assert!(shape1.is_compatible(&shape2)); // different batch is ok
        assert!(!shape1.is_compatible(&shape3)); // different vars is not ok
    }

    #[test]
    fn test_shape_serialization() {
        let shape = TSShape::new(32, 3, 100);
        let json = serde_json::to_string(&shape).unwrap();
        let restored: TSShape = serde_json::from_str(&json).unwrap();
        assert_eq!(shape, restored);
    }
}
