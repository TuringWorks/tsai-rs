//! Data split types for train/validation/test.

use serde::{Deserialize, Serialize};

/// Represents the data split type for training, validation, or testing.
///
/// This is used by transforms to apply different behavior based on the split.
/// For example, data augmentation is typically only applied during training.
///
/// # Example
///
/// ```rust
/// use tsai_core::Split;
///
/// let split = Split::Train;
/// assert!(split.is_train());
/// assert!(!split.is_valid());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Split {
    /// Training split - augmentations and regularization are applied.
    #[default]
    Train,
    /// Validation split - no augmentation, used for hyperparameter tuning.
    Valid,
    /// Test split - no augmentation, used for final evaluation.
    Test,
}

impl Split {
    /// Check if this is the training split.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tsai_core::Split;
    /// assert!(Split::Train.is_train());
    /// assert!(!Split::Valid.is_train());
    /// ```
    #[must_use]
    pub const fn is_train(&self) -> bool {
        matches!(self, Split::Train)
    }

    /// Check if this is the validation split.
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        matches!(self, Split::Valid)
    }

    /// Check if this is the test split.
    #[must_use]
    pub const fn is_test(&self) -> bool {
        matches!(self, Split::Test)
    }

    /// Check if this is an evaluation split (valid or test).
    ///
    /// Useful for disabling augmentations during evaluation.
    #[must_use]
    pub const fn is_eval(&self) -> bool {
        matches!(self, Split::Valid | Split::Test)
    }

    /// Get the split index (0=Train, 1=Valid, 2=Test).
    ///
    /// This mirrors the `split_idx` convention in Python tsai.
    #[must_use]
    pub const fn index(&self) -> usize {
        match self {
            Split::Train => 0,
            Split::Valid => 1,
            Split::Test => 2,
        }
    }

    /// Create a Split from an index.
    ///
    /// Returns None if the index is out of range.
    #[must_use]
    pub const fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(Split::Train),
            1 => Some(Split::Valid),
            2 => Some(Split::Test),
            _ => None,
        }
    }
}

impl std::fmt::Display for Split {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Split::Train => write!(f, "train"),
            Split::Valid => write!(f, "valid"),
            Split::Test => write!(f, "test"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_checks() {
        assert!(Split::Train.is_train());
        assert!(!Split::Train.is_valid());
        assert!(!Split::Train.is_test());
        assert!(!Split::Train.is_eval());

        assert!(!Split::Valid.is_train());
        assert!(Split::Valid.is_valid());
        assert!(!Split::Valid.is_test());
        assert!(Split::Valid.is_eval());

        assert!(!Split::Test.is_train());
        assert!(!Split::Test.is_valid());
        assert!(Split::Test.is_test());
        assert!(Split::Test.is_eval());
    }

    #[test]
    fn test_split_index() {
        assert_eq!(Split::Train.index(), 0);
        assert_eq!(Split::Valid.index(), 1);
        assert_eq!(Split::Test.index(), 2);

        assert_eq!(Split::from_index(0), Some(Split::Train));
        assert_eq!(Split::from_index(1), Some(Split::Valid));
        assert_eq!(Split::from_index(2), Some(Split::Test));
        assert_eq!(Split::from_index(3), None);
    }

    #[test]
    fn test_split_display() {
        assert_eq!(format!("{}", Split::Train), "train");
        assert_eq!(format!("{}", Split::Valid), "valid");
        assert_eq!(format!("{}", Split::Test), "test");
    }
}
