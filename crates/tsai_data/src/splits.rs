//! Dataset splitting utilities.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::dataset::TSDataset;
use crate::error::{DataError, Result};
use tsai_core::Seed;

/// Strategy for splitting datasets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SplitStrategy {
    /// Random split with given ratio.
    Random {
        /// Ratio for the first split (e.g., 0.8 for 80% train).
        ratio: f32,
    },
    /// Stratified split maintaining class distribution.
    Stratified {
        /// Ratio for the first split.
        ratio: f32,
    },
    /// Fixed indices for each split.
    Fixed {
        /// Indices for the first split.
        first_indices: Vec<usize>,
        /// Indices for the second split.
        second_indices: Vec<usize>,
    },
}

/// Split a dataset into train and test sets.
///
/// # Arguments
///
/// * `dataset` - The dataset to split
/// * `test_ratio` - Ratio for the test set (e.g., 0.2 for 20%)
/// * `seed` - Random seed for reproducibility
///
/// # Returns
///
/// A tuple of (train_dataset, test_dataset).
pub fn train_test_split(
    dataset: &TSDataset,
    test_ratio: f32,
    seed: Seed,
) -> Result<(TSDataset, TSDataset)> {
    if test_ratio <= 0.0 || test_ratio >= 1.0 {
        return Err(DataError::SplitError(format!(
            "test_ratio must be between 0 and 1, got {}",
            test_ratio
        )));
    }

    let n = dataset.len();
    let n_test = (n as f32 * test_ratio).round() as usize;
    let n_test = n_test.max(1).min(n - 1);

    let mut rng = ChaCha8Rng::seed_from_u64(seed.value());
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);

    let train_indices: Vec<usize> = indices[n_test..].to_vec();
    let test_indices: Vec<usize> = indices[..n_test].to_vec();

    let train = dataset.subset(&train_indices)?;
    let test = dataset.subset(&test_indices)?;

    Ok((train, test))
}

/// Split a dataset into train, validation, and test sets.
///
/// # Arguments
///
/// * `dataset` - The dataset to split
/// * `valid_ratio` - Ratio for the validation set
/// * `test_ratio` - Ratio for the test set
/// * `seed` - Random seed for reproducibility
///
/// # Returns
///
/// A tuple of (train_dataset, valid_dataset, test_dataset).
pub fn train_valid_test_split(
    dataset: &TSDataset,
    valid_ratio: f32,
    test_ratio: f32,
    seed: Seed,
) -> Result<(TSDataset, TSDataset, TSDataset)> {
    let total_ratio = valid_ratio + test_ratio;
    if total_ratio <= 0.0 || total_ratio >= 1.0 {
        return Err(DataError::SplitError(format!(
            "valid_ratio + test_ratio must be between 0 and 1, got {}",
            total_ratio
        )));
    }

    let n = dataset.len();
    let n_valid = (n as f32 * valid_ratio).round() as usize;
    let n_test = (n as f32 * test_ratio).round() as usize;

    let n_valid = n_valid.max(1);
    let n_test = n_test.max(1);

    if n_valid + n_test >= n {
        return Err(DataError::SplitError(format!(
            "Not enough samples for split: {} samples, but need {} for valid and {} for test",
            n, n_valid, n_test
        )));
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed.value());
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);

    let test_indices: Vec<usize> = indices[..n_test].to_vec();
    let valid_indices: Vec<usize> = indices[n_test..n_test + n_valid].to_vec();
    let train_indices: Vec<usize> = indices[n_test + n_valid..].to_vec();

    let train = dataset.subset(&train_indices)?;
    let valid = dataset.subset(&valid_indices)?;
    let test = dataset.subset(&test_indices)?;

    Ok((train, valid, test))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    fn create_test_dataset(n: usize) -> TSDataset {
        let x = Array3::zeros((n, 3, 50));
        let y = Array2::zeros((n, 1));
        TSDataset::from_arrays(x, Some(y)).unwrap()
    }

    #[test]
    fn test_train_test_split() {
        let ds = create_test_dataset(100);
        let (train, test) = train_test_split(&ds, 0.2, Seed::new(42)).unwrap();

        assert_eq!(train.len() + test.len(), 100);
        assert!(test.len() >= 15 && test.len() <= 25); // ~20%
    }

    #[test]
    fn test_train_test_split_determinism() {
        let ds = create_test_dataset(100);
        let (train1, test1) = train_test_split(&ds, 0.2, Seed::new(42)).unwrap();
        let (train2, test2) = train_test_split(&ds, 0.2, Seed::new(42)).unwrap();

        assert_eq!(train1.len(), train2.len());
        assert_eq!(test1.len(), test2.len());
    }

    #[test]
    fn test_train_valid_test_split() {
        let ds = create_test_dataset(100);
        let (train, valid, test) =
            train_valid_test_split(&ds, 0.1, 0.1, Seed::new(42)).unwrap();

        assert_eq!(train.len() + valid.len() + test.len(), 100);
        assert!(valid.len() >= 8 && valid.len() <= 12); // ~10%
        assert!(test.len() >= 8 && test.len() <= 12); // ~10%
    }

    #[test]
    fn test_split_invalid_ratio() {
        let ds = create_test_dataset(100);
        assert!(train_test_split(&ds, 0.0, Seed::new(42)).is_err());
        assert!(train_test_split(&ds, 1.0, Seed::new(42)).is_err());
    }
}
