//! Time series dataset types.

use ndarray::{Array2, Array3};

use crate::error::{DataError, Result};

/// A dataset of time series samples.
///
/// Stores time series data in the `(N, V, L)` format:
/// - `N`: Number of samples
/// - `V`: Variables/channels
/// - `L`: Sequence length
///
/// # Example
///
/// ```rust,ignore
/// use tsai_data::TSDataset;
/// use ndarray::Array3;
///
/// let x = Array3::<f32>::zeros((100, 3, 50));
/// let y = Array2::<f32>::zeros((100, 1));
/// let dataset = TSDataset::from_arrays(x, Some(y))?;
/// ```
#[derive(Debug, Clone)]
pub struct TSDataset {
    /// Input data (N, V, L)
    x: Array3<f32>,
    /// Optional targets (N, T) where T is the target dimension
    y: Option<Array2<f32>>,
    /// Optional sample weights
    weights: Option<Vec<f32>>,
}

impl TSDataset {
    /// Create a new dataset from arrays.
    ///
    /// # Arguments
    ///
    /// * `x` - Input array of shape (N, V, L)
    /// * `y` - Optional target array of shape (N, T)
    ///
    /// # Errors
    ///
    /// Returns an error if the batch dimensions don't match.
    pub fn from_arrays(x: Array3<f32>, y: Option<Array2<f32>>) -> Result<Self> {
        let n_samples = x.shape()[0];

        if let Some(ref targets) = y {
            if targets.shape()[0] != n_samples {
                return Err(DataError::InvalidShape(format!(
                    "x has {} samples but y has {} samples",
                    n_samples,
                    targets.shape()[0]
                )));
            }
        }

        Ok(Self { x, y, weights: None })
    }

    /// Create an empty dataset.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            x: Array3::zeros((0, 0, 0)),
            y: None,
            weights: None,
        }
    }

    /// Get the number of samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.x.shape()[0]
    }

    /// Check if the dataset is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.x.shape()[0] == 0
    }

    /// Get the number of variables.
    #[must_use]
    pub fn n_vars(&self) -> usize {
        self.x.shape()[1]
    }

    /// Get the sequence length.
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.x.shape()[2]
    }

    /// Get the shape as (N, V, L).
    #[must_use]
    pub fn shape(&self) -> (usize, usize, usize) {
        let s = self.x.shape();
        (s[0], s[1], s[2])
    }

    /// Get a reference to the input data.
    #[must_use]
    pub fn x(&self) -> &Array3<f32> {
        &self.x
    }

    /// Get a reference to the targets.
    #[must_use]
    pub fn y(&self) -> Option<&Array2<f32>> {
        self.y.as_ref()
    }

    /// Check if the dataset has targets.
    #[must_use]
    pub fn has_targets(&self) -> bool {
        self.y.is_some()
    }

    /// Set sample weights.
    pub fn set_weights(&mut self, weights: Vec<f32>) -> Result<()> {
        if weights.len() != self.len() {
            return Err(DataError::InvalidShape(format!(
                "Expected {} weights but got {}",
                self.len(),
                weights.len()
            )));
        }
        self.weights = Some(weights);
        Ok(())
    }

    /// Get sample weights.
    #[must_use]
    pub fn weights(&self) -> Option<&[f32]> {
        self.weights.as_deref()
    }

    /// Get a sample by index.
    ///
    /// Returns the input slice and optionally the target.
    pub fn get(&self, index: usize) -> Result<(ndarray::ArrayView2<'_, f32>, Option<ndarray::ArrayView1<'_, f32>>)> {
        if index >= self.len() {
            return Err(DataError::IndexOutOfBounds {
                index,
                length: self.len(),
            });
        }

        let x = self.x.index_axis(ndarray::Axis(0), index);
        let y = self.y.as_ref().map(|y| y.index_axis(ndarray::Axis(0), index));

        Ok((x, y))
    }

    /// Get a subset of samples by indices.
    pub fn subset(&self, indices: &[usize]) -> Result<Self> {
        for &idx in indices {
            if idx >= self.len() {
                return Err(DataError::IndexOutOfBounds {
                    index: idx,
                    length: self.len(),
                });
            }
        }

        let x = ndarray::stack(
            ndarray::Axis(0),
            &indices
                .iter()
                .map(|&i| self.x.index_axis(ndarray::Axis(0), i))
                .collect::<Vec<_>>(),
        )
        .map_err(|e| DataError::Other(e.to_string()))?;

        let y = self.y.as_ref().map(|y| {
            ndarray::stack(
                ndarray::Axis(0),
                &indices
                    .iter()
                    .map(|&i| y.index_axis(ndarray::Axis(0), i))
                    .collect::<Vec<_>>(),
            )
            .unwrap()
        });

        let weights = self.weights.as_ref().map(|w| indices.iter().map(|&i| w[i]).collect());

        Ok(Self { x, y, weights })
    }

    /// Concatenate two datasets.
    pub fn concat(&self, other: &Self) -> Result<Self> {
        if self.is_empty() {
            return Ok(other.clone());
        }
        if other.is_empty() {
            return Ok(self.clone());
        }

        if self.n_vars() != other.n_vars() || self.seq_len() != other.seq_len() {
            return Err(DataError::InvalidShape(format!(
                "Cannot concatenate datasets with different shapes: ({}, {}) vs ({}, {})",
                self.n_vars(),
                self.seq_len(),
                other.n_vars(),
                other.seq_len()
            )));
        }

        let x = ndarray::concatenate(ndarray::Axis(0), &[self.x.view(), other.x.view()])
            .map_err(|e| DataError::Other(e.to_string()))?;

        let y = match (&self.y, &other.y) {
            (Some(y1), Some(y2)) => Some(
                ndarray::concatenate(ndarray::Axis(0), &[y1.view(), y2.view()])
                    .map_err(|e| DataError::Other(e.to_string()))?,
            ),
            _ => None,
        };

        let weights = match (&self.weights, &other.weights) {
            (Some(w1), Some(w2)) => {
                let mut w = w1.clone();
                w.extend(w2);
                Some(w)
            }
            _ => None,
        };

        Ok(Self { x, y, weights })
    }
}

/// A collection of datasets for train/valid/test splits.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_data::{TSDataset, TSDatasets};
///
/// let datasets = TSDatasets::new(train_ds)
///     .with_valid(valid_ds)
///     .with_test(test_ds);
/// ```
#[derive(Debug, Clone)]
pub struct TSDatasets {
    /// Training dataset
    train: TSDataset,
    /// Validation dataset
    valid: Option<TSDataset>,
    /// Test dataset
    test: Option<TSDataset>,
    /// Unlabeled dataset
    unlabeled: Option<TSDataset>,
}

impl TSDatasets {
    /// Create a new TSDatasets with just a training set.
    #[must_use]
    pub fn new(train: TSDataset) -> Self {
        Self {
            train,
            valid: None,
            test: None,
            unlabeled: None,
        }
    }

    /// Add a validation dataset.
    #[must_use]
    pub fn with_valid(mut self, valid: TSDataset) -> Self {
        self.valid = Some(valid);
        self
    }

    /// Add a test dataset.
    #[must_use]
    pub fn with_test(mut self, test: TSDataset) -> Self {
        self.test = Some(test);
        self
    }

    /// Add an unlabeled dataset.
    #[must_use]
    pub fn with_unlabeled(mut self, unlabeled: TSDataset) -> Self {
        self.unlabeled = Some(unlabeled);
        self
    }

    /// Get the training dataset.
    #[must_use]
    pub fn train(&self) -> &TSDataset {
        &self.train
    }

    /// Get the validation dataset.
    #[must_use]
    pub fn valid(&self) -> Option<&TSDataset> {
        self.valid.as_ref()
    }

    /// Get the test dataset.
    #[must_use]
    pub fn test(&self) -> Option<&TSDataset> {
        self.test.as_ref()
    }

    /// Get the unlabeled dataset.
    #[must_use]
    pub fn unlabeled(&self) -> Option<&TSDataset> {
        self.unlabeled.as_ref()
    }

    /// Get the number of variables (from training set).
    #[must_use]
    pub fn n_vars(&self) -> usize {
        self.train.n_vars()
    }

    /// Get the sequence length (from training set).
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.train.seq_len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_dataset(n: usize, v: usize, l: usize) -> TSDataset {
        let x = Array3::zeros((n, v, l));
        let y = Array2::zeros((n, 1));
        TSDataset::from_arrays(x, Some(y)).unwrap()
    }

    #[test]
    fn test_dataset_creation() {
        let ds = create_test_dataset(100, 3, 50);
        assert_eq!(ds.len(), 100);
        assert_eq!(ds.n_vars(), 3);
        assert_eq!(ds.seq_len(), 50);
        assert!(ds.has_targets());
    }

    #[test]
    fn test_dataset_get() {
        let ds = create_test_dataset(100, 3, 50);
        let (x, y) = ds.get(0).unwrap();
        assert_eq!(x.shape(), &[3, 50]);
        assert!(y.is_some());
    }

    #[test]
    fn test_dataset_subset() {
        let ds = create_test_dataset(100, 3, 50);
        let subset = ds.subset(&[0, 10, 20]).unwrap();
        assert_eq!(subset.len(), 3);
    }

    #[test]
    fn test_dataset_concat() {
        let ds1 = create_test_dataset(50, 3, 50);
        let ds2 = create_test_dataset(50, 3, 50);
        let combined = ds1.concat(&ds2).unwrap();
        assert_eq!(combined.len(), 100);
    }

    #[test]
    fn test_datasets() {
        let train = create_test_dataset(100, 3, 50);
        let valid = create_test_dataset(20, 3, 50);

        let datasets = TSDatasets::new(train).with_valid(valid);

        assert_eq!(datasets.train().len(), 100);
        assert_eq!(datasets.valid().unwrap().len(), 20);
        assert!(datasets.test().is_none());
    }
}
