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

/// Configuration for SlidingWindow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlidingWindowConfig {
    /// Window size (number of time steps per window).
    pub window_size: usize,
    /// Stride between consecutive windows.
    pub stride: usize,
    /// Horizon size for forecasting (output steps).
    pub horizon: usize,
    /// Whether to get only complete windows.
    pub drop_incomplete: bool,
}

impl Default for SlidingWindowConfig {
    fn default() -> Self {
        Self {
            window_size: 50,
            stride: 1,
            horizon: 1,
            drop_incomplete: true,
        }
    }
}

impl SlidingWindowConfig {
    /// Create a new SlidingWindow configuration.
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            ..Default::default()
        }
    }

    /// Set the stride.
    #[must_use]
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Set the forecast horizon.
    #[must_use]
    pub fn with_horizon(mut self, horizon: usize) -> Self {
        self.horizon = horizon;
        self
    }

    /// Set whether to drop incomplete windows.
    #[must_use]
    pub fn with_drop_incomplete(mut self, drop: bool) -> Self {
        self.drop_incomplete = drop;
        self
    }
}

/// Create sliding windows from a time series.
///
/// This is useful for:
/// - Creating training samples for forecasting
/// - Generating windows for classification
/// - Preparing data for RNNs/Transformers
///
/// # Arguments
///
/// * `data` - Input time series of shape (n_vars, seq_len)
/// * `config` - Sliding window configuration
///
/// # Returns
///
/// Tuple of (windows, targets) where:
/// - windows: Vec of (n_vars, window_size) arrays
/// - targets: Vec of (n_vars, horizon) arrays (for forecasting)
pub fn sliding_window(
    data: &ndarray::Array2<f32>,
    config: &SlidingWindowConfig,
) -> Result<(Vec<ndarray::Array2<f32>>, Vec<ndarray::Array2<f32>>)> {
    use ndarray::s;

    let (n_vars, seq_len) = (data.nrows(), data.ncols());

    if config.window_size == 0 {
        return Err(DataError::SplitError(
            "window_size must be > 0".to_string(),
        ));
    }

    let total_len = config.window_size + config.horizon;
    if seq_len < total_len {
        return Err(DataError::SplitError(format!(
            "Sequence length {} is shorter than window_size + horizon = {}",
            seq_len, total_len
        )));
    }

    let mut windows = Vec::new();
    let mut targets = Vec::new();

    let mut start = 0;
    while start + total_len <= seq_len {
        // Extract window
        let window = data.slice(s![.., start..start + config.window_size]);
        windows.push(window.to_owned());

        // Extract target (horizon)
        if config.horizon > 0 {
            let target_start = start + config.window_size;
            let target = data.slice(s![.., target_start..target_start + config.horizon]);
            targets.push(target.to_owned());
        }

        start += config.stride;
    }

    // Handle incomplete last window if not dropping
    if !config.drop_incomplete && start + config.window_size <= seq_len {
        let window = data.slice(s![.., start..start + config.window_size]);
        windows.push(window.to_owned());

        // Target might be incomplete or missing
        let remaining = seq_len - (start + config.window_size);
        if remaining > 0 {
            let target_start = start + config.window_size;
            let target_end = (target_start + config.horizon).min(seq_len);
            let mut target = ndarray::Array2::zeros((n_vars, config.horizon));
            for i in 0..n_vars {
                for j in 0..(target_end - target_start) {
                    target[[i, j]] = data[[i, target_start + j]];
                }
            }
            targets.push(target);
        }
    }

    Ok((windows, targets))
}

/// Create sliding windows for a 3D dataset (batch, vars, seq_len).
///
/// Applies sliding window to each sample in the batch.
pub fn sliding_window_batch(
    data: &ndarray::Array3<f32>,
    config: &SlidingWindowConfig,
) -> Result<(ndarray::Array3<f32>, ndarray::Array3<f32>)> {
    let (batch_size, n_vars, _seq_len) = data.dim();

    let mut all_windows = Vec::new();
    let mut all_targets = Vec::new();

    for i in 0..batch_size {
        let sample = data.slice(ndarray::s![i, .., ..]).to_owned();
        let (windows, targets) = sliding_window(&sample, config)?;
        all_windows.extend(windows);
        all_targets.extend(targets);
    }

    if all_windows.is_empty() {
        return Err(DataError::SplitError(
            "No windows created from batch".to_string(),
        ));
    }

    let n_windows = all_windows.len();

    // Stack into 3D arrays
    let windows_3d = ndarray::Array3::from_shape_fn(
        (n_windows, n_vars, config.window_size),
        |(i, j, k)| all_windows[i][[j, k]],
    );

    let targets_3d = if config.horizon > 0 {
        ndarray::Array3::from_shape_fn((n_windows, n_vars, config.horizon), |(i, j, k)| {
            all_targets[i][[j, k]]
        })
    } else {
        ndarray::Array3::zeros((n_windows, n_vars, 0))
    };

    Ok((windows_3d, targets_3d))
}

/// Configuration for TimeSplitter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSplitterConfig {
    /// Train ratio (fraction of time for training).
    pub train_ratio: f32,
    /// Validation ratio (fraction of time for validation).
    pub valid_ratio: f32,
    /// Gap between train/valid and test (for forecasting).
    pub gap: usize,
}

impl Default for TimeSplitterConfig {
    fn default() -> Self {
        Self {
            train_ratio: 0.7,
            valid_ratio: 0.1,
            gap: 0,
        }
    }
}

impl TimeSplitterConfig {
    /// Create a new TimeSplitter configuration.
    pub fn new(train_ratio: f32, valid_ratio: f32) -> Self {
        Self {
            train_ratio,
            valid_ratio,
            gap: 0,
        }
    }

    /// Set the gap between splits.
    #[must_use]
    pub fn with_gap(mut self, gap: usize) -> Self {
        self.gap = gap;
        self
    }
}

/// Time-based splitter that preserves temporal order.
///
/// Unlike random splits, this ensures that:
/// - Training data comes before validation data
/// - Validation data comes before test data
/// - No data leakage from future to past
///
/// This is essential for time series forecasting to avoid look-ahead bias.
///
/// # Arguments
///
/// * `dataset` - The dataset to split (assumed to be in temporal order)
/// * `config` - TimeSplitter configuration
///
/// # Returns
///
/// Tuple of (train_dataset, valid_dataset, test_dataset).
pub fn time_split(
    dataset: &TSDataset,
    config: &TimeSplitterConfig,
) -> Result<(TSDataset, TSDataset, TSDataset)> {
    let n = dataset.len();
    let total_ratio = config.train_ratio + config.valid_ratio;

    if total_ratio <= 0.0 || total_ratio >= 1.0 {
        return Err(DataError::SplitError(format!(
            "train_ratio + valid_ratio must be between 0 and 1, got {}",
            total_ratio
        )));
    }

    let n_train = (n as f32 * config.train_ratio).round() as usize;
    let n_valid = (n as f32 * config.valid_ratio).round() as usize;

    let n_train = n_train.max(1);
    let n_valid = if config.valid_ratio > 0.0 {
        n_valid.max(1)
    } else {
        0
    };

    // Ensure we have enough samples
    let required = n_train + n_valid + config.gap + 1;
    if required > n {
        return Err(DataError::SplitError(format!(
            "Not enough samples: {} available, but need at least {} (train={}, valid={}, gap={}, test>=1)",
            n, required, n_train, n_valid, config.gap
        )));
    }

    // Create indices preserving temporal order
    let train_end = n_train;
    let valid_start = train_end + config.gap;
    let valid_end = valid_start + n_valid;
    let test_start = valid_end + config.gap;

    let train_indices: Vec<usize> = (0..train_end).collect();
    let valid_indices: Vec<usize> = if n_valid > 0 {
        (valid_start..valid_end).collect()
    } else {
        vec![]
    };
    let test_indices: Vec<usize> = (test_start..n).collect();

    let train = dataset.subset(&train_indices)?;
    let valid = if n_valid > 0 {
        dataset.subset(&valid_indices)?
    } else {
        // Empty dataset
        dataset.subset(&[])?
    };
    let test = dataset.subset(&test_indices)?;

    Ok((train, valid, test))
}

/// Get indices for time-based split without creating datasets.
///
/// Useful when you want to split indices but manage datasets yourself.
pub fn time_split_indices(n: usize, config: &TimeSplitterConfig) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    let total_ratio = config.train_ratio + config.valid_ratio;

    if total_ratio <= 0.0 || total_ratio >= 1.0 {
        return Err(DataError::SplitError(format!(
            "train_ratio + valid_ratio must be between 0 and 1, got {}",
            total_ratio
        )));
    }

    let n_train = (n as f32 * config.train_ratio).round() as usize;
    let n_valid = (n as f32 * config.valid_ratio).round() as usize;

    let n_train = n_train.max(1);
    let n_valid = if config.valid_ratio > 0.0 {
        n_valid.max(1)
    } else {
        0
    };

    let train_end = n_train;
    let valid_start = train_end + config.gap;
    let valid_end = valid_start + n_valid;
    let test_start = valid_end + config.gap;

    let train_indices: Vec<usize> = (0..train_end).collect();
    let valid_indices: Vec<usize> = if n_valid > 0 {
        (valid_start..valid_end.min(n)).collect()
    } else {
        vec![]
    };
    let test_indices: Vec<usize> = (test_start..n).collect();

    Ok((train_indices, valid_indices, test_indices))
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

/// Configuration for Walk-forward cross-validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardConfig {
    /// Number of folds.
    pub n_folds: usize,
    /// Whether to use expanding window (true) or sliding window (false).
    pub expanding: bool,
    /// Minimum training size (as fraction or absolute count).
    pub min_train_size: WalkForwardSize,
    /// Test size for each fold (as fraction or absolute count).
    pub test_size: WalkForwardSize,
    /// Gap between train and test sets.
    pub gap: usize,
}

/// Size specification for walk-forward splits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalkForwardSize {
    /// Fraction of total data.
    Fraction(f32),
    /// Absolute number of samples.
    Absolute(usize),
}

impl Default for WalkForwardConfig {
    fn default() -> Self {
        Self {
            n_folds: 5,
            expanding: true,
            min_train_size: WalkForwardSize::Fraction(0.2),
            test_size: WalkForwardSize::Fraction(0.1),
            gap: 0,
        }
    }
}

impl WalkForwardConfig {
    /// Create a new walk-forward configuration.
    pub fn new(n_folds: usize) -> Self {
        Self {
            n_folds,
            ..Default::default()
        }
    }

    /// Set whether to use expanding or sliding window.
    #[must_use]
    pub fn with_expanding(mut self, expanding: bool) -> Self {
        self.expanding = expanding;
        self
    }

    /// Set minimum training size.
    #[must_use]
    pub fn with_min_train_size(mut self, size: WalkForwardSize) -> Self {
        self.min_train_size = size;
        self
    }

    /// Set test size.
    #[must_use]
    pub fn with_test_size(mut self, size: WalkForwardSize) -> Self {
        self.test_size = size;
        self
    }

    /// Set gap between train and test.
    #[must_use]
    pub fn with_gap(mut self, gap: usize) -> Self {
        self.gap = gap;
        self
    }
}

/// A single fold from walk-forward cross-validation.
#[derive(Debug, Clone)]
pub struct WalkForwardFold {
    /// Fold index (0-based).
    pub fold_idx: usize,
    /// Training indices.
    pub train_indices: Vec<usize>,
    /// Test indices.
    pub test_indices: Vec<usize>,
}

/// Walk-forward cross-validation for time series.
///
/// This is a time-series aware cross-validation strategy that:
/// - Never uses future data to predict the past
/// - Trains on historical data and tests on future data
/// - Supports both expanding window and sliding window modes
///
/// ## Expanding Window Mode
/// Each fold uses all data from start up to fold boundary:
/// - Fold 1: Train [0, t1), Test [t1, t2)
/// - Fold 2: Train [0, t2), Test [t2, t3)
/// - Fold 3: Train [0, t3), Test [t3, t4)
///
/// ## Sliding Window Mode
/// Each fold uses a fixed-size training window:
/// - Fold 1: Train [s1, t1), Test [t1, t2)
/// - Fold 2: Train [s2, t2), Test [t2, t3)
/// - Fold 3: Train [s3, t3), Test [t3, t4)
///
/// # Arguments
///
/// * `n` - Total number of samples
/// * `config` - Walk-forward configuration
///
/// # Returns
///
/// Vector of WalkForwardFold structs containing train/test indices for each fold.
pub fn walk_forward_split(n: usize, config: &WalkForwardConfig) -> Result<Vec<WalkForwardFold>> {
    if config.n_folds < 2 {
        return Err(DataError::SplitError(
            "n_folds must be at least 2".to_string(),
        ));
    }

    // Calculate sizes
    let min_train = match config.min_train_size {
        WalkForwardSize::Fraction(f) => ((n as f32) * f).round() as usize,
        WalkForwardSize::Absolute(s) => s,
    };

    let test_size = match config.test_size {
        WalkForwardSize::Fraction(f) => ((n as f32) * f).round() as usize,
        WalkForwardSize::Absolute(s) => s,
    };

    let min_train = min_train.max(1);
    let test_size = test_size.max(1);

    // Calculate total required space
    let required = min_train + config.gap + (config.n_folds * test_size);
    if required > n {
        return Err(DataError::SplitError(format!(
            "Not enough samples: {} available, but need at least {} for {} folds",
            n, required, config.n_folds
        )));
    }

    // Calculate available space for folds after minimum training
    let available = n - min_train - config.gap;
    let fold_step = available / config.n_folds;

    if fold_step < test_size {
        return Err(DataError::SplitError(format!(
            "Fold step {} is smaller than test_size {}",
            fold_step, test_size
        )));
    }

    let mut folds = Vec::with_capacity(config.n_folds);

    for fold_idx in 0..config.n_folds {
        // Test window position
        let test_start = min_train + config.gap + (fold_idx * fold_step);
        let test_end = (test_start + test_size).min(n);

        // Training window
        let train_end = test_start - config.gap;
        let train_start = if config.expanding {
            0
        } else {
            // Sliding window: maintain min_train size
            if train_end > min_train {
                train_end - min_train
            } else {
                0
            }
        };

        let train_indices: Vec<usize> = (train_start..train_end).collect();
        let test_indices: Vec<usize> = (test_start..test_end).collect();

        folds.push(WalkForwardFold {
            fold_idx,
            train_indices,
            test_indices,
        });
    }

    Ok(folds)
}

/// Create walk-forward splits from a dataset.
///
/// Convenience function that creates dataset subsets for each fold.
pub fn walk_forward_split_dataset(
    dataset: &TSDataset,
    config: &WalkForwardConfig,
) -> Result<Vec<(TSDataset, TSDataset)>> {
    let n = dataset.len();
    let folds = walk_forward_split(n, config)?;

    let mut result = Vec::with_capacity(folds.len());
    for fold in folds {
        let train = dataset.subset(&fold.train_indices)?;
        let test = dataset.subset(&fold.test_indices)?;
        result.push((train, test));
    }

    Ok(result)
}

/// Iterator over walk-forward folds.
pub struct WalkForwardIterator {
    folds: Vec<WalkForwardFold>,
    current: usize,
}

impl WalkForwardIterator {
    /// Create a new walk-forward iterator.
    pub fn new(n: usize, config: &WalkForwardConfig) -> Result<Self> {
        let folds = walk_forward_split(n, config)?;
        Ok(Self { folds, current: 0 })
    }
}

impl Iterator for WalkForwardIterator {
    type Item = WalkForwardFold;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.folds.len() {
            let fold = self.folds[self.current].clone();
            self.current += 1;
            Some(fold)
        } else {
            None
        }
    }
}

impl ExactSizeIterator for WalkForwardIterator {
    fn len(&self) -> usize {
        self.folds.len() - self.current
    }
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
