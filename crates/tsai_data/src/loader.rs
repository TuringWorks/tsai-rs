//! Dataloader implementations for batched iteration.

use burn::prelude::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use crate::dataset::TSDataset;
use crate::error::{DataError, Result};
use tsai_core::{Seed, Split, TSBatch, TSTensor};

/// A dataloader that produces batches from a dataset.
///
/// Supports shuffling, deterministic sampling with seeds, and
/// variable-length sequences with padding.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_data::{TSDataset, TSDataLoader};
/// use tsai_core::Seed;
///
/// let dataset = TSDataset::from_arrays(x, Some(y))?;
/// let loader = TSDataLoader::builder(dataset)
///     .batch_size(32)
///     .shuffle(true)
///     .seed(Seed::new(42))
///     .build()?;
///
/// for batch in loader.iter() {
///     // process batch
/// }
/// ```
pub struct TSDataLoader {
    dataset: TSDataset,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    seed: Option<Seed>,
    split: Split,
}

impl TSDataLoader {
    /// Create a new dataloader builder.
    #[must_use]
    pub fn builder(dataset: TSDataset) -> TSDataLoaderBuilder {
        TSDataLoaderBuilder::new(dataset)
    }

    /// Get the dataset.
    #[must_use]
    pub fn dataset(&self) -> &TSDataset {
        &self.dataset
    }

    /// Get the batch size.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get the number of batches.
    #[must_use]
    pub fn n_batches(&self) -> usize {
        let n = self.dataset.len();
        if self.drop_last {
            n / self.batch_size
        } else {
            (n + self.batch_size - 1) / self.batch_size
        }
    }

    /// Get the total number of samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    /// Check if the loader is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }

    /// Create an iterator over batches.
    ///
    /// # Type Parameters
    ///
    /// * `B` - The Burn backend to use for tensors
    #[must_use]
    pub fn iter<B: Backend>(&self, device: &B::Device) -> TSDataLoaderIter<'_, B> {
        TSDataLoaderIter::new(self, device.clone())
    }

    /// Get the data split type.
    #[must_use]
    pub fn split(&self) -> Split {
        self.split
    }
}

/// Builder for TSDataLoader.
pub struct TSDataLoaderBuilder {
    dataset: TSDataset,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    seed: Option<Seed>,
    split: Split,
}

impl TSDataLoaderBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new(dataset: TSDataset) -> Self {
        Self {
            dataset,
            batch_size: 32,
            shuffle: false,
            drop_last: false,
            seed: None,
            split: Split::Train,
        }
    }

    /// Set the batch size.
    #[must_use]
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Enable or disable shuffling.
    #[must_use]
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Enable or disable dropping the last incomplete batch.
    #[must_use]
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Set the random seed for shuffling.
    #[must_use]
    pub fn seed(mut self, seed: Seed) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the data split type.
    #[must_use]
    pub fn split(mut self, split: Split) -> Self {
        self.split = split;
        self
    }

    /// Build the dataloader.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn build(self) -> Result<TSDataLoader> {
        if self.batch_size == 0 {
            return Err(DataError::InvalidBatchSize(
                "Batch size must be greater than 0".to_string(),
            ));
        }

        if self.dataset.is_empty() {
            return Err(DataError::EmptyDataset);
        }

        Ok(TSDataLoader {
            dataset: self.dataset,
            batch_size: self.batch_size,
            shuffle: self.shuffle,
            drop_last: self.drop_last,
            seed: self.seed,
            split: self.split,
        })
    }
}

/// Iterator over batches from a TSDataLoader.
pub struct TSDataLoaderIter<'a, B: Backend> {
    loader: &'a TSDataLoader,
    device: B::Device,
    indices: Vec<usize>,
    current_batch: usize,
    n_batches: usize,
}

impl<'a, B: Backend> TSDataLoaderIter<'a, B> {
    fn new(loader: &'a TSDataLoader, device: B::Device) -> Self {
        let n = loader.dataset.len();
        let mut indices: Vec<usize> = (0..n).collect();

        if loader.shuffle {
            let mut rng = if let Some(seed) = loader.seed {
                ChaCha8Rng::seed_from_u64(seed.value())
            } else {
                ChaCha8Rng::from_entropy()
            };
            indices.shuffle(&mut rng);
        }

        Self {
            loader,
            device,
            indices,
            current_batch: 0,
            n_batches: loader.n_batches(),
        }
    }
}

impl<'a, B: Backend> Iterator for TSDataLoaderIter<'a, B> {
    type Item = Result<TSBatch<B>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_batch >= self.n_batches {
            return None;
        }

        let start = self.current_batch * self.loader.batch_size;
        let end = std::cmp::min(start + self.loader.batch_size, self.indices.len());
        let batch_indices: Vec<usize> = self.indices[start..end].to_vec();

        self.current_batch += 1;

        // Get the batch data
        let batch_result = self.create_batch(&batch_indices);
        Some(batch_result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.n_batches - self.current_batch;
        (remaining, Some(remaining))
    }
}

impl<'a, B: Backend> TSDataLoaderIter<'a, B> {
    fn create_batch(&self, indices: &[usize]) -> Result<TSBatch<B>> {
        let dataset = &self.loader.dataset;
        let batch_size = indices.len();
        let n_vars = dataset.n_vars();
        let seq_len = dataset.seq_len();

        // Collect data into ndarray first
        let mut x_data = ndarray::Array3::<f32>::zeros((batch_size, n_vars, seq_len));
        let mut y_data = dataset
            .y()
            .map(|y| ndarray::Array2::<f32>::zeros((batch_size, y.shape()[1])));

        for (i, &idx) in indices.iter().enumerate() {
            let (x_sample, y_sample) = dataset.get(idx)?;
            x_data
                .index_axis_mut(ndarray::Axis(0), i)
                .assign(&x_sample);

            if let (Some(ref mut y), Some(ys)) = (&mut y_data, y_sample) {
                y.index_axis_mut(ndarray::Axis(0), i).assign(&ys);
            }
        }

        // Convert to Burn tensors
        let x_flat: Vec<f32> = x_data.iter().copied().collect();
        let x_tensor: Tensor<B, 3> =
            Tensor::<B, 1>::from_floats(x_flat.as_slice(), &self.device).reshape([batch_size, n_vars, seq_len]);

        let ts_tensor = TSTensor::new(x_tensor)?;

        let y_tensor = y_data.map(|y| {
            let y_flat: Vec<f32> = y.iter().copied().collect();
            let y_cols = y.shape()[1];
            Tensor::<B, 1>::from_floats(y_flat.as_slice(), &self.device).reshape([batch_size, y_cols])
        });

        match y_tensor {
            Some(y) => Ok(TSBatch::with_target(ts_tensor, y)?),
            None => Ok(TSBatch::new(ts_tensor)),
        }
    }
}

impl<'a, B: Backend> ExactSizeIterator for TSDataLoaderIter<'a, B> {}

/// Paired dataloaders for training and validation.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_data::{TSDataset, TSDataLoaders};
/// use tsai_core::Seed;
///
/// let dls = TSDataLoaders::new(train_ds, valid_ds)
///     .batch_size(32)
///     .seed(Seed::new(42))
///     .build()?;
///
/// for batch in dls.train().iter() {
///     // training
/// }
/// ```
pub struct TSDataLoaders {
    train: TSDataLoader,
    valid: TSDataLoader,
}

impl TSDataLoaders {
    /// Create a new builder.
    #[must_use]
    pub fn builder(train: TSDataset, valid: TSDataset) -> TSDataLoadersBuilder {
        TSDataLoadersBuilder::new(train, valid)
    }

    /// Get the training dataloader.
    #[must_use]
    pub fn train(&self) -> &TSDataLoader {
        &self.train
    }

    /// Get the validation dataloader.
    #[must_use]
    pub fn valid(&self) -> &TSDataLoader {
        &self.valid
    }

    /// Get the batch size.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.train.batch_size()
    }

    /// Get the number of variables.
    #[must_use]
    pub fn n_vars(&self) -> usize {
        self.train.dataset().n_vars()
    }

    /// Get the sequence length.
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.train.dataset().seq_len()
    }
}

/// Builder for TSDataLoaders.
pub struct TSDataLoadersBuilder {
    train: TSDataset,
    valid: TSDataset,
    batch_size: usize,
    shuffle_train: bool,
    seed: Option<Seed>,
}

impl TSDataLoadersBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new(train: TSDataset, valid: TSDataset) -> Self {
        Self {
            train,
            valid,
            batch_size: 32,
            shuffle_train: true,
            seed: None,
        }
    }

    /// Set the batch size for both loaders.
    #[must_use]
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Enable or disable shuffling for the training loader.
    #[must_use]
    pub fn shuffle_train(mut self, shuffle: bool) -> Self {
        self.shuffle_train = shuffle;
        self
    }

    /// Set the random seed.
    #[must_use]
    pub fn seed(mut self, seed: Seed) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Build the dataloaders.
    pub fn build(self) -> Result<TSDataLoaders> {
        let mut train_builder = TSDataLoader::builder(self.train)
            .batch_size(self.batch_size)
            .shuffle(self.shuffle_train)
            .split(Split::Train);

        if let Some(seed) = self.seed {
            train_builder = train_builder.seed(seed.derive("train"));
        }

        let valid_builder = TSDataLoader::builder(self.valid)
            .batch_size(self.batch_size)
            .shuffle(false)
            .split(Split::Valid);

        Ok(TSDataLoaders {
            train: train_builder.build()?,
            valid: valid_builder.build()?,
        })
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
    fn test_loader_builder() {
        let ds = create_test_dataset(100);
        let loader = TSDataLoader::builder(ds)
            .batch_size(32)
            .shuffle(true)
            .build()
            .unwrap();

        assert_eq!(loader.batch_size(), 32);
        assert_eq!(loader.n_batches(), 4); // ceil(100/32) = 4
    }

    #[test]
    fn test_loader_n_batches() {
        let ds = create_test_dataset(100);

        // Without drop_last
        let loader = TSDataLoader::builder(ds.clone())
            .batch_size(32)
            .drop_last(false)
            .build()
            .unwrap();
        assert_eq!(loader.n_batches(), 4);

        // With drop_last
        let loader = TSDataLoader::builder(ds)
            .batch_size(32)
            .drop_last(true)
            .build()
            .unwrap();
        assert_eq!(loader.n_batches(), 3);
    }

    #[test]
    fn test_loaders_builder() {
        let train = create_test_dataset(100);
        let valid = create_test_dataset(20);

        let dls = TSDataLoaders::builder(train, valid)
            .batch_size(32)
            .seed(Seed::new(42))
            .build()
            .unwrap();

        assert_eq!(dls.batch_size(), 32);
        assert_eq!(dls.train().len(), 100);
        assert_eq!(dls.valid().len(), 20);
    }
}
