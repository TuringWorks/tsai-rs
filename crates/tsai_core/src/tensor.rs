//! Time series tensor types.

use burn::prelude::*;

use crate::error::{CoreError, Result};
use crate::shape::TSShape;

/// A time series tensor wrapper with shape metadata.
///
/// Wraps a Burn tensor and ensures the shape follows the `(B, V, L)` convention:
/// - `B`: Batch size
/// - `V`: Variables/channels
/// - `L`: Sequence length
///
/// # Type Parameters
///
/// * `B` - The Burn backend type
///
/// # Example
///
/// ```rust,ignore
/// use tsai_core::TSTensor;
/// use burn::backend::NdArray;
///
/// // Create from raw tensor
/// let tensor = Tensor::<NdArray, 3>::zeros([32, 3, 100], &device);
/// let ts_tensor = TSTensor::new(tensor)?;
/// ```
#[derive(Debug, Clone)]
pub struct TSTensor<B: Backend> {
    inner: Tensor<B, 3>,
    shape: TSShape,
}

impl<B: Backend> TSTensor<B> {
    /// Create a new TSTensor from a Burn tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - A 3D Burn tensor with shape `(batch, vars, len)`
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is not 3-dimensional.
    pub fn new(tensor: Tensor<B, 3>) -> Result<Self> {
        let dims = tensor.dims();
        let shape = TSShape::new(dims[0], dims[1], dims[2]);
        Ok(Self {
            inner: tensor,
            shape,
        })
    }

    /// Create a TSTensor filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `shape` - The desired shape
    /// * `device` - The device to create the tensor on
    pub fn zeros(shape: TSShape, device: &B::Device) -> Self {
        let dims = shape.as_array();
        let tensor = Tensor::zeros(dims, device);
        Self {
            inner: tensor,
            shape,
        }
    }

    /// Create a TSTensor filled with ones.
    ///
    /// # Arguments
    ///
    /// * `shape` - The desired shape
    /// * `device` - The device to create the tensor on
    pub fn ones(shape: TSShape, device: &B::Device) -> Self {
        let dims = shape.as_array();
        let tensor = Tensor::ones(dims, device);
        Self {
            inner: tensor,
            shape,
        }
    }

    /// Get the shape metadata.
    #[must_use]
    pub const fn shape(&self) -> TSShape {
        self.shape
    }

    /// Get the batch size.
    #[must_use]
    pub const fn batch(&self) -> usize {
        self.shape.batch()
    }

    /// Get the number of variables.
    #[must_use]
    pub const fn vars(&self) -> usize {
        self.shape.vars()
    }

    /// Get the sequence length.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.shape.len()
    }

    /// Check if the tensor is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.shape.is_empty()
    }

    /// Get a reference to the underlying Burn tensor.
    #[must_use]
    pub const fn inner(&self) -> &Tensor<B, 3> {
        &self.inner
    }

    /// Consume self and return the underlying Burn tensor.
    #[must_use]
    pub fn into_inner(self) -> Tensor<B, 3> {
        self.inner
    }

    /// Get the device the tensor is on.
    pub fn device(&self) -> B::Device {
        self.inner.device()
    }

    /// Clone the tensor to a new device.
    pub fn to_device(&self, device: &B::Device) -> Self {
        Self {
            inner: self.inner.clone().to_device(device),
            shape: self.shape,
        }
    }

    /// Transpose the tensor (swap vars and len dimensions).
    ///
    /// Changes shape from `(B, V, L)` to `(B, L, V)`.
    pub fn transpose(&self) -> Self {
        let transposed = self.inner.clone().swap_dims(1, 2);
        Self {
            inner: transposed,
            shape: TSShape::new(self.shape.batch(), self.shape.len(), self.shape.vars()),
        }
    }
}

/// A mask tensor for variable-length sequences.
///
/// Values of 1.0 indicate valid positions, 0.0 indicate padding.
#[derive(Debug, Clone)]
pub struct TSMaskTensor<B: Backend> {
    inner: Tensor<B, 3>,
    shape: TSShape,
}

impl<B: Backend> TSMaskTensor<B> {
    /// Create a new mask tensor.
    pub fn new(tensor: Tensor<B, 3>) -> Result<Self> {
        let dims = tensor.dims();
        let shape = TSShape::new(dims[0], dims[1], dims[2]);
        Ok(Self {
            inner: tensor,
            shape,
        })
    }

    /// Create an all-ones mask (no padding).
    pub fn all_valid(shape: TSShape, device: &B::Device) -> Self {
        let dims = shape.as_array();
        let tensor = Tensor::ones(dims, device);
        Self {
            inner: tensor,
            shape,
        }
    }

    /// Get the underlying tensor.
    #[must_use]
    pub const fn inner(&self) -> &Tensor<B, 3> {
        &self.inner
    }

    /// Get the shape.
    #[must_use]
    pub const fn shape(&self) -> TSShape {
        self.shape
    }
}

/// A batch of time series data with optional labels and masks.
///
/// This is the primary data structure passed through dataloaders and transforms.
#[derive(Debug, Clone)]
pub struct TSBatch<B: Backend> {
    /// Input time series tensor (B, V, L).
    pub x: TSTensor<B>,

    /// Optional target tensor.
    pub y: Option<Tensor<B, 2>>,

    /// Optional attention mask for variable-length sequences.
    pub mask: Option<TSMaskTensor<B>>,
}

impl<B: Backend> TSBatch<B> {
    /// Create a new batch with just input data.
    pub fn new(x: TSTensor<B>) -> Self {
        Self {
            x,
            y: None,
            mask: None,
        }
    }

    /// Create a batch with input and target.
    pub fn with_target(x: TSTensor<B>, y: Tensor<B, 2>) -> Result<Self> {
        let x_batch = x.batch();
        let y_batch = y.dims()[0];

        if x_batch != y_batch {
            return Err(CoreError::ShapeMismatch(format!(
                "x batch size {} != y batch size {}",
                x_batch, y_batch
            )));
        }

        Ok(Self {
            x,
            y: Some(y),
            mask: None,
        })
    }

    /// Add a mask to the batch.
    pub fn with_mask(mut self, mask: TSMaskTensor<B>) -> Result<Self> {
        if self.x.shape() != mask.shape() {
            return Err(CoreError::ShapeMismatch(format!(
                "x shape {:?} != mask shape {:?}",
                self.x.shape(),
                mask.shape()
            )));
        }
        self.mask = Some(mask);
        Ok(self)
    }

    /// Get the batch size.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.x.batch()
    }

    /// Get the device.
    pub fn device(&self) -> B::Device {
        self.x.device()
    }

    /// Move the batch to a device.
    pub fn to_device(self, device: &B::Device) -> Self {
        Self {
            x: self.x.to_device(device),
            y: self.y.map(|y| y.to_device(device)),
            mask: self.mask.map(|m| TSMaskTensor {
                inner: m.inner.to_device(device),
                shape: m.shape,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Tests require a backend to be enabled
    // These are placeholder tests that will work with any backend

    #[test]
    fn test_ts_shape_basic() {
        let shape = TSShape::new(32, 3, 100);
        assert_eq!(shape.batch(), 32);
        assert_eq!(shape.vars(), 3);
        assert_eq!(shape.len(), 100);
    }
}
