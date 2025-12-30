//! Model traits for training.
//!
//! Defines traits that models must implement to work with the training system.

use burn::module::AutodiffModule;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

/// Trait for time series classification models.
///
/// This trait must be implemented by models that can be trained
/// for classification tasks.
pub trait TSClassificationModel<B: AutodiffBackend>: AutodiffModule<B> + Clone + Send {
    /// Forward pass returning logits.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape (batch, vars, seq_len)
    ///
    /// # Returns
    ///
    /// Logits tensor of shape (batch, n_classes)
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2>;

    /// Forward pass returning probabilities.
    fn forward_probs(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let logits = self.forward(x);
        burn::tensor::activation::softmax(logits, 1)
    }
}

/// Trait for time series regression models.
pub trait TSRegressionModel<B: AutodiffBackend>: AutodiffModule<B> + Clone + Send {
    /// Forward pass returning predictions.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape (batch, vars, seq_len)
    ///
    /// # Returns
    ///
    /// Predictions tensor of shape (batch, n_outputs)
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2>;
}

/// Trait for time series forecasting models.
pub trait TSForecastingModel<B: AutodiffBackend>: AutodiffModule<B> + Clone + Send {
    /// Forward pass returning forecasts.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape (batch, vars, seq_len)
    ///
    /// # Returns
    ///
    /// Forecasts tensor of shape (batch, horizon) or (batch, vars, horizon)
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3>;
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_trait_definitions() {
        // Traits are defined, implementation tests would go in model crate
    }
}
