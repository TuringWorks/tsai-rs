//! Model registry for dynamic model creation.
//!
//! The registry allows creating models dynamically by name from JSON configuration.
//!
//! # Example
//!
//! ```rust,ignore
//! use tsai_models::registry::default_registry;
//! use serde_json::json;
//!
//! let registry = default_registry::<NdArray>(&device);
//! let config = json!({
//!     "n_vars": 3,
//!     "seq_len": 100,
//!     "n_classes": 5
//! });
//! let model = registry.create("InceptionTimePlus", &config, &device)?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use burn::prelude::*;
use burn::tensor::backend::Backend;
use serde_json::Value;
use thiserror::Error;

use crate::{
    HydraPlus, HydraPlusConfig, InceptionTimePlus, InceptionTimePlusConfig, OmniScaleCNN,
    OmniScaleCNNConfig, RNNPlus, RNNPlusConfig, RNNType, TSPerceiver, TSPerceiverConfig,
    TSTConfig, TSTPlus, XCMPlus, XCMPlusConfig,
};

/// Error type for model registry operations.
#[derive(Error, Debug)]
pub enum RegistryError {
    /// Model not found in registry.
    #[error("Model '{0}' not found in registry")]
    ModelNotFound(String),

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Model creation failed.
    #[error("Failed to create model: {0}")]
    CreationFailed(String),
}

/// Result type for registry operations.
pub type Result<T> = std::result::Result<T, RegistryError>;

/// Trait for models that can be created from config.
///
/// Note: We don't require Send + Sync here because Burn's Module types
/// use interior mutability (OnceCell) that doesn't implement Sync.
pub trait TSModel<B: Backend> {
    /// Forward pass returning class logits.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2>;

    /// Get the model name.
    fn name(&self) -> &str;
}

/// Type alias for model constructor.
pub type ModelConstructor<B> =
    Arc<dyn Fn(&Value, &<B as Backend>::Device) -> Result<Box<dyn TSModel<B>>> + Send + Sync>;

/// Registry for dynamically creating models by name.
pub struct ModelRegistry<B: Backend> {
    models: HashMap<String, ModelConstructor<B>>,
}

impl<B: Backend> Default for ModelRegistry<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> ModelRegistry<B> {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    /// Register a model constructor.
    ///
    /// # Arguments
    ///
    /// * `name` - The name to register the model under
    /// * `constructor` - A function that creates the model from config
    pub fn register<F>(&mut self, name: &str, constructor: F)
    where
        F: Fn(&Value, &<B as Backend>::Device) -> Result<Box<dyn TSModel<B>>> + Send + Sync + 'static,
    {
        self.models.insert(name.to_string(), Arc::new(constructor));
    }

    /// Create a model by name.
    ///
    /// # Arguments
    ///
    /// * `name` - The registered name of the model
    /// * `config` - JSON configuration for the model
    /// * `device` - The device to create the model on
    pub fn create(
        &self,
        name: &str,
        config: &Value,
        device: &<B as Backend>::Device,
    ) -> Result<Box<dyn TSModel<B>>> {
        let constructor = self
            .models
            .get(name)
            .ok_or_else(|| RegistryError::ModelNotFound(name.to_string()))?;
        constructor(config, device)
    }

    /// List all registered model names.
    pub fn list(&self) -> Vec<&str> {
        self.models.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a model is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.models.contains_key(name)
    }
}

// ============================================================================
// TSModel implementations for each model type
// ============================================================================

impl<B: Backend> TSModel<B> for InceptionTimePlus<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        self.forward(x)
    }

    fn name(&self) -> &str {
        "InceptionTimePlus"
    }
}

impl<B: Backend> TSModel<B> for OmniScaleCNN<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        self.forward(x)
    }

    fn name(&self) -> &str {
        "OmniScaleCNN"
    }
}

impl<B: Backend> TSModel<B> for XCMPlus<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        self.forward(x)
    }

    fn name(&self) -> &str {
        "XCMPlus"
    }
}

impl<B: Backend> TSModel<B> for TSTPlus<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        self.forward(x)
    }

    fn name(&self) -> &str {
        "TSTPlus"
    }
}

impl<B: Backend> TSModel<B> for TSPerceiver<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        self.forward(x)
    }

    fn name(&self) -> &str {
        "TSPerceiver"
    }
}

impl<B: Backend> TSModel<B> for HydraPlus<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        self.forward(x)
    }

    fn name(&self) -> &str {
        "HydraPlus"
    }
}

impl<B: Backend> TSModel<B> for RNNPlus<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        self.forward(x)
    }

    fn name(&self) -> &str {
        "RNNPlus"
    }
}

// ============================================================================
// Helper functions for parsing configs
// ============================================================================

fn get_usize(config: &Value, key: &str) -> Result<usize> {
    config
        .get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .ok_or_else(|| RegistryError::InvalidConfig(format!("Missing or invalid '{}'", key)))
}

fn get_usize_or(config: &Value, key: &str, default: usize) -> usize {
    config
        .get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(default)
}

fn get_f64_or(config: &Value, key: &str, default: f64) -> f64 {
    config
        .get(key)
        .and_then(|v| v.as_f64())
        .unwrap_or(default)
}

// ============================================================================
// Default registry with all models
// ============================================================================

/// Create a registry with all available models pre-registered.
///
/// # Available Models
///
/// - `InceptionTimePlus` - InceptionTime with improvements
/// - `OmniScaleCNN` - Multi-scale CNN
/// - `XCMPlus` - Explainable CNN
/// - `TSTPlus` - Time Series Transformer
/// - `TSPerceiver` - Perceiver for time series
/// - `HydraPlus` - Hybrid ROCKET
/// - `RNNPlus` - LSTM/GRU with improvements
///
/// # Required Config Fields
///
/// All models require:
/// - `n_vars`: Number of input variables/channels
/// - `seq_len`: Sequence length
/// - `n_classes`: Number of output classes
///
/// # Example
///
/// ```rust,ignore
/// use tsai_models::registry::default_registry;
/// use serde_json::json;
///
/// let registry = default_registry::<NdArray>();
/// let config = json!({
///     "n_vars": 3,
///     "seq_len": 100,
///     "n_classes": 5,
///     "n_blocks": 6,
///     "n_filters": 32
/// });
/// let model = registry.create("InceptionTimePlus", &config, &device)?;
/// ```
pub fn default_registry<B: Backend>() -> ModelRegistry<B> {
    let mut registry = ModelRegistry::new();

    // InceptionTimePlus
    registry.register("InceptionTimePlus", |config, device| {
        let n_vars = get_usize(config, "n_vars")?;
        let seq_len = get_usize(config, "seq_len")?;
        let n_classes = get_usize(config, "n_classes")?;

        let model_config = InceptionTimePlusConfig {
            n_vars,
            seq_len,
            n_classes,
            n_blocks: get_usize_or(config, "n_blocks", 6),
            n_filters: get_usize_or(config, "n_filters", 32),
            kernel_sizes: [9, 19, 39],
            bottleneck_dim: get_usize_or(config, "bottleneck_dim", 32),
            dropout: get_f64_or(config, "dropout", 0.0),
        };

        Ok(Box::new(model_config.init::<B>(device)) as Box<dyn TSModel<B>>)
    });

    // OmniScaleCNN
    registry.register("OmniScaleCNN", |config, device| {
        let n_vars = get_usize(config, "n_vars")?;
        let seq_len = get_usize(config, "seq_len")?;
        let n_classes = get_usize(config, "n_classes")?;

        let model_config = OmniScaleCNNConfig::new(n_vars, seq_len, n_classes)
            .with_n_filters(get_usize_or(config, "n_filters", 64))
            .with_dropout(get_f64_or(config, "dropout", 0.1));

        Ok(Box::new(model_config.init::<B>(device)) as Box<dyn TSModel<B>>)
    });

    // XCMPlus
    registry.register("XCMPlus", |config, device| {
        let n_vars = get_usize(config, "n_vars")?;
        let seq_len = get_usize(config, "seq_len")?;
        let n_classes = get_usize(config, "n_classes")?;

        let model_config = XCMPlusConfig {
            n_vars,
            seq_len,
            n_classes,
            n_filters: get_usize_or(config, "n_filters", 128),
            window_sizes: vec![10, 20, 40],
        };

        Ok(Box::new(model_config.init::<B>(device)) as Box<dyn TSModel<B>>)
    });

    // TSTPlus
    registry.register("TSTPlus", |config, device| {
        let n_vars = get_usize(config, "n_vars")?;
        let seq_len = get_usize(config, "seq_len")?;
        let n_classes = get_usize(config, "n_classes")?;
        let d_model = get_usize_or(config, "d_model", 64);

        let model_config = TSTConfig {
            n_vars,
            seq_len,
            n_classes,
            d_model,
            n_heads: get_usize_or(config, "n_heads", 4),
            n_layers: get_usize_or(config, "n_layers", 3),
            d_ff: get_usize_or(config, "d_ff", d_model * 4),
            dropout: get_f64_or(config, "dropout", 0.1),
            use_pe: true,
        };

        Ok(Box::new(model_config.init::<B>(device)) as Box<dyn TSModel<B>>)
    });

    // TSPerceiver
    registry.register("TSPerceiver", |config, device| {
        let n_vars = get_usize(config, "n_vars")?;
        let seq_len = get_usize(config, "seq_len")?;
        let n_classes = get_usize(config, "n_classes")?;

        let model_config = TSPerceiverConfig::new(n_vars, seq_len, n_classes)
            .with_d_latent(get_usize_or(config, "d_latent", 256))
            .with_n_latents(get_usize_or(config, "n_latents", 64))
            .with_n_cross_layers(get_usize_or(config, "n_cross_layers", 2))
            .with_n_self_layers(get_usize_or(config, "n_self_layers", 6))
            .with_dropout(get_f64_or(config, "dropout", 0.1));

        Ok(Box::new(model_config.init::<B>(device)) as Box<dyn TSModel<B>>)
    });

    // HydraPlus
    registry.register("HydraPlus", |config, device| {
        let n_vars = get_usize(config, "n_vars")?;
        let seq_len = get_usize(config, "seq_len")?;
        let n_classes = get_usize(config, "n_classes")?;

        let model_config = HydraPlusConfig::new(n_vars, seq_len, n_classes)
            .with_n_groups(get_usize_or(config, "n_groups", 8))
            .with_kernels_per_group(get_usize_or(config, "kernels_per_group", 8))
            .with_kernel_length(get_usize_or(config, "kernel_length", 9))
            .with_hidden_dim(get_usize_or(config, "hidden_dim", 128))
            .with_dropout(get_f64_or(config, "dropout", 0.1));

        Ok(Box::new(model_config.init::<B>(device)) as Box<dyn TSModel<B>>)
    });

    // RNNPlus
    registry.register("RNNPlus", |config, device| {
        let n_vars = get_usize(config, "n_vars")?;
        let seq_len = get_usize(config, "seq_len")?;
        let n_classes = get_usize(config, "n_classes")?;

        let model_config = RNNPlusConfig {
            n_vars,
            seq_len,
            n_classes,
            hidden_size: get_usize_or(config, "hidden_size", 128),
            n_layers: get_usize_or(config, "n_layers", 2),
            rnn_type: RNNType::LSTM,
            bidirectional: true,
            dropout: get_f64_or(config, "dropout", 0.1),
        };

        Ok(Box::new(model_config.init::<B>(device)) as Box<dyn TSModel<B>>)
    });

    registry
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use serde_json::json;

    type TestBackend = NdArray;

    #[test]
    fn test_registry_new() {
        let registry: ModelRegistry<TestBackend> = ModelRegistry::new();
        assert!(registry.list().is_empty());
    }

    #[test]
    fn test_default_registry_contains_all_models() {
        let registry: ModelRegistry<TestBackend> = default_registry();
        let models = registry.list();

        assert!(registry.contains("InceptionTimePlus"));
        assert!(registry.contains("OmniScaleCNN"));
        assert!(registry.contains("XCMPlus"));
        assert!(registry.contains("TSTPlus"));
        assert!(registry.contains("TSPerceiver"));
        assert!(registry.contains("HydraPlus"));
        assert!(registry.contains("RNNPlus"));
        assert_eq!(models.len(), 7);
    }

    #[test]
    fn test_create_inception_time_plus() {
        let registry: ModelRegistry<TestBackend> = default_registry();
        let device = Default::default();
        let config = json!({
            "n_vars": 3,
            "seq_len": 50,
            "n_classes": 5,
            "n_blocks": 2,
            "n_filters": 16
        });

        let model = registry.create("InceptionTimePlus", &config, &device);
        assert!(model.is_ok());
        assert_eq!(model.unwrap().name(), "InceptionTimePlus");
    }

    #[test]
    fn test_create_omniscale_cnn() {
        let registry: ModelRegistry<TestBackend> = default_registry();
        let device = Default::default();
        // Use seq_len that generates odd kernel sizes (powers of 2 up to seq_len/2)
        let config = json!({
            "n_vars": 2,
            "seq_len": 100,
            "n_classes": 3
        });

        let model = registry.create("OmniScaleCNN", &config, &device);
        assert!(model.is_ok());
        assert_eq!(model.unwrap().name(), "OmniScaleCNN");
    }

    #[test]
    fn test_create_tst_plus() {
        let registry: ModelRegistry<TestBackend> = default_registry();
        let device = Default::default();
        let config = json!({
            "n_vars": 3,
            "seq_len": 50,
            "n_classes": 5,
            "d_model": 32,
            "n_heads": 2,
            "n_layers": 2
        });

        let model = registry.create("TSTPlus", &config, &device);
        assert!(model.is_ok());
        assert_eq!(model.unwrap().name(), "TSTPlus");
    }

    #[test]
    fn test_create_hydra_plus() {
        let registry: ModelRegistry<TestBackend> = default_registry();
        let device = Default::default();
        let config = json!({
            "n_vars": 2,
            "seq_len": 100,
            "n_classes": 3,
            "n_groups": 4,
            "kernels_per_group": 4
        });

        let model = registry.create("HydraPlus", &config, &device);
        assert!(model.is_ok());
        assert_eq!(model.unwrap().name(), "HydraPlus");
    }

    #[test]
    fn test_model_not_found() {
        let registry: ModelRegistry<TestBackend> = default_registry();
        let device = Default::default();
        let config = json!({});

        let result = registry.create("NonExistentModel", &config, &device);
        assert!(result.is_err());
        match result {
            Err(RegistryError::ModelNotFound(name)) => assert_eq!(name, "NonExistentModel"),
            _ => panic!("Expected ModelNotFound error"),
        }
    }

    #[test]
    fn test_invalid_config() {
        let registry: ModelRegistry<TestBackend> = default_registry();
        let device = Default::default();
        let config = json!({
            "n_vars": 3
            // Missing n_classes and seq_len
        });

        let result = registry.create("InceptionTimePlus", &config, &device);
        assert!(result.is_err());
        match result {
            Err(RegistryError::InvalidConfig(_)) => (),
            _ => panic!("Expected InvalidConfig error"),
        }
    }
}
