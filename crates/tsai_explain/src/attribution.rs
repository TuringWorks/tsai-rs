//! Attribution map computation.

use burn::prelude::*;
use serde::{Deserialize, Serialize};

/// Method for computing attribution maps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttributionMethod {
    /// Gradient-weighted Class Activation Mapping.
    GradCAM,
    /// Input × Gradient.
    InputGradient,
    /// Integrated Gradients.
    IntegratedGradients,
    /// Attention weights (for transformers).
    Attention,
}

/// Attribution map over time series.
#[derive(Debug, Clone)]
pub struct AttributionMap<B: Backend> {
    /// The attribution values.
    pub values: Tensor<B, 3>,
    /// The method used.
    pub method: AttributionMethod,
    /// Target class (for classification).
    pub target_class: Option<usize>,
}

impl<B: Backend> AttributionMap<B> {
    /// Create a new attribution map.
    pub fn new(values: Tensor<B, 3>, method: AttributionMethod) -> Self {
        Self {
            values,
            method,
            target_class: None,
        }
    }

    /// Set the target class.
    pub fn with_target_class(mut self, class: usize) -> Self {
        self.target_class = Some(class);
        self
    }

    /// Get the shape of the attribution map.
    pub fn shape(&self) -> [usize; 3] {
        self.values.dims()
    }

    /// Normalize the attribution values to [0, 1].
    pub fn normalize(&self) -> Self {
        // Get min and max as scalars
        let min_val: f32 = self.values.clone().min().into_scalar().elem();
        let max_val: f32 = self.values.clone().max().into_scalar().elem();
        let range = max_val - min_val;

        // Avoid division by zero
        let normalized = if range > 1e-8 {
            (self.values.clone() - min_val) / range
        } else {
            self.values.clone()
        };

        Self {
            values: normalized,
            method: self.method,
            target_class: self.target_class,
        }
    }

    /// Get the mean attribution per variable.
    /// Returns tensor of shape (batch, vars, 1) - mean over time steps.
    pub fn mean_per_variable(&self) -> Tensor<B, 3> {
        self.values.clone().mean_dim(2)
    }

    /// Get the mean attribution per time step.
    /// Returns tensor of shape (batch, 1, seq_len) - mean over variables.
    pub fn mean_per_step(&self) -> Tensor<B, 3> {
        self.values.clone().mean_dim(1)
    }
}

/// Compute GradCAM attribution for CNN models.
///
/// # Arguments
///
/// * `activations` - Activations from the last conv layer (batch, channels, len)
/// * `gradients` - Gradients w.r.t. activations (batch, channels, len)
///
/// # Returns
///
/// Attribution map of shape (batch, 1, len).
pub fn grad_cam<B: Backend>(
    activations: Tensor<B, 3>,
    gradients: Tensor<B, 3>,
) -> AttributionMap<B> {
    // Global average pool the gradients: (batch, channels, len) -> (batch, channels, 1)
    let weights = gradients.mean_dim(2);

    // Weight the activations: (batch, channels, len) * (batch, channels, 1) -> (batch, channels, len)
    let weighted = activations * weights;

    // Sum across channels: (batch, channels, len) -> (batch, 1, len)
    let cam = weighted.sum_dim(1);

    // ReLU
    let cam = cam.clamp_min(0.0);

    AttributionMap::new(cam, AttributionMethod::GradCAM)
}

/// Compute Input × Gradient attribution.
///
/// # Arguments
///
/// * `input` - Model input (batch, vars, len)
/// * `gradients` - Gradients w.r.t. input (batch, vars, len)
///
/// # Returns
///
/// Attribution map of same shape as input.
pub fn input_gradient<B: Backend>(input: Tensor<B, 3>, gradients: Tensor<B, 3>) -> AttributionMap<B> {
    let attribution = input * gradients;
    AttributionMap::new(attribution.abs(), AttributionMethod::InputGradient)
}

/// Configuration for Integrated Gradients.
#[derive(Debug, Clone)]
pub struct IntegratedGradientsConfig {
    /// Number of steps for Riemann approximation of the integral.
    pub n_steps: usize,
    /// Baseline type for the path integral.
    pub baseline: BaselineType,
}

impl Default for IntegratedGradientsConfig {
    fn default() -> Self {
        Self {
            n_steps: 50,
            baseline: BaselineType::Zeros,
        }
    }
}

/// Type of baseline to use for Integrated Gradients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BaselineType {
    /// Zero baseline (most common).
    Zeros,
    /// Random baseline sampled from standard normal.
    Random,
    /// Mean of the input batch.
    Mean,
}

/// Compute Integrated Gradients attribution.
///
/// Integrated Gradients is a path-based attribution method that computes
/// the integral of gradients along a straight line path from a baseline
/// to the input. This satisfies important axioms like sensitivity and
/// implementation invariance.
///
/// Reference: Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017.
///
/// # Arguments
///
/// * `input` - Model input (batch, vars, len)
/// * `baseline` - Baseline input (batch, vars, len), typically zeros
/// * `gradients_fn` - Function that computes gradients for a given interpolated input
/// * `config` - Configuration for the method
///
/// # Returns
///
/// Attribution map of same shape as input.
///
/// # Example
///
/// ```rust,ignore
/// use tsai_explain::attribution::{integrated_gradients, IntegratedGradientsConfig};
///
/// let config = IntegratedGradientsConfig::default();
/// let attribution = integrated_gradients(&input, &baseline, |x| model.gradients(x), &config);
/// ```
pub fn integrated_gradients<B: Backend, F>(
    input: &Tensor<B, 3>,
    baseline: &Tensor<B, 3>,
    gradients_fn: F,
    config: &IntegratedGradientsConfig,
) -> AttributionMap<B>
where
    F: Fn(&Tensor<B, 3>) -> Tensor<B, 3>,
{
    let [batch_size, n_vars, seq_len] = input.dims();
    let device = input.device();
    let n_steps = config.n_steps.max(1);

    // Compute the difference (input - baseline)
    let diff = input.clone() - baseline.clone();

    // Initialize accumulator for the Riemann sum
    let mut integral = Tensor::<B, 3>::zeros([batch_size, n_vars, seq_len], &device);

    // Riemann sum approximation using trapezoidal rule
    for step in 0..=n_steps {
        // Alpha goes from 0 to 1
        let alpha = step as f32 / n_steps as f32;

        // Interpolated input: baseline + alpha * (input - baseline)
        let interpolated = baseline.clone() + diff.clone() * alpha;

        // Compute gradients at this point
        let grads = gradients_fn(&interpolated);

        // Add to integral (trapezoidal rule: weight endpoints by 0.5)
        let weight = if step == 0 || step == n_steps {
            0.5
        } else {
            1.0
        };

        integral = integral + grads * weight;
    }

    // Scale by step size and multiply by (input - baseline)
    let step_size = 1.0 / n_steps as f32;
    let attribution = diff * integral * step_size;

    AttributionMap::new(attribution.abs(), AttributionMethod::IntegratedGradients)
}

/// Create a zero baseline for Integrated Gradients.
///
/// # Arguments
///
/// * `shape` - Shape of the baseline (batch, vars, len)
/// * `device` - Device to create the tensor on
pub fn zero_baseline<B: Backend>(shape: [usize; 3], device: &B::Device) -> Tensor<B, 3> {
    Tensor::zeros(shape, device)
}

/// Create a random baseline for Integrated Gradients.
///
/// Uses standard normal distribution to match typical data normalization.
///
/// # Arguments
///
/// * `shape` - Shape of the baseline (batch, vars, len)
/// * `device` - Device to create the tensor on
pub fn random_baseline<B: Backend>(shape: [usize; 3], device: &B::Device) -> Tensor<B, 3> {
    Tensor::random(shape, burn::tensor::Distribution::Normal(0.0, 1.0), device)
}

/// Compute Attention-based attribution for transformer models.
///
/// Aggregates attention weights across layers and heads to produce
/// an attribution map showing which parts of the input the model
/// attends to.
///
/// # Arguments
///
/// * `attention_weights` - List of attention weight tensors from each layer
///                         Each tensor has shape (batch, n_heads, seq_len, seq_len)
/// * `aggregation` - How to aggregate across layers and heads
///
/// # Returns
///
/// Attribution map of shape (batch, 1, seq_len) showing attention to each position.
pub fn attention_attribution<B: Backend>(
    attention_weights: &[Tensor<B, 4>],
    aggregation: AttentionAggregation,
) -> AttributionMap<B> {
    if attention_weights.is_empty() {
        panic!("At least one attention weight tensor is required");
    }

    let first = &attention_weights[0];
    let [batch_size, _n_heads, seq_len, _] = first.dims();
    let device = first.device();

    // Start with identity for attention rollout, or zeros for mean
    let mut aggregated = match aggregation {
        AttentionAggregation::Rollout => {
            // Identity matrix for each batch
            Tensor::<B, 2>::eye(seq_len, &device)
                .unsqueeze::<3>()
                .repeat_dim(0, batch_size)
        }
        AttentionAggregation::Mean | AttentionAggregation::Last => {
            Tensor::<B, 3>::zeros([batch_size, seq_len, seq_len], &device)
        }
    };

    for (layer_idx, attn) in attention_weights.iter().enumerate() {
        // Average across heads: (batch, n_heads, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
        let attn_mean_4d = attn.clone().mean_dim(1);
        // Reshape to (batch, seq_len, seq_len)
        let attn_mean = attn_mean_4d.reshape([batch_size, seq_len, seq_len]);

        match aggregation {
            AttentionAggregation::Rollout => {
                // Attention rollout: multiply attention matrices
                // Add residual connection (identity) to attention
                let eye = Tensor::<B, 2>::eye(seq_len, &device)
                    .unsqueeze::<3>()
                    .repeat_dim(0, batch_size);
                let attn_with_residual = (attn_mean + eye.clone()) / 2.0;

                // Matrix multiplication for rollout
                aggregated = aggregated.matmul(attn_with_residual);
            }
            AttentionAggregation::Mean => {
                aggregated = aggregated + attn_mean;
            }
            AttentionAggregation::Last => {
                if layer_idx == attention_weights.len() - 1 {
                    aggregated = attn_mean;
                }
            }
        }
    }

    // For mean aggregation, divide by number of layers
    if matches!(aggregation, AttentionAggregation::Mean) {
        aggregated = aggregated / (attention_weights.len() as f32);
    }

    // Sum across query positions: (batch, seq_len, seq_len) -> (batch, 1, seq_len)
    let attribution = aggregated.sum_dim(1);

    AttributionMap::new(attribution, AttributionMethod::Attention)
}

/// How to aggregate attention weights across layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionAggregation {
    /// Attention rollout: multiply attention matrices.
    /// Reference: Abnar & Zuidema, "Quantifying Attention Flow in Transformers", ACL 2020.
    Rollout,
    /// Simple mean across all layers.
    Mean,
    /// Use only the last layer's attention.
    Last,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tsai_core::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_attribution_method() {
        let method = AttributionMethod::GradCAM;
        assert_eq!(method, AttributionMethod::GradCAM);
    }

    #[test]
    fn test_attribution_map_shape() {
        let device = Default::default();
        let values = Tensor::<TestBackend, 3>::zeros([2, 3, 10], &device);
        let map = AttributionMap::new(values, AttributionMethod::InputGradient);

        assert_eq!(map.shape(), [2, 3, 10]);
        assert_eq!(map.method, AttributionMethod::InputGradient);
        assert!(map.target_class.is_none());
    }

    #[test]
    fn test_attribution_map_with_target_class() {
        let device = Default::default();
        let values = Tensor::<TestBackend, 3>::zeros([2, 3, 10], &device);
        let map = AttributionMap::new(values, AttributionMethod::GradCAM).with_target_class(5);

        assert_eq!(map.target_class, Some(5));
    }

    #[test]
    fn test_attribution_map_normalize() {
        let device = Default::default();
        // Create tensor with known range [0, 10]
        let data: Vec<f32> = (0..60).map(|i| i as f32 / 6.0).collect();
        let values = Tensor::<TestBackend, 1>::from_floats(data.as_slice(), &device)
            .reshape([2, 3, 10]);
        let map = AttributionMap::new(values, AttributionMethod::InputGradient);

        let normalized = map.normalize();
        let norm_vals = normalized.values;

        // Check that normalized values are in [0, 1]
        let min: f32 = norm_vals.clone().min().into_scalar().elem();
        let max: f32 = norm_vals.max().into_scalar().elem();

        assert!(min >= 0.0 - 1e-6);
        assert!(max <= 1.0 + 1e-6);
    }

    #[test]
    fn test_grad_cam() {
        let device = Default::default();
        let batch_size = 2;
        let channels = 16;
        let seq_len = 50;

        let activations = Tensor::<TestBackend, 3>::ones([batch_size, channels, seq_len], &device);
        let gradients = Tensor::<TestBackend, 3>::ones([batch_size, channels, seq_len], &device);

        let cam = grad_cam(activations, gradients);

        assert_eq!(cam.shape(), [batch_size, 1, seq_len]);
        assert_eq!(cam.method, AttributionMethod::GradCAM);
    }

    #[test]
    fn test_input_gradient() {
        let device = Default::default();
        let input = Tensor::<TestBackend, 3>::ones([2, 3, 10], &device);
        let gradients = Tensor::<TestBackend, 3>::ones([2, 3, 10], &device) * 2.0;

        let attr = input_gradient(input, gradients);

        assert_eq!(attr.shape(), [2, 3, 10]);
        assert_eq!(attr.method, AttributionMethod::InputGradient);

        // Check that values are |input * gradient| = |1 * 2| = 2
        let sum: f32 = attr.values.sum().into_scalar().elem();
        assert!((sum - 2.0 * 60.0).abs() < 1e-5);
    }

    #[test]
    fn test_zero_baseline() {
        let device = Default::default();
        let baseline = zero_baseline::<TestBackend>([2, 3, 10], &device);

        assert_eq!(baseline.dims(), [2, 3, 10]);

        let sum: f32 = baseline.sum().into_scalar().elem();
        assert!(sum.abs() < 1e-6);
    }

    #[test]
    fn test_random_baseline_shape() {
        let device = Default::default();
        let baseline = random_baseline::<TestBackend>([2, 3, 10], &device);

        assert_eq!(baseline.dims(), [2, 3, 10]);
    }

    #[test]
    fn test_integrated_gradients_config() {
        let config = IntegratedGradientsConfig::default();
        assert_eq!(config.n_steps, 50);
        assert_eq!(config.baseline, BaselineType::Zeros);
    }

    #[test]
    fn test_integrated_gradients() {
        let device = Default::default();
        let input = Tensor::<TestBackend, 3>::ones([1, 2, 5], &device);
        let baseline = zero_baseline([1, 2, 5], &device);

        // Simple gradient function: returns input directly
        let gradients_fn = |x: &Tensor<TestBackend, 3>| x.clone();

        let config = IntegratedGradientsConfig {
            n_steps: 10,
            baseline: BaselineType::Zeros,
        };

        let attr = integrated_gradients(&input, &baseline, gradients_fn, &config);

        assert_eq!(attr.shape(), [1, 2, 5]);
        assert_eq!(attr.method, AttributionMethod::IntegratedGradients);
    }

    #[test]
    fn test_baseline_type_serde() {
        // Test serialization/deserialization
        let baseline = BaselineType::Random;
        let json = serde_json::to_string(&baseline).unwrap();
        let decoded: BaselineType = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, BaselineType::Random);
    }

    #[test]
    fn test_attention_aggregation_enum() {
        // Test all variants exist
        let _rollout = AttentionAggregation::Rollout;
        let _mean = AttentionAggregation::Mean;
        let _last = AttentionAggregation::Last;

        assert_eq!(AttentionAggregation::Mean, AttentionAggregation::Mean);
    }

    #[test]
    fn test_attention_attribution_mean() {
        let device = Default::default();
        let batch_size = 2;
        let n_heads = 4;
        let seq_len = 8;

        // Create two layers of attention weights
        let attn1 = Tensor::<TestBackend, 4>::ones([batch_size, n_heads, seq_len, seq_len], &device)
            / (seq_len as f32);
        let attn2 = attn1.clone();

        let attr = attention_attribution(&[attn1, attn2], AttentionAggregation::Mean);

        assert_eq!(attr.shape(), [batch_size, 1, seq_len]);
        assert_eq!(attr.method, AttributionMethod::Attention);
    }

    #[test]
    fn test_attention_attribution_last() {
        let device = Default::default();
        let batch_size = 1;
        let n_heads = 2;
        let seq_len = 4;

        let attn1 = Tensor::<TestBackend, 4>::ones([batch_size, n_heads, seq_len, seq_len], &device);
        let attn2 = Tensor::<TestBackend, 4>::ones([batch_size, n_heads, seq_len, seq_len], &device)
            * 2.0;

        let attr = attention_attribution(&[attn1, attn2], AttentionAggregation::Last);

        assert_eq!(attr.shape(), [batch_size, 1, seq_len]);
        // Last layer's attention should be used (multiplied by 2)
    }

    #[test]
    fn test_mean_per_variable() {
        let device = Default::default();
        let values = Tensor::<TestBackend, 3>::ones([2, 3, 10], &device);
        let map = AttributionMap::new(values, AttributionMethod::InputGradient);

        let mean_var = map.mean_per_variable();
        // Shape should be (batch, vars, 1) -> unsqueezed at dim 2
        assert_eq!(mean_var.dims(), [2, 3, 1]);
    }

    #[test]
    fn test_mean_per_step() {
        let device = Default::default();
        let values = Tensor::<TestBackend, 3>::ones([2, 3, 10], &device);
        let map = AttributionMap::new(values, AttributionMethod::InputGradient);

        let mean_step = map.mean_per_step();
        // Shape should be (batch, 1, seq_len) -> unsqueezed at dim 1
        assert_eq!(mean_step.dims(), [2, 1, 10]);
    }
}
