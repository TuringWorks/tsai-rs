//! Python bindings for tsai-rs.
//!
//! This module provides PyO3 bindings to expose tsai-rs functionality to Python.
//!
//! # Usage from Python
//!
//! ```python
//! import tsai_rs
//!
//! # Create model config
//! config = tsai_rs.InceptionTimePlusConfig(
//!     n_vars=1,
//!     seq_len=100,
//!     n_classes=5
//! )
//!
//! # Compute confusion matrix
//! cm = tsai_rs.confusion_matrix(preds, targets, n_classes=5)
//! print(f"Accuracy: {cm.accuracy()}")
//!
//! # Compute GASF image
//! gasf = tsai_rs.compute_gasf(time_series)
//! ```

use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

// ============================================================================
// Model configuration bindings
// ============================================================================

/// Configuration for InceptionTimePlus model.
#[pyclass]
#[derive(Clone)]
pub struct InceptionTimePlusConfig {
    inner: tsai_models::InceptionTimePlusConfig,
}

#[pymethods]
impl InceptionTimePlusConfig {
    #[new]
    #[pyo3(signature = (n_vars, seq_len, n_classes, n_blocks=6, n_filters=32, bottleneck_dim=32, dropout=0.0))]
    fn new(
        n_vars: usize,
        seq_len: usize,
        n_classes: usize,
        n_blocks: usize,
        n_filters: usize,
        bottleneck_dim: usize,
        dropout: f64,
    ) -> Self {
        Self {
            inner: tsai_models::InceptionTimePlusConfig {
                n_vars,
                seq_len,
                n_classes,
                n_blocks,
                n_filters,
                kernel_sizes: [9, 19, 39],
                bottleneck_dim,
                dropout,
            },
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "InceptionTimePlusConfig(n_vars={}, seq_len={}, n_classes={}, n_blocks={}, n_filters={})",
            self.inner.n_vars, self.inner.seq_len, self.inner.n_classes,
            self.inner.n_blocks, self.inner.n_filters
        )
    }

    #[getter]
    fn n_vars(&self) -> usize {
        self.inner.n_vars
    }

    #[getter]
    fn seq_len(&self) -> usize {
        self.inner.seq_len
    }

    #[getter]
    fn n_classes(&self) -> usize {
        self.inner.n_classes
    }

    #[getter]
    fn n_blocks(&self) -> usize {
        self.inner.n_blocks
    }

    #[getter]
    fn n_filters(&self) -> usize {
        self.inner.n_filters
    }

    /// Convert to JSON string.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}

/// Configuration for PatchTST model.
#[pyclass]
#[derive(Clone)]
pub struct PatchTSTConfig {
    inner: tsai_models::PatchTSTConfig,
}

#[pymethods]
impl PatchTSTConfig {
    /// Create a config for classification.
    #[staticmethod]
    fn for_classification(n_vars: usize, seq_len: usize, n_classes: usize) -> Self {
        Self {
            inner: tsai_models::PatchTSTConfig::for_classification(n_vars, seq_len, n_classes),
        }
    }

    /// Create a config for forecasting.
    #[staticmethod]
    fn for_forecasting(n_vars: usize, seq_len: usize, horizon: usize) -> Self {
        Self {
            inner: tsai_models::PatchTSTConfig::for_forecasting(n_vars, seq_len, horizon),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PatchTSTConfig(n_vars={}, seq_len={}, n_outputs={}, patch_len={}, n_patches={})",
            self.inner.n_vars, self.inner.seq_len, self.inner.n_outputs,
            self.inner.patch_len, self.inner.n_patches()
        )
    }

    #[getter]
    fn n_vars(&self) -> usize {
        self.inner.n_vars
    }

    #[getter]
    fn seq_len(&self) -> usize {
        self.inner.seq_len
    }

    #[getter]
    fn n_outputs(&self) -> usize {
        self.inner.n_outputs
    }

    #[getter]
    fn patch_len(&self) -> usize {
        self.inner.patch_len
    }

    #[getter]
    fn stride(&self) -> usize {
        self.inner.stride
    }

    #[getter]
    fn n_patches(&self) -> usize {
        self.inner.n_patches()
    }

    #[getter]
    fn d_model(&self) -> usize {
        self.inner.d_model
    }

    #[getter]
    fn n_heads(&self) -> usize {
        self.inner.n_heads
    }

    #[getter]
    fn n_layers(&self) -> usize {
        self.inner.n_layers
    }

    /// Convert to JSON string.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}

/// Configuration for RNNPlus model.
#[pyclass]
#[derive(Clone)]
pub struct RNNPlusConfig {
    inner: tsai_models::RNNPlusConfig,
}

#[pymethods]
impl RNNPlusConfig {
    #[new]
    #[pyo3(signature = (n_vars, seq_len, n_outputs, hidden_dim=128, n_layers=2, rnn_type="lstm", bidirectional=true, dropout=0.1))]
    fn new(
        n_vars: usize,
        seq_len: usize,
        n_outputs: usize,
        hidden_dim: usize,
        n_layers: usize,
        rnn_type: &str,
        bidirectional: bool,
        dropout: f64,
    ) -> PyResult<Self> {
        let rnn_type = match rnn_type.to_lowercase().as_str() {
            "lstm" => tsai_models::RNNType::LSTM,
            "gru" => tsai_models::RNNType::GRU,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                "rnn_type must be 'lstm' or 'gru'"
            )),
        };

        Ok(Self {
            inner: tsai_models::RNNPlusConfig {
                n_vars,
                seq_len,
                n_outputs,
                hidden_dim,
                n_layers,
                rnn_type,
                bidirectional,
                dropout,
            },
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "RNNPlusConfig(n_vars={}, seq_len={}, n_outputs={}, hidden_dim={}, n_layers={})",
            self.inner.n_vars, self.inner.seq_len, self.inner.n_outputs,
            self.inner.hidden_dim, self.inner.n_layers
        )
    }
}

// ============================================================================
// Training configuration bindings
// ============================================================================

/// Training configuration.
#[pyclass]
#[derive(Clone)]
pub struct LearnerConfig {
    inner: tsai_train::LearnerConfig,
}

#[pymethods]
impl LearnerConfig {
    #[new]
    #[pyo3(signature = (lr=1e-3, weight_decay=0.01, grad_clip=1.0, mixed_precision=false))]
    fn new(lr: f64, weight_decay: f64, grad_clip: f64, mixed_precision: bool) -> Self {
        Self {
            inner: tsai_train::LearnerConfig {
                lr,
                weight_decay,
                grad_clip,
                mixed_precision,
            },
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "LearnerConfig(lr={}, weight_decay={}, grad_clip={})",
            self.inner.lr, self.inner.weight_decay, self.inner.grad_clip
        )
    }

    #[getter]
    fn lr(&self) -> f64 {
        self.inner.lr
    }

    #[getter]
    fn weight_decay(&self) -> f64 {
        self.inner.weight_decay
    }

    #[getter]
    fn grad_clip(&self) -> f64 {
        self.inner.grad_clip
    }

    #[getter]
    fn mixed_precision(&self) -> bool {
        self.inner.mixed_precision
    }
}

// ============================================================================
// Learning rate scheduler bindings
// ============================================================================

/// One-cycle learning rate scheduler.
#[pyclass]
pub struct OneCycleLR {
    inner: tsai_train::OneCycleLR,
}

#[pymethods]
impl OneCycleLR {
    /// Create a simple one-cycle scheduler.
    #[staticmethod]
    fn simple(max_lr: f64, total_steps: usize) -> Self {
        Self {
            inner: tsai_train::OneCycleLR::simple(max_lr, total_steps),
        }
    }

    /// Get learning rate at a given step.
    fn get_lr(&self, step: usize) -> f64 {
        self.inner.get_lr(step)
    }

    /// Get learning rates for a range of steps.
    fn get_lr_schedule<'py>(&self, py: Python<'py>, n_steps: usize) -> Bound<'py, PyArray1<f64>> {
        let lrs: Vec<f64> = (0..n_steps).map(|s| self.inner.get_lr(s)).collect();
        PyArray1::from_vec(py, lrs)
    }

    fn __repr__(&self) -> String {
        "OneCycleLR(...)".to_string()
    }
}

// ============================================================================
// Analysis bindings
// ============================================================================

/// Confusion matrix for classification.
#[pyclass]
pub struct ConfusionMatrix {
    inner: tsai_analysis::ConfusionMatrix,
}

#[pymethods]
impl ConfusionMatrix {
    /// Compute accuracy.
    fn accuracy(&self) -> f64 {
        self.inner.accuracy()
    }

    /// Compute macro-averaged F1 score.
    fn macro_f1(&self) -> f64 {
        self.inner.macro_f1()
    }

    /// Compute precision for a specific class.
    fn precision(&self, class: usize) -> f64 {
        self.inner.precision(class)
    }

    /// Compute recall for a specific class.
    fn recall(&self, class: usize) -> f64 {
        self.inner.recall(class)
    }

    /// Compute F1 score for a specific class.
    fn f1(&self, class: usize) -> f64 {
        self.inner.f1(class)
    }

    /// Get the confusion matrix as a 2D array.
    fn matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<usize>> {
        let mat = self.inner.matrix();
        PyArray2::from_owned_array(py, mat)
    }

    fn __repr__(&self) -> String {
        format!(
            "ConfusionMatrix(accuracy={:.4}, macro_f1={:.4})",
            self.inner.accuracy(), self.inner.macro_f1()
        )
    }
}

/// Compute confusion matrix from predictions and targets.
///
/// Args:
///     preds: Predicted class indices (numpy array of integers)
///     targets: True class indices (numpy array of integers)
///     n_classes: Number of classes
///
/// Returns:
///     ConfusionMatrix object with accuracy, F1, precision, recall methods
#[pyfunction]
fn confusion_matrix(
    preds: PyReadonlyArray1<i64>,
    targets: PyReadonlyArray1<i64>,
    n_classes: usize,
) -> ConfusionMatrix {
    let preds_vec: Vec<usize> = preds.as_array().iter().map(|&x| x as usize).collect();
    let targets_vec: Vec<usize> = targets.as_array().iter().map(|&x| x as usize).collect();

    let cm = tsai_analysis::confusion_matrix(&preds_vec, &targets_vec, n_classes);
    ConfusionMatrix { inner: cm }
}

/// A record of a top loss sample.
#[pyclass]
#[derive(Clone)]
pub struct TopLoss {
    /// Index of the sample in the dataset.
    #[pyo3(get)]
    index: usize,
    /// Loss value.
    #[pyo3(get)]
    loss: f32,
    /// True class label.
    #[pyo3(get)]
    target: usize,
    /// Predicted class label.
    #[pyo3(get)]
    pred: usize,
    /// Prediction probability for the predicted class.
    #[pyo3(get)]
    prob: f32,
}

#[pymethods]
impl TopLoss {
    fn __repr__(&self) -> String {
        format!(
            "TopLoss(index={}, loss={:.4}, target={}, pred={}, prob={:.4})",
            self.index, self.loss, self.target, self.pred, self.prob
        )
    }
}

/// Find the top K samples with highest losses.
///
/// Args:
///     losses: Per-sample losses (numpy array of floats)
///     targets: True class indices (numpy array of integers)
///     preds: Predicted class indices (numpy array of integers)
///     probs: Prediction probabilities for predicted class (numpy array of floats)
///     k: Number of top losses to return
///
/// Returns:
///     List of TopLoss records, sorted by descending loss
#[pyfunction]
fn top_losses(
    losses: PyReadonlyArray1<f32>,
    targets: PyReadonlyArray1<i64>,
    preds: PyReadonlyArray1<i64>,
    probs: PyReadonlyArray1<f32>,
    k: usize,
) -> Vec<TopLoss> {
    let losses_vec: Vec<f32> = losses.as_array().to_vec();
    let targets_vec: Vec<usize> = targets.as_array().iter().map(|&x| x as usize).collect();
    let preds_vec: Vec<usize> = preds.as_array().iter().map(|&x| x as usize).collect();
    let probs_vec: Vec<f32> = probs.as_array().to_vec();

    let top = tsai_analysis::top_losses(&losses_vec, &targets_vec, &preds_vec, &probs_vec, k);

    top.into_iter()
        .map(|t| TopLoss {
            index: t.index,
            loss: t.loss,
            target: t.target,
            pred: t.pred,
            prob: t.prob,
        })
        .collect()
}

// ============================================================================
// Transform bindings
// ============================================================================

/// Compute Gramian Angular Summation Field (GASF) for a time series.
///
/// GASF is an image encoding of time series that preserves temporal dependencies.
/// The resulting image can be used with 2D CNNs for time series classification.
///
/// Args:
///     series: 1D time series as numpy array
///     size: Output image size (default: length of series)
///
/// Returns:
///     2D GASF image as numpy array of shape (size, size)
#[pyfunction]
#[pyo3(signature = (series, size=None))]
fn compute_gasf<'py>(
    py: Python<'py>,
    series: PyReadonlyArray1<f32>,
    size: Option<usize>,
) -> Bound<'py, PyArray2<f32>> {
    let series_vec: Vec<f32> = series.as_array().to_vec();
    let image_size = size.unwrap_or(series_vec.len());

    let gasf = tsai_transforms::TSToGASF::new(image_size);
    let result = gasf.compute(&series_vec);

    // Convert Vec<Vec<f32>> to Array2
    let rows = result.len();
    let cols = if rows > 0 { result[0].len() } else { 0 };
    let mut array = ndarray::Array2::<f32>::zeros((rows, cols));
    for (i, row) in result.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            array[[i, j]] = val;
        }
    }

    PyArray2::from_owned_array(py, array)
}

/// Compute Gramian Angular Difference Field (GADF) for a time series.
///
/// GADF is similar to GASF but uses angular differences instead of sums.
///
/// Args:
///     series: 1D time series as numpy array
///     size: Output image size (default: length of series)
///
/// Returns:
///     2D GADF image as numpy array of shape (size, size)
#[pyfunction]
#[pyo3(signature = (series, size=None))]
fn compute_gadf<'py>(
    py: Python<'py>,
    series: PyReadonlyArray1<f32>,
    size: Option<usize>,
) -> Bound<'py, PyArray2<f32>> {
    let series_vec: Vec<f32> = series.as_array().to_vec();
    let image_size = size.unwrap_or(series_vec.len());

    let gadf = tsai_transforms::TSToGADF::new(image_size);
    let result = gadf.compute(&series_vec);

    let rows = result.len();
    let cols = if rows > 0 { result[0].len() } else { 0 };
    let mut array = ndarray::Array2::<f32>::zeros((rows, cols));
    for (i, row) in result.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            array[[i, j]] = val;
        }
    }

    PyArray2::from_owned_array(py, array)
}

/// Compute Recurrence Plot for a time series.
///
/// Recurrence plots visualize the recurrence of states in a time series.
///
/// Args:
///     series: 1D time series as numpy array
///     threshold: Distance threshold for recurrence (default: 0.1)
///
/// Returns:
///     2D binary recurrence plot as numpy array
#[pyfunction]
#[pyo3(signature = (series, threshold=0.1))]
fn compute_recurrence_plot<'py>(
    py: Python<'py>,
    series: PyReadonlyArray1<f32>,
    threshold: f32,
) -> Bound<'py, PyArray2<f32>> {
    let series_vec: Vec<f32> = series.as_array().to_vec();
    let size = series_vec.len();

    let rp = tsai_transforms::TSToRP::new(
        tsai_transforms::RecurrencePlotConfig::new(size, threshold),
    );
    let result = rp.compute(&series_vec);

    let rows = result.len();
    let cols = if rows > 0 { result[0].len() } else { 0 };
    let mut array = ndarray::Array2::<f32>::zeros((rows, cols));
    for (i, row) in result.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            array[[i, j]] = val;
        }
    }

    PyArray2::from_owned_array(py, array)
}

// ============================================================================
// Utility functions
// ============================================================================

/// Get the version of the tsai-rs library.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

// ============================================================================
// Module definition
// ============================================================================

/// tsai-rs: Time Series AI in Rust
///
/// Python bindings for the tsai-rs deep learning library for time series.
///
/// This module provides:
/// - Model configurations (InceptionTimePlus, PatchTST, RNNPlus)
/// - Training utilities (LearnerConfig, OneCycleLR scheduler)
/// - Analysis tools (confusion_matrix, top_losses)
/// - Time series to image transforms (GASF, GADF, Recurrence Plot)
///
/// Example:
///     >>> import tsai_rs
///     >>> import numpy as np
///     >>>
///     >>> # Configure a model
///     >>> config = tsai_rs.InceptionTimePlusConfig(n_vars=1, seq_len=100, n_classes=5)
///     >>> print(config)
///     InceptionTimePlusConfig(n_vars=1, seq_len=100, n_classes=5, n_blocks=6, n_filters=32)
///     >>>
///     >>> # Compute confusion matrix
///     >>> preds = np.array([0, 1, 2, 0, 1])
///     >>> targets = np.array([0, 1, 1, 0, 2])
///     >>> cm = tsai_rs.confusion_matrix(preds, targets, n_classes=3)
///     >>> print(f"Accuracy: {cm.accuracy():.2%}")
///     Accuracy: 60.00%
#[pymodule]
fn tsai_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Version
    m.add_function(wrap_pyfunction!(version, m)?)?;

    // Model configs
    m.add_class::<InceptionTimePlusConfig>()?;
    m.add_class::<PatchTSTConfig>()?;
    m.add_class::<RNNPlusConfig>()?;

    // Training
    m.add_class::<LearnerConfig>()?;
    m.add_class::<OneCycleLR>()?;

    // Analysis
    m.add_class::<ConfusionMatrix>()?;
    m.add_class::<TopLoss>()?;
    m.add_function(wrap_pyfunction!(confusion_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(top_losses, m)?)?;

    // Transforms
    m.add_function(wrap_pyfunction!(compute_gasf, m)?)?;
    m.add_function(wrap_pyfunction!(compute_gadf, m)?)?;
    m.add_function(wrap_pyfunction!(compute_recurrence_plot, m)?)?;

    Ok(())
}
