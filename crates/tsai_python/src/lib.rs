//! Python bindings for tsai-rs.
//!
//! This module provides PyO3 bindings to expose tsai-rs functionality to Python.

use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, ToPyArray};
use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use tsai_train::Scheduler;

// ============================================================================
// Data loading and UCR datasets
// ============================================================================

/// Get list of available UCR univariate datasets.
#[pyfunction]
fn get_UCR_univariate_list() -> Vec<&'static str> {
    tsai_data::ucr::list_datasets().to_vec()
}

/// Get list of available UCR multivariate datasets.
#[pyfunction]
fn get_UCR_multivariate_list() -> Vec<&'static str> {
    vec![
        "ArticularyWordRecognition", "AtrialFibrillation", "BasicMotions",
        "CharacterTrajectories", "Cricket", "DuckDuckGeese", "EigenWorms",
        "Epilepsy", "ERing", "EthanolConcentration", "FaceDetection",
        "FingerMovements", "HandMovementDirection", "Handwriting", "Heartbeat",
        "InsectWingbeat", "JapaneseVowels", "Libras", "LSST", "MotorImagery",
        "NATOPS", "PEMS-SF", "PenDigits", "PhonemeSpectra", "RacketSports",
        "SelfRegulationSCP1", "SelfRegulationSCP2", "SpokenArabicDigits",
        "StandWalkJump", "UWaveGestureLibrary",
    ]
}

/// Load a UCR dataset.
#[pyfunction]
#[pyo3(signature = (dsid, return_split=true))]
fn get_UCR_data<'py>(
    py: Python<'py>,
    dsid: &str,
    return_split: bool,
) -> PyResult<PyObject> {
    let dataset = tsai_data::ucr::UCRDataset::load(dsid, None)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load dataset '{}': {}", dsid, e)))?;

    let train_x = dataset.train.x().clone();
    let train_y = dataset.train.y().cloned();
    let test_x = dataset.test.x().clone();
    let test_y = dataset.test.y().cloned();

    if return_split {
        let x_train = train_x.to_pyarray_bound(py);
        let y_train = match train_y {
            Some(y) => {
                let y_1d: Vec<f32> = y.iter().cloned().collect();
                ndarray::Array1::from_vec(y_1d).to_pyarray_bound(py)
            },
            None => ndarray::Array1::<f32>::zeros(0).to_pyarray_bound(py),
        };
        let x_test = test_x.to_pyarray_bound(py);
        let y_test = match test_y {
            Some(y) => {
                let y_1d: Vec<f32> = y.iter().cloned().collect();
                ndarray::Array1::from_vec(y_1d).to_pyarray_bound(py)
            },
            None => ndarray::Array1::<f32>::zeros(0).to_pyarray_bound(py),
        };

        let result = pyo3::types::PyTuple::new_bound(py, [
            x_train.into_any(),
            y_train.into_any(),
            x_test.into_any(),
            y_test.into_any(),
        ]);
        Ok(result.into())
    } else {
        let n_train = train_x.shape()[0];
        let n_test = test_x.shape()[0];

        let combined_x = ndarray::concatenate(
            ndarray::Axis(0),
            &[train_x.view(), test_x.view()]
        ).map_err(|e| PyRuntimeError::new_err(format!("Failed to concatenate: {}", e)))?;

        let x_arr = combined_x.to_pyarray_bound(py);

        let y_arr = if let (Some(ty), Some(tey)) = (train_y, test_y) {
            let ty_1d: Vec<f32> = ty.iter().cloned().collect();
            let tey_1d: Vec<f32> = tey.iter().cloned().collect();
            let mut combined_y = ty_1d;
            combined_y.extend(tey_1d);
            ndarray::Array1::from_vec(combined_y).to_pyarray_bound(py)
        } else {
            ndarray::Array1::<f32>::zeros(0).to_pyarray_bound(py)
        };

        let train_indices: Vec<i64> = (0..n_train as i64).collect();
        let test_indices: Vec<i64> = (n_train as i64..(n_train + n_test) as i64).collect();
        let train_arr = ndarray::Array1::from_vec(train_indices).to_pyarray_bound(py);
        let test_arr = ndarray::Array1::from_vec(test_indices).to_pyarray_bound(py);
        let splits = pyo3::types::PyTuple::new_bound(py, [train_arr.into_any(), test_arr.into_any()]);

        let result = pyo3::types::PyTuple::new_bound(py, [
            x_arr.into_any(),
            y_arr.into_any(),
            splits.into_any(),
        ]);
        Ok(result.into())
    }
}

/// Combine split data arrays into X, y and splits.
#[pyfunction]
fn combine_split_data<'py>(
    py: Python<'py>,
    x_list: Vec<PyReadonlyArray3<f32>>,
    y_list: Vec<PyReadonlyArray1<f32>>,
) -> PyResult<PyObject> {
    if x_list.is_empty() || y_list.is_empty() {
        return Err(PyValueError::new_err("X_list and y_list cannot be empty"));
    }
    if x_list.len() != y_list.len() {
        return Err(PyValueError::new_err("X_list and y_list must have same length"));
    }

    let mut sizes = Vec::new();
    let mut all_x_data = Vec::new();
    let mut all_y_data = Vec::new();

    for (x, y) in x_list.iter().zip(y_list.iter()) {
        let x_arr = x.as_array();
        let y_arr = y.as_array();
        sizes.push(x_arr.shape()[0]);

        for sample in x_arr.outer_iter() {
            all_x_data.push(sample.to_owned());
        }
        all_y_data.extend(y_arr.iter().cloned());
    }

    let n_total = sizes.iter().sum();
    let n_vars = x_list[0].as_array().shape()[1];
    let seq_len = x_list[0].as_array().shape()[2];

    let mut combined_x = ndarray::Array3::<f32>::zeros((n_total, n_vars, seq_len));
    let mut idx = 0;
    for sample in all_x_data {
        combined_x.slice_mut(ndarray::s![idx, .., ..]).assign(&sample);
        idx += 1;
    }

    let x_arr = combined_x.to_pyarray_bound(py);
    let y_arr = ndarray::Array1::from_vec(all_y_data).to_pyarray_bound(py);

    let mut splits = Vec::new();
    let mut start = 0i64;
    for size in &sizes {
        let indices: Vec<i64> = (start..start + *size as i64).collect();
        splits.push(ndarray::Array1::from_vec(indices).to_pyarray_bound(py));
        start += *size as i64;
    }

    let splits_tuple = pyo3::types::PyTuple::new_bound(py, splits);
    let result = pyo3::types::PyTuple::new_bound(py, [
        x_arr.into_any(),
        y_arr.into_any(),
        splits_tuple.into_any(),
    ]);
    Ok(result.into())
}

/// Create train/test split indices.
#[pyfunction]
#[pyo3(signature = (n_samples, test_size=0.2, shuffle=true, seed=None))]
fn train_test_split_indices<'py>(
    py: Python<'py>,
    n_samples: usize,
    test_size: f64,
    shuffle: bool,
    seed: Option<u64>,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let mut indices: Vec<usize> = (0..n_samples).collect();

    if shuffle {
        let mut rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };
        indices.shuffle(&mut rng);
    }

    let n_test = (n_samples as f64 * test_size).round() as usize;
    let n_train = n_samples - n_test;

    let train_indices: Vec<i64> = indices[..n_train].iter().map(|&x| x as i64).collect();
    let test_indices: Vec<i64> = indices[n_train..].iter().map(|&x| x as i64).collect();

    (
        ndarray::Array1::from_vec(train_indices).to_pyarray_bound(py),
        ndarray::Array1::from_vec(test_indices).to_pyarray_bound(py),
    )
}

// ============================================================================
// Dataset wrapper
// ============================================================================

#[pyclass]
pub struct TSDataset {
    x: ndarray::Array3<f32>,
    y: Option<ndarray::Array1<f32>>,
}

#[pymethods]
impl TSDataset {
    #[new]
    #[pyo3(signature = (x, y=None))]
    fn new(
        x: PyReadonlyArray3<f32>,
        y: Option<PyReadonlyArray1<f32>>,
    ) -> Self {
        Self {
            x: x.as_array().to_owned(),
            y: y.map(|arr| arr.as_array().to_owned()),
        }
    }

    fn __len__(&self) -> usize {
        self.x.shape()[0]
    }

    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        let s = self.x.shape();
        (s[0], s[1], s[2])
    }

    #[getter]
    fn n_vars(&self) -> usize {
        self.x.shape()[1]
    }

    #[getter]
    fn seq_len(&self) -> usize {
        self.x.shape()[2]
    }

    fn get_x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
        self.x.to_pyarray_bound(py)
    }

    fn get_y<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f32>>> {
        self.y.clone().map(|y| y.to_pyarray_bound(py))
    }

    fn subset(&self, indices: PyReadonlyArray1<i64>) -> PyResult<Self> {
        let indices: Vec<usize> = indices.as_array().iter().map(|&i| i as usize).collect();

        let n = self.x.shape()[0];
        for &i in &indices {
            if i >= n {
                return Err(PyValueError::new_err(format!("Index {} out of bounds for dataset of size {}", i, n)));
            }
        }

        let n_vars = self.x.shape()[1];
        let seq_len = self.x.shape()[2];
        let mut new_x = ndarray::Array3::<f32>::zeros((indices.len(), n_vars, seq_len));

        for (new_i, &old_i) in indices.iter().enumerate() {
            new_x.slice_mut(ndarray::s![new_i, .., ..])
                .assign(&self.x.slice(ndarray::s![old_i, .., ..]));
        }

        let new_y = self.y.as_ref().map(|y| {
            let subset: Vec<f32> = indices.iter().map(|&i| y[i]).collect();
            ndarray::Array1::from_vec(subset)
        });

        Ok(Self { x: new_x, y: new_y })
    }

    fn __repr__(&self) -> String {
        let (n, v, l) = self.shape();
        format!("TSDataset(samples={}, vars={}, seq_len={})", n, v, l)
    }
}

// ============================================================================
// Model configuration bindings
// ============================================================================

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

    #[getter] fn n_vars(&self) -> usize { self.inner.n_vars }
    #[getter] fn seq_len(&self) -> usize { self.inner.seq_len }
    #[getter] fn n_classes(&self) -> usize { self.inner.n_classes }
    #[getter] fn n_blocks(&self) -> usize { self.inner.n_blocks }
    #[getter] fn n_filters(&self) -> usize { self.inner.n_filters }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyclass]
#[derive(Clone)]
pub struct ResNetPlusConfig {
    inner: tsai_models::ResNetPlusConfig,
}

#[pymethods]
impl ResNetPlusConfig {
    #[new]
    #[pyo3(signature = (n_vars, seq_len, n_classes, n_blocks=3, n_filters=64))]
    fn new(
        n_vars: usize,
        seq_len: usize,
        n_classes: usize,
        n_blocks: usize,
        n_filters: usize,
    ) -> Self {
        Self {
            inner: tsai_models::ResNetPlusConfig {
                n_vars,
                seq_len,
                n_classes,
                n_blocks,
                n_filters: vec![n_filters; n_blocks],
                kernel_size: 8,
            },
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ResNetPlusConfig(n_vars={}, seq_len={}, n_classes={})",
            self.inner.n_vars, self.inner.seq_len, self.inner.n_classes
        )
    }

    #[getter] fn n_vars(&self) -> usize { self.inner.n_vars }
    #[getter] fn seq_len(&self) -> usize { self.inner.seq_len }
    #[getter] fn n_classes(&self) -> usize { self.inner.n_classes }
}

#[pyclass]
#[derive(Clone)]
pub struct PatchTSTConfig {
    inner: tsai_models::PatchTSTConfig,
}

#[pymethods]
impl PatchTSTConfig {
    #[staticmethod]
    fn for_classification(n_vars: usize, seq_len: usize, n_classes: usize) -> Self {
        Self {
            inner: tsai_models::PatchTSTConfig::for_classification(n_vars, seq_len, n_classes),
        }
    }

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

    #[getter] fn n_vars(&self) -> usize { self.inner.n_vars }
    #[getter] fn seq_len(&self) -> usize { self.inner.seq_len }
    #[getter] fn n_outputs(&self) -> usize { self.inner.n_outputs }
    #[getter] fn patch_len(&self) -> usize { self.inner.patch_len }
    #[getter] fn stride(&self) -> usize { self.inner.stride }
    #[getter] fn n_patches(&self) -> usize { self.inner.n_patches() }
    #[getter] fn d_model(&self) -> usize { self.inner.d_model }
    #[getter] fn n_heads(&self) -> usize { self.inner.n_heads }
    #[getter] fn n_layers(&self) -> usize { self.inner.n_layers }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyclass]
#[derive(Clone)]
pub struct TSTConfig {
    inner: tsai_models::TSTConfig,
}

#[pymethods]
impl TSTConfig {
    #[new]
    #[pyo3(signature = (n_vars, seq_len, n_classes, d_model=128, n_heads=8, n_layers=3, dropout=0.1))]
    fn new(
        n_vars: usize,
        seq_len: usize,
        n_classes: usize,
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        dropout: f64,
    ) -> Self {
        Self {
            inner: tsai_models::TSTConfig {
                n_vars,
                seq_len,
                n_classes,
                d_model,
                n_heads,
                n_layers,
                d_ff: d_model * 4,
                dropout,
                use_pe: true,
            },
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TSTConfig(n_vars={}, seq_len={}, n_classes={}, d_model={})",
            self.inner.n_vars, self.inner.seq_len, self.inner.n_classes, self.inner.d_model
        )
    }

    #[getter] fn n_vars(&self) -> usize { self.inner.n_vars }
    #[getter] fn seq_len(&self) -> usize { self.inner.seq_len }
    #[getter] fn n_classes(&self) -> usize { self.inner.n_classes }
    #[getter] fn d_model(&self) -> usize { self.inner.d_model }
    #[getter] fn n_heads(&self) -> usize { self.inner.n_heads }
    #[getter] fn n_layers(&self) -> usize { self.inner.n_layers }
}

#[pyclass]
#[derive(Clone)]
pub struct RNNPlusConfig {
    inner: tsai_models::RNNPlusConfig,
}

#[pymethods]
impl RNNPlusConfig {
    #[new]
    #[pyo3(signature = (n_vars, seq_len, n_classes, hidden_size=128, n_layers=2, rnn_type="lstm", bidirectional=true, dropout=0.1))]
    fn new(
        n_vars: usize,
        seq_len: usize,
        n_classes: usize,
        hidden_size: usize,
        n_layers: usize,
        rnn_type: &str,
        bidirectional: bool,
        dropout: f64,
    ) -> PyResult<Self> {
        let rnn_type = match rnn_type.to_lowercase().as_str() {
            "lstm" => tsai_models::RNNType::LSTM,
            "gru" => tsai_models::RNNType::GRU,
            _ => return Err(PyValueError::new_err("rnn_type must be 'lstm' or 'gru'")),
        };

        Ok(Self {
            inner: tsai_models::RNNPlusConfig {
                n_vars,
                seq_len,
                n_classes,
                hidden_size,
                n_layers,
                rnn_type,
                bidirectional,
                dropout,
            },
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "RNNPlusConfig(n_vars={}, seq_len={}, n_classes={}, hidden_size={})",
            self.inner.n_vars, self.inner.seq_len, self.inner.n_classes, self.inner.hidden_size
        )
    }

    #[getter] fn n_vars(&self) -> usize { self.inner.n_vars }
    #[getter] fn seq_len(&self) -> usize { self.inner.seq_len }
    #[getter] fn n_classes(&self) -> usize { self.inner.n_classes }
    #[getter] fn hidden_size(&self) -> usize { self.inner.hidden_size }
}

#[pyclass]
#[derive(Clone)]
pub struct MiniRocketConfig {
    inner: tsai_models::MiniRocketConfig,
}

#[pymethods]
impl MiniRocketConfig {
    #[new]
    #[pyo3(signature = (n_vars, seq_len, n_classes, n_features=10000, seed=42))]
    fn new(
        n_vars: usize,
        seq_len: usize,
        n_classes: usize,
        n_features: usize,
        seed: u64,
    ) -> Self {
        Self {
            inner: tsai_models::MiniRocketConfig {
                n_vars,
                seq_len,
                n_classes,
                n_features,
                seed,
            },
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "MiniRocketConfig(n_vars={}, seq_len={}, n_classes={}, n_features={})",
            self.inner.n_vars, self.inner.seq_len, self.inner.n_classes, self.inner.n_features
        )
    }

    #[getter] fn n_vars(&self) -> usize { self.inner.n_vars }
    #[getter] fn seq_len(&self) -> usize { self.inner.seq_len }
    #[getter] fn n_classes(&self) -> usize { self.inner.n_classes }
    #[getter] fn n_features(&self) -> usize { self.inner.n_features }
}

// ============================================================================
// Training configuration bindings
// ============================================================================

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

    #[getter] fn lr(&self) -> f64 { self.inner.lr }
    #[getter] fn weight_decay(&self) -> f64 { self.inner.weight_decay }
    #[getter] fn grad_clip(&self) -> f64 { self.inner.grad_clip }
    #[getter] fn mixed_precision(&self) -> bool { self.inner.mixed_precision }
}

#[pyclass]
pub struct OneCycleLR {
    inner: tsai_train::OneCycleLR,
}

#[pymethods]
impl OneCycleLR {
    #[staticmethod]
    fn simple(max_lr: f64, total_steps: usize) -> Self {
        Self {
            inner: tsai_train::OneCycleLR::simple(max_lr, total_steps),
        }
    }

    fn get_lr(&self, step: usize) -> f64 {
        self.inner.get_lr(step)
    }

    fn get_lr_schedule<'py>(&self, py: Python<'py>, n_steps: usize) -> Bound<'py, PyArray1<f64>> {
        let lrs: Vec<f64> = (0..n_steps).map(|s| self.inner.get_lr(s)).collect();
        ndarray::Array1::from_vec(lrs).to_pyarray_bound(py)
    }

    fn __repr__(&self) -> String {
        "OneCycleLR(...)".to_string()
    }
}

// ============================================================================
// Analysis bindings
// ============================================================================

#[pyclass]
pub struct ConfusionMatrix {
    inner: tsai_analysis::ConfusionMatrix,
}

#[pymethods]
impl ConfusionMatrix {
    fn accuracy(&self) -> f64 { self.inner.accuracy() as f64 }
    fn macro_f1(&self) -> f64 { self.inner.macro_f1() as f64 }
    fn precision(&self, class: usize) -> f64 { self.inner.precision(class) as f64 }
    fn recall(&self, class: usize) -> f64 { self.inner.recall(class) as f64 }
    fn f1(&self, class: usize) -> f64 { self.inner.f1(class) as f64 }

    fn matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<usize>> {
        let mat = &self.inner.matrix;
        let n_classes = self.inner.n_classes;
        let mut arr = ndarray::Array2::<usize>::zeros((n_classes, n_classes));
        for (i, row) in mat.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                arr[[i, j]] = val;
            }
        }
        arr.to_pyarray_bound(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "ConfusionMatrix(accuracy={:.4}, macro_f1={:.4})",
            self.inner.accuracy(), self.inner.macro_f1()
        )
    }
}

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

#[pyclass]
#[derive(Clone)]
pub struct TopLoss {
    #[pyo3(get)] index: usize,
    #[pyo3(get)] loss: f32,
    #[pyo3(get)] target: usize,
    #[pyo3(get)] pred: usize,
    #[pyo3(get)] prob: f32,
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

    let rows = result.len();
    let cols = if rows > 0 { result[0].len() } else { 0 };
    let mut array = ndarray::Array2::<f32>::zeros((rows, cols));
    for (i, row) in result.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            array[[i, j]] = val;
        }
    }

    array.to_pyarray_bound(py)
}

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

    array.to_pyarray_bound(py)
}

#[pyfunction]
#[pyo3(signature = (series, _threshold=0.1))]
fn compute_recurrence_plot<'py>(
    py: Python<'py>,
    series: PyReadonlyArray1<f32>,
    _threshold: f32,
) -> Bound<'py, PyArray2<f32>> {
    let series_vec: Vec<f32> = series.as_array().to_vec();
    let size = series_vec.len();

    let rp = tsai_transforms::TSToRP::new(size);
    let result = rp.compute(&series_vec);

    let rows = result.len();
    let cols = if rows > 0 { result[0].len() } else { 0 };
    let mut array = ndarray::Array2::<f32>::zeros((rows, cols));
    for (i, row) in result.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            array[[i, j]] = val;
        }
    }

    array.to_pyarray_bound(py)
}

#[pyfunction]
#[pyo3(signature = (x, by_sample=true))]
fn ts_standardize<'py>(
    py: Python<'py>,
    x: PyReadonlyArray3<f32>,
    by_sample: bool,
) -> Bound<'py, PyArray3<f32>> {
    let x_arr = x.as_array().to_owned();
    let (n_samples, n_vars, seq_len) = (x_arr.shape()[0], x_arr.shape()[1], x_arr.shape()[2]);

    let mut result = ndarray::Array3::<f32>::zeros((n_samples, n_vars, seq_len));

    if by_sample {
        for i in 0..n_samples {
            let sample = x_arr.slice(ndarray::s![i, .., ..]);
            let mean = sample.mean().unwrap_or(0.0);
            let std = sample.std(0.0);
            let std = if std < 1e-8 { 1.0 } else { std };

            for v in 0..n_vars {
                for t in 0..seq_len {
                    result[[i, v, t]] = (x_arr[[i, v, t]] - mean) / std;
                }
            }
        }
    } else {
        let mean = x_arr.mean().unwrap_or(0.0);
        let std = x_arr.std(0.0);
        let std = if std < 1e-8 { 1.0 } else { std };

        for i in 0..n_samples {
            for v in 0..n_vars {
                for t in 0..seq_len {
                    result[[i, v, t]] = (x_arr[[i, v, t]] - mean) / std;
                }
            }
        }
    }

    result.to_pyarray_bound(py)
}

#[pyfunction]
#[pyo3(signature = (x, std=0.1, seed=None))]
fn add_gaussian_noise<'py>(
    py: Python<'py>,
    x: PyReadonlyArray3<f32>,
    std: f32,
    seed: Option<u64>,
) -> Bound<'py, PyArray3<f32>> {
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let x_arr = x.as_array();
    let mut result = x_arr.to_owned();

    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };

    for val in result.iter_mut() {
        let noise: f32 = rng.gen::<f32>() * 2.0 - 1.0;
        *val += noise * std;
    }

    result.to_pyarray_bound(py)
}

#[pyfunction]
#[pyo3(signature = (x, scale_range=(0.8, 1.2), seed=None))]
fn mag_scale<'py>(
    py: Python<'py>,
    x: PyReadonlyArray3<f32>,
    scale_range: (f32, f32),
    seed: Option<u64>,
) -> Bound<'py, PyArray3<f32>> {
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let x_arr = x.as_array();
    let (n_samples, n_vars, seq_len) = (x_arr.shape()[0], x_arr.shape()[1], x_arr.shape()[2]);
    let mut result = ndarray::Array3::<f32>::zeros((n_samples, n_vars, seq_len));

    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };

    for i in 0..n_samples {
        let scale = rng.gen_range(scale_range.0..scale_range.1);
        for v in 0..n_vars {
            for t in 0..seq_len {
                result[[i, v, t]] = x_arr[[i, v, t]] * scale;
            }
        }
    }

    result.to_pyarray_bound(py)
}

// ============================================================================
// Feature extraction bindings
// ============================================================================

/// Extract features from a time series.
#[pyfunction]
#[pyo3(signature = (series, feature_set="efficient"))]
fn extract_features<'py>(
    py: Python<'py>,
    series: PyReadonlyArray1<f32>,
    feature_set: &str,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
    let series_vec: Vec<f32> = series.as_array().to_vec();

    let fs = match feature_set.to_lowercase().as_str() {
        "minimal" => tsai_analysis::FeatureSet::Minimal,
        "efficient" => tsai_analysis::FeatureSet::Efficient,
        "comprehensive" => tsai_analysis::FeatureSet::Comprehensive,
        "all" => tsai_analysis::FeatureSet::All,
        _ => return Err(PyValueError::new_err("feature_set must be 'minimal', 'efficient', 'comprehensive', or 'all'")),
    };

    let features = tsai_analysis::extract_features(&series_vec, fs);

    let dict = pyo3::types::PyDict::new_bound(py);
    for (name, value) in features.iter() {
        dict.set_item(name, *value)?;
    }

    Ok(dict)
}

/// Extract features from multiple time series (batch).
#[pyfunction]
#[pyo3(signature = (series_batch, feature_set="efficient"))]
fn extract_features_batch<'py>(
    py: Python<'py>,
    series_batch: PyReadonlyArray2<f32>,
    feature_set: &str,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
    let batch = series_batch.as_array();
    let n_samples = batch.shape()[0];

    let fs = match feature_set.to_lowercase().as_str() {
        "minimal" => tsai_analysis::FeatureSet::Minimal,
        "efficient" => tsai_analysis::FeatureSet::Efficient,
        "comprehensive" => tsai_analysis::FeatureSet::Comprehensive,
        "all" => tsai_analysis::FeatureSet::All,
        _ => return Err(PyValueError::new_err("feature_set must be 'minimal', 'efficient', 'comprehensive', or 'all'")),
    };

    // Get feature names from first sample
    let first_series: Vec<f32> = batch.row(0).to_vec();
    let first_features = tsai_analysis::extract_features(&first_series, fs.clone());
    let feature_names: Vec<String> = first_features.keys().cloned().collect();

    // Extract features for all samples
    let mut all_features: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
    for name in &feature_names {
        all_features.insert(name.clone(), Vec::with_capacity(n_samples));
    }

    for i in 0..n_samples {
        let series: Vec<f32> = batch.row(i).to_vec();
        let features = tsai_analysis::extract_features(&series, fs.clone());
        for (name, value) in features {
            if let Some(vec) = all_features.get_mut(&name) {
                vec.push(value);
            }
        }
    }

    let dict = pyo3::types::PyDict::new_bound(py);
    for (name, values) in all_features {
        let arr = ndarray::Array1::from_vec(values).to_pyarray_bound(py);
        dict.set_item(name, arr)?;
    }

    Ok(dict)
}

// ============================================================================
// Additional dataset bindings
// ============================================================================

/// Get list of available UEA multivariate datasets.
#[pyfunction]
fn get_UEA_list() -> Vec<String> {
    tsai_data::uea::list_uea_datasets().iter().map(|s| s.to_string()).collect()
}

/// Get list of available TSER regression datasets.
#[pyfunction]
fn get_TSER_list() -> Vec<String> {
    tsai_data::tser::list_tser_datasets().map(String::from).collect()
}

/// Get list of available Monash forecasting datasets.
#[pyfunction]
fn get_forecasting_list() -> Vec<String> {
    tsai_data::forecasting::list_forecasting_datasets().map(String::from).collect()
}

// ============================================================================
// HPO bindings
// ============================================================================

#[pyclass]
pub struct PyHyperparameterSpace {
    inner: tsai_train::HyperparameterSpace,
}

#[pymethods]
impl PyHyperparameterSpace {
    #[new]
    fn new() -> Self {
        Self {
            inner: tsai_train::HyperparameterSpace::new(),
        }
    }

    fn add_float(&mut self, name: &str, values: Vec<f64>) -> PyResult<()> {
        self.inner.add_float(name, &values);
        Ok(())
    }

    fn add_float_range(&mut self, name: &str, min: f64, max: f64, log_scale: bool) -> PyResult<()> {
        self.inner.add_float_range(name, min, max, log_scale);
        Ok(())
    }

    fn add_int(&mut self, name: &str, values: Vec<i64>) -> PyResult<()> {
        self.inner.add_int(name, &values);
        Ok(())
    }

    fn add_int_range(&mut self, name: &str, min: i64, max: i64) -> PyResult<()> {
        self.inner.add_int_range(name, min, max);
        Ok(())
    }

    fn add_bool(&mut self, name: &str) -> PyResult<()> {
        self.inner.add_bool(name);
        Ok(())
    }

    fn add_categorical(&mut self, name: &str, options: Vec<String>) -> PyResult<()> {
        let opts: Vec<&str> = options.iter().map(|s| s.as_str()).collect();
        self.inner.add_categorical(name, &opts);
        Ok(())
    }

    fn grid_size(&self) -> usize {
        self.inner.grid_size()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("HyperparameterSpace(n_params={}, grid_size={})", self.inner.len(), self.inner.grid_size())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyParamSet {
    inner: tsai_train::ParamSet,
}

#[pymethods]
impl PyParamSet {
    fn get_float(&self, name: &str) -> PyResult<f64> {
        self.inner.get_float(name)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn get_int(&self, name: &str) -> PyResult<i64> {
        self.inner.get_int(name)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn get_bool(&self, name: &str) -> PyResult<bool> {
        self.inner.get_bool(name)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn get_categorical(&self, name: &str) -> PyResult<String> {
        self.inner.get_categorical(name)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new_bound(py);
        for (name, value) in self.inner.iter() {
            match value {
                tsai_train::ParamValue::Float(v) => dict.set_item(name, *v)?,
                tsai_train::ParamValue::Int(v) => dict.set_item(name, *v)?,
                tsai_train::ParamValue::Bool(v) => dict.set_item(name, *v)?,
                tsai_train::ParamValue::Categorical(v) => dict.set_item(name, v)?,
            }
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        let params: Vec<String> = self.inner.iter()
            .map(|(k, v)| format!("{}={:?}", k, v))
            .collect();
        format!("ParamSet({})", params.join(", "))
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyTrialResult {
    #[pyo3(get)]
    trial: usize,
    #[pyo3(get)]
    score: f64,
    params: tsai_train::ParamSet,
}

#[pymethods]
impl PyTrialResult {
    #[getter]
    fn get_params(&self) -> PyParamSet {
        PyParamSet { inner: self.params.clone() }
    }

    fn __repr__(&self) -> String {
        format!("TrialResult(trial={}, score={:.4})", self.trial, self.score)
    }
}

#[pyclass]
pub struct PySearchResult {
    #[pyo3(get)]
    best_score: f64,
    #[pyo3(get)]
    n_trials: usize,
    best_params: tsai_train::ParamSet,
    all_trials: Vec<tsai_train::TrialResult>,
}

#[pymethods]
impl PySearchResult {
    #[getter]
    fn get_best_params(&self) -> PyParamSet {
        PyParamSet { inner: self.best_params.clone() }
    }

    fn get_all_trials(&self) -> Vec<PyTrialResult> {
        self.all_trials.iter().map(|t| PyTrialResult {
            trial: t.trial,
            score: t.score,
            params: t.params.clone(),
        }).collect()
    }

    fn top_n(&self, n: usize) -> Vec<PyTrialResult> {
        let mut sorted = self.all_trials.clone();
        sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        sorted.into_iter().take(n).map(|t| PyTrialResult {
            trial: t.trial,
            score: t.score,
            params: t.params,
        }).collect()
    }

    fn __repr__(&self) -> String {
        format!("SearchResult(best_score={:.4}, n_trials={})", self.best_score, self.n_trials)
    }
}

// ============================================================================
// Utility functions
// ============================================================================

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
fn my_setup<'py>(py: Python<'py>) -> PyResult<PyObject> {
    let info = pyo3::types::PyDict::new_bound(py);
    info.set_item("tsai_rs", version())?;

    println!("tsai-rs version: {}", version());
    println!("Available UCR datasets: {}", get_UCR_univariate_list().len());

    Ok(info.unbind().into())
}

// ============================================================================
// Module definition
// ============================================================================

#[pymodule]
fn tsai_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(my_setup, m)?)?;

    // Data loading
    m.add_function(wrap_pyfunction!(get_UCR_univariate_list, m)?)?;
    m.add_function(wrap_pyfunction!(get_UCR_multivariate_list, m)?)?;
    m.add_function(wrap_pyfunction!(get_UCR_data, m)?)?;
    m.add_function(wrap_pyfunction!(get_UEA_list, m)?)?;
    m.add_function(wrap_pyfunction!(get_TSER_list, m)?)?;
    m.add_function(wrap_pyfunction!(get_forecasting_list, m)?)?;
    m.add_function(wrap_pyfunction!(combine_split_data, m)?)?;
    m.add_function(wrap_pyfunction!(train_test_split_indices, m)?)?;
    m.add_class::<TSDataset>()?;

    // Model configs
    m.add_class::<InceptionTimePlusConfig>()?;
    m.add_class::<ResNetPlusConfig>()?;
    m.add_class::<PatchTSTConfig>()?;
    m.add_class::<TSTConfig>()?;
    m.add_class::<RNNPlusConfig>()?;
    m.add_class::<MiniRocketConfig>()?;

    // Training
    m.add_class::<LearnerConfig>()?;
    m.add_class::<OneCycleLR>()?;

    // Analysis
    m.add_class::<ConfusionMatrix>()?;
    m.add_class::<TopLoss>()?;
    m.add_function(wrap_pyfunction!(confusion_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(top_losses, m)?)?;

    // Feature extraction
    m.add_function(wrap_pyfunction!(extract_features, m)?)?;
    m.add_function(wrap_pyfunction!(extract_features_batch, m)?)?;

    // Transforms
    m.add_function(wrap_pyfunction!(compute_gasf, m)?)?;
    m.add_function(wrap_pyfunction!(compute_gadf, m)?)?;
    m.add_function(wrap_pyfunction!(compute_recurrence_plot, m)?)?;
    m.add_function(wrap_pyfunction!(ts_standardize, m)?)?;
    m.add_function(wrap_pyfunction!(add_gaussian_noise, m)?)?;
    m.add_function(wrap_pyfunction!(mag_scale, m)?)?;

    // HPO
    m.add_class::<PyHyperparameterSpace>()?;
    m.add_class::<PyParamSet>()?;
    m.add_class::<PyTrialResult>()?;
    m.add_class::<PySearchResult>()?;

    Ok(())
}
