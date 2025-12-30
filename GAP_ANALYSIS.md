# tsai vs tsai-rs: Comprehensive Fit-Gap Analysis

**Generated:** December 2024
**tsai Version:** 0.4.1 (Python)
**tsai-rs Version:** 0.1.0 (Rust)

---

## Executive Summary

This document provides a comprehensive fit-gap analysis between the Python `tsai` library (timeseriesAI) and its Rust port `tsai-rs`. The analysis covers all major components including models, data handling, transforms, training infrastructure, and explainability features.

### Overall Status

| Category | tsai (Python) | tsai-rs (Rust) | Coverage |
|----------|---------------|----------------|----------|
| **Models** | 40+ architectures | 10 architectures | **25%** |
| **Augmentation Transforms** | 40+ transforms | 5 transforms | **12%** |
| **Label Mixing** | 4 transforms | 3 (stubbed) | **25%** |
| **Imaging Transforms** | 7 transforms | 4 transforms | **57%** |
| **Loss Functions** | 7+ custom losses | 3 losses | **43%** |
| **Metrics** | 10+ metrics | 4 metrics | **40%** |
| **Callbacks** | 10+ callbacks | 3 callbacks | **30%** |
| **Data I/O** | Multiple formats | 4 formats | **80%** |
| **Analysis Tools** | 5+ tools | 4 tools | **80%** |
| **Explainability** | Full suite | Partial | **60%** |

**Overall Feature Parity: ~35%**

---

## 1. Model Architectures

### 1.1 CNN Models

| Model | tsai (Python) | tsai-rs (Rust) | Status | Notes |
|-------|---------------|----------------|--------|-------|
| FCN | ✅ | ❌ | **GAP** | Fully Convolutional Network |
| ResNet | ✅ | ✅ | **FIT** | ResNetPlus implemented |
| ResCNN | ✅ | ❌ | **GAP** | 1D Residual CNN |
| InceptionTime | ✅ | ✅ | **FIT** | InceptionTimePlus implemented |
| XceptionTime | ✅ | ❌ | **GAP** | Xception adaptation |
| OmniScaleCNN | ✅ | ❌ | **GAP** | Multi-scale 1D CNN |
| XCM | ✅ | ✅ | **FIT** | XCMPlus implemented |
| TCN | ✅ | ❌ | **GAP** | Temporal Convolutional Network |
| mWDN | ✅ | ❌ | **GAP** | Multi-level Wavelet Decomposition |

### 1.2 Transformer Models

| Model | tsai (Python) | tsai-rs (Rust) | Status | Notes |
|-------|---------------|----------------|--------|-------|
| TransformerModel | ✅ | ❌ | **GAP** | Base Transformer |
| TST/TSTPlus | ✅ | ✅ | **FIT** | Time Series Transformer |
| TSiT/TSiTPlus | ✅ | ❌ | **GAP** | Vision Transformer adaptation |
| PatchTST | ✅ | ✅ | **FIT** | ICLR 2023 model |
| TSPerceiver | ✅ | ❌ | **GAP** | Perceiver IO adaptation |
| TSSequencerPlus | ✅ | ❌ | **GAP** | Sequencer adaptation |
| TabTransformer | ✅ | ✅ | **FIT** | Tabular + TS Transformer |
| GatedTabTransformer | ✅ | ❌ | **GAP** | Gated variant |
| TabFusionTransformer | ✅ | ❌ | **GAP** | Fusion variant |

### 1.3 RNN Models

| Model | tsai (Python) | tsai-rs (Rust) | Status | Notes |
|-------|---------------|----------------|--------|-------|
| LSTM | ✅ | ✅ | **FIT** | Part of RNNPlus |
| GRU | ✅ | ✅ | **FIT** | Part of RNNPlus |
| RNNPlus | ✅ | ✅ | **FIT** | Fully implemented |
| RNNAttention | ✅ | ❌ | **GAP** | RNN with attention |
| LSTMAttention | ✅ | ❌ | **GAP** | LSTM with attention |
| GRUAttention | ✅ | ❌ | **GAP** | GRU with attention |

### 1.4 Hybrid Models

| Model | tsai (Python) | tsai-rs (Rust) | Status | Notes |
|-------|---------------|----------------|--------|-------|
| RNN_FCN | ✅ | ❌ | **GAP** | RNN + FCN hybrid |
| LSTM-FCN | ✅ | ❌ | **GAP** | LSTM + FCN |
| GRU-FCN | ✅ | ❌ | **GAP** | GRU + FCN |
| MLSTM-FCN | ✅ | ❌ | **GAP** | Multi-LSTM + FCN |
| TransformerRNNPlus | ✅ | ❌ | **GAP** | Transformer + RNN |
| ConvTranPlus | ✅ | ❌ | **GAP** | Conv + Transformer |

### 1.5 ROCKET Family

| Model | tsai (Python) | tsai-rs (Rust) | Status | Notes |
|-------|---------------|----------------|--------|-------|
| ROCKET | ✅ | ❌ | **GAP** | Original ROCKET |
| MiniRocket | ✅ | ✅ | **FIT** | Implemented |
| MultiRocketPlus | ✅ | ❌ | **GAP** | Multi-variate ROCKET |
| HydraPlus | ✅ | ❌ | **GAP** | Hydra model |
| HydraMultiRocketPlus | ✅ | ❌ | **GAP** | Combined Hydra + ROCKET |

### 1.6 Other Models

| Model | tsai (Python) | tsai-rs (Rust) | Status | Notes |
|-------|---------------|----------------|--------|-------|
| MLP | ✅ | ❌ | **GAP** | Multilayer Perceptron |
| gMLP | ✅ | ❌ | **GAP** | Gated MLP |
| TabModel | ✅ | ❌ | **GAP** | Tabular model |
| MultiInputNet | ✅ | ❌ | **GAP** | Multi-modal |
| XResNet1d | ✅ | ❌ | **GAP** | XResNet for 1D |

### Model Gap Summary

- **Implemented:** 10 models (InceptionTimePlus, ResNetPlus, XCMPlus, TSTPlus, PatchTST, MiniRocket, RNNPlus, TabTransformer, LSTM, GRU)
- **Missing:** 30+ models
- **Priority Gaps:** FCN, XceptionTime, OmniScaleCNN, RNN_FCN variants, Attention-based RNNs, ROCKET, MultiRocketPlus

---

## 2. Data Augmentation Transforms

### 2.1 Basic Transforms

| Transform | tsai (Python) | tsai-rs (Rust) | Status | Notes |
|-----------|---------------|----------------|--------|-------|
| TSIdentity | ✅ | ✅ | **FIT** | Identity transform |
| TSShuffle_HLs | ✅ | ❌ | **GAP** | OHLC shuffle |
| TSShuffleSteps | ✅ | ❌ | **GAP** | Step shuffling |

### 2.2 Noise & Distortion

| Transform | tsai (Python) | tsai-rs (Rust) | Status | Notes |
|-----------|---------------|----------------|--------|-------|
| TSGaussianNoise | ✅ | ✅ | **FIT** | Implemented |
| TSMagAddNoise | ✅ | ❌ | **GAP** | Additive noise |
| TSMagMulNoise | ✅ | ❌ | **GAP** | Multiplicative noise |
| TSTimeNoise | ✅ | ❌ | **GAP** | Time-axis noise |
| TSRandomFreqNoise | ✅ | ❌ | **GAP** | Wavelet-based noise |

### 2.3 Warping & Scaling

| Transform | tsai (Python) | tsai-rs (Rust) | Status | Notes |
|-----------|---------------|----------------|--------|-------|
| TSMagWarp | ✅ | ❌ | **GAP** | Magnitude warping |
| TSTimeWarp | ✅ | ⚠️ | **PARTIAL** | Stubbed, needs spline impl |
| TSWindowWarp | ✅ | ❌ | **GAP** | Window warping |
| TSMagScale | ✅ | ✅ | **FIT** | Magnitude scaling |
| TSMagScalePerVar | ✅ | ❌ | **GAP** | Per-variable scaling |
| TSRandomTrend | ✅ | ❌ | **GAP** | Random trend |

### 2.4 Temporal Transformations

| Transform | tsai (Python) | tsai-rs (Rust) | Status | Notes |
|-----------|---------------|----------------|--------|-------|
| TSRandomShift | ✅ | ❌ | **GAP** | Random shifting |
| TSHorizontalFlip | ✅ | ❌ | **GAP** | Time reversal |
| TSVerticalFlip | ✅ | ❌ | **GAP** | Value negation |
| TSTranslateX | ✅ | ❌ | **GAP** | X-axis translation |

### 2.5 Resampling & Resolution

| Transform | tsai (Python) | tsai-rs (Rust) | Status | Notes |
|-----------|---------------|----------------|--------|-------|
| TSRandomTimeScale | ✅ | ❌ | **GAP** | Time scaling |
| TSRandomTimeStep | ✅ | ❌ | **GAP** | Random step |
| TSResampleSteps | ✅ | ❌ | **GAP** | Step resampling |
| TSResize | ✅ | ❌ | **GAP** | Resize sequence |
| TSRandomSize | ✅ | ❌ | **GAP** | Random resize |
| TSRandomLowRes | ✅ | ❌ | **GAP** | Low resolution |
| TSDownUpScale | ✅ | ❌ | **GAP** | Down/up scaling |

### 2.6 Cropping & Slicing

| Transform | tsai (Python) | tsai-rs (Rust) | Status | Notes |
|-----------|---------------|----------------|--------|-------|
| TSRandomResizedCrop | ✅ | ❌ | **GAP** | Random crop + resize |
| TSWindowSlicing | ✅ | ❌ | **GAP** | Window slicing |
| TSRandomZoomOut | ✅ | ❌ | **GAP** | Zoom out |
| TSRandomCropPad | ✅ | ❌ | **GAP** | Crop and pad |

### 2.7 Masking & Dropout

| Transform | tsai (Python) | tsai-rs (Rust) | Status | Notes |
|-----------|---------------|----------------|--------|-------|
| TSCutOut | ✅ | ⚠️ | **PARTIAL** | Stubbed, needs tensor ops |
| TSTimeStepOut | ✅ | ❌ | **GAP** | Step dropout |
| TSVarOut | ✅ | ❌ | **GAP** | Variable dropout |
| TSMaskOut | ✅ | ❌ | **GAP** | Mask-based dropout |
| TSInputDropout | ✅ | ❌ | **GAP** | Input dropout |
| TSSelfDropout | ✅ | ❌ | **GAP** | Self-dropout |

### 2.8 Smoothing & Filtering

| Transform | tsai (Python) | tsai-rs (Rust) | Status | Notes |
|-----------|---------------|----------------|--------|-------|
| TSBlur | ✅ | ❌ | **GAP** | Blur filter |
| TSSmooth | ✅ | ❌ | **GAP** | Smoothing |
| TSFreqDenoise | ✅ | ❌ | **GAP** | Frequency denoising |

### 2.9 Advanced

| Transform | tsai (Python) | tsai-rs (Rust) | Status | Notes |
|-----------|---------------|----------------|--------|-------|
| TSRandomConv | ✅ | ❌ | **GAP** | Random convolution |
| RandAugment | ✅ | ❌ | **GAP** | Auto augmentation |

### Transform Gap Summary

- **Implemented:** 4 transforms (GaussianNoise, MagScale, Identity, Compose)
- **Partial:** 2 transforms (TimeWarp, CutOut - need completion)
- **Missing:** 35+ transforms
- **Priority Gaps:** TimeWarp completion, MagWarp, WindowWarp, RandomResizedCrop, masking transforms, RandAugment

---

## 3. Label Mixing Transforms

| Transform | tsai (Python) | tsai-rs (Rust) | Status | Notes |
|-----------|---------------|----------------|--------|-------|
| MixUp1d | ✅ | ⚠️ | **PARTIAL** | Stubbed, needs tensor mixing |
| CutMix1d | ✅ | ⚠️ | **PARTIAL** | Stubbed |
| IntraClassCutMix1d | ✅ | ⚠️ | **PARTIAL** | Stubbed |
| MixHandler1d | ✅ | ❌ | **GAP** | Base handler |

---

## 4. Imaging Transforms

| Transform | tsai (Python) | tsai-rs (Rust) | Status | Notes |
|-----------|---------------|----------------|--------|-------|
| TSToGASF | ✅ | ✅ | **FIT** | Gramian Angular Summation Field |
| TSToGADF | ✅ | ✅ | **FIT** | Gramian Angular Difference Field |
| TSToMTF | ✅ | ✅ | **FIT** | Markov Transition Field |
| TSToRP | ✅ | ✅ | **FIT** | Recurrence Plot |
| TSToJRP | ✅ | ❌ | **GAP** | Joint Recurrence Plot |
| TSToPlot | ✅ | ❌ | **GAP** | Matplotlib plot |
| TSToMat | ✅ | ❌ | **GAP** | Matrix visualization |

---

## 5. Training Infrastructure

### 5.1 Loss Functions

| Loss | tsai (Python) | tsai-rs (Rust) | Status | Notes |
|------|---------------|----------------|--------|-------|
| CrossEntropyLoss | ✅ | ✅ | **FIT** | Standard CE |
| MSELoss | ✅ | ✅ | **FIT** | Mean Squared Error |
| HuberLoss | ✅ | ⚠️ | **PARTIAL** | Simplified impl |
| LogCoshLoss | ✅ | ❌ | **GAP** | Log-cosh loss |
| FocalLoss | ✅ | ❌ | **GAP** | Class imbalance |
| CenterLoss | ✅ | ❌ | **GAP** | Feature discrimination |
| CenterPlusLoss | ✅ | ❌ | **GAP** | Combined center loss |
| TweedieLoss | ✅ | ❌ | **GAP** | Probabilistic loss |
| MaskedLossWrapper | ✅ | ❌ | **GAP** | NaN handling |

### 5.2 Metrics

| Metric | tsai (Python) | tsai-rs (Rust) | Status | Notes |
|--------|---------------|----------------|--------|-------|
| Accuracy | ✅ | ✅ | **FIT** | Classification accuracy |
| MSE | ✅ | ✅ | **FIT** | Mean Squared Error |
| MAE | ✅ | ✅ | **FIT** | Mean Absolute Error |
| RMSE | ✅ | ❌ | **GAP** | Root MSE |
| F1Score | ✅ | ⚠️ | **PARTIAL** | Uses accuracy proxy |
| Precision | ✅ | ❌ | **GAP** | Per confusion matrix |
| Recall | ✅ | ❌ | **GAP** | Per confusion matrix |
| MAPE | ✅ | ❌ | **GAP** | Mean Abs Percentage Error |

### 5.3 Schedulers

| Scheduler | tsai (Python) | tsai-rs (Rust) | Status | Notes |
|-----------|---------------|----------------|--------|-------|
| OneCycleLR | ✅ | ✅ | **FIT** | Fully implemented |
| CosineAnnealingLR | ✅ | ✅ | **FIT** | Implemented |
| StepLR | ✅ | ✅ | **FIT** | Implemented |
| ExponentialLR | ✅ | ❌ | **GAP** | Exponential decay |
| ReduceLROnPlateau | ✅ | ❌ | **GAP** | Adaptive LR |

### 5.4 Callbacks

| Callback | tsai (Python) | tsai-rs (Rust) | Status | Notes |
|----------|---------------|----------------|--------|-------|
| ProgressCallback | ✅ | ✅ | **FIT** | Progress reporting |
| EarlyStopping | ✅ | ✅ | **FIT** | Early stopping |
| SaveModel | ✅ | ❌ | **GAP** | Model checkpointing |
| ShowGraph | ✅ | ❌ | **GAP** | Training visualization |
| TransformScheduler | ✅ | ❌ | **GAP** | Transform scheduling |
| WeightedPerSampleLoss | ✅ | ❌ | **GAP** | Sample weighting |
| BatchSubsampler | ✅ | ❌ | **GAP** | Batch subsampling |
| PredictionDynamics | ✅ | ❌ | **GAP** | Prediction tracking |
| NoisyStudent | ✅ | ❌ | **GAP** | Semi-supervised |

### 5.5 Optimizers

| Optimizer | tsai (Python) | tsai-rs (Rust) | Status | Notes |
|-----------|---------------|----------------|--------|-------|
| Adam | ✅ | ✅ | **FIT** | Via Burn |
| AdamW | ✅ | ✅ | **FIT** | Via Burn |
| SGD | ✅ | ✅ | **FIT** | Via Burn |
| RAdam | ✅ | ❌ | **GAP** | Rectified Adam |
| Ranger | ✅ | ❌ | **GAP** | Ranger optimizer |

---

## 6. Data Handling

### 6.1 Data I/O

| Format | tsai (Python) | tsai-rs (Rust) | Status | Notes |
|--------|---------------|----------------|--------|-------|
| NumPy .npy | ✅ | ✅ | **FIT** | Implemented |
| NumPy .npz | ✅ | ✅ | **FIT** | Implemented |
| CSV | ✅ | ✅ | **FIT** | Implemented |
| Parquet | ✅ | ✅ | **FIT** | Via Polars |
| Zarr | ✅ | ❌ | **GAP** | Array storage |
| HDF5 | ✅ | ❌ | **GAP** | Hierarchical data |
| PyTorch tensors | ✅ | ❌ | **N/A** | Python-specific |

### 6.2 Dataset Features

| Feature | tsai (Python) | tsai-rs (Rust) | Status | Notes |
|---------|---------------|----------------|--------|-------|
| TSDataset | ✅ | ✅ | **FIT** | Core dataset |
| TSDatasets | ✅ | ✅ | **FIT** | Train/valid/test |
| TSDataLoader | ✅ | ✅ | **FIT** | Batch loading |
| TSDataLoaders | ✅ | ✅ | **FIT** | Paired loaders |
| Random split | ✅ | ✅ | **FIT** | train_test_split |
| Stratified split | ✅ | ✅ | **FIT** | StratifiedSampler |
| Walk-forward CV | ✅ | ❌ | **GAP** | Time series CV |
| SlidingWindow | ✅ | ❌ | **GAP** | Window creation |
| TimeSplitter | ✅ | ❌ | **GAP** | Time-based split |

### 6.3 External Datasets

| Feature | tsai (Python) | tsai-rs (Rust) | Status | Notes |
|---------|---------------|----------------|--------|-------|
| UCR datasets (128) | ✅ | ❌ | **GAP** | Auto-download |
| UEA datasets (30) | ✅ | ❌ | **GAP** | Multivariate |
| Regression datasets (15) | ✅ | ❌ | **GAP** | External data |
| Forecasting datasets (62) | ✅ | ❌ | **GAP** | External data |
| Cache management | ✅ | ✅ | **FIT** | cache_dir() |

---

## 7. Inference

| Feature | tsai (Python) | tsai-rs (Rust) | Status | Notes |
|---------|---------------|----------------|--------|-------|
| get_X_preds | ✅ | ⚠️ | **PARTIAL** | Predictions struct exists |
| with_input option | ✅ | ✅ | **FIT** | Predictions.x |
| with_decoded option | ✅ | ✅ | **FIT** | Predictions.decoded |
| with_loss option | ✅ | ✅ | **FIT** | Predictions.losses |
| Batch inference | ✅ | ✅ | **FIT** | DataLoader support |
| load_learner | ✅ | ❌ | **GAP** | Model loading |
| Model export | ✅ | ❌ | **GAP** | Safetensors planned |

---

## 8. Analysis & Explainability

### 8.1 Analysis Tools

| Tool | tsai (Python) | tsai-rs (Rust) | Status | Notes |
|------|---------------|----------------|--------|-------|
| Confusion Matrix | ✅ | ✅ | **FIT** | Full implementation |
| Top Losses | ✅ | ✅ | **FIT** | Full implementation |
| Feature Importance | ✅ | ✅ | **FIT** | Permutation-based |
| Step Importance | ✅ | ✅ | **FIT** | Temporal importance |
| Calibration | ✅ | ❌ | **GAP** | Confidence calibration |
| Classification Report | ✅ | ❌ | **GAP** | Per-class metrics |

### 8.2 Explainability

| Feature | tsai (Python) | tsai-rs (Rust) | Status | Notes |
|---------|---------------|----------------|--------|-------|
| GradCAM | ✅ | ✅ | **FIT** | Implemented |
| Input × Gradient | ✅ | ✅ | **FIT** | Implemented |
| Integrated Gradients | ✅ | ⚠️ | **PARTIAL** | Enum exists, impl pending |
| Attention Visualization | ✅ | ⚠️ | **PARTIAL** | Enum exists, impl pending |
| Activation Capture | ✅ | ✅ | **FIT** | Full implementation |
| Gradient Capture | ✅ | ✅ | **FIT** | Full implementation |

---

## 9. Integration Features

| Feature | tsai (Python) | tsai-rs (Rust) | Status | Notes |
|---------|---------------|----------------|--------|-------|
| fastai integration | ✅ | ❌ | **N/A** | Python-specific |
| sklearn pipelines | ✅ | ❌ | **GAP** | Could add similar |
| sktime integration | ✅ | ❌ | **GAP** | ROCKET via sktime |
| tsfresh integration | ✅ | ❌ | **GAP** | Feature extraction |
| Weights & Biases | ✅ | ⚠️ | **PARTIAL** | Feature flag exists |
| Optuna HPO | ✅ | ❌ | **GAP** | Hyperparameter tuning |
| PyTorch 2.0 | ✅ | ❌ | **N/A** | Different framework |
| ONNX export | ✅ | ❌ | **GAP** | Model export |

---

## 10. Priority Gap Recommendations

### High Priority (Core Functionality)

1. **Models to Add:**
   - FCN (foundational CNN)
   - XceptionTime
   - OmniScaleCNN
   - ROCKET (original algorithm)
   - RNN_FCN variants
   - Attention-based RNNs (LSTMAttention, GRUAttention)

2. **Transforms to Complete:**
   - TimeWarp (complete spline interpolation)
   - CutOut (complete tensor operations)
   - MixUp1d, CutMix1d (complete implementations)
   - MagWarp, WindowWarp

3. **Training Infrastructure:**
   - Complete HuberLoss implementation
   - Proper F1Score metric
   - SaveModel callback
   - Model checkpointing and loading

### Medium Priority (Enhanced Functionality)

4. **Models:**
   - TSiT/TSiTPlus
   - TSPerceiver
   - MultiRocketPlus
   - HydraPlus
   - gMLP

5. **Transforms:**
   - RandAugment
   - TSRandomResizedCrop
   - Masking transforms (TSMaskOut, TSVarOut)
   - Smoothing/filtering transforms

6. **Training:**
   - FocalLoss
   - MaskedLossWrapper
   - WeightedPerSampleLoss
   - ShowGraph callback

### Low Priority (Advanced Features)

7. **Data:**
   - UCR/UEA dataset auto-download
   - SlidingWindow
   - Walk-forward cross-validation

8. **Integration:**
   - tsfresh feature extraction
   - ONNX export
   - Optuna integration

---

## 11. Implementation Recommendations

### Architecture Considerations

1. **Backend Flexibility:** tsai-rs correctly uses Burn framework for backend abstraction, allowing CPU (ndarray), GPU (WGPU), and PyTorch (tch) backends.

2. **Memory Efficiency:** Consider implementing:
   - Lazy data loading
   - Memory-mapped datasets
   - Gradient checkpointing for large models

3. **Parallelism:** Current rayon integration for CPU parallelism is good. Consider:
   - Async data loading
   - Multi-GPU support

### Code Quality

1. **Current Strengths:**
   - No unsafe code
   - Comprehensive error handling with thiserror
   - Full serde serialization support
   - Deterministic seeding with ChaCha8

2. **Improvements Needed:**
   - Remove TODO stubs in transforms
   - Complete metric implementations
   - Add model saving/loading

### Testing

1. **Current Coverage:** Good unit test coverage (~100+ tests)

2. **Needed Tests:**
   - End-to-end training integration tests
   - Model accuracy benchmarks
   - Cross-backend compatibility tests
   - Memory leak tests for training loops

---

## 12. Conclusion

tsai-rs provides a solid foundation with approximately **35% feature parity** with the Python tsai library. The core infrastructure (datasets, dataloaders, basic models, training loop) is well-implemented. The main gaps are:

1. **Model diversity** - Only 10 of 40+ models implemented
2. **Augmentation transforms** - Only 5 of 40+ transforms implemented
3. **Training utilities** - Missing advanced losses, callbacks, and checkpointing

The Rust implementation benefits from:
- Type safety and memory safety
- Potential for better performance
- Clean architecture with proper abstractions
- Good backend flexibility via Burn

**Recommended next steps:**
1. Complete partial implementations (TimeWarp, MixUp, etc.)
2. Add model checkpointing
3. Implement FCN and XceptionTime models
4. Add RandAugment for automatic augmentation
5. Complete metrics (F1, RMSE)

---

*This analysis was generated by comparing tsai v0.4.1 documentation and source with tsai-rs v0.1.0 implementation.*
