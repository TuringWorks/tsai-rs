//! Trait implementations for model training.
//!
//! Implements `TSClassificationModel` trait for all classification models.

use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use tsai_core::TSClassificationModel;

use crate::cnn::{FCN, InceptionTimePlus, OmniScaleCNN, ResNetPlus, XCMPlus, XceptionTime};
use crate::rnn::{RNNAttention, RNNPlus};
use crate::rocket::HydraPlus;
use crate::transformer::{PatchTST, TSPerceiver, TSiTPlus, TSTPlus};

// CNN Models

impl<B: AutodiffBackend> TSClassificationModel<B> for InceptionTimePlus<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        self.forward(x)
    }
}

impl<B: AutodiffBackend> TSClassificationModel<B> for XceptionTime<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        self.forward(x)
    }
}

impl<B: AutodiffBackend> TSClassificationModel<B> for FCN<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        self.forward(x)
    }
}

impl<B: AutodiffBackend> TSClassificationModel<B> for ResNetPlus<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        self.forward(x)
    }
}

impl<B: AutodiffBackend> TSClassificationModel<B> for OmniScaleCNN<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        self.forward(x)
    }
}

impl<B: AutodiffBackend> TSClassificationModel<B> for XCMPlus<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        self.forward(x)
    }
}

// Transformer Models

impl<B: AutodiffBackend> TSClassificationModel<B> for TSTPlus<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        self.forward(x)
    }
}

impl<B: AutodiffBackend> TSClassificationModel<B> for TSiTPlus<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        self.forward(x)
    }
}

impl<B: AutodiffBackend> TSClassificationModel<B> for TSPerceiver<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        self.forward(x)
    }
}

impl<B: AutodiffBackend> TSClassificationModel<B> for PatchTST<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        self.forward(x)
    }
}

// ROCKET Models
// Note: Rocket, MiniRocket, MultiRocket take pre-extracted 2D features, not raw 3D time series
// They should use a different trait (TSFeatureModel) or be wrapped with feature extraction

impl<B: AutodiffBackend> TSClassificationModel<B> for HydraPlus<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        self.forward(x)
    }
}

// RNN Models

impl<B: AutodiffBackend> TSClassificationModel<B> for RNNPlus<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        self.forward(x)
    }
}

impl<B: AutodiffBackend> TSClassificationModel<B> for RNNAttention<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        self.forward(x)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_traits_compile() {
        // Trait implementations compile - actual tests in integration tests
    }
}
