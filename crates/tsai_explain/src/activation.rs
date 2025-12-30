//! Activation and gradient capture utilities.

use std::collections::HashMap;

use burn::prelude::*;

/// Captured activations from model layers.
#[derive(Debug, Clone)]
pub struct ActivationCapture<B: Backend> {
    /// Activations by layer name.
    activations: HashMap<String, Tensor<B, 4>>,
}

impl<B: Backend> ActivationCapture<B> {
    /// Create a new activation capture.
    pub fn new() -> Self {
        Self {
            activations: HashMap::new(),
        }
    }

    /// Store an activation.
    pub fn store(&mut self, name: &str, activation: Tensor<B, 4>) {
        self.activations.insert(name.to_string(), activation);
    }

    /// Get an activation by name.
    pub fn get(&self, name: &str) -> Option<&Tensor<B, 4>> {
        self.activations.get(name)
    }

    /// Get all layer names.
    pub fn names(&self) -> Vec<&str> {
        self.activations.keys().map(|s| s.as_str()).collect()
    }

    /// Clear all stored activations.
    pub fn clear(&mut self) {
        self.activations.clear();
    }
}

impl<B: Backend> Default for ActivationCapture<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Captured gradients from model layers.
#[derive(Debug, Clone)]
pub struct GradientCapture<B: Backend> {
    /// Gradients by layer name.
    gradients: HashMap<String, Tensor<B, 4>>,
}

impl<B: Backend> GradientCapture<B> {
    /// Create a new gradient capture.
    pub fn new() -> Self {
        Self {
            gradients: HashMap::new(),
        }
    }

    /// Store a gradient.
    pub fn store(&mut self, name: &str, gradient: Tensor<B, 4>) {
        self.gradients.insert(name.to_string(), gradient);
    }

    /// Get a gradient by name.
    pub fn get(&self, name: &str) -> Option<&Tensor<B, 4>> {
        self.gradients.get(name)
    }

    /// Get all layer names.
    pub fn names(&self) -> Vec<&str> {
        self.gradients.keys().map(|s| s.as_str()).collect()
    }

    /// Clear all stored gradients.
    pub fn clear(&mut self) {
        self.gradients.clear();
    }
}

impl<B: Backend> Default for GradientCapture<B> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tsai_core::backend::NdArray;

    #[test]
    fn test_activation_capture() {
        let capture: ActivationCapture<NdArray> = ActivationCapture::new();
        assert!(capture.names().is_empty());
    }

    #[test]
    fn test_gradient_capture() {
        let capture: GradientCapture<NdArray> = GradientCapture::new();
        assert!(capture.names().is_empty());
    }
}
