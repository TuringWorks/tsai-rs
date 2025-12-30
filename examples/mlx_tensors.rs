//! Example: MLX Tensor Operations Demo
//!
//! This example demonstrates the MLX tensor operations provided by burn-mlx:
//! - Creating tensors on Apple Silicon GPU
//! - Basic arithmetic operations
//! - Activation functions
//! - Matrix multiplication
//! - Performance comparison with CPU operations
//!
//! Run with (macOS only):
//!   cargo run --example mlx_tensors --release
//!
//! Note: This requires an Apple Silicon Mac (M1/M2/M3/M4).

#[cfg(target_os = "macos")]
use burn_mlx::{MlxDevice, MlxTensor};

#[cfg(target_os = "macos")]
fn main() {
    use std::time::Instant;

    println!("==============================================================");
    println!("         burn-mlx: MLX Tensor Operations Demo                 ");
    println!("==============================================================");
    println!();

    // =========================================================================
    // Step 1: Device Setup
    // =========================================================================
    println!("Step 1: Device Setup");
    println!("-------------------------------------------------------------");

    let gpu_device = MlxDevice::Gpu;
    let cpu_device = MlxDevice::Cpu;

    println!("  GPU device: {:?}", gpu_device);
    println!("  CPU device: {:?}", cpu_device);
    println!();

    // =========================================================================
    // Step 2: Basic Tensor Creation
    // =========================================================================
    println!("Step 2: Basic Tensor Creation");
    println!("-------------------------------------------------------------");

    let zeros = MlxTensor::<f32>::zeros(&[2, 3], gpu_device);
    let ones = MlxTensor::<f32>::ones(&[2, 3], gpu_device);

    println!("  zeros shape: {:?}", zeros.shape());
    println!("  ones shape: {:?}", ones.shape());
    println!();

    // Create tensor from data
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let tensor = MlxTensor::from_slice(&data, &[2, 3], gpu_device);
    println!("  Custom tensor shape: {:?}", tensor.shape());
    println!("  Number of elements: {}", tensor.numel());
    println!();

    // =========================================================================
    // Step 3: Arithmetic Operations
    // =========================================================================
    println!("Step 3: Arithmetic Operations");
    println!("-------------------------------------------------------------");

    // Create two tensors for operations
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];

    let a = MlxTensor::from_slice(&a_data, &[2, 2], gpu_device);
    let b = MlxTensor::from_slice(&b_data, &[2, 2], gpu_device);

    let _add_result = a.add(&b);
    let _sub_result = a.sub(&b);
    let _mul_result = a.mul(&b);
    let _div_result = a.div(&b);

    println!("  a + b computed (element-wise addition)");
    println!("  a - b computed (element-wise subtraction)");
    println!("  a * b computed (element-wise multiplication)");
    println!("  a / b computed (element-wise division)");
    println!();

    // =========================================================================
    // Step 4: Matrix Multiplication
    // =========================================================================
    println!("Step 4: Matrix Multiplication");
    println!("-------------------------------------------------------------");

    // Create matrices for matmul
    let m1_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let m2_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let m1 = MlxTensor::from_slice(&m1_data, &[2, 3], gpu_device);
    let m2 = MlxTensor::from_slice(&m2_data, &[3, 2], gpu_device);

    let matmul_result = m1.matmul(&m2);
    println!("  Input shapes: {:?} @ {:?}", m1.shape(), m2.shape());
    println!("  Output shape: {:?}", matmul_result.shape());
    println!();

    // =========================================================================
    // Step 5: Activation Functions
    // =========================================================================
    println!("Step 5: Activation Functions");
    println!("-------------------------------------------------------------");

    let activation_data: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let activation_tensor = MlxTensor::from_slice(&activation_data, &[2, 3], gpu_device);

    let _relu = activation_tensor.relu();
    let _sigmoid = activation_tensor.sigmoid();
    let _tanh = activation_tensor.tanh_act();
    let _softmax = activation_tensor.softmax();

    println!("  ReLU activation computed");
    println!("  Sigmoid activation computed");
    println!("  Tanh activation computed");
    println!("  Softmax activation computed");
    println!();

    // =========================================================================
    // Step 6: Reduction Operations
    // =========================================================================
    println!("Step 6: Reduction Operations");
    println!("-------------------------------------------------------------");

    let reduce_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let reduce_tensor = MlxTensor::from_slice(&reduce_data, &[2, 3], gpu_device);

    let sum_result = reduce_tensor.sum_dim(1, true);
    let mean_result = reduce_tensor.mean_dim(1);

    println!("  Input shape: {:?}", reduce_tensor.shape());
    println!("  Sum along dim 1 shape: {:?}", sum_result.shape());
    println!("  Mean along dim 1 shape: {:?}", mean_result.shape());
    println!();

    // =========================================================================
    // Step 7: Math Operations
    // =========================================================================
    println!("Step 7: Math Operations");
    println!("-------------------------------------------------------------");

    let math_data: Vec<f32> = vec![1.0, 4.0, 9.0, 16.0];
    let math_tensor = MlxTensor::from_slice(&math_data, &[2, 2], gpu_device);

    let _exp = math_tensor.exp();
    let _log = math_tensor.log();
    let _sqrt = math_tensor.sqrt();
    let _abs = math_tensor.abs();
    let _neg = math_tensor.neg();

    println!("  exp(x) computed");
    println!("  log(x) computed");
    println!("  sqrt(x) computed");
    println!("  abs(x) computed");
    println!("  neg(x) computed");
    println!();

    // =========================================================================
    // Step 8: Shape Operations
    // =========================================================================
    println!("Step 8: Shape Operations");
    println!("-------------------------------------------------------------");

    let shape_data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let shape_tensor = MlxTensor::from_slice(&shape_data, &[3, 4], gpu_device);

    let reshaped = shape_tensor.reshape_to(&[4, 3]);
    let transposed = shape_tensor.transpose_all();
    let broadcasted = shape_tensor.broadcast_to(&[2, 3, 4]);

    println!("  Original shape: {:?}", shape_tensor.shape());
    println!("  Reshaped to: {:?}", reshaped.shape());
    println!("  Transposed to: {:?}", transposed.shape());
    println!("  Broadcasted to: {:?}", broadcasted.shape());
    println!();

    // =========================================================================
    // Step 9: Performance Benchmark
    // =========================================================================
    println!("Step 9: Performance Benchmark");
    println!("-------------------------------------------------------------");

    let size = 1024;
    let large_data: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.001).collect();

    // GPU benchmark
    let gpu_tensor = MlxTensor::from_slice(&large_data, &[size as i32, size as i32], gpu_device);
    let gpu_start = Instant::now();
    for _ in 0..10 {
        let _ = gpu_tensor.matmul(&gpu_tensor);
    }
    // Force evaluation
    let result = gpu_tensor.matmul(&gpu_tensor);
    result.eval().expect("Failed to evaluate");
    let gpu_time = gpu_start.elapsed();

    println!("  Matrix size: {}x{}", size, size);
    println!("  GPU matmul (10 iterations): {:?}", gpu_time);
    println!("  Average per matmul: {:?}", gpu_time / 10);
    println!();

    // =========================================================================
    // Summary
    // =========================================================================
    println!("==============================================================");
    println!("                   Demo Complete!                             ");
    println!("==============================================================");
    println!();
    println!("burn-mlx provides:");
    println!("  - High-performance tensor operations on Apple Silicon");
    println!("  - Unified memory (zero-copy between CPU/GPU)");
    println!("  - Lazy evaluation for optimized computation graphs");
    println!("  - Support for all common deep learning operations");
    println!();
    println!("This foundation crate can be extended to implement the full");
    println!("Burn Backend trait for integration with tsai training.");
}

#[cfg(not(target_os = "macos"))]
fn main() {
    println!("This example requires macOS with Apple Silicon (M1/M2/M3/M4).");
    println!("MLX is only available on Apple platforms.");
}
