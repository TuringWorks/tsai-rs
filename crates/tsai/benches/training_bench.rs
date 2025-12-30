//! Benchmarks for training performance.
//!
//! Run with: cargo bench --bench training_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use burn::prelude::*;
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;
use ndarray::{Array2, Array3};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use tsai_data::{TSDataLoaders, TSDataset};
use tsai_models::{InceptionTimePlus, InceptionTimePlusConfig};
use tsai_train::{ClassificationTrainer, ClassificationTrainerConfig};

type TrainBackend = Autodiff<NdArray>;

/// Create synthetic time series data for benchmarking.
fn create_synthetic_data(
    n_samples: usize,
    n_vars: usize,
    seq_len: usize,
    n_classes: usize,
) -> (Array3<f32>, Array2<f32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let mut x_data = Vec::with_capacity(n_samples * n_vars * seq_len);
    let mut y_data = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let class = i % n_classes;
        y_data.push(class as f32);

        for _v in 0..n_vars {
            for t in 0..seq_len {
                let value = (class as f32) * 0.5 + (t as f32 / seq_len as f32) + rng.gen::<f32>() * 0.1;
                x_data.push(value);
            }
        }
    }

    let x = Array3::from_shape_vec((n_samples, n_vars, seq_len), x_data).unwrap();
    let y = Array2::from_shape_vec((n_samples, 1), y_data).unwrap();

    (x, y)
}

fn bench_model_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_forward");

    let device = <TrainBackend as Backend>::Device::default();

    for batch_size in [8, 16, 32, 64].iter() {
        // Create model
        let config = InceptionTimePlusConfig {
            n_vars: 3,
            seq_len: 100,
            n_classes: 5,
            n_blocks: 6,
            n_filters: 32,
            kernel_sizes: [9, 19, 39],
            bottleneck_dim: 32,
            dropout: 0.0,
        };
        let model: InceptionTimePlus<TrainBackend> = config.init(&device);

        // Create input tensor
        let (x, _) = create_synthetic_data(*batch_size, 3, 100, 5);
        let data: Vec<f32> = x.iter().copied().collect();
        let tensor_data = burn::tensor::TensorData::new(data, [*batch_size, 3, 100]);
        let tensor: Tensor<TrainBackend, 3> = Tensor::from_data(tensor_data, &device);

        group.bench_with_input(
            BenchmarkId::new("InceptionTimePlus", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    let output = model.forward(black_box(tensor.clone()));
                    black_box(output)
                })
            },
        );
    }

    group.finish();
}

fn bench_dataset_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataset_iteration");

    let device = <NdArray as Backend>::Device::default();

    for n_samples in [100, 500, 1000].iter() {
        let (x, y) = create_synthetic_data(*n_samples, 3, 100, 5);

        // Split into train/valid
        let train_samples = *n_samples * 4 / 5;
        let x_train = x.slice(ndarray::s![..train_samples, .., ..]).to_owned();
        let y_train = y.slice(ndarray::s![..train_samples, ..]).to_owned();
        let x_valid = x.slice(ndarray::s![train_samples.., .., ..]).to_owned();
        let y_valid = y.slice(ndarray::s![train_samples.., ..]).to_owned();

        let train_ds = TSDataset::from_arrays(x_train, Some(y_train)).unwrap();
        let valid_ds = TSDataset::from_arrays(x_valid, Some(y_valid)).unwrap();

        let dls = TSDataLoaders::builder(train_ds, valid_ds)
            .batch_size(32)
            .build()
            .unwrap();

        group.bench_with_input(
            BenchmarkId::new("train_iteration", n_samples),
            n_samples,
            |b, _| {
                b.iter(|| {
                    let mut count = 0;
                    for batch_result in dls.train().iter::<NdArray>(&device) {
                        let _batch = batch_result.unwrap();
                        count += 1;
                    }
                    black_box(count)
                })
            },
        );
    }

    group.finish();
}

fn bench_single_training_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("training_step");
    group.sample_size(10); // Fewer samples due to expensive training

    let device = <TrainBackend as Backend>::Device::default();

    // Create small dataset
    let (x, y) = create_synthetic_data(64, 3, 100, 5);
    let train_ds = TSDataset::from_arrays(x.clone(), Some(y.clone())).unwrap();
    let valid_ds = TSDataset::from_arrays(x, Some(y)).unwrap();

    let dls = TSDataLoaders::builder(train_ds, valid_ds)
        .batch_size(16)
        .build()
        .unwrap();

    // Create model
    let config = InceptionTimePlusConfig {
        n_vars: 3,
        seq_len: 100,
        n_classes: 5,
        n_blocks: 2, // Smaller for benchmarking
        n_filters: 16,
        kernel_sizes: [9, 19, 39],
        bottleneck_dim: 16,
        dropout: 0.0,
    };
    let model: InceptionTimePlus<TrainBackend> = config.init(&device);

    // Configure trainer for 1 epoch
    let trainer_config = ClassificationTrainerConfig {
        n_epochs: 1,
        lr: 1e-3,
        weight_decay: 0.01,
        grad_clip: 1.0,
        verbose: false,
        early_stopping_patience: 0,
        early_stopping_min_delta: 0.001,
    };

    group.bench_function("single_epoch", |b| {
        b.iter(|| {
            let trainer = ClassificationTrainer::<TrainBackend>::new(
                trainer_config.clone(),
                device.clone(),
            );
            let model_clone = model.clone();
            let result = trainer.fit_with_forward(
                model_clone,
                &dls,
                |m, x| m.forward(x),
                |m, x| m.forward(x),
            );
            black_box(result.unwrap())
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_model_forward,
    bench_dataset_iteration,
    bench_single_training_step,
);
criterion_main!(benches);
