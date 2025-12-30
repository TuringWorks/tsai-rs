//! Benchmarks comparing SIMD vs scalar operations.
//!
//! Run with: cargo bench -p tsai_compute --bench simd_ops

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tsai_compute::backend::cpu::simd::ops;

fn bench_add_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_f32");

    for size in [1024, 4096, 16384, 65536, 262144].iter() {
        let a: Vec<f32> = (0..*size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..*size).map(|i| i as f32 * 0.5).collect();
        let mut out = vec![0.0f32; *size];

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("simd", size), size, |bench, _| {
            bench.iter(|| {
                ops::add_f32_avx2(black_box(&a), black_box(&b), black_box(&mut out));
                black_box(out[0])
            })
        });

        group.bench_with_input(BenchmarkId::new("scalar", size), size, |bench, _| {
            bench.iter(|| {
                ops::add_f32_scalar(black_box(&a), black_box(&b), black_box(&mut out));
                black_box(out[0])
            })
        });
    }

    group.finish();
}

fn bench_mul_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul_f32");

    for size in [1024, 4096, 16384, 65536, 262144].iter() {
        let a: Vec<f32> = (0..*size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..*size).map(|i| (i as f32 + 1.0).recip()).collect();
        let mut out = vec![0.0f32; *size];

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("simd", size), size, |bench, _| {
            bench.iter(|| {
                ops::mul_f32_avx2(black_box(&a), black_box(&b), black_box(&mut out));
                black_box(out[0])
            })
        });

        group.bench_with_input(BenchmarkId::new("scalar", size), size, |bench, _| {
            bench.iter(|| {
                ops::mul_f32_scalar(black_box(&a), black_box(&b), black_box(&mut out));
                black_box(out[0])
            })
        });
    }

    group.finish();
}

fn bench_sum_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_f32");

    for size in [1024, 4096, 16384, 65536, 262144].iter() {
        let data: Vec<f32> = (0..*size).map(|i| (i as f32).sin()).collect();

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("simd", size), size, |bench, _| {
            bench.iter(|| ops::sum_f32_avx2(black_box(&data)))
        });

        group.bench_with_input(BenchmarkId::new("scalar", size), size, |bench, _| {
            bench.iter(|| ops::sum_f32_scalar(black_box(&data)))
        });
    }

    group.finish();
}

fn bench_dot_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_f32");

    for size in [1024, 4096, 16384, 65536, 262144].iter() {
        let a: Vec<f32> = (0..*size).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..*size).map(|i| (i as f32).cos()).collect();

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("simd", size), size, |bench, _| {
            bench.iter(|| ops::dot_f32_avx2(black_box(&a), black_box(&b)))
        });

        group.bench_with_input(BenchmarkId::new("scalar", size), size, |bench, _| {
            bench.iter(|| ops::dot_f32_scalar(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_add_f32,
    bench_mul_f32,
    bench_sum_f32,
    bench_dot_f32,
);
criterion_main!(benches);
