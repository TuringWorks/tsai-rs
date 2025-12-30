//! Benchmarks for device discovery and memory operations.
//!
//! Run with: cargo bench -p tsai_compute --bench device_ops

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tsai_compute::{
    Buffer, BufferUsage, ComputeBackend, CpuBackend, HardwareDiscovery,
    RoundRobinScheduler, Scheduler, SimpleScheduler, Workload, WorkloadScheduler,
};

fn bench_hardware_discovery(c: &mut Criterion) {
    let mut group = c.benchmark_group("discovery");

    group.bench_function("full_discovery", |bench| {
        bench.iter(|| {
            let pool = HardwareDiscovery::discover_all().unwrap();
            black_box(pool.device_count())
        })
    });

    group.bench_function("cpu_only", |bench| {
        bench.iter(|| {
            let devices = CpuBackend::enumerate_devices().unwrap();
            black_box(devices.len())
        })
    });

    // Test memoized discovery - first call initializes cache
    let _ = tsai_compute::get_device_pool();

    group.bench_function("cached_lookup", |bench| {
        bench.iter(|| {
            // This should be <50ns since it returns cached result
            let pool = tsai_compute::get_device_pool().unwrap();
            black_box(pool.device_count())
        })
    });

    group.finish();
}

fn bench_buffer_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_allocation");

    let devices = CpuBackend::enumerate_devices().unwrap();
    let backend = CpuBackend::new(&devices[0]).unwrap();

    for size in [1024usize, 4096, 16384, 65536, 262144, 1048576].iter() {
        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(BenchmarkId::new("allocate", size), size, |bench, &size| {
            bench.iter(|| {
                let buffer = backend.allocate_buffer(size, BufferUsage::HOST_VISIBLE).unwrap();
                black_box(buffer.size())
            })
        });
    }

    group.finish();
}

fn bench_buffer_map(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_map");

    let devices = CpuBackend::enumerate_devices().unwrap();
    let backend = CpuBackend::new(&devices[0]).unwrap();

    for size in [1024usize, 16384, 262144].iter() {
        let buffer = backend.allocate_buffer(*size, BufferUsage::HOST_VISIBLE).unwrap();

        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(BenchmarkId::new("map_write", size), size, |bench, &size| {
            let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
            bench.iter(|| {
                let mut mapping = buffer.map().unwrap();
                mapping.as_slice_mut().copy_from_slice(&data);
                black_box(())
            })
        });

        group.bench_with_input(BenchmarkId::new("map_read", size), size, |bench, _| {
            bench.iter(|| {
                let mapping = buffer.map().unwrap();
                let sum: u64 = mapping.as_slice().iter().map(|&b| b as u64).sum();
                black_box(sum)
            })
        });
    }

    group.finish();
}

fn bench_device_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("device_selection");

    let pool = HardwareDiscovery::discover_all().unwrap();

    group.bench_function("best_device", |bench| {
        bench.iter(|| black_box(pool.best_device()))
    });

    group.bench_function("cpu_devices", |bench| {
        bench.iter(|| black_box(pool.cpu_devices()))
    });

    let scheduler = WorkloadScheduler::new();
    let workload = Workload::new()
        .with_flops(1_000_000_000)
        .with_memory(1024 * 1024 * 100);

    group.bench_function("workload_scheduler", |bench| {
        bench.iter(|| {
            let device = scheduler.select_device(&pool, &workload);
            black_box(device)
        })
    });

    let simple = SimpleScheduler;
    group.bench_function("simple_scheduler", |bench| {
        bench.iter(|| {
            let device = simple.select_device(&pool, &workload);
            black_box(device)
        })
    });

    let round_robin = RoundRobinScheduler::new();
    group.bench_function("round_robin_scheduler", |bench| {
        bench.iter(|| {
            let device = round_robin.select_device(&pool, &workload);
            black_box(device)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_hardware_discovery,
    bench_buffer_allocation,
    bench_buffer_map,
    bench_device_selection,
);
criterion_main!(benches);
