//! Example: Time Series Feature Extraction
//!
//! This example demonstrates how to use tsai-rs's tsfresh-style feature extraction
//! to compute statistical features from time series data.
//!
//! Run with: cargo run --example feature_extraction

use tsai_analysis::{extract_features, extract_multivariate_features, FeatureExtractor, FeatureSet};

fn main() {
    println!("=== Time Series Feature Extraction ===\n");

    // Generate sample time series data
    let series: Vec<f32> = (0..100)
        .map(|i| {
            let t = i as f32 / 10.0;
            (t * 0.5).sin() + 0.1 * (t * 3.0).cos() + 0.05 * rand_f32()
        })
        .collect();

    println!("Input series: {} data points", series.len());
    println!("First 10 values: {:?}\n", &series[..10]);

    // Extract features with different feature sets
    println!("--- Minimal Feature Set (6 features) ---");
    let minimal = extract_features(&series, FeatureSet::Minimal);
    for (name, value) in &minimal {
        println!("  {}: {:.4}", name, value);
    }

    println!("\n--- Efficient Feature Set (15 features) ---");
    let efficient = extract_features(&series, FeatureSet::Efficient);
    println!("  Features extracted: {}", efficient.len());
    let mut sorted: Vec<_> = efficient.iter().collect();
    sorted.sort_by_key(|(k, _)| *k);
    for (name, value) in sorted.iter().take(10) {
        println!("  {}: {:.4}", name, value);
    }
    println!("  ... and {} more", efficient.len() - 10);

    println!("\n--- Comprehensive Feature Set (30+ features) ---");
    let comprehensive = extract_features(&series, FeatureSet::Comprehensive);
    println!("  Features extracted: {}", comprehensive.len());

    println!("\n--- All Features (50+ features) ---");
    let all = extract_features(&series, FeatureSet::All);
    println!("  Features extracted: {}", all.len());

    // Using FeatureExtractor for batch processing
    println!("\n--- Batch Feature Extraction ---");
    let mut extractor = FeatureExtractor::new(FeatureSet::Efficient);

    let batch: Vec<Vec<f32>> = (0..5)
        .map(|_| {
            (0..100)
                .map(|i| (i as f32 / 10.0).sin() + 0.1 * rand_f32())
                .collect()
        })
        .collect();

    let (batch_features, feature_names) = extractor.transform(&batch);
    println!("  Processed {} time series", batch_features.len());
    println!("  Features per series: {}", feature_names.len());

    // Multivariate feature extraction
    println!("\n--- Multivariate Feature Extraction ---");
    let multivariate: Vec<Vec<f32>> = (0..3)
        .map(|ch| {
            (0..100)
                .map(|i| {
                    let t = i as f32 / 10.0;
                    (t * (ch as f32 + 1.0) * 0.3).sin()
                })
                .collect()
        })
        .collect();

    let multi_features = extract_multivariate_features(&multivariate, FeatureSet::Efficient);
    println!("  Input: {} channels x {} timesteps", multivariate.len(), multivariate[0].len());
    println!("  Features extracted: {}", multi_features.len());

    // Show some cross-channel features
    let cross_features: Vec<_> = multi_features
        .iter()
        .filter(|(k, _)| k.contains("cross") || k.contains("corr"))
        .collect();
    println!("  Cross-channel features:");
    for (name, value) in cross_features.iter().take(5) {
        println!("    {}: {:.4}", name, value);
    }

    println!("\n=== Feature Extraction Complete ===");
}

// Simple random number generator for the example
fn rand_f32() -> f32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    ((nanos % 1000) as f32 / 1000.0) - 0.5
}
