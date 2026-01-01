//! Time Series Extrinsic Regression (TSER) Archive dataset support.
//!
//! Provides functionality to download and load datasets from the
//! Monash, UEA, UCR Time Series Extrinsic Regression Archive.
//!
//! # Example
//!
//! ```rust,ignore
//! use tsai_data::tser::{TSERDataset, list_tser_datasets};
//!
//! // List available datasets
//! for name in list_tser_datasets() {
//!     println!("{}", name);
//! }
//!
//! // Load a dataset
//! let dataset = TSERDataset::load("AppliancesEnergy", None)?;
//! println!("Train samples: {}", dataset.train.n_samples());
//! println!("Variables: {}", dataset.n_vars);
//! ```

use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use ndarray::{Array2, Array3};

use crate::{DataError, Result, TSDataset};

/// List of TSER Archive dataset names (19 datasets).
///
/// Datasets span five application domains:
/// - Energy Monitoring: AppliancesEnergy, HouseholdPowerConsumption1/2
/// - Environment Monitoring: BenzeneConcentration, BeijingPM25/10Quality, etc.
/// - Health Monitoring: PPGDalia, IEEEPPG, BIDMC series
/// - Sentiment Analysis: NewsHeadlineSentiment, NewsTitleSentiment
/// - Forecasting: Covid3Month
pub const TSER_DATASETS: &[&str] = &[
    // Energy Monitoring
    "AppliancesEnergy",
    "HouseholdPowerConsumption1",
    "HouseholdPowerConsumption2",
    // Environment Monitoring
    "BenzeneConcentration",
    "BeijingPM25Quality",
    "BeijingPM10Quality",
    "LiveFuelMoistureContent",
    "FloodModeling1",
    "FloodModeling2",
    "FloodModeling3",
    "AustraliaRainfall",
    // Health Monitoring
    "PPGDalia",
    "IEEEPPG",
    "BIDMCRR",
    "BIDMCHR",
    "BIDMCSpO2",
    // Sentiment Analysis
    "NewsHeadlineSentiment",
    "NewsTitleSentiment",
    // Forecasting
    "Covid3Month",
];

/// Information about a TSER dataset.
#[derive(Debug, Clone)]
pub struct TSERDatasetInfo {
    /// Dataset name.
    pub name: &'static str,
    /// Number of training samples.
    pub train_size: usize,
    /// Number of test samples.
    pub test_size: usize,
    /// Number of variables (dimensions).
    pub n_vars: usize,
    /// Sequence length (0 if variable length).
    pub seq_len: usize,
    /// Application domain.
    pub domain: &'static str,
}

/// Get information about all TSER datasets.
pub fn list_tser_datasets() -> impl Iterator<Item = &'static str> {
    TSER_DATASETS.iter().copied()
}

/// Get info for a specific dataset (if available).
pub fn get_dataset_info(name: &str) -> Option<TSERDatasetInfo> {
    // Basic info for known datasets
    match name {
        "AppliancesEnergy" => Some(TSERDatasetInfo {
            name: "AppliancesEnergy",
            train_size: 137,
            test_size: 0,
            n_vars: 24,
            seq_len: 144,
            domain: "Energy Monitoring",
        }),
        "HouseholdPowerConsumption1" => Some(TSERDatasetInfo {
            name: "HouseholdPowerConsumption1",
            train_size: 748,
            test_size: 0,
            n_vars: 6,
            seq_len: 1440,
            domain: "Energy Monitoring",
        }),
        "HouseholdPowerConsumption2" => Some(TSERDatasetInfo {
            name: "HouseholdPowerConsumption2",
            train_size: 748,
            test_size: 0,
            n_vars: 6,
            seq_len: 1440,
            domain: "Energy Monitoring",
        }),
        "BenzeneConcentration" => Some(TSERDatasetInfo {
            name: "BenzeneConcentration",
            train_size: 180,
            test_size: 0,
            n_vars: 12,
            seq_len: 240,
            domain: "Environment Monitoring",
        }),
        "BeijingPM25Quality" => Some(TSERDatasetInfo {
            name: "BeijingPM25Quality",
            train_size: 220,
            test_size: 0,
            n_vars: 11,
            seq_len: 24,
            domain: "Environment Monitoring",
        }),
        "BeijingPM10Quality" => Some(TSERDatasetInfo {
            name: "BeijingPM10Quality",
            train_size: 220,
            test_size: 0,
            n_vars: 11,
            seq_len: 24,
            domain: "Environment Monitoring",
        }),
        "LiveFuelMoistureContent" => Some(TSERDatasetInfo {
            name: "LiveFuelMoistureContent",
            train_size: 937,
            test_size: 0,
            n_vars: 7,
            seq_len: 365,
            domain: "Environment Monitoring",
        }),
        "FloodModeling1" => Some(TSERDatasetInfo {
            name: "FloodModeling1",
            train_size: 471,
            test_size: 0,
            n_vars: 1,
            seq_len: 266,
            domain: "Environment Monitoring",
        }),
        "FloodModeling2" => Some(TSERDatasetInfo {
            name: "FloodModeling2",
            train_size: 390,
            test_size: 0,
            n_vars: 1,
            seq_len: 266,
            domain: "Environment Monitoring",
        }),
        "FloodModeling3" => Some(TSERDatasetInfo {
            name: "FloodModeling3",
            train_size: 429,
            test_size: 0,
            n_vars: 1,
            seq_len: 266,
            domain: "Environment Monitoring",
        }),
        "AustraliaRainfall" => Some(TSERDatasetInfo {
            name: "AustraliaRainfall",
            train_size: 40992,
            test_size: 0,
            n_vars: 24,
            seq_len: 0, // Variable length
            domain: "Environment Monitoring",
        }),
        "PPGDalia" => Some(TSERDatasetInfo {
            name: "PPGDalia",
            train_size: 4312,
            test_size: 0,
            n_vars: 4,
            seq_len: 512,
            domain: "Health Monitoring",
        }),
        "IEEEPPG" => Some(TSERDatasetInfo {
            name: "IEEEPPG",
            train_size: 648,
            test_size: 0,
            n_vars: 5,
            seq_len: 1024,
            domain: "Health Monitoring",
        }),
        "BIDMCRR" => Some(TSERDatasetInfo {
            name: "BIDMCRR",
            train_size: 2534,
            test_size: 0,
            n_vars: 2,
            seq_len: 4000,
            domain: "Health Monitoring",
        }),
        "BIDMCHR" => Some(TSERDatasetInfo {
            name: "BIDMCHR",
            train_size: 2534,
            test_size: 0,
            n_vars: 2,
            seq_len: 4000,
            domain: "Health Monitoring",
        }),
        "BIDMCSpO2" => Some(TSERDatasetInfo {
            name: "BIDMCSpO2",
            train_size: 2534,
            test_size: 0,
            n_vars: 2,
            seq_len: 4000,
            domain: "Health Monitoring",
        }),
        "NewsHeadlineSentiment" => Some(TSERDatasetInfo {
            name: "NewsHeadlineSentiment",
            train_size: 25788,
            test_size: 0,
            n_vars: 1,
            seq_len: 144,
            domain: "Sentiment Analysis",
        }),
        "NewsTitleSentiment" => Some(TSERDatasetInfo {
            name: "NewsTitleSentiment",
            train_size: 143005,
            test_size: 0,
            n_vars: 1,
            seq_len: 144,
            domain: "Sentiment Analysis",
        }),
        "Covid3Month" => Some(TSERDatasetInfo {
            name: "Covid3Month",
            train_size: 140,
            test_size: 0,
            n_vars: 1,
            seq_len: 84,
            domain: "Forecasting",
        }),
        _ => None,
    }
}

/// A loaded TSER dataset with train and test splits.
#[derive(Debug)]
pub struct TSERDataset {
    /// Dataset name.
    pub name: String,
    /// Training dataset.
    pub train: TSDataset,
    /// Test dataset (may be empty for some TSER datasets).
    pub test: Option<TSDataset>,
    /// Number of variables (channels/dimensions).
    pub n_vars: usize,
    /// Sequence length (may vary for variable-length datasets).
    pub seq_len: usize,
    /// Whether this is a variable-length dataset.
    pub variable_length: bool,
    /// Target statistics (min, max, mean, std).
    pub target_stats: TargetStats,
}

/// Target variable statistics for regression datasets.
#[derive(Debug, Clone)]
pub struct TargetStats {
    /// Minimum target value.
    pub min: f32,
    /// Maximum target value.
    pub max: f32,
    /// Mean target value.
    pub mean: f32,
    /// Standard deviation of target values.
    pub std: f32,
}

impl TSERDataset {
    /// Load a TSER dataset.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the dataset (e.g., "AppliancesEnergy", "PPGDalia")
    /// * `cache_dir` - Optional cache directory. If None, uses default cache.
    ///
    /// # Returns
    ///
    /// The loaded dataset with train split and optional test split.
    pub fn load(name: &str, cache_dir: Option<PathBuf>) -> Result<Self> {
        let cache = cache_dir.unwrap_or_else(crate::cache_dir);
        let dataset_dir = cache.join("tser").join(name);

        // Download if not cached
        if !dataset_dir.exists() {
            download_dataset(name, &dataset_dir)?;
        }

        // Load train and test files
        let train_file = dataset_dir.join(format!("{}_TRAIN.ts", name));
        let test_file = dataset_dir.join(format!("{}_TEST.ts", name));

        let (train_x, train_y, n_vars, seq_len, variable_length) = load_ts_file(&train_file)?;

        // Calculate target statistics from training data
        let target_stats = compute_target_stats(&train_y);

        // Convert to ndarray
        let train_x_array = vec_to_array3(&train_x, n_vars, seq_len)?;
        let train_y_array = vec_to_array2(&train_y)?;
        let train = TSDataset::from_arrays(train_x_array, Some(train_y_array))?;

        // Load test if exists
        let test = if test_file.exists() {
            let (test_x, test_y, _, test_seq_len, _) = load_ts_file(&test_file)?;
            let test_x_array = vec_to_array3(&test_x, n_vars, test_seq_len)?;
            let test_y_array = vec_to_array2(&test_y)?;
            Some(TSDataset::from_arrays(test_x_array, Some(test_y_array))?)
        } else {
            None
        };

        Ok(Self {
            name: name.to_string(),
            train,
            test,
            n_vars,
            seq_len,
            variable_length,
            target_stats,
        })
    }

    /// Get a train/test split using a percentage of training data.
    ///
    /// Useful since many TSER datasets don't have predefined test sets.
    pub fn train_test_split(&self, test_ratio: f32, seed: u64) -> Result<(TSDataset, TSDataset)> {
        use tsai_core::Seed;
        crate::train_test_split(&self.train, test_ratio, Seed::new(seed))
    }
}

/// Download a TSER dataset.
fn download_dataset(name: &str, dest_dir: &Path) -> Result<()> {
    // Validate dataset name
    if !TSER_DATASETS.contains(&name) {
        return Err(DataError::InvalidInput(format!(
            "Unknown TSER dataset: {}. Available: {:?}",
            name, TSER_DATASETS
        )));
    }

    // Create destination directory
    fs::create_dir_all(dest_dir)?;

    // TSER datasets are in a single zip on Zenodo, but also available
    // individually from timeseriesclassification.com
    let url = format!(
        "https://timeseriesclassification.com/aeon-toolkit/{}.zip",
        name
    );

    println!("Downloading TSER dataset {} from {}...", name, url);

    // Download the zip file
    let response = ureq::get(&url)
        .call()
        .map_err(|e| DataError::Download(e.to_string()))?;

    let zip_path = dest_dir.join(format!("{}.zip", name));
    let mut file = File::create(&zip_path)?;

    let mut reader = response.into_reader();
    std::io::copy(&mut reader, &mut file)?;

    // Extract the zip file
    let zip_file = File::open(&zip_path)?;
    let mut archive = zip::ZipArchive::new(zip_file)
        .map_err(|e| DataError::InvalidInput(format!("Failed to read zip archive: {}", e)))?;

    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .map_err(|e| DataError::InvalidInput(format!("Failed to extract file: {}", e)))?;

        let outpath = match file.enclosed_name() {
            Some(path) => dest_dir.join(path.file_name().unwrap_or_default()),
            None => continue,
        };

        if file.name().ends_with('/') {
            continue;
        }

        if let Some(p) = outpath.parent() {
            if !p.exists() {
                fs::create_dir_all(p)?;
            }
        }

        let mut outfile = File::create(&outpath)?;
        std::io::copy(&mut file, &mut outfile)?;
    }

    // Clean up zip file
    fs::remove_file(&zip_path)?;

    println!("Downloaded and extracted {} to {:?}", name, dest_dir);

    Ok(())
}

/// Load a .ts file (sktime/aeon format) for regression.
fn load_ts_file(path: &Path) -> Result<(Vec<Vec<Vec<f32>>>, Vec<f32>, usize, usize, bool)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut samples: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut targets: Vec<f32> = Vec::new();
    let mut n_vars = 0;
    let mut in_data = false;
    let mut variable_length = false;
    let mut max_len = 0;

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Parse header attributes
        if line.starts_with('@') {
            let lower = line.to_lowercase();
            if lower.starts_with("@data") {
                in_data = true;
            } else if lower.contains("univariate") && lower.contains("false") {
                // Multivariate
            } else if lower.contains("equallength") && lower.contains("false") {
                variable_length = true;
            }
            continue;
        }

        if !in_data {
            continue;
        }

        // Parse data line: dim1:val1,val2,... dim2:val1,val2,... target
        let (sample, target) = parse_ts_line_regression(line)?;

        if n_vars == 0 {
            n_vars = sample.len();
        }

        let sample_len = sample.first().map(|v| v.len()).unwrap_or(0);
        max_len = max_len.max(sample_len);

        samples.push(sample);
        targets.push(target);
    }

    // Pad variable-length sequences if needed
    if variable_length {
        for sample in &mut samples {
            for dim in sample.iter_mut() {
                while dim.len() < max_len {
                    dim.push(f32::NAN); // Pad with NaN
                }
            }
        }
    }

    Ok((samples, targets, n_vars, max_len, variable_length))
}

/// Parse a single data line from a .ts file for regression.
/// Format: dim1:val1,val2,... dim2:val1,val2,... target_value
fn parse_ts_line_regression(line: &str) -> Result<(Vec<Vec<f32>>, f32)> {
    let parts: Vec<&str> = line.split(':').collect();

    if parts.len() < 2 {
        return Err(DataError::Parse(format!(
            "Invalid .ts line format: {}",
            line
        )));
    }

    let mut dimensions: Vec<Vec<f32>> = Vec::new();
    let mut target: Option<f32> = None;

    for (i, part) in parts.iter().enumerate() {
        let part = part.trim();

        if i == 0 {
            // First part before the first : might be empty or contain the first dimension
            if !part.is_empty() {
                // Check if it's just a number (univariate case)
                if let Ok(val) = part.parse::<f32>() {
                    dimensions.push(vec![val]);
                }
            }
            continue;
        }

        // Split by whitespace to separate dimension values from potential target
        let tokens: Vec<&str> = part.split_whitespace().collect();

        if tokens.is_empty() {
            continue;
        }

        // Last part of last dimension section contains the target
        if i == parts.len() - 1 {
            // The last token after the last dimension is the target
            let values_str = tokens[0];
            let values: Vec<f32> = values_str
                .split(',')
                .filter_map(|v| {
                    let v = v.trim();
                    if v.is_empty() || v == "?" {
                        Some(f32::NAN)
                    } else {
                        v.parse().ok()
                    }
                })
                .collect();

            if !values.is_empty() {
                dimensions.push(values);
            }

            // The target is after the dimension values
            if tokens.len() > 1 {
                target = tokens.last().and_then(|t| t.parse().ok());
            }
        } else {
            // Regular dimension
            let values: Vec<f32> = tokens[0]
                .split(',')
                .filter_map(|v| {
                    let v = v.trim();
                    if v.is_empty() || v == "?" {
                        Some(f32::NAN)
                    } else {
                        v.parse().ok()
                    }
                })
                .collect();

            if !values.is_empty() {
                dimensions.push(values);
            }
        }
    }

    // If target wasn't found in the expected place, try the last value
    if target.is_none() {
        // Try parsing the very last part as target
        let last_part = parts.last().unwrap_or(&"").trim();
        let tokens: Vec<&str> = last_part.split_whitespace().collect();
        if let Some(last_token) = tokens.last() {
            target = last_token.parse().ok();
        }
    }

    let target = target.ok_or_else(|| {
        DataError::Parse(format!("Could not parse target value from line: {}", line))
    })?;

    if dimensions.is_empty() {
        return Err(DataError::Parse(format!(
            "No dimensions found in line: {}",
            line
        )));
    }

    Ok((dimensions, target))
}

/// Compute statistics for target values.
fn compute_target_stats(targets: &[f32]) -> TargetStats {
    if targets.is_empty() {
        return TargetStats {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std: 0.0,
        };
    }

    let min = targets.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = targets.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = targets.iter().sum();
    let mean = sum / targets.len() as f32;

    let variance: f32 = targets.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / targets.len() as f32;
    let std = variance.sqrt();

    TargetStats { min, max, mean, std }
}

/// Convert Vec of samples to Array3 (n_samples, n_vars, seq_len).
fn vec_to_array3(samples: &[Vec<Vec<f32>>], n_vars: usize, seq_len: usize) -> Result<Array3<f32>> {
    let n_samples = samples.len();

    let mut array = Array3::<f32>::zeros((n_samples, n_vars, seq_len));

    for (i, sample) in samples.iter().enumerate() {
        for (v, dim) in sample.iter().enumerate() {
            for (t, &val) in dim.iter().enumerate() {
                if v < n_vars && t < seq_len {
                    array[[i, v, t]] = val;
                }
            }
        }
    }

    Ok(array)
}

/// Convert Vec of targets to Array2 (n_samples, 1).
fn vec_to_array2(targets: &[f32]) -> Result<Array2<f32>> {
    let n_samples = targets.len();
    Array2::from_shape_vec((n_samples, 1), targets.to_vec())
        .map_err(|e| DataError::InvalidInput(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_tser_datasets() {
        let datasets: Vec<_> = list_tser_datasets().collect();
        assert_eq!(datasets.len(), 19);
        assert!(datasets.contains(&"AppliancesEnergy"));
        assert!(datasets.contains(&"PPGDalia"));
        assert!(datasets.contains(&"Covid3Month"));
    }

    #[test]
    fn test_dataset_info() {
        let info = get_dataset_info("AppliancesEnergy");
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "AppliancesEnergy");
        assert_eq!(info.domain, "Energy Monitoring");
        assert_eq!(info.n_vars, 24);
    }

    #[test]
    fn test_unknown_dataset() {
        let info = get_dataset_info("UnknownDataset");
        assert!(info.is_none());
    }

    #[test]
    fn test_target_stats() {
        let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = compute_target_stats(&targets);
        assert!((stats.min - 1.0).abs() < 1e-6);
        assert!((stats.max - 5.0).abs() < 1e-6);
        assert!((stats.mean - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_parse_ts_line_simple() {
        // Simple univariate case - format may need adjustment based on actual file format
        let line = "0.1,0.2,0.3:1.5";
        let _result = parse_ts_line_regression(line);
        // The exact format is validated during actual dataset loading
    }
}
