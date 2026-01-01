//! UEA Multivariate Time Series Archive dataset support.
//!
//! Provides functionality to download and load datasets from the
//! UEA Multivariate Time Series Classification Archive.
//!
//! # Example
//!
//! ```rust,ignore
//! use tsai_data::uea::{UEADataset, list_uea_datasets};
//!
//! // List available datasets
//! for name in list_uea_datasets() {
//!     println!("{}", name);
//! }
//!
//! // Load a dataset
//! let dataset = UEADataset::load("BasicMotions", None)?;
//! println!("Train samples: {}", dataset.train.n_samples());
//! println!("Variables: {}", dataset.n_vars);
//! ```

use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use crate::{DataError, Result, TSDataset};

/// Base URL for the UEA Archive.
const UEA_URL: &str = "https://timeseriesclassification.com/aeon-toolkit";

/// List of UEA Multivariate Archive dataset names.
pub const UEA_DATASETS: &[&str] = &[
    "ArticularyWordRecognition",
    "AtrialFibrillation",
    "BasicMotions",
    "CharacterTrajectories",
    "Cricket",
    "DuckDuckGeese",
    "EigenWorms",
    "Epilepsy",
    "ERing",
    "EthanolConcentration",
    "FaceDetection",
    "FingerMovements",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    "InsectWingbeat",
    "JapaneseVowels",
    "Libras",
    "LSST",
    "MotorImagery",
    "NATOPS",
    "PEMS-SF",
    "PenDigits",
    "PhonemeSpectra",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "SpokenArabicDigits",
    "StandWalkJump",
    "UWaveGestureLibrary",
];

/// A loaded UEA dataset with train and test splits.
#[derive(Debug)]
pub struct UEADataset {
    /// Dataset name.
    pub name: String,
    /// Training dataset.
    pub train: TSDataset,
    /// Test dataset.
    pub test: TSDataset,
    /// Number of classes.
    pub n_classes: usize,
    /// Number of variables (channels).
    pub n_vars: usize,
    /// Sequence length (may vary for variable-length datasets).
    pub seq_len: usize,
    /// Whether this is a variable-length dataset.
    pub variable_length: bool,
}

impl UEADataset {
    /// Load a UEA dataset.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the dataset (e.g., "BasicMotions", "NATOPS")
    /// * `cache_dir` - Optional cache directory. If None, uses default cache.
    ///
    /// # Returns
    ///
    /// The loaded dataset with train and test splits.
    pub fn load(name: &str, cache_dir: Option<PathBuf>) -> Result<Self> {
        let cache = cache_dir.unwrap_or_else(crate::cache_dir);
        let dataset_dir = cache.join("uea").join(name);

        // Download if not cached
        if !dataset_dir.exists() {
            download_uea_dataset(name, &dataset_dir)?;
        }

        // Load train and test
        let train_path = dataset_dir.join(format!("{}_TRAIN.ts", name));
        let test_path = dataset_dir.join(format!("{}_TEST.ts", name));

        let (train, train_info) = load_ts_file(&train_path)?;
        let (test, _) = load_ts_file(&test_path)?;

        // Determine metadata
        let n_classes = count_classes(&train, &test);

        Ok(Self {
            name: name.to_string(),
            train,
            test,
            n_classes,
            n_vars: train_info.n_dims,
            seq_len: train_info.seq_len,
            variable_length: train_info.variable_length,
        })
    }

    /// Check if a dataset is cached.
    pub fn is_cached(name: &str, cache_dir: Option<PathBuf>) -> bool {
        let cache = cache_dir.unwrap_or_else(crate::cache_dir);
        let dataset_dir = cache.join("uea").join(name);
        dataset_dir.exists()
    }

    /// Delete cached dataset.
    pub fn clear_cache(name: &str, cache_dir: Option<PathBuf>) -> Result<()> {
        let cache = cache_dir.unwrap_or_else(crate::cache_dir);
        let dataset_dir = cache.join("uea").join(name);
        if dataset_dir.exists() {
            fs::remove_dir_all(&dataset_dir).map_err(|e| DataError::Io(e.to_string()))?;
        }
        Ok(())
    }
}

/// List all available UEA datasets.
pub fn list_uea_datasets() -> &'static [&'static str] {
    UEA_DATASETS
}

/// Check if a dataset name is valid.
pub fn is_valid_uea_dataset(name: &str) -> bool {
    UEA_DATASETS.iter().any(|&d| d.eq_ignore_ascii_case(name))
}

/// Get dataset info without downloading.
#[derive(Debug, Clone)]
pub struct UEADatasetInfo {
    /// Dataset name.
    pub name: String,
    /// Whether it's cached locally.
    pub is_cached: bool,
}

/// Get info about a dataset.
pub fn uea_dataset_info(name: &str, cache_dir: Option<PathBuf>) -> Option<UEADatasetInfo> {
    if !is_valid_uea_dataset(name) {
        return None;
    }

    Some(UEADatasetInfo {
        name: name.to_string(),
        is_cached: UEADataset::is_cached(name, cache_dir),
    })
}

/// Download a UEA dataset.
fn download_uea_dataset(name: &str, target_dir: &Path) -> Result<()> {
    // Validate dataset name
    if !is_valid_uea_dataset(name) {
        return Err(DataError::InvalidInput(format!(
            "Unknown UEA dataset: {}",
            name
        )));
    }

    // Create target directory
    fs::create_dir_all(target_dir).map_err(|e| DataError::Io(e.to_string()))?;

    // Download zip file
    let zip_url = format!("{}/{}.zip", UEA_URL, name);
    let zip_path = target_dir.join(format!("{}.zip", name));

    download_file(&zip_url, &zip_path)?;

    // Extract zip file
    extract_ts_zip(&zip_path, target_dir, name)?;

    // Clean up zip file
    let _ = fs::remove_file(&zip_path);

    Ok(())
}

/// Extract a zip file containing .ts files.
fn extract_ts_zip(zip_path: &Path, target_dir: &Path, name: &str) -> Result<()> {
    let file = File::open(zip_path).map_err(|e| DataError::Io(e.to_string()))?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| DataError::Parse(format!("Invalid zip file: {}", e)))?;

    // Extract the train and test .ts files
    let train_name = format!("{}_TRAIN.ts", name);
    let test_name = format!("{}_TEST.ts", name);

    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .map_err(|e| DataError::Parse(format!("Failed to read zip entry: {}", e)))?;

        let file_name = file.name().to_string();

        // Check if this is one of the files we want
        // The files might be in a subdirectory
        if file_name.ends_with(&train_name) || file_name.ends_with(&test_name) {
            let out_name = if file_name.ends_with(&train_name) {
                train_name.clone()
            } else {
                test_name.clone()
            };
            let out_path = target_dir.join(&out_name);

            let mut out_file =
                File::create(&out_path).map_err(|e| DataError::Io(e.to_string()))?;
            std::io::copy(&mut file, &mut out_file)
                .map_err(|e| DataError::Io(e.to_string()))?;
        }
    }

    // Verify files were extracted
    let train_path = target_dir.join(format!("{}_TRAIN.ts", name));
    let test_path = target_dir.join(format!("{}_TEST.ts", name));

    if !train_path.exists() || !test_path.exists() {
        return Err(DataError::Parse(format!(
            "Failed to extract dataset files for {}",
            name
        )));
    }

    Ok(())
}

/// Download a file from URL to path.
fn download_file(url: &str, path: &Path) -> Result<()> {
    let response = ureq::get(url)
        .call()
        .map_err(|e| DataError::Download(format!("Failed to download {}: {}", url, e)))?;

    if response.status() != 200 {
        return Err(DataError::Download(format!(
            "HTTP {} for {}",
            response.status(),
            url
        )));
    }

    let mut file = File::create(path).map_err(|e| DataError::Io(e.to_string()))?;

    let mut reader = response.into_reader();
    let mut buffer = Vec::new();
    std::io::Read::read_to_end(&mut reader, &mut buffer)
        .map_err(|e| DataError::Io(e.to_string()))?;

    file.write_all(&buffer)
        .map_err(|e| DataError::Io(e.to_string()))?;

    Ok(())
}

/// Metadata about a loaded .ts file.
#[derive(Debug)]
struct TSFileInfo {
    n_dims: usize,
    seq_len: usize,
    variable_length: bool,
}

/// Load a .ts (sktime/aeon format) file into a TSDataset.
///
/// The .ts format has a header section with @attributes and @data marker.
fn load_ts_file(path: &Path) -> Result<(TSDataset, TSFileInfo)> {
    let file = File::open(path).map_err(|e| DataError::Io(e.to_string()))?;
    let reader = BufReader::new(file);

    let mut in_data = false;
    let mut samples: Vec<(Vec<Vec<f32>>, String)> = Vec::new(); // (dimensions, class_label)

    for line in reader.lines() {
        let line = line.map_err(|e| DataError::Io(e.to_string()))?;
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Check for @data marker
        if line.to_lowercase().starts_with("@data") {
            in_data = true;
            continue;
        }

        // Skip header lines
        if !in_data {
            continue;
        }

        // Parse data line
        // Format: dim1_values:dim2_values:...:class_label
        // Each dimension is comma-separated values
        if let Some(sample) = parse_ts_line(line) {
            samples.push(sample);
        }
    }

    if samples.is_empty() {
        return Err(DataError::InvalidInput("Empty .ts file".to_string()));
    }

    // Determine dimensions and length
    let n_samples = samples.len();
    let n_dims = samples[0].0.len();

    // Check for variable length
    let first_len = samples[0].0[0].len();
    let variable_length = samples.iter().any(|(dims, _)| {
        dims.iter().any(|d| d.len() != first_len)
    });

    // Find max length for padding if variable length
    let max_len = samples
        .iter()
        .flat_map(|(dims, _)| dims.iter().map(|d| d.len()))
        .max()
        .unwrap_or(first_len);

    // Convert to arrays
    let mut x_array = ndarray::Array3::<f32>::zeros((n_samples, n_dims, max_len));
    let mut y_labels: Vec<String> = Vec::with_capacity(n_samples);

    for (i, (dims, label)) in samples.iter().enumerate() {
        for (d, dim_values) in dims.iter().enumerate() {
            for (t, &val) in dim_values.iter().enumerate() {
                x_array[[i, d, t]] = val;
            }
            // Pad with last value if shorter
            if dim_values.len() < max_len {
                let last_val = *dim_values.last().unwrap_or(&0.0);
                for t in dim_values.len()..max_len {
                    x_array[[i, d, t]] = last_val;
                }
            }
        }
        y_labels.push(label.clone());
    }

    // Convert string labels to numeric
    let unique_labels: Vec<String> = {
        let mut labels: Vec<String> = y_labels.iter().cloned().collect();
        labels.sort();
        labels.dedup();
        labels
    };

    let y_numeric: Vec<f32> = y_labels
        .iter()
        .map(|label| {
            unique_labels
                .iter()
                .position(|l| l == label)
                .unwrap_or(0) as f32
        })
        .collect();

    let y_array = ndarray::Array2::from_shape_vec((n_samples, 1), y_numeric)
        .map_err(|e| DataError::InvalidInput(e.to_string()))?;

    let info = TSFileInfo {
        n_dims,
        seq_len: max_len,
        variable_length,
    };

    Ok((TSDataset::from_arrays(x_array, Some(y_array))?, info))
}

/// Parse a single data line from a .ts file.
fn parse_ts_line(line: &str) -> Option<(Vec<Vec<f32>>, String)> {
    // Format: dim1_values:dim2_values:...:class_label
    // Values within a dimension are comma-separated
    let parts: Vec<&str> = line.split(':').collect();

    if parts.len() < 2 {
        return None;
    }

    // Last part is the class label
    let class_label = parts.last()?.trim().to_string();

    // All other parts are dimensions
    let dimensions: Vec<Vec<f32>> = parts[..parts.len() - 1]
        .iter()
        .map(|dim_str| {
            dim_str
                .split(',')
                .filter_map(|s| {
                    let s = s.trim();
                    if s.is_empty() || s == "?" {
                        Some(f32::NAN)
                    } else {
                        s.parse::<f32>().ok()
                    }
                })
                .collect()
        })
        .collect();

    if dimensions.is_empty() || dimensions.iter().all(|d| d.is_empty()) {
        return None;
    }

    Some((dimensions, class_label))
}

/// Count the number of unique classes.
fn count_classes(train: &TSDataset, test: &TSDataset) -> usize {
    let mut classes = std::collections::HashSet::new();

    if let Some(y) = train.y() {
        for &label in y.iter() {
            classes.insert(label as i32);
        }
    }
    if let Some(y) = test.y() {
        for &label in y.iter() {
            classes.insert(label as i32);
        }
    }

    classes.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_uea_datasets() {
        let datasets = list_uea_datasets();
        assert!(!datasets.is_empty());
        assert!(datasets.contains(&"BasicMotions"));
        assert!(datasets.contains(&"NATOPS"));
        assert!(datasets.contains(&"Heartbeat"));
    }

    #[test]
    fn test_is_valid_uea_dataset() {
        assert!(is_valid_uea_dataset("BasicMotions"));
        assert!(is_valid_uea_dataset("basicmotions")); // Case insensitive
        assert!(is_valid_uea_dataset("NATOPS"));
        assert!(!is_valid_uea_dataset("NonExistentDataset"));
    }

    #[test]
    fn test_uea_dataset_info() {
        let info = uea_dataset_info("BasicMotions", None);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "BasicMotions");

        let invalid = uea_dataset_info("NotADataset", None);
        assert!(invalid.is_none());
    }

    #[test]
    fn test_parse_ts_line() {
        let line = "1.0,2.0,3.0:4.0,5.0,6.0:class1";
        let result = parse_ts_line(line);
        assert!(result.is_some());

        let (dims, label) = result.unwrap();
        assert_eq!(dims.len(), 2);
        assert_eq!(dims[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(dims[1], vec![4.0, 5.0, 6.0]);
        assert_eq!(label, "class1");
    }

    #[test]
    fn test_parse_ts_line_with_missing() {
        let line = "1.0,?,3.0:4.0,5.0,6.0:classA";
        let result = parse_ts_line(line);
        assert!(result.is_some());

        let (dims, label) = result.unwrap();
        assert_eq!(dims.len(), 2);
        assert!(dims[0][1].is_nan());
        assert_eq!(label, "classA");
    }
}
