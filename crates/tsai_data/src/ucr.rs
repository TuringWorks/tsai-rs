//! UCR Time Series Archive dataset support.
//!
//! Provides functionality to download and load datasets from the
//! UCR Time Series Classification Archive.
//!
//! # Example
//!
//! ```rust,ignore
//! use tsai_data::ucr::{UCRDataset, list_datasets};
//!
//! // List available datasets
//! for name in list_datasets() {
//!     println!("{}", name);
//! }
//!
//! // Load a dataset
//! let dataset = UCRDataset::load("NATOPS", None)?;
//! println!("Train samples: {}", dataset.train.n_samples());
//! println!("Test samples: {}", dataset.test.n_samples());
//! ```

use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use crate::{DataError, Result, TSDataset};

/// Base URL for the UCR Archive.
/// URL for downloading UCR datasets as zip files.
const UCR_URL: &str = "https://timeseriesclassification.com/aeon-toolkit";

/// List of UCR Archive dataset names (subset of most popular ones).
pub const UCR_DATASETS: &[&str] = &[
    "ACSF1",
    "Adiac",
    "AllGestureWiimoteX",
    "AllGestureWiimoteY",
    "AllGestureWiimoteZ",
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "BME",
    "Car",
    "CBF",
    "Chinatown",
    "ChlorineConcentration",
    "CinCECGTorso",
    "Coffee",
    "Computers",
    "CricketX",
    "CricketY",
    "CricketZ",
    "Crop",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW",
    "DodgerLoopDay",
    "DodgerLoopGame",
    "DodgerLoopWeekend",
    "Earthquakes",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "ElectricDevices",
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "EthanolLevel",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "FiftyWords",
    "Fish",
    "FordA",
    "FordB",
    "FreezerRegularTrain",
    "FreezerSmallTrain",
    "Fungi",
    "GestureMidAirD1",
    "GestureMidAirD2",
    "GestureMidAirD3",
    "GesturePebbleZ1",
    "GesturePebbleZ2",
    "GunPoint",
    "GunPointAgeSpan",
    "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung",
    "Ham",
    "HandOutlines",
    "Haptics",
    "Herring",
    "HouseTwenty",
    "InlineSkate",
    "InsectEPGRegularTrain",
    "InsectEPGSmallTrain",
    "InsectWingbeatSound",
    "ItalyPowerDemand",
    "LargeKitchenAppliances",
    "Lightning2",
    "Lightning7",
    "Mallat",
    "Meat",
    "MedicalImages",
    "MelbournePedestrian",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxTW",
    "MixedShapesRegularTrain",
    "MixedShapesSmallTrain",
    "MoteStrain",
    "NATOPS",
    "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2",
    "OliveOil",
    "OSULeaf",
    "PhalangesOutlinesCorrect",
    "Phoneme",
    "PickupGestureWiimoteZ",
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
    "PLAID",
    "Plane",
    "PowerCons",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxTW",
    "RefrigerationDevices",
    "Rock",
    "ScreenType",
    "SemgHandGenderCh2",
    "SemgHandMovementCh2",
    "SemgHandSubjectCh2",
    "ShakeGestureWiimoteZ",
    "ShapeletSim",
    "ShapesAll",
    "SmallKitchenAppliances",
    "SmoothSubspace",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "StarLightCurves",
    "Strawberry",
    "SwedishLeaf",
    "Symbols",
    "SyntheticControl",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Trace",
    "TwoLeadECG",
    "TwoPatterns",
    "UMD",
    "UWaveGestureLibraryAll",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "Wafer",
    "Wine",
    "WordSynonyms",
    "Worms",
    "WormsTwoClass",
    "Yoga",
];

/// A loaded UCR dataset with train and test splits.
#[derive(Debug)]
pub struct UCRDataset {
    /// Dataset name.
    pub name: String,
    /// Training dataset.
    pub train: TSDataset,
    /// Test dataset.
    pub test: TSDataset,
    /// Number of classes.
    pub n_classes: usize,
    /// Sequence length.
    pub seq_len: usize,
}

impl UCRDataset {
    /// Load a UCR dataset.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the dataset (e.g., "NATOPS", "ECG200")
    /// * `cache_dir` - Optional cache directory. If None, uses default cache.
    ///
    /// # Returns
    ///
    /// The loaded dataset with train and test splits.
    pub fn load(name: &str, cache_dir: Option<PathBuf>) -> Result<Self> {
        let cache = cache_dir.unwrap_or_else(crate::cache_dir);
        let dataset_dir = cache.join("ucr").join(name);

        // Download if not cached
        if !dataset_dir.exists() {
            download_dataset(name, &dataset_dir)?;
        }

        // Load train and test
        let train_path = dataset_dir.join(format!("{}_TRAIN.tsv", name));
        let test_path = dataset_dir.join(format!("{}_TEST.tsv", name));

        let train = load_tsv(&train_path)?;
        let test = load_tsv(&test_path)?;

        // Determine metadata
        let n_classes = count_classes(&train, &test);
        let seq_len = train.seq_len();

        Ok(Self {
            name: name.to_string(),
            train,
            test,
            n_classes,
            seq_len,
        })
    }

    /// Check if a dataset is cached.
    pub fn is_cached(name: &str, cache_dir: Option<PathBuf>) -> bool {
        let cache = cache_dir.unwrap_or_else(crate::cache_dir);
        let dataset_dir = cache.join("ucr").join(name);
        dataset_dir.exists()
    }

    /// Delete cached dataset.
    pub fn clear_cache(name: &str, cache_dir: Option<PathBuf>) -> Result<()> {
        let cache = cache_dir.unwrap_or_else(crate::cache_dir);
        let dataset_dir = cache.join("ucr").join(name);
        if dataset_dir.exists() {
            fs::remove_dir_all(&dataset_dir).map_err(|e| DataError::Io(e.to_string()))?;
        }
        Ok(())
    }
}

/// List all available UCR datasets.
pub fn list_datasets() -> &'static [&'static str] {
    UCR_DATASETS
}

/// Check if a dataset name is valid.
pub fn is_valid_dataset(name: &str) -> bool {
    UCR_DATASETS.iter().any(|&d| d.eq_ignore_ascii_case(name))
}

/// Get dataset info without downloading.
#[derive(Debug, Clone)]
pub struct UCRDatasetInfo {
    /// Dataset name.
    pub name: String,
    /// Whether it's cached locally.
    pub is_cached: bool,
}

/// Get info about a dataset.
pub fn dataset_info(name: &str, cache_dir: Option<PathBuf>) -> Option<UCRDatasetInfo> {
    if !is_valid_dataset(name) {
        return None;
    }

    Some(UCRDatasetInfo {
        name: name.to_string(),
        is_cached: UCRDataset::is_cached(name, cache_dir),
    })
}

/// Download a UCR dataset.
fn download_dataset(name: &str, target_dir: &Path) -> Result<()> {
    // Validate dataset name
    if !is_valid_dataset(name) {
        return Err(DataError::InvalidInput(format!(
            "Unknown UCR dataset: {}",
            name
        )));
    }

    // Create target directory
    fs::create_dir_all(target_dir).map_err(|e| DataError::Io(e.to_string()))?;

    // Download zip file
    let zip_url = format!("{}/{}.zip", UCR_URL, name);
    let zip_path = target_dir.join(format!("{}.zip", name));

    download_file(&zip_url, &zip_path)?;

    // Extract zip file
    extract_zip(&zip_path, target_dir, name)?;

    // Clean up zip file
    let _ = fs::remove_file(&zip_path);

    Ok(())
}

/// Extract a zip file.
fn extract_zip(zip_path: &Path, target_dir: &Path, name: &str) -> Result<()> {
    let file = File::open(zip_path).map_err(|e| DataError::Io(e.to_string()))?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| DataError::Parse(format!("Invalid zip file: {}", e)))?;

    // Extract the train and test txt files
    let train_name = format!("{}_TRAIN.txt", name);
    let test_name = format!("{}_TEST.txt", name);

    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .map_err(|e| DataError::Parse(format!("Failed to read zip entry: {}", e)))?;

        let file_name = file.name().to_string();

        // Only extract the txt files we need
        if file_name == train_name || file_name == test_name {
            // Rename to .tsv for our loader
            let out_name = file_name.replace(".txt", ".tsv");
            let out_path = target_dir.join(&out_name);

            let mut out_file =
                File::create(&out_path).map_err(|e| DataError::Io(e.to_string()))?;
            std::io::copy(&mut file, &mut out_file)
                .map_err(|e| DataError::Io(e.to_string()))?;
        }
    }

    // Verify files were extracted
    let train_path = target_dir.join(format!("{}_TRAIN.tsv", name));
    let test_path = target_dir.join(format!("{}_TEST.tsv", name));

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
    // Use ureq for HTTP requests
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

/// Load a TSV file into a TSDataset.
///
/// UCR TSV format: first column is the class label, remaining columns are the time series.
fn load_tsv(path: &Path) -> Result<TSDataset> {
    let file = File::open(path).map_err(|e| DataError::Io(e.to_string()))?;
    let reader = BufReader::new(file);

    let mut x_data: Vec<Vec<f32>> = Vec::new();
    let mut y_data: Vec<f32> = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(|e| DataError::Io(e.to_string()))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Split by tab or whitespace
        let parts: Vec<&str> = line.split(['\t', ' ']).filter(|s| !s.is_empty()).collect();

        if parts.is_empty() {
            continue;
        }

        // First column is label
        let label: f32 = parts[0]
            .parse()
            .map_err(|_| DataError::Parse("Failed to parse label".to_string()))?;

        // Remaining columns are time series values
        let values: Vec<f32> = parts[1..]
            .iter()
            .map(|s| s.parse::<f32>().unwrap_or(f32::NAN))
            .collect();

        y_data.push(label);
        x_data.push(values);
    }

    if x_data.is_empty() {
        return Err(DataError::InvalidInput("Empty TSV file".to_string()));
    }

    // Convert to ndarray
    let n_samples = x_data.len();
    let seq_len = x_data[0].len();

    // Univariate time series: shape (n_samples, 1, seq_len)
    let mut x_array = ndarray::Array3::<f32>::zeros((n_samples, 1, seq_len));
    for (i, series) in x_data.iter().enumerate() {
        for (t, &val) in series.iter().enumerate() {
            x_array[[i, 0, t]] = val;
        }
    }

    // Labels: shape (n_samples, 1)
    let y_array = ndarray::Array2::from_shape_vec(
        (n_samples, 1),
        y_data.iter().map(|&l| remap_labels(&y_data, l)).collect(),
    )
    .map_err(|e| DataError::InvalidInput(e.to_string()))?;

    TSDataset::from_arrays(x_array, Some(y_array))
}

/// Remap labels to 0-indexed contiguous integers.
fn remap_labels(all_labels: &[f32], label: f32) -> f32 {
    let mut unique: Vec<i32> = all_labels.iter().map(|&l| l as i32).collect();
    unique.sort();
    unique.dedup();

    unique
        .iter()
        .position(|&l| l == label as i32)
        .unwrap_or(0) as f32
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
    fn test_list_datasets() {
        let datasets = list_datasets();
        assert!(!datasets.is_empty());
        assert!(datasets.contains(&"NATOPS"));
        assert!(datasets.contains(&"ECG200"));
        assert!(datasets.contains(&"FordA"));
    }

    #[test]
    fn test_is_valid_dataset() {
        assert!(is_valid_dataset("NATOPS"));
        assert!(is_valid_dataset("natops")); // Case insensitive
        assert!(is_valid_dataset("ECG200"));
        assert!(!is_valid_dataset("NonExistentDataset"));
    }

    #[test]
    fn test_dataset_info() {
        let info = dataset_info("NATOPS", None);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "NATOPS");

        let invalid = dataset_info("NotADataset", None);
        assert!(invalid.is_none());
    }

    #[test]
    fn test_remap_labels() {
        let labels = vec![1.0, 2.0, 1.0, 3.0, 2.0];
        assert_eq!(remap_labels(&labels, 1.0), 0.0);
        assert_eq!(remap_labels(&labels, 2.0), 1.0);
        assert_eq!(remap_labels(&labels, 3.0), 2.0);
    }
}
