//! Monash Time Series Forecasting Repository dataset support.
//!
//! Provides functionality to download and load datasets from the
//! Monash Time Series Forecasting Archive for time series forecasting tasks.
//!
//! # Example
//!
//! ```rust,ignore
//! use tsai_data::forecasting::{ForecastingDataset, list_forecasting_datasets};
//!
//! // List available datasets
//! for name in list_forecasting_datasets() {
//!     println!("{}", name);
//! }
//!
//! // Load a dataset
//! let dataset = ForecastingDataset::load("nn5_daily", None)?;
//! println!("Number of series: {}", dataset.n_series);
//! println!("Forecast horizon: {}", dataset.forecast_horizon);
//! ```

use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use ndarray::{Array2, Array3};

use crate::{DataError, Result, TSDataset};

/// List of Monash Forecasting Archive dataset names.
///
/// Datasets span multiple domains including:
/// - Competition: M1, M3, M4, Tourism, NN5, CIF
/// - Energy: Electricity, Solar, Wind, London Smart Meters
/// - Traffic: San Francisco Traffic, Pedestrian Counts, Vehicle Trips
/// - Nature: Weather, COVID Deaths, Sunspot, River Flow
/// - Economic: FRED-MD, Bitcoin
/// - Sales: Dominick, Car Parts
pub const FORECASTING_DATASETS: &[&str] = &[
    // Competition datasets
    "m1_yearly",
    "m1_quarterly",
    "m1_monthly",
    "m3_yearly",
    "m3_quarterly",
    "m3_monthly",
    "m3_other",
    "m4_yearly",
    "m4_quarterly",
    "m4_monthly",
    "m4_weekly",
    "m4_daily",
    "m4_hourly",
    "tourism_yearly",
    "tourism_quarterly",
    "tourism_monthly",
    "nn5_daily",
    "nn5_weekly",
    "cif_2016",
    // Energy
    "electricity_hourly",
    "electricity_weekly",
    "solar_10_minutes",
    "solar_weekly",
    "wind_farms_minutely",
    "london_smart_meters",
    "australian_electricity_demand",
    "solar_4_seconds",
    "wind_4_seconds",
    // Traffic & Transport
    "traffic_hourly",
    "traffic_weekly",
    "pedestrian_counts",
    "vehicle_trips",
    "rideshare",
    // Nature & Weather
    "weather",
    "temperature_rain",
    "covid_deaths",
    "sunspot",
    "saugeenday",
    "us_births",
    "kdd_cup_2018",
    // Economic & Sales
    "fred_md",
    "bitcoin",
    "dominick",
    "car_parts",
    "hospital",
    // Web
    "kaggle_web_traffic_daily",
    "kaggle_web_traffic_weekly",
];

/// Time series frequency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Frequency {
    /// Secondly (S)
    Secondly,
    /// Minutely (T)
    Minutely,
    /// Hourly (H)
    Hourly,
    /// Daily (D)
    Daily,
    /// Weekly (W)
    Weekly,
    /// Monthly (M)
    Monthly,
    /// Quarterly (Q)
    Quarterly,
    /// Yearly (Y)
    Yearly,
    /// Unknown or variable
    Unknown,
}

impl Frequency {
    /// Parse frequency from string.
    pub fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "S" | "SECONDLY" => Self::Secondly,
            "T" | "MINUTELY" | "MIN" => Self::Minutely,
            "H" | "HOURLY" => Self::Hourly,
            "D" | "DAILY" => Self::Daily,
            "W" | "WEEKLY" => Self::Weekly,
            "M" | "MONTHLY" => Self::Monthly,
            "Q" | "QUARTERLY" => Self::Quarterly,
            "Y" | "YEARLY" | "A" | "ANNUAL" => Self::Yearly,
            _ => Self::Unknown,
        }
    }

    /// Get default forecast horizon for this frequency.
    pub fn default_horizon(&self) -> usize {
        match self {
            Self::Secondly => 60,
            Self::Minutely => 60,
            Self::Hourly => 48,
            Self::Daily => 30,
            Self::Weekly => 8,
            Self::Monthly => 12,
            Self::Quarterly => 4,
            Self::Yearly => 4,
            Self::Unknown => 10,
        }
    }
}

/// Information about a forecasting dataset.
#[derive(Debug, Clone)]
pub struct ForecastingDatasetInfo {
    /// Dataset name.
    pub name: &'static str,
    /// Number of time series in the dataset.
    pub n_series: usize,
    /// Typical series length.
    pub series_length: &'static str,
    /// Data frequency.
    pub frequency: &'static str,
    /// Domain/category.
    pub domain: &'static str,
    /// Whether multivariate.
    pub multivariate: bool,
}

/// Get list of all available forecasting datasets.
pub fn list_forecasting_datasets() -> impl Iterator<Item = &'static str> {
    FORECASTING_DATASETS.iter().copied()
}

/// Get info for a specific dataset.
pub fn get_dataset_info(name: &str) -> Option<ForecastingDatasetInfo> {
    match name {
        "m1_yearly" => Some(ForecastingDatasetInfo {
            name: "m1_yearly",
            n_series: 181,
            series_length: "15-58",
            frequency: "Yearly",
            domain: "Competition",
            multivariate: false,
        }),
        "m3_monthly" => Some(ForecastingDatasetInfo {
            name: "m3_monthly",
            n_series: 1428,
            series_length: "66-144",
            frequency: "Monthly",
            domain: "Competition",
            multivariate: false,
        }),
        "m4_daily" => Some(ForecastingDatasetInfo {
            name: "m4_daily",
            n_series: 4227,
            series_length: "107-9933",
            frequency: "Daily",
            domain: "Competition",
            multivariate: false,
        }),
        "nn5_daily" => Some(ForecastingDatasetInfo {
            name: "nn5_daily",
            n_series: 111,
            series_length: "791",
            frequency: "Daily",
            domain: "Banking",
            multivariate: false,
        }),
        "electricity_hourly" => Some(ForecastingDatasetInfo {
            name: "electricity_hourly",
            n_series: 321,
            series_length: "26304",
            frequency: "Hourly",
            domain: "Energy",
            multivariate: false,
        }),
        "traffic_hourly" => Some(ForecastingDatasetInfo {
            name: "traffic_hourly",
            n_series: 862,
            series_length: "17544",
            frequency: "Hourly",
            domain: "Transport",
            multivariate: false,
        }),
        "weather" => Some(ForecastingDatasetInfo {
            name: "weather",
            n_series: 3010,
            series_length: "1332-65981",
            frequency: "Daily",
            domain: "Nature",
            multivariate: false,
        }),
        "tourism_monthly" => Some(ForecastingDatasetInfo {
            name: "tourism_monthly",
            n_series: 366,
            series_length: "91-333",
            frequency: "Monthly",
            domain: "Tourism",
            multivariate: false,
        }),
        "covid_deaths" => Some(ForecastingDatasetInfo {
            name: "covid_deaths",
            n_series: 266,
            series_length: "212",
            frequency: "Daily",
            domain: "Health",
            multivariate: false,
        }),
        "fred_md" => Some(ForecastingDatasetInfo {
            name: "fred_md",
            n_series: 107,
            series_length: "728",
            frequency: "Monthly",
            domain: "Economic",
            multivariate: false,
        }),
        "rideshare" => Some(ForecastingDatasetInfo {
            name: "rideshare",
            n_series: 156,
            series_length: "541",
            frequency: "Hourly",
            domain: "Transport",
            multivariate: true,
        }),
        _ => None,
    }
}

/// A single time series from a forecasting dataset.
#[derive(Debug, Clone)]
pub struct TimeSeries {
    /// Series identifier/name.
    pub id: String,
    /// Time series values.
    pub values: Vec<f32>,
    /// Start timestamp (if available).
    pub start: Option<String>,
    /// Series frequency.
    pub frequency: Frequency,
}

/// A loaded forecasting dataset.
#[derive(Debug)]
pub struct ForecastingDataset {
    /// Dataset name.
    pub name: String,
    /// All time series in the dataset.
    pub series: Vec<TimeSeries>,
    /// Number of series.
    pub n_series: usize,
    /// Minimum series length.
    pub min_length: usize,
    /// Maximum series length.
    pub max_length: usize,
    /// Default forecast horizon.
    pub forecast_horizon: usize,
    /// Data frequency.
    pub frequency: Frequency,
    /// Whether dataset has missing values.
    pub has_missing: bool,
    /// Whether multivariate.
    pub multivariate: bool,
}

impl ForecastingDataset {
    /// Load a forecasting dataset.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the dataset (e.g., "nn5_daily", "electricity_hourly")
    /// * `cache_dir` - Optional cache directory. If None, uses default cache.
    ///
    /// # Returns
    ///
    /// The loaded dataset with all time series.
    pub fn load(name: &str, cache_dir: Option<PathBuf>) -> Result<Self> {
        let cache = cache_dir.unwrap_or_else(crate::cache_dir);
        let dataset_dir = cache.join("forecasting").join(name);

        // Download if not cached
        if !dataset_dir.exists() {
            download_dataset(name, &dataset_dir)?;
        }

        // Find and load .tsf file
        let tsf_file = find_tsf_file(&dataset_dir)?;
        load_tsf_file(&tsf_file, name)
    }

    /// Create train/test split for forecasting.
    ///
    /// For each series, the last `horizon` values are used as test.
    ///
    /// # Arguments
    ///
    /// * `horizon` - Number of steps to forecast (test set size per series)
    ///
    /// # Returns
    ///
    /// Tuple of (train_dataset, test_dataset) as TSDataset.
    pub fn train_test_split(&self, horizon: Option<usize>) -> Result<(TSDataset, TSDataset)> {
        let h = horizon.unwrap_or(self.forecast_horizon);

        let mut train_data: Vec<Vec<f32>> = Vec::new();
        let mut test_data: Vec<Vec<f32>> = Vec::new();

        for series in &self.series {
            if series.values.len() > h {
                let split_point = series.values.len() - h;
                train_data.push(series.values[..split_point].to_vec());
                test_data.push(series.values[split_point..].to_vec());
            } else {
                // Series too short, use all for training
                train_data.push(series.values.clone());
                test_data.push(vec![f32::NAN; h]);
            }
        }

        // Pad to uniform length
        let train_max_len = train_data.iter().map(|s| s.len()).max().unwrap_or(0);
        let test_max_len = test_data.iter().map(|s| s.len()).max().unwrap_or(h);

        // Convert to arrays
        let train_x = series_to_array3(&train_data, train_max_len)?;
        let test_x = series_to_array3(&test_data, test_max_len)?;

        // No labels for forecasting - use dummy
        let train_y = Array2::<f32>::zeros((train_data.len(), 1));
        let test_y = Array2::<f32>::zeros((test_data.len(), 1));

        let train = TSDataset::from_arrays(train_x, Some(train_y))?;
        let test = TSDataset::from_arrays(test_x, Some(test_y))?;

        Ok((train, test))
    }

    /// Create sliding window samples for training.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Size of input window
    /// * `horizon` - Size of forecast horizon (target)
    /// * `stride` - Step size between windows (default: 1)
    ///
    /// # Returns
    ///
    /// TSDataset where X is input windows and Y is target values.
    pub fn create_windows(
        &self,
        window_size: usize,
        horizon: usize,
        stride: Option<usize>,
    ) -> Result<TSDataset> {
        let stride = stride.unwrap_or(1);
        let mut x_data: Vec<Vec<f32>> = Vec::new();
        let mut y_data: Vec<Vec<f32>> = Vec::new();

        for series in &self.series {
            let total_len = window_size + horizon;
            if series.values.len() >= total_len {
                let mut i = 0;
                while i + total_len <= series.values.len() {
                    let x_window = series.values[i..i + window_size].to_vec();
                    let y_window = series.values[i + window_size..i + total_len].to_vec();
                    x_data.push(x_window);
                    y_data.push(y_window);
                    i += stride;
                }
            }
        }

        if x_data.is_empty() {
            return Err(DataError::InvalidInput(
                "No valid windows could be created. Try smaller window_size or horizon.".to_string(),
            ));
        }

        // Convert to arrays
        let x_array = series_to_array3(&x_data, window_size)?;
        let y_array = series_to_array2(&y_data, horizon)?;

        TSDataset::from_arrays(x_array, Some(y_array))
    }

    /// Get a single series by index.
    pub fn get_series(&self, idx: usize) -> Option<&TimeSeries> {
        self.series.get(idx)
    }

    /// Get a single series by ID.
    pub fn get_series_by_id(&self, id: &str) -> Option<&TimeSeries> {
        self.series.iter().find(|s| s.id == id)
    }
}

/// Download a forecasting dataset.
fn download_dataset(name: &str, dest_dir: &Path) -> Result<()> {
    // Map dataset name to Zenodo URL
    let zenodo_id = get_zenodo_id(name)?;

    fs::create_dir_all(dest_dir)?;

    let url = format!(
        "https://zenodo.org/record/{}/files/{}.zip?download=1",
        zenodo_id, name
    );

    println!("Downloading forecasting dataset {} from Zenodo...", name);

    // Try direct download from timeseriesclassification.com first
    let alt_url = format!(
        "https://forecastingdata.org/files/{}.zip",
        name
    );

    let response = ureq::get(&alt_url)
        .call()
        .or_else(|_| ureq::get(&url).call())
        .map_err(|e| DataError::Download(e.to_string()))?;

    let zip_path = dest_dir.join(format!("{}.zip", name));
    let mut file = File::create(&zip_path)?;

    let mut reader = response.into_reader();
    std::io::copy(&mut reader, &mut file)?;

    // Extract
    let zip_file = File::open(&zip_path)?;
    let mut archive = zip::ZipArchive::new(zip_file)
        .map_err(|e| DataError::InvalidInput(format!("Failed to read zip: {}", e)))?;

    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .map_err(|e| DataError::InvalidInput(format!("Failed to extract: {}", e)))?;

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

    fs::remove_file(&zip_path)?;
    println!("Downloaded and extracted {} to {:?}", name, dest_dir);

    Ok(())
}

/// Get Zenodo record ID for a dataset.
fn get_zenodo_id(name: &str) -> Result<&'static str> {
    // Zenodo record IDs for Monash forecasting datasets
    let id = match name {
        "m1_yearly" | "m1_quarterly" | "m1_monthly" => "4656193",
        "m3_yearly" | "m3_quarterly" | "m3_monthly" | "m3_other" => "4656298",
        "m4_yearly" | "m4_quarterly" | "m4_monthly" | "m4_weekly" | "m4_daily" | "m4_hourly" => "4656410",
        "tourism_yearly" | "tourism_quarterly" | "tourism_monthly" => "4656103",
        "nn5_daily" | "nn5_weekly" => "4656125",
        "cif_2016" => "4656042",
        "electricity_hourly" | "electricity_weekly" => "4656140",
        "solar_10_minutes" | "solar_weekly" => "4656144",
        "traffic_hourly" | "traffic_weekly" => "4656132",
        "weather" => "4654822",
        "covid_deaths" => "4656009",
        "fred_md" => "4654833",
        _ => {
            return Err(DataError::InvalidInput(format!(
                "Unknown dataset: {}. Available: {:?}",
                name, FORECASTING_DATASETS
            )));
        }
    };
    Ok(id)
}

/// Find the .tsf file in a directory.
fn find_tsf_file(dir: &Path) -> Result<PathBuf> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map_or(false, |ext| ext == "tsf") {
            return Ok(path);
        }
    }
    Err(DataError::InvalidInput(format!(
        "No .tsf file found in {:?}",
        dir
    )))
}

/// Load a .tsf file (Time Series Forecasting format).
fn load_tsf_file(path: &Path, name: &str) -> Result<ForecastingDataset> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut series: Vec<TimeSeries> = Vec::new();
    let mut frequency = Frequency::Unknown;
    let mut forecast_horizon: Option<usize> = None;
    let mut has_missing = false;
    let mut in_data = false;

    // Parse metadata and data

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
                continue;
            }

            // Parse attribute
            if let Some(attr_line) = line.strip_prefix('@') {
                let parts: Vec<&str> = attr_line.splitn(2, ' ').collect();
                if parts.len() >= 2 {
                    let attr_name = parts[0].to_lowercase();
                    let attr_value = parts[1].trim().to_string();

                    match attr_name.as_str() {
                        "frequency" => frequency = Frequency::from_str(&attr_value),
                        "horizon" | "forecast_horizon" => {
                            forecast_horizon = attr_value.parse().ok();
                        }
                        "missing" => has_missing = attr_value.to_lowercase() == "true",
                        _ => {}
                    }
                }
            }
            continue;
        }

        if !in_data {
            continue;
        }

        // Parse data line
        // Format: series_id:value1,value2,value3,...
        // Or with timestamp: series_id|timestamp:value1,value2,...
        if let Some((id_part, values_part)) = line.split_once(':') {
            let (series_id, start) = if id_part.contains('|') {
                let parts: Vec<&str> = id_part.split('|').collect();
                (parts[0].to_string(), Some(parts.get(1).unwrap_or(&"").to_string()))
            } else {
                (id_part.to_string(), None)
            };

            let values: Vec<f32> = values_part
                .split(',')
                .filter_map(|v| {
                    let v = v.trim();
                    if v.is_empty() || v == "?" || v.to_lowercase() == "nan" {
                        Some(f32::NAN)
                    } else {
                        v.parse().ok()
                    }
                })
                .collect();

            if !values.is_empty() {
                series.push(TimeSeries {
                    id: series_id,
                    values,
                    start,
                    frequency,
                });
            }
        }
    }

    if series.is_empty() {
        return Err(DataError::InvalidInput(format!(
            "No time series found in {:?}",
            path
        )));
    }

    let n_series = series.len();
    let min_length = series.iter().map(|s| s.values.len()).min().unwrap_or(0);
    let max_length = series.iter().map(|s| s.values.len()).max().unwrap_or(0);
    let horizon = forecast_horizon.unwrap_or_else(|| frequency.default_horizon());

    Ok(ForecastingDataset {
        name: name.to_string(),
        series,
        n_series,
        min_length,
        max_length,
        forecast_horizon: horizon,
        frequency,
        has_missing,
        multivariate: false, // Determined from data structure
    })
}

/// Convert series data to Array3 (n_series, 1, seq_len).
fn series_to_array3(data: &[Vec<f32>], max_len: usize) -> Result<Array3<f32>> {
    let n_series = data.len();
    let mut array = Array3::<f32>::from_elem((n_series, 1, max_len), f32::NAN);

    for (i, series) in data.iter().enumerate() {
        for (t, &val) in series.iter().enumerate() {
            if t < max_len {
                array[[i, 0, t]] = val;
            }
        }
    }

    Ok(array)
}

/// Convert targets to Array2 (n_samples, horizon).
fn series_to_array2(data: &[Vec<f32>], horizon: usize) -> Result<Array2<f32>> {
    let n_samples = data.len();
    let mut array = Array2::<f32>::from_elem((n_samples, horizon), f32::NAN);

    for (i, series) in data.iter().enumerate() {
        for (t, &val) in series.iter().enumerate() {
            if t < horizon {
                array[[i, t]] = val;
            }
        }
    }

    Ok(array)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_forecasting_datasets() {
        let datasets: Vec<_> = list_forecasting_datasets().collect();
        assert!(datasets.len() >= 40);
        assert!(datasets.contains(&"nn5_daily"));
        assert!(datasets.contains(&"electricity_hourly"));
        assert!(datasets.contains(&"weather"));
    }

    #[test]
    fn test_frequency_parsing() {
        assert_eq!(Frequency::from_str("D"), Frequency::Daily);
        assert_eq!(Frequency::from_str("H"), Frequency::Hourly);
        assert_eq!(Frequency::from_str("M"), Frequency::Monthly);
        assert_eq!(Frequency::from_str("Y"), Frequency::Yearly);
    }

    #[test]
    fn test_default_horizons() {
        assert_eq!(Frequency::Daily.default_horizon(), 30);
        assert_eq!(Frequency::Hourly.default_horizon(), 48);
        assert_eq!(Frequency::Monthly.default_horizon(), 12);
        assert_eq!(Frequency::Yearly.default_horizon(), 4);
    }

    #[test]
    fn test_dataset_info() {
        let info = get_dataset_info("nn5_daily");
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.n_series, 111);
        assert_eq!(info.domain, "Banking");
    }

    #[test]
    fn test_series_to_array3() {
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0],
        ];
        let array = series_to_array3(&data, 4).unwrap();
        assert_eq!(array.shape(), &[2, 1, 4]);
        assert_eq!(array[[0, 0, 0]], 1.0);
        assert_eq!(array[[0, 0, 2]], 3.0);
        assert!(array[[0, 0, 3]].is_nan());
        assert_eq!(array[[1, 0, 0]], 4.0);
        assert!(array[[1, 0, 2]].is_nan());
    }
}
