//! I/O utilities for reading time series data.

use std::path::Path;

use ndarray::{Array2, Array3};

use crate::dataset::TSDataset;
use crate::error::{DataError, Result};

/// Read a time series dataset from a NumPy .npy file.
///
/// The file should contain a 3D array of shape (N, V, L).
///
/// # Arguments
///
/// * `path` - Path to the .npy file
///
/// # Returns
///
/// The loaded array.
pub fn read_npy<P: AsRef<Path>>(path: P) -> Result<Array3<f32>> {
    use ndarray_npy::ReadNpyExt;

    let file = std::fs::File::open(path.as_ref())?;
    let reader = std::io::BufReader::new(file);

    // Try reading as f32 first
    match Array3::<f32>::read_npy(reader) {
        Ok(arr) => Ok(arr),
        Err(e) => {
            // Try reading as f64 and converting
            let file = std::fs::File::open(path.as_ref())?;
            let reader = std::io::BufReader::new(file);
            let arr_f64: Array3<f64> = Array3::<f64>::read_npy(reader)
                .map_err(|_| DataError::FormatError(format!("Failed to read npy file: {}", e)))?;
            Ok(arr_f64.mapv(|x| x as f32))
        }
    }
}

/// Read time series data from a NumPy .npz archive.
///
/// Expects keys "x" and optionally "y" in the archive.
///
/// # Arguments
///
/// * `path` - Path to the .npz file
///
/// # Returns
///
/// A tuple of (x_array, optional_y_array).
pub fn read_npz<P: AsRef<Path>>(path: P) -> Result<(Array3<f32>, Option<Array2<f32>>)> {
    let file = std::fs::File::open(path.as_ref())?;
    let mut npz = ndarray_npy::NpzReader::new(file)
        .map_err(|e| DataError::FormatError(format!("Failed to read npz file: {}", e)))?;

    // Read x
    let x: Array3<f32> = npz
        .by_name("x")
        .map_err(|e| DataError::FormatError(format!("Failed to read 'x' from npz: {}", e)))?;

    // Try to read y
    let y: Option<Array2<f32>> = npz.by_name("y").ok();

    Ok((x, y))
}

/// Read time series data from a CSV file.
///
/// The CSV should have a header row and be structured as:
/// - First column: sample index or ID
/// - Remaining columns: time series values (flattened or per-variable)
///
/// # Arguments
///
/// * `path` - Path to the CSV file
/// * `n_vars` - Number of variables (channels)
/// * `seq_len` - Sequence length
/// * `has_labels` - Whether the last column contains labels
///
/// # Returns
///
/// A TSDataset.
#[cfg(feature = "polars-io")]
pub fn read_csv<P: AsRef<Path>>(
    path: P,
    n_vars: usize,
    seq_len: usize,
    has_labels: bool,
) -> Result<TSDataset> {
    use polars::prelude::*;

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(path.as_ref().to_path_buf()))
        .map_err(|e| DataError::FormatError(format!("Failed to create CSV reader: {}", e)))?
        .finish()
        .map_err(|e| DataError::FormatError(format!("Failed to read CSV: {}", e)))?;

    let n_rows = df.height();
    let n_cols = df.width();

    // Calculate expected columns
    let expected_data_cols = n_vars * seq_len;
    let total_expected = if has_labels {
        expected_data_cols + 1
    } else {
        expected_data_cols
    };

    if n_cols != total_expected {
        return Err(DataError::FormatError(format!(
            "Expected {} columns, got {}",
            total_expected, n_cols
        )));
    }

    // Extract data
    let mut x = Array3::<f32>::zeros((n_rows, n_vars, seq_len));
    let data_cols = if has_labels { n_cols - 1 } else { n_cols };

    for (col_idx, col) in df.get_columns()[..data_cols].iter().enumerate() {
        let var_idx = col_idx / seq_len;
        let step_idx = col_idx % seq_len;

        let values = col
            .cast(&DataType::Float32)
            .map_err(|e| DataError::FormatError(format!("Failed to cast column: {}", e)))?;
        let values = values
            .f32()
            .map_err(|e| DataError::FormatError(format!("Failed to get f32 values: {}", e)))?;

        for (row_idx, val) in values.into_iter().enumerate() {
            x[[row_idx, var_idx, step_idx]] = val.unwrap_or(0.0);
        }
    }

    // Extract labels if present
    let y = if has_labels {
        let label_col = &df.get_columns()[n_cols - 1];
        let labels = label_col
            .cast(&DataType::Float32)
            .map_err(|e| DataError::FormatError(format!("Failed to cast label column: {}", e)))?;
        let labels = labels
            .f32()
            .map_err(|e| DataError::FormatError(format!("Failed to get f32 labels: {}", e)))?;

        let mut y = Array2::<f32>::zeros((n_rows, 1));
        for (row_idx, val) in labels.into_iter().enumerate() {
            y[[row_idx, 0]] = val.unwrap_or(0.0);
        }
        Some(y)
    } else {
        None
    };

    TSDataset::from_arrays(x, y)
}

/// Read time series data from a Parquet file.
///
/// # Arguments
///
/// * `path` - Path to the Parquet file
/// * `x_columns` - Column names for input data
/// * `y_column` - Optional column name for labels
/// * `n_vars` - Number of variables
/// * `seq_len` - Sequence length
///
/// # Returns
///
/// A TSDataset.
#[cfg(feature = "polars-io")]
pub fn read_parquet<P: AsRef<Path>>(
    path: P,
    x_columns: &[&str],
    y_column: Option<&str>,
    n_vars: usize,
    seq_len: usize,
) -> Result<TSDataset> {
    use polars::prelude::*;

    let file = std::fs::File::open(path.as_ref())?;
    let df = ParquetReader::new(file)
        .finish()
        .map_err(|e| DataError::FormatError(format!("Failed to read Parquet: {}", e)))?;

    let n_rows = df.height();

    // Validate column count
    if x_columns.len() != n_vars * seq_len {
        return Err(DataError::FormatError(format!(
            "Expected {} x columns, got {}",
            n_vars * seq_len,
            x_columns.len()
        )));
    }

    // Extract data
    let mut x = Array3::<f32>::zeros((n_rows, n_vars, seq_len));

    for (col_idx, col_name) in x_columns.iter().enumerate() {
        let var_idx = col_idx / seq_len;
        let step_idx = col_idx % seq_len;

        let col = df
            .column(col_name)
            .map_err(|e| DataError::FormatError(format!("Column '{}' not found: {}", col_name, e)))?;
        let values = col
            .cast(&DataType::Float32)
            .map_err(|e| DataError::FormatError(format!("Failed to cast column: {}", e)))?;
        let values = values
            .f32()
            .map_err(|e| DataError::FormatError(format!("Failed to get f32 values: {}", e)))?;

        for (row_idx, val) in values.into_iter().enumerate() {
            x[[row_idx, var_idx, step_idx]] = val.unwrap_or(0.0);
        }
    }

    // Extract labels if present
    let y = if let Some(y_col) = y_column {
        let col = df
            .column(y_col)
            .map_err(|e| DataError::FormatError(format!("Column '{}' not found: {}", y_col, e)))?;
        let labels = col
            .cast(&DataType::Float32)
            .map_err(|e| DataError::FormatError(format!("Failed to cast label column: {}", e)))?;
        let labels = labels
            .f32()
            .map_err(|e| DataError::FormatError(format!("Failed to get f32 labels: {}", e)))?;

        let mut y = Array2::<f32>::zeros((n_rows, 1));
        for (row_idx, val) in labels.into_iter().enumerate() {
            y[[row_idx, 0]] = val.unwrap_or(0.0);
        }
        Some(y)
    } else {
        None
    };

    TSDataset::from_arrays(x, y)
}

#[cfg(test)]
mod tests {
    // I/O tests require actual files, so we'll test the data structures instead

    use super::*;

    #[test]
    fn test_dataset_creation() {
        let x = Array3::<f32>::zeros((100, 3, 50));
        let y = Array2::<f32>::zeros((100, 1));
        let ds = TSDataset::from_arrays(x, Some(y)).unwrap();

        assert_eq!(ds.len(), 100);
        assert_eq!(ds.n_vars(), 3);
        assert_eq!(ds.seq_len(), 50);
    }
}
