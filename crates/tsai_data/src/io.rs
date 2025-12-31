//! I/O utilities for reading time series data.

use std::fs::File;
use std::io::BufReader;
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
/// - First column: sample index or ID (ignored)
/// - Remaining columns: time series values (flattened as var0_t0, var0_t1, ..., var1_t0, ...)
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
pub fn read_csv<P: AsRef<Path>>(
    path: P,
    n_vars: usize,
    seq_len: usize,
    has_labels: bool,
) -> Result<TSDataset> {
    let file = File::open(path.as_ref())?;
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(BufReader::new(file));

    // Calculate expected columns
    let expected_data_cols = n_vars * seq_len;

    // Collect all records
    let mut rows: Vec<Vec<f32>> = Vec::new();
    let mut labels: Vec<f32> = Vec::new();

    for result in reader.records() {
        let record = result.map_err(|e| DataError::FormatError(format!("CSV parse error: {}", e)))?;

        let n_cols = record.len();
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

        // Parse data columns
        let data_cols = if has_labels { n_cols - 1 } else { n_cols };
        let mut row_data: Vec<f32> = Vec::with_capacity(data_cols);

        for i in 0..data_cols {
            let val: f32 = record.get(i)
                .ok_or_else(|| DataError::FormatError("Missing column".to_string()))?
                .parse()
                .unwrap_or(0.0);
            row_data.push(val);
        }
        rows.push(row_data);

        // Parse label if present
        if has_labels {
            let label: f32 = record.get(n_cols - 1)
                .ok_or_else(|| DataError::FormatError("Missing label column".to_string()))?
                .parse()
                .unwrap_or(0.0);
            labels.push(label);
        }
    }

    let n_samples = rows.len();
    if n_samples == 0 {
        return Err(DataError::InvalidInput("Empty CSV file".to_string()));
    }

    // Build the 3D array
    let mut x = Array3::<f32>::zeros((n_samples, n_vars, seq_len));
    for (row_idx, row_data) in rows.iter().enumerate() {
        for (col_idx, &val) in row_data.iter().enumerate() {
            let var_idx = col_idx / seq_len;
            let step_idx = col_idx % seq_len;
            x[[row_idx, var_idx, step_idx]] = val;
        }
    }

    // Build labels array if present
    let y = if has_labels {
        let mut y_arr = Array2::<f32>::zeros((n_samples, 1));
        for (i, &label) in labels.iter().enumerate() {
            y_arr[[i, 0]] = label;
        }
        Some(y_arr)
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
/// * `x_columns` - Column names for input data (in order: var0_t0, var0_t1, ..., var1_t0, ...)
/// * `y_column` - Optional column name for labels
/// * `n_vars` - Number of variables
/// * `seq_len` - Sequence length
///
/// # Returns
///
/// A TSDataset.
pub fn read_parquet<P: AsRef<Path>>(
    path: P,
    x_columns: &[&str],
    y_column: Option<&str>,
    n_vars: usize,
    seq_len: usize,
) -> Result<TSDataset> {
    use arrow_array::cast::AsArray;
    use arrow_array::types::Float32Type;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    // Validate column count
    if x_columns.len() != n_vars * seq_len {
        return Err(DataError::FormatError(format!(
            "Expected {} x columns, got {}",
            n_vars * seq_len,
            x_columns.len()
        )));
    }

    let file = File::open(path.as_ref())?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| DataError::FormatError(format!("Failed to read Parquet: {}", e)))?;

    let reader = builder.build()
        .map_err(|e| DataError::FormatError(format!("Failed to build Parquet reader: {}", e)))?;

    // Collect all batches
    let mut all_x_data: Vec<Vec<f32>> = Vec::new();
    let mut all_y_data: Vec<f32> = Vec::new();
    let mut n_samples = 0;

    for batch_result in reader {
        let batch = batch_result
            .map_err(|e| DataError::FormatError(format!("Failed to read batch: {}", e)))?;

        let batch_size = batch.num_rows();
        n_samples += batch_size;

        // Initialize storage for this batch
        let mut batch_x: Vec<Vec<f32>> = vec![vec![0.0; x_columns.len()]; batch_size];

        // Extract x columns
        for (col_idx, col_name) in x_columns.iter().enumerate() {
            let col = batch.column_by_name(col_name)
                .ok_or_else(|| DataError::FormatError(format!("Column '{}' not found", col_name)))?;

            // Try to cast to Float32Array
            let float_arr = col.as_primitive_opt::<Float32Type>()
                .ok_or_else(|| DataError::FormatError(format!(
                    "Column '{}' is not a float type", col_name
                )))?;

            for (row_idx, val) in float_arr.iter().enumerate() {
                batch_x[row_idx][col_idx] = val.unwrap_or(0.0);
            }
        }

        all_x_data.extend(batch_x);

        // Extract y column if present
        if let Some(y_col) = y_column {
            let col = batch.column_by_name(y_col)
                .ok_or_else(|| DataError::FormatError(format!("Column '{}' not found", y_col)))?;

            let float_arr = col.as_primitive_opt::<Float32Type>()
                .ok_or_else(|| DataError::FormatError(format!(
                    "Column '{}' is not a float type", y_col
                )))?;

            for val in float_arr.iter() {
                all_y_data.push(val.unwrap_or(0.0));
            }
        }
    }

    if n_samples == 0 {
        return Err(DataError::InvalidInput("Empty Parquet file".to_string()));
    }

    // Build the 3D array
    let mut x = Array3::<f32>::zeros((n_samples, n_vars, seq_len));
    for (row_idx, row_data) in all_x_data.iter().enumerate() {
        for (col_idx, &val) in row_data.iter().enumerate() {
            let var_idx = col_idx / seq_len;
            let step_idx = col_idx % seq_len;
            x[[row_idx, var_idx, step_idx]] = val;
        }
    }

    // Build labels array if present
    let y = if y_column.is_some() && !all_y_data.is_empty() {
        let mut y_arr = Array2::<f32>::zeros((n_samples, 1));
        for (i, &label) in all_y_data.iter().enumerate() {
            y_arr[[i, 0]] = label;
        }
        Some(y_arr)
    } else {
        None
    };

    TSDataset::from_arrays(x, y)
}

#[cfg(test)]
mod tests {
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
