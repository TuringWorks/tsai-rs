//! Hyperparameter optimization utilities.
//!
//! This module provides tools for hyperparameter search:
//! - [`GridSearch`] for exhaustive grid search
//! - [`RandomSearch`] for random sampling
//! - [`HyperparameterSpace`] for defining search spaces
//!
//! # Example
//!
//! ```rust,ignore
//! use tsai_train::hpo::{GridSearch, HyperparameterSpace, ParamValue, SearchResult};
//!
//! let mut space = HyperparameterSpace::new();
//! space.add_float("learning_rate", &[1e-4, 1e-3, 1e-2]);
//! space.add_int("batch_size", &[16, 32, 64]);
//! space.add_categorical("optimizer", &["adam", "sgd"]);
//!
//! let search = GridSearch::new(space);
//! let result = search.run(|params| {
//!     // Train model with these params and return validation score
//!     let lr = params.get_float("learning_rate")?;
//!     let bs = params.get_int("batch_size")?;
//!     // ... train and evaluate ...
//!     Ok(0.95) // return validation metric
//! })?;
//!
//! println!("Best params: {:?}", result.best_params);
//! println!("Best score: {}", result.best_score);
//! ```

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tsai_core::Seed;

/// A hyperparameter value that can be of different types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParamValue {
    /// Floating point value (for learning rates, dropout, etc.)
    Float(f64),
    /// Integer value (for batch size, hidden units, etc.)
    Int(i64),
    /// Boolean value (for flags, enable/disable features)
    Bool(bool),
    /// Categorical value (for optimizer names, activation functions, etc.)
    Categorical(String),
}

impl ParamValue {
    /// Get the value as a float, if it is one.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ParamValue::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Get the value as an integer, if it is one.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            ParamValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Get the value as a boolean, if it is one.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ParamValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Get the value as a categorical string, if it is one.
    pub fn as_categorical(&self) -> Option<&str> {
        match self {
            ParamValue::Categorical(v) => Some(v),
            _ => None,
        }
    }
}

/// A set of hyperparameters for a single trial.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParamSet {
    params: HashMap<String, ParamValue>,
}

impl ParamSet {
    /// Create an empty parameter set.
    pub fn new() -> Self {
        Self {
            params: HashMap::new(),
        }
    }

    /// Insert a parameter value.
    pub fn insert(&mut self, name: impl Into<String>, value: ParamValue) {
        self.params.insert(name.into(), value);
    }

    /// Get a parameter value.
    pub fn get(&self, name: &str) -> Option<&ParamValue> {
        self.params.get(name)
    }

    /// Get a float parameter, returning an error if not found or wrong type.
    pub fn get_float(&self, name: &str) -> Result<f64, HpoError> {
        self.get(name)
            .ok_or_else(|| HpoError::ParamNotFound(name.to_string()))?
            .as_float()
            .ok_or_else(|| HpoError::TypeMismatch {
                name: name.to_string(),
                expected: "float".to_string(),
            })
    }

    /// Get an integer parameter, returning an error if not found or wrong type.
    pub fn get_int(&self, name: &str) -> Result<i64, HpoError> {
        self.get(name)
            .ok_or_else(|| HpoError::ParamNotFound(name.to_string()))?
            .as_int()
            .ok_or_else(|| HpoError::TypeMismatch {
                name: name.to_string(),
                expected: "int".to_string(),
            })
    }

    /// Get a boolean parameter, returning an error if not found or wrong type.
    pub fn get_bool(&self, name: &str) -> Result<bool, HpoError> {
        self.get(name)
            .ok_or_else(|| HpoError::ParamNotFound(name.to_string()))?
            .as_bool()
            .ok_or_else(|| HpoError::TypeMismatch {
                name: name.to_string(),
                expected: "bool".to_string(),
            })
    }

    /// Get a categorical parameter, returning an error if not found or wrong type.
    pub fn get_categorical(&self, name: &str) -> Result<String, HpoError> {
        self.get(name)
            .ok_or_else(|| HpoError::ParamNotFound(name.to_string()))?
            .as_categorical()
            .map(String::from)
            .ok_or_else(|| HpoError::TypeMismatch {
                name: name.to_string(),
                expected: "categorical".to_string(),
            })
    }

    /// Iterate over all parameters.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &ParamValue)> {
        self.params.iter()
    }
}

/// Definition of a single hyperparameter with its possible values.
#[derive(Debug, Clone)]
pub enum ParamDef {
    /// Float parameter with list of values to try.
    Float(Vec<f64>),
    /// Float parameter with range (min, max) for random sampling.
    FloatRange { min: f64, max: f64, log_scale: bool },
    /// Integer parameter with list of values to try.
    Int(Vec<i64>),
    /// Integer parameter with range (min, max) for random sampling.
    IntRange { min: i64, max: i64 },
    /// Boolean parameter (tries both true and false).
    Bool,
    /// Categorical parameter with list of options.
    Categorical(Vec<String>),
}

impl ParamDef {
    /// Get all discrete values for grid search.
    fn values(&self) -> Vec<ParamValue> {
        match self {
            ParamDef::Float(vals) => vals.iter().map(|&v| ParamValue::Float(v)).collect(),
            ParamDef::FloatRange { min, max, .. } => {
                // For grid search, sample 5 points from range
                (0..5)
                    .map(|i| {
                        let t = i as f64 / 4.0;
                        ParamValue::Float(*min + t * (*max - *min))
                    })
                    .collect()
            }
            ParamDef::Int(vals) => vals.iter().map(|&v| ParamValue::Int(v)).collect(),
            ParamDef::IntRange { min, max } => (*min..=*max).map(ParamValue::Int).collect(),
            ParamDef::Bool => vec![ParamValue::Bool(false), ParamValue::Bool(true)],
            ParamDef::Categorical(opts) => {
                opts.iter().map(|s| ParamValue::Categorical(s.clone())).collect()
            }
        }
    }

    /// Sample a random value.
    fn sample(&self, rng: &mut ChaCha8Rng) -> ParamValue {
        match self {
            ParamDef::Float(vals) => {
                let idx = rng.gen_range(0..vals.len());
                ParamValue::Float(vals[idx])
            }
            ParamDef::FloatRange { min, max, log_scale } => {
                let val = if *log_scale {
                    let log_min = min.ln();
                    let log_max = max.ln();
                    (log_min + rng.gen::<f64>() * (log_max - log_min)).exp()
                } else {
                    *min + rng.gen::<f64>() * (*max - *min)
                };
                ParamValue::Float(val)
            }
            ParamDef::Int(vals) => {
                let idx = rng.gen_range(0..vals.len());
                ParamValue::Int(vals[idx])
            }
            ParamDef::IntRange { min, max } => {
                ParamValue::Int(rng.gen_range(*min..=*max))
            }
            ParamDef::Bool => ParamValue::Bool(rng.gen()),
            ParamDef::Categorical(opts) => {
                let idx = rng.gen_range(0..opts.len());
                ParamValue::Categorical(opts[idx].clone())
            }
        }
    }
}

/// Defines the hyperparameter search space.
#[derive(Debug, Clone, Default)]
pub struct HyperparameterSpace {
    params: HashMap<String, ParamDef>,
    order: Vec<String>, // preserve insertion order
}

impl HyperparameterSpace {
    /// Create an empty search space.
    pub fn new() -> Self {
        Self {
            params: HashMap::new(),
            order: Vec::new(),
        }
    }

    /// Add a float parameter with discrete values.
    pub fn add_float(&mut self, name: &str, values: &[f64]) -> &mut Self {
        self.params.insert(name.to_string(), ParamDef::Float(values.to_vec()));
        if !self.order.contains(&name.to_string()) {
            self.order.push(name.to_string());
        }
        self
    }

    /// Add a float parameter with a continuous range.
    pub fn add_float_range(&mut self, name: &str, min: f64, max: f64, log_scale: bool) -> &mut Self {
        self.params.insert(
            name.to_string(),
            ParamDef::FloatRange { min, max, log_scale },
        );
        if !self.order.contains(&name.to_string()) {
            self.order.push(name.to_string());
        }
        self
    }

    /// Add an integer parameter with discrete values.
    pub fn add_int(&mut self, name: &str, values: &[i64]) -> &mut Self {
        self.params.insert(name.to_string(), ParamDef::Int(values.to_vec()));
        if !self.order.contains(&name.to_string()) {
            self.order.push(name.to_string());
        }
        self
    }

    /// Add an integer parameter with a range.
    pub fn add_int_range(&mut self, name: &str, min: i64, max: i64) -> &mut Self {
        self.params.insert(name.to_string(), ParamDef::IntRange { min, max });
        if !self.order.contains(&name.to_string()) {
            self.order.push(name.to_string());
        }
        self
    }

    /// Add a boolean parameter.
    pub fn add_bool(&mut self, name: &str) -> &mut Self {
        self.params.insert(name.to_string(), ParamDef::Bool);
        if !self.order.contains(&name.to_string()) {
            self.order.push(name.to_string());
        }
        self
    }

    /// Add a categorical parameter.
    pub fn add_categorical(&mut self, name: &str, options: &[&str]) -> &mut Self {
        self.params.insert(
            name.to_string(),
            ParamDef::Categorical(options.iter().map(|s| s.to_string()).collect()),
        );
        if !self.order.contains(&name.to_string()) {
            self.order.push(name.to_string());
        }
        self
    }

    /// Get the number of parameters.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Check if the space is empty.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Calculate total number of combinations for grid search.
    pub fn grid_size(&self) -> usize {
        self.order
            .iter()
            .filter_map(|name| self.params.get(name))
            .map(|def| def.values().len())
            .product()
    }

    /// Generate all combinations for grid search.
    fn generate_grid(&self) -> Vec<ParamSet> {
        let param_values: Vec<(&String, Vec<ParamValue>)> = self
            .order
            .iter()
            .filter_map(|name| {
                self.params.get(name).map(|def| (name, def.values()))
            })
            .collect();

        if param_values.is_empty() {
            return vec![ParamSet::new()];
        }

        // Generate all combinations using cartesian product
        let mut combinations = vec![ParamSet::new()];
        for (name, values) in param_values {
            let mut new_combinations = Vec::new();
            for combo in &combinations {
                for value in &values {
                    let mut new_combo = combo.clone();
                    new_combo.insert(name.clone(), value.clone());
                    new_combinations.push(new_combo);
                }
            }
            combinations = new_combinations;
        }

        combinations
    }

    /// Sample a random parameter set.
    fn sample(&self, rng: &mut ChaCha8Rng) -> ParamSet {
        let mut params = ParamSet::new();
        for name in &self.order {
            if let Some(def) = self.params.get(name) {
                params.insert(name.clone(), def.sample(rng));
            }
        }
        params
    }
}

/// Result of a single trial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialResult {
    /// The parameters used for this trial.
    pub params: ParamSet,
    /// The score achieved (higher is better).
    pub score: f64,
    /// The trial number.
    pub trial: usize,
}

/// Results of a hyperparameter search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// The best parameters found.
    pub best_params: ParamSet,
    /// The best score achieved.
    pub best_score: f64,
    /// All trial results.
    pub all_trials: Vec<TrialResult>,
    /// Total number of trials run.
    pub n_trials: usize,
}

impl SearchResult {
    /// Get the top N results.
    pub fn top_n(&self, n: usize) -> Vec<&TrialResult> {
        let mut sorted: Vec<_> = self.all_trials.iter().collect();
        sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        sorted.into_iter().take(n).collect()
    }

    /// Get trial result at a specific index.
    pub fn get_trial(&self, idx: usize) -> Option<&TrialResult> {
        self.all_trials.get(idx)
    }
}

/// Error type for HPO operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum HpoError {
    /// Parameter not found in the set.
    #[error("Parameter not found: {0}")]
    ParamNotFound(String),

    /// Type mismatch when retrieving parameter.
    #[error("Type mismatch for parameter '{name}': expected {expected}")]
    TypeMismatch {
        /// The parameter name.
        name: String,
        /// The expected type.
        expected: String,
    },

    /// Error during trial evaluation.
    #[error("Trial error: {0}")]
    TrialError(String),

    /// Empty search space.
    #[error("Search space is empty")]
    EmptySpace,
}

/// Grid search optimizer that tries all combinations.
#[derive(Debug, Clone)]
pub struct GridSearch {
    space: HyperparameterSpace,
    maximize: bool,
    verbose: bool,
}

impl GridSearch {
    /// Create a new grid search with the given space.
    pub fn new(space: HyperparameterSpace) -> Self {
        Self {
            space,
            maximize: true,
            verbose: true,
        }
    }

    /// Set whether to maximize (true) or minimize (false) the objective.
    pub fn maximize(mut self, maximize: bool) -> Self {
        self.maximize = maximize;
        self
    }

    /// Set whether to print progress.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Run the grid search with the given objective function.
    ///
    /// The objective function receives a `ParamSet` and should return a score.
    /// Higher scores are better when `maximize=true` (default).
    pub fn run<F>(&self, mut objective: F) -> Result<SearchResult, HpoError>
    where
        F: FnMut(&ParamSet) -> Result<f64, HpoError>,
    {
        if self.space.is_empty() {
            return Err(HpoError::EmptySpace);
        }

        let grid = self.space.generate_grid();
        let total = grid.len();

        if self.verbose {
            eprintln!("Starting grid search with {} combinations", total);
        }

        let mut best_score = if self.maximize { f64::NEG_INFINITY } else { f64::INFINITY };
        let mut best_params = ParamSet::new();
        let mut all_trials = Vec::new();

        for (i, params) in grid.iter().enumerate() {
            let score = objective(params)?;

            let is_better = if self.maximize {
                score > best_score
            } else {
                score < best_score
            };

            if is_better {
                best_score = score;
                best_params = params.clone();
            }

            all_trials.push(TrialResult {
                params: params.clone(),
                score,
                trial: i,
            });

            if self.verbose {
                let marker = if is_better { " *" } else { "" };
                eprintln!("Trial {}/{}: score = {:.6}{}", i + 1, total, score, marker);
            }
        }

        Ok(SearchResult {
            best_params,
            best_score,
            all_trials,
            n_trials: total,
        })
    }
}

/// Random search optimizer that samples randomly from the space.
#[derive(Debug, Clone)]
pub struct RandomSearch {
    space: HyperparameterSpace,
    n_trials: usize,
    seed: Seed,
    maximize: bool,
    verbose: bool,
}

impl RandomSearch {
    /// Create a new random search with the given space and number of trials.
    pub fn new(space: HyperparameterSpace, n_trials: usize) -> Self {
        Self {
            space,
            n_trials,
            seed: Seed::new(42),
            maximize: true,
            verbose: true,
        }
    }

    /// Set the random seed.
    pub fn seed(mut self, seed: Seed) -> Self {
        self.seed = seed;
        self
    }

    /// Set whether to maximize (true) or minimize (false) the objective.
    pub fn maximize(mut self, maximize: bool) -> Self {
        self.maximize = maximize;
        self
    }

    /// Set whether to print progress.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Run the random search with the given objective function.
    pub fn run<F>(&self, mut objective: F) -> Result<SearchResult, HpoError>
    where
        F: FnMut(&ParamSet) -> Result<f64, HpoError>,
    {
        if self.space.is_empty() {
            return Err(HpoError::EmptySpace);
        }

        let mut rng = self.seed.to_rng();

        if self.verbose {
            eprintln!("Starting random search with {} trials", self.n_trials);
        }

        let mut best_score = if self.maximize { f64::NEG_INFINITY } else { f64::INFINITY };
        let mut best_params = ParamSet::new();
        let mut all_trials = Vec::new();

        for i in 0..self.n_trials {
            let params = self.space.sample(&mut rng);
            let score = objective(&params)?;

            let is_better = if self.maximize {
                score > best_score
            } else {
                score < best_score
            };

            if is_better {
                best_score = score;
                best_params = params.clone();
            }

            all_trials.push(TrialResult {
                params,
                score,
                trial: i,
            });

            if self.verbose {
                let marker = if is_better { " *" } else { "" };
                eprintln!("Trial {}/{}: score = {:.6}{}", i + 1, self.n_trials, score, marker);
            }
        }

        Ok(SearchResult {
            best_params,
            best_score,
            all_trials,
            n_trials: self.n_trials,
        })
    }
}

/// Successive Halving optimizer for efficient hyperparameter search.
///
/// This implements the Successive Halving algorithm which:
/// 1. Starts with many configurations and a small budget per trial
/// 2. Keeps the top fraction of configurations at each round
/// 3. Increases the budget for survivors
///
/// This is more efficient than grid/random search when training is expensive.
#[derive(Debug, Clone)]
pub struct SuccessiveHalving {
    space: HyperparameterSpace,
    n_configs: usize,
    min_budget: usize,
    max_budget: usize,
    eta: usize,
    seed: Seed,
    maximize: bool,
    verbose: bool,
}

impl SuccessiveHalving {
    /// Create a new Successive Halving search.
    ///
    /// # Arguments
    /// * `space` - The hyperparameter search space
    /// * `n_configs` - Initial number of configurations to try
    /// * `min_budget` - Minimum budget (e.g., epochs) per trial
    /// * `max_budget` - Maximum budget for best configurations
    /// * `eta` - Reduction factor (typically 3)
    pub fn new(
        space: HyperparameterSpace,
        n_configs: usize,
        min_budget: usize,
        max_budget: usize,
        eta: usize,
    ) -> Self {
        Self {
            space,
            n_configs,
            min_budget,
            max_budget,
            eta,
            seed: Seed::new(42),
            maximize: true,
            verbose: true,
        }
    }

    /// Set the random seed.
    pub fn seed(mut self, seed: Seed) -> Self {
        self.seed = seed;
        self
    }

    /// Set whether to maximize (true) or minimize (false) the objective.
    pub fn maximize(mut self, maximize: bool) -> Self {
        self.maximize = maximize;
        self
    }

    /// Set whether to print progress.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Run Successive Halving with the given objective function.
    ///
    /// The objective function receives a `ParamSet` and a budget (e.g., epochs),
    /// and should return a score.
    pub fn run<F>(&self, mut objective: F) -> Result<SearchResult, HpoError>
    where
        F: FnMut(&ParamSet, usize) -> Result<f64, HpoError>,
    {
        if self.space.is_empty() {
            return Err(HpoError::EmptySpace);
        }

        let mut rng = self.seed.to_rng();

        // Sample initial configurations
        let mut configs: Vec<ParamSet> = (0..self.n_configs)
            .map(|_| self.space.sample(&mut rng))
            .collect();

        let mut all_trials = Vec::new();
        let mut budget = self.min_budget;
        let mut round = 0;
        let mut trial_id = 0;

        while configs.len() > 1 && budget <= self.max_budget {
            if self.verbose {
                eprintln!(
                    "Round {}: {} configs, budget = {}",
                    round,
                    configs.len(),
                    budget
                );
            }

            // Evaluate all configurations with current budget
            let mut scores: Vec<(usize, f64)> = Vec::new();
            for (i, params) in configs.iter().enumerate() {
                let score = objective(params, budget)?;
                scores.push((i, score));

                all_trials.push(TrialResult {
                    params: params.clone(),
                    score,
                    trial: trial_id,
                });
                trial_id += 1;

                if self.verbose {
                    eprintln!("  Config {}: score = {:.6}", i, score);
                }
            }

            // Sort by score
            if self.maximize {
                scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            } else {
                scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            }

            // Keep top 1/eta configurations
            let n_keep = (configs.len() / self.eta).max(1);
            let keep_indices: Vec<usize> = scores.iter().take(n_keep).map(|(i, _)| *i).collect();

            configs = keep_indices.iter().map(|&i| configs[i].clone()).collect();
            budget *= self.eta;
            round += 1;
        }

        // Find best result
        let (best_params, best_score) = if self.maximize {
            all_trials
                .iter()
                .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
                .map(|t| (t.params.clone(), t.score))
                .unwrap_or_else(|| (ParamSet::new(), f64::NEG_INFINITY))
        } else {
            all_trials
                .iter()
                .min_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
                .map(|t| (t.params.clone(), t.score))
                .unwrap_or_else(|| (ParamSet::new(), f64::INFINITY))
        };

        Ok(SearchResult {
            best_params,
            best_score,
            all_trials,
            n_trials: trial_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_set() {
        let mut params = ParamSet::new();
        params.insert("lr", ParamValue::Float(0.001));
        params.insert("batch_size", ParamValue::Int(32));
        params.insert("use_bn", ParamValue::Bool(true));
        params.insert("optimizer", ParamValue::Categorical("adam".to_string()));

        assert_eq!(params.get_float("lr").unwrap(), 0.001);
        assert_eq!(params.get_int("batch_size").unwrap(), 32);
        assert!(params.get_bool("use_bn").unwrap());
        assert_eq!(params.get_categorical("optimizer").unwrap(), "adam");
    }

    #[test]
    fn test_hyperparameter_space() {
        let mut space = HyperparameterSpace::new();
        space
            .add_float("lr", &[0.001, 0.01])
            .add_int("batch_size", &[16, 32])
            .add_categorical("optimizer", &["adam", "sgd"]);

        assert_eq!(space.len(), 3);
        assert_eq!(space.grid_size(), 8); // 2 * 2 * 2
    }

    #[test]
    fn test_grid_search() {
        let mut space = HyperparameterSpace::new();
        space
            .add_float("x", &[1.0, 2.0])
            .add_float("y", &[10.0, 20.0]);

        let search = GridSearch::new(space).verbose(false);

        // Objective: maximize x + y
        let result = search
            .run(|params| {
                let x = params.get_float("x")?;
                let y = params.get_float("y")?;
                Ok(x + y)
            })
            .unwrap();

        assert_eq!(result.n_trials, 4);
        assert_eq!(result.best_score, 22.0); // 2.0 + 20.0
        assert_eq!(result.best_params.get_float("x").unwrap(), 2.0);
        assert_eq!(result.best_params.get_float("y").unwrap(), 20.0);
    }

    #[test]
    fn test_grid_search_minimize() {
        let mut space = HyperparameterSpace::new();
        space.add_float("x", &[1.0, 2.0, 3.0]);

        let search = GridSearch::new(space).verbose(false).maximize(false);

        // Objective: minimize x^2
        let result = search
            .run(|params| {
                let x = params.get_float("x")?;
                Ok(x * x)
            })
            .unwrap();

        assert_eq!(result.best_score, 1.0); // 1.0^2
        assert_eq!(result.best_params.get_float("x").unwrap(), 1.0);
    }

    #[test]
    fn test_random_search() {
        let mut space = HyperparameterSpace::new();
        space.add_float_range("x", 0.0, 10.0, false);

        let search = RandomSearch::new(space, 20)
            .seed(Seed::new(42))
            .verbose(false);

        // Objective: minimize (x - 5)^2
        let result = search
            .run(|params| {
                let x = params.get_float("x")?;
                Ok(-((x - 5.0).powi(2))) // Negate because we maximize
            })
            .unwrap();

        assert_eq!(result.n_trials, 20);
        // Best x should be close to 5.0
        let best_x = result.best_params.get_float("x").unwrap();
        assert!(best_x > 2.0 && best_x < 8.0);
    }

    #[test]
    fn test_random_search_log_scale() {
        let mut space = HyperparameterSpace::new();
        space.add_float_range("lr", 1e-5, 1e-1, true);

        let search = RandomSearch::new(space, 10)
            .seed(Seed::new(123))
            .verbose(false);

        let result = search.run(|params| {
            let lr = params.get_float("lr")?;
            assert!(lr >= 1e-5 && lr <= 1e-1);
            Ok(lr)
        });

        assert!(result.is_ok());
    }

    #[test]
    fn test_successive_halving() {
        let mut space = HyperparameterSpace::new();
        space.add_float("x", &[1.0, 2.0, 3.0, 4.0, 5.0]);

        let search = SuccessiveHalving::new(space, 5, 1, 4, 2)
            .seed(Seed::new(42))
            .verbose(false);

        // Objective: maximize x (budget doesn't matter for this simple test)
        let result = search
            .run(|params, _budget| {
                let x = params.get_float("x")?;
                Ok(x)
            })
            .unwrap();

        // The best should gravitate towards higher x values
        assert!(result.best_score >= 3.0);
    }

    #[test]
    fn test_top_n_results() {
        let mut space = HyperparameterSpace::new();
        space.add_int("x", &[1, 2, 3, 4, 5]);

        let search = GridSearch::new(space).verbose(false);

        let result = search
            .run(|params| {
                let x = params.get_int("x")?;
                Ok(x as f64)
            })
            .unwrap();

        let top3 = result.top_n(3);
        assert_eq!(top3.len(), 3);
        assert_eq!(top3[0].score, 5.0);
        assert_eq!(top3[1].score, 4.0);
        assert_eq!(top3[2].score, 3.0);
    }

    #[test]
    fn test_empty_space_error() {
        let space = HyperparameterSpace::new();
        let search = GridSearch::new(space).verbose(false);

        let result = search.run(|_| Ok(0.0));
        assert!(matches!(result, Err(HpoError::EmptySpace)));
    }

    #[test]
    fn test_param_not_found_error() {
        let params = ParamSet::new();
        let result = params.get_float("nonexistent");
        assert!(matches!(result, Err(HpoError::ParamNotFound(_))));
    }

    #[test]
    fn test_type_mismatch_error() {
        let mut params = ParamSet::new();
        params.insert("x", ParamValue::Int(42));

        let result = params.get_float("x");
        assert!(matches!(result, Err(HpoError::TypeMismatch { .. })));
    }

    #[test]
    fn test_bool_parameter() {
        let mut space = HyperparameterSpace::new();
        space.add_bool("use_dropout");

        assert_eq!(space.grid_size(), 2);

        let search = GridSearch::new(space).verbose(false);
        let result = search
            .run(|params| {
                let use_dropout = params.get_bool("use_dropout")?;
                Ok(if use_dropout { 1.0 } else { 0.0 })
            })
            .unwrap();

        assert_eq!(result.n_trials, 2);
    }

    #[test]
    fn test_int_range() {
        let mut space = HyperparameterSpace::new();
        space.add_int_range("hidden", 64, 66);

        let search = GridSearch::new(space).verbose(false);
        let result = search
            .run(|params| {
                let hidden = params.get_int("hidden")?;
                assert!(hidden >= 64 && hidden <= 66);
                Ok(hidden as f64)
            })
            .unwrap();

        assert_eq!(result.n_trials, 3); // 64, 65, 66
    }
}
