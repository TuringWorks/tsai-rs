//! Example: Hyperparameter Optimization
//!
//! This example demonstrates how to use tsai-rs's HPO module to find
//! optimal hyperparameters using grid search and random search.
//!
//! Run with: cargo run --example hpo

use tsai_core::Seed;
use tsai_train::{GridSearch, HyperparameterSpace, RandomSearch, SuccessiveHalving};

fn main() {
    println!("=== Hyperparameter Optimization ===\n");

    // Define a search space
    let mut space = HyperparameterSpace::new();
    space
        .add_float("learning_rate", &[1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
        .add_int("batch_size", &[16, 32, 64, 128])
        .add_int("n_layers", &[2, 3, 4])
        .add_float("dropout", &[0.0, 0.1, 0.2, 0.3])
        .add_categorical("optimizer", &["adam", "sgd", "adamw"]);

    println!("Search Space:");
    println!("  learning_rate: [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]");
    println!("  batch_size: [16, 32, 64, 128]");
    println!("  n_layers: [2, 3, 4]");
    println!("  dropout: [0.0, 0.1, 0.2, 0.3]");
    println!("  optimizer: [adam, sgd, adamw]");
    println!("  Total combinations: {}\n", space.grid_size());

    // Simulated objective function
    // In practice, this would train a model and return validation accuracy
    let objective = |params: &tsai_train::ParamSet| -> Result<f64, tsai_train::HpoError> {
        let lr = params.get_float("learning_rate")?;
        let batch_size = params.get_int("batch_size")?;
        let n_layers = params.get_int("n_layers")?;
        let dropout = params.get_float("dropout")?;
        let optimizer = params.get_categorical("optimizer")?;

        // Simulate a score based on hyperparameters
        // (In reality, you'd train and evaluate a model here)
        let mut score = 0.7;

        // Prefer medium learning rates
        score += match lr {
            x if x < 1e-3 => 0.05,
            x if x < 5e-3 => 0.15,
            _ => 0.08,
        };

        // Prefer batch size 32 or 64
        score += match batch_size {
            32 | 64 => 0.1,
            _ => 0.05,
        };

        // Prefer 3 layers
        score += match n_layers {
            3 => 0.08,
            _ => 0.04,
        };

        // Prefer moderate dropout
        score += match dropout {
            x if x > 0.05 && x < 0.25 => 0.05,
            _ => 0.02,
        };

        // Adam works best
        score += match optimizer.as_str() {
            "adam" | "adamw" => 0.05,
            _ => 0.02,
        };

        // Add some noise
        score += (score * 0.05_f64).sin() * 0.02;

        Ok(score)
    };

    // 1. Grid Search (subset for demo)
    println!("--- Grid Search (subset) ---");
    let mut small_space = HyperparameterSpace::new();
    small_space
        .add_float("learning_rate", &[1e-3, 5e-3])
        .add_int("batch_size", &[32, 64])
        .add_int("n_layers", &[3])
        .add_float("dropout", &[0.1])
        .add_categorical("optimizer", &["adam"]);

    let grid_search = GridSearch::new(small_space).verbose(false);

    let grid_result = grid_search.run(objective.clone()).expect("Grid search failed");
    println!("Best score: {:.4}", grid_result.best_score);
    println!("Best params:");
    for (name, value) in grid_result.best_params.iter() {
        println!("  {}: {:?}", name, value);
    }
    println!("Trials run: {}\n", grid_result.n_trials);

    // 2. Random Search
    println!("--- Random Search (20 trials) ---");
    let random_search = RandomSearch::new(space.clone(), 20)
        .seed(Seed::new(42))
        .verbose(false);

    let random_result = random_search.run(objective.clone()).expect("Random search failed");
    println!("Best score: {:.4}", random_result.best_score);
    println!("Best params:");
    for (name, value) in random_result.best_params.iter() {
        println!("  {}: {:?}", name, value);
    }

    println!("\nTop 3 configurations:");
    for (i, trial) in random_result.top_n(3).iter().enumerate() {
        println!("  {}. Score: {:.4}", i + 1, trial.score);
    }

    // 3. Successive Halving
    println!("\n--- Successive Halving ---");
    let sh_space = space.clone();
    let successive_halving = SuccessiveHalving::new(
        sh_space,
        9,    // initial configurations
        1,    // min budget (e.g., epochs)
        9,    // max budget
        3,    // reduction factor (eta)
    )
    .seed(Seed::new(42))
    .verbose(false);

    // Objective with budget parameter
    let budget_objective = |params: &tsai_train::ParamSet, budget: usize| -> Result<f64, tsai_train::HpoError> {
        let base_score = objective(params)?;
        // Score improves slightly with more budget (epochs)
        let budget_bonus = (budget as f64 / 10.0) * 0.02;
        Ok(base_score + budget_bonus)
    };

    let sh_result = successive_halving.run(budget_objective).expect("Successive halving failed");
    println!("Best score: {:.4}", sh_result.best_score);
    println!("Best params:");
    for (name, value) in sh_result.best_params.iter() {
        println!("  {}: {:?}", name, value);
    }
    println!("Total trials: {}", sh_result.n_trials);

    // 4. Using continuous ranges
    println!("\n--- Random Search with Continuous Ranges ---");
    let mut continuous_space = HyperparameterSpace::new();
    continuous_space
        .add_float_range("learning_rate", 1e-5, 1e-1, true) // log scale
        .add_int_range("hidden_size", 64, 256)
        .add_bool("use_dropout");

    let range_search = RandomSearch::new(continuous_space, 10)
        .seed(Seed::new(123))
        .verbose(false);

    let range_objective = |params: &tsai_train::ParamSet| -> Result<f64, tsai_train::HpoError> {
        let lr = params.get_float("learning_rate")?;
        let hidden = params.get_int("hidden_size")?;
        let use_dropout = params.get_bool("use_dropout")?;

        // Simulate score
        let lr_score = if lr > 1e-4 && lr < 1e-2 { 0.9 } else { 0.7 };
        let hidden_score = (hidden as f64 / 256.0) * 0.1;
        let dropout_bonus = if use_dropout { 0.02 } else { 0.0 };

        Ok(lr_score + hidden_score + dropout_bonus)
    };

    let range_result = range_search.run(range_objective).expect("Range search failed");
    println!("Best score: {:.4}", range_result.best_score);
    println!("Best params:");
    println!("  learning_rate: {:.6}", range_result.best_params.get_float("learning_rate").unwrap());
    println!("  hidden_size: {}", range_result.best_params.get_int("hidden_size").unwrap());
    println!("  use_dropout: {}", range_result.best_params.get_bool("use_dropout").unwrap());

    println!("\n=== HPO Complete ===");
}
