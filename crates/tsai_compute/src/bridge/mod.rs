//! Integration bridges with other frameworks.

#[cfg(feature = "burn-bridge")]
pub mod burn;

// Re-export bridge types when available
#[cfg(feature = "burn-bridge")]
pub use self::burn::*;
