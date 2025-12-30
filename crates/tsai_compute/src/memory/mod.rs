//! Memory management abstractions.
//!
//! This module provides unified buffer types, memory pooling,
//! and cross-device transfer utilities.

mod buffer;
mod pool;
mod transfer;

pub use buffer::*;
pub use pool::*;
pub use transfer::*;
