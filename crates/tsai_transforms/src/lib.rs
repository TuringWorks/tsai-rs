//! # tsai_transforms
//!
//! Time series augmentations, label-mixing, and imaging transforms.
//!
//! This crate provides:
//! - Data augmentation transforms (noise, warping, masking, etc.)
//! - Label-mixing callbacks (MixUp, CutMix)
//! - Imaging transforms (Recurrence Plots, GAF, MTF)
//!
//! ## Augmentation Example
//!
//! ```rust,ignore
//! use tsai_transforms::augment::{GaussianNoise, Compose};
//! use tsai_core::{Transform, Split};
//!
//! let transform = Compose::new()
//!     .add(GaussianNoise::new(0.1))
//!     .add(TimeWarp::new(0.2));
//!
//! let batch = transform.apply(batch, Split::Train)?;
//! ```

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod augment;
pub mod imaging;
pub mod label_mix;

pub use augment::*;
pub use imaging::*;
pub use label_mix::*;
