//! CNN models for time series.

mod fcn;
mod inception;
mod omniscale;
mod resnet;
mod tcn;
mod xcm;
mod xception;

pub use fcn::{ConvBlock, FCN, FCNConfig};
pub use inception::{InceptionBlock, InceptionTimePlus, InceptionTimePlusConfig};
pub use omniscale::{OmniScaleCNN, OmniScaleCNNConfig};
pub use resnet::{ResNetBlock, ResNetPlus, ResNetPlusConfig};
pub use tcn::{TCN, TCNBlock, TCNBlockConfig, TCNConfig};
pub use xcm::{XCMPlus, XCMPlusConfig};
pub use xception::{SeparableConv1d, XceptionBlock, XceptionTime, XceptionTimeConfig};
