//! CNN models for time series.

mod fcn;
mod inception;
mod mlp;
mod mwdn;
mod omniscale;
mod rescnn;
mod resnet;
mod tcn;
mod xcm;
mod xception;
mod xresnet;

pub use fcn::{ConvBlock, FCN, FCNConfig};
pub use inception::{InceptionBlock, InceptionTimePlus, InceptionTimePlusConfig};
pub use mlp::{MLP, MLPConfig};
pub use mwdn::{MWDNConfig, WaveletType, MWDN};
pub use omniscale::{OmniScaleCNN, OmniScaleCNNConfig};
pub use rescnn::{ResCNN, ResCNNBlock, ResCNNConfig};
pub use resnet::{ResNetBlock, ResNetPlus, ResNetPlusConfig};
pub use tcn::{TCN, TCNBlock, TCNBlockConfig, TCNConfig};
pub use xcm::{XCMPlus, XCMPlusConfig};
pub use xception::{SeparableConv1d, XceptionBlock, XceptionTime, XceptionTimeConfig};
pub use xresnet::{XResNet1d, XResNet1dConfig};
