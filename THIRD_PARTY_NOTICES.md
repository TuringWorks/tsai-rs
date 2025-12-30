# Third Party Notices

This project is a Rust port of the Python `tsai` library. This document provides
attribution and license notices for third-party components.

## Upstream Project

### tsai (timeseriesAI)

This project is based on and inspired by the Python tsai library.

- **Repository**: https://github.com/timeseriesAI/tsai
- **License**: Apache-2.0
- **Copyright**: 2020-2024 Ignacio Oguiza

The architecture designs, API patterns, and algorithmic implementations in tsai-rs
are derived from or inspired by the original tsai library. We gratefully acknowledge
the work of the tsai maintainers and contributors.

## Research Papers

The model architectures implemented in this library are based on the following research:

### InceptionTime

> Fawaz, H.I., Lucas, B., Forestier, G., Pelletier, C., Schmidt, D.F., Weber, J.,
> Webb, G.I., Idoumghar, L., Muller, P.A. and Petitjean, F., 2020.
> InceptionTime: Finding AlexNet for time series classification.
> Data Mining and Knowledge Discovery, 34(6), pp.1936-1962.

### ROCKET / MiniRocket

> Dempster, A., Petitjean, F. and Webb, G.I., 2020.
> ROCKET: exceptionally fast and accurate time series classification using random convolutional kernels.
> Data Mining and Knowledge Discovery, 34(5), pp.1454-1495.

> Dempster, A., Schmidt, D.F. and Webb, G.I., 2021.
> MINIROCKET: A Very Fast (Almost) Deterministic Transform for Time Series Classification.
> In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining.

### PatchTST

> Nie, Y., Nguyen, N.H., Sinthong, P. and Kalagnanam, J., 2023.
> A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.
> In International Conference on Learning Representations.

### TST (Time Series Transformer)

> Zerveas, G., Jayaraman, S., Patel, D., Bhamidipaty, A. and Eickhoff, C., 2021.
> A Transformer-based Framework for Multivariate Time Series Representation Learning.
> In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining.

### MixUp / CutMix

> Zhang, H., Cisse, M., Dauphin, Y.N. and Lopez-Paz, D., 2018.
> mixup: Beyond empirical risk minimization.
> In International Conference on Learning Representations.

> Yun, S., Han, D., Oh, S.J., Chun, S., Choe, J. and Yoo, Y., 2019.
> CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features.
> In Proceedings of the IEEE/CVF International Conference on Computer Vision.

## Rust Dependencies

This project uses the following Rust crates, each with their own licenses:

- **burn** (Apache-2.0 / MIT): Deep learning framework
- **ndarray** (Apache-2.0 / MIT): N-dimensional array library
- **polars** (MIT): DataFrame library
- **serde** (Apache-2.0 / MIT): Serialization framework
- **thiserror** (Apache-2.0 / MIT): Error handling
- **rand** (Apache-2.0 / MIT): Random number generation
- **rayon** (Apache-2.0 / MIT): Parallel computation
- **clap** (Apache-2.0 / MIT): Command line argument parsing
- **tracing** (MIT): Application-level tracing

For the complete list of dependencies and their licenses, please refer to the
`Cargo.lock` file and run `cargo license` in the project root.

## Acknowledgments

We thank:

- The tsai team for creating an excellent time series deep learning library
- The Burn team for the Rust deep learning framework
- All researchers whose work is implemented in this library
- The Rust community for the excellent ecosystem of crates

## License

tsai-rs is licensed under the Apache License, Version 2.0.
See the LICENSE file for the full license text.
