# Changelog

PolyUQ was developed within the `oma_uq` research repository from 2022 and
extracted as a standalone package in 2025. The dates below are the original
development milestones from the source history.

## [Unreleased]
- `UncertainVariable.proposal`: declare the sampling (proposal) distribution a
  primary variable's samples actually came from. `probabilities_imp` then forms
  the importance weights as `target pdf / proposal pdf` instead of assuming the
  uniform-over-support proposal that `sample_qmc` implies. Needed for *observed*
  aleatory samples -- e.g. the measured excitation level of each data block in an
  experimental OMA study -- where the uniform assumption would silently bias the
  weighted ensemble towards the product of the assumed and the observed density.
  `proposal = None` (the default) is bit-identical to the previous behaviour.
- `sample_qmc(given_samples={name: values})`: substitute given sample values for
  the drawn ones, for the same use case. Warns when the values fall outside the
  truncated support or when the variable declares no proposal.

## [0.1.0] - 2025 (public release)
- Extracted `uncertainty` package from oma_uq into standalone `polyuq`.
- Renamed package `uncertainty` -> `polyuq`; GPL v3; packaging, tests, docs.

## Development history (in oma_uq)
- 2023-02  optimize_inc / incompleteness work for the modal-beam case study
- 2022-04  interpolation, discrete random variables, load/save state,
           additional interpolators, convergence study, PolyUQ-DataManager bridge
- 2022-03  PolyUQ class profiled and tested
- 2022-02  polymorphic UQ with fixed-size sampling
- 2022-01  polymorphic UQ methodology started
- 2021-04  DataManager infrastructure (predates PolyUQ; general result orchestration)
