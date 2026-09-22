# Validation record

The sample campaign was executed on 2026-09-22 with Windows, CPython 3.14.5,
NumPy 2.5.3, and MSVC 19.44.35219.0, using Release compilation.
This record reports observed results; it is not a certification.

Local checks passed: 42 Python tests (98% Python statement coverage), CMake/MSVC
Release compilation, native CTest 1/1, Ruff lint/format, clang-format, source
distribution and wheel builds, installed-wheel tests and CLI, and `pip check`.
The installed-wheel test run loaded modules from site-packages, not the source
tree. MSBuild emitted temporary-directory/path-property warnings (MSB8029 and
MSB8012) during isolated packaging; packaging succeeded and the resulting
wheel passed all tests. These infrastructure warnings remain documented.

## Analytical campaign

All cases use L=1, duration=0.5, CFL fraction=0.8, and exact sine cell averages.
Each pair of methods is run on 40, 80, 160, and 320 cells (24 solves).

| Case | Velocity | Physical diffusivity | Upwind finest L1 order | MC finest L1 order |
| --- | ---: | ---: | ---: | ---: |
| Advection | 1.0 | 0 | 0.977804 | 2.001037 |
| Mixed | -0.7 | 0.01 | 0.983969 | 1.981563 |
| Heat | 0 | 0.02 | 1.999972 | 1.999972 |

At 160 cells in pure advection, upwind's apparent diffusivity is 0.003124624;
the leading small-wavenumber prediction is 0.003125. MC gives 0.0000014115.
These measure fundamental-mode damping only; they do not account for phase
error or nonlinear transfer to other harmonics.

The largest absolute smooth-case mass drift is 8.3045e-14. Two additional
200-cell pulses after one circuit remain in [0,1] within a 1e-12 tolerance.
Their mean absolute errors are 0.112592 (upwind) and 0.023262 (MC).

Read `results/convergence.csv` for full-precision errors, time-step counts,
attenuation, and orders. `results/validation.json` contains explicit pass/fail
gates. The figure was inspected for labels, clipping, and consistency with CSVs.

## Independent checks

- Python tests cover randomized signed data, mass, bounds, total variation,
  positive/negative/zero velocities, advection/diffusion, reflection symmetry,
  constants, zero time, input ownership, invalid inputs, non-unit domains,
  analytical overlap averages, and CLI output.
- An upwind Fourier-mode test uses the independently derived spatial eigenvalue
  and the SSPRK3 stability polynomial, rather than duplicating stencil code.
- A time-refinement test compares the heat equation with the exact
  semidiscrete Fourier decay, isolating third-order time error from spatial error.
- Native CTest runs 12 conservation/boundedness cases with active Release checks.
- The campaign performs three MC convergence gates (>1.7), smooth mass (<1e-12),
  and pulse bounds. It is intentionally small enough to execute during CI.

`requirements-repro.txt` captures the local environment. Build/test results and
remote CI completion are recorded by the project-series index and automation
memory as well as the repository's actual Actions logs. No benchmark timing is
claimed: the benchmarks here concern numerical error against analytical references.
