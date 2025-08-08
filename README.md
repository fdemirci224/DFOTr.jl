<div style="display: flex; align-items: center; justify-content: space-between; width: 100%;">
  <img src="https://github.com/user-attachments/assets/59c3f790-5bcc-4932-82bc-fe4ca168d15f" alt="DFO-TR Logo" width="100" height="100">
  <h1 style="margin: 0; font-size: 24px;">DERIVATIVE-FREE OPTIMIZATION TRUST-REGION</h1>
</div>
---


# DFOTr.jl — Derivative-Free Optimization Trust-Region

This repository provides a pure-Julia implementation of a Derivative-Free Optimization Trust-Region (DFO-TR) solver. The method builds a local quadratic surrogate from function evaluations only (no gradients), solves a trust-region subproblem, and adapts the radius based on agreement between predicted and actual reductions.

- Module: `DFOTr`
- Main entry point: `DFOTr.dfo_tr()`
- Dependencies: Julia standard libraries only (LinearAlgebra, Statistics, Printf)

## Installation

This package is not yet registered. You can add it locally during development or directly from a Git URL once published.

- Local path (recommended while developing):
  ```julia
  using Pkg
  Pkg.develop(path = "c:/path/to/DFOTr_TEST/DFOTr_TEST")
  ```
- From a Git URL (replace with your repo URL when available):
  ```julia
  using Pkg
  Pkg.add(url = "https://github.com/your-user/DFOTr.jl.git")
  ```

## Quick Start

```julia
using DFOTr

# Black-box objective
sphere(x) = sum(x.^2)

x0 = [2.0, 3.0]

# Run solver (verbosity=1 prints compact iteration table)
result = DFOTr.dfo_tr(sphere, x0; verbosity=1, maxfev=200)

println("opt point     = ", result.x)
println("opt value     = ", result.fun)
println("iterations    = ", result.iteration)
println("successful    = ", result.iter_suc)
println("func evals    = ", result.func_eval)
println("final radius  = ", result.delta)
```
Verbosity levels:
- 0: Silent
- 1: Compact iteration table
- 2: Detailed diagnostics per iteration with optional CSV/JSON logging

## Algorithm Options

Options are provided via `DFOTr.DFOParams` and can be overridden as keywords in `DFOTr.dfo_tr`.

```julia
DFOTr.DFOParams(; 
    init_delta::Float64 = 1.0,      # initial trust-region radius
    tol_delta::Float64 = 1e-10,     # minimum radius to stop
    max_delta::Float64 = 100.0,     # maximum radius
    gamma1::Float64 = 0.8,          # shrink factor
    gamma2::Float64 = 1.5,          # expand factor
    eta0::Float64 = 0.0,            # acceptance threshold
    eta1::Float64 = 0.25,           # shrink threshold
    eta2::Float64 = 0.75,           # expand threshold
    tol_f::Float64 = 1e-15,         # |f_k - f_{k+1}| tolerance
    tol_norm_g::Float64 = 1e-15,    # gradient norm (model g)
    tol_norm_H::Float64 = 1e-10,    # Hessian Frobenius norm
    maxfev::Int = 1000,             # max function evaluations
    min_del_crit::Float64 = 1e-8,   # criticality radius floor
    min_s_crit::Float64 = 0.1       # criticality step floor
)
```

Example:
```julia
result = DFOTr.dfo_tr(sphere, x0; init_delta=0.5, maxfev=300, tol_norm_g=1e-8, verbosity=2)
```

## Algorithm Description

- __Model construction__: Build a quadratic model m(x) = f0 + gᵀs + 0.5 sᵀHs from the sample set using full interpolation or a minimum-Frobenius-norm regularized approach (see `src/ModelBuilder.jl`).
- __Trust-region step__: Solve the trust-region subproblem using a Moré–Sorensen-style method with robust fallbacks (e.g., Cauchy point) (see `src/TrustRegion.jl`).
- __Acceptance & radius update__: Compare actual vs predicted reduction to compute ρ. Accept if ρ ≥ η₀ and update the radius with γ₁/γ₂ depending on ρ vs thresholds η₁, η₂.
- __Sample set management__: Maintain a poised set and replace points using distance and f-value criteria (see `src/Utils.jl`).
- __Stopping__: Triggered when tolerances are met (small gradient norm, small |Δf|, or small radius) or when `maxfev` is reached.

Key files:
- `src/DFOTr.jl`: `dfo_tr` driver and types (`DFOParams`, `DFOResult`).
- `src/ModelBuilder.jl`: Quadratic model construction and regularization.
- `src/TrustRegion.jl`: Trust-region subproblem solver.
- `src/Utils.jl`: Utilities and sample-set management.
- `src/Logger.jl`: CSV/JSON iteration logger for verbosity ≥ 2.

## Result Object

`DFOTr.DFOResult` contains:
- `x::Vector{Float64}`: final point
- `fun::Float64`: final objective value
- `iteration::Int`: number of iterations
- `iter_suc::Int`: successful iterations
- `func_eval::Int`: function evaluations
- `delta::Float64`: final trust-region radius

## Testing

Run the tests via Pkg:
```julia
using Pkg
Pkg.test("DFOTr")
```

## References

- A. R. Conn, K. Scheinberg, and L. N. Vicente. "Introduction to Derivative-Free Optimization". SIAM, 2009.
- Original Python implementation by Anahita Hassanzadeh

## License

This project is licensed under the EPL-2.0 license. See `LICENSE`.
