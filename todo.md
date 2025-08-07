# TODO: Derivative-Free Trust Region Solver (DFO-TR)

## Core Modules
- [ ] ✅ Create `DFO.jl`: Main optimization loop (port of `dfo_tr`)
- [ ] ✅ Create `ModelBuilder.jl`: Build quadratic models (port of `quad_Frob.py`)
- [ ] ✅ Create `TrustRegion.jl`: Solve TR subproblem (port of `trust_sub.py`)
- [ ] ✅ Create `Utils.jl`: Helper functions (distance sorting, reshaping)

## DFO.jl
- [ ] Implement parameter struct
- [ ] Implement verbosity and stopping conditions
- [ ] Port main loop: model construction, TR solve, evaluation, radius update
- [ ] Add interpolation set update logic
- [ ] Add criticality step handling

## ModelBuilder.jl
- [ ] Shift and center sample points
- [ ] Handle full quadratic model interpolation
- [ ] Handle minimum Frobenius norm model (KKT system)
- [ ] Use SVD with condition number regularization

## TrustRegion.jl
- [ ] Implement eigen-decomposition of symmetric Hessian
- [ ] Project gradient to eigenbasis
- [ ] Solve secular equation using 1D root-finding (rfzero)
- [ ] Compute trust-region step
- [ ] Handle indefinite and nearly singular Hessians

## Utils.jl
- [ ] Implement point sorting by distance
- [ ] Implement model reduction logging
- [ ] Add matrix and vector reshaping utilities

## Testing
- [ ] Write `test_dfo.jl` with Rosenbrock and Sphere
- [ ] Add regression test for known solution
- [ ] Test step acceptance and radius update logic
