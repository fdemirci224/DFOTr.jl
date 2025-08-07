# RULES: DFO-TR Solver Development

## 1. Architecture
- Modular structure: keep model building, TR solver, and main loop in separate files
- Reuse common logic via `Utils.jl`
- Follow Julia's module system for namespacing

## 2. Code Style
- Use descriptive function names and type annotations
- Use `struct` for parameters and results
- Avoid global variables
- Follow Julia conventions (snake_case for variables, PascalCase for types)

## 3. Numerical Stability
- Use SVD with a tolerance fallback in underdetermined cases
- Clip small singular values to `ϵ^5` for stability
- Handle floating-point comparisons with tolerances (e.g., `≈` or `isapprox`)

## 4. Trust Region Method
- Always ensure trial steps satisfy the trust-region constraint
- Use More–Sorensen method for solving subproblem
- Handle indefinite Hessians with secular equation root-finding

## 5. Documentation & Testing
- Add docstrings for all public functions
- Create minimal examples in `test_dfo.jl`
- Include verbose logging for debugging (optional toggle)

## 6. Performance
- Use `StaticArrays` where applicable for small vector dimensions
- Avoid unnecessary allocations in inner loops
- Preallocate where possible

## 7. Extensibility
- Allow pluggable sample generation methods
- Parameter struct should support overrides from keyword arguments
