# DFO.jl - Main DFO-TR algorithm
# Port of dfo_tr.py from https://github.com/TheClimateCorporation/dfo-algorithm

module DFO

using LinearAlgebra
using Statistics
include("ModelBuilder.jl")
include("TrustRegion.jl")
include("Utils.jl")

export DFOParams, DFOResult, dfo_tr

"""
    DFOParams

Parameters for the DFO-TR trust region algorithm.
Corresponds to the `params` class in the Python implementation.
"""
struct DFOParams
    # Trust region parameters
    init_delta::Float64      # Initial trust region radius
    tol_delta::Float64       # Minimum delta to stop
    max_delta::Float64       # Maximum possible delta
    gamma1::Float64          # Radius shrink factor
    gamma2::Float64          # Radius expansion factor
    
    # Step acceptance parameters
    eta0::Float64            # Step acceptance threshold (pred/ared)
    eta1::Float64            # Level to shrink radius
    eta2::Float64            # Level to expand radius
    
    # Convergence tolerances
    tol_f::Float64           # Threshold for |fprev - fnew|
    tol_norm_g::Float64      # Threshold for norm of gradient
    tol_norm_H::Float64      # Threshold for Frobenius norm of H
    maxfev::Int              # Maximum number of iterations
    
    # Criticality step parameters
    min_del_crit::Float64    # Minimum delta for criticality step
    min_s_crit::Float64      # Minimum step for criticality step
end

"""
    DFOParams(; kwargs...)

Create DFOParams with default values, allowing keyword overrides.
"""
function DFOParams(; 
    init_delta::Float64 = 1.0,
    tol_delta::Float64 = 1e-10,
    max_delta::Float64 = 100.0,
    gamma1::Float64 = 0.8,
    gamma2::Float64 = 1.5,
    eta0::Float64 = 0.0,
    eta1::Float64 = 0.25,
    eta2::Float64 = 0.75,
    tol_f::Float64 = 1e-15,
    tol_norm_g::Float64 = 1e-15,
    tol_norm_H::Float64 = 1e-10,
    maxfev::Int = 1000,
    min_del_crit::Float64 = 1e-8,
    min_s_crit::Float64 = 0.1
)
    return DFOParams(
        init_delta, tol_delta, max_delta, gamma1, gamma2,
        eta0, eta1, eta2, tol_f, tol_norm_g, tol_norm_H, maxfev,
        min_del_crit, min_s_crit
    )
end

"""
    DFOResult

Result structure for DFO-TR optimization.
Corresponds to the `result` class in the Python implementation.
"""
struct DFOResult
    x::Vector{Float64}       # Optimal point found
    fun::Float64             # Optimal value
    iteration::Int           # Number of iterations
    iter_suc::Int           # Number of successful iterations
    func_eval::Int          # Number of function evaluations
    delta::Float64          # Final trust region radius
end

"""
    build_initial_sample(x_initial::Vector{Float64}, delta::Float64, sample_gen::String="auto")

Build initial sample set for DFO-TR algorithm.
Corresponds to `_build_initial_sample` in Python implementation.
"""
function build_initial_sample(x_initial::Vector{Float64}, delta::Float64, sample_gen::String="auto")
    n = length(x_initial)
    
    if sample_gen == "auto"
        # Create coordinate directions: +/- 0.5*delta*e_i
        Y = zeros(n, 2*n + 1)
        Y[:, 1] = x_initial  # Center point
        
        for i in 1:n
            Y[:, i+1] = x_initial + 0.5 * delta * Utils.unit_vector(n, i)
            Y[:, n+i+1] = x_initial - 0.5 * delta * Utils.unit_vector(n, i)
        end
    else
        error("Only 'auto' sample generation is currently supported")
    end
    
    return Y
end

"""
    dfo_tr(bb_func, x_initial::Vector{Float64}; options...)

Main DFO-TR trust region algorithm.

# Arguments
- `bb_func`: Black-box function to minimize (takes Vector{Float64}, returns Float64)
- `x_initial`: Starting point
- `options`: Keyword arguments for DFOParams and verbosity

# Returns
- `DFOResult`: Optimization result
"""
function dfo_tr(bb_func, x_initial::Vector{Float64}; 
                verbosity::Int=1, sample_gen::String="auto", kwargs...)
    
    # Set parameters
    par = DFOParams(; kwargs...)
    
    # Initialize algorithm parameters
    n = length(x_initial)
    maxY = div((n+1) * (n+2), 2)  # Maximum points for quadratic model
    minY = n + 1                  # Minimum points for quadratic model
    
    # Initialize counters
    iteration = 0
    iter_suc = 0
    func_eval = 0
    delta = par.init_delta
    
    # Build initial sample set
    Y = build_initial_sample(x_initial, delta, sample_gen)
    nY = size(Y, 2)
    
    # Evaluate function at sample points
    f_values = zeros(nY)
    for i in 1:nY
        f_values[i] = bb_func(Y[:, i])
        func_eval += 1
    end
    
    # Find best point as initial center
    best_idx = argmin(f_values)
    x = copy(Y[:, best_idx])
    f = f_values[best_idx]
    
    # Print initial report
    if verbosity > 0
        println("\n Iteration Report \n")
        println("|it |suc|  objective  | TR_radius  |    rho    | |Y|")
        println("| $iteration |---| $(round(f, digits=6)) | $(round(delta, digits=6)) | --------- | $nY")
    end
    
    # Main trust region loop
    while true
        success = false
        
        # Build quadratic model
        H, g = ModelBuilder.quad_frob(Y, f_values)
        
        # Check stopping criteria
        normg = norm(g)
        if normg <= par.tol_norm_g || delta < par.tol_delta
            break
        end
        
        # Solve trust region subproblem
        if norm(H, 2) > par.tol_norm_H
            s, val = TrustRegion.trust_sub(g, H, delta)
        else
            # Steepest descent step
            s = -(delta / normg) * g
            val = dot(g, s) + 0.5 * dot(s, H * s)
        end
        
        # Check for very small reduction
        if abs(val) < par.tol_f || iteration > par.maxfev
            break
        end
        
        # Evaluate function at trial point
        x_trial = x + s
        f_trial = bb_func(x_trial)
        func_eval += 1
        
        # Compute ratio of actual to predicted reduction
        pred = -val  # Predicted reduction (val is negative)
        ared = f - f_trial  # Actual reduction
        rho = pred > 0 ? ared / pred : -1.0
        
        # Update iterate and trust region radius
        if rho >= par.eta0 && ared > 0
            # Accept step
            success = true
            iter_suc += 1
            x = copy(x_trial)
            f = f_trial
            
            # Expand radius if very successful
            if rho >= par.eta2
                delta = min(par.gamma2 * delta, par.max_delta)
            end
        else
            # Reject step, shrink radius if we have enough points
            if nY >= minY
                delta = par.gamma1 * delta
            end
        end
        
        iteration += 1
        
        # Print iteration report
        if verbosity > 0
            println("| $iteration | $(success ? 1 : 0) | $(round(f, digits=6)) | $(round(delta, digits=6)) | $(round(rho, digits=6)) | $nY")
            if verbosity > 1
                println("x = ", x')
            end
        end
        
        # Update sample set
        Y, f_values, nY = Utils.update_sample_set(Y, f_values, x, x_trial, f_trial, success, maxY)
        
        # Criticality step handling
        if delta < par.min_del_crit && norm(s) < par.min_s_crit
            # Keep only points within 100*delta radius
            distances = [norm(Y[:, i] - x) for i in 1:nY]
            keep_indices = findall(d -> d < 100 * delta, distances)
            Y = Y[:, keep_indices]
            f_values = f_values[keep_indices]
            nY = length(keep_indices)
        end
    end
    
    # Final report
    if verbosity > 0
        println("\n*****************REPORT************************")
        println("Norm of the gradient of the model is $normg.")
        println("***************Final Report**************")
        println("|iter | #success| #fevals| final fvalue | final tr_radius|")
        println("| $iteration |    $iter_suc   |   $func_eval   |   $f   |  $delta  ")
    end
    
    return DFOResult(x, f, iteration, iter_suc, func_eval, delta)
end

end # module
