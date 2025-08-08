# DFOTrustRegion.jl - Main DFO-TR algorithm

module DFOTrustRegion

using LinearAlgebra
using Statistics
using Printf
include("ModelBuilder.jl")
include("TrustRegion.jl")
include("Utils.jl")
include("Logger.jl")

export DFOParams, DFOResult, dfo_tr

using .Logger

"""
    DFOParams

Parameters for the DFO-TR trust region algorithm.
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
    x::Vector{Float64}      # Optimal point found
    fun::Float64            # Optimal value
    iteration::Int          # Number of iterations
    iter_suc::Int           # Number of successful iterations
    func_eval::Int          # Number of function evaluations
    delta::Float64          # Final trust region radius
end

"""
    build_initial_sample(x_initial, delta, sample_gen="auto")

Build initial sample set for DFO-TR algorithm.
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
    dfo_tr(bb_func, x_initial; verbosity=1, log_file="", kwargs...)

Main DFO-TR trust region algorithm.

# Verbosity levels:
- 0: Silent mode (no output)
- 1: Standard mode (compact iteration summary)
- 2: Debug mode (detailed info + optional log file)
"""
function dfo_tr(bb_func, x_initial::Vector{Float64}; 
                verbosity::Int=1, log_file::String="", sample_gen::String="auto", kwargs...)
    
    # Set parameters
    par = DFOParams(; kwargs...)
    
    # Initialize logger
    logger = IterationLogger(verbosity)
    
    # Initialize algorithm parameters
    n = length(x_initial)
    maxY = div((n+1) * (n+2), 2)
    minY = n + 1
    
    # Initialize counters
    iteration = 0
    iter_suc = 0
    func_eval = 0
    delta = par.init_delta
    
    # Build initial sample set 
    x_initial_col = reshape(x_initial, n, 1)  
    Y_initial = build_initial_sample(x_initial, delta, sample_gen)
    nY = size(Y_initial, 2)
    
    # Pre-allocate larger matrices for efficiency
    max_points = maxY + 10  # Extra buffer
    Y = zeros(n, max_points)
    f_values = zeros(max_points)
    Y[:, 1:nY] = Y_initial
    
    # Pre-allocate working arrays
    distances = Vector{Float64}(undef, max_points)
    perm = Vector{Int}(undef, max_points)
    nY_ref = Ref(nY)
    
    # Point tracking for debug mode
    visited_points = Vector{Vector{Float64}}()
    visited_values = Float64[]
    
    # Evaluate function at sample points 
    for i in 1:nY
        f_values[i] = bb_func(Y[:, i])
        func_eval += 1
        if verbosity >= 2
            push!(visited_points, copy(@view Y[:, i]))
            push!(visited_values, f_values[i])
        end
    end
    
    # Find best point as initial center 
    best_idx = argmin(@view f_values[1:nY])
    x = copy(Y[:, best_idx])
    f = f_values[best_idx]
    
    # Print initial report
    if verbosity >= 1
        println("\nDFO-TR Iteration Report")
        @printf("| %3s | %3s | %13s | %11s | %11s | %5s\n", "it", "suc", "objective", "TR_radius", "rho", "|Y|")
        @printf("| %3d | %3s | %13.6e | %11.5f | %11s | %5d\n", iteration, "---", f, delta, "-----------", nY)
    end
    
    # Main trust region loop 
    normg = NaN
    while true
        success = 0  
        
        # Build quadratic model 
        H, g = ModelBuilder.quad_frob(@view(Y[:, 1:nY]), @view(f_values[1:nY]))
        
        # Check stopping criteria 
        normg = norm(g)
        if normg <= par.tol_norm_g || delta < par.tol_delta
            break
        end
        
        # Solve trust region subproblem 
        if norm(H, 2) > par.tol_norm_H  
            s, val = TrustRegion.trust_sub(g, H, delta; verbosity=verbosity)
        else
            # Steepest descent step
            s = -(delta / normg) * g
            val = dot(g, s) + 0.5 * dot(s, H * s)
        end
        
        # Model value at new point
        fmod = f + val
        
        # Check for very small reduction
        if abs(val) < par.tol_f || iteration > par.maxfev
            break
        end
        
        # Evaluate function at trial point
        x_trial = x + s
        f_trial = bb_func(x_trial)
        func_eval += 1
        if verbosity >= 2
            push!(visited_points, copy(x_trial))
            push!(visited_values, f_trial)
        end
        
        # Compute ratio of actual to predicted reduction
        pred = f - fmod  # Predicted reduction
        ared = f - f_trial  # Actual reduction
        eps_pred = 1e-12
        abs_acc_thresh = 1e-12
        denom = max(abs(pred), eps_pred)
        rho = ared / denom
        
        # Update iterate and trust region radius
        if (rho >= par.eta0 && (f - f_trial) > 0) || (ared >= abs_acc_thresh && abs(pred) <= eps_pred)
            # Accept step 
            success = 1
            iter_suc += 1
            
            # Update center
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
        
        # Print iteration report and log data
        if verbosity >= 1
            @printf("| %3d | %3d | %13.6e | %11.5f | %11.6f | %5d\n", iteration, success, f, delta, rho, nY)
        end
        
        # Print detailed iteration report
        if verbosity >= 2
            println("    x: ", round.(x', digits=6))
            println("    g: ", round.(g', digits=6), " (norm: ", round(normg, digits=6), ")")
            println("    s: ", round.(s', digits=6), " (norm: ", round(norm(s), digits=6), ")")
            println("    val: ", round(val, digits=6), " (predicted reduction)")
            # Hessian diagnostics
            Hdiag = [H[i,i] for i in 1:n]
            Heigs = eigvals(Symmetric(H))
            println("    H diag: ", round.(Hdiag', digits=6))
            k = min(n, 5)
            println("    H eigs[1:$k]: ", round.(Heigs[1:k]', digits=6))
            if n <= 6
                println("    H (full): ")
                println(round.(H, digits=6))
            end
            if success == 1
                println("    ared: ", round(ared, digits=6), " (actual reduction)")
                println("    rho: ", round(rho, digits=6))
            end
            
            # Log iteration data
            # Prepare compact Hessian for logging (lower-triangular vectorization)
            Hvech = Vector{Float64}(undef, div(n*(n+1),2))
            idx = 1
            @inbounds for j in 1:n
                for i in j:n
                    Hvech[idx] = H[i,j]
                    idx += 1
                end
            end
            iter_data = Dict{String, Any}(
                "iteration" => iteration,
                "success" => success,
                "objective" => f,
                "tr_radius" => delta,
                "rho" => rho,
                "norm_g" => normg,
                "step_norm" => norm(s),
                "predicted_reduction" => val,
                "actual_reduction" => success == 1 ? ared : NaN,
                "func_eval" => func_eval,
                "sample_size" => nY,
                "H_diag" => Hdiag,
                "H_eigs" => Heigs,
                "H_vech" => Hvech
            )
            log_iteration(logger, iter_data)
        end
        
        # Update sample set
        Utils.update_sample_set!(Y, f_values, nY_ref, x, x_trial, f_trial, success == 1, maxY, distances, perm)
        nY = nY_ref[]
        
        # Criticality step handling
        if delta < par.min_del_crit && norm(s) < par.min_s_crit
            # Keep only points within 100*delta radius
            kept_count = 0
            @inbounds for i in 1:nY
                if distances[i] < 100 * delta
                    kept_count += 1
                    if kept_count != i
                        Y[:, kept_count] = Y[:, i]
                        f_values[kept_count] = f_values[i]
                    end
                end
            end
            nY = kept_count
            nY_ref[] = nY
        end
    end
    
    # Final report
    if verbosity >= 1
        println("\n" * "="^50)
        println("DFO-TR Final Report")
        println("Gradient norm: $(round(normg, digits=8))")
        println("Iterations: $iteration | Successful: $iter_suc | Function evals: $func_eval")
        println("Final objective: $(round(f, digits=8))")
        println("Final trust radius: $(round(delta, digits=8))")
        println("="^50)
    end
    
    # Save log file if requested
    if verbosity >= 2 && !isempty(log_file)
        save_log(logger, log_file)
    end
    
    return DFOResult(x, f, iteration, iter_suc, func_eval, delta)
end

end # module
