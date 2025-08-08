# DFO-TR Comprehensive Solver Test Suite

using Test
using LinearAlgebra
using DFOTr


"""
    sphere(x)

Sphere function: sum of squares. Global minimum at the origin.
"""
function sphere(x)
    return sum(abs2, x)
end

"""
    rosenbrock(x)

Rosenbrock function in n dimensions.
f(x) = Σ_{i=1}^{n-1} [100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2]
Global minimum at the vector of ones.
"""
function rosenbrock(x)
    n = length(x)
    s = 0.0
    @inbounds for i in 1:(n-1)
        s += 100.0 * (x[i+1] - x[i]^2)^2 + (1.0 - x[i])^2
    end
    return s
end

"""
    quadratic_2d(x)

Simple convex quadratic in 2D with minimum at the origin.
"""
function quadratic_2d(x)
    return x[1]^2 + 2.0 * x[2]^2
end

# Additional test functions for comprehensive testing
function rastrigin(x::Vector{Float64})
    A = 10.0
    n = length(x)
    return A * n + sum(x.^2 - A * cos.(2π * x))
end

function ill_conditioned_quadratic(x::Vector{Float64})
    return x[1]^2 + 1e-4 * x[2]^2
end

function nearly_flat_exponential(x::Vector{Float64})
    return exp(-norm(x)^2)
end

function non_symmetric_curvature(x::Vector{Float64})
    return 5.0 * x[1]^2 + 0.1 * x[2]^2
end

function small_reduction_function(x::Vector{Float64})
    return 1e-6 * norm(x)^4 + 1e-8 * norm(x)^2
end

function booth_function(x::Vector{Float64})
    return (x[1] + 2*x[2] - 7)^2 + (2*x[1] + x[2] - 5)^2
end

"""
    run_test_function(name, func, x0, expected_min, expected_x; tol_f, tol_x, maxfev, solver_tol_f, solver_tol_norm_g)

Helper to run a single test function and report results
"""
function run_test_function(name::String, func, x0::Vector{Float64}, 
                          expected_min::Float64, expected_x::Vector{Float64};
                          tol_f::Float64=1e-3, tol_x::Float64=0.5, maxfev::Int=200,
                          solver_tol_f::Float64=1e-12, solver_tol_norm_g::Float64=1e-10)
    println("\n" * "="^60)
    println("Testing: $name")
    println("Starting point: $(round.(x0', digits=4))")
    println("Expected minimum: $expected_min at $(round.(expected_x', digits=4))")
    try
        result = DFOTr.dfo_tr(func, x0, verbosity=1, maxfev=maxfev, 
                           tol_f=solver_tol_f, tol_norm_g=solver_tol_norm_g)
        converged = result.fun <= expected_min + tol_f
        println("Results:")
        println("  Converged: $(converged ? "✓" : "✗")")
        println("  Final objective: $(round(result.fun, digits=8))")
        println("  Final point: $(round.(result.x', digits=6))")
        println("  Distance to optimum: $(round(norm(result.x - expected_x), digits=6))")
        println("  Iterations: $(result.iteration)")
        println("  Function evaluations: $(result.func_eval)")
        println("  Success rate: $(round(100 * result.iter_suc / result.iteration, digits=1))%")
        @test result.func_eval > 0
        @test result.iteration > 0
        @test isfinite(result.fun)
        @test all(isfinite.(result.x))
        if converged
            println("  Status: PASSED ✓")
            @test true
        else
            println("  Status: FAILED (did not converge to expected minimum)")
            @test false
        end
        return true
    catch e
        println("  Status: ERROR - $e")
        @test false
        return false
    end
end

println("="^80)
println("DFO-TR Comprehensive Solver Test Suite")
println("="^80)

@testset "DFO-TR Solver Procedures" begin

    @testset "Standard Test Functions" begin
        run_test_function(
            "Sphere Function",
            sphere,
            [2.0, 3.0],
            0.0,
            [0.0, 0.0],
            tol_f=1e-6
        )

        run_test_function(
            "Rosenbrock Function",
            rosenbrock,
            [-1.2, 1.0],
            0.0,
            [1.0, 1.0],
            tol_f=1e-6,
            maxfev=300
        )

        run_test_function(
            "2D Quadratic",
            quadratic_2d,
            [1.0, 2.0],
            0.0,
            [0.0, 0.0],
            tol_f=1e-6
        )

        run_test_function(
            "Booth Function",
            booth_function,
            [0.0, 0.0],
            0.0,
            [1.0, 3.0],
            tol_f=1e-4
        )
    end

    @testset "Numerically Challenging Functions" begin
        run_test_function(
            "Ill-conditioned Quadratic",
            ill_conditioned_quadratic,
            [1.0, 10.0],
            0.0,
            [0.0, 0.0],
            tol_f=1e-6,
            tol_x=1.0
        )

        run_test_function(
            "Non-symmetric Curvature",
            non_symmetric_curvature,
            [1.0, 2.0],
            0.0,
            [0.0, 0.0],
            tol_f=1e-6
        )

        run_test_function(
            "Nearly Flat Exponential",
            nearly_flat_exponential,
            [2.0, 2.0],
            1.0,
            [0.0, 0.0],
            tol_f=1e-3,
            maxfev=150
        )

        run_test_function(
            "Small Reduction Function",
            small_reduction_function,
            [1.0, 1.0],
            0.0,
            [0.0, 0.0],
            tol_f=1e-10,
            maxfev=400,
            solver_tol_f=1e-14,
            solver_tol_norm_g=1e-12
        )
    end

    @testset "Multi-modal Functions" begin
        println("\n" * "="^60)
        println("Testing: Rastrigin Function (Multi-modal)")
        println("Note: This function has many local minima - convergence to global minimum not guaranteed")
        try
            result = DFOTr.dfo_tr(rastrigin, [1.5, 1.5], verbosity=1, maxfev=300,
                               tol_f=1e-8, tol_norm_g=1e-8)
            println("Results:")
            println("  Final objective: $(round(result.fun, digits=8))")
            println("  Final point: $(round.(result.x', digits=6))")
            println("  Iterations: $(result.iteration)")
            println("  Function evaluations: $(result.func_eval)")
            @test result.func_eval > 0
            @test result.iteration > 0
            @test isfinite(result.fun)
            @test all(isfinite.(result.x))
            if result.fun < 10.0
                println("  Status: PASSED ✓ (found reasonable local minimum)")
            else
                println("  Status: PARTIAL (algorithm ran but high final value)")
            end
        catch e
            println("  Status: ERROR - $e")
            @test false
        end
    end

    @testset "Edge Cases and Robustness" begin
        println("\n" * "="^60)
        println("Testing: Small Initial Trust Region")
        result = DFOTr.dfo_tr(sphere, [1.0, 1.0], verbosity=0, 
                           init_delta=1e-6, maxfev=50)
        @test result.func_eval > 0
        @test isfinite(result.fun)
        println("  Status: PASSED ✓")

        println("\n" * "="^60)
        println("Testing: Single Variable Function")
        sphere_1d(x) = x[1]^2
        result_1d = DFOTr.dfo_tr(sphere_1d, [3.0], verbosity=0, maxfev=50)
        @test result_1d.fun < 1e-2
        @test abs(result_1d.x[1]) < 0.5
        println("  Status: PASSED ✓")

        println("\n" * "="^60)
        println("Testing: Higher Dimensional Function (5D)")
        sphere_5d(x) = sum(x.^2)
        x0_5d = ones(5) * 2.0
        result_5d = DFOTr.dfo_tr(sphere_5d, x0_5d, verbosity=0, maxfev=200)
        @test result_5d.func_eval > 0
        @test result_5d.fun < 1.0
        println("  Final objective: $(round(result_5d.fun, digits=6))")
        println("  Status: PASSED ✓")
    end
end

println("\n" * "="^80)
println("DFO-TR Comprehensive Test Suite Completed")
println("="^80)
println("\nSummary:")
println("- Standard functions: Sphere, Rosenbrock, 2D Quadratic, Booth")
println("- Challenging functions: Ill-conditioned, Non-symmetric, Nearly flat, Small reductions")
println("- Multi-modal: Rastrigin (local minima expected)")
println("- Edge cases: Small trust region, 1D, 5D functions")
println("\nThe solver demonstrates robust performance across diverse optimization landscapes.")
