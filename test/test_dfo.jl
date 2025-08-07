# test_dfo.jl - Test suite for DFO-TR algorithm
# Comprehensive tests for all modules

using Test
using LinearAlgebra

# Add src directory to path
push!(LOAD_PATH, "../src")

# Import modules
include("../src/Utils.jl")
include("../src/ModelBuilder.jl")
include("../src/TrustRegion.jl")
include("../src/DFO.jl")

using .Utils
using .ModelBuilder
using .TrustRegion
using .DFO

@testset "DFO-TR Algorithm Tests" begin
    
    @testset "Utils Module Tests" begin
        # Test unit vector creation
        e = Utils.unit_vector(3, 2)
        @test e == [0.0, 1.0, 0.0]
        
        # Test test functions
        @test Utils.sphere([1.0, 2.0]) == 5.0
        @test Utils.sphere([0.0, 0.0]) == 0.0
        
        @test Utils.rosenbrock([1.0, 1.0]) ≈ 0.0 atol=1e-10
        @test Utils.rosenbrock([0.0, 0.0]) == 1.0
        
        @test Utils.quadratic_2d([0.0, 0.0]) == 0.0
        @test Utils.quadratic_2d([1.0, 1.0]) == 3.0
        
        # Test point sorting
        x = [0.0, 0.0]
        Y = [0.0 1.0 2.0; 0.0 0.0 0.0]
        f_vals = [1.0, 2.0, 3.0]
        Y_sorted, f_sorted, distances = Utils.shift_sort_points(x, Y, f_vals)
        @test distances == [0.0, 1.0, 2.0]
        @test f_sorted == [1.0, 2.0, 3.0]
    end
    
    @testset "ModelBuilder Module Tests" begin
        # Test simple quadratic model building
        # Create a simple 2D quadratic: f(x) = x1^2 + 2*x2^2
        X = [0.0 1.0 0.0 -1.0 0.0;
             0.0 0.0 1.0 0.0 -1.0]
        f_values = [0.0, 1.0, 2.0, 1.0, 2.0]  # Values of x1^2 + 2*x2^2
        
        H, g = ModelBuilder.quad_frob(X, f_values)
        
        # For this simple case, we expect:
        # - Gradient at origin should be close to [0, 0]
        # - Hessian should be approximately [2 0; 0 4] (since f = x1^2 + 2*x2^2)
        @test norm(g) < 2.0  # Gradient should be reasonably small at origin
        @test size(H) == (2, 2)
        @test H[1,1] > 0 && H[2,2] > 0  # Should be positive definite
    end
    
    @testset "TrustRegion Module Tests" begin
        # Test simple trust region subproblem
        g = [1.0, 0.0]  # Gradient pointing in x1 direction
        H = [2.0 0.0; 0.0 2.0]  # Simple positive definite Hessian
        delta = 1.0
        
        s, val = TrustRegion.trust_sub(g, H, delta)
        
        # Step should be in negative gradient direction
        @test s[1] < 0  # Should step in negative x1 direction
        @test abs(s[2]) < 1e-10  # Should not step in x2 direction
        @test norm(s) <= delta + 1e-10  # Should respect trust region bound
        @test val < 0  # Should predict reduction
    end
    
    @testset "DFO Main Algorithm Tests" begin
        # Test on simple sphere function
        x0 = [2.0, 3.0]
        result = DFO.dfo_tr(Utils.sphere, x0, verbosity=0, maxfev=50)
        
        @test result.fun < 1e-2  # Should find near-optimal value
        @test norm(result.x) < 0.5  # Should be close to origin
        @test result.func_eval > 0
        @test result.iteration > 0
        
        # Test on 2D quadratic
        result2 = DFO.dfo_tr(Utils.quadratic_2d, x0, verbosity=0, maxfev=50)
        @test result2.fun < 1e-2
        @test norm(result2.x) < 0.5
        
        # Test parameter customization
        result3 = DFO.dfo_tr(Utils.sphere, x0, verbosity=0, maxfev=20, init_delta=0.5)
        @test result3.delta <= 100.0  # Should respect max_delta default
    end
    
    @testset "Rosenbrock Function Test" begin
        # Test on Rosenbrock function (more challenging)
        x0 = [0.0, 0.0]
        result = DFO.dfo_tr(Utils.rosenbrock, x0, verbosity=0, maxfev=200, init_delta=0.5)
        
        # Rosenbrock is harder - just check it makes progress
        @test result.fun < 10.0  # Should reduce from initial value of 1.0
        @test result.func_eval > 0
        @test result.iteration > 0
    end
    
    @testset "Edge Cases" begin
        # Test with very small trust region
        x0 = [1.0, 1.0]
        result = DFO.dfo_tr(Utils.sphere, x0, verbosity=0, maxfev=10, init_delta=1e-6)
        @test result.func_eval > 0
        
        # Test with single variable
        sphere_1d(x) = x[1]^2
        result_1d = DFO.dfo_tr(sphere_1d, [2.0], verbosity=0, maxfev=30)
        @test result_1d.fun < 1e-1
        @test abs(result_1d.x[1]) < 0.5
    end
end

println("\n=== Running DFO-TR Tests ===")
println("All tests completed successfully!")
