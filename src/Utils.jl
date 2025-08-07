# Utils.jl - Utility functions
# Helper functions for DFO-TR algorithm

module Utils

using LinearAlgebra

export unit_vector, shift_sort_points, update_sample_set, sphere, rosenbrock, quadratic_2d

"""
    unit_vector(n::Int, i::Int)

Create unit vector of length n with 1.0 in position i.
"""
function unit_vector(n::Int, i::Int)
    e = zeros(n)
    e[i] = 1.0
    return e
end

"""
    shift_sort_points(x::Vector{Float64}, Y::Matrix{Float64}, f_values::Vector{Float64})

Shift points to origin (centered at x) and sort by distance from x.
Corresponds to `_shift_sort_points` in Python implementation.

# Arguments
- `x`: Current center point
- `Y`: Sample points matrix (n × m)
- `f_values`: Function values at sample points

# Returns
- `Y_sorted`: Sorted sample points
- `f_sorted`: Sorted function values
- `distances`: Distances from center (sorted)
"""
function shift_sort_points(x::Vector{Float64}, Y::Matrix{Float64}, f_values::Vector{Float64})
    n, m = size(Y)
    
    # Compute distances from center
    distances = zeros(m)
    for i in 1:m
        distances[i] = norm(Y[:, i] - x)
    end
    
    # Sort by distance
    perm = sortperm(distances)
    
    Y_sorted = Y[:, perm]
    f_sorted = f_values[perm]
    distances_sorted = distances[perm]
    
    return Y_sorted, f_sorted, distances_sorted
end

"""
    update_sample_set(Y::Matrix{Float64}, f_values::Vector{Float64}, x::Vector{Float64}, 
                     x_trial::Vector{Float64}, f_trial::Float64, success::Bool, maxY::Int)

Update sample set based on trial point and success status.
Corresponds to sample set update logic in Python dfo_tr implementation.

# Arguments
- `Y`: Current sample points matrix
- `f_values`: Current function values
- `x`: Current center point
- `x_trial`: Trial point
- `f_trial`: Function value at trial point
- `success`: Whether trial was successful
- `maxY`: Maximum number of sample points

# Returns
- `Y_new`: Updated sample points
- `f_new`: Updated function values
- `nY_new`: New number of sample points
"""
function update_sample_set(Y::Matrix{Float64}, f_values::Vector{Float64}, x::Vector{Float64},
                          x_trial::Vector{Float64}, f_trial::Float64, success::Bool, maxY::Int)
    n, nY = size(Y)
    
    # Sort points by distance from current center
    Y_sorted, f_sorted, distances = shift_sort_points(x, Y, f_values)
    
    if success
        # Successful iteration
        if nY < maxY
            # Add new point
            Y_new = hcat(Y_sorted, x_trial)
            f_new = vcat(f_sorted, f_trial)
            nY_new = nY + 1
        else
            # Replace furthest point
            Y_new = copy(Y_sorted)
            Y_new[:, end] = x_trial
            f_new = copy(f_sorted)
            f_new[end] = f_trial
            nY_new = nY
        end
    else
        # Unsuccessful iteration
        trial_distance = norm(x_trial - x)
        
        if nY >= maxY
            # Only add if closer than furthest point
            if trial_distance <= distances[end]
                Y_new = copy(Y_sorted)
                Y_new[:, end] = x_trial
                f_new = copy(f_sorted)
                f_new[end] = f_trial
                nY_new = nY
            else
                # Keep existing set
                Y_new = Y_sorted
                f_new = f_sorted
                nY_new = nY
            end
        else
            # Add new point
            Y_new = hcat(Y_sorted, x_trial)
            f_new = vcat(f_sorted, f_trial)
            nY_new = nY + 1
        end
    end
    
    # Re-sort the updated set
    Y_final, f_final, _ = shift_sort_points(x, Y_new, f_new)
    
    return Y_final, f_final, nY_new
end

# Test functions for validation

"""
    sphere(x::Vector{Float64})

Sphere test function: f(x) = sum(x_i^2)
Global minimum at origin with value 0.
"""
function sphere(x::Vector{Float64})
    return sum(x.^2)
end

"""
    rosenbrock(x::Vector{Float64})

Rosenbrock test function: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
Global minimum at (1,1,...,1) with value 0.
"""
function rosenbrock(x::Vector{Float64})
    n = length(x)
    if n < 2
        error("Rosenbrock function requires at least 2 variables")
    end
    
    result = 0.0
    for i in 1:(n-1)
        result += 100.0 * (x[i+1] - x[i]^2)^2 + (1.0 - x[i])^2
    end
    
    return result
end

"""
    quadratic_2d(x::Vector{Float64})

Simple 2D quadratic test function: f(x) = x[1]^2 + 2*x[2]^2
Global minimum at origin with value 0.
"""
function quadratic_2d(x::Vector{Float64})
    if length(x) != 2
        error("quadratic_2d requires exactly 2 variables")
    end
    return x[1]^2 + 2.0 * x[2]^2
end

end # module
