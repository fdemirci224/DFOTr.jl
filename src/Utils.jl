# Utils.jl - Utility functions
# Helper functions for DFO-TR algorithm

module Utils

using LinearAlgebra

export unit_vector, shift_sort_points, update_sample_set

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
    shift_sort_points!(Y::Matrix{Float64}, f_values::Vector{Float64}, x::Vector{Float64}, 
                      distances::Vector{Float64}, perm::Vector{Int})

In-place version: shift points and sort by distance from x.
Corresponds to `_shift_sort_points` in Python implementation.

# Arguments (modified in-place)
- `Y`: Sample points matrix (n × m) - will be sorted
- `f_values`: Function values at sample points - will be sorted  
- `x`: Current center point
- `distances`: Pre-allocated distance buffer
- `perm`: Pre-allocated permutation buffer

# Returns
- `distances`: Sorted distances from center
"""
function shift_sort_points!(Y::AbstractMatrix{Float64}, f_values::AbstractVector{Float64}, x::Vector{Float64}, 
                           distances::AbstractVector{Float64}, perm::AbstractVector{Int})
    n, m = size(Y)
    
    # Compute distances from center (in-place)
    @inbounds for i in 1:m
        dist_sq = 0.0
        for j in 1:n
            diff = Y[j, i] - x[j]
            dist_sq += diff * diff
        end
        distances[i] = sqrt(dist_sq)
    end
    
    # Sort by distance (in-place)
    sortperm!(perm, distances)
    
    # Apply permutation in-place
    Y[:, :] = Y[:, perm]
    f_values[:] = f_values[perm]
    distances[:] = distances[perm]
    
    return distances
end

# Convenience wrapper that allocates (for backward compatibility)
function shift_sort_points(x::Vector{Float64}, Y::Matrix{Float64}, f_values::Vector{Float64})
    n, m = size(Y)
    distances = Vector{Float64}(undef, m)
    perm = Vector{Int}(undef, m)
    
    Y_copy = copy(Y)
    f_copy = copy(f_values)
    
    distances_sorted = shift_sort_points!(Y_copy, f_copy, x, distances, perm)
    return Y_copy, f_copy, distances_sorted
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

"""
    update_sample_set!(Y, f_values, nY_ref, x, x_trial, f_trial, success, maxY, distances, perm)

In-place sample set update matching Python dfo_tr.py logic exactly.
Corresponds to sample set update logic in Python dfo_tr implementation (lines 273-302).

# Arguments (modified in-place)
- `Y`: Sample points matrix - will be modified
- `f_values`: Function values - will be modified
- `nY_ref`: Reference to current number of points - will be updated
- `x`: Current center point
- `x_trial`: Trial point to potentially add
- `f_trial`: Function value at trial point
- `success`: Whether trial was successful
- `maxY`: Maximum number of sample points
- `distances`: Pre-allocated distance buffer
- `perm`: Pre-allocated permutation buffer
"""
function update_sample_set!(Y::AbstractMatrix{Float64}, f_values::AbstractVector{Float64}, nY_ref::Ref{Int},
                           x::Vector{Float64}, x_trial::Vector{Float64}, f_trial::Float64, 
                           success::Bool, maxY::Int, distances::AbstractVector{Float64}, perm::AbstractVector{Int})
    nY = nY_ref[]
    n = size(Y, 1)
    x_new = success ? x_trial : x

    # First, sort points by distance from the *new* center to find the furthest point
    Y_view_pre = @view Y[:, 1:nY]
    f_view_pre = @view f_values[1:nY]
    dist_view_pre = @view distances[1:nY]
    perm_view_pre = @view perm[1:nY]
    shift_sort_points!(Y_view_pre, f_view_pre, x_new, dist_view_pre, perm_view_pre)

    if success
        # Successful iteration: replace the furthest point or add the new one
        if nY >= maxY
            Y[:, nY] = x_trial
            f_values[nY] = f_trial
        else
            nY += 1
            Y[:, nY] = x_trial
            f_values[nY] = f_trial
            nY_ref[] = nY
        end
    else
        # Unsuccessful iteration: only add if it improves the geometry (closer than furthest) or if set is not full
        trial_distance = norm(x_trial - x_new)
        if nY >= maxY
            if trial_distance < distances[nY]
                Y[:, nY] = x_trial
                f_values[nY] = f_trial
            end
        else
            nY += 1
            Y[:, nY] = x_trial
            f_values[nY] = f_trial
            nY_ref[] = nY
        end
    end

    # Finally, re-sort the potentially modified set around the new center point
    nY = nY_ref[]
    Y_view_post = @view Y[:, 1:nY]
    f_view_post = @view f_values[1:nY]
    dist_view_post = @view distances[1:nY]
    perm_view_post = @view perm[1:nY]
    shift_sort_points!(Y_view_post, f_view_post, x_new, dist_view_post, perm_view_post)
end

# Backward compatibility wrapper
function update_sample_set(Y::Matrix{Float64}, f_values::Vector{Float64}, x::Vector{Float64},
                          x_trial::Vector{Float64}, f_trial::Float64, success::Bool, maxY::Int)
    n, nY = size(Y)
    
    # Allocate larger matrices to handle potential growth
    Y_new = zeros(n, maxY + 1)
    f_new = zeros(maxY + 1)
    Y_new[:, 1:nY] = Y
    f_new[1:nY] = f_values
    
    distances = Vector{Float64}(undef, maxY + 1)
    perm = Vector{Int}(undef, maxY + 1)
    nY_ref = Ref(nY)
    
    update_sample_set!(Y_new, f_new, nY_ref, x, x_trial, f_trial, success, maxY, distances, perm)
    
    nY_final = nY_ref[]
    return Y_new[:, 1:nY_final], f_new[1:nY_final], nY_final
end

end # module
