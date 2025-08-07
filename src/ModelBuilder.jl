# ModelBuilder.jl - Quadratic model construction
# Port of quad_Frob.py from https://github.com/TheClimateCorporation/dfo-algorithm

module ModelBuilder

using LinearAlgebra

export quad_frob

"""
    compute_coeffs(W::Matrix{Float64}, tol_svd::Float64, b::Vector{Float64}, option::String)

Compute model coefficients by solving the system of equations using SVD.
Corresponds to `_compute_coeffs` in Python implementation.
"""
function compute_coeffs(W::Matrix{Float64}, tol_svd::Float64, b::Vector{Float64}, option::String)
    if option == "partial"
        U, S, Vt = svd(W)
    else
        U, S, Vt = svd(W, full=false)
    end
    
    # Regularize small singular values
    S[S .< tol_svd] .= tol_svd
    Sinv = diagm(1.0 ./ S)
    V = Vt'
    
    # Compute coefficients
    lambda_0 = V * Sinv * U' * b
    return lambda_0
end

"""
    quad_frob(X::Matrix{Float64}, F_values::Vector{Float64})

Build quadratic model from sample points and function values.
Returns Hessian H and gradient g for model: g'*s + 0.5*s'*H*s + α

If number of points < (n+1)(n+2)/2, builds minimum Frobenius norm model.
Otherwise, builds full interpolation model.

Corresponds to `quad_Frob` in Python implementation.

# Arguments
- `X`: Sample points matrix (n × m)
- `F_values`: Function values at sample points (m,)

# Returns
- `H`: Hessian matrix (n × n)
- `g`: Gradient vector (n,)
"""
function quad_frob(X::Matrix{Float64}, F_values::Vector{Float64})
    # Tolerance for SVD regularization
    eps_val = eps(Float64)
    tol_svd = eps_val^5
    
    n, m = size(X)
    
    # Initialize outputs
    H = zeros(n, n)
    g = zeros(n)
    
    # Shift points to origin (center at first point)
    Y = X .- X[:, 1:1]  # Broadcasting to subtract first column from all columns
    
    # Maximum points for full quadratic model
    max_points = div((n+1) * (n+2), 2)
    
    if m < max_points
        # Minimum Frobenius norm model (underdetermined case)
        # Solve KKT conditions from page 81 of Intro to DFO book
        
        b = vcat(F_values, zeros(n+1))
        
        # Construct quadratic terms matrix A
        A = 0.5 * (Y' * Y).^2
        
        # Build constraint matrix W
        # Top part: [A, ones, Y']
        top = hcat(A, ones(m, 1), Y')
        
        # Bottom part: [ones'; Y; zeros]
        temp = vcat(ones(1, m), Y)
        bottom = hcat(temp, zeros(n+1, n+1))
        
        W = vcat(top, bottom)
        
        # Solve for coefficients
        lambda_0 = compute_coeffs(W, tol_svd, b, "partial")
        
        # Extract gradient (linear coefficients)
        g = lambda_0[m+2:end]
        
        # Reconstruct Hessian from quadratic coefficients
        H = zeros(n, n)
        for j in 1:m
            y_j = Y[:, j]
            H += lambda_0[j] * (y_j * y_j')
        end
        
    else
        # Full interpolation model (determined/overdetermined case)
        b = F_values
        
        # Build polynomial basis matrix
        # Number of quadratic terms: n(n+1)/2
        num_quad_terms = div(n * (n + 1), 2)
        phi_Q = zeros(m, num_quad_terms)
        
        for i in 1:m
            y = Y[:, i]
            
            # Construct upper triangular part of y*y' - 0.5*diag(y.^2)
            aux_H = y * y' - 0.5 * diagm(y.^2)
            
            # Extract upper triangular elements
            aux = Float64[]
            for j in 1:n
                append!(aux, aux_H[j:n, j])
            end
            
            phi_Q[i, :] = aux
        end
        
        # Build full constraint matrix: [ones, Y', phi_Q]
        W = hcat(ones(m, 1), Y', phi_Q)
        
        # Solve for coefficients
        lambda_0 = compute_coeffs(W, tol_svd, b, "full")
        
        # Extract gradient
        g = lambda_0[2:n+1]
        
        # Reconstruct symmetric Hessian
        H = zeros(n, n)
        cont = n + 2
        
        for j in 1:n
            len_j = n - j + 1
            H[j:n, j] = lambda_0[cont:cont+len_j-1]
            cont += len_j
        end
        
        # Make H symmetric
        H = H + H' - diagm(diag(H))
    end
    
    return H, g
end

end # module
