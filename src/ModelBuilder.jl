# ModelBuilder.jl - Quadratic model construction for DFO-TR

module ModelBuilder

using LinearAlgebra

export quad_frob

"""
    quad_frob(X, F_values)

Build quadratic model from sample points and function values.
Returns Hessian H and gradient g for model: g'*s + 0.5*s'*H*s + constant.

Uses minimum Frobenius norm model if underdetermined, full interpolation otherwise.
"""
function quad_frob(X::AbstractMatrix{Float64}, F_values::AbstractVector{Float64})
    tol_svd = eps(Float64)^5
    n, m = size(X)
    
    H = zeros(n, n)
    g = zeros(n, 1)
    
    # Shift points relative to first point
    Y = zeros(n, m)
    x0 = X[:, 1]
    @inbounds for i in 1:m
        for j in 1:n
            Y[j, i] = X[j, i] - x0[j]
        end
    end
    
    max_points = div((n+1) * (n+2), 2)
    
    if m < max_points
        # Minimum Frobenius norm model
        b = vcat(F_values, zeros(n+1))
        
        YTY = Y' * Y
        A = 0.5 * (YTY .* YTY)
        
        C = hcat(ones(m, 1), Y')
        top = hcat(A, C)
        bottom = hcat(C', zeros(n + 1, n + 1))
        W = vcat(top, bottom)
        
        Fqr = qr(W, ColumnNorm())
        lambda_0 = Fqr \ b
        
        g = lambda_0[m+2:end]
        
        fill!(H, 0.0)
        @inbounds for j in 1:m
            coeff = lambda_0[j]
            if abs(coeff) > 1e-16
                y_j = @view Y[:, j]
                for i1 in 1:n
                    for i2 in 1:n
                        H[i1, i2] += coeff * y_j[i1] * y_j[i2]
                    end
                end
            end
        end
        
    else
        # Full interpolation model
        b = F_values
        
        num_quad_terms = div(n * (n + 1), 2)
        phi_Q = zeros(m, num_quad_terms)
        
        @inbounds for i in 1:m
            y = @view Y[:, i]
            aux_idx = 1
            for j1 in 1:n
                for j2 in j1:n
                    if j1 == j2
                        phi_Q[i, aux_idx] = 0.5 * y[j1]^2
                    else
                        phi_Q[i, aux_idx] = y[j1] * y[j2]
                    end
                    aux_idx += 1
                end
            end
        end
        
        W = hcat(ones(m, 1), Y', phi_Q)
        
        Fqr = qr(W, ColumnNorm())
        lambda_0 = Fqr \ b

        g = lambda_0[2:n+1]
        
        fill!(H, 0.0)
        cont = n + 2
        
        @inbounds for j in 1:n
            len_j = n - j + 1
            for k in 1:len_j
                H[j+k-1, j] = lambda_0[cont + k - 1]
            end
            cont += len_j
        end
        
        # Build symmetric Hessian from lower triangular part
        H_diag = diag(H)
        H = H + H' - diagm(H_diag)
    end
    
    return H, vec(g)
end

end # module
