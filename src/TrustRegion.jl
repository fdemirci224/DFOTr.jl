module TrustRegion

"""
TrustRegion.jl - Trust region subproblem solver
Port of trust_sub.py from https://github.com/TheClimateCorporation/dfo-algorithm
"""

using LinearAlgebra

export trust_sub

"""
    secular_eqn(lambda_val::Float64, eigval::Vector{Float64}, alpha::Vector{Float64}, delta::Float64)

Evaluate the secular equation at lambda_val.
Corresponds to `_secular_eqn` in Python implementation.
"""
function secular_eqn(lambda_val::Float64, eigval::Vector{Float64}, alpha::Vector{Float64}, delta::Float64)
    n = length(eigval)
    
    # Compute denominator: eigval + lambda_val
    denom = eigval .+ lambda_val
    
    # Handle division by zero
    ratio = zeros(n)
    for i in 1:n
        if abs(denom[i]) > eps(Float64)
            ratio[i] = alpha[i] / denom[i]
        else
            ratio[i] = 0.0  # Set to zero when denominator is zero
        end
    end
    
    # Compute norm squared
    norm_sq = sum(ratio.^2)
    
    # Return secular equation value
    if norm_sq > 0
        return 1.0/delta - 1.0/sqrt(norm_sq)
    else
        return 1.0/delta
    end
end

"""
    rfzero(x::Float64, itbnd::Int, eigval::Vector{Float64}, alpha::Vector{Float64}, delta::Float64, tol::Float64)

Find zero of secular equation to the right of x using modified bisection/interpolation.
Corresponds to `rfzero` in Python implementation.
"""
function rfzero(x::Float64, itbnd::Int, eigval::Vector{Float64}, alpha::Vector{Float64}, delta::Float64, tol::Float64)
    itfun = 0
    
    # Initial step size
    dx = abs(x) > 0 ? abs(x) / 2 : 0.5
    
    # Initial points
    a = x
    c = a
    fa = secular_eqn(a, eigval, alpha, delta)
    itfun += 1
    
    b = x + max(dx, 1.0)
    fb = secular_eqn(b, eigval, alpha, delta)
    itfun += 1
    
    # Find sign change
    while (fa > 0) == (fb > 0) && itfun < itbnd
        dx *= 2
        b = x + dx
        fb = secular_eqn(b, eigval, alpha, delta)
        itfun += 1
    end
    
    fc = fb
    
    # Initialize variables for interpolation
    d = b - a
    e = d
    
    # Main iteration loop
    while abs(fb) > tol && itfun < itbnd
        # Ensure b is best result, c is on opposite side
        if (fb > 0) == (fc > 0)
            c = a
            fc = fa
            d = b - a
            e = d
        end
        
        if abs(fc) < abs(fb)
            a, b, c = b, c, a
            fa, fb, fc = fb, fc, fa
        end
        
        # Convergence test
        m = 0.5 * (c - b)
        rel_tol = 2.0 * tol * max(abs(b), 1.0)
        
        if abs(m) <= rel_tol || abs(fb) < tol
            break
        end
        
        # Choose bisection or interpolation
        if abs(e) < rel_tol || abs(fa) <= abs(fb)
            # Bisection
            d = e = m
        else
            # Interpolation
            s = fb / fa
            if a == c
                # Linear interpolation
                p = 2.0 * m * s
                q = 1.0 - s
            else
                # Inverse quadratic interpolation
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            end
            
            if p > 0
                q = -q
            else
                p = -p
            end
            
            # Check if interpolation is acceptable
            if 2.0 * p < 3.0 * m * q - abs(rel_tol * q) && p < abs(0.5 * e * q)
                e = d
                d = p / q
            else
                d = e = m
            end
        end
        
        # Next point
        a = b
        fa = fb
        
        if abs(d) > rel_tol
            b += d
        else
            b += b > c ? -rel_tol : rel_tol
        end
        
        fb = secular_eqn(b, eigval, alpha, delta)
        itfun += 1
    end
    
    return b, c, itfun
end

"""
    compute_step(alpha::Vector{Float64}, eigval::Vector{Float64}, V::Matrix{Float64}, lambda_val::Float64)

Compute trust region step given eigendecomposition and Lagrange multiplier.
Corresponds to `compute_step` in Python implementation.
"""
function compute_step(alpha::Vector{Float64}, eigval::Vector{Float64}, V::Matrix{Float64}, lambda_val::Float64)
    n = length(eigval)
    w = eigval .+ lambda_val
    
    # Compute coefficients
    coeff = zeros(n)
    for i in 1:n
        if abs(w[i]) > eps(Float64)
            coeff[i] = alpha[i] / w[i]
        elseif abs(alpha[i]) < eps(Float64)
            coeff[i] = 0.0
        else
            coeff[i] = Inf  # This case should be handled by caller
        end
    end
    
    # Replace NaN and Inf with 0
    coeff[.!isfinite.(coeff)] .= 0.0
    
    # Compute step
    s = V * coeff
    nrms = norm(s)
    
    return coeff, s, nrms, w
end

"""
    trust_sub(g::Vector{Float64}, H::Matrix{Float64}, delta::Float64)

Solve trust region subproblem: min g'*s + 0.5*s'*H*s subject to ||s|| <= delta

Uses More-Sorensen approach with eigendecomposition and secular equation.
Corresponds to `trust_sub` in Python implementation.

# Arguments
- `g`: Gradient vector
- `H`: Hessian matrix
- `delta`: Trust region radius

# Returns
- `s`: Trust region step
- `val`: Predicted reduction in objective
"""
function trust_sub(g::Vector{Float64}, H::Matrix{Float64}, delta::Float64)
    tol = 1e-12
    tol_seqeq = 1e-8
    itbnd = 50
    s_factor = 0.8
    b_factor = 1.2
    
    n = length(g)
    
    # Eigendecomposition of symmetrized Hessian
    H_sym = 0.5 * (H + H')
    eigval, V = eigen(H_sym)
    
    # Find minimum eigenvalue
    jmin = argmin(eigval)
    mineig = eigval[jmin]
    
    # Project gradient onto eigenvector basis
    alpha = -V' * g
    sig = sign(alpha[jmin]) + (alpha[jmin] == 0 ? 1 : 0)
    
    lambda_val = 0.0
    key = 0
    
    # Positive definite case
    if mineig > 0
        lambda_val = 0.0
        coeff, s, nrms, w = compute_step(alpha, eigval, V, lambda_val)
        
        if nrms < b_factor * delta
            key = 1
        else
            laminit = 0.0
        end
    else
        laminit = -mineig
    end
    
    # Indefinite case - solve secular equation
    if key == 0
        if secular_eqn(laminit, eigval, alpha, delta) > 0
            lambda_val, c, count = rfzero(laminit, itbnd, eigval, alpha, delta, tol)
            
            if abs(secular_eqn(lambda_val, eigval, alpha, delta)) <= tol_seqeq
                key = 2
                coeff, s, nrms, w = compute_step(alpha, eigval, V, lambda_val)
                
                if nrms > b_factor * delta || nrms < s_factor * delta
                    key = 5
                    lambda_val = -mineig
                end
            else
                key = 3
                lambda_val = -mineig
            end
        else
            key = 4
            lambda_val = -mineig
        end
        
        # Handle hard case
        if key > 2
            # Zero out components where eigval + lambda is nearly zero
            alpha_mod = copy(alpha)
            for i in 1:n
                if abs(eigval[i] + lambda_val) < 10 * eps(Float64) * max(abs(eigval[i]), 1.0)
                    alpha_mod[i] = 0.0
                end
            end
            
            coeff, s, nrms, w = compute_step(alpha_mod, eigval, V, lambda_val)
            
            # Add component in null space if step too small
            if nrms < s_factor * delta
                beta = sqrt(delta^2 - nrms^2)
                s += beta * sig * V[:, jmin]
            end
            
            # Solve secular equation if step too large
            if nrms > b_factor * delta
                lambda_val, c, count = rfzero(laminit, itbnd, eigval, alpha_mod, delta, tol)
                coeff, s, nrms, w = compute_step(alpha_mod, eigval, V, lambda_val)
            end
        end
    end
    
    # Compute predicted reduction
    val = dot(g, s) + 0.5 * dot(s, H * s)
    
    return s, val
end

end # module
