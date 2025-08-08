module TrustRegion

"""
TrustRegion.jl - Trust region subproblem solver for DFO-TR
"""

using LinearAlgebra

export trust_sub

"""
    secular_eqn(lambda_val::Float64, eigval::Vector{Float64}, alpha::Vector{Float64}, delta::Float64)

Evaluate the secular equation at lambda_val.
"""
function secular_eqn(lambda_val::Float64, eigval::Vector{Float64}, alpha::Vector{Float64}, delta::Float64)
    n = length(eigval)

    # Robust masked division: ratio_i = alpha_i / (eigval_i + lambda), zero where denom ~ 0
    denom = eigval .+ lambda_val
    reltol = 1e-15
    scale = abs.(eigval) .+ abs(lambda_val) .+ 1.0
    mask = abs.(denom) .> (reltol .* scale)

    ratio = zeros(n)
    @inbounds ratio[mask] .= alpha[mask] ./ denom[mask]

    # Compute norm squared
    norm_sq = sum(ratio.^2)

    # Return secular equation value: 1/δ - 1/||ratio||
    if norm_sq > 0
        return 1.0 / delta - 1.0 / sqrt(norm_sq)
    else
        return 1.0 / delta
    end
end

"""
    rfzero(x::Float64, itbnd::Int, eigval::Vector{Float64}, alpha::Vector{Float64}, delta::Float64, tol::Float64)

Find root of secular equation using Ridders' method.
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
"""
function compute_step(alpha::Vector{Float64}, eigval::Vector{Float64}, V::Matrix{Float64}, lambda_val::Float64)
    n = length(eigval)
    w = eigval .+ lambda_val

    # Robust, Python-equivalent safe division:
    # coeff = alpha / (eigval + lambda), with zeros where denominator ~ 0
    # Use a relative tolerance to decide near-zero denominators
    reltol = 1e-15
    scale = abs.(eigval) .+ abs(lambda_val) .+ 1.0
    mask = abs.(w) .> (reltol .* scale)

    coeff = zeros(n)
    @inbounds coeff[mask] .= alpha[mask] ./ w[mask]

    # Compute step
    s = V * coeff
    nrms = norm(s)

    return coeff, s, nrms, w
end

"""
    trust_sub(g, H, delta; verbosity=0)

Solve trust region subproblem: min g'*s + 0.5*s'*H*s subject to ||s|| <= delta

Uses More-Sorensen method with eigendecomposition.
"""
function trust_sub(g::Vector{Float64}, H::Matrix{Float64}, delta::Float64; verbosity::Int=0)
    tol = 1e-12
    tol_seqeq = 1e-8
    itbnd = 50
    s_factor = 0.8
    b_factor = 1.2
    
    n = length(g)
    
    # Initialize variables that must be defined in all paths
    s = zeros(n)
    coeff = zeros(n)
    nrms = 0.0
    w = zeros(n)
    
    # Eigendecomposition of symmetrized Hessian
    # D, V = LA.eigh(0.5 * (H.T + H))
    H_sym = Symmetric(0.5 * (H + H'))  
    eigval, V = eigen(H_sym)
    
    # Find minimum eigenvalue
    jmin = argmin(eigval)
    mineig = eigval[jmin]
    
    # Project gradient onto eigenvector basis
    alpha = -V' * g
    # Robust augmentation orientation: choose direction that decreases model
    # Equivalent to sig = -sign(dot(V[:, jmin], g)), with near-zero safeguard
    vmin = V[:, jmin]
    vmin_g = dot(vmin, g)
    eps_scale = 1.0e-14 * max(norm(g), 1.0)
    sig = abs(vmin_g) < eps_scale ? 1.0 : -sign(vmin_g)
    
    lambda_val = 0.0
    key = 0
    laminit = 0.0
    
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
            # Hard case: solution is on the boundary.
            lambda_val = -mineig  # Default lambda for hard case
            if jmin > 0 && abs(alpha[jmin]) < 1.0e-8 * norm(alpha)
                if verbosity >= 4
                    println("--- HARD CASE ---")
                end
                lambda_val = -eigval[jmin]
                if verbosity >= 4
                    println("Initial lambda_val = ", lambda_val)
                end
                
                alpha_mod = copy(alpha)
                alpha_mod[jmin] = 0.0
                
                coeff, s, nrms, w = compute_step(alpha_mod, eigval, V, lambda_val)
                if verbosity >= 4
                    println("After first compute_step: nrms = ", nrms, ", delta = ", delta)
                end

                # This logic must exactly match the Python reference's if/elif structure.
                if nrms < s_factor * delta
                    # Add component in null space if step too small
                    beta = sqrt(delta^2 - nrms^2)
                    s += beta * sig * V[:, jmin]
                    if verbosity >= 4
                        println("Step too small. beta = ", beta, ", new_nrms = ", norm(s))
                    end
                elseif nrms > b_factor * delta
                    # Re-solve secular equation if step too large
                    if verbosity >= 4
                        println("Step too large. Calling rfzero...")
                    end
                    lambda_val, c, count = rfzero(laminit, itbnd, eigval, alpha_mod, delta, tol)
                    if verbosity >= 4
                        println("rfzero returned lambda_val = ", lambda_val)
                    end
                    coeff, s, nrms, w = compute_step(alpha_mod, eigval, V, lambda_val)
                    if verbosity >= 4
                        println("After rfzero and compute_step: nrms = ", nrms)
                    end
                end
                if verbosity >= 4
                    println("--- END HARD CASE ---")
                end
            else
                # Simple hard case: compute step with lambda = -mineig and add null-space component
                coeff, s, nrms, w = compute_step(alpha, eigval, V, lambda_val)
                # Augment along the minimum-eigenvalue direction to reach the boundary
                if nrms < (1.0 - 1e-14) * delta
                    beta = sqrt(max(delta^2 - nrms^2, 0.0))
                    s += beta * sig * V[:, jmin]
                end
            end
        end
    end
    
    # Compute predicted reduction
    val = dot(g, s) + 0.5 * dot(s, H * s)
    
    return s, val
end

end # module
