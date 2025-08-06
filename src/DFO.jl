module DFO

export sphere

"""Simple Sphere test function"""
sphere(x::AbstractVector) = sum(abs2, x)

end # module
