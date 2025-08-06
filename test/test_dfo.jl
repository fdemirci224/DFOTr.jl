include("../src/DFO.jl")
using Test, .DFO

@test sphere([1.0, 2.0]) == 5.0
