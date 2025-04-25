using OMEinsum
using Graphs, GraphIO
using CUDA, LinearAlgebra
using CSV, DataFrames
using TropicalNumbers, CuTropicalGEMM
using GenericTensorNetworks

function profile(id::Int, sc)
    graph = loadgraph("../networks/sc$(sc)/graph_$id.dot")
    code = readjson("../networks/sc$(sc)/eincode_$id.json")

    tn = GenericTensorNetwork(IndependentSet(graph), code, Dict{Int, Int}())
    @show contraction_complexity(tn)

    # warmup
    res = solve(tn, SizeMax(), T = Float32, usecuda = true)
    @show id, Array(res)[]
    
    CUDA.@profile solve(tn, SizeMax(), T = Float32, usecuda = true)    

    return nothing
end

function main(sc)
    profile(1, sc)
end

main(31)
