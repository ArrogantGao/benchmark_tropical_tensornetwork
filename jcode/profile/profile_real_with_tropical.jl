using GenericTensorNetworks
using OMEinsum
using Graphs, GraphIO
using CUDA, LinearAlgebra
using CSV, DataFrames

include("../tropical_gemm.jl")

function generate_tensors(g)
    tensors = []

    for i in 1:ne(g)
        push!(tensors, CUDA.rand(Float32, 2, 2))
    end

    for j in 1:nv(g)
        push!(tensors, CUDA.rand(Float32, 2))
    end

    return tensors
end

function profile(id::Int, sc)
    graph = loadgraph("../networks/sc$(sc)/graph_$id.dot")
    code = readjson("../networks/sc$(sc)/eincode_$id.json")

    tensors = generate_tensors(graph)

    # warmup
    res = code(tensors...)

    CUDA.@profile code(tensors...)

    return nothing
end

function main(sc)
    profile(1, sc)
end

main(31)
