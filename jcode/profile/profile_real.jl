using GenericTensorNetworks
using OMEinsum
using Graphs, GraphIO
using CUDA, LinearAlgebra
using CSV, DataFrames


function generate_tensors(g)
    tensors = []

    for i in 1:ne(g)
        push!(tensors, rand(Float32, (2, 2)))
    end

    for j in 1:nv(g)
        push!(tensors, rand(Float32, 2))
    end

    return CuArray.(tensors)
end

function profile(id::Int, sc)
    graph = loadgraph("../networks/sc$(sc)/graph_$id.dot")
    code = readjson("../networks/sc$(sc)/eincode_$id.json")

    tensors = generate_tensors(graph)

    # warmup
    code(tensors...)

    CUDA.@profile code(tensors...) 
    
    nothing
end

function main()
    profile(1, 31)
end

main()
