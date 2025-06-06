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

function benchmark(id::Int, sc)
    graph = loadgraph("../networks/sc$(sc)/graph_$id.dot")
    code = readjson("../networks/sc$(sc)/eincode_$id.json")

    tensors = generate_tensors(graph)

    # warmup
    code(tensors...)

    # 10 times runs for benchmark
    t = @elapsed begin
        for i in 1:10
            code(tensors...)
        end
    end

    return t / 10
end

function main(sc)
    df = CSV.write("../data/benchmark_julia_real_sc$(sc).csv", DataFrame(id = Int[], time = Float64[]))

    for id in 1:10
        t = benchmark(id, sc)
        CSV.write(df, DataFrame(id = [id], time = [t]), append = true)
    end

    return df
end

main(31)
