using OMEinsum
using Graphs, GraphIO
using CUDA, LinearAlgebra
using CSV, DataFrames
using TropicalNumbers, CuTropicalGEMM
using GenericTensorNetworks

function benchmark(id::Int, sc)
    graph = loadgraph("../networks/sc$(sc)/graph_$id.dot")
    code = readjson("../networks/sc$(sc)/eincode_$id.json")

    tn = GenericTensorNetwork(IndependentSet(graph), code, Dict{Int, Int}())
    @show contraction_complexity(tn)

    # warmup
    res = solve(tn, SizeMax(), T = Float32, usecuda = true)
    @show id, Array(res)[]

    # 10 times runs for benchmark
    t = @elapsed begin
        for i in 1:5
            @show id, i
            solve(tn, SizeMax(), T = Float32, usecuda = true)
        end
    end

    @show id, t / 5

    return t / 5, Array(res)[]
end

function main(sc)
    df = CSV.write("../data/benchmark_julia_tropical_sc$(sc).csv", DataFrame(id = Int[], time = Float64[], res = []))

    for id in 1:10
        t, res = benchmark(id, sc)
        CSV.write(df, DataFrame(id = [id], time = [t], res = [res.n]), append = true)
    end

    return df
end

main(31)
