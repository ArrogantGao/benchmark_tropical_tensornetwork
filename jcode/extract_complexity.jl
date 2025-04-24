using OMEinsum
using CSV, DataFrames

for sc in [31, 32]
df = CSV.write("../data/complexity_sc$(sc).csv", DataFrame(id = Int[], sc = Int[], tc = Float64[]))

id = collect(1:10)
ccs = []

for i in id
    eins = readjson("../networks/sc$(sc)/eincode_$(i).json")
    push!(ccs, contraction_complexity(eins, uniformsize(eins, 2)))
end

CSV.write("../data/complexity_sc$(sc).csv", DataFrame(id = id, sc = [cc.sc for cc in ccs], tc = [cc.tc for cc in ccs]))
end
