using OMEinsum
using CSV, DataFrames

df = CSV.write("../data/complexity.csv", DataFrame(id = Int[], sc = Int[], tc = Float64[]))

id = collect(1:10)
ccs = []

for i in id
    eins = readjson("../networks/eincode_$(i).json")
    push!(ccs, contraction_complexity(eins, uniformsize(eins, 2)))
end

CSV.write("../data/complexity.csv", DataFrame(id = id, sc = [cc.sc for cc in ccs], tc = [cc.tc for cc in ccs]))