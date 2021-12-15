include("CUDAcore.jl")
include("measure.jl")
using JLD2
using DelimitedFiles

function main()
    L = 4
    U::Float64 = 8
    μ::Float64 = 4
    Temp::Float64 = 0.72
    Nt = 40
    l = lattice(L,U,μ,Temp,Nt)
    # Aux = jldopen("Aux.jld2")
    # AuxField = Aux["AuxField"]
    AuxField = rand([-1,1],l.Ns,Nt)
    MultBup, MultBdn = initMultBudt(l,AuxField)
    Gup = invoneplus(MultBup[Nt+1])
    Gdn = invoneplus(MultBdn[Nt+1])
    Nsweep = 400
    S = zeros(Nsweep,1)
    Sfile = "Sπ.dat"
    for sweep = 1:Nsweep
        sweep!(l,AuxField,Gup,Gdn,MultBup,MultBdn)
        S[sweep] = Sπ(Gup,Gdn,l)
        println(S[sweep])
    end
    open(Sfile,"w") do io
        writedlm(io, d, '\t')
    end
    jldsave("Aux.jld2";AuxField)
end

main()