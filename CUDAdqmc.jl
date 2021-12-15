include("CUDAcore.jl")
include("measure.jl")

function main()
    L = 4
    U = 8.0
    μ = 4.0
    Temp = 0.5
    Nt = 40
    l = lattice(L,U,μ,Temp,Nt)
    AuxField = rand([-1,1],l.Ns,Nt)
    MultBup, MultBdn = initMultBudt(l,AuxField)
    Gup = invoneplus(MultBup[Nt+1])
    Gdn = invoneplus(MultBdn[Nt+1])
    Nsweep = 100
    for sweep = 1:Nsweep
        sweep!(l,AuxField,Gup,Gdn,MultBup,MultBdn)
        println(energy(Gup,Gdn,l))
    end
end

main()