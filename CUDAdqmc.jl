include("CUDAcore.jl")
include("measure.jl")

CUDA.allowscalar(false)

function main()
    L = 10
    U::Float64 = 8
    μ::Float64 = 4
    Temp::Float64 = 0.5
    Nt = 40
    l = lattice(L,U,μ,Temp,Nt)
    AuxField = rand([-1,1],l.Ns,Nt)
    MultBup, MultBdn = initMultBudt(l,AuxField)
    Gup = invoneplus(MultBup[Nt+1])
    Gdn = invoneplus(MultBdn[Nt+1])
    for sweep = 1:10
        @time sweep!(l,AuxField,Gup,Gdn,MultBup,MultBdn)
    end
    println(occupy(Gup,Gdn))
    println(doubleoccupy(Gup,Gdn))
end

main()