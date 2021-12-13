include("CUDAqrudt.jl")
using CUDA
using Adapt
using LinearAlgebra
using Test

struct lattice
    L::Int
    Ns::Int
    U::Float64
    μ::Float64
    Temp::Float64
    Nt::Int
    Δτ::Float64
    λ::Float64
    expmΔτT::CuArray{Float64}

    function lattice(L::Int,U::Float64,μ::Float64,Temp::Float64,Nt::Int)
        Ns = L^3
        Δτ = 1/(Temp*Nt)
        λ = acosh(exp(abs(U)*Δτ/2))
        expmΔτT = adapt(CuArray,exp(-Δτ * initT(L)))
        new(L,Ns,U,μ,Temp,Nt,Δτ,λ,expmΔτT)
    end
end

function initT(L::Int)
    Tmatrix = zeros(Int,L^3,L^3)
    index::Int = 1
    for z = 1:L, y = 1:L, x = 1:L
        Tmatrix[index, x == 1 ? index+L-1 : index-1] = -1
        Tmatrix[index, x == L ? index-L+1 : index+1] = -1
        Tmatrix[index, y == 1 ? index+L*(L-1) : index-L] = -1
        Tmatrix[index, y == L ? index-L*(L-1) : index+L] = -1
        Tmatrix[index, z == 1 ? index+L^2*(L-1) : index-L^2] = -1
        Tmatrix[index, z == L ? index-L^2*(L-1) : index+L^2] = -1
        index += 1
    end
    Tmatrix
end

function initMultBudt(l::lattice,AuxField::Matrix{Int})
    MultBup = Vector{CUDAUDT}(undef,l.Nt+2)
    MultBdn = Vector{CUDAUDT}(undef,l.Nt+2)
    UDTI = CUDAudt(CuMatrix(Diagonal(ones(l.Ns))))
    MultBup[1] = copy(UDTI)
    MultBdn[1] = copy(UDTI)
    MultBup[l.Nt+2] = copy(UDTI)
    MultBdn[l.Nt+2] = copy(UDTI)
    for i = 2:l.Nt+1
        MultBup[i] = CUDAudtMult(MultBup[i-1],
            CUDAudt(l.expmΔτT*Diagonal(exp.(cu( AuxField[:,i-1])*l.λ .+ (l.μ - l.U/2)*l.Δτ))))
        MultBdn[i] = CUDAudtMult(MultBdn[i-1],
            CUDAudt(l.expmΔτT*Diagonal(exp.(cu(-AuxField[:,i-1])*l.λ .+ (l.μ - l.U/2)*l.Δτ))))
    end
    MultBup, MultBdn
end

function flipslice!(slice::Int,l::lattice,AuxField::Matrix{Int},Gup::CuArray{Float64},Gdn::CuArray{Float64})
    γup = exp.(-2*l.λ*AuxField[:,slice]).-1
    γdn = exp.( 2*l.λ*AuxField[:,slice]).-1
    Rup = 0
    Rdn = 0
    gtmp = similar(Gup[1,:])
    @inbounds for site = 1:l.Ns
        CUDA.@allowscalar Rup = 1+(1-Float64(Gup[site,site]))*γup[site]
        CUDA.@allowscalar Rdn = 1+(1-Float64(Gdn[site,site]))*γdn[site]
        if rand() < min(1,Rup * Rdn)
            AuxField[site,slice] *= -1
                updateg!(site,γup[site]/Rup,Gup,gtmp)
                updateg!(site,γdn[site]/Rdn,Gdn,gtmp)
        end
    end
    nothing
end

function updateg!(site::Int,prop::Float64,g::CuArray{Float64},gtmp::CuArray{Float64})
    gtmp[:] = -g[site,:]
    CUDA.@allowscalar gtmp[site] += 1
    @views g[:,:] = g[:,:] - prop * g[:,site] * transpose(gtmp)
    nothing
end

function updateRight!(slice::Int,l::lattice,AuxField::Matrix{Int},
    MultBup::Vector{CUDAUDT},MultBdn::Vector{CUDAUDT},gup::CuArray{Float64},gdn::CuArray{Float64})
    MultBup[l.Nt-slice+2] = CUDAudtMult(CUDAudt(l.expmΔτT * Diagonal(exp.(cu( AuxField[:,slice])*l.λ 
        .+ (l.μ - l.U/2)*l.Δτ))), MultBup[l.Nt-slice+3])
    gup[:,:] = invoneplus(CUDAudtMult(MultBup[l.Nt-slice+2],MultBup[l.Nt-slice+1]))
    MultBdn[l.Nt-slice+2] = CUDAudtMult(CUDAudt(l.expmΔτT * Diagonal(exp.(cu(-AuxField[:,slice])*l.λ 
        .+ (l.μ - l.U/2)*l.Δτ))), MultBdn[l.Nt-slice+3])
    gdn[:,:] = invoneplus(CUDAudtMult(MultBdn[l.Nt-slice+2],MultBdn[l.Nt-slice+1]))
    nothing
end

function updateLeft!(slice::Int,l::lattice,AuxField::Matrix{Int},
    MultBup::Vector{CUDAUDT},MultBdn::Vector{CUDAUDT},gup::CuArray{Float64},gdn::CuArray{Float64})
    MultBup[l.Nt-slice+2] = CUDAudtMult(MultBup[l.Nt-slice+1],
        CUDAudt(l.expmΔτT * Diagonal(exp.(cu( AuxField[:,slice])*l.λ .+ (l.μ - l.U/2)*l.Δτ))))
    gup[:,:] = invoneplus(CUDAudtMult(MultBup[l.Nt-slice+2],MultBup[l.Nt-slice+3]))
    MultBdn[l.Nt-slice+2] = CUDAudtMult(MultBdn[l.Nt-slice+1], 
        CUDAudt(l.expmΔτT * Diagonal(exp.(cu(-AuxField[:,slice])*l.λ .+ (l.μ - l.U/2)*l.Δτ))))
    gdn[:,:] = invoneplus(CUDAudtMult(MultBdn[l.Nt-slice+2],MultBdn[l.Nt-slice+3]))
    nothing
end

function sweep!(l::lattice,AuxField::Matrix{Int},Gup::CuArray{Float64},Gdn::CuArray{Float64},
    MultBup::Vector{CUDAUDT},MultBdn::Vector{CUDAUDT})
    for slice = 1:l.Nt
        flipslice!(slice,l,AuxField,Gup,Gdn)
        updateRight!(slice,l,AuxField,MultBup,MultBdn,Gup,Gdn)
    end
    for slice = l.Nt:-1:1
        flipslice!(slice,l,AuxField,Gup,Gdn)
        updateLeft!(slice,l,AuxField,MultBup,MultBdn,Gup,Gdn)
    end
end