using CUDA
using Adapt
using LinearAlgebra

mutable struct CUDAUDT{Type <: Real}
	U::CuMatrix{Type}
	D::CuVector{Type}
	T::CuMatrix{Type}
end

# iteration for destructuring into components
Base.iterate(S::CUDAUDT) = (S.U, Val(:D))
Base.iterate(S::CUDAUDT, ::Val{:D}) = (S.D, Val(:T))
Base.iterate(S::CUDAUDT, ::Val{:T}) = (S.T, Val(:done))
Base.copy(S::CUDAUDT) = CUDAUDT(S.U, S.D, S.T)

function CUDAudt(A::Matrix{Float64})
    CuA = adapt(CuArray,A)
    U, R = qr(CuA)
    D = diag(R)
    CUDAUDT(CuMatrix(U), D, Diagonal(1 ./ D) * R)
end

function CUDAudt(CuA::CuMatrix{Float64})
    U, R = qr(CuA)
    D = diag(R)
    CUDAUDT(CuMatrix(U), D, Diagonal(1 ./ D) * R)
end

function CUDAudtMult(A::CUDAUDT{Float64},B::CUDAUDT{Float64})
    CuMat = A.T * B.U
    lmul!(Diagonal(A.D), CuMat)
    rmul!(CuMat, Diagonal(B.D))
    F = CUDAudt(CuMat)
    CUDAUDT(A.U * F.U, F.D, F.T * B.T)
end

function invoneplus(F::CUDAUDT;u = similar(F.U),t = similar(F.T))
    U, D, T = F
    m = U' / T
    m[diagind(m)] .+= D
    utmp, d, ttmp = CUDAudt(m)
    mul!(u, U, utmp)
    mul!(t, ttmp, T)
    return inv(t)*Diagonal(1 ./ d)*u'
end