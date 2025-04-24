using CUDA, LinearAlgebra
using TropicalGemmC_jll

const Symbol_FP32 = (:FP32, "FP32")
const Symbol_FP64 = (:FP64, "FP64")
const Symbol_INT32 = (:INT32, "INT32")
const Symbol_INT64 = (:INT64, "INT64")
const Symbol_Bool = (:Bool, "Bool")

const CTranspose{T} = Transpose{T, <:CuVecOrMat{T}}

function dims_match(A::T1, B::T2, C::T3) where{T1, T2, T3}

    @assert size(A, 1) == size(C, 1)
    @assert size(B, 2) == size(C, 2)
    @assert size(A, 2) == size(B, 1)

    return size(A, 1), size(B, 2), size(A, 2)
end

for (TA, tA) in [(:CuVecOrMat, 'N'), (:CTranspose, 'T')]
    for (TB, tB) in [(:CuVecOrMat, 'N'), (:CTranspose, 'T')]
        for (TT, CT, funcname, lib) in [(:Float32, :Cfloat, :FLOAT_maxplus, :lib_TropicalMaxPlus_FP32), (:Float64, :Cdouble, :DOUBLE_maxplus, :lib_TropicalMaxPlus_FP64)]
            @eval function matmul!(C::CuVecOrMat{T}, A::$TA{T}, B::$TB{T}, α::T, β::T, stream::CuStream = stream()) where {T<:$TT}
                M, N, K = dims_match(A, B, C)
                if K == 0 && M * N != 0
                    return rmul!(C, β)
                elseif M * N == 0
                    return C
                else
                    @ccall $lib.$funcname(M::Cint, N::Cint, K::Cint, pointer(parent(A))::CuPtr{$CT}, pointer(parent(B))::CuPtr{$CT}, pointer(C)::CuPtr{$CT}, α::$CT, β::$CT, $tA::Cchar, $tB::Cchar, stream::CUDA.CUstream)::Cvoid
                end
                return C
            end
        end
    end
end

const CuTropicalBlasTypes = Union{Float32, Float64}

function _convert(x::Bool, ::Type{TF}) where TF <: CuTropicalBlasTypes
    return x ? zero(TF) : TF(-Inf)
end

# overload the LinearAlgebra.mul!
for TA in [:CuVecOrMat, :CTranspose]
    for TB in [:CuVecOrMat, :CTranspose]
        @eval function LinearAlgebra.mul!(C::CuVecOrMat{T}, A::$TA{T}, B::$TB{T}, α::Number, β::Number) where {T <: CuTropicalBlasTypes}
            C = matmul!(C, A, B, _convert(α, T), _convert(β, T))
            return C
        end
    end
end
