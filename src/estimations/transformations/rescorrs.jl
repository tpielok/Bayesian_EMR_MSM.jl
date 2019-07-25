struct ResCorrs{T<:AbstractFloat} <: AbstractVector{Array{T,2}}
    L_mats::Array{Array{T,2},1}

    num_params::Integer
    num_layers::Integer
end

function ResCorrs(vec::Array{T,1}, num_params::Integer, num_layers::Integer) where T<:Real
    ResCorrs{T}(
        [reshape(vec[((div(l*(l+1),2)-1)*num_params^2+1):((div((l+1)*(l+2),2)-1)*
        num_params^2)],num_params, (l+1)*num_params) for l in 1:num_layers],
        num_params,
        num_layers)
end

Base.vec(rescorrs::ResCorrs{T}) where T <: Real =
    Array{T}(vcat([vec(l) for l in rescorrs.L_mats]...))
Base.print(rescorrs::ResCorrs) = print(vec(rescorrs))
Base.length(::Type{ResCorrs}, num_params, num_layers) =
    num_params*num_params*(div((num_layers+2)*(num_layers+1),2)-1)
Base.size(rescorrs::ResCorrs) = length(vec(rescorrs))

Base.getindex(rescorrs::ResCorrs{T}, idxs::Integer...) where T <: Real =
    rescorrs.L_mats[idxs...]
