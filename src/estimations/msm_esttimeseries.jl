struct MSM_EstTimeseries_Dist{T<:AbstractFloat} <: MSM_Timeseries{T}
    x::Array{T,2}
    residuals::Array{T,4}
    timesteps::Array{T,1}
end
