abstract type MSM_Timeseries{T<:AbstractFloat} end

struct MSM_Timeseries_Point{T<:AbstractFloat} <: MSM_Timeseries{T}
    x::Array{T,2}
    residuals::Array{T,3}
    timesteps::Array{T,1}

    MSM_Timeseries_Point{T}(x::Array{T,2},
        residuals::Array{T,3},
        timesteps::Array{T,1}) where T <: Real =
            new(x,residuals,timesteps)
end

function MSM_Timeseries_Point{T}(num_obs::Integer, num_params::Integer,
    num_layers::Integer, timesteps::Array{T,1}) where T <: Real

    MSM_Timeseries_Point{T}(Array{T}(undef,num_obs, num_params),
        Array{T}(undef, num_obs, num_params, num_layers), timesteps)
end

function MSM_Timeseries_Point{T}(num_obs::Integer, num_params::Integer,
    num_layers::Integer, timeStep::T = one(T)) where T <: Real

    MSM_Timeseries_Point{T}(num_obs, num_params, num_layers,
        repeat([timeStep], num_obs))
end

timesteps(ts::S) where {S<:MSM_Timeseries} = ts.timesteps

values(ts::MSM_Timeseries_Point{T}) where T <:Real = ts.x
residuals(ts::MSM_Timeseries_Point{T}) where T <:Real  = ts.residuals

Base.length(ts::S) where {S<:MSM_Timeseries} = size(values(ts), 1)
params(ts::S) where {S<:MSM_Timeseries} = size(values(ts), 2)
layers(ts::S) where {S<:MSM_Timeseries} = size(residuals(ts), 3)

function Base.copy(ts::S) where {S<:MSM_Timeseries{T}} where {T <:Real}
    S(copy(ts.x), copy(ts.residuals), copy(ts.timesteps))
end
