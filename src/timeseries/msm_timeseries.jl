abstract type MSM_Timeseries{T<:AbstractFloat} end

struct MSM_Timeseries_Point{T<:AbstractFloat} <: MSM_Timeseries{T}
    x::AbstractArray{T,2}
    residuals::AbstractArray{T,3}
    timesteps::AbstractArray{T,1}

    MSM_Timeseries_Point{T}(x::AbstractArray{T,2},
        residuals::AbstractArray{T,3},
        timesteps::AbstractArray{T,1}) where T <: Real =
            new(x,residuals,timesteps)
end

struct MSM_Timeseries_Dist{T<:AbstractFloat} <: MSM_Timeseries{T}
    x::AbstractArray{T,3}
    residuals::AbstractArray{T,4}
    timesteps::AbstractArray{T,1}

    ts_points::Array{MSM_Timeseries_Point{T},1}

    MSM_Timeseries_Dist{T}(x::AbstractArray{T,3},
        residuals::AbstractArray{T,4},
        timesteps::AbstractArray{T,1}) where T <: Real =
        new(x,residuals,timesteps,
            [MSM_Timeseries_Point{T}(
                view(x,:,:,i),
                view(residuals,:,:,:,i),
                view(timesteps,:))
                for i in 1:size(x,3)
            ])
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

MSM_Timeseries_Point{T}(timeseries::MSM_Timeseries_Dist{T},
    aggregate_fun::Function)  where T <: Real =
        MSM_Timeseries_Point{T}(
            mapslices(aggregate_fun, timeseries.x; dims=(3))[:,:,1],
            mapslices(aggregate_fun, timeseries.residuals; dims=(4))[:,:,:,1],
            timeseries.timesteps
        )

function MSM_Timeseries_Dist{T}(num_obs::Integer, num_params::Integer,
    num_layers::Integer, num_samples::Integer, timesteps::Array{T,1}) where T <: Real

    MSM_Timeseries_Dist{T}(Array{T}(undef,num_obs, num_params, num_samples),
        Array{T}(undef, num_obs, num_params, num_layers, num_samples), timesteps)
end

function MSM_Timeseries(timeseries::S,
    num_preds::Integer,
    timeStep::T = one(T),
    num_samples::Integer = 1) where S <: MSM_Timeseries{T} where T<:Real

    MSM_Timeseries(S, params(timeseries),
        layers(timeseries), repeat([timeStep], num_preds), num_samples)
end

function MSM_Timeseries(timeseries::S,
    timeSteps::Array{T,1},
    num_samples::Integer = 1) where S <: MSM_Timeseries{T} where T<:Real

    MSM_Timeseries(S, params(timeseries),
        layers(timeseries), timeSteps, num_samples)
end

MSM_Timeseries(::Type{MSM_Timeseries_Point{T}},
    num_params::Integer, num_layers::Integer, timeSteps::Array{T,1},
    num_samples::Integer = 1) where T <: Real =
    ifelse(num_samples == 1,
        MSM_Timeseries_Point{T}(
            length(timeSteps),
            num_params,
            num_layers,
            timeSteps),
        MSM_Timeseries_Dist{T}(length(timeSteps),
            num_params,
            num_layers,
            num_samples,
            timeSteps
            )
    )

timesteps(ts::S) where {S<:MSM_Timeseries} = ts.timesteps
values(ts::S) where {S<:MSM_Timeseries} = ts.x
residuals(ts::S) where {S<:MSM_Timeseries}  = ts.residuals

Base.length(ts::S) where {S<:MSM_Timeseries} = size(values(ts), 1)

params(ts::S) where {S<:MSM_Timeseries} = size(values(ts), 2)
layers(ts::S) where {S<:MSM_Timeseries} = size(residuals(ts), 3)
samples(ts::MSM_Timeseries_Dist) = size(values(ts),3)

function Base.copy(ts::S) where {S<:MSM_Timeseries{T}} where {T <:Real}
    S(copy(ts.x), copy(ts.residuals), copy(ts.timesteps))
end
