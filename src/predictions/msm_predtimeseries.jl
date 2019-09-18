struct MSM_PredTimeseries_Dist{T<:AbstractFloat} <: MSM_Timeseries{T}
    x::Array{T,3}
    residuals::Array{T,4}
    timesteps::Array{T,1}
end

samples(ts::MSM_PredTimeseries_Dist{T}) where {T<:Real} =
    size(values(ts),3)

function MSM_PredTimeseries_Dist{T}(num_obs::Integer, num_params::Integer,
    num_layers::Integer, num_samples::Integer, timesteps::Array{T,1}) where T <: Real

    MSM_PredTimeseries_Dist{T}(Array{T}(undef,num_obs, num_params, num_samples),
        Array{T}(undef, num_obs, num_params, num_layers, num_samples), timesteps)
end

function MSM_PredTimeseries(timeseries::S,
    num_preds::Integer,
    timeStep::T = one(T),
    num_samples::Integer = 1) where S <: MSM_Timeseries{T} where T<:Real

    MSM_PredTimeseries(S, params(timeseries),
        layers(timeseries), repeat([timeStep], num_preds), num_samples)
end

function MSM_PredTimeseries(timeseries::S,
    timeSteps::Array{T,1},
    num_samples::Integer = 1) where S <: MSM_Timeseries{T} where T<:Real

    MSM_PredTimeseries(S, params(timeseries),
        layers(timeseries), timeSteps, num_samples)
end

MSM_PredTimeseries(timeseries::MSM_PredTimeseries_Dist,
    aggregate_fun::Function) =
        MSM_Timeseries_Point{Float64}(
            mapslices(aggregate_fun, timeseries.x; dims=(3))[:,:,1],
            mapslices(aggregate_fun, timeseries.residuals; dims=(4))[:,:,:,1],
            timeseries.timesteps
        )

MSM_PredTimeseries(::Type{MSM_Timeseries_Point{T}},
    num_params::Integer, num_layers::Integer, timeSteps::Array{T,1},
    num_samples::Integer = 1) where T <: Real =
    ifelse(num_samples == 1,
        MSM_Timeseries_Point{T}(
            length(timeSteps),
            num_params,
            num_layers,
            timeSteps),
        MSM_PredTimeseries_Dist{T}(length(timeSteps),
            num_params,
            num_layers,
            num_samples,
            timeSteps
            )
    )
