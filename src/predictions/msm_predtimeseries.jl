struct MSM_PredTimeseries_Dist{T<:AbstractFloat} <: MSM_Timeseries{T}
    x::Array{T,3}
    residuals::Array{T,4}
    timeSteps::Array{T,1}
end

function MSM_PredTimeseries(timeseries::S,
    num_preds::Integer,
    timeStep::T = one(T)) where S <: MSM_Timeseries{T} where T<:Real

    MSM_PredTimeseries(S, params(timeseries),
        layers(timeseries), repeat([timeStep], num_preds))
end

function MSM_PredTimeseries(timeseries::S,
    timeSteps::Array{T,1}) where S <: MSM_Timeseries{T} where T<:Real

    MSM_PredTimeseries(S, params(timeseries),
        layers(timeseries), timeSteps)
end

MSM_PredTimeseries(::Type{MSM_Timeseries_Point{T}},
    num_params, num_layers, timeSteps) where T <: Real =
    MSM_Timeseries_Point{T}(length(timeSteps), num_params, num_layers, timeSteps)
