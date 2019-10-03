struct EMR_MSM_Estimate{T<:AbstractFloat,
        M<:EMR_MSM_Model_Estimate{T},
        S<:MSM_Timeseries{T},
        P<:MSM_Timeseries{T}}
    model::M
    timeseries::AbstractArray{S,1}
    pred_timeseries::AbstractArray{P,1}
end

EMR_MSM_Estimate(model::M,timeseries::AbstractArray{S,1},
    pred_timeseries::AbstractArray{P,1}) where
    {T<:AbstractFloat, M<:EMR_MSM_Model_Estimate{T},
     S<:MSM_Timeseries{T},P<:MSM_Timeseries{T}} =
     EMR_MSM_Estimate{T,M,S,P}(model, timeseries, pred_timeseries)

function EMR_MSM_Estimate(timeseries::AbstractArray{S,1},
    num_layers::Integer,
    num_samples::Integer,
    num_chains::Integer,
    tau0::T = one(T)) where
    {S <: MSM_Timeseries{T}} where T <: Real

    dist_model, pred_timeseries = EMR_MSM_Model_DistEstimate(timeseries,
                    num_layers, num_samples, num_chains, tau0)

    EMR_MSM_Estimate{T, EMR_MSM_Model_DistEstimate{T}, S,
        typeof(pred_timeseries[1])}(
        dist_model,
        timeseries,
        pred_timeseries
    )
end
