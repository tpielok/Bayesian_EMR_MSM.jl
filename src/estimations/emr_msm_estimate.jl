# todo: remove and integrate with model

struct EMR_MSM_Estimate{T<:AbstractFloat, M<:EMR_MSM_Model_Estimate{T}, S<:MSM_Timeseries{T}}
    model::M
    timeseries::S
end

EMR_MSM_Estimate(model::M,timeseries::S) where
    {T<:AbstractFloat, M<:EMR_MSM_Model_Estimate{T},
     S<:MSM_Timeseries{T}} =
    EMR_MSM_Estimate{T,M,S}(model, timeseries)

EMR_MSM_Estimate(timeseries::S, num_layers::Integer, num_samples::Integer,
    num_chains::Integer,
    tau0::T = one(T)) where
    S <: MSM_Timeseries{T} where T <: Real =
    EMR_MSM_Estimate{T, EMR_MSM_Model_DistEstimate{T}, S}(
        EMR_MSM_Model_DistEstimate(timeseries, num_layers, num_samples,
        num_chains,
        tau0),
        timeseries
    )
