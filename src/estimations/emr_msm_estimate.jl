struct EMR_MSM_Estimate{T<:AbstractFloat,
        M<:EMR_MSM_Model_Estimate{T},
        S<:MSM_Timeseries{T},
        P<:MSM_Timeseries{T}}
    model::M
    timeseries::AbstractArray{S,1}
    pred_timeseries::AbstractArray{P,1}
    conv_info::Union{Nothing, DataFrames.DataFrame}
end

EMR_MSM_Estimate(model::M,timeseries::AbstractArray{S,1},
    pred_timeseries::AbstractArray{P,1},
    conv_info::Union{Nothing, DataFrames.DataFrame} = nothing) where
    {T<:AbstractFloat, M<:EMR_MSM_Model_Estimate{T},
     S<:MSM_Timeseries{T},P<:MSM_Timeseries{T}} =
     EMR_MSM_Estimate{T,M,S,P}(model, timeseries, pred_timeseries,
        conv_info)

function EMR_MSM_Estimate(timeseries::AbstractArray{S,1},
    num_layers::Integer,
    num_samples::Integer,
    num_chains::Integer,
    tau0::T = one(T),
    rand_seed::Integer=123) where
    {S <: MSM_Timeseries{T}} where T <: Real

    dist_model, pred_timeseries, conv_info = EMR_MSM_Model_DistEstimate(timeseries,
                    num_layers, num_samples, num_chains, tau0; rand_seed = rand_seed)

    EMR_MSM_Estimate{T, EMR_MSM_Model_DistEstimate{T}, S,
        typeof(pred_timeseries[1])}(
        dist_model,
        timeseries,
        pred_timeseries,
        conv_info
    )
end

function write(output::String, est::EMR_MSM_Estimate)
    mkpath(output)
    est_dir  = joinpath(output,"est")
    pred_dir = joinpath(output,"pred")

    mkpath(est_dir)
    mkpath(pred_dir)

    write(joinpath(output,"model"), est.model)
    for i in 1:length(est.timeseries)
        write(joinpath(est_dir,"ts-"*string(i)), est.timeseries[i])
    end
    for i in 1:length(est.pred_timeseries)
        write(joinpath(pred_dir,"ts-"*string(i)), est.pred_timeseries[i])
    end

    if !isnothing(est.conv_info)
        CSV.write(joinpath(output,"conv_info.csv"),est.conv_info)
    end
end
