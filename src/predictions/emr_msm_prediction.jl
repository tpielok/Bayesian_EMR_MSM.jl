struct EMR_MSM_Prediction{T<:AbstractFloat, S<:MSM_Timeseries{T}}
    est::EMR_MSM_Estimate
    pred_timeseries::S
    start_ind::Integer
end

# TODO: enlarge existing prediction - maybe type problems
function EMR_MSM_Prediction(
    est::EMR_MSM_Estimate{T, M, S},
    timesteps::Array{T,1},
    num_samples::Integer = 1,
    start_ind::Integer = length(est.timeseries),
    last_layer_residuals::Union{Nothing, AbstractArray{T}} = nothing;
    rand_last_layer_res = true
    ) where {T<:AbstractFloat, M<:EMR_MSM_Model_Estimate{T},
            S<:MSM_Timeseries{T}}

    pred_timeseries = MSM_Timeseries(est.timeseries, timesteps)
    EMR_MSM_Prediction(pred_timeseries, est, start_ind, last_layer_residuals;
        rand_last_layer_res = rand_last_layer_res)
end

function EMR_MSM_Prediction(
    est::EMR_MSM_Estimate{T, M, S}, num_preds::Integer,
    timeStep::T = one(T),
    num_samples::Integer = 1,
    start_ind::Integer = length(est.timeseries),
    last_layer_residuals::Union{Nothing, AbstractArray{T}} = nothing;
    rand_last_layer_res = true
    ) where {T<:AbstractFloat, M<:EMR_MSM_Model_Estimate{T},
            S<:MSM_Timeseries{T}}

    pred_timeseries = MSM_Timeseries(est.timeseries, num_preds, timeStep,
        num_samples)

    EMR_MSM_Prediction(pred_timeseries, est, start_ind, last_layer_residuals;
        rand_last_layer_res = rand_last_layer_res)
end

function EMR_MSM_Prediction(pred_timeseries::MSM_Timeseries_Point{T},
        est::EMR_MSM_Estimate{T, EMR_MSM_Model_PointEstimate{T}, S},
        start_ind::Integer = length(est.timeseries),
        last_layer_residuals::Union{Nothing, AbstractArray{T}} = nothing;
        rand_last_layer_res = true
        ) where {T<:AbstractFloat, S<:MSM_Timeseries{T}}

    num_layers = layers(pred_timeseries)
    num_params = params(pred_timeseries)

    r = residuals(pred_timeseries)
    x = values(pred_timeseries)
    L_mats = est.model.RCorrs
    A_mat = est.model.Lin_mat
    B_mats = est.model.Quad_mats
    F = est.model.F
    μ = est.model.μ
    Σ = est.model.Σ

    x[1,:] = values(est.timeseries)[start_ind, :]
    r[1,:,:] = residuals(est.timeseries)[start_ind, :, :]
    tS = timesteps(pred_timeseries)

    for i=2:length(pred_timeseries)
        r[i,:,num_layers] = if isnothing(last_layer_residuals)
            rand(MvNormal(μ, Σ))
        elseif rand_last_layer_res
            last_layer_residuals[rand(1:size(rand_last_layer_res,1)),:]
        else
            last_layer_residuals[i,:]
        end

        for l = (num_layers-1):-1:1
            r[i,:,l] = tS[i]*(L_mats[l]*vcat(x[i-1,:],vec(r[i-1,:,1:l]))) +
                r[i-1,:,l+1] + r[i-1,:,l]
        end

        x[i,:] = tS[i]*(-A_mat*x[i-1,:] +
            [transpose(x[i-1,:])*B_mats[:,:,k]*x[i-1,:] for k in 1:num_params] +
            F) + r[i-1,:,1] + x[i-1,:]
    end

    EMR_MSM_Prediction{T, MSM_Timeseries_Point{T}}(est, pred_timeseries, start_ind)
end

function EMR_MSM_Prediction(pred_timeseries::MSM_Timeseries_Dist{T},
        est::EMR_MSM_Estimate{T, EMR_MSM_Model_DistEstimate{T}, S},
        start_ind::Integer = length(est.timeseries),
        last_layer_residuals::Union{Nothing, AbstractArray{T}} = nothing;
        rand_last_layer_res = true
        ) where {T<:AbstractFloat, S<:MSM_Timeseries{T}}

    for which_ts in 1:samples(pred_timeseries)
        r_ind = rand(1:size(est.model))
        EMR_MSM_Prediction(
            MSM_Timeseries_Point{T}(
                view(pred_timeseries.x,:,:,which_ts),
                view(pred_timeseries.residuals,:,:,:,which_ts),
                view(pred_timeseries.timesteps,:)),
            EMR_MSM_Estimate(
                est.model.estimates[r_ind],
                est.timeseries,
                est.pred_timeseries.ts_points[r_ind]
                ),
            start_ind,
            if !isnothing(last_layer_residuals)
                if typeof(last_layer_residuals) <: AbstractArray{T,3}
                    last_layer_residuals[:,:,which_ts]
                elseif typeof(last_layer_residuals) <: AbstractArray{T,2}
                    last_layer_residuals
                else
                    nothing
                end
            else
                nothing
            end;
            rand_last_layer_res = rand_last_layer_res
            )
    end

    EMR_MSM_Prediction{T, MSM_Timeseries_Dist{T}}(
        est, pred_timeseries, start_ind)
end
