struct EMR_MSM_Prediction{T<:AbstractFloat, S<:MSM_Timeseries{T}}
    est::EMR_MSM_Estimate
    pred_timeseries::AbstractArray{S,1}
    start_ind::Array{Integer,1}
end

# TODO: enlarge existing prediction
function EMR_MSM_Prediction(
    est::EMR_MSM_Estimate{T, M, S},
    timesteps::Array{A,1},
    num_samples::Integer = 1,
    start_ind::Array{I,1} = length.(est.timeseries),
    last_layer_residuals::Union{Nothing, Array{B,1}} = nothing;
    rand_last_layer_res = true,
    use_last_layer = true
    ) where {T<:AbstractFloat, M<:EMR_MSM_Model_Estimate{T},
            S<:MSM_Timeseries{T}, A<:AbstractArray{T,1},
            B<:AbstractArray{T,1}, I<:Integer}

    pred_timeseries = [MSM_Timeseries(est.pred_timeseries[ts], timesteps[ts],
                        num_samples)
                        for ts in 1:length(est.pred_timeseries)]
    EMR_MSM_Prediction(pred_timeseries, est, start_ind, last_layer_residuals;
        rand_last_layer_res = rand_last_layer_res,
        use_last_layer = use_last_layer)
end

function EMR_MSM_Prediction(
    est::EMR_MSM_Estimate{T, M, S},
    num_preds::Integer,
    timestep::Array{T,1} = [one(T) for i in 1:length(est.timeseries)],
    num_samples::Integer = 1,
    start_ind::Array{I,1} = length.(est.timeseries),
    last_layer_residuals::Union{Nothing, Array{B,1}} = nothing;
    rand_last_layer_res = true,
    use_last_layer = true
    ) where {T<:AbstractFloat, M<:EMR_MSM_Model_Estimate{T},
            B<:AbstractArray{T,1}, S<:MSM_Timeseries{T}, I<:Integer}

    pred_timeseries = [MSM_Timeseries(est.pred_timeseries[ts], num_preds, timestep[ts],
        num_samples)
        for ts in 1:length(est.pred_timeseries)]

    EMR_MSM_Prediction(pred_timeseries, est, start_ind, last_layer_residuals;
        rand_last_layer_res = rand_last_layer_res,
        use_last_layer = use_last_layer)
end

# TODO use est.pred_ts not est.timeseries !!

function EMR_MSM_Prediction(pred_timeseries::Array{MSM_Timeseries_Point{T},1},
        est::EMR_MSM_Estimate{T, EMR_MSM_Model_PointEstimate{T}, S,
            MSM_Timeseries_Dist{T}},
        start_ind::Array{I,1} = length.(est.timeseries),
        last_layer_residuals::Union{Nothing, Array{B,1}} = nothing;
        rand_last_layer_res = true,
        use_last_layer = true
        ) where {T<:AbstractFloat, S<:MSM_Timeseries{T},
            B<:AbstractArray{T,1}, I<:Integer}

    EMR_MSM_Prediction(pred_timeseries,
        EMR_MSM_Estimate(est.model,est.timeseries,
        [rand(ps.ts_points) for ps in est.pred_timeseries]),
        start_ind,
        last_layer_residuals;
        rand_last_layer_res = rand_last_layer_res,
        use_last_layer = use_last_layer
        )
end

function EMR_MSM_Prediction(pred_timeseries::Array{MSM_Timeseries_Point{T},1},
        est::EMR_MSM_Estimate{T, EMR_MSM_Model_PointEstimate{T}, S,
            MSM_Timeseries_Point{T}},
        start_ind::Array{I,1} = length.(est.timeseries),
        last_layer_residuals::Union{Nothing, Array{B,1}} = nothing;
        rand_last_layer_res = true,
        use_last_layer = true
        ) where {T<:AbstractFloat, S<:MSM_Timeseries{T},
            B<:AbstractArray{T,1}, I<:Integer}

    num_layers = layers(pred_timeseries[1])
    num_params = params(pred_timeseries[1])

    L_mats = est.model.RCorrs
    A_mat = est.model.Lin_mat
    B_mats = est.model.Quad_mats
    F = est.model.F
    μ = est.model.μ
    Σ = est.model.Σ

    for ts in 1:length(pred_timeseries)
        r = residuals(pred_timeseries[ts])
        x = values(pred_timeseries[ts])

        x[1,:] = values(est.timeseries[ts])[start_ind[ts], :]
        r[1,:,:] = residuals(est.pred_timeseries[ts])[start_ind[ts], :, :]
        pred_timesteps = timesteps(pred_timeseries[ts])

        for i=2:length(pred_timeseries[ts])
            r[i,:,num_layers] = if isnothing(last_layer_residuals)
                ifelse(use_last_layer,rand(MvNormal(μ, Σ)), zeros(T,num_params))
            elseif rand_last_layer_res
                last_layer_residuals[ts][rand(1:size(rand_last_layer_res,1)),:]
            else
                last_layer_residuals[ts][i,:]
            end

            for l = (num_layers-1):-1:1
                r[i,:,l] = pred_timesteps[i]*(L_mats[l]*vcat(x[i-1,:],vec(r[i-1,:,1:l]))) +
                    r[i-1,:,l+1] + r[i-1,:,l]
            end

            x[i,:] = pred_timesteps[i]*(-A_mat*x[i-1,:] +
                [transpose(x[i-1,:])*B_mats[:,:,k]*x[i-1,:] for k in 1:num_params] +
                F) + r[i-1,:,1] + x[i-1,:]
        end

    end
    EMR_MSM_Prediction{T, MSM_Timeseries_Point{T}}(est, pred_timeseries, start_ind)
end

function EMR_MSM_Prediction(pred_timeseries::Array{MSM_Timeseries_Dist{T},1},
        est::EMR_MSM_Estimate{T, EMR_MSM_Model_DistEstimate{T}, S},
        start_ind::Array{I,1} = length.(est.timeseries),
        last_layer_residuals::Union{Nothing, Array{B,1}} = nothing;
        rand_last_layer_res = true,
        use_last_layer = true
        ) where {T<:AbstractFloat, S<:MSM_Timeseries{T}, B<:AbstractArray{T,1},  I<:Integer}

    for ts in 1:length(pred_timeseries)
        for which_ts in 1:samples(pred_timeseries[ts])
            r_ind = rand(1:size(est.model))
            EMR_MSM_Prediction(
                [MSM_Timeseries_Point{T}(
                    view(pred_timeseries[ts].x,:,:,which_ts),
                    view(pred_timeseries[ts].residuals,:,:,:,which_ts),
                    view(pred_timeseries[ts].timesteps,:))],
                EMR_MSM_Estimate(
                    est.model.estimates[r_ind],
                    [est.timeseries[ts]],
                    [est.pred_timeseries[ts].ts_points[r_ind]]
                    ),
                [start_ind[ts]],
                if !isnothing(last_layer_residuals)
                    if typeof(last_layer_residuals[ts]) <: AbstractArray{T,3}
                        [last_layer_residuals[ts][:,:,which_ts]]
                    elseif typeof(last_layer_residuals[ts]) <: AbstractArray{T,2}
                        [last_layer_residuals[ts]]
                    else
                        nothing
                    end
                else
                    nothing
                end ; rand_last_layer_res = rand_last_layer_res,
                use_last_layer = use_last_layer)
        end
    end

    EMR_MSM_Prediction{T, MSM_Timeseries_Dist{T}}(
        est, pred_timeseries, start_ind)
end

function write(output::String, pred::EMR_MSM_Prediction)
    mkpath(output)
    for i in 1:length(pred.pred_timeseries)
        write(joinpath(output,"ts-"*string(i)), pred.pred_timeseries[i])
    end
end
