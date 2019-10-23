using Bayesian_EMR_MSM
using Test

using Distributions
using LinearAlgebra
import Random

@testset "emr_msm_estimate" begin

    @testset "EMR_MSM_PointEstimate" begin
        Bayesian_EMR_MSM.cmdstan_home!("/cmdstan")
        delta = 1/50

        seed = 2
        rand_gen = Normal(0,5)
        num_obs    = 7
        num_params = 2
        num_layers = 3
        num_est_layers = 2
        num_ts = 2
        σ = 0.001
        Σ = Diagonal(repeat([σ],num_params))
        μ = zeros(num_params)
        num_pred = floor(Int64, delta*500)
        timestep = 0.0015/delta
        F_zero = false
        A_zero = false
        B_zero = false

        num_samples = 10
        num_chains = 2

        Random.seed!(seed)

        # actually only 1 needed
        x_start   = [rand(rand_gen, num_obs, num_params) for ts in 1:num_ts]
        res_start = [rand(rand_gen, num_obs, num_params, num_layers+1)/100
                        for ts in 1:num_ts]

        F = ifelse(F_zero,zeros(num_params),rand(rand_gen, num_params))
        A = ifelse(A_zero,zeros(num_params,num_params),rand(rand_gen, num_params, num_params))
        B = ifelse(B_zero,zeros(num_params,num_params,num_params),rand(rand_gen, num_params, num_params, num_params))

        for i in 1:num_params
            B[:,:,i] = 0.5*(B[:,:,i] + transpose(B[:,:,i]))
        end

        L = Bayesian_EMR_MSM.ResCorrs(rand(rand_gen,
        length(Bayesian_EMR_MSM.ResCorrs,
        num_params, num_layers)), num_params, num_layers)

        model = Bayesian_EMR_MSM.EMR_MSM_Model_PointEstimate(F, A, B, L, μ, Σ)

        timeseries_start = [Bayesian_EMR_MSM.MSM_Timeseries_Point{Float64}(
        x_start[ts],
        res_start[ts],
        repeat([1.0], num_obs)
        ) for ts in 1:num_ts]

        test_point_est = Bayesian_EMR_MSM.EMR_MSM_Estimate(model, timeseries_start,
            timeseries_start
        )

        Bayesian_EMR_MSM.write("point_est", test_point_est)

        pred = Bayesian_EMR_MSM.EMR_MSM_Prediction(test_point_est, num_pred, repeat([timestep],num_ts);
                use_last_layer=false)

        Bayesian_EMR_MSM.write("pred_point", pred)

        print(Bayesian_EMR_MSM.DataFrame(pred.pred_timeseries[1]))

        tau0 = 100.0
        num_pred_samples = 100

        dist_est = Bayesian_EMR_MSM.EMR_MSM_Estimate(pred.pred_timeseries, num_est_layers,
        num_samples, num_chains, tau0)

        Bayesian_EMR_MSM.write("dist_est", dist_est)

        mean_est = Bayesian_EMR_MSM.EMR_MSM_Model_PointEstimate(
                    dist_est.model,
                    num_params,
                    num_est_layers,
                    mean)

        test_pred_ts_med = Bayesian_EMR_MSM.MSM_Timeseries_Point{Float64}(
            dist_est.pred_timeseries,
            x -> median(x))

        dist_pred = Bayesian_EMR_MSM.EMR_MSM_Prediction(
            dist_est, num_pred, repeat([timestep],num_ts), num_pred_samples,
            repeat([1],num_ts);
            use_last_layer=false
        )

        Bayesian_EMR_MSM.write("pred_dist", dist_pred)

        @test true == true
    end

end
