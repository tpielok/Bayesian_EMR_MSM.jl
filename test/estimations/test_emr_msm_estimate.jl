using Bayesian_EMR_MSM
using Test

using Distributions
using LinearAlgebra
import Random

@testset "emr_msm_estimate" begin

    @testset "EMR_MSM_PointEstimate" begin
        Bayesian_EMR_MSM.cmdstan_home!("/cmdstan")
        delta = 1/50

        seed = 1
        rand_gen = Normal(0,5)
        num_obs    = 1
        num_params = 2
        num_layers = 2
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

        x_start   = rand(rand_gen, num_obs, num_params)
        res_start = rand(rand_gen, num_obs, num_params, num_layers+1)

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

        timeseries_start = Bayesian_EMR_MSM.MSM_Timeseries_Point{Float64}(
        x_start,
        res_start,
        repeat([1.0], num_obs)
        )

        test_point_est = Bayesian_EMR_MSM.EMR_MSM_Estimate(model, [timeseries_start],
            [Bayesian_EMR_MSM.MSM_Timeseries_Dist{Float64}(
                Array{Float64}(undef,0,0,1),
                Array{Float64}(undef, 0, 0, 0, 1), [1.0])]
        )

        pred = Bayesian_EMR_MSM.EMR_MSM_Prediction(test_point_est, num_pred, [timestep])

        tau0 = 100.0
        num_pred_samples = 100

        dist_est = Bayesian_EMR_MSM.EMR_MSM_Estimate(pred.pred_timeseries, num_layers,
        num_samples, num_chains, tau0)

        mean_est = Bayesian_EMR_MSM.EMR_MSM_Model_PointEstimate(
                    dist_est.model,
                    num_params,
                    num_layers,
                    mean)

        test_pred_ts_med = Bayesian_EMR_MSM.MSM_Timeseries_Point{Float64}(
            dist_est.pred_timeseries,
            x -> median(x))

        dist_pred = Bayesian_EMR_MSM.EMR_MSM_Prediction(
            dist_est, num_pred, [timestep], num_pred_samples
        )

        print(dist_pred.pred_timeseries[1].x)

        @test true == true
    end

end
