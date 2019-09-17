using Bayesian_EMR_MSM
using Test

using Distributions
import Random

@testset "emr_msm_estimate" begin

    @testset "EMR_MSM_PointEstimate" begin
        Bayesian_EMR_MSM.cmdstan_home!("/cmdstan")
        seed = 7
        rand_gen = Normal(0,5)
        num_obs    = 1
        num_params = 2
        num_layers = 0
        σ = 0.002
        num_pred = 100
        timestep = 0.0015
        F_zero = false
        A_zero = false
        B_zero = false

        num_samples = 1000
        num_chains = 4

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

        model = Bayesian_EMR_MSM.EMR_MSM_Model_PointEstimate(F, A, B, L,
        σ
        )

        timeseries_start = Bayesian_EMR_MSM.MSM_Timeseries_Point{Float64}(
        x_start,
        res_start,
        repeat([1.0], num_obs)
        )

        test_point_est = Bayesian_EMR_MSM.EMR_MSM_Estimate(model, timeseries_start)

        pred = Bayesian_EMR_MSM.EMR_MSM_Prediction(test_point_est, num_pred, timestep)

        copy(pred.pred_timeseries)

        tau0 = 100.0

        dist_est = Bayesian_EMR_MSM.EMR_MSM_Estimate(pred.pred_timeseries, 0,
        num_samples, num_chains, tau0)

        @test true == true
    end

end
