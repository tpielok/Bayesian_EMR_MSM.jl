using Bayesian_EMR_MSM
using Test

using Distributions
import Random

@testset "emr_msm_estimate" begin

    @testset "EMR_MSM_PointEstimate" begin
        #Random.seed!(3)
        #rand_gen = Normal(0,10)
        Random.seed!(7)
        rand_gen = Normal(0,2)

        num_obs    = 1
        num_params = 2
        num_layers = 0
        σ = 0.002
        num_pred = 1000
        num_samples = 500
        timestep = 0.0005

        x_start   = rand(rand_gen, num_obs, num_params)
        res_start = rand(rand_gen, num_obs, num_params, num_layers+1)

        F = rand(rand_gen, num_params)
        A = rand(rand_gen, num_params, num_params)
        B = rand(rand_gen, num_params, num_params, num_params)
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

        dist_est = Bayesian_EMR_MSM.EMR_MSM_Estimate(pred.pred_timeseries, 0,
            num_samples)


        mean_est = Bayesian_EMR_MSM.EMR_MSM_Model_PointEstimate(
            dist_est.model,
            num_params,
            num_layers,
            mean)

        print("\nMean-Est:\n")
        print(mean_est)

        median_est = Bayesian_EMR_MSM.EMR_MSM_Model_PointEstimate(
            dist_est.model,
            num_params,
            num_layers,
            median)

        print("\nMedian-Est:\n")
        print(median_est)

        @test true == true
    end

end
