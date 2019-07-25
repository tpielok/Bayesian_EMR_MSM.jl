# todo: create sampler_instructions type

abstract type EMR_MSM_Model_Estimate{T<:AbstractFloat} end

struct EMR_MSM_Model_PointEstimate{T<:AbstractFloat} <: EMR_MSM_Model_Estimate{T}
    F::Array{T,1}
    Lin_mat::Array{T,2}
    Quad_mats::Array{T,3}
    RCorrs::ResCorrs{T}
    σ::T
end

struct EMR_MSM_Model_DistEstimate{T<:AbstractFloat} <: EMR_MSM_Model_Estimate{T}
    estimates::Array{EMR_MSM_Model_PointEstimate{T},1}
end

function EMR_MSM_Model_PointEstimate(F::S, A::S, B::S, L::S,
    σ::T, num_params::Integer, num_layers::Integer) where
    S <: AbstractVector{T} where T <: Real
    L_mats = ResCorrs(L, num_params, num_layers)

    Lin_mat  = reshape(A,(num_params, num_params))
    Quad_mats = reshape(B, (num_params, num_params, num_params))

    EMR_MSM_Model_PointEstimate{T}(F, Lin_mat, Quad_mats, L_mats, σ)
end

# Model for main-level regression
@model regr(x, dx, timesteps, num_params, num_obs) = begin
    logsigma ~ Normal(0,1)
    sigma = exp(logsigma)

    tau ~ Half(Cauchy(0,sigma/sqrt(num_obs)))

    F = Array{Float64}(undef, num_params)
    A = Array{Float64}(undef, num_params^2)
    B = Array{Float64}(undef, num_params^3)

    lambda_F = Array{Float64}(undef, size(F,1))
    lambda_A = Array{Float64}(undef, size(A,1))
    lambda_B = Array{Float64}(undef, size(B,1))
    lambda_F ~ [Half(Cauchy(0,1))]
    lambda_A ~ [Half(Cauchy(0,1))]
    lambda_B ~ [Half(Cauchy(0,1))]

    F ~ MvNormal(tau*lambda_F)
    A ~ MvNormal(tau*lambda_A)
    B ~ MvNormal(tau*lambda_B)

    A_mats = reshape(A,(num_params,num_params))
    B_mats = reshape(B,(num_params,num_params,num_params))

    for i=1:num_obs-1
        A_v = -A_mats*x[i,:]
        B_v = [transpose(B_mats[:,:,k]*x[i,:])*x[i,:] for k in 1:num_params]
        for j=1:num_params
            dx[num_params*(i-1)+j] ~
                Normal(timesteps[i+1]*(A_v[j] + B_v[j] + F[j]),sigma)
        end
    end
end

function EMR_MSM_Model_DistEstimate(timeseries::MSM_Timeseries_Point{T},
    num_layers::Integer, num_samples::Integer) where T <: Real

    num_obs = length(timeseries)
    num_params = params(timeseries)

    σ = 0.002
    num_pred = 100

    rand_gen = Normal(0,0.03)
    L = rand(rand_gen,
    length(ResCorrs, num_params, num_layers))

    x = values(timeseries)
    dx = vec(transpose(x[2:num_obs,:]) - transpose(x[1:num_obs-1, :]))

    print(dx)

    chn = sample(regr(x,dx, timesteps(timeseries), num_params, num_obs),
        Turing.NUTS(num_samples,  0.65))

    print(chn)

    EMR_MSM_Model_DistEstimate{T}([EMR_MSM_Model_PointEstimate(
        Array{T}(chn[:F].value[j, 1:num_params, 1]),
        Array{T}(chn[:A].value[j, 1:num_params^2, 1]),
        Array{T}(chn[:B].value[j, 1:num_params^3, 1]),
        Array{T}(undef,length(ResCorrs, num_params, num_layers)),
        exp(chn[:logsigma].value[j, 1, 1]), num_params, num_layers
    ) for j in 1:num_samples])
end

function EMR_MSM_Model_PointEstimate(dist::EMR_MSM_Model_DistEstimate{T},
    num_params::Integer, num_layers::Integer,
    aggregate_fun::Function) where T <: Real

    F = mapslices(aggregate_fun,hcat([vec(est.F) for est in dist.estimates]...);dims=2)[:,1]
    A = mapslices(aggregate_fun,hcat([vec(est.Lin_mat) for est in dist.estimates]...);dims=2)[:,1]
    B = mapslices(aggregate_fun,hcat([vec(est.Quad_mats) for est in dist.estimates]...);dims=2)[:,1]
    if num_layers > 0
        L = mapslices(aggregate_fun,hcat([vec(est.RCorrs) for est in dist.estimates]...);dims=2)[:,1]
    else
        L = Array{T}(undef, 0)
    end
    σ = aggregate_fun([est.σ for est in dist.estimates])

    EMR_MSM_Model_PointEstimate(F, A, B, L,
        σ, num_params, num_layers)
end


Base.size(emr_msm::EMR_MSM_Model_DistEstimate) = size(emr_msm.estimates, 1)
