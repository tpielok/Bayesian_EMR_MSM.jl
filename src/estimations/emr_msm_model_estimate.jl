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

const ts_mv_stanmodel = "
functions{
    matrix sym_mat(vector v, int num_params){
        int c;
        matrix[num_params,num_params] sym;

        c = 1;
        sym = rep_matrix(0,num_params,num_params);

        for (i in 1:(num_params-1)){
            for(j in (i+1):num_params){
                sym[i,j] = v[c];
                c = c + 1;
            }
        }

        sym = sym + sym' + diag_matrix(v[((num_params*num_params-num_params)/2+1):
                                    ((num_params*num_params+num_params)/2)]);
        return sym;
    }
}
data {
  int<lower=0> num_data;
  int<lower=0> num_params;
  matrix[num_data, num_params] x;
  matrix[num_data-1, num_params] dx;
  vector[num_data] time_steps;
  real<lower=0> scale_global;
  real<lower=1> nu_global;            // degree of freedom for the half-t prior
  real<lower=1> nu_local;             // degree of freedom for the half-t prior
}
transformed data{
  int num_params2 = num_params*num_params;
  int num_sym_params = (num_params*num_params + num_params) / 2;
  int num_all_sym_params = num_params*(num_params*num_params + num_params) / 2;
}
parameters {
  real<lower=0> r1_global;
  real<lower=0> r2_global;
  real logsigma;

  vector<lower=0>[num_params] r1_local_f;
  vector<lower=0>[num_params] r2_local_f;
  vector<lower=0>[num_params2] r1_local_l;
  vector<lower=0>[num_params2] r2_local_l;
  vector<lower=0>[num_all_sym_params] r1_local_q;
  vector<lower=0>[num_all_sym_params] r2_local_q;

  vector[num_params] f;
  vector[num_params2] l;
  vector[num_all_sym_params] q;
}
transformed parameters {
  real<lower=0> tau;

  vector<lower=0>[num_params] lambda_f;
  vector<lower=0>[num_params2] lambda_l;
  vector<lower=0>[num_all_sym_params] lambda_q;

  real sigma;
  vector[num_params] trafo_f;
  vector[num_params2] trafo_l;
  vector[num_all_sym_params] trafo_q;
  matrix[num_params,num_params] trafo_qq[num_params];

  matrix[num_data-1, num_params] trafo_dx_hat;

  trafo_dx_hat = rep_matrix(0, num_data-1, num_params);

  tau = r1_global * sqrt(r2_global);
  lambda_f = r1_local_f .* sqrt(r2_local_f);
  lambda_l = r1_local_l .* sqrt(r2_local_l);
  lambda_q = r1_local_q .* sqrt(r2_local_q);

  trafo_f = f .* lambda_f * tau;
  trafo_l = l .* lambda_l * tau;
  trafo_q = q .* lambda_q * tau;

  sigma = exp(logsigma);

  for (i in 1:(num_data-1)){
    trafo_dx_hat[i] = -x[i]*to_matrix(trafo_l, num_params, num_params)' + trafo_f';
    for (j in 1:num_params){
        trafo_qq[j] = sym_mat(
                trafo_q[(1+(j-1)*num_sym_params):(j*num_sym_params)],
                num_params);
        trafo_dx_hat[i,j] = trafo_dx_hat[i,j] +
            quad_form(trafo_qq[j],
                x[i]');
    }
    trafo_dx_hat[i] = time_steps[i+1]*trafo_dx_hat[i];
  }

}
model {
  r1_global ~ normal(0.0, scale_global*sigma);
  r2_global ~ inv_gamma(0.5*nu_global, 0.5*nu_global);

  r1_local_f ~ normal(0.0, 1);
  r2_local_f ~ inv_gamma(0.5*nu_local, 0.5*nu_local);
  r1_local_l ~ normal(0.0, 1);
  r2_local_l ~ inv_gamma(0.5*nu_local, 0.5*nu_local);
  r1_local_q ~ normal(0.0, 1);
  r2_local_q ~ inv_gamma(0.5*nu_local, 0.5*nu_local);

  f ~ normal(0,1);
  l ~ normal(0,1);
  q ~ normal(0,1);

  to_vector(dx) ~ normal(to_vector(trafo_dx_hat), sigma);
}
";

function EMR_MSM_Model_DistEstimate(timeseries::MSM_Timeseries_Point{T},
    num_layers::Integer, num_samples::Integer, num_chains::Integer, tau0::T=one(T)) where T <: Real

    num_obs = length(timeseries)
    num_params = params(timeseries)

    x = values(timeseries)
    dx = (x[2:num_obs,:]) - (x[1:num_obs-1, :])

    stanmodel = Stanmodel(name="timeseries_mv" * string(rand(1:10000)),
        model=ts_mv_stanmodel,
        random=CmdStan.Random(123), nchains=num_chains, num_samples=num_samples)

    ts_data = Dict("num_data" => num_obs,
            "num_params" => num_params,
            "x" => x,
            "dx" => dx,
            "time_steps" => timesteps(timeseries),
            "scale_global" => tau0/sqrt(num_obs),
            "nu_global" => 1,
            "nu_local" => 1
            );
    rc, chns, cnames = stan(stanmodel, ts_data, "./", CmdStanDir=CMDSTAN_HOME)

    Bayesian_EMR_MSM.EMR_MSM_Model_DistEstimate{T}([Bayesian_EMR_MSM.EMR_MSM_Model_PointEstimate(
        [vec(chns.value[:,"trafo_f." * string(i),
        (div(j-1,num_samples)+1)])[mod(j-1,num_samples)+1] for i in 1:num_params],
        [vec(chns.value[:,"trafo_l." * string(i),
        (div(j-1,num_samples)+1)])[mod(j-1,num_samples)+1] for i in 1:num_params^2],
        vcat([vec(hcat([vec([chns.value[:,"trafo_qq." * string(i) *"."* string(k) *"." * string(l),
        (div(j-1,num_samples)+1)][mod(j-1,num_samples)+1]
        for k in 1:num_params]) for l in 1:num_params]...)) for i in 1:num_params]...),
            Array{T}(undef,length(Bayesian_EMR_MSM.ResCorrs, num_params, num_layers)),
            chns.value[:,"sigma",(div(j-1,num_samples)+1)][mod(j-1,num_samples)+1],
            num_params, num_layers) for j in 1:(num_samples*num_chains)])
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

cmdstan_home!(home::String) = set_cmdstan_home!(home)

Base.size(emr_msm::EMR_MSM_Model_DistEstimate) = size(emr_msm.estimates, 1)
