# todo: create sampler_instructions type

abstract type EMR_MSM_Model_Estimate{T<:AbstractFloat} end

struct EMR_MSM_Model_PointEstimate{T<:AbstractFloat} <: EMR_MSM_Model_Estimate{T}
    F::Array{T,1}
    Lin_mat::Array{T,2}
    Quad_mats::Array{T,3}
    RCorrs::ResCorrs{T}
    μ::AbstractArray{T,1} # mean of the deepest residual layer
    Σ::AbstractArray{T,2} # covariance of the deepes residual layer
end

# TODO add F, Lin_mat ... views ...
struct EMR_MSM_Model_DistEstimate{T<:AbstractFloat} <: EMR_MSM_Model_Estimate{T}
    estimates::Array{EMR_MSM_Model_PointEstimate{T},1}
end

function EMR_MSM_Model_PointEstimate(F::S, A::S, B::S, L::S,
    μ::S, Σ::S, num_params::Integer, num_layers::Integer) where
    S <: AbstractVector{T} where T <: Real
    L_mats = ResCorrs(L, num_params, num_layers)

    Lin_mat  = reshape(A,(num_params, num_params))
    Quad_mats = reshape(B, (num_params, num_params, num_params))

    Σ = reshape(Σ,(num_params, num_params))

    EMR_MSM_Model_PointEstimate{T}(F, Lin_mat, Quad_mats, L_mats, μ, Σ)
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

    μ = mapslices(aggregate_fun,hcat([vec(est.μ) for est in dist.estimates]...);dims=2)[:,1]
    Σ = mapslices(aggregate_fun,hcat([vec(est.Σ) for est in dist.estimates]...);dims=2)[:,1]

    EMR_MSM_Model_PointEstimate(F, A, B, L,
        μ, Σ, num_params, num_layers)
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

    int get_rc_ind(int cur_layer, int num_params2){
        return num_params2*((cur_layer+2)*(cur_layer+1)/2 - 1);
    }

    matrix predict(row_vector start, matrix diff, int num_params, int num_data){
        matrix[num_data, num_params] pred;
        pred[1] = start;
        for(i in 2:num_data){
                pred[i] = pred[i-1] + diff[i-1];
        }
        return pred;
    }
}
data {
  int<lower=0> num_data;
  int<lower=0> num_params;
  int<lower=0> num_layers;
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
  int num_rescor_params =
    num_params*num_params*((num_layers+2)*(num_layers+1)/2 - 1);
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
  vector<lower=0>[num_rescor_params] r1_local_rc;
  vector<lower=0>[num_rescor_params] r2_local_rc;

  vector[num_params] f;
  vector[num_params2] l;
  vector[num_all_sym_params] q;
  vector[num_rescor_params] rc;
}
transformed parameters {
  real<lower=0> tau;

  vector<lower=0>[num_params] lambda_f;
  vector<lower=0>[num_params2] lambda_l;
  vector<lower=0>[num_all_sym_params] lambda_q;
  vector<lower=0>[num_rescor_params] lambda_rc;

  real sigma;
  vector[num_params] trafo_f;
  vector[num_params2] trafo_l;
  vector[num_all_sym_params] trafo_q;
  vector[num_rescor_params] trafo_rc;
  matrix[num_params,num_params] trafo_qq[num_params];

  matrix[num_data-1,(num_layers+1)*num_params] res_vec;
  matrix[num_data-1, num_params] trafo_dx_hat;
  matrix[num_data-1, num_params] r[num_layers+1];
  matrix[num_data-1, num_params] dr[num_layers];
  matrix[num_data-1, num_params] trafo_r_hat[num_layers];
  matrix[num_data-1, num_params] trafo_dr_hat[num_layers];
  matrix[num_data-1, num_params] trafo_r_tilde[num_layers];
  matrix[num_data-1, num_params] trafo_dr_tilde[num_layers];

  trafo_dx_hat = rep_matrix(0, num_data-1, num_params);
  res_vec = rep_matrix(0, num_data-1,(num_layers+1)*num_params);
  for(i in 1:num_layers){
      trafo_dr_hat[i] = rep_matrix(0, num_data-1, num_params);
      trafo_r_hat[i] = rep_matrix(0, num_data-1, num_params);
      trafo_r_tilde[i] = rep_matrix(0, num_data-1, num_params);
      dr[i] = rep_matrix(0, num_data-1, num_params);
      r[i] = rep_matrix(0, num_data-1, num_params);
      trafo_dr_tilde[i] = rep_matrix(0, num_data-1, num_params);
  }

  tau = r1_global * sqrt(r2_global);
  lambda_f = r1_local_f .* sqrt(r2_local_f);
  lambda_l = r1_local_l .* sqrt(r2_local_l);
  lambda_q = r1_local_q .* sqrt(r2_local_q);
  lambda_rc = r1_local_rc .* sqrt(r2_local_rc);

  trafo_f = f .* lambda_f * tau;
  trafo_l = l .* lambda_l * tau;
  trafo_q = q .* lambda_q * tau;
  trafo_rc = rc .* lambda_rc * tau;

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

  r[1] = dx - trafo_dx_hat;

  if(num_layers > 0){

      res_vec[:,:num_params] = x[:(num_data-1),:];
      for(i in 1:num_layers){
        dr[i][1:(num_data-2),:] = r[i][2:,:] - r[i][:(num_data-2),:];
        res_vec[:,(i*num_params+1):((i+1)*num_params)] = r[i];
        for(j in 1:(num_data-1-i)){
            trafo_dr_hat[i][j] = time_steps[j+1] *
                res_vec[j,:(num_params*(i+1))] * to_matrix(trafo_rc[
                (get_rc_ind(i-1, num_params2)+1):get_rc_ind(i, num_params2)],
                num_params, (i+1)*num_params)';
        }

        r[i+1] = dr[i] - trafo_dr_hat[i];
      }

      trafo_dr_tilde[num_layers] = trafo_dr_hat[num_layers];
      for(i in (-num_layers):(-1)){
          trafo_r_tilde[-i] = predict(r[-i][1], trafo_dr_tilde[-i],
            num_params, num_data-1);
          if(-i > 1){
                trafo_dr_tilde[-1-i] = trafo_dr_hat[-1-i] +
                    trafo_r_tilde[-i];
          }
      }

      trafo_dx_hat = trafo_dx_hat + trafo_r_tilde[1];
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
  r1_local_rc ~ normal(0.0, 1);
  r2_local_rc ~ inv_gamma(0.5*nu_local, 0.5*nu_local);

  f ~ normal(0,1);
  l ~ normal(0,1);
  q ~ normal(0,1);

  to_vector(dx[1:(num_data-1-num_layers)]) ~ normal(
    to_vector(trafo_dx_hat[1:(num_data-1-num_layers)]), sigma);
}
";

function EMR_MSM_Model_DistEstimate(timeseries::MSM_Timeseries_Point{T},
    num_layers::Integer, num_samples::Integer, num_chains::Integer, tau0::T=one(T);
    uncorr = true) where T <: Real

    num_obs = length(timeseries)
    num_params = params(timeseries)

    x = values(timeseries)
    dx = (x[2:num_obs,:]) - (x[1:num_obs-1, :])

    stanmodel = Stanmodel(name="timeseries_mv" * string(rand(1:10000)),
        model=ts_mv_stanmodel,
        random=CmdStan.Random(123), nchains=num_chains, num_samples=num_samples)

    ts_data = Dict("num_data" => num_obs,
            "num_params" => num_params,
            "num_layers" => num_layers,
            "x" => x,
            "dx" => dx,
            "time_steps" => timesteps(timeseries),
            "scale_global" => tau0/sqrt(num_obs),
            "nu_global" => 1,
            "nu_local" => 1
            );
    rc, chns, cnames = stan(stanmodel, ts_data, "./", CmdStanDir=CMDSTAN_HOME)

    dx_est = [hcat([[chns.value[:,"trafo_dx_hat." * string(i) *"."* string(p), (div(j-1,num_samples)+1)
                        ][mod(j-1,num_samples)+1] for i in 1:(num_obs-1)]
                        for p in 1:num_params]...)
                        for j in 1:(num_samples*num_chains)]

    res_est = [[hcat([[chns.value[:,"r."* string(l) *"." * string(i) *"."* string(p), (div(j-1,num_samples)+1)
                        ][mod(j-1,num_samples)+1] for i in 1:(num_obs-1)]
                        for p in 1:num_params]...)
                        for l in 1:(num_layers+1)]
                        for j in 1:(num_samples*num_chains)]

    dx_est_t = reshape(hcat(dx_est...),(num_obs-1,num_params,num_samples*num_chains))
    res_est_t = reshape(hcat(hcat(res_est...)...),
        (num_obs-1,num_params,num_layers+1,num_samples*num_chains))

    x_est = Array{Float64,3}(undef, num_obs-1, num_params, num_samples*num_chains)
    for j in 1:(num_samples*num_chains)
        x_est[1,:,j] = x[1,:]
        for i in 2:(num_obs-1)
            x_est[i,:,j] = x_est[i-1,:,j] + dx_est_t[i-1,:,j]
        end
    end

    pred_timeseries = MSM_Timeseries_Dist{T}(x_est, res_est_t,
        timesteps(timeseries)[1:(num_obs-1)])

    return EMR_MSM_Model_DistEstimate{T}([EMR_MSM_Model_PointEstimate(
        [vec(chns.value[:,"trafo_f." * string(i),
        (div(j-1,num_samples)+1)])[mod(j-1,num_samples)+1] for i in 1:num_params],
        [vec(chns.value[:,"trafo_l." * string(i),
        (div(j-1,num_samples)+1)])[mod(j-1,num_samples)+1] for i in 1:num_params^2],
        vcat([vec(hcat([vec([chns.value[:,"trafo_qq." * string(i) *"."* string(k) *"." * string(l),
        (div(j-1,num_samples)+1)][mod(j-1,num_samples)+1]
            for k in 1:num_params]) for l in 1:num_params]...)) for i in 1:num_params]...),
        if num_layers == 0
            Array{T}(undef,length(ResCorrs, num_params, num_layers))
        else
            [vec(chns.value[:,"trafo_rc." * string(i),
            (div(j-1,num_samples)+1)])[mod(j-1,num_samples)+1] for i in 1:length(ResCorrs, num_params, num_layers)]
        end
        ,
        ifelse(uncorr, zeros(T, num_params),mean.([dx_est_t[:,i,j] for i in 1:num_params])),
        ifelse(uncorr, collect(vec(Diagonal(cov(dx_est_t[:,:,j])))),
            vec(cov(dx_est_t[:,:,j]))),
        num_params,
        num_layers
        ) for j in 1:(num_samples*num_chains)]),
        pred_timeseries
end



cmdstan_home!(home::String) = set_cmdstan_home!(home)

Base.size(emr_msm::EMR_MSM_Model_DistEstimate) = size(emr_msm.estimates, 1)
