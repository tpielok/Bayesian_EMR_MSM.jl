# Distributions needed for horseshoe priors

offset = log(2)

struct HalfNormal <: ContinuousUnivariateDistribution
    n::Normal
end

Distributions.logpdf(d::HalfNormal, x::Real) = offset +
    Distributions.logpdf(d.n, x)
Distributions.minimum(d::HalfNormal) = d.n.μ
Distributions.maximum(d::HalfNormal) = +Inf

struct HalfCauchy <: ContinuousUnivariateDistribution
    c::Cauchy
end

Half(n::Normal) = HalfNormal(n)
Half(c::Cauchy) = HalfCauchy(c)

Distributions.logpdf(d::HalfCauchy, x::Real) = offset +
    Distributions.logpdf(d.c, x)
Distributions.minimum(d::HalfCauchy) = d.c.μ
Distributions.maximum(d::HalfCauchy) = +Inf
