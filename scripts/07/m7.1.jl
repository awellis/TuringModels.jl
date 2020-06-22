using DataFrames, Query, Turing, StatsFuns
  
d = DataFrame(species = ["afarensis", "africanus", "habilis", "boisei", "rudolfensis", "ergaster", "sapiens"], 
         brain = [438, 452, 612, 521, 752, 871, 1350], 
         mass = [37.0, 35.5, 34.5, 41.5, 55.5, 61.0, 53.5])



# d = d |> @mutate(mass_std = (_.mass - mean(d.mass)),
#          brain_std = _.brain / maximum(d.brain)) |> DataFrame

# TODO: use population standard deviation to be consistent with Rethinking
scale(x) = (x .- mean(x)) ./ std(x, corrected = true)
d.mass_std = scale(d.mass)
d.brain_std = d.brain ./ maximum(d.brain)


@model model7_1(brain_std, mass_std) = begin
    log_σ ~ Normal(0, 1)
    σ = exp(log_σ)
    α ~ Normal(0.5, 1)
    β ~ Normal(0, 10)
    for i ∈ eachindex(brain_std)
        brain_std[i] ~ Normal(α + β * mass_std[i], σ)
    end
end

brain_std = d.brain_std
mass_std = d.mass_std

model = model7_1(brain_std, mass_std)

m7_1 = sample(model, NUTS(), 2000, save_state = true)

describe(m7_1)

loglik = 
[logprob"brain_std = brain_std[i] | mass_std = mass_std[i], chain = m7_1" for i ∈ eachindex(brain_std)]

ll = DataFrame(loglik)

function WAIC(ll::DataFrame)
    n_samples, n_obs = size(ll)
    logsums = [logsumexp(col) for col ∈ eachcol(ll)]
    lppd = logsums .- log(n_samples)
    pWAIC = [var(col) for col ∈ eachcol(ll)]
    sum_lppd = sum(lppd) 
    sum_pWAIC =sum(pWAIC)
    waic = -2 * (sum_lppd - sum_pWAIC)
    waic_vec = -2 .* (lppd .- pWAIC)
    se = sqrt(n_obs * var(waic_vec))

    return (waic = waic, se = se, lppd = sum_lppd, pwaic = sum_pWAIC)
end

waic, se, lppd, pWAIC = WAIC(ll)