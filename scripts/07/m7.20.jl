using DataFrames, Query, Turing, StatsFuns
  
using RDatasets

cars = dataset("datasets", "cars")


@model model7_20(dist, speed) = begin
    σ ~ Exponential(1)
    α ~ Normal(0, 100)
    β ~ Normal(0, 10)
    for i ∈ eachindex(dist)
        dist[i] ~ Normal(α + β * speed[i], σ)
    end
end

dist = cars.Dist
speed = cars.Speed


model = model7_20(dist, speed)


m7_20 = sample(model, NUTS(), 2000, save_state = true)

describe(m7_20)

loglik = 
[logprob"dist = dist[i] | speed = speed[i], chain = m7_20" for i ∈ eachindex(dist)]

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