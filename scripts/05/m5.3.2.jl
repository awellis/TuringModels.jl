# using TuringModels
using CSV, DataFrames, Query, Turing, Optim, StatsBase, StatsPlots

# define precis from StatisticalRethinking module
using NamedArrays
function precis(df::DataFrame; digits=3, depth=Inf, alpha=0.11)
  m = zeros(length(names(df)), 5)
  for (indx, col) in enumerate(names(df))
    m[indx, 1] = mean(df[:, col])
    m[indx, 2] = std(df[:, col])
    q = quantile(df[:, col], [alpha/2, 0.5, 1-alpha/2])
    m[indx, 3] = q[1]
    m[indx, 4] = q[2]
    m[indx, 5] = q[3]
  end
  NamedArray(
    round.(m, digits=digits), 
    (names(df), ["mean", "sd", "5.5%", "50%", "94.5%"]), 
    ("Rows", "Cols"))
end



# ### snippet 5.3

# TODO: use population standard deviation to be consistent with Rethinking?
scale(x) = (x .- mean(x)) ./ std(x)

df = CSV.read(joinpath(@__DIR__, "data", "milk.csv"))
categorical!(df, :clade)

df = df |> @mutate(clade_id = levelcode(_.clade)) |> DataFrame

df.K = scale(df.kcal_per_g)

K = df.K
clade = df.clade_id

@model model5_9(K, clade) = begin
    n_clades = length(unique(clade))
    σ ~ Exponential(1)
    α ~ filldist(Normal(0, 0.5), n_clades)
    μ = α[clade]
    @. K ~ Normal(μ, σ)
end

model = model5_9(K, clade)

m5_9 = sample(model, NUTS(), 2000)
loglik = logprob"K = K | chain = m5_9, clade = clade, model = model5_9"

describe(m5_9)
plot(m5_9)

params = ["α["] .* map(string, 1:4) .* "]"
dict = Dict(zip(params, levels(df.clade)))

post = DataFrame(m5_9)
rename!(post, dict)
post_long = DataFrames.stack(post)

using VegaLite

post_long |>
  @filter(_.variable != "σ") |>

  @vlplot(y={"variable:o", title = ""}) +
  @vlplot(
    mark={
        :point,
        filled=true
    },
    x={
        "mean(value)",
        scale={zero=false},
        title="expected kcal (std)"},
    color={value=:black}) +
  @vlplot(
    mark={
        :errorbar,
        extent=:stdev
     },
     x={:value, title="expected kcal (std)"})



using Random
Random.seed!(63);

df.house = sample(repeat(1:4, inner = 8), nrow(df))

@model model5_10(K, clade, house) = begin
    n_clades = length(unique(clade))
    n_houses = length(unique(house))
    σ ~ Exponential(1)
    α ~ filldist(Normal(0, 0.5), n_clades)
    h ~ filldist(Normal(0, 0.5), n_houses)
    μ = α[clade] + h[house]
    @. K ~ Normal(μ, σ)
end

K, clade, house = df.K, df.clade_id, df.house
model = model5_10(K, clade, house)

m5_10 = sample(model, NUTS(), 2000)


post = DataFrame(m5_10)

function recover_levels(var::String, factor::CategoricalArray)
    flevels = levels(factor)
    nlevels = length(flevels)
    params = [var * "["] .* map(string, 1:nlevels) .* "]"
    dict = Dict(zip(params, flevels))
    return dict
end


clades = recover_levels("α", df.clade)
rename!(post, clades)

houses = categorical(["Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"])
houses = recover_levels("h", houses)
rename!(post, houses)

post_long = DataFrames.stack(post)


post_long |>
  @filter(_.variable != "σ") |>

  @vlplot(y={"variable:o", title = ""}) +
  @vlplot(
    mark={
        :point,
        filled=true
    },
    x={
        "mean(value)",
        scale={zero=false},
        title="expected kcal (std)"},
    color={value=:black}) +
  @vlplot(
    mark={
        :errorbar,
        extent=:stdev
     },
     x={:value, title="expected kcal (std)"})