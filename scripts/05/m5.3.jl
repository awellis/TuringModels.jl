# using TuringModels
using CSV, DataFrames, Query, Turing, Optim, StatsBase
# ### snippet 5.3

df = CSV.read(joinpath(@__DIR__, "data", "Howell1.csv"))

# create categorical sex variable
df = df |> @mutate(sexcat = _.male == 1 ? "male" : "female") |> DataFrame
categorical!(df, :sexcat)

```
get levels code to create index variable:
this could have been done like this:
    df = df |> @mutate(sexcat = _.male == 1 ? 2 : 1) |> DataFrame
for the same result, but this code will do the job if you already
have a categorial variable.
```

df = df |> @mutate(sex = levelcode(_.sexcat)) |> DataFrame

@model model5_8(y, group) = begin
    n_groups = length(unique(group))
    σ ~ Uniform(0, 50)
    α ~ filldist(Normal(178, 20), n_groups)
    μ = α[group]
    @. y ~ Normal(μ, σ)
end

y = df.height
group = df.sex

model = model5_8(y, group)
m5_8 = optimize(model, MAP())

m5_8 = sample(model, NUTS(), MCMCThreads(), 2000, 4)
describe(m5_8)