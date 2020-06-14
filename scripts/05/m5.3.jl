# using TuringModels
using CSV, DataFrames, Query, Turing, Optim, StatsBase

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

post = DataFrame(m5_8) 

# post |> @rename("α[1]" => "male")

# retrieve category names
dict = zip(["α[1]", "α[2]"], levels(df.sexcat)) |> Dict
rename!(post, dict)

post = post |> @mutate(diff = _.female - _.male) |> DataFrame

precis(post)