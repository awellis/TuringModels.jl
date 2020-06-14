

function recover_levels(var::String, factor::CategoricalArray)
    flevels = levels(factor)
    nlevels = length(flevels)
    params = [var * "["] .* map(string, 1:nlevels) .* "]"
    dict = Dict(zip(params, flevels))
    return dict
end


# usage example
# dict = recover_levels("α", df.clade)
# rename!(post, dict)

