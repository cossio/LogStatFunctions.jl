module LogStatFunctionsChainRulesCoreExt

using LogStatFunctions: logmeanexp, logvarexp, logstdexp
import ChainRulesCore

function ChainRulesCore.frule((_, Δx), ::typeof(logmeanexp), x::AbstractArray{<:Real}; dims = :)
    Ω = logmeanexp(x; dims)
    n = length(x) ÷ length(Ω)
    ΔΩ = sum(exp.(x .- Ω) .* Δx; dims) ./ n
    return Ω, ΔΩ
end

function ChainRulesCore.rrule(::typeof(logmeanexp), x::AbstractArray{<:Real}; dims = :)
    Ω = logmeanexp(x; dims)
    n = length(x) ÷ length(Ω)
    return Ω, _∂x_pullback(exp.(x .- Ω) ./ n, x)
end

function ChainRulesCore.frule(
        (_, Δx), ::typeof(logvarexp), x::AbstractArray{<:Real};
        dims = :, corrected::Bool = true, logmean = logmeanexp(x; dims)
    )
    Ω = logvarexp(x; dims, corrected, logmean)
    ΔΩ = sum(_∂x_logvarexp(x, logmean, dims) .* Δx; dims)
    return Ω, ΔΩ
end

function ChainRulesCore.rrule(
        ::typeof(logvarexp), x::AbstractArray{<:Real};
        dims = :, corrected::Bool = true, logmean = logmeanexp(x; dims)
    )
    Ω = logvarexp(x; dims, corrected, logmean)
    return Ω, _∂x_pullback(_∂x_logvarexp(x, logmean, dims), x)
end

function ChainRulesCore.frule(
        (_, Δx), ::typeof(logstdexp), x::AbstractArray{<:Real};
        dims = :, corrected::Bool = true, logmean = logmeanexp(x; dims)
    )
    Ω = logstdexp(x; dims, corrected, logmean)
    ΔΩ = sum(_∂x_logvarexp(x, logmean, dims) ./ 2 .* Δx; dims)
    return Ω, ΔΩ
end

function ChainRulesCore.rrule(
        ::typeof(logstdexp), x::AbstractArray{<:Real};
        dims = :, corrected::Bool = true, logmean = logmeanexp(x; dims)
    )
    Ω = logstdexp(x; dims, corrected, logmean)
    return Ω, _∂x_pullback(_∂x_logvarexp(x, logmean, dims) / 2, x)
end

# ∂/∂xⱼ log(var(exp.(x))) = 2 exp(xⱼ) (exp(xⱼ) - m) / Σᵢ (exp(xᵢ) - m)², with m = exp(logmean).
# The m-dependence on x drops out because Σᵢ (exp(xᵢ) - m) = 0, and `corrected` only shifts
# the result by a constant, so the gradient is the same either way.
function _∂x_logvarexp(x::AbstractArray{<:Real}, logmean, dims)
    d = x .- logmean
    e = expm1.(d)
    return (2 .* exp.(d) .* e) ./ sum(abs2, e; dims)
end

function _∂x_pullback(∂x, x)
    project_x = ChainRulesCore.ProjectTo(x)
    function pullback(Ω̄)
        x̄ = ChainRulesCore.InplaceableThunk(
            Δ -> Δ .+= Ω̄ .* ∂x,
            ChainRulesCore.@thunk(project_x(Ω̄ .* ∂x)),
        )
        return ChainRulesCore.NoTangent(), x̄
    end
    return pullback
end

end # module
