module LogStatFunctionsChainRulesCoreExt

using LogStatFunctions: logmeanexp, logvarexp, logstdexp
using LogExpFunctions: logsumexp, softmax
import ChainRulesCore

function ChainRulesCore.frule((_, Δx), ::typeof(logmeanexp), x::AbstractArray{<:Real}; dims = :)
    Ω = logmeanexp(x; dims)
    ΔΩ = sum(softmax(x; dims) .* Δx; dims)
    return Ω, ΔΩ
end

function ChainRulesCore.rrule(::typeof(logmeanexp), x::AbstractArray{<:Real}; dims = :)
    Ω = logmeanexp(x; dims)
    return Ω, _∂x_pullback(softmax(x; dims), x)
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

# ∂/∂xⱼ log(var(exp.(x))) = 2 exp(xⱼ) (exp(xⱼ) - m) / Σᵢ (exp(xᵢ) - m)², with m = exp(logmean);
# the dependence of m on x drops out since Σᵢ (exp(xᵢ) - m) = 0, and `corrected` only shifts
# the result by a constant. Log-domain to avoid under/overflow of the squared terms.
function _∂x_logvarexp(x::AbstractArray{<:Real}, logmean, dims)
    d = x .- logmean
    l = log.(abs.(expm1.(d)))
    S = logsumexp(2 .* l; dims)
    return sign.(d) .* 2 .* exp.(d .+ l .- S)
end

function _∂x_pullback(∂x, x)
    project_x = ChainRulesCore.ProjectTo(x)
    function pullback(Ω̄)
        ΔΩ = ChainRulesCore.unthunk(Ω̄)
        x̄ = ChainRulesCore.InplaceableThunk(
            Δ -> Δ .+= ΔΩ .* ∂x,
            ChainRulesCore.@thunk(project_x(ΔΩ .* ∂x)),
        )
        return ChainRulesCore.NoTangent(), x̄
    end
    return pullback
end

end # module
