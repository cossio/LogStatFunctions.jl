module LogStatFunctionsChainRulesCoreExt

using LogStatFunctions: logmeanexp, logvarexp, logstdexp
using LogExpFunctions: log1mexp, logsumexp
import ChainRulesCore

function ChainRulesCore.frule((_, Δx), ::typeof(logmeanexp), x::AbstractArray{<:Real}; dims = :)
    Ω = logmeanexp(x; dims)
    ΔΩ = sum(_softmax(x, dims) .* Δx; dims)
    return Ω, ΔΩ
end

function ChainRulesCore.rrule(::typeof(logmeanexp), x::AbstractArray{<:Real}; dims = :)
    Ω = logmeanexp(x; dims)
    return Ω, _∂x_pullback(_softmax(x, dims), x)
end

function ChainRulesCore.frule(
        (_, Δx), ::typeof(logvarexp), x::AbstractArray{<:Real};
        dims = :, corrected::Bool = true, logmean = logmeanexp(x; dims)
    )
    Ω = logvarexp(x; dims, corrected, logmean)
    ΔΩ = sum(_∂x_logvarexp(x, dims) .* Δx; dims)
    return Ω, ΔΩ
end

function ChainRulesCore.rrule(
        ::typeof(logvarexp), x::AbstractArray{<:Real};
        dims = :, corrected::Bool = true, logmean = logmeanexp(x; dims)
    )
    Ω = logvarexp(x; dims, corrected, logmean)
    return Ω, _∂x_pullback(_∂x_logvarexp(x, dims), x)
end

function ChainRulesCore.frule(
        (_, Δx), ::typeof(logstdexp), x::AbstractArray{<:Real};
        dims = :, corrected::Bool = true, logmean = logmeanexp(x; dims)
    )
    Ω = logstdexp(x; dims, corrected, logmean)
    ΔΩ = sum(_∂x_logvarexp(x, dims) ./ 2 .* Δx; dims)
    return Ω, ΔΩ
end

function ChainRulesCore.rrule(
        ::typeof(logstdexp), x::AbstractArray{<:Real};
        dims = :, corrected::Bool = true, logmean = logmeanexp(x; dims)
    )
    Ω = logstdexp(x; dims, corrected, logmean)
    return Ω, _∂x_pullback(_∂x_logvarexp(x, dims) / 2, x)
end

# ∂/∂xⱼ log(mean(exp.(x))) = exp(xⱼ) / Σᵢ exp(xᵢ), i.e. softmax(x). Normalizing by the
# actual sum of the max-centered exponentials (rather than dividing exp(x - logmean) by n)
# makes a large common offset in x cancel exactly instead of leaking ulp-level errors of
# the offset's magnitude into the gradient.
function _softmax(x::AbstractArray{<:Real}, dims)
    y = exp.(x .- maximum(x; dims))
    return y ./ sum(y; dims)
end

# ∂/∂xⱼ log(var(exp.(x))) = 2 exp(xⱼ) (exp(xⱼ) - m) / Σᵢ (exp(xᵢ) - m)², with m = exp(logmean).
# The m-dependence on x drops out because Σᵢ (exp(xᵢ) - m) = 0, and `corrected` only shifts
# the result by a constant, so the gradient is the same either way. The `logmean` keyword is
# a cache of logmeanexp(x; dims), and the rules differentiate under that assumption (a
# ChainRules pullback cannot attribute tangents to keyword arguments, so treating logmean as
# an independent constant would silently drop the mean's own x-dependence); it is therefore
# recomputed here rather than taken from the caller. The gradient is translation-invariant,
# so everything is computed from max-centered values (a large common offset cancels exactly
# in t and never enters the arithmetic), and in the log domain (squaring expm1(d) directly
# can underflow even when the gradient itself is representable, e.g. for nearly equal
# entries). The centered log-mean uses log1p(mean(expm1(t))), which retains offsets below
# the epsilon of logsumexp(t) - log(n) (e.g. lm = -1e-200 for x = [-1e-200, 1e-200]); the
# expm1 terms are accumulated in at least Float64, since in half precision individually
# tiny terms round to -1 and their collective contribution to the mean is lost.
function _∂x_logvarexp(x::AbstractArray{<:Real}, dims)
    t = x .- maximum(x; dims)
    s = sum(_wexpm1, t; dims)
    n = length(x) ÷ length(s)
    # mean(exp.(t)) ≥ 1/n since the max entry contributes exp(0) = 1, so lm ≥ -log(n); the
    # floor catches sums rounding s to exactly -n (log1p(-1) == -Inf), which can only
    # happen when the max dominates and -log(n) is the correct value.
    lm = max.(log1p.(s ./ n), -log(n))
    # narrow back to the input's precision elementwise (via values, not eltype, so arrays
    # with abstract element types still work)
    d = oftype.(float.(t), t .- lm)
    # log(abs(expm1(d))) without materializing expm1(d), which can overflow in half
    # precision even though d ≤ log(n) keeps the final gradient representable.
    l = max.(d, 0) .+ log1mexp.(-abs.(d))
    S = logsumexp(2 .* l; dims)
    return sign.(d) .* 2 .* exp.(d .+ l .- S)
end

# expm1 with the accumulation widened to at least Float64 (BigFloat stays BigFloat)
_wexpm1(u::Real) = expm1(convert(promote_type(Float64, typeof(u)), u))

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
