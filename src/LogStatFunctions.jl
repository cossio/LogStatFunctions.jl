module LogStatFunctions

using LogExpFunctions: logsumexp, logsubexp

export logmeanexp, logvarexp, logstdexp

"""
    logmeanexp(A; dims=:)

Computes `log.(mean(exp.(A); dims))`, in a numerically stable way.
"""
function logmeanexp(A::AbstractArray; dims = :)
    R = logsumexp(A; dims)
    N = length(A) ÷ length(R)
    return sub!(R, log(N))
end

"""
    logvarexp(A; dims=:, corrected=true, logmean=logmeanexp(A; dims))

Computes `log.(var(exp.(A); dims))`, in a numerically stable way.
"""
function logvarexp(
        A::AbstractArray; dims = :, corrected::Bool = true, logmean = logmeanexp(A; dims)
    )
    R = logsumexp(2logsubexp.(A, logmean); dims)
    N = length(A) ÷ length(R)
    if corrected
        return sub!(R, log(N - 1))
    else
        return sub!(R, log(N))
    end
end

"""
    logstdexp(A; dims=:, corrected=true, logmean=logmeanexp(A; dims))

Computes `log.(std(exp.(A); dims))`, in a numerically stable way.
"""
function logstdexp(
        A::AbstractArray; dims = :, corrected::Bool = true, logmean = logmeanexp(A; dims)
    )
    return halve!(logvarexp(A; dims, corrected, logmean))
end

function sub!(R::AbstractArray, c::Real)
    if ismutabletype(typeof(R))
        return R .-= c
    else
        return R .- convert(real(eltype(R)), c)
    end
end
sub!(R::Number, c::Real) = R - convert(real(typeof(R)), c)

function halve!(R::AbstractArray)
    if ismutabletype(typeof(R))
        return R ./= 2
    else
        return R ./ 2
    end
end
halve!(R::Number) = R / 2

end # module
