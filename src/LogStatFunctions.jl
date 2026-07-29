module LogStatFunctions

using LogExpFunctions: logsumexp, logsumexp!, logsubexp

export logmeanexp, logvarexp, logstdexp
export logmeanexp!, logvarexp!, logstdexp!

"""
    logmeanexp(A; dims=:)

Computes `log.(mean(exp.(A); dims))`, in a numerically stable way.
"""
function logmeanexp(A::AbstractArray; dims = :)
    R = logsumexp(A; dims)
    N = length(A) ÷ length(R)
    return sub!(R, loglen(R, N))
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
        return sub!(R, loglen(R, N - 1))
    else
        return sub!(R, loglen(R, N))
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

"""
    logmeanexp!(out, A)

In-place version of [`logmeanexp`](@ref): computes `log.(mean(exp.(A); dims))` over the
singleton dimensions of `out`, and writes the result to `out`.
"""
function logmeanexp!(out::AbstractArray, A::AbstractArray)
    logsumexp!(out, A)
    N = length(A) ÷ length(out)
    out .-= log(N)
    return out
end

"""
    logvarexp!(out, A; corrected=true, logmean=logmeanexp!(similar(out), A))

In-place version of [`logvarexp`](@ref): computes `log.(var(exp.(A); dims, corrected))`
over the singleton dimensions of `out`, and writes the result to `out`.
"""
function logvarexp!(
        out::AbstractArray, A::AbstractArray;
        corrected::Bool = true, logmean = logmeanexp!(similar(out), A)
    )
    logsumexp!(out, 2logsubexp.(A, logmean))
    N = length(A) ÷ length(out)
    if corrected
        out .-= log(N - 1)
    else
        out .-= log(N)
    end
    return out
end

"""
    logstdexp!(out, A; corrected=true, logmean=logmeanexp!(similar(out), A))

In-place version of [`logstdexp`](@ref): computes `log.(std(exp.(A); dims, corrected))`
over the singleton dimensions of `out`, and writes the result to `out`.
"""
function logstdexp!(
        out::AbstractArray, A::AbstractArray;
        corrected::Bool = true, logmean = logmeanexp!(similar(out), A)
    )
    logvarexp!(out, A; corrected, logmean)
    out ./= 2
    return out
end

loglen(R::AbstractArray, N::Integer) = log(convert(real(eltype(R)), N))
loglen(R::Number, N::Integer) = log(convert(real(typeof(R)), N))

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
