```@meta
CurrentModule = LogStatFunctions
```

# LogStatFunctions.jl

Numerically stable logarithms of the mean, variance, and standard deviation of
exponentials — a small companion to
[LogExpFunctions.jl](https://github.com/JuliaStats/LogExpFunctions.jl).

Given an array `A`, these compute `log`-of-a-statistic-of-`exp.(A)` without ever
forming `exp.(A)` (which would overflow for large entries):

| function | computes |
|---|---|
| `logmeanexp(A; dims=:)` | `log.(mean(exp.(A); dims))` |
| `logvarexp(A; dims=:, corrected=true)` | `log.(var(exp.(A); dims, corrected))` |
| `logstdexp(A; dims=:, corrected=true)` | `log.(std(exp.(A); dims, corrected))` |

```julia
using LogStatFunctions

A = 1000 .* randn(10^4)      # exp.(A) would overflow Float64
logmeanexp(A)                # finite and accurate
logstdexp(A; corrected=false)
```

`logvarexp` and `logstdexp` also accept `logmean` to reuse a precomputed
`logmeanexp(A; dims)`.

Each function has an in-place variant (`logmeanexp!`, `logvarexp!`,
`logstdexp!`) that writes the result into a preallocated output array,
reducing over the singleton dimensions of `out`:

```julia
A = randn(10, 5)
out = zeros(1, 5)
logmeanexp!(out, A)          # same as logmeanexp(A; dims=1)
```

All three functions have
[ChainRules](https://github.com/JuliaDiff/ChainRulesCore.jl) derivative rules
for real arrays (loaded automatically when ChainRulesCore is in the
environment), so they can be differentiated with ChainRules-based AD packages
such as Zygote. Complex arrays are not covered by these rules.

## Reference

```@docs
logmeanexp
logvarexp
logstdexp
logmeanexp!
logvarexp!
logstdexp!
```
