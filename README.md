# LogStatFunctions.jl

[![CI](https://github.com/cossio/LogStatFunctions.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/cossio/LogStatFunctions.jl/actions/workflows/ci.yml)
[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://cossio.github.io/LogStatFunctions.jl/stable)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://cossio.github.io/LogStatFunctions.jl/dev)
[![codecov](https://codecov.io/gh/cossio/LogStatFunctions.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/cossio/LogStatFunctions.jl)

Numerically stable logarithms of the mean, variance, and standard deviation of
exponentials.

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
