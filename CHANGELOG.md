# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

- Add a ChainRulesCore extension defining `frule`s and `rrule`s for
  `logmeanexp`, `logvarexp` and `logstdexp` on real arrays, making them
  differentiable with ChainRules-based AD packages (e.g. Zygote).
- Support immutable arrays (e.g. `StaticArrays`) and complex arrays.
- Results preserve the input eltype (e.g. `Float32` in → `Float32` out), and
  the `log(N)` normalization is computed in the result's precision (so e.g.
  `BigFloat` inputs no longer lose accuracy to a `Float64` `log(N)`).
- Reductions mutate the freshly allocated `logsumexp` output in place,
  avoiding an extra allocation.
- Allow `LogExpFunctions` v1.0; require at least v0.3.26.
- Raise the minimum supported Julia to v1.10.
- Add an online manual: https://cossio.github.io/LogStatFunctions.jl/

## v2.1.0

- Preserve the input eltype in the returned results.

## v2.0.0

- **Breaking**: export `logmeanexp`, `logvarexp` and `logstdexp`, which
  previously had to be qualified or imported explicitly.

## v1.0.2

- No user-facing changes.

## v1.0.1

- Support Julia v1.8.

## v1.0.0

- Register at https://github.com/cossio/CossioJuliaRegistry.
