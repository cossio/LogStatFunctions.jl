# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

- Results preserve the input eltype (e.g. `Float32` in → `Float32` out) for
  complex arrays too, and reductions mutate the fresh `logsumexp` output in
  place to avoid an extra allocation.
- Support immutable arrays (e.g. `StaticArrays`), which are no longer mutated.
- Allow `LogExpFunctions` v1.0; require at least v0.3.26.
- Raise the minimum supported Julia to v1.10.
- Add a Documenter manual, Runic formatting checks, coverage upload, Aqua
  ambiguity checks and ExplicitImports tests, and restore TagBot.
- Declare `test` and `docs` as Pkg workspace projects, and stop tracking
  `Manifest.toml` files.

## v2.1.0

- Preserve the input eltype in the returned results.

## v2.0.0

- **Breaking**: export `logmeanexp`, `logvarexp` and `logstdexp`, which
  previously had to be qualified or imported explicitly.

## v1.0.2

- Remove the docs build, TagBot and CompatHelper workflows.

## v1.0.1

- Support Julia v1.8.

## v1.0.0

- Register at https://github.com/cossio/CossioJuliaRegistry.
