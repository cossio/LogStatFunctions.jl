# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small Julia package providing numerically stable `log`-of-statistic-of-`exp` functions (`logmeanexp`, `logvarexp`, `logstdexp`) that avoid forming `exp.(A)` and thus avoid overflow. It is a companion to LogExpFunctions.jl and builds everything on top of its `logsumexp` and `logsubexp`.

The entire implementation lives in `src/LogStatFunctions.jl` (a single module, ~60 lines).

## Workspace layout

The root `Project.toml` declares a Pkg workspace (`[workspace] projects = ["test", "docs"]`), so `test/` and `docs/` are workspace members sharing the root `Manifest.toml` (gitignored). Do not create `test/Manifest.toml` or `docs/Manifest.toml`. Sub-projects list `LogStatFunctions` in their `[deps]` and it resolves to the local package via the workspace — no `Pkg.develop` needed.

Workspaces need Julia 1.12+. On 1.10/1.11 the `[workspace]` table is ignored, so `--project=test` and `--project=docs` have no manifest of their own and cannot see the local package — activating them directly fails. `Pkg.test()` from the root still works on those versions, because it resolves the test target fresh.

`test/runtests.jl` only includes the other test files, each wrapped in a bare module (`functions.jl` for the functional tests, `aqua.jl` for Aqua.jl quality checks, `explicit_imports.jl` for ExplicitImports.jl checks). Add new test files following that pattern.

## Commands

Run the full test suite from the repo root:

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

Run a single test file directly (faster on repeat runs; uses the shared workspace manifest, so this one needs Julia 1.12+):

```sh
julia --project=test test/functions.jl
```

Build the docs (output in `docs/build/`, gitignored; also Julia 1.12+):

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'   # first time
julia --project=docs docs/make.jl
```

Check formatting with Runic (CI enforces Runic 1.7 on the whole repo):

```sh
julia --project=<env-with-Runic> -m Runic --check --diff .   # drop --check to apply fixes
```

## CI

Four workflows in `.github/workflows/`: `ci.yml` (tests on latest stable Julia, uploads coverage to Codecov — thresholds in `codecov.yml` — plus a `downgrade` job that tests the oldest supported Julia against the compat lower bounds of the deps), `format.yml` (Runic check), `docs.yml` (Documenter build + deploy to GitHub Pages), `TagBot.yml` (tags and GitHub releases after registry merges).

## Conventions and constraints

- Minimum supported Julia is 1.10 (see `[compat]` in Project.toml). The `[workspace]` table is ignored by Julia < 1.12 but `Pkg.test` still works there, since it resolves the test target fresh.
- All three functions follow the same shape: accept `dims = :` (whole-array reduction returns a scalar-like result via `logsumexp`), compute `N` as `length(A) ÷ length(R)` (the reduction factor), and apply the `log(N)` correction through the `sub!`/`halve!` helpers, which mutate the freshly allocated `logsumexp` output when it is mutable and fall back to allocating for immutable arrays (e.g. `StaticArrays`) and scalars. `logvarexp`/`logstdexp` additionally take `corrected::Bool` (Bessel correction, matching `Statistics.var`/`std`) and `logmean` to reuse a precomputed `logmeanexp`. Keep new functions consistent with this keyword API, and keep docstrings in the `Computes `log.(f(exp.(A); dims))`` style.
- Public functions need docstrings AND an entry in the `@docs` block of `docs/src/index.md` — Documenter fails the build on docstrings missing from the manual. ExplicitImports enforces that every import is explicit (`using X: y`) and public in its source module; Aqua's checks (ambiguities, stale deps, compat bounds…) also run as part of the test suite.
- Tests verify against the naive `log.(f(exp.(A)))` on small random arrays across all `dims` combinations, plus eltype preservation, complex and static arrays, overflow safety, and mathematical property tests (Jensen's inequality, singleton-dimension identity). Follow that pattern when adding functionality.

## Registering a new version

`.claude/skills/register-new-version/` documents the release procedure (ColPrac version choice, Registrator comment, TagBot). Note that the package is currently registered in https://github.com/cossio/CossioJuliaRegistry, not in the General registry.
