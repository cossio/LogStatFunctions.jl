# Tests GPU compatibility without a physical GPU, using JLArrays (the reference
# GPUArrays.jl backend). With allowscalar(false), any code path falling back to
# scalar indexing errors out, just like CuArray on CI. The setting is session-global,
# which is intentional: all JLArrays tests live in this file, and any future test
# using JLArrays should run under allowscalar(false) too.
using Test: @test, @testset
using JLArrays: JLArray, JLArrays
using LogStatFunctions: logmeanexp, logvarexp, logstdexp
using LogStatFunctions: logmeanexp!, logvarexp!, logstdexp!

JLArrays.allowscalar(false)

@testset "logmeanexp, logvarexp, logstdexp" begin
    A = randn(11, 7, 5)
    jl_A = JLArray(A)
    for dims in (1, 2, 3, (1, 2), (2, 3), (1, 3), (1, 2, 3))
        jl_R = logmeanexp(jl_A; dims)
        @test jl_R isa JLArray
        @test Array(jl_R) ≈ logmeanexp(A; dims)
        for corrected in (true, false)
            jl_R = logvarexp(jl_A; dims, corrected)
            @test jl_R isa JLArray
            @test Array(jl_R) ≈ logvarexp(A; dims, corrected)
            jl_R = logstdexp(jl_A; dims, corrected)
            @test jl_R isa JLArray
            @test Array(jl_R) ≈ logstdexp(A; dims, corrected)
        end
    end
    # whole-array reduction returns a scalar (this is not scalar indexing)
    @test logmeanexp(jl_A) ≈ logmeanexp(A)
    @test logvarexp(jl_A) ≈ logvarexp(A)
    @test logstdexp(jl_A) ≈ logstdexp(A)
end

@testset "logmeanexp!, logvarexp!, logstdexp!" begin
    A = randn(11, 7, 5)
    jl_A = JLArray(A)
    for dims in (1, 2, 3, (1, 2), (2, 3), (1, 3), (1, 2, 3))
        jl_out = JLArray(similar(logmeanexp(A; dims)))
        @test logmeanexp!(jl_out, jl_A) === jl_out
        @test Array(jl_out) ≈ logmeanexp(A; dims)
        for corrected in (true, false)
            jl_out = JLArray(similar(logvarexp(A; dims)))
            @test logvarexp!(jl_out, jl_A; corrected) === jl_out
            @test Array(jl_out) ≈ logvarexp(A; dims, corrected)
            jl_out = JLArray(similar(logstdexp(A; dims)))
            @test logstdexp!(jl_out, jl_A; corrected) === jl_out
            @test Array(jl_out) ≈ logstdexp(A; dims, corrected)
        end
    end
end

@testset "logmean argument" begin
    A = randn(11, 7)
    jl_A = JLArray(A)
    for dims in (1, 2)
        jl_logmean = logmeanexp(jl_A; dims)
        @test Array(logvarexp(jl_A; dims, logmean = jl_logmean)) ≈ logvarexp(A; dims)
        @test Array(logstdexp(jl_A; dims, logmean = jl_logmean)) ≈ logstdexp(A; dims)
        jl_out = JLArray(similar(logvarexp(A; dims)))
        @test Array(logvarexp!(jl_out, jl_A; logmean = jl_logmean)) ≈ logvarexp(A; dims)
        @test Array(logstdexp!(jl_out, jl_A; logmean = jl_logmean)) ≈ logstdexp(A; dims)
    end
    logmean = logmeanexp(jl_A)
    @test logvarexp(jl_A; logmean) ≈ logvarexp(A)
    @test logstdexp(jl_A; logmean) ≈ logstdexp(A)
end

@testset "eltype preservation" begin
    A = randn(Float32, 11, 7)
    jl_A = JLArray(A)
    @test logmeanexp(jl_A) isa Float32
    @test logvarexp(jl_A) isa Float32
    @test logstdexp(jl_A) isa Float32
    for dims in (1, 2, (1, 2))
        @test eltype(logmeanexp(jl_A; dims)) === Float32
        for corrected in (true, false)
            @test eltype(logvarexp(jl_A; dims, corrected)) === Float32
            @test eltype(logstdexp(jl_A; dims, corrected)) === Float32
        end
    end
end

@testset "no overflow" begin
    # exp.(A) overflows Float64, but the log-domain computation stays finite
    jl_A = JLArray(1000 .* randn(1000))
    @test isfinite(logmeanexp(jl_A))
    @test isfinite(logvarexp(jl_A))
    @test isfinite(logstdexp(jl_A))
    jl_B = JLArray(1000 .* randn(1000, 3))
    @test all(isfinite, Array(logmeanexp(jl_B; dims = 1)))
    @test all(isfinite, Array(logvarexp(jl_B; dims = 1)))
    @test all(isfinite, Array(logstdexp(jl_B; dims = 1)))
end
