using Test: @test, @testset, @inferred
using Statistics: mean, var, std
using StaticArrays: SA
using LogStatFunctions: logmeanexp, logvarexp, logstdexp

@testset "logmeanexp, logvarexp, logstdexp" begin
    A = randn(11, 7, 5)
    for dims in (1, 2, 3, (1, 2), (2, 3), (1, 3), (1, 2, 3), :)
        @test logmeanexp(A; dims) ≈ log.(mean(exp.(A); dims))
        for corrected in (true, false)
            @test logvarexp(A; dims, corrected) ≈ log.(var(exp.(A); dims, corrected))
            @test logstdexp(A; dims, corrected) ≈ log.(std(exp.(A); dims, corrected))
        end
    end
    @test logvarexp(A; dims = 2) ≈ log.(var(exp.(A); dims = 2))
    @test logstdexp(A; dims = 2) ≈ log.(std(exp.(A); dims = 2))
    @test logmeanexp(A) ≈ log.(mean(exp.(A)))
    @test logvarexp(A) ≈ log.(var(exp.(A)))
    @test logstdexp(A) ≈ log.(std(exp.(A)))
end

@testset "no overflow" begin
    # exp.(A) overflows Float64, but the log-domain computation stays finite
    A = 1000 .* randn(1000)
    @test isfinite(logmeanexp(A))
    @test isfinite(logvarexp(A))
    @test isfinite(logstdexp(A))
    # shifting by a constant shifts the log-mean by the same constant
    B = randn(1000)
    @test logmeanexp(B .+ 1000) ≈ logmeanexp(B) + 1000
end

@testset "eltype preservation" begin
    A = randn(Float32, 11, 7)
    @test @inferred(logmeanexp(A)) isa Float32
    @test @inferred(logvarexp(A)) isa Float32
    @test @inferred(logstdexp(A)) isa Float32
    for dims in (1, 2, (1, 2))
        @test eltype(logmeanexp(A; dims)) === Float32
        for corrected in (true, false)
            @test eltype(logvarexp(A; dims, corrected)) === Float32
            @test eltype(logstdexp(A; dims, corrected)) === Float32
        end
    end
end

@testset "complex arrays" begin
    A = randn(ComplexF64, 5)
    @test logmeanexp(A) ≈ log(mean(exp.(A)))
    @test logmeanexp(A) isa ComplexF64
    @test logmeanexp(randn(ComplexF32, 4)) isa ComplexF32
end

@testset "immutable (static) arrays" begin
    S = SA[1.0 2.0; 3.0 4.0]
    @test logmeanexp(S) ≈ log(mean(exp.(S)))
    @test logmeanexp(S; dims = 1) ≈ log.(mean(exp.(S); dims = 1))
    for corrected in (true, false)
        @test logvarexp(S; dims = 1, corrected) ≈ log.(var(exp.(S); dims = 1, corrected))
        @test logstdexp(S; dims = 1, corrected) ≈ log.(std(exp.(S); dims = 1, corrected))
    end
end

@testset "logmean argument" begin
    A = randn(11, 7)
    for dims in (1, 2, :)
        logmean = logmeanexp(A; dims)
        @test logvarexp(A; dims, logmean) ≈ logvarexp(A; dims)
        @test logstdexp(A; dims, logmean) ≈ logstdexp(A; dims)
    end
end

@testset "logmeanexp properties" begin
    X = randn(5, 3)
    # associativity of the mean across dimensions
    @test only(logmeanexp(logmeanexp(X; dims = 1); dims = 2)) ≈ logmeanexp(X)
    # AM–GM / Jensen: -logmeanexp(-x) ≤ logmeanexp(x)
    @test only(logmeanexp(-logmeanexp(-X; dims = 1); dims = 2)) ≤ only(-logmeanexp(-logmeanexp(X; dims = 1); dims = 2))
    x = randn()
    @test logmeanexp([x]) ≈ x
    # mean over a singleton dimension is the identity
    X = randn(5, 3, 1)
    @test logmeanexp(X; dims = 3) ≈ X
    @test -logmeanexp(-X) ≤ logmeanexp(X)
end
