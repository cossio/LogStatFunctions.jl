using Test: @test, @testset, @inferred
using Statistics: mean, var, std
using LogStatFunctions: logmeanexp, logvarexp, logstdexp

@testset "logmeanexp, logvarexp" begin
    A = randn(5,3,2)
    for dims in (2, (1,2), :)
        @test logmeanexp(A; dims) ≈ log.(mean(exp.(A); dims))
        for corrected in (true, false)
            @test logvarexp(A; dims, corrected) ≈ log.(var(exp.(A); dims, corrected))
            @test logstdexp(A; dims, corrected) ≈ log.(std(exp.(A); dims, corrected))
        end
    end
    @test logvarexp(A; dims=2) ≈ log.(var(exp.(A); dims=2))
    @test logstdexp(A; dims=2) ≈ log.(std(exp.(A); dims=2))
    @test logmeanexp(A) ≈ log.(mean(exp.(A)))
    @test logvarexp(A) ≈ log.(var(exp.(A)))
    @test logstdexp(A) ≈ log.(std(exp.(A)))
end

@testset "logmeanexp properties" begin
    X = randn(1000, 1000)
    @test only(logmeanexp(logmeanexp(X; dims=1); dims=2)) ≈ logmeanexp(X)
    @test only(logmeanexp(-logmeanexp(-X; dims=1); dims=2)) ≤ only(-logmeanexp(-logmeanexp(X; dims=1); dims=2))
    x = randn()
    @test logmeanexp([x]) ≈ x
    X = randn(1000, 1000, 1)
    @test logmeanexp(X; dims=3) ≈ X
    @test -logmeanexp(-X) ≤ logmeanexp(X)
end
