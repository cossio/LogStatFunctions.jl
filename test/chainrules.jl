using Test: @test, @testset
using ChainRulesTestUtils: test_frule, test_rrule
using ChainRulesCore: NoTangent, frule, rrule, unthunk
using LogStatFunctions: logmeanexp, logvarexp, logstdexp

@testset "chainrules" begin
    for x in (randn(10), randn(10, 8)), dims in (:, 1, 1:2, 2)
        dims isa Colon || all(d ≤ ndims(x) for d in dims) || continue
        test_frule(logmeanexp, x; fkwargs = (; dims))
        test_rrule(logmeanexp, x; fkwargs = (; dims))
        for corrected in (true, false)
            test_frule(logvarexp, x; fkwargs = (; dims, corrected))
            test_rrule(logvarexp, x; fkwargs = (; dims, corrected))
            test_frule(logstdexp, x; fkwargs = (; dims, corrected))
            test_rrule(logstdexp, x; fkwargs = (; dims, corrected))
        end
    end
end

@testset "chainrules extreme values" begin
    # Nearly equal entries: the primal and its gradient are representable, but squaring
    # exp(xᵢ) - m outside the log domain underflows and used to yield Inf/NaN gradients.
    x = [-1.0e-200, 1.0e-200]
    for (f, h) in ((logvarexp, 1), (logstdexp, 2))
        Ω, pb = rrule(f, x)
        @test isfinite(Ω)
        x̄ = unthunk(pb(1.0)[2])
        @test collect(x̄) ≈ [-1.0e200, 1.0e200] ./ h
        Ω, ΔΩ = frule((NoTangent(), [1.0, 0.0]), f, x)
        @test isfinite(Ω)
        @test ΔΩ ≈ -1.0e200 / h
        @test frule((NoTangent(), [0.0, 1.0]), f, x)[2] ≈ 1.0e200 / h
    end
end

@testset "chainrules abstract eltype" begin
    x = Real[0.0, 1.0]
    for f in (logmeanexp, logvarexp, logstdexp)
        @test unthunk(rrule(f, x)[2](1.0)[2]) ≈ unthunk(rrule(f, [0.0, 1.0])[2](1.0)[2])
        @test frule((NoTangent(), [1.0, 0.0]), f, x)[2] ≈
            frule((NoTangent(), [1.0, 0.0]), f, [0.0, 1.0])[2]
    end
end
