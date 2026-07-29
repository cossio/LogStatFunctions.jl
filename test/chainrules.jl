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

@testset "chainrules low-precision large reduction" begin
    # In Float16 the sum of 4095 expm1(-20) ≈ -1 terms rounds to -4096 == -n, which used to
    # drive the centered log-mean to -Inf and the gradients to NaN. The max entry dominates,
    # so its logvarexp gradient is ≈ 2 (and half that for logstdexp).
    for m in (4095, 65503)
        x = Float16[0; fill(Float16(-20), m)]
        for (f, h) in ((logvarexp, 1), (logstdexp, 2))
            Ω, pb = rrule(f, x)
            @test isfinite(Ω)
            x̄ = unthunk(pb(one(Float16))[2])
            @test all(isfinite, x̄)
            @test x̄[1] ≈ 2 / h rtol = 0.05
            Ω, ΔΩ = frule((NoTangent(), [1; zeros(m)]), f, x)
            @test isfinite(Ω)
            @test ΔΩ ≈ 2 / h rtol = 0.05
        end
    end
    # Individually tiny but collectively significant tail: expm1(-8.5) rounds to -1 in
    # Float16, yet the 4095 tail entries contribute most of the mean. The tail gradients
    # are ≈ -1e-7 (subnormal in Float16, hence the loose tolerance).
    x = Float16[0; fill(Float16(-8.5), 4095)]
    x̄ = unthunk(rrule(logvarexp, x)[2](one(Float16))[2])
    @test all(isfinite, x̄)
    @test x̄[1] ≈ 2 rtol = 0.05
    @test x̄[2] ≈ -9.9e-8 rtol = 0.5
    # Nearly adjacent subnormals: the centered offsets (±2^-25) sit below the Float16
    # subnormal spacing, but the gradients (±2^14, and half that for logstdexp) are
    # exactly representable.
    x = Float16[zeros(2048); fill(nextfloat(Float16(0)), 2048)]
    for (f, h) in ((logvarexp, 1), (logstdexp, 2))
        x̄ = unthunk(rrule(f, x)[2](one(Float16))[2])
        @test all(isfinite, x̄)
        @test x̄[1] ≈ -16384 / h rtol = 0.05
        @test x̄[end] ≈ 16384 / h rtol = 0.05
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

@testset "chainrules large common offset" begin
    # The gradients are translation-invariant. The spread is a multiple of ulp(c) for every
    # offset c below, so x .+ c is exact and the gradients must agree to machine precision.
    x = collect((1:10) .* 0.125)
    for f in (logmeanexp, logvarexp, logstdexp)
        g = unthunk(rrule(f, x)[2](1.0)[2])
        Δx = [1.0; zeros(9)]
        ΔΩ = frule((NoTangent(), Δx), f, x)[2]
        for c in (1.0e12, 1.0e15)
            @test unthunk(rrule(f, x .+ c)[2](1.0)[2]) ≈ g rtol = 1.0e-10
            @test frule((NoTangent(), Δx), f, x .+ c)[2] ≈ ΔΩ rtol = 1.0e-10
        end
    end
end
