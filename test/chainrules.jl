using Test: @testset
using ChainRulesTestUtils: test_frule, test_rrule
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
