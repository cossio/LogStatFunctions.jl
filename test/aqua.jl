import Aqua
import LogStatFunctions
using Test: @testset

@testset "aqua" begin
    Aqua.test_all(LogStatFunctions; ambiguities = false)
end
