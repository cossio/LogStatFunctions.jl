import ExplicitImports
import LogStatFunctions
using Test: @testset

@testset "ExplicitImports" begin
    ExplicitImports.test_explicit_imports(LogStatFunctions)
end
