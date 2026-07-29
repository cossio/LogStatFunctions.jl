import Documenter
import LogStatFunctions

Documenter.makedocs(
    modules = [LogStatFunctions],
    sitename = "LogStatFunctions.jl",
    repo = Documenter.Remotes.GitHub("cossio", "LogStatFunctions.jl"),
    pages = ["Home" => "index.md"]
)

Documenter.deploydocs(
    repo = "github.com/cossio/LogStatFunctions.jl.git",
    devbranch = "main"
)
