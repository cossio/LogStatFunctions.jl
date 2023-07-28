using LogStatFunctions
using Documenter

DocMeta.setdocmeta!(LogStatFunctions, :DocTestSetup, :(using LogStatFunctions); recursive=true)

makedocs(;
    modules=[LogStatFunctions],
    authors="JFdCD <j.cossio.diaz@gmail.com>",
    repo="https://github.com/cossio/LogStatFunctions.jl/blob/{commit}{path}#{line}",
    sitename="LogStatFunctions.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
