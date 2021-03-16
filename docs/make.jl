using KeyedDistributions
using Documenter

DocMeta.setdocmeta!(KeyedDistributions, :DocTestSetup, :(using KeyedDistributions); recursive=true)

makedocs(;
    modules=[KeyedDistributions],
    authors="Invenia Technical Computing Corporation",
    repo="https://github.com/invenia/KeyedDistributions.jl/blob/{commit}{path}#{line}",
    sitename="KeyedDistributions.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://invenia.github.io/KeyedDistributions.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
    checkdocs=:exports,
    strict=true,
)

deploydocs(;
    repo="github.com/invenia/KeyedDistributions.jl",
    devbranch = "main",
    push_preview = true,
)
