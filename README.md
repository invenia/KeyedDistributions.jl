# KeyedDistributions

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://invenia.github.io/KeyedDistributions.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://invenia.github.io/KeyedDistributions.jl/dev)
[![Build Status](https://github.com/invenia/KeyedDistributions.jl/workflows/CI/badge.svg)](https://github.com/invenia/KeyedDistributions.jl/actions)
[![Coverage](https://codecov.io/gh/invenia/KeyedDistributions.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/invenia/KeyedDistributions.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

KeyedDistributions.jl provides thin wrappers of [`Distribution`](https://juliastats.org/Distributions.jl/latest/types/#Distributions) and [`Sampleable`](https://juliastats.org/Distributions.jl/latest/types/#Sampleable), to store keys for the variates.

```julia
julia> using KeyedDistributions, Distributions;

julia> kd = KeyedDistribution(MvNormal(3, 1.0), [:x, :y, :z]);
```

Methods for `Distribution` and `Sampleable` return [`KeyedArray`](https://github.com/mcabbott/AxisKeys.jl)s in place of regular `Array`s, where applicable.

```julia
julia> mean(kd)
1-dimensional KeyedArray(...) with keys:
↓   3-element Vector{Symbol}
And data, 3-element Zeros{Float64}:
 (:x)  0.0
 (:y)  0.0
 (:z)  0.0
 
julia> rand(kd, 2)
2-dimensional KeyedArray(...) with keys:
↓   3-element Vector{Symbol}
→   2-element OneTo{Int}
And data, 3×2 Matrix{Float64}:
        (1)           (2)
  (:x)   -1.11227      -0.279841
  (:y)    0.00784496    0.871718
  (:z)   -0.930186     -0.8922
```

In this way, KeyedDistributions.jl extends the [AxisKeys](https://github.com/mcabbott/AxisKeys.jl) ecosystem to [Distributions.jl](https://github.com/JuliaStats/Distributions.jl).
