module KeyedDistributions

using AutoHashEquals
using AxisKeys
using AxisKeys: keyless
using Distributions
using Random: AbstractRNG

export KeyedDistribution, KeyedSampleable
export axiskeys, distribution


for T in (:Distribution, :Sampleable)
    KeyedT = Symbol(:Keyed, T)
    @eval begin
        @auto_hash_equals struct $KeyedT{F, S, D<:$T{F, S}} <: $T{F, S}
            d::D
            keys::AbstractVector

            """
            $($KeyedT)(d<:$($T), keys::AbstractVector)

            Stores `keys` for each variate alongside the `$($T)` `d`.
            """
            function $KeyedT(d::$T{F, S}, keys::AbstractVector) where {F, S}
                length(d) == length(keys) || throw(DimensionMismatch(
                        "number of keys ($(length(keys))) must match " *
                        "number of variates ($(length(d)))"
                    ))
                return new{F, S, typeof(d)}(d, keys)
            end
        end
    end
end

"""
    KeyedDistribution(d::Distribution)

Constructs a [`KeyedDistribution`](@ref) using keys stored in `d`.
The keys are copied from the first axis of the first parameter in `d`.
"""
function KeyedDistribution(d::D) where D <: Distribution
    first_param = getfield(d, 1)
    keys = first(axiskeys(first_param))  # axiskeys guaranteed to be Tuple{AbstractVector}?
    return KeyedDistribution(d, keys)
end

const KeyedDistOrSampleable = Union{KeyedDistribution, KeyedSampleable}

# Access methods

"""
    distribution(::KeyedDistribution) -> Distribution
    distribution(::KeyedSampleable{F, S, D}) -> D

Return the wrapped distribution.
"""
distribution(d::KeyedDistOrSampleable) = d.d

# AxisKeys functionality

Base.parent(d::KeyedDistOrSampleable) = d.d

AxisKeys.keyless(d::KeyedDistOrSampleable) = parent(d)

"""
    axiskeys(s::Sampleable)

Return the keys for the variates of the Sampleable.
For an [`KeyedDistribution`](@ref) or [`KeyedSampleable`](@ref) this
is the keys it was constructed with.
For any other `Sampleable` this is equal to `1:length(s)`.
"""
AxisKeys.axiskeys(d::KeyedDistOrSampleable) = tuple(d.keys)
AxisKeys.axiskeys(d::Sampleable) = tuple(Base.OneTo(length(d)))

# Standard functions to overload for new Distribution and/or Sampleable
# https://juliastats.org/Distributions.jl/latest/extends/#Create-New-Samplers-and-Distributions

function Distributions._rand!(
    rng::AbstractRNG,
    d::KeyedDistOrSampleable,
    x::AbstractVector{T}
) where T<:Real
    sample = Distributions._rand!(rng, parent(d), x)
    return KeyedArray(sample, axiskeys(d))
end

Base.length(d::KeyedDistOrSampleable) = length(keyless(d))

Distributions.sampler(d::KeyedDistribution) = sampler(keyless(d))

Base.eltype(d::KeyedDistribution) = eltype(keyless(d))

Distributions._logpdf(d::KeyedDistribution, x::AbstractArray) =
    Distributions._logpdf(parent(d), x)

# Also need to overload `rand` methods to return a KeyedArray

Base.rand(rng::AbstractRNG, d::KeyedDistOrSampleable) =
    KeyedArray(rand(rng, parent(d)), axiskeys(d))

Base.rand(rng::AbstractRNG, d::KeyedDistOrSampleable, n::Int) =
    KeyedArray(rand(rng, parent(d), n), (first(axiskeys(d)), Base.OneTo(n)))

# Statistics functions for Distribution

Distributions.mean(d::KeyedDistribution) = KeyedArray(mean(keyless(d)), axiskeys(d))

Distributions.var(d::KeyedDistribution) = KeyedArray(var(keyless(d)), axiskeys(d))

Distributions.cov(d::KeyedDistribution) =
    KeyedArray(cov(keyless(d)), (first(axiskeys(d)), first(axiskeys(d))))

Distributions.entropy(d::KeyedDistribution) = entropy(keyless(d))
Distributions.entropy(d::KeyedDistribution, b::Real) = entropy(keyless(d), b)

end
