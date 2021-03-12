module KeyedDistributions

using AutoHashEquals
using AxisKeys
using Distributions
using Random: AbstractRNG

export KeyedDistribution, KeyedSampleable
export axiskeys, distribution


# Constructors

for T in (:Distribution, :Sampleable)
    KeyedT = Symbol(:Keyed, T)
    @eval begin
        """
            $($KeyedT)(d<:$($T), keys::Tuple{Vararg{AbstractVector}})

        Stores `keys` for each variate alongside the [`$($T)`](@ref) `d`,
        supporting all of the common functions of a [`$($T)`](@ref).
        Common functions that return an [`AbstractArray`](@ref), such as [`rand`](@ref),
        will return a [`KeyedArray`](@ref) with keys derived from the `$($T)`.

        The type of `keys` is restricted to be consistent with
        [AxisKeys.jl](https://github.com/mcabbott/AxisKeys.jl).
        The length of the `keys` tuple must be the number of dimensions, which is 1 for
        univariate and multivariate distributions, and 2 for matrix-variate distributions.
        The length of each key vector in must match the length along each dimension.
        """
        @auto_hash_equals struct $KeyedT{F<:VariateForm, S<:ValueSupport, D<:$T{F, S}} <: $T{F, S}
            d::D
            keys::Tuple{Vararg{AbstractVector}}

            function $KeyedT(d::$T{F, S}, keys::Tuple{Vararg{AbstractVector}}) where {F, S}
                length(d) == prod(length, keys) || throw(ArgumentError(
                    "number of keys ($(prod(length, keys))) must match " *
                    "number of variates ($(length(d)))"))

                if F == Matrixvariate
                    lengths = map(v -> length(v), keys)
                    lengths == size(d) || throw(ArgumentError(
                        "lengths of key vectors $(lengths) must match " *
                        "size of distribution $(size(d))"))
                end

                return new{F, S, typeof(d)}(d, keys)
            end
        end

        """
            $($KeyedT)(d<:$($T), keys::AbstractVector)

        Constructor for [`$($KeyedT)`](@ref) with one dimension of variates.
        The elements of `keys` correspond to the variates of the distribution.
        """
        $KeyedT(d::$T{F, S}, keys::AbstractVector) where {F, S} = $KeyedT(d, (keys, ))
    end
end

"""
    KeyedDistribution(d::Distribution)

Constructs a [`KeyedDistribution`](@ref) using the keys of the first field stored in `d`,
or if there are no keys, `1:n` for the length `n` of each dimension.
"""
function KeyedDistribution(d::Distribution)
    first_field = getfield(d, 1)
    return KeyedDistribution(d, _keys(first_field))
end

_keys(x::KeyedArray) = axiskeys(x)
_keys(x) = map(Base.OneTo, size(x))

const KeyedDistOrSampleable = Union{KeyedDistribution, KeyedSampleable}

# Access methods

"""
    distribution(::KeyedDistribution) -> Distribution
    distribution(::KeyedSampleable{F, S, D}) -> D

Return the wrapped distribution.
"""
distribution(d::KeyedDistOrSampleable) = d.d

# AxisKeys functionality

"""
    axiskeys(s::Sampleable)

Return the keys for the variates of the `Sampleable`.
For a [`KeyedDistribution`](@ref) or [`KeyedSampleable`](@ref)
this is the keys it was constructed with.
For any other `Sampleable` this is equal to `1:length(s)`.
"""
AxisKeys.axiskeys(d::KeyedDistOrSampleable) = d.keys

# Standard functions to overload for new Distribution and/or Sampleable
# https://juliastats.org/Distributions.jl/latest/extends/#Create-New-Samplers-and-Distributions

function Distributions._rand!(
    rng::AbstractRNG,
    d::KeyedDistOrSampleable,
    x::AbstractVector{<:Real}
)
    sample = Distributions._rand!(rng, distribution(d), x)
    return KeyedArray(sample, axiskeys(d))
end

Base.length(d::KeyedDistOrSampleable) = length(distribution(d))

function Distributions.size(d::KeyedDistribution{F}) where F<:Matrixvariate
    return size(distribution(d))
end

Distributions.sampler(d::KeyedDistribution) = sampler(distribution(d))

Base.eltype(d::KeyedDistribution) = eltype(distribution(d))

function Distributions._logpdf(d::KeyedDistribution, x::AbstractArray)
    # Does not support KeyedArray as parameter of Distribution
    # https://github.com/mcabbott/AxisKeys.jl/issues/54
    return Distributions._logpdf(distribution(d), x)
end

# Also need to overload `rand` methods to return KeyedArrays:

function Base.rand(rng::AbstractRNG, d::KeyedDistOrSampleable)
    sample = rand(rng, distribution(d))
    ndims(sample) == 0 && return sample  # univariate returns a Number
    return KeyedArray(sample, axiskeys(d))
end

function Base.rand(rng::AbstractRNG, d::KeyedDistOrSampleable, n::Int)
    samples = rand(rng, distribution(d), n)
    ndims(samples) == 1 && return KeyedArray(samples, Base.OneTo(n))  # univariate
    return KeyedArray(samples, (first(axiskeys(d)), Base.OneTo(n)))
end

function Base.rand(rng::AbstractRNG, d::KeyedDistribution{F}, n::Int) where F<:Matrixvariate
    # Distributions.rand returns a vector of matrices
    samples = [KeyedArray(x, axiskeys(d)) for x in rand(rng, distribution(d), n)]
    return KeyedArray(samples, Base.OneTo(n))
end

# Statistics functions for Distribution

Distributions.mean(d::KeyedDistribution) = KeyedArray(mean(distribution(d)), axiskeys(d))

Distributions.var(d::KeyedDistribution) = KeyedArray(var(distribution(d)), axiskeys(d))

function Distributions.cov(d::KeyedDistribution)
    keys = vcat(axiskeys(d)...)
    return KeyedArray(cov(distribution(d)), (keys, keys))
end

Distributions.entropy(d::KeyedDistribution) = entropy(distribution(d))
Distributions.entropy(d::KeyedDistribution, b::Real) = entropy(distribution(d), b)

end
