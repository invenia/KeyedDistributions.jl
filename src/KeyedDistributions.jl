module KeyedDistributions

using AutoHashEquals
using AxisKeys
using Distributions
using Distributions: GenericMvTDist
using IterTools
using LinearAlgebra: Symmetric
using PDMatsExtras: submat
using Random: AbstractRNG

export KeyedDistribution, KeyedSampleable
export KeyedMvNormal, KeyedGenericMvTDist, MvNormalLike, MvTLike
export axiskeys, distribution

for T in (:Distribution, :Sampleable)
    KeyedT = Symbol(:Keyed, T)
    @eval begin
        """
            $($KeyedT)(d<:$($T), keys::Tuple{Vararg{AbstractVector}})

        Stores `keys` for each variate alongside the `$($T)` `d`,
        supporting all of the common functions of a `$($T)`.
        Common functions that return an `AbstractArray`, such as `rand`,
        will return a `KeyedArray` with keys derived from the `$($T)`.

        The type of `keys` is restricted to be consistent with
        [AxisKeys.jl](https://github.com/mcabbott/AxisKeys.jl).
        The length of the `keys` tuple must be the number of dimensions, which is 1 for
        univariate and multivariate distributions, and 2 for matrix-variate distributions.
        The length of each key vector in must match the length along each dimension.

        !!! Note
            For distributions that can be marginalized exactly, the $($KeyedT)) can be
            marginalised via the indexing or lookup syntax just like `KeyedArray`s.
            i.e. One can use square or round brackets to retain certain indices or keys and
            marginalise out the others. For example for `D::KeyedMvNormal` over `:a, :b, :c`:
             - `D(:a)` or D(1) will marginalise out `:b, :c` and return a `KeyedMvNormal`
               over `:a`.
             - `D([:a, :b])` or `D[[1, 2]]` will marginalise out `:c` and return a
               `KeyedMvNormal` over `:a, :b`.
        """
        @auto_hash_equals struct $KeyedT{F<:VariateForm, S<:ValueSupport, D<:$T{F, S}} <: $T{F, S}
            d::D
            keys::Tuple{Vararg{AbstractVector}}

            function $KeyedT(d::$T{F, S}, keys::Tuple{Vararg{AbstractVector}}) where {F, S}
                key_lengths = map(length, keys)
                key_lengths == _size(d) || throw(ArgumentError(
                    "lengths of key vectors $key_lengths must match " *
                    "size of distribution $(_size(d))"
                ))

                return new{F, S, typeof(d)}(d, keys)
            end
        end

        """
            $($KeyedT)(d<:$($T), keys::AbstractVector)

        Constructor for [`$($KeyedT)`](@ref) with one dimension of variates.
        The elements of `keys` correspond to the variates of the distribution.
        """
        $KeyedT(d::$T{F, S}, keys::AbstractVector) where {F, S} = $KeyedT(d, (keys, ))

        # Allows marginalisation via lookup syntax, using getindex.
        (d::$KeyedT)(keys...) = d[first(map(AxisKeys.findindex, keys, axiskeys(d)))]
    end
end

_size(d) = (length(d),)
_size(d::Sampleable{<:Matrixvariate}) = size(d)

"""
    KeyedDistribution(d::Distribution)

Constructs a [`KeyedDistribution`](@ref) using the keys of the parameter that matches `size(d)`.
If the parameter has no keys, uses `1:n` for the length `n` of each dimension.
"""
KeyedDistribution(d::Distribution) = KeyedDistribution(d, _keys(d))

_keys(d::Distribution) = _keys(first(filter(x -> size(x) == size(d), params(d))))
_keys(x::KeyedArray) = axiskeys(x)
_keys(x) = map(Base.OneTo, size(x))

const KeyedDistOrSampleable = Union{KeyedDistribution, KeyedSampleable}

# MvNormal and MvT are particularly useful - these make it easier to dispatch
const KeyedMvNormal = KeyedDistribution{Multivariate, Continuous, <:MvNormal}
const MvNormalLike = Union{MvNormal, KeyedMvNormal}

const KeyedGenericMvTDist = KeyedDistribution{Multivariate, Continuous, <:GenericMvTDist}
const MvTLike = Union{GenericMvTDist, KeyedGenericMvTDist}

# Use submat to preserve the covariance matrix PDMat type
function Base.getindex(d::KeyedMvNormal, i::Vector)::KeyedMvNormal
    return KeyedDistribution(MvNormal(d.d.μ[i], submat(d.d.Σ, i)), axiskeys(d)[1][i])
end

function Base.getindex(d::KeyedMvNormal, i::Integer)::KeyedDistribution
    return KeyedDistribution(Normal(d.d.μ[i], d.d.Σ[i, i]), [axiskeys(d)[1][i]])
end

function Base.getindex(d::KeyedGenericMvTDist, i::Vector)::KeyedGenericMvTDist
    return KeyedDistribution(MvTDist(d.d.df, d.d.μ[i], submat(d.d.Σ, i)), axiskeys(d)[1][i])
end

# Access methods

"""
    distribution(::KeyedDistribution) -> Distribution
    distribution(::KeyedSampleable{F, S, D}) -> D

Return the wrapped distribution.
"""
distribution(d::KeyedDistOrSampleable) = d.d

# AxisKeys functionality

"""
    axiskeys(d::Union{KeyedDistribution, KeyedSampleable})

Return the keys for the variates of the `KeyedDistribution` or `KeyedSampleable`.
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

Distributions.size(d::KeyedDistribution{<:Matrixvariate}) = size(distribution(d))

Distributions.sampler(d::KeyedDistribution) = sampler(distribution(d))

Distributions.params(d::KeyedDistOrSampleable) = params(distribution(d))

Base.eltype(d::KeyedDistribution) = eltype(distribution(d))

function Distributions._logpdf(d::KeyedDistribution, x::AbstractArray)
    # Workaround when KeyedArray is parameter of Distribution
    # https://github.com/mcabbott/AxisKeys.jl/issues/54
    dist = distribution(d)
    T = typeof(dist)
    args = map(_maybe_parent, params(dist))
    unkeyed_dist = T.name.wrapper(args...)

    return Distributions._logpdf(unkeyed_dist, x)
end

_maybe_parent(x) = x
_maybe_parent(x::AbstractArray) = parent(x)

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

function Base.rand(rng::AbstractRNG, d::KeyedDistribution{<:Matrixvariate}, n::Int)
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

# Univariate Distributions only

for f in (:logpdf, :quantile, :mgf, :cf)
    @eval Distributions.$f(d::KeyedDistribution{<:Univariate}, x) = $f(distribution(d), x)
end

for f in (:minimum, :maximum, :modes, :mode, :skewness, :kurtosis)
    @eval Distributions.$f(d::KeyedDistribution{<:Univariate}) = $f(distribution(d))
end

# Needed to avoid method ambiguity
Distributions.cdf(d::KeyedDistribution{<:Univariate}, x::Real) = cdf(distribution(d), x)

function Distributions.insupport(d::KeyedDistribution{<:Univariate}, x::Real)
    return insupport(distribution(d), x)
end

end
