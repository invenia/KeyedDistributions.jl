using AxisKeys
using Distributions
using Distributions: GenericMvTDist
using KeyedDistributions
using LinearAlgebra
using NamedDims
using PDMats: PDMat
using PDMatsExtras: WoodburyPDMat, submat
using StableRNGs
using Statistics
using Test

@testset "KeyedDistributions.jl" begin
    @testset "Common" begin
        X = rand(StableRNG(1234), 10, 3)
        m = vec(mean(X; dims=1))
        s = cov(X; dims=1)
        d = MvNormal(m, s)
        keys = ([:a, :b, :c], )
        names = (:id, )
        named_keys = NamedTuple{names}(keys)

        @testset "subtyping" begin
            @test KeyedDistribution <: Distribution
            @test KeyedDistribution <: Sampleable
            @test <:(KeyedDistribution{F, S} where {F, S}, Distribution{F, S} where {F, S})
            @test <:(KeyedDistribution{F, S} where {F, S}, Sampleable{F, S} where {F, S})

            @test !(KeyedSampleable <: Distribution)
            @test KeyedSampleable <: Sampleable
            @test <:(KeyedSampleable{F, S} where {F, S}, Sampleable{F, S} where {F, S})

            @test KeyedDistribution(d, keys) isa Distribution{Multivariate}
            @test KeyedSampleable(d, keys) isa Sampleable{Multivariate}

            @test KeyedDistribution(d; named_keys...) isa Distribution{Multivariate}
            @test KeyedSampleable(d; named_keys...) isa Sampleable{Multivariate}

            # Input not a Distribution
            @test_throws MethodError KeyedDistribution(X, keys)

            @test MvNormal <: MvNormalLike
            @test KeyedMvNormal <: MvNormalLike
            @test GenericMvTDist <: MvTLike
            @test KeyedGenericMvTDist <: MvTLike
            @test MvNormalLike <: MultivariateDistribution
            @test MvTLike <: MultivariateDistribution

        end

        @testset "1-dimensional constructor" begin
            @test KeyedDistribution(d, id=keys[1]) isa KeyedDistribution
            @test KeyedSampleable(d, id=keys[1]) isa KeyedSampleable

            # Input not a Distribution
            @test_throws MethodError KeyedDistribution(X, keys[1])
        end

        @testset "$T" for T in (KeyedDistribution, KeyedSampleable)
            kd = T(d; named_keys...)

            @testset "base functions" begin
                @test kd isa Sampleable
                @test distribution(kd) == d
                @test axiskeys(kd) == keys
                @test dimnames(kd) == names
                @test length(kd) == length(d) == 3
                @test isequal(kd, T(d; id=[:a, :b, :c]))
                @test params(kd) == params(d) == (m, s)
                @test ==(kd, T(d; named_keys...))
            end

            @testset "sampling" begin
                # Samples from the distribution both wrapped and unwrapped are the same
                @test rand(StableRNG(1), d) == rand(StableRNG(1), kd)
                @test rand(StableRNG(1), d, 3) == rand(StableRNG(1), kd, 3)

                rng = StableRNG(1)

                @testset "_rand!" begin
                    expected = [0.4209843517343097, 0.40546557072031986, 0.5573596971895245]
                    observed = Distributions._rand!(rng, kd, zeros(Float64, 3))
                    @test observed isa KeyedArray
                    @test isapprox(observed, expected)
                    @test isapprox(observed(:a), expected[1])
                    @test axiskeys(observed) == axiskeys(kd)
                    @test dimnames(observed) == dimnames(kd)
                end

                @testset "one-sample method" begin
                    expected = [0.3308452209411066, -0.1078587452687392, 0.3177486760818843]
                    observed = rand(rng, kd)
                    @test observed isa KeyedArray
                    @test isapprox(observed, expected)
                    @test isapprox(observed(:a), expected[1])
                    @test axiskeys(observed) == axiskeys(kd)
                    @test dimnames(observed) == dimnames(kd)
                end

                @testset "multi-sample method" begin
                    expected = [
                        0.8744990592397803 0.6468838205306433;
                        0.5919869675852324 0.04861847876961256;
                        0.24068367188141987 0.3277215089605162
                    ]
                    observed = rand(rng, kd, 2)
                    @test observed isa KeyedArray
                    @test isapprox(observed, expected)
                    @test isapprox(observed(:a), expected[1, :])
                    @test axiskeys(observed) == (first(axiskeys(kd)), Base.OneTo(2))
                    @test dimnames(observed) == (first(dimnames(kd)), :sample)
                end
            end
        end
    end

    @testset "KeyedDistribution only" begin
        X = rand(StableRNG(1234), 10, 3)
        m = vec(mean(X; dims=1))
        s = cov(X; dims=1)
        keys = ([:a, :b, :c], )
        names = (:id,)
        named_keys = NamedTuple{names}(keys)

        _make_dist(::Type{MvNormal}, m, s) = MvNormal(m, s)
        _make_dist(::Type{GenericMvTDist}, m, s) = GenericMvTDist(3, m, PDMat(s))

        @testset "$D" for D in (MvNormal, GenericMvTDist)

            d = _make_dist(D, m, s)
            d_keyed = _make_dist(D, KeyedArray(m; named_keys...), s)

            kd_keys = KeyedDistribution(d; named_keys...)
            kd_no_keys = KeyedDistribution(d_keyed)
            kds = (keys=kd_keys, no_keys=kd_no_keys)

            @testset "$case" for case in (:keys, :no_keys)
                kd = kds[case]

                @testset "base functions" begin
                    @test kd isa Distribution
                    @test distribution(kd) == d
                    @test axiskeys(kd) == keys
                    @test dimnames(kd) == names
                    @test AxisKeys.haskeys(kd)
                    @test AxisKeys.keyless(kd) == d
                    @test AxisKeys.keyless_unname(kd) == d
                    @test sampler(kd) == sampler(d)
                    @test eltype(kd) == eltype(d) == Float64
                end

                @testset "statistical functions" begin
                    @test mean(kd) isa KeyedArray{Float64, 1}
                    @test parent(mean(kd)) == mean(d) == m

                    @test var(kd) isa KeyedArray{Float64, 1}
                    @test parent(var(kd)) == var(d)

                    @test cov(kd) isa KeyedArray{Float64, 2}
                    @test parent(cov(kd)) == cov(d)

                    @test cor(kd) isa KeyedArray{Float64, 2}
                    @test parent(cor(kd)) == cor(d)

                    @test entropy(kd) == entropy(d)
                    @test entropy(kd, 2) == entropy(d, 2)

                    @test Distributions._logpdf(kd, m) == Distributions._logpdf(d, m)

                    # statistical functions commute with accessor methods
                    for f in (mean, var, cov)
                        @test f(distribution(kd)) == parent(f(kd))
                    end
                end
            end
        end

        @testset "default keys and dimnames" begin
            kd = KeyedDistribution(MvNormal(m, s))
            @test axiskeys(kd) == (Base.OneTo(3), )
            @test dimnames(kd) == (:_, )
        end
    end

    @testset "Distributions types" begin
        @testset "Univariate" begin
            d = Normal(0.5, 0.2)
            kd = KeyedDistribution(d; id=["variate"])
            rng = StableRNG(1)

            @test kd isa UnivariateDistribution
            @test axiskeys(kd) == (["variate"], )
            @test dimnames(kd) == (:id, )
            @test length(kd) == length(d) == 1

            @test logpdf(kd, 1) == logpdf(d, 1) ≈ -2.4345006207705726
            @test cdf(kd, 1) == cdf(d, 1) ≈ 0.9937903346742238
            @test quantile(kd, 0.5) == quantile(d, 0.5) == 0.5
            @test insupport(kd, 1) == insupport(d, 1) == true
            @test mgf(kd, 1) == mgf(d, 1) ≈ 1.6820276496988864
            @test cf(kd, 0) == cf(d, 0) == 1.0 + 0.0im
            @test entropy(kd) == entropy(d) ≈ -0.1904993792294276
            @test entropy(kd, 2) == entropy(d, 2) ≈ -0.27483250970672124

            @test minimum(kd) == minimum(d) == -Inf
            @test maximum(kd) == maximum(d) == Inf
            @test modes(kd) == modes(d) == [0.5]
            @test mode(kd) == mode(d) == 0.5
            @test skewness(kd) == skewness(d) == 0.0
            @test kurtosis(kd) == kurtosis(d) == 0.0
            @test kurtosis(kd, false) == kurtosis(d, false) == 3.0

            @test isapprox(rand(rng, kd), 0.39349598502717537)
            @test isapprox(rand(rng, kd, 2), [0.519693102856957, 0.6505773044249047])
        end

        @testset "Matrix-variate" begin
            d = Wishart(7.0, Matrix(1.0I, 2, 2))
            md = KeyedDistribution(d; dim1=["x1", "x2"], dim2=["y1", "y2"])
            get_rng() = StableRNG(1)

            @test md isa MatrixDistribution
            @test length(md) == length(d) == 4
            @test size(md) == size(d) == (2, 2)
            @test axiskeys(md) == (["x1", "x2"], ["y1", "y2"])
            @test dimnames(md) == (:dim1, :dim2)

            @testset "sample" begin
                expected = rand(get_rng(), d)
                observed = rand(get_rng(), md)
                @test observed isa KeyedArray
                @test observed == expected
                @test isapprox(observed("x1", :), expected[1, :])
                @test axiskeys(observed) == axiskeys(md)
                @test dimnames(observed) == dimnames(md)
            end

            @testset "multi-sample" begin
                expected = rand(get_rng(), d, 2)
                observed = rand(get_rng(), md, 2)
                @test observed isa KeyedArray
                @test observed == expected
                @test axiskeys(observed) == (Base.OneTo(2),)
                @test dimnames(observed) == (:sample,)

                @test observed[1] isa KeyedArray
                @test isapprox(observed(1), expected[1])
                @test axiskeys(observed[1]) == axiskeys(md)
                @test dimnames(observed[1]) == dimnames(md)
            end
        end

        @testset "Sampleable not <:Distribution" begin
            samp = Distributions.MultinomialSampler(5, [0.5, 0.5])
            ksamp = KeyedSampleable(samp; id=["number1", "number2"])
            get_rng = StableRNG(1)

            @test ksamp isa Sampleable
            @test !(ksamp isa Distribution)
            @test axiskeys(ksamp) == (["number1", "number2"], )
            @test dimnames(ksamp) == (:id,)
            @test length(ksamp) == length(samp) == 2

            @test rand(get_rng, ksamp) == [3, 2]
            @test rand(get_rng, ksamp, 2) == [1 1; 4 4]
        end
    end

    @testset "Invalid keys $T" for T in (KeyedDistribution, KeyedSampleable)
        # Wrong number of keys
        @test_throws ArgumentError T(MvNormal(I(3)), ["foo"])
        @test_throws ArgumentError T(MvNormal(I(3)); id=["foo"])
        # Wrong key lengths
        @test_throws ArgumentError T(MvNormal(I(3)), ([:a, :b, :c], [:x]))
        @test_throws ArgumentError T(MvNormal(I(3)); dim1=[:a, :b, :c], dim2=[:x])
        @test_throws ArgumentError T(Wishart(7.0, Matrix(1.0I, 2, 2)), (["foo"], ["bar"]))
        @test_throws ArgumentError T(Wishart(7.0, Matrix(1.0I, 2, 2)); dim1=["foo"], dim2=["bar"])
        # AxisKeys requires key vectors to be AbstractVector
        @test_throws MethodError T(MvNormal(I(3)), (:a, :b, :c))
    end

    @testset "marginalising" begin

        X = rand(StableRNG(1234), 10, 3)
        m = vec(mean(X; dims=1))
        s = cov(X; dims=1)
        keys = ([:a, :b, :c], )
        names = (:id,)
        named_keys = NamedTuple{names}(keys)

        W = WoodburyPDMat(
            randn(StableRNG(1234), 3, 2),
            Diagonal(rand(StableRNG(1234), 2,)),
            Diagonal(rand(StableRNG(1234), 3,)),
        )

        @testset "KeyedMvNormal constructed with named keys" begin
            d = KeyedDistribution(MvNormal(m, s); named_keys...)
            @test d([:a, :b, :c]) == d[[1, 2, 3]] == d

            d13 = KeyedDistribution(MvNormal(m[[1, 3]], s[[1, 3], [1, 3]]); id=[:a, :c])
            @test d([:a, :c]) == d[[1, 3]] == d13

            @test d([:a]) == d[[1]] == KeyedDistribution(MvNormal(m[[1]], s[[1], [1]]); id=[:a])
            @test d(:a) == d[1] == KeyedDistribution(Normal(m[1], s[1, 1]), [:a])
        end

        @testset "KeyedMvNormal constructed without named keys" begin
            d = KeyedDistribution(MvNormal(m, s))
            @test d([1, 2, 3]) == d[[1, 2, 3]] == d

            d13 = KeyedDistribution(MvNormal(m[[1, 3]], s[[1, 3], [1, 3]]), [1, 3])
            @test d([1, 3]) == d[[1, 3]] == d13

            @test d([1]) == d[[1]] == KeyedDistribution(MvNormal(m[[1]], s[[1], [1]]), [1])
            @test d(1) == d[1] == KeyedDistribution(Normal(m[1], s[1, 1]), [1])
        end


        @testset "KeyedMvTDist constructed with named keys" begin
            d = KeyedDistribution(MvTDist(3, m, s); named_keys...)
            @test d([:a, :b, :c]) == d[[1, 2, 3]] == d

            d13 = KeyedDistribution(MvTDist(3, m[[1, 3]], s[[1, 3], [1, 3]]); id=[:a, :c])
            @test d([:a, :c]) == d[[1, 3]] == d13

            @test d([:a]) == d[[1]] == KeyedDistribution(MvTDist(3, m[[1]], s[[1], [1]]); id=[:a])
        end

        @testset "KeyedMvTDist constructed without named keys" begin
            d = KeyedDistribution(MvTDist(3, m, s))
            @test d([1, 2, 3]) == d[[1, 2, 3]] == d

            d13 = KeyedDistribution(MvTDist(3, m[[1, 3]], s[[1, 3], [1, 3]]), [1, 3])
            @test d([1, 3]) == d[[1, 3]] == d13

            @test d([1]) == d[[1]] == KeyedDistribution(MvTDist(3, m[[1]], s[[1], [1]]), [1])
        end

        @testset "KeyedGenericMvTDist with WoodburyPDMat" begin
            d = KeyedDistribution(GenericMvTDist(3, m, W))
            @test d([1, 2, 3]) == d[[1, 2, 3]] == d

            d13 = KeyedDistribution(GenericMvTDist(3, m[[1, 3]], submat(W, [1, 3])), [1, 3])
            @test d([1, 3]) == d[[1, 3]] == d13
            @test d([1, 3]).d.Σ isa WoodburyPDMat

            @test d([1]) == d[[1]] == KeyedDistribution(GenericMvTDist(3, m[[1]], submat(W, [1])), [1])
        end
    end

    @testset "NamedDims functions" begin
        X = rand(StableRNG(1234), 10, 3)
        m = vec(mean(X; dims=1))
        s = cov(X; dims=1)
        keys = ([:a, :b, :c], )
        names = (:id, )
        named_keys = NamedTuple{names}(keys)

        kd = KeyedDistribution(MvNormal(m, s); named_keys...)

        @test NamedDims.dim(kd, :id) == 1
        @test dimnames(unname(kd)) == (:_, )
        @test dimnames(rename(kd, (:name,))) == (:name, )
        @test dimnames(rename(kd, :id => :name)) == (:name, )
    end

    @testset "Equality comparison with wrapped type" begin
        d = MvNormal([1.0, 2.0], [1.0, 1.0]);

        kd = KeyedDistribution(d, 1:length(d));

        @test kd == d
        @test d == kd

        ks = KeyedSampleable(d, 1:length(d));

        @test ks == d
        @test d == ks
    end
end
