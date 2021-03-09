using AxisKeys
using Distributions
using KeyedDistributions
using LinearAlgebra
using Random
using Statistics
using Test

@testset "KeyedDistributions.jl" begin
    X = rand(MersenneTwister(1234), 10, 3)
    m = vec(mean(X; dims=1))
    s = cov(X; dims=1)
    d = MvNormal(m, s)
    keys = [:a, :b, :c]
    kd = KeyedDistribution(d, keys)

    @testset "Common" begin
        @testset "$T" for T in (KeyedDistribution, KeyedSampleable)
            kd = T(d, keys)

            @testset "base functions" begin
                @test kd isa Sampleable
                @test distribution(kd) == d
                @test parent(kd) == d
                @test AxisKeys.keyless(kd) == d
                @test axiskeys(kd) == (keys, )
                @test length(kd) == length(d) == 3
                @test isequal(kd, T(d, [:a, :b, :c]))
                @test ==(kd, T(d, keys))
            end

            @testset "sampling" begin
                # Samples from the distribution both wrapped and unwrapped should be the same.
                @test rand(MersenneTwister(1), d) == rand(MersenneTwister(1), kd)
                @test rand(MersenneTwister(1), d, 3) == rand(MersenneTwister(1), kd, 3)

                rng = MersenneTwister(1)

                @testset "one-sample method" begin
                    expected = [0.6048058078690228, 0.5560878435408364, 0.41599188102577894]
                    observed = rand(rng, kd)
                    @test observed isa KeyedArray
                    @test isapprox(observed, expected)
                    @test isapprox(observed(:a), expected[1])
                    @test first(axiskeys(observed)) == first(axiskeys(kd))
                end

                @testset "multi-sample method" begin
                    expected = [
                        0.5333595810495181 1.0686337176543628;
                        0.12636240168741192 -0.44724174718706566;
                        0.579000443516295 0.914066493850531
                    ]
                    observed = rand(rng, kd, 2)
                    @test observed isa KeyedArray
                    @test isapprox(observed, expected)
                    @test isapprox(observed(:a), expected[1, :])
                    @test first(axiskeys(observed)) == first(axiskeys(kd))
                end
            end
        end
    end

    @testset "Non-Distribution Sampleable only" begin
        sampler = Distributions.MultinomialSampler(3, [0.1, 0.2, 0.3])

        @test axiskeys(sampler) == (Base.OneTo(3), )
    end

    @testset "KeyedDistribution only" begin
        @testset "Inner keys constructor" begin
            kd2 = KeyedDistribution(MvNormal(KeyedArray(m, keys), s))

            @test kd2 isa Distribution
            @test distribution(kd2) == d
            @test axiskeys(kd2) == (keys, )
            @test mean(kd2) == m
        end

        kd = KeyedDistribution(d, keys)

        @testset "base functions" begin
            @test kd isa Distribution
            @test sampler(kd) == sampler(d)
            @test eltype(kd) == eltype(d) == Float64
        end

        @testset "statistical functions" begin
            @test mean(kd) isa KeyedArray{Float64, 1}
            @test parent(mean(kd)) == mean(d) == m

            @test var(kd) isa KeyedArray{Float64, 1}
            @test parent(var(kd)) == var(d) == diag(s)

            @test cov(kd) isa KeyedArray{Float64, 2}
            @test parent(cov(kd)) == cov(d) == s

            @test entropy(kd) isa Number
            @test entropy(kd) == entropy(d)
            @test entropy(kd, 2) == entropy(d, 2)

            @test Distributions._logpdf(kd, m) isa Number
            @test Distributions._logpdf(kd, m) == Distributions._logpdf(d, m)

            # statistical functions commute with parent on KeyedArray/KeyedDistribution
            for f in (mean, var, cov)
                @test f(parent(kd)) == parent(f(kd))
            end
        end
    end

    @testset "Invalid keys $T" for T in (KeyedDistribution, KeyedSampleable)
        @test_throws DimensionMismatch T(MvNormal(ones(3)), ["foo"])
        @test_throws MethodError T(MvNormal(ones(3)), (:a, :b, :c))  # because Tuple
    end

end
