using AxisKeys
using Distributions
using KeyedDistributions
using LinearAlgebra
using Random
using Statistics
using Test

@testset "KeyedDistributions.jl" begin
    # Write your tests here.

    @testset "Inner keys constructor" begin
        keys = [:a, :b]
        m = KeyedArray([1., 2.], keys)
        d = MvNormal(m, [1., 1.])
        kd = KeyedDistribution(d)

        @test distribution(kd) == d
        @test axiskeys(kd) == (keys, )
        @test mean(kd) == m
    end

    @testset "Common" begin
        X = rand(MersenneTwister(1234), 10, 5)
        m = vec(mean(X; dims=1))
        s = cov(X; dims=1)
        d = MvNormal(m, s)
        keys = [:a, :b, :c, :d, :e]

        @testset for T in (KeyedDistribution, KeyedSampleable)
            kd = T(d, keys)

            @testset "base functions" begin
                @test kd isa Sampleable
                @test distribution(kd) == d
                @test parent(kd) == d
                @test axiskeys(kd) == (keys, )
                @test length(kd) == length(d) == 5
                @test eltype(kd) == eltype(d) == Float64
                @test isequal(kd, T(d, [:a, :b, :c, :d, :e]))
                @test ==(kd, T(d, keys))
            end

            @testset "statistical functions" begin
                @test mean(kd) isa KeyedArray{Float64, 1}
                @test parent(mean(kd)) == mean(d) == m
                # @test axisnames(mean(kd)) == (:variates,)

                @test var(kd) isa KeyedArray{Float64, 1}
                @test parent(var(kd)) == var(d) == diag(s)
                # @test axisnames(var(kd)) == (:variates,)

                @test cov(kd) isa KeyedArray{Float64, 2}
                @test parent(cov(kd)) == cov(d) == s
                # @test axisnames(cov(kd)) == (:variates, :variates_)

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

            @testset "sampling" begin
                # Samples from the distribution both wrapped and unwrapped should be the same.
                @test rand(MersenneTwister(1), d) == rand(MersenneTwister(1), kd)
                @test rand(MersenneTwister(1), d, 3) == rand(MersenneTwister(1), kd, 3)

                rng = MersenneTwister(1)

                @testset "one-sample method" begin
                    expected = [
                        0.6048058078690228,
                        0.5560878435408365,
                        0.41599188102577894,
                        0.4756226986245742,
                        0.15366818427047801,
                    ]
                    observed = rand(rng, kd)
                    @test observed isa KeyedArray
                    @test isapprox(observed, expected)
                    @test isapprox(observed(:a), expected[1])
                    # @test axisnames(observed) == (:variates,)
                    @test first(axiskeys(observed)) == first(axiskeys(kd))
                end

                @testset "multi-sample method" begin
                    expected = [
                        0.6080151671094673,
                        1.2415182218538203,
                        -4.4285504138930065e-5,
                        0.7298398256256964,
                        0.2103467702699237,
                    ]
                    observed = rand(rng, kd, 1)
                    @test observed isa KeyedArray
                    @test isapprox(observed, expected)
                    @test isapprox(observed(:a), [expected[1]])
                    # @test axisnames(observed) == (:variates, :samples)
                    @test first(axiskeys(observed)) == first(axiskeys(kd))
                end
            end
        end
    end

    @testset "Invalid keys $T" for T in (KeyedDistribution, KeyedSampleable)
        @test_throws DimensionMismatch T(MvNormal(ones(3)), ["foo"])
    end

end
