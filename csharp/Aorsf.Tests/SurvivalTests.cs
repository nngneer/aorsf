using Xunit;

namespace Aorsf.Tests;

public class SurvivalTests
{
    private static (double[,] X, double[,] y) GenerateData(int nSamples, int nFeatures, int seed = 42)
    {
        var random = new Random(seed);
        var X = new double[nSamples, nFeatures];
        var y = new double[nSamples, 2];  // time, status

        for (int i = 0; i < nSamples; i++)
        {
            for (int j = 0; j < nFeatures; j++)
            {
                X[i, j] = random.NextDouble() * 2 - 1;
            }
            y[i, 0] = random.NextDouble() * 100 + 1;  // time
            y[i, 1] = random.Next(2);  // status: 0 or 1
        }

        return (X, y);
    }

    [Fact]
    public void FitPredict_ReturnsRiskScores()
    {
        var (features, outcomes) = GenerateData(200, 5);

        using var forest = new ObliqueForestSurvival
        {
            TreeCount = 10,
            RandomState = 42
        };

        forest.Fit(features, outcomes);

        Assert.True(forest.IsFitted);

        var predictions = forest.Predict(features);

        Assert.Equal(features.GetLength(0), predictions.Length);
    }

    [Fact]
    public void UniqueTimes_IsPopulated()
    {
        var (features, outcomes) = GenerateData(200, 5);

        using var forest = new ObliqueForestSurvival
        {
            TreeCount = 10,
            RandomState = 42
        };

        forest.Fit(features, outcomes);

        Assert.NotNull(forest.UniqueTimes);
        Assert.True(forest.UniqueTimes!.Length > 0);
    }

    [Fact]
    public void OutOfBagConcordance_IsComputed()
    {
        var (features, outcomes) = GenerateData(200, 5);

        using var forest = new ObliqueForestSurvival
        {
            TreeCount = 50,
            RandomState = 42
        };

        forest.Fit(features, outcomes);

        // Concordance should be between 0 and 1
        Assert.True(forest.OutOfBagConcordance >= 0);
        Assert.True(forest.OutOfBagConcordance <= 1);
    }

    [Fact]
    public void FeatureImportance_IsComputed()
    {
        var (features, outcomes) = GenerateData(200, 5);

        using var forest = new ObliqueForestSurvival
        {
            TreeCount = 10,
            Importance = ImportanceType.Negate,
            RandomState = 42
        };

        forest.Fit(features, outcomes);

        Assert.NotNull(forest.FeatureImportances);
        Assert.Equal(5, forest.FeatureImportances!.Length);
    }

    [Fact]
    public void Predict_BeforeFit_ThrowsException()
    {
        var forest = new ObliqueForestSurvival();
        var features = new double[10, 5];

        Assert.Throws<InvalidOperationException>(() => forest.Predict(features));
    }

    [Fact]
    public void Fit_WrongOutcomeColumns_ThrowsException()
    {
        var features = new double[100, 5];
        var outcomes = new double[100, 1];  // Wrong: should be 2 columns

        using var forest = new ObliqueForestSurvival();

        Assert.Throws<ArgumentException>(() => forest.Fit(features, outcomes));
    }

    [Fact]
    public void Dispose_ReleasesResources()
    {
        var (features, outcomes) = GenerateData(100, 5);

        var forest = new ObliqueForestSurvival { TreeCount = 5 };
        forest.Fit(features, outcomes);
        forest.Dispose();

        // Double dispose should be safe
        forest.Dispose();
    }
}
