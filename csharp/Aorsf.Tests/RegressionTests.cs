using Xunit;

namespace Aorsf.Tests;

public class RegressionTests
{
    private static (double[,] X, double[] y) GenerateData(int nSamples, int nFeatures, int seed = 42)
    {
        var random = new Random(seed);
        var X = new double[nSamples, nFeatures];
        var y = new double[nSamples];

        for (int i = 0; i < nSamples; i++)
        {
            double sum = 0;
            for (int j = 0; j < nFeatures; j++)
            {
                X[i, j] = random.NextDouble() * 2 - 1;
                if (j < 2) sum += X[i, j];
            }
            // y is a linear combination with noise
            y[i] = sum + random.NextDouble() * 0.5;
        }

        return (X, y);
    }

    [Fact]
    public void FitPredict_ReturnsValidPredictions()
    {
        var (features, targets) = GenerateData(200, 5);

        using var regressor = new ObliqueForestRegressor
        {
            TreeCount = 10,
            RandomState = 42
        };

        regressor.Fit(features, targets);

        Assert.True(regressor.IsFitted);

        var predictions = regressor.Predict(features);

        Assert.Equal(targets.Length, predictions.Length);
        // Predictions should be in reasonable range
        Assert.All(predictions, p => Assert.True(p > -10 && p < 10));
    }

    [Fact]
    public void Score_ReturnsR2()
    {
        var (features, targets) = GenerateData(200, 5);

        using var regressor = new ObliqueForestRegressor
        {
            TreeCount = 50,
            RandomState = 42
        };

        regressor.Fit(features, targets);
        var score = regressor.Score(features, targets);

        // R2 should be reasonable on training data
        Assert.True(score > 0);  // Better than mean prediction
        Assert.True(score <= 1.0);
    }

    [Fact]
    public void FeatureImportance_IsComputed()
    {
        var (features, targets) = GenerateData(200, 5);

        using var regressor = new ObliqueForestRegressor
        {
            TreeCount = 10,
            Importance = ImportanceType.Negate,
            RandomState = 42
        };

        regressor.Fit(features, targets);

        Assert.NotNull(regressor.FeatureImportances);
        Assert.Equal(5, regressor.FeatureImportances!.Length);
        Assert.True(regressor.FeatureImportances.Any(x => Math.Abs(x) > 0.001));
    }

    [Fact]
    public void OutOfBagR2_IsComputed()
    {
        var (features, targets) = GenerateData(200, 5);

        using var regressor = new ObliqueForestRegressor
        {
            TreeCount = 50,
            RandomState = 42
        };

        regressor.Fit(features, targets);

        // OOB R2 should exist (might be low for random data)
        Assert.True(regressor.OutOfBagR2 >= -10);  // Allow some negative values for bad models
        Assert.True(regressor.OutOfBagR2 <= 1);
    }

    [Fact]
    public void Predict_BeforeFit_ThrowsException()
    {
        var regressor = new ObliqueForestRegressor();
        var features = new double[10, 5];

        Assert.Throws<InvalidOperationException>(() => regressor.Predict(features));
    }

    [Fact]
    public void Fit_MismatchedDimensions_ThrowsException()
    {
        var features = new double[100, 5];
        var targets = new double[50];  // Wrong size

        using var regressor = new ObliqueForestRegressor();

        Assert.Throws<ArgumentException>(() => regressor.Fit(features, targets));
    }

    [Fact]
    public void Dispose_ReleasesResources()
    {
        var (features, targets) = GenerateData(100, 5);

        var regressor = new ObliqueForestRegressor { TreeCount = 5 };
        regressor.Fit(features, targets);
        regressor.Dispose();

        // Double dispose should be safe
        regressor.Dispose();
    }
}
