using Xunit;

namespace Aorsf.Tests;

public class ClassificationTests
{
    private static (double[,] X, int[] y) GenerateData(int nSamples, int nFeatures, int seed = 42)
    {
        var random = new Random(seed);
        var X = new double[nSamples, nFeatures];
        var y = new int[nSamples];

        for (int i = 0; i < nSamples; i++)
        {
            double sum = 0;
            for (int j = 0; j < nFeatures; j++)
            {
                X[i, j] = random.NextDouble() * 2 - 1;
                if (j < 2) sum += X[i, j];
            }
            y[i] = sum > 0 ? 1 : 0;
        }

        return (X, y);
    }

    [Fact]
    public void FitPredict_ReturnsValidPredictions()
    {
        var (features, labels) = GenerateData(200, 5);

        using var classifier = new ObliqueForestClassifier
        {
            TreeCount = 10,
            RandomState = 42
        };

        classifier.Fit(features, labels);

        Assert.True(classifier.IsFitted);

        var predictions = classifier.Predict(features);

        Assert.Equal(labels.Length, predictions.Length);
        Assert.All(predictions, p => Assert.True(p == 0 || p == 1));
    }

    [Fact(Skip = "Probability prediction not fully implemented in C API")]
    public void PredictProbability_ReturnsProbabilities()
    {
        var (features, labels) = GenerateData(100, 5);

        using var classifier = new ObliqueForestClassifier
        {
            TreeCount = 10,
            RandomState = 42
        };

        classifier.Fit(features, labels);
        var probabilities = classifier.PredictProbability(features);

        Assert.Equal(features.GetLength(0), probabilities.GetLength(0));
        Assert.Equal(2, probabilities.GetLength(1));  // Binary classification

        // Probabilities should sum to 1
        for (int i = 0; i < probabilities.GetLength(0); i++)
        {
            double sum = probabilities[i, 0] + probabilities[i, 1];
            Assert.True(Math.Abs(sum - 1.0) < 0.01);
        }
    }

    [Fact]
    public void Score_ReturnsAccuracy()
    {
        var (features, labels) = GenerateData(200, 5);

        using var classifier = new ObliqueForestClassifier
        {
            TreeCount = 50,
            RandomState = 42
        };

        classifier.Fit(features, labels);
        var score = classifier.Score(features, labels);

        Assert.True(score > 0.5);  // Better than random
        Assert.True(score <= 1.0);
    }

    [Fact]
    public void FeatureImportance_IsComputed()
    {
        var (features, labels) = GenerateData(200, 5);

        using var classifier = new ObliqueForestClassifier
        {
            TreeCount = 10,
            Importance = ImportanceType.Negate,
            RandomState = 42
        };

        classifier.Fit(features, labels);

        Assert.NotNull(classifier.FeatureImportances);
        Assert.Equal(5, classifier.FeatureImportances!.Length);
        Assert.True(classifier.FeatureImportances.Any(x => Math.Abs(x) > 0.001));
    }

    [Fact(Skip = "OOB evaluation not fully implemented in C API")]
    public void OutOfBagScore_IsAvailable()
    {
        var (features, labels) = GenerateData(200, 5);

        using var classifier = new ObliqueForestClassifier
        {
            TreeCount = 50,
            RandomState = 42
        };

        classifier.Fit(features, labels);

        // OOB score computed - value depends on model quality
        // Just verify it's a valid number (not NaN or infinite)
        Assert.False(double.IsNaN(classifier.OutOfBagScore));
        Assert.False(double.IsInfinity(classifier.OutOfBagScore));
    }

    [Fact]
    public void Predict_BeforeFit_ThrowsException()
    {
        var classifier = new ObliqueForestClassifier();
        var features = new double[10, 5];

        Assert.Throws<InvalidOperationException>(() => classifier.Predict(features));
    }

    [Fact]
    public void Fit_MismatchedDimensions_ThrowsException()
    {
        var features = new double[100, 5];
        var labels = new int[50];  // Wrong size

        using var classifier = new ObliqueForestClassifier();

        Assert.Throws<ArgumentException>(() => classifier.Fit(features, labels));
    }

    [Fact]
    public void Dispose_ReleasesResources()
    {
        var (features, labels) = GenerateData(100, 5);

        var classifier = new ObliqueForestClassifier { TreeCount = 5 };
        classifier.Fit(features, labels);
        classifier.Dispose();

        // Double dispose should be safe
        classifier.Dispose();
    }
}
