# Aorsf.NET

.NET bindings for the aorsf (Accelerated Oblique Random Forests) library.

## Installation

```bash
dotnet add package Aorsf
```

## Quick Start

### Classification

```csharp
using Aorsf;

// Create and configure classifier
var classifier = new ObliqueForestClassifier
{
    TreeCount = 100,
    MaxFeatures = 5,
    Importance = ImportanceType.Negate
};

// Fit model
classifier.Fit(trainingFeatures, trainingLabels);

// Make predictions
int[] predictions = classifier.Predict(testFeatures);

// Evaluate
double accuracy = classifier.Score(testFeatures, testLabels);
Console.WriteLine($"Accuracy: {accuracy:P2}");

// Feature importance
if (classifier.FeatureImportances != null)
{
    for (int i = 0; i < classifier.FeatureImportances.Length; i++)
        Console.WriteLine($"Feature {i}: {classifier.FeatureImportances[i]:F4}");
}
```

### Regression

```csharp
using Aorsf;

var regressor = new ObliqueForestRegressor
{
    TreeCount = 100
};

// Fit model
regressor.Fit(features, targets);

// Predict
double[] predictions = regressor.Predict(testFeatures);

// Evaluate R-squared
double r2 = regressor.Score(testFeatures, testTargets);
```

### Survival Analysis

```csharp
using Aorsf;

var survivalForest = new ObliqueForestSurvival
{
    TreeCount = 100
};

// outcomes should be [n_samples, 2] with columns: time, status
survivalForest.Fit(features, outcomes);

// Predict risk scores (higher = higher risk)
double[] riskScores = survivalForest.Predict(testFeatures);
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `TreeCount` | int | 500 | Number of trees in the forest |
| `MaxFeatures` | int? | null | Features per split (null = sqrt(n_features)) |
| `MinSamplesLeaf` | int | 5 | Minimum samples in a leaf node |
| `MinSamplesSplit` | int | 10 | Minimum samples to consider a split |
| `Importance` | ImportanceType | None | Variable importance method |
| `LinearComboMethod` | LinearComboMethod | Glm | Method for computing linear combinations |
| `ThreadCount` | int | 0 | Number of threads (0 = auto-detect) |
| `RandomState` | uint? | null | Random seed for reproducibility |

## Variable Importance Methods

| Method | Description | Speed |
|--------|-------------|-------|
| `Negate` | Negation importance (recommended) | Fast |
| `Permute` | Permutation importance | Moderate |
| `Anova` | ANOVA-based importance | Fast |

## Model Serialization

Save and load trained models in binary or JSON format.

### Save to File

```csharp
// Save in binary format (default, fast and compact)
classifier.Save("model.orsf");

// Save in JSON format (human-readable)
classifier.Save("model.json", SerializationFormat.Json);

// Include importance scores and metadata
classifier.Save("model.orsf", SerializationFormat.Binary,
    SerializationFlags.IncludeImportance | SerializationFlags.IncludeMetadata);
```

### Load from File

```csharp
// Load model (format auto-detected)
var classifier = ObliqueForestClassifier.Load("model.orsf");
var regressor = ObliqueForestRegressor.Load("model.orsf");
var survival = ObliqueForestSurvival.Load("model.orsf");
```

### Serialize to Bytes

```csharp
// Save to byte array (useful for databases, network transfer)
byte[] data = classifier.SaveToBytes(SerializationFormat.Binary,
    SerializationFlags.IncludeImportance);

// Load from byte array
var loaded = ObliqueForestClassifier.LoadFromBytes(data);
```

### Serialization Flags

| Flag | Description |
|------|-------------|
| `None` | Core model only |
| `IncludeImportance` | Include variable importance scores |
| `IncludeOobData` | Include out-of-bag data for partial dependence |
| `IncludeMetadata` | Include feature names and statistics |
| `All` | Include all optional data |

### Cross-Language Models

Models saved with the C# wrapper can be loaded in Python and vice versa, enabling training in one language and deployment in another.

## Platform Support

- Windows x64
- Linux x64
- macOS x64
- macOS ARM64 (Apple Silicon)

## License

MIT License - see [LICENSE](https://github.com/ropensci/aorsf/blob/main/LICENSE.md) for details.

## References

- Jaeger et al. (2023). "Accelerated and interpretable oblique random survival forests." *Journal of Computational and Graphical Statistics*. DOI: 10.1080/10618600.2023.2231048
- [aorsf R package documentation](https://docs.ropensci.org/aorsf/)
