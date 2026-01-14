/*
 * Penguin Classification Example with Aorsf.NET
 *
 * This example demonstrates fitting an oblique random forest classifier
 * to predict penguin species using the Palmer Penguins dataset.
 *
 * This is the C# equivalent of the R example from the aorsf README:
 *
 *     penguin_fit <- orsf(data = penguins_orsf,
 *                         n_tree = 5,
 *                         formula = species ~ .)
 */

using Aorsf;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine(new string('=', 60));
        Console.WriteLine("Penguin Classification with Oblique Random Forest");
        Console.WriteLine(new string('=', 60));

        // Load data
        Console.WriteLine("\nLoading Palmer Penguins dataset...");
        var (X, y, featureNames, speciesNames) = LoadPenguins();
        Console.WriteLine($"Dataset shape: ({X.GetLength(0)}, {X.GetLength(1)})");
        Console.WriteLine($"Species: {string.Join(", ", speciesNames)}");

        Console.WriteLine($"\nFeatures: {string.Join(", ", featureNames)}");
        Console.WriteLine($"N observations: {y.Length}");
        Console.WriteLine($"N classes: {speciesNames.Length}");
        Console.WriteLine($"N predictors: {X.GetLength(1)}");

        // Fit oblique classification random forest
        Console.WriteLine("\n" + new string('-', 60));
        Console.WriteLine("Fitting Oblique Random Classification Forest...");
        Console.WriteLine(new string('-', 60));

        using var clf = new ObliqueForestClassifier
        {
            TreeCount = 5,
            MaxFeatures = 3,  // N predictors per node (like R default sqrt(7) ~ 3)
            MinSamplesLeaf = 5,
            Importance = ImportanceType.Anova,
            RandomState = 42
        };
        clf.Fit(X, y);

        // Print model summary (similar to R output)
        Console.WriteLine("\n---------- Oblique random classification forest");
        Console.WriteLine();
        Console.WriteLine($"     Linear combinations: Accelerated Logistic regression");
        Console.WriteLine($"          N observations: {y.Length}");
        Console.WriteLine($"               N classes: {speciesNames.Length}");
        Console.WriteLine($"                 N trees: {clf.TreeCount}");
        Console.WriteLine($"      N predictors total: {X.GetLength(1)}");
        Console.WriteLine($"   N predictors per node: {clf.MaxFeatures}");
        Console.WriteLine($" Min observations in leaf: {clf.MinSamplesLeaf}");
        Console.WriteLine($"          OOB stat value: {clf.OutOfBagScore:F2}");
        Console.WriteLine($"           OOB stat type: AUC-ROC");
        Console.WriteLine($"     Variable importance: anova");
        Console.WriteLine();
        Console.WriteLine(new string('-', 41));

        // Show predictions
        Console.WriteLine("\n" + new string('=', 60));
        Console.WriteLine("Predictions");
        Console.WriteLine(new string('=', 60));

        // Get first 5 samples
        var xFirst5 = GetRows(X, 0, 5);
        var predictions = clf.Predict(xFirst5);
        var probabilities = clf.PredictProbability(xFirst5);

        Console.WriteLine("\nFirst 5 predictions:");
        Console.WriteLine($"{"Actual",-12} {"Predicted",-12} {"Probabilities"}");
        Console.WriteLine(new string('-', 50));
        for (int i = 0; i < 5; i++)
        {
            var actual = speciesNames[y[i]];
            var predicted = speciesNames[predictions[i]];
            var probs = string.Join(", ", Enumerable.Range(0, speciesNames.Length)
                .Select(j => probabilities[i, j].ToString("F2")));
            Console.WriteLine($"{actual,-12} {predicted,-12} [{probs}]");
        }

        // Variable importance
        if (clf.FeatureImportances != null)
        {
            Console.WriteLine("\n" + new string('=', 60));
            Console.WriteLine("Variable Importance (ANOVA)");
            Console.WriteLine(new string('=', 60));

            // Sort by importance (descending)
            var importanceOrder = clf.FeatureImportances
                .Select((value, index) => (value, index))
                .OrderByDescending(x => x.value)
                .Select(x => x.index)
                .ToArray();

            Console.WriteLine($"\n{"Feature",-20} {"Importance",12}");
            Console.WriteLine(new string('-', 35));
            foreach (var idx in importanceOrder)
            {
                Console.WriteLine($"{featureNames[idx],-20} {clf.FeatureImportances[idx],12:F4}");
            }
        }

        // Full model with more trees
        Console.WriteLine("\n" + new string('=', 60));
        Console.WriteLine("Full Model (100 trees)");
        Console.WriteLine(new string('=', 60));

        using var clfFull = new ObliqueForestClassifier
        {
            TreeCount = 100,
            Importance = ImportanceType.Negate,
            RandomState = 42
        };
        clfFull.Fit(X, y);

        Console.WriteLine($"\nTraining Accuracy: {clfFull.Score(X, y):F3}");

        if (clfFull.FeatureImportances != null)
        {
            Console.WriteLine("\nVariable Importance (Negation):");
            var importanceOrder = clfFull.FeatureImportances
                .Select((value, index) => (value, index))
                .OrderByDescending(x => x.value)
                .Select(x => x.index)
                .Take(5)
                .ToArray();

            foreach (var idx in importanceOrder)
            {
                Console.WriteLine($"  {featureNames[idx],-20} {clfFull.FeatureImportances[idx]:F4}");
            }
        }

        Console.WriteLine("\n" + new string('=', 60));
        Console.WriteLine("Example complete!");
        Console.WriteLine(new string('=', 60));
    }

    static (double[,] X, int[] y, string[] featureNames, string[] speciesNames) LoadPenguins()
    {
        // Find the CSV file (in parent directory or current directory)
        var csvPath = FindFile("penguins.csv");
        if (csvPath == null)
            throw new FileNotFoundException("penguins.csv not found");

        var lines = File.ReadAllLines(csvPath);
        var header = ParseCsvLine(lines[0]);

        // Find column indices
        int speciesCol = Array.IndexOf(header, "species");
        int islandCol = Array.IndexOf(header, "island");
        int sexCol = Array.IndexOf(header, "sex");

        // Parse data
        var rows = new List<string[]>();
        for (int i = 1; i < lines.Length; i++)
        {
            if (!string.IsNullOrWhiteSpace(lines[i]))
                rows.Add(ParseCsvLine(lines[i]));
        }

        // Get unique values for categorical columns
        var speciesValues = rows.Select(r => r[speciesCol]).Distinct().OrderBy(x => x).ToArray();
        var islandValues = rows.Select(r => r[islandCol]).Distinct().OrderBy(x => x).ToArray();
        var sexValues = rows.Select(r => r[sexCol]).Distinct().OrderBy(x => x).ToArray();

        // Feature columns (all except species)
        var featureNames = header.Where(h => h != "species").ToArray();
        int nFeatures = featureNames.Length;
        int nSamples = rows.Count;

        // Create arrays
        var X = new double[nSamples, nFeatures];
        var y = new int[nSamples];

        for (int i = 0; i < nSamples; i++)
        {
            var row = rows[i];

            // Target: species
            y[i] = Array.IndexOf(speciesValues, row[speciesCol]);

            // Features
            int fIdx = 0;
            for (int j = 0; j < header.Length; j++)
            {
                if (header[j] == "species") continue;

                string value = row[j];
                if (header[j] == "island")
                    X[i, fIdx] = Array.IndexOf(islandValues, value);
                else if (header[j] == "sex")
                    X[i, fIdx] = Array.IndexOf(sexValues, value);
                else
                    X[i, fIdx] = double.Parse(value);

                fIdx++;
            }
        }

        return (X, y, featureNames, speciesValues);
    }

    static string[] ParseCsvLine(string line)
    {
        // Simple CSV parser that handles quoted fields
        var result = new List<string>();
        bool inQuotes = false;
        var current = new System.Text.StringBuilder();

        for (int i = 0; i < line.Length; i++)
        {
            char c = line[i];
            if (c == '"')
            {
                inQuotes = !inQuotes;
            }
            else if (c == ',' && !inQuotes)
            {
                result.Add(current.ToString());
                current.Clear();
            }
            else
            {
                current.Append(c);
            }
        }
        result.Add(current.ToString());

        return result.ToArray();
    }

    static string? FindFile(string filename)
    {
        // Check current directory and parent directories
        var searchPaths = new[]
        {
            Path.Combine(AppContext.BaseDirectory, filename),
            Path.Combine(AppContext.BaseDirectory, "..", filename),
            Path.Combine(AppContext.BaseDirectory, "..", "..", filename),
            Path.Combine(Directory.GetCurrentDirectory(), filename),
            Path.Combine(Directory.GetCurrentDirectory(), "..", filename),
        };

        foreach (var path in searchPaths)
        {
            if (File.Exists(path))
                return path;
        }

        return null;
    }

    static double[,] GetRows(double[,] matrix, int startRow, int count)
    {
        int cols = matrix.GetLength(1);
        var result = new double[count, cols];
        for (int i = 0; i < count; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = matrix[startRow + i, j];
            }
        }
        return result;
    }
}
