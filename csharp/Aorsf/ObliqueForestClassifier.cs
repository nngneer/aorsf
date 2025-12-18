using System;
using System.Runtime.InteropServices;
using Aorsf.Native;

namespace Aorsf
{
    /// <summary>
    /// Oblique random forest classifier using linear combination splits.
    /// </summary>
    public class ObliqueForestClassifier : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;
        private int _featureCount;
        private int _classCount;

        // Configuration properties (.NET-idiomatic naming)

        /// <summary>Number of trees in the forest. Default: 500.</summary>
        public int TreeCount { get; set; } = 500;

        /// <summary>Number of features to consider at each split. Null = sqrt(n_features).</summary>
        public int? MaxFeatures { get; set; }

        /// <summary>Minimum number of observations in a leaf node. Default: 5.</summary>
        public int MinSamplesLeaf { get; set; } = 5;

        /// <summary>Minimum number of observations required to consider a split. Default: 10.</summary>
        public int MinSamplesSplit { get; set; } = 10;

        /// <summary>Number of random splits to evaluate at each node. Default: 5.</summary>
        public int SplitCount { get; set; } = 5;

        /// <summary>Method for computing feature importance. Default: None.</summary>
        public ImportanceType Importance { get; set; } = ImportanceType.None;

        /// <summary>Method for computing linear combinations. Default: Glm.</summary>
        public LinearComboMethod LinearComboMethod { get; set; } = LinearComboMethod.Glm;

        /// <summary>Number of threads for parallel processing. 0 = auto-detect.</summary>
        public int ThreadCount { get; set; } = 0;

        /// <summary>Random seed for reproducibility. Null = random.</summary>
        public uint? RandomState { get; set; }

        /// <summary>Verbosity level (0 = silent).</summary>
        public int Verbosity { get; set; } = 0;

        // Fitted attributes (read-only after fitting)

        /// <summary>Feature importance scores (available after fitting with Importance != None).</summary>
        public double[]? FeatureImportances { get; private set; }

        /// <summary>Out-of-bag accuracy score.</summary>
        public double OutOfBagScore { get; private set; }

        /// <summary>Number of features seen during fitting.</summary>
        public int FeatureCount => _featureCount;

        /// <summary>Number of classes.</summary>
        public int ClassCount => _classCount;

        /// <summary>Whether the model has been fitted.</summary>
        public bool IsFitted => _handle != IntPtr.Zero &&
                                NativeMethods.aorsf_forest_is_fitted(_handle) != 0;

        /// <summary>
        /// Fit the classifier to training data.
        /// </summary>
        /// <param name="features">Training feature matrix [n_samples, n_features].</param>
        /// <param name="labels">Target class labels [n_samples].</param>
        /// <param name="sampleWeights">Optional sample weights [n_samples].</param>
        public void Fit(double[,] features, int[] labels, double[]? sampleWeights = null)
        {
            if (features == null) throw new ArgumentNullException(nameof(features));
            if (labels == null) throw new ArgumentNullException(nameof(labels));

            int sampleCount = features.GetLength(0);
            int featureCount = features.GetLength(1);

            if (labels.Length != sampleCount)
                throw new ArgumentException("Features and labels must have same number of samples.");

            // Determine number of classes
            int minClass = int.MaxValue, maxClass = int.MinValue;
            foreach (var label in labels)
            {
                if (label < minClass) minClass = label;
                if (label > maxClass) maxClass = label;
            }
            _classCount = maxClass - minClass + 1;

            // Convert to flat row-major array
            double[] featuresFlat = new double[sampleCount * featureCount];
            for (int i = 0; i < sampleCount; i++)
                for (int j = 0; j < featureCount; j++)
                    featuresFlat[i * featureCount + j] = features[i, j];

            double[] labelsDouble = new double[sampleCount];
            for (int i = 0; i < sampleCount; i++)
                labelsDouble[i] = labels[i] - minClass;  // Zero-based

            // Initialize configuration
            var config = new NativeMethods.AorsfConfig();
            NativeMethods.aorsf_config_init(ref config, (int)TreeType.Classification);

            config.NTree = TreeCount;
            config.Mtry = MaxFeatures ?? 0;
            config.LeafMinObs = MinSamplesLeaf;
            config.SplitMinObs = MinSamplesSplit;
            config.NSplit = SplitCount;
            config.ViType = (int)Importance;
            config.LincombType = (int)LinearComboMethod;
            config.NThread = ThreadCount;
            config.Seed = RandomState ?? 0;
            config.Verbosity = Verbosity;
            config.Oobag = 1;

            // Dispose previous handle if exists
            if (_handle != IntPtr.Zero)
            {
                NativeMethods.aorsf_forest_destroy(_handle);
                _handle = IntPtr.Zero;
            }

            // Create forest
            int err = NativeMethods.aorsf_forest_create(out _handle, ref config);
            AorsfException.ThrowIfError(err);

            // Create data
            IntPtr dataHandle;
            err = NativeMethods.aorsf_data_create(
                out dataHandle, featuresFlat, sampleCount, featureCount,
                labelsDouble, 1, sampleWeights, _classCount);
            AorsfException.ThrowIfError(err);

            try
            {
                // Fit model
                err = NativeMethods.aorsf_forest_fit(_handle, dataHandle);
                AorsfException.ThrowIfError(err);

                _featureCount = NativeMethods.aorsf_forest_get_n_features(_handle);

                // Get OOB score
                err = NativeMethods.aorsf_forest_get_oob_error(_handle, out double oobError);
                if (err == NativeMethods.AORSF_SUCCESS)
                    OutOfBagScore = 1.0 - oobError;  // Convert error to accuracy

                // Get importance if computed
                if (Importance != ImportanceType.None)
                {
                    FeatureImportances = new double[_featureCount];
                    err = NativeMethods.aorsf_forest_get_importance(
                        _handle, FeatureImportances, _featureCount);
                    if (err != NativeMethods.AORSF_SUCCESS)
                        FeatureImportances = null;
                }
            }
            finally
            {
                NativeMethods.aorsf_data_destroy(dataHandle);
            }
        }

        /// <summary>
        /// Predict class labels for samples.
        /// </summary>
        /// <param name="features">Feature matrix [n_samples, n_features].</param>
        /// <returns>Predicted class labels [n_samples].</returns>
        public int[] Predict(double[,] features)
        {
            if (!IsFitted)
                throw new InvalidOperationException("Model not fitted. Call Fit() first.");

            int sampleCount = features.GetLength(0);
            int featureCount = features.GetLength(1);

            if (featureCount != _featureCount)
                throw new ArgumentException($"Expected {_featureCount} features, got {featureCount}.");

            // Convert to flat row-major array
            double[] featuresFlat = new double[sampleCount * featureCount];
            for (int i = 0; i < sampleCount; i++)
                for (int j = 0; j < featureCount; j++)
                    featuresFlat[i * featureCount + j] = features[i, j];

            // Get predictions
            double[] predictions = new double[sampleCount];
            int err = NativeMethods.aorsf_forest_predict(
                _handle, featuresFlat, sampleCount, featureCount,
                (int)PredictionType.Class, predictions, sampleCount);
            AorsfException.ThrowIfError(err);

            // Convert to int array
            int[] result = new int[sampleCount];
            for (int i = 0; i < sampleCount; i++)
                result[i] = (int)predictions[i];

            return result;
        }

        /// <summary>
        /// Predict class probabilities for samples.
        /// </summary>
        /// <param name="features">Feature matrix [n_samples, n_features].</param>
        /// <returns>Class probabilities [n_samples, n_classes].</returns>
        public double[,] PredictProbability(double[,] features)
        {
            if (!IsFitted)
                throw new InvalidOperationException("Model not fitted. Call Fit() first.");

            int sampleCount = features.GetLength(0);
            int featureCount = features.GetLength(1);

            if (featureCount != _featureCount)
                throw new ArgumentException($"Expected {_featureCount} features, got {featureCount}.");

            // Convert to flat row-major array
            double[] featuresFlat = new double[sampleCount * featureCount];
            for (int i = 0; i < sampleCount; i++)
                for (int j = 0; j < featureCount; j++)
                    featuresFlat[i * featureCount + j] = features[i, j];

            // Get output dimensions
            int err = NativeMethods.aorsf_predict_get_dims(
                _handle, sampleCount, (int)PredictionType.Probability,
                out int outRows, out int outCols);
            AorsfException.ThrowIfError(err);

            // Get predictions
            double[] predictions = new double[outRows * outCols];
            err = NativeMethods.aorsf_forest_predict(
                _handle, featuresFlat, sampleCount, featureCount,
                (int)PredictionType.Probability, predictions, predictions.Length);
            AorsfException.ThrowIfError(err);

            // Convert to 2D array
            double[,] result = new double[outRows, outCols];
            for (int i = 0; i < outRows; i++)
                for (int j = 0; j < outCols; j++)
                    result[i, j] = predictions[i * outCols + j];

            return result;
        }

        /// <summary>
        /// Calculate accuracy score on test data.
        /// </summary>
        /// <param name="features">Test feature matrix [n_samples, n_features].</param>
        /// <param name="labels">True class labels [n_samples].</param>
        /// <returns>Classification accuracy (0.0 to 1.0).</returns>
        public double Score(double[,] features, int[] labels)
        {
            var predictions = Predict(features);
            int correct = 0;
            for (int i = 0; i < labels.Length; i++)
                if (predictions[i] == labels[i])
                    correct++;
            return (double)correct / labels.Length;
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (_handle != IntPtr.Zero)
                {
                    NativeMethods.aorsf_forest_destroy(_handle);
                    _handle = IntPtr.Zero;
                }
                _disposed = true;
            }
        }

        ~ObliqueForestClassifier()
        {
            Dispose(false);
        }
    }
}
