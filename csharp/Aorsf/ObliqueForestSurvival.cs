using System;
using System.Runtime.InteropServices;
using Aorsf.Native;

namespace Aorsf
{
    /// <summary>
    /// Oblique random survival forest using linear combination splits.
    /// </summary>
    public class ObliqueForestSurvival : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;
        private int _featureCount;

        // Configuration properties (.NET-idiomatic naming)

        /// <summary>Number of trees in the forest. Default: 500.</summary>
        public int TreeCount { get; set; } = 500;

        /// <summary>Number of features to consider at each split. Null = sqrt(n_features).</summary>
        public int? MaxFeatures { get; set; }

        /// <summary>Minimum number of observations in a leaf node. Default: 5.</summary>
        public int MinSamplesLeaf { get; set; } = 5;

        /// <summary>Minimum number of events in a leaf node. Default: 1.</summary>
        public int MinEventsLeaf { get; set; } = 1;

        /// <summary>Minimum number of observations required to consider a split. Default: 10.</summary>
        public int MinSamplesSplit { get; set; } = 10;

        /// <summary>Minimum number of events required to consider a split. Default: 5.</summary>
        public int MinEventsSplit { get; set; } = 5;

        /// <summary>Number of random splits to evaluate at each node. Default: 5.</summary>
        public int SplitCount { get; set; } = 5;

        /// <summary>Split rule for survival analysis. Default: LogRank.</summary>
        public SplitRule SplitRule { get; set; } = SplitRule.LogRank;

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

        /// <summary>Out-of-bag concordance index.</summary>
        public double OutOfBagConcordance { get; private set; }

        /// <summary>Unique event times from training data.</summary>
        public double[]? UniqueTimes { get; private set; }

        /// <summary>Number of features seen during fitting.</summary>
        public int FeatureCount => _featureCount;

        /// <summary>Whether the model has been fitted.</summary>
        public bool IsFitted => _handle != IntPtr.Zero &&
                                NativeMethods.aorsf_forest_is_fitted(_handle) != 0;

        /// <summary>
        /// Fit the survival forest to training data.
        /// </summary>
        /// <param name="features">Training feature matrix [n_samples, n_features].</param>
        /// <param name="outcomes">Survival outcomes [n_samples, 2] with columns (time, status).</param>
        /// <param name="sampleWeights">Optional sample weights [n_samples].</param>
        public void Fit(double[,] features, double[,] outcomes, double[]? sampleWeights = null)
        {
            if (features == null) throw new ArgumentNullException(nameof(features));
            if (outcomes == null) throw new ArgumentNullException(nameof(outcomes));

            int sampleCount = features.GetLength(0);
            int featureCount = features.GetLength(1);

            if (outcomes.GetLength(0) != sampleCount)
                throw new ArgumentException("Features and outcomes must have same number of samples.");
            if (outcomes.GetLength(1) != 2)
                throw new ArgumentException("Outcomes must have 2 columns: (time, status).");

            // Convert to flat row-major arrays
            double[] featuresFlat = new double[sampleCount * featureCount];
            for (int i = 0; i < sampleCount; i++)
                for (int j = 0; j < featureCount; j++)
                    featuresFlat[i * featureCount + j] = features[i, j];

            double[] outcomesFlat = new double[sampleCount * 2];
            for (int i = 0; i < sampleCount; i++)
            {
                outcomesFlat[i * 2 + 0] = outcomes[i, 0];  // time
                outcomesFlat[i * 2 + 1] = outcomes[i, 1];  // status
            }

            // Initialize configuration
            var config = new NativeMethods.AorsfConfig();
            NativeMethods.aorsf_config_init(ref config, (int)TreeType.Survival);

            config.NTree = TreeCount;
            config.Mtry = MaxFeatures ?? 0;
            config.LeafMinObs = MinSamplesLeaf;
            config.LeafMinEvents = MinEventsLeaf;
            config.SplitMinObs = MinSamplesSplit;
            config.SplitMinEvents = MinEventsSplit;
            config.NSplit = SplitCount;
            config.SplitRule = (int)SplitRule;
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
                outcomesFlat, 2, sampleWeights, 0);
            AorsfException.ThrowIfError(err);

            try
            {
                // Fit model
                err = NativeMethods.aorsf_forest_fit(_handle, dataHandle);
                AorsfException.ThrowIfError(err);

                _featureCount = NativeMethods.aorsf_forest_get_n_features(_handle);

                // Get OOB concordance
                err = NativeMethods.aorsf_forest_get_oob_error(_handle, out double oobError);
                if (err == NativeMethods.AORSF_SUCCESS)
                    OutOfBagConcordance = oobError;

                // Get unique times
                int nTimes = 0;
                err = NativeMethods.aorsf_forest_get_unique_times(_handle, null, ref nTimes);
                if (err == NativeMethods.AORSF_SUCCESS && nTimes > 0)
                {
                    UniqueTimes = new double[nTimes];
                    err = NativeMethods.aorsf_forest_get_unique_times(_handle, UniqueTimes, ref nTimes);
                    if (err != NativeMethods.AORSF_SUCCESS)
                        UniqueTimes = null;
                }

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
        /// Predict risk scores for samples. Higher values indicate higher risk.
        /// </summary>
        /// <param name="features">Feature matrix [n_samples, n_features].</param>
        /// <returns>Risk scores [n_samples].</returns>
        public double[] Predict(double[,] features)
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
                (int)PredictionType.Risk, predictions, sampleCount);
            AorsfException.ThrowIfError(err);

            return predictions;
        }

        /// <summary>
        /// Predict mortality (higher = higher probability of death).
        /// </summary>
        /// <param name="features">Feature matrix [n_samples, n_features].</param>
        /// <returns>Mortality scores [n_samples].</returns>
        public double[] PredictMortality(double[,] features)
        {
            if (!IsFitted)
                throw new InvalidOperationException("Model not fitted. Call Fit() first.");

            int sampleCount = features.GetLength(0);
            int featureCount = features.GetLength(1);

            if (featureCount != _featureCount)
                throw new ArgumentException($"Expected {_featureCount} features, got {featureCount}.");

            double[] featuresFlat = new double[sampleCount * featureCount];
            for (int i = 0; i < sampleCount; i++)
                for (int j = 0; j < featureCount; j++)
                    featuresFlat[i * featureCount + j] = features[i, j];

            double[] predictions = new double[sampleCount];
            int err = NativeMethods.aorsf_forest_predict(
                _handle, featuresFlat, sampleCount, featureCount,
                (int)PredictionType.Mortality, predictions, sampleCount);
            AorsfException.ThrowIfError(err);

            return predictions;
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

        ~ObliqueForestSurvival()
        {
            Dispose(false);
        }
    }
}
