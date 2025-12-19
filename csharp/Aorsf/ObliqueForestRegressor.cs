using System;
using System.Runtime.InteropServices;
using Aorsf.Native;

namespace Aorsf
{
    /// <summary>
    /// Oblique random forest regressor using linear combination splits.
    /// </summary>
    public class ObliqueForestRegressor : IDisposable
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

        /// <summary>Out-of-bag R-squared score.</summary>
        public double OutOfBagR2 { get; private set; }

        /// <summary>Number of features seen during fitting.</summary>
        public int FeatureCount => _featureCount;

        /// <summary>Whether the model has been fitted.</summary>
        public bool IsFitted => _handle != IntPtr.Zero &&
                                NativeMethods.aorsf_forest_is_fitted(_handle) != 0;

        /// <summary>
        /// Fit the regressor to training data.
        /// </summary>
        /// <param name="features">Training feature matrix [n_samples, n_features].</param>
        /// <param name="targets">Target values [n_samples].</param>
        /// <param name="sampleWeights">Optional sample weights [n_samples].</param>
        public void Fit(double[,] features, double[] targets, double[]? sampleWeights = null)
        {
            if (features == null) throw new ArgumentNullException(nameof(features));
            if (targets == null) throw new ArgumentNullException(nameof(targets));

            int sampleCount = features.GetLength(0);
            int featureCount = features.GetLength(1);

            if (targets.Length != sampleCount)
                throw new ArgumentException("Features and targets must have same number of samples.");

            // Convert to flat row-major array
            double[] featuresFlat = new double[sampleCount * featureCount];
            for (int i = 0; i < sampleCount; i++)
                for (int j = 0; j < featureCount; j++)
                    featuresFlat[i * featureCount + j] = features[i, j];

            // Initialize configuration
            var config = new NativeMethods.AorsfConfig();
            NativeMethods.aorsf_config_init(ref config, (int)TreeType.Regression);

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
                targets, 1, sampleWeights, 0);
            AorsfException.ThrowIfError(err);

            try
            {
                // Fit model
                err = NativeMethods.aorsf_forest_fit(_handle, dataHandle);
                AorsfException.ThrowIfError(err);

                _featureCount = NativeMethods.aorsf_forest_get_n_features(_handle);

                // Get OOB R-squared
                err = NativeMethods.aorsf_forest_get_oob_error(_handle, out double oobError);
                if (err == NativeMethods.AORSF_SUCCESS)
                    OutOfBagR2 = oobError;

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
        /// Predict target values for samples.
        /// </summary>
        /// <param name="features">Feature matrix [n_samples, n_features].</param>
        /// <returns>Predicted values [n_samples].</returns>
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
                (int)PredictionType.Mean, predictions, sampleCount);
            AorsfException.ThrowIfError(err);

            return predictions;
        }

        /// <summary>
        /// Calculate R-squared score on test data.
        /// </summary>
        /// <param name="features">Test feature matrix [n_samples, n_features].</param>
        /// <param name="targets">True target values [n_samples].</param>
        /// <returns>R-squared score.</returns>
        public double Score(double[,] features, double[] targets)
        {
            var predictions = Predict(features);

            // Calculate mean of actual values
            double mean = 0;
            for (int i = 0; i < targets.Length; i++)
                mean += targets[i];
            mean /= targets.Length;

            // Calculate SS_res and SS_tot
            double ssRes = 0, ssTot = 0;
            for (int i = 0; i < targets.Length; i++)
            {
                double diff = targets[i] - predictions[i];
                ssRes += diff * diff;
                double diffMean = targets[i] - mean;
                ssTot += diffMean * diffMean;
            }

            return 1.0 - (ssRes / ssTot);
        }

        /// <summary>
        /// Save the fitted model to a file.
        /// </summary>
        /// <param name="filepath">Path to save the model.</param>
        /// <param name="format">Serialization format (Binary or Json).</param>
        /// <param name="flags">Optional flags to control what data is included.</param>
        public void Save(string filepath, SerializationFormat format = SerializationFormat.Binary,
                         SerializationFlags flags = SerializationFlags.IncludeImportance)
        {
            if (!IsFitted)
                throw new InvalidOperationException("Model not fitted. Call Fit() first.");

            int err = NativeMethods.aorsf_forest_save_file(
                _handle, filepath, (int)format, (uint)flags);
            AorsfException.ThrowIfError(err);
        }

        /// <summary>
        /// Save the fitted model to a byte array.
        /// </summary>
        /// <param name="format">Serialization format (Binary or Json).</param>
        /// <param name="flags">Optional flags to control what data is included.</param>
        /// <returns>Byte array containing the serialized model.</returns>
        public byte[] SaveToBytes(SerializationFormat format = SerializationFormat.Binary,
                                   SerializationFlags flags = SerializationFlags.IncludeImportance)
        {
            if (!IsFitted)
                throw new InvalidOperationException("Model not fitted. Call Fit() first.");

            int err = NativeMethods.aorsf_forest_get_save_size(
                _handle, (int)format, (uint)flags, out UIntPtr size);
            AorsfException.ThrowIfError(err);

            byte[] buffer = new byte[(int)size];
            err = NativeMethods.aorsf_forest_save(
                _handle, (int)format, (uint)flags, buffer, size, out UIntPtr written);
            AorsfException.ThrowIfError(err);

            if ((int)written != buffer.Length)
            {
                byte[] result = new byte[(int)written];
                Array.Copy(buffer, result, (int)written);
                return result;
            }
            return buffer;
        }

        /// <summary>
        /// Load a model from a file.
        /// </summary>
        /// <param name="filepath">Path to the saved model.</param>
        /// <returns>Loaded regressor.</returns>
        public static ObliqueForestRegressor Load(string filepath)
        {
            int err = NativeMethods.aorsf_forest_load_file(out IntPtr handle, filepath);
            AorsfException.ThrowIfError(err);

            var regressor = new ObliqueForestRegressor();
            regressor._handle = handle;
            regressor._featureCount = NativeMethods.aorsf_forest_get_n_features(handle);

            // Try to get importance if available
            regressor.FeatureImportances = new double[regressor._featureCount];
            err = NativeMethods.aorsf_forest_get_importance(
                handle, regressor.FeatureImportances, regressor._featureCount);
            if (err != NativeMethods.AORSF_SUCCESS)
                regressor.FeatureImportances = null;

            // Try to get OOB score
            err = NativeMethods.aorsf_forest_get_oob_error(handle, out double oobError);
            if (err == NativeMethods.AORSF_SUCCESS)
                regressor.OutOfBagR2 = oobError;

            return regressor;
        }

        /// <summary>
        /// Load a model from a byte array.
        /// </summary>
        /// <param name="data">Byte array containing the serialized model.</param>
        /// <returns>Loaded regressor.</returns>
        public static ObliqueForestRegressor LoadFromBytes(byte[] data)
        {
            int err = NativeMethods.aorsf_forest_load(out IntPtr handle, data, (UIntPtr)data.Length);
            AorsfException.ThrowIfError(err);

            var regressor = new ObliqueForestRegressor();
            regressor._handle = handle;
            regressor._featureCount = NativeMethods.aorsf_forest_get_n_features(handle);

            // Try to get importance if available
            regressor.FeatureImportances = new double[regressor._featureCount];
            err = NativeMethods.aorsf_forest_get_importance(
                handle, regressor.FeatureImportances, regressor._featureCount);
            if (err != NativeMethods.AORSF_SUCCESS)
                regressor.FeatureImportances = null;

            // Try to get OOB score
            err = NativeMethods.aorsf_forest_get_oob_error(handle, out double oobError);
            if (err == NativeMethods.AORSF_SUCCESS)
                regressor.OutOfBagR2 = oobError;

            return regressor;
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

        ~ObliqueForestRegressor()
        {
            Dispose(false);
        }
    }
}
