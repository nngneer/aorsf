namespace Aorsf
{
    public enum TreeType
    {
        Classification = 1,
        Regression = 2,
        Survival = 3
    }

    public enum ImportanceType
    {
        None = 0,
        Negate = 1,
        Permute = 2,
        Anova = 3
    }

    public enum SplitRule
    {
        LogRank = 1,
        Concordance = 2,
        Gini = 3,
        Variance = 4
    }

    public enum LinearComboMethod
    {
        Glm = 1,
        Random = 2,
        GlmNet = 3
    }

    public enum PredictionType
    {
        Risk = 1,
        Survival = 2,
        CumulativeHazard = 3,
        Mortality = 4,
        Mean = 5,
        Probability = 6,
        Class = 7
    }

    /// <summary>
    /// Serialization format for saving/loading models.
    /// </summary>
    public enum SerializationFormat
    {
        /// <summary>
        /// Binary format - fast and compact.
        /// </summary>
        Binary = 0,

        /// <summary>
        /// JSON format - human-readable, good for debugging.
        /// </summary>
        Json = 1
    }

    /// <summary>
    /// Flags for controlling what data is included in serialization.
    /// </summary>
    [Flags]
    public enum SerializationFlags : uint
    {
        /// <summary>
        /// No extra data included.
        /// </summary>
        None = 0,

        /// <summary>
        /// Include variable importance data.
        /// </summary>
        IncludeImportance = 0x01,

        /// <summary>
        /// Include out-of-bag data (for partial dependence, etc.).
        /// </summary>
        IncludeOobData = 0x02,

        /// <summary>
        /// Include metadata (feature names, etc.).
        /// </summary>
        IncludeMetadata = 0x04,

        /// <summary>
        /// Include all optional data.
        /// </summary>
        All = IncludeImportance | IncludeOobData | IncludeMetadata
    }
}
