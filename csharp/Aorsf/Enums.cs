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
}
