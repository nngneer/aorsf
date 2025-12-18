"""
Survival analysis example with ObliqueForestSurvival.

This example demonstrates:
- Training a survival forest on simulated survival data
- Making risk predictions
- Evaluating with the concordance index (C-index)
- Predicting survival probabilities
"""

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from pyaorsf import ObliqueForestSurvival
from pyaorsf.utils import concordance_index


def generate_survival_data(n_samples=500, n_features=10, random_state=42):
    """
    Generate simulated survival data.

    The survival time is generated based on a Cox model where the hazard
    depends on a linear combination of features.
    """
    np.random.seed(random_state)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # True coefficients (only first 3 features matter)
    true_coef = np.zeros(n_features)
    true_coef[:3] = [0.5, -0.3, 0.4]

    # Linear predictor
    lp = X @ true_coef

    # Generate survival times (exponential baseline hazard)
    scale = np.exp(-lp) * 10  # Higher risk = shorter time
    time = np.random.exponential(scale)

    # Generate censoring times (uniform censoring)
    censor_time = np.random.uniform(0, 30, n_samples)

    # Observed time and status
    observed_time = np.minimum(time, censor_time)
    status = (time <= censor_time).astype(float)

    # Create y array: [time, status]
    y = np.column_stack([observed_time, status])

    return X, y, true_coef


def main():
    # Generate survival data
    print("Generating simulated survival data...")
    X, y, true_coef = generate_survival_data(n_samples=500, n_features=10)

    time, status = y[:, 0], y[:, 1]
    print(f"Dataset shape: {X.shape}")
    print(f"Number of events: {int(status.sum())} ({100*status.mean():.1f}%)")
    print(f"Time range: [{time.min():.2f}, {time.max():.2f}]")
    print(f"True coefficients: {true_coef[:5]}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train ObliqueForestSurvival
    print("\nTraining ObliqueForestSurvival...")
    surv = ObliqueForestSurvival(
        n_trees=100,
        mtry=3,
        leaf_min_obs=10,
        leaf_min_events=3,
        n_threads=4,
        random_state=42
    )
    surv.fit(X_train, y_train)

    # Evaluate with C-index
    train_cindex = surv.score(X_train, y_train)
    test_cindex = surv.score(X_test, y_test)
    print(f"Train C-index: {train_cindex:.4f}")
    print(f"Test C-index: {test_cindex:.4f}")

    # Risk predictions
    risk = surv.predict(X_test)
    print(f"\nRisk score range: [{risk.min():.3f}, {risk.max():.3f}]")
    print(f"Sample risk scores (first 5): {risk[:5].round(3)}")

    # Survival probability predictions
    survival = surv.predict_survival(X_test)
    print(f"\nSurvival probability shape: {survival.shape}")
    print(f"Sample survival probs (first 5): {survival[:5].ravel().round(3)}")

    # Cross-validation
    print("\nCross-validation (5-fold C-index):")
    surv_cv = ObliqueForestSurvival(n_trees=50, random_state=42)
    scores = cross_val_score(surv_cv, X, y, cv=5)
    print(f"  C-index: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # Using in a pipeline
    print("\nUsing in a Pipeline with StandardScaler:")
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('surv', ObliqueForestSurvival(n_trees=100, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    pipe_cindex = pipe.score(X_test, y_test)
    print(f"  Pipeline C-index: {pipe_cindex:.4f}")

    # Demonstrate relationship between risk and time
    print("\nRisk-time relationship (higher risk should correlate with shorter time):")
    time_test = y_test[:, 0]
    correlation = np.corrcoef(risk, time_test)[0, 1]
    print(f"  Correlation(risk, time): {correlation:.3f}")
    print("  (Negative correlation expected: higher risk -> shorter survival)")


if __name__ == "__main__":
    main()
