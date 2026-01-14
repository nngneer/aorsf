"""
Penguin Classification Example with pyaorsf

This example demonstrates fitting an oblique random forest classifier
to predict penguin species using the Palmer Penguins dataset.

This is the Python equivalent of the R example from the aorsf README:

    penguin_fit <- orsf(data = penguins_orsf,
                        n_tree = 5,
                        formula = species ~ .)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Import pyaorsf
from pyaorsf import ObliqueForestClassifier


def load_penguins():
    """Load the Palmer Penguins dataset."""
    data_path = Path(__file__).parent / "penguins.csv"
    df = pd.read_csv(data_path)
    return df


def prepare_data(df):
    """
    Prepare penguin data for modeling.

    Converts categorical variables to numeric and separates features from target.
    """
    # Target variable
    y = df['species'].astype('category').cat.codes.values
    species_names = df['species'].astype('category').cat.categories.tolist()

    # Feature columns (everything except species)
    feature_cols = [col for col in df.columns if col != 'species']

    # Convert categorical features to numeric
    X_df = df[feature_cols].copy()

    # Encode categorical columns
    for col in X_df.select_dtypes(include=['object']).columns:
        X_df[col] = X_df[col].astype('category').cat.codes

    X = X_df.values.astype(np.float64)
    feature_names = feature_cols

    return X, y, feature_names, species_names


def main():
    print("=" * 60)
    print("Penguin Classification with Oblique Random Forest")
    print("=" * 60)

    # Load data
    print("\nLoading Palmer Penguins dataset...")
    df = load_penguins()
    print(f"Dataset shape: {df.shape}")
    print(f"Species: {df['species'].unique().tolist()}")

    # Prepare data
    X, y, feature_names, species_names = prepare_data(df)
    print(f"\nFeatures: {feature_names}")
    print(f"N observations: {len(y)}")
    print(f"N classes: {len(species_names)}")
    print(f"N predictors: {X.shape[1]}")

    # Fit oblique classification random forest
    # This is equivalent to:
    #   penguin_fit <- orsf(data = penguins_orsf, n_tree = 5, formula = species ~ .)
    print("\n" + "-" * 60)
    print("Fitting Oblique Random Classification Forest...")
    print("-" * 60)

    clf = ObliqueForestClassifier(
        n_trees=5,
        mtry=3,  # N predictors per node (like R default sqrt(7) ~ 3)
        leaf_min_obs=5,
        importance='anova',  # Variable importance method
        random_state=42
    )
    clf.fit(X, y)

    # Print model summary (similar to R output)
    print("\n---------- Oblique random classification forest")
    print()
    print(f"     Linear combinations: Accelerated Logistic regression")
    print(f"          N observations: {len(y)}")
    print(f"               N classes: {len(species_names)}")
    print(f"                 N trees: {clf.n_trees}")
    print(f"      N predictors total: {X.shape[1]}")
    print(f"   N predictors per node: {clf.mtry if clf.mtry else int(np.sqrt(X.shape[1]))}")
    print(f" Min observations in leaf: {clf.leaf_min_obs}")

    # Get true OOB score (AUC-ROC) from the fitted model
    if clf.oob_score_ is not None:
        print(f"          OOB stat value: {clf.oob_score_:.2f}")
        print(f"           OOB stat type: AUC-ROC")
    else:
        # Fallback to training accuracy if OOB not available
        train_accuracy = clf.score(X, y)
        print(f"          OOB stat value: {train_accuracy:.2f}")
        print(f"           OOB stat type: Accuracy (training)")
    print(f"     Variable importance: anova")
    print()
    print("-" * 41)

    # Show predictions
    print("\n" + "=" * 60)
    print("Predictions")
    print("=" * 60)

    # Predict on first 5 samples
    predictions = clf.predict(X[:5])
    probabilities = clf.predict_proba(X[:5])

    print("\nFirst 5 predictions:")
    print(f"{'Actual':<12} {'Predicted':<12} {'Probabilities'}")
    print("-" * 50)
    for i in range(5):
        actual = species_names[y[i]]
        predicted = species_names[predictions[i]]
        probs = ", ".join([f"{p:.2f}" for p in probabilities[i]])
        print(f"{actual:<12} {predicted:<12} [{probs}]")

    # Variable importance
    if clf.feature_importances_ is not None:
        print("\n" + "=" * 60)
        print("Variable Importance (ANOVA)")
        print("=" * 60)

        # Sort by importance
        importance_order = np.argsort(clf.feature_importances_)[::-1]
        print(f"\n{'Feature':<20} {'Importance':>12}")
        print("-" * 35)
        for idx in importance_order:
            print(f"{feature_names[idx]:<20} {clf.feature_importances_[idx]:>12.4f}")

    # Cross-validation
    print("\n" + "=" * 60)
    print("Cross-Validation")
    print("=" * 60)

    from sklearn.model_selection import cross_val_score

    clf_cv = ObliqueForestClassifier(n_trees=50, random_state=42)
    scores = cross_val_score(clf_cv, X, y, cv=5, scoring='accuracy')
    print(f"\n5-Fold CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

    # Full model with more trees
    print("\n" + "=" * 60)
    print("Full Model (100 trees)")
    print("=" * 60)

    clf_full = ObliqueForestClassifier(
        n_trees=100,
        importance='negate',
        random_state=42
    )
    clf_full.fit(X, y)

    print(f"\nTraining Accuracy: {clf_full.score(X, y):.3f}")

    if clf_full.feature_importances_ is not None:
        print("\nVariable Importance (Negation):")
        importance_order = np.argsort(clf_full.feature_importances_)[::-1]
        for idx in importance_order[:5]:
            print(f"  {feature_names[idx]:<20} {clf_full.feature_importances_[idx]:.4f}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
