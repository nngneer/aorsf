"""
Classification example with ObliqueForestClassifier.

This example demonstrates:
- Training a classifier on the breast cancer dataset
- Making predictions and evaluating accuracy
- Using cross-validation
- Comparing with scikit-learn's RandomForestClassifier
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from pyaorsf import ObliqueForestClassifier


def main():
    # Load breast cancer dataset
    print("Loading breast cancer dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Class distribution: {np.bincount(y)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train ObliqueForestClassifier
    print("\nTraining ObliqueForestClassifier...")
    orf = ObliqueForestClassifier(
        n_trees=100,
        mtry=5,
        leaf_min_obs=5,
        n_threads=4,
        random_state=42
    )
    orf.fit(X_train, y_train)

    # Evaluate
    train_acc = orf.score(X_train, y_train)
    test_acc = orf.score(X_test, y_test)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    # Predictions and probabilities
    predictions = orf.predict(X_test)
    probabilities = orf.predict_proba(X_test)
    print(f"\nSample predictions (first 5):")
    print(f"  True labels:  {y_test[:5]}")
    print(f"  Predictions:  {predictions[:5]}")
    print(f"  Probabilities: {probabilities[:5].round(3)}")

    # Cross-validation comparison with RandomForest
    print("\nCross-validation comparison (5-fold):")

    orf_cv = ObliqueForestClassifier(n_trees=100, random_state=42)
    orf_scores = cross_val_score(orf_cv, X, y, cv=5)
    print(f"  ObliqueForest: {orf_scores.mean():.4f} (+/- {orf_scores.std():.4f})")

    rf_cv = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_scores = cross_val_score(rf_cv, X, y, cv=5)
    print(f"  RandomForest:  {rf_scores.mean():.4f} (+/- {rf_scores.std():.4f})")

    # Using in a pipeline
    print("\nUsing in a Pipeline with StandardScaler:")
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', ObliqueForestClassifier(n_trees=100, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    pipe_acc = pipe.score(X_test, y_test)
    print(f"  Pipeline accuracy: {pipe_acc:.4f}")


if __name__ == "__main__":
    main()
