"""
Regression example with ObliqueForestRegressor.

This example demonstrates:
- Training a regressor on the California housing dataset
- Making predictions and evaluating R^2 score
- Using cross-validation
- Comparing with scikit-learn's RandomForestRegressor
"""

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error

from pyaorsf import ObliqueForestRegressor


def main():
    # Load California housing dataset
    print("Loading California housing dataset...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = data.feature_names

    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train ObliqueForestRegressor
    print("\nTraining ObliqueForestRegressor...")
    orf = ObliqueForestRegressor(
        n_trees=100,
        mtry=3,
        leaf_min_obs=5,
        n_threads=4,
        random_state=42
    )
    orf.fit(X_train, y_train)

    # Evaluate
    train_r2 = orf.score(X_train, y_train)
    test_r2 = orf.score(X_test, y_test)
    print(f"Train R^2: {train_r2:.4f}")
    print(f"Test R^2: {test_r2:.4f}")

    # Predictions
    predictions = orf.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    print(f"\nError metrics:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")

    print(f"\nSample predictions (first 5):")
    print(f"  True values:  {y_test[:5].round(3)}")
    print(f"  Predictions:  {predictions[:5].round(3)}")

    # Cross-validation comparison with RandomForest
    print("\nCross-validation comparison (5-fold R^2):")

    # Use smaller subset for faster CV
    X_sub, y_sub = X[:5000], y[:5000]

    orf_cv = ObliqueForestRegressor(n_trees=50, random_state=42)
    orf_scores = cross_val_score(orf_cv, X_sub, y_sub, cv=5, scoring='r2')
    print(f"  ObliqueForest: {orf_scores.mean():.4f} (+/- {orf_scores.std():.4f})")

    rf_cv = RandomForestRegressor(n_estimators=50, random_state=42)
    rf_scores = cross_val_score(rf_cv, X_sub, y_sub, cv=5, scoring='r2')
    print(f"  RandomForest:  {rf_scores.mean():.4f} (+/- {rf_scores.std():.4f})")

    # Using in a pipeline
    print("\nUsing in a Pipeline with StandardScaler:")
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reg', ObliqueForestRegressor(n_trees=100, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    pipe_r2 = pipe.score(X_test, y_test)
    print(f"  Pipeline R^2: {pipe_r2:.4f}")


if __name__ == "__main__":
    main()
