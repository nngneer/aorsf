"""
Custom callbacks example.

This example demonstrates:
- Using custom linear combination functions
- Using custom out-of-bag evaluation functions
- Implementing a simple PCA-based linear combination
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from pyaorsf import ObliqueForestClassifier


def pca_lincomb(x, y, w):
    """
    PCA-based linear combination.

    Uses the first principal component as the split direction.
    """
    # Center the data
    x_centered = x - np.mean(x, axis=0)

    # Compute covariance matrix
    cov = np.cov(x_centered.T)

    # Get first eigenvector (largest eigenvalue)
    if cov.ndim == 0:  # Single feature
        return np.array([1.0])

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigenvectors are sorted by eigenvalue ascending, get the last one
    coefs = eigenvectors[:, -1]

    return coefs


def logistic_lincomb(x, y, w):
    """
    Logistic regression-based linear combination.

    Fits logistic regression and uses coefficients as split direction.
    """
    # Ensure y is 1D
    y_flat = y.ravel() if y.ndim > 1 else y

    # Handle case with only one class
    if len(np.unique(y_flat)) < 2:
        coefs = np.zeros(x.shape[1])
        coefs[0] = 1.0
        return coefs

    try:
        # Fit logistic regression
        lr = LogisticRegression(
            C=1.0,
            solver='lbfgs',
            max_iter=100,
            random_state=42
        )
        lr.fit(x, y_flat, sample_weight=w)
        return lr.coef_.ravel()
    except Exception:
        # Fallback to first feature
        coefs = np.zeros(x.shape[1])
        coefs[0] = 1.0
        return coefs


def random_sparse_lincomb(x, y, w):
    """
    Random sparse linear combination.

    Selects a random subset of features with random coefficients.
    """
    n_features = x.shape[1]
    n_select = max(1, n_features // 3)

    # Random selection
    selected = np.random.choice(n_features, n_select, replace=False)

    coefs = np.zeros(n_features)
    coefs[selected] = np.random.randn(n_select)

    # Normalize
    norm = np.linalg.norm(coefs)
    if norm > 0:
        coefs /= norm

    return coefs


def custom_accuracy_eval(y, w, p):
    """
    Custom accuracy evaluation for OOB.

    Parameters
    ----------
    y : ndarray of shape (n_samples, n_targets)
        True labels.
    w : ndarray of shape (n_samples,)
        Sample weights.
    p : ndarray
        Predicted probabilities.

    Returns
    -------
    score : float
        Weighted accuracy.
    """
    y_true = y[:, 0] if y.ndim > 1 else y

    # Convert probabilities to predictions
    if p.ndim > 1:
        y_pred = np.argmax(p, axis=1)
    else:
        y_pred = (p > 0.5).astype(float)

    # Weighted accuracy
    correct = (y_true == y_pred).astype(float)
    return np.average(correct, weights=w)


def custom_brier_eval(y, w, p):
    """
    Custom Brier score evaluation for OOB.

    Returns negative Brier score (so higher is better).
    """
    y_true = y[:, 0] if y.ndim > 1 else y

    # Get probability for class 1
    if p.ndim > 1:
        p_class1 = p[:, 1] if p.shape[1] > 1 else p[:, 0]
    else:
        p_class1 = p

    # Brier score (negative so higher is better)
    brier = np.average((y_true - p_class1) ** 2, weights=w)
    return -brier


def main():
    # Generate sample data
    print("Generating classification data...")
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")

    # Standard oblique forest
    print("\n1. Standard ObliqueForestClassifier:")
    clf_standard = ObliqueForestClassifier(
        n_trees=50,
        random_state=42
    )
    clf_standard.fit(X_train, y_train)
    print(f"   Test accuracy: {clf_standard.score(X_test, y_test):.4f}")

    # PCA-based linear combination
    print("\n2. PCA-based linear combination:")
    clf_pca = ObliqueForestClassifier(
        n_trees=50,
        lincomb_func=pca_lincomb,
        random_state=42
    )
    clf_pca.fit(X_train, y_train)
    print(f"   Test accuracy: {clf_pca.score(X_test, y_test):.4f}")

    # Logistic regression-based linear combination
    print("\n3. Logistic regression-based linear combination:")
    clf_logistic = ObliqueForestClassifier(
        n_trees=50,
        lincomb_func=logistic_lincomb,
        random_state=42
    )
    clf_logistic.fit(X_train, y_train)
    print(f"   Test accuracy: {clf_logistic.score(X_test, y_test):.4f}")

    # Random sparse linear combination
    print("\n4. Random sparse linear combination:")
    clf_sparse = ObliqueForestClassifier(
        n_trees=50,
        lincomb_func=random_sparse_lincomb,
        random_state=42
    )
    clf_sparse.fit(X_train, y_train)
    print(f"   Test accuracy: {clf_sparse.score(X_test, y_test):.4f}")

    # Custom OOB evaluation with accuracy
    print("\n5. Custom OOB evaluation (accuracy):")
    clf_oob_acc = ObliqueForestClassifier(
        n_trees=50,
        oobag_eval_func=custom_accuracy_eval,
        random_state=42
    )
    clf_oob_acc.fit(X_train, y_train)
    print(f"   Test accuracy: {clf_oob_acc.score(X_test, y_test):.4f}")

    # Custom OOB evaluation with Brier score
    print("\n6. Custom OOB evaluation (Brier score):")
    clf_oob_brier = ObliqueForestClassifier(
        n_trees=50,
        oobag_eval_func=custom_brier_eval,
        random_state=42
    )
    clf_oob_brier.fit(X_train, y_train)
    print(f"   Test accuracy: {clf_oob_brier.score(X_test, y_test):.4f}")

    # Combined: custom lincomb + custom eval
    print("\n7. Combined: Logistic lincomb + Brier eval:")
    clf_combined = ObliqueForestClassifier(
        n_trees=50,
        lincomb_func=logistic_lincomb,
        oobag_eval_func=custom_brier_eval,
        random_state=42
    )
    clf_combined.fit(X_train, y_train)
    print(f"   Test accuracy: {clf_combined.score(X_test, y_test):.4f}")


if __name__ == "__main__":
    main()
