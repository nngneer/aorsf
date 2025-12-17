"""
Scikit-learn compatible oblique random forest estimators.
"""

import numpy as np
from typing import Optional, Union, List

# Will import from C++ bindings once implemented
# from .._pyaorsf import ForestClassification, ForestRegression


class ObliqueForestClassifier:
    """
    Oblique Random Forest Classifier.

    An oblique random forest uses linear combinations of features for splitting,
    rather than single features. This can result in more efficient trees and
    smoother decision boundaries.

    Parameters
    ----------
    n_trees : int, default=500
        Number of trees in the forest.
    mtry : int, default=None
        Number of features to consider at each split. If None, uses sqrt(n_features).
    leaf_min_obs : int, default=5
        Minimum number of observations in a leaf node.
    split_min_obs : int, default=10
        Minimum number of observations required to attempt a split.
    split_min_stat : float, default=None
        Minimum improvement in split statistic to make a split.
    split_rule : str, default='gini'
        Split rule to use. One of 'gini' or 'cstat'.
    n_threads : int, default=1
        Number of threads to use for parallel processing. Use 0 for all available.
    random_state : int, default=None
        Random seed for reproducibility.
    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        Feature importances computed using negation method.
    n_classes_ : int
        Number of classes.
    classes_ : ndarray
        Unique class labels.
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> from pyaorsf import ObliqueForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=10)
    >>> clf = ObliqueForestClassifier(n_trees=10)
    >>> clf.fit(X, y)
    >>> clf.predict(X[:5])
    """

    def __init__(
        self,
        n_trees: int = 500,
        mtry: Optional[int] = None,
        leaf_min_obs: int = 5,
        split_min_obs: int = 10,
        split_min_stat: Optional[float] = None,
        split_rule: str = 'gini',
        n_threads: int = 1,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        self.n_trees = n_trees
        self.mtry = mtry
        self.leaf_min_obs = leaf_min_obs
        self.split_min_obs = split_min_obs
        self.split_min_stat = split_min_stat
        self.split_rule = split_rule
        self.n_threads = n_threads
        self.random_state = random_state
        self.verbose = verbose

        # Attributes set during fit
        self._forest = None
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_in_ = None
        self.feature_importances_ = None

    def fit(self, X, y, sample_weight=None):
        """
        Fit the oblique random forest classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        self : ObliqueForestClassifier
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Convert y to integer labels
        y_encoded = np.searchsorted(self.classes_, y)

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)

        # TODO: Call C++ forest training once bindings are implemented
        # self._forest = ForestClassification(...)
        # self._forest.train(X, y_encoded, sample_weight, ...)

        # Placeholder for feature importances
        self.feature_importances_ = np.zeros(self.n_features_in_)

        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        X = np.asarray(X, dtype=np.float64)

        # TODO: Call C++ forest prediction once bindings are implemented
        # return self._forest.predict_proba(X)

        # Placeholder
        n_samples = X.shape[0]
        return np.ones((n_samples, self.n_classes_)) / self.n_classes_

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy.
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'n_trees': self.n_trees,
            'mtry': self.mtry,
            'leaf_min_obs': self.leaf_min_obs,
            'split_min_obs': self.split_min_obs,
            'split_min_stat': self.split_min_stat,
            'split_rule': self.split_rule,
            'n_threads': self.n_threads,
            'random_state': self.random_state,
            'verbose': self.verbose,
        }

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class ObliqueForestRegressor:
    """
    Oblique Random Forest Regressor.

    An oblique random forest uses linear combinations of features for splitting,
    rather than single features. This can result in more efficient trees and
    smoother predictions.

    Parameters
    ----------
    n_trees : int, default=500
        Number of trees in the forest.
    mtry : int, default=None
        Number of features to consider at each split. If None, uses n_features/3.
    leaf_min_obs : int, default=5
        Minimum number of observations in a leaf node.
    split_min_obs : int, default=10
        Minimum number of observations required to attempt a split.
    split_min_stat : float, default=None
        Minimum improvement in split statistic to make a split.
    n_threads : int, default=1
        Number of threads to use for parallel processing. Use 0 for all available.
    random_state : int, default=None
        Random seed for reproducibility.
    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        Feature importances computed using negation method.
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> from pyaorsf import ObliqueForestRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=10)
    >>> reg = ObliqueForestRegressor(n_trees=10)
    >>> reg.fit(X, y)
    >>> reg.predict(X[:5])
    """

    def __init__(
        self,
        n_trees: int = 500,
        mtry: Optional[int] = None,
        leaf_min_obs: int = 5,
        split_min_obs: int = 10,
        split_min_stat: Optional[float] = None,
        n_threads: int = 1,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        self.n_trees = n_trees
        self.mtry = mtry
        self.leaf_min_obs = leaf_min_obs
        self.split_min_obs = split_min_obs
        self.split_min_stat = split_min_stat
        self.n_threads = n_threads
        self.random_state = random_state
        self.verbose = verbose

        # Attributes set during fit
        self._forest = None
        self.n_features_in_ = None
        self.feature_importances_ = None

    def fit(self, X, y, sample_weight=None):
        """
        Fit the oblique random forest regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        self : ObliqueForestRegressor
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        self.n_features_in_ = X.shape[1]

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)

        # TODO: Call C++ forest training once bindings are implemented
        # self._forest = ForestRegression(...)
        # self._forest.train(X, y, sample_weight, ...)

        # Placeholder for feature importances
        self.feature_importances_ = np.zeros(self.n_features_in_)

        return self

    def predict(self, X):
        """
        Predict target values for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        X = np.asarray(X, dtype=np.float64)

        # TODO: Call C++ forest prediction once bindings are implemented
        # return self._forest.predict(X)

        # Placeholder
        return np.zeros(X.shape[0])

    def score(self, X, y, sample_weight=None):
        """
        Return the coefficient of determination (R^2) of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            R^2 score.
        """
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X), sample_weight=sample_weight)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'n_trees': self.n_trees,
            'mtry': self.mtry,
            'leaf_min_obs': self.leaf_min_obs,
            'split_min_obs': self.split_min_obs,
            'split_min_stat': self.split_min_stat,
            'n_threads': self.n_threads,
            'random_state': self.random_state,
            'verbose': self.verbose,
        }

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
