"""
Scikit-learn compatible oblique random forest estimators.
"""

import numpy as np
from typing import Optional, Union, List

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from . import _pyaorsf


class ObliqueForestClassifier(BaseEstimator, ClassifierMixin):
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
    split_min_stat : float, default=0.0
        Minimum improvement in split statistic to make a split.
    split_rule : str, default='gini'
        Split rule to use. One of 'gini' or 'cstat'.
    split_max_cuts : int, default=5
        Maximum number of cut-points to evaluate.
    split_max_retry : int, default=3
        Maximum retries if split fails.
    sample_fraction : float, default=0.632
        Fraction of samples to use for each tree (bootstrap).
    sample_with_replacement : bool, default=True
        Whether to sample with replacement.
    lincomb_type : str, default='random'
        Type of linear combination. One of 'random', 'glm', 'glmnet'.
    lincomb_eps : float, default=1e-9
        Epsilon for linear combination convergence.
    lincomb_iter_max : int, default=20
        Maximum iterations for linear combination.
    lincomb_scale : bool, default=True
        Whether to scale features in linear combination.
    lincomb_alpha : float, default=0.5
        Alpha for elastic net (if using glmnet).
    n_threads : int, default=1
        Number of threads to use for parallel processing.
    random_state : int, default=None
        Random seed for reproducibility.
    verbose : int, default=0
        Verbosity level.
    importance : str, default=None
        Variable importance method. One of 'negate', 'permute', 'anova', or None.
        If None, variable importance is not computed.
    lincomb_func : callable, default=None
        Custom function for computing linear combinations at each node.
        Signature: lincomb_func(x, y, w) -> coefficients
        Where x is (n_obs, n_features), y is (n_obs, n_outcomes), w is (n_obs,).
        Returns coefficient array of shape (n_features,) or (n_features, 1).
    oobag_eval_func : callable, default=None
        Custom function for out-of-bag evaluation.
        Signature: oobag_eval_func(y, w, p) -> float
        Where y is (n_obs, n_outcomes), w is (n_obs,), p is (n_obs,).
        Returns scalar accuracy metric.

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        Feature importances. Only available if importance parameter is set.
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
        split_min_stat: float = 0.0,
        split_rule: str = 'gini',
        split_max_cuts: int = 5,
        split_max_retry: int = 3,
        sample_fraction: float = 0.632,
        sample_with_replacement: bool = True,
        lincomb_type: str = 'random',
        lincomb_eps: float = 1e-9,
        lincomb_iter_max: int = 20,
        lincomb_scale: bool = True,
        lincomb_alpha: float = 0.5,
        n_threads: int = 1,
        random_state: Optional[int] = None,
        verbose: int = 0,
        importance: Optional[str] = None,
        lincomb_func=None,
        oobag_eval_func=None,
    ):
        self.n_trees = n_trees
        self.mtry = mtry
        self.leaf_min_obs = leaf_min_obs
        self.split_min_obs = split_min_obs
        self.split_min_stat = split_min_stat
        self.split_rule = split_rule
        self.split_max_cuts = split_max_cuts
        self.split_max_retry = split_max_retry
        self.sample_fraction = sample_fraction
        self.sample_with_replacement = sample_with_replacement
        self.lincomb_type = lincomb_type
        self.lincomb_eps = lincomb_eps
        self.lincomb_iter_max = lincomb_iter_max
        self.lincomb_scale = lincomb_scale
        self.lincomb_alpha = lincomb_alpha
        self.n_threads = n_threads
        self.random_state = random_state
        self.verbose = verbose
        self.importance = importance
        self.lincomb_func = lincomb_func
        self.oobag_eval_func = oobag_eval_func

        # Attributes set during fit
        self._forest_data = None
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_in_ = None
        self.feature_importances_ = None

    def _get_split_rule(self):
        """Convert string split rule to enum value."""
        rules = {
            'gini': _pyaorsf.SplitRule.GINI.value,
            'cstat': _pyaorsf.SplitRule.CONCORD.value,
        }
        return rules.get(self.split_rule.lower(), _pyaorsf.SplitRule.GINI.value)

    def _get_lincomb_type(self):
        """Convert string lincomb type to enum value."""
        types = {
            'random': _pyaorsf.LinearCombo.RANDOM.value,
            'glm': _pyaorsf.LinearCombo.GLM.value,
            'glmnet': _pyaorsf.LinearCombo.GLMNET.value,
        }
        return types.get(self.lincomb_type.lower(), _pyaorsf.LinearCombo.RANDOM.value)

    def _get_vi_type(self):
        """Convert string importance type to enum value."""
        if self.importance is None:
            return _pyaorsf.VariableImportance.NONE.value
        types = {
            'negate': _pyaorsf.VariableImportance.NEGATE.value,
            'permute': _pyaorsf.VariableImportance.PERMUTE.value,
            'anova': _pyaorsf.VariableImportance.ANOVA.value,
        }
        vi_type = types.get(self.importance.lower())
        if vi_type is None:
            raise ValueError(
                f"importance must be one of 'negate', 'permute', 'anova', or None, "
                f"got '{self.importance}'"
            )
        return vi_type

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
        X, y = check_X_y(X, y, dtype=np.float64)

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Encode y to integer labels (0, 1, ..., n_classes-1)
        y_encoded = np.searchsorted(self.classes_, y).astype(np.float64)
        y_2d = y_encoded.reshape(-1, 1)

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)

        # Determine mtry
        mtry = self.mtry
        if mtry is None:
            mtry = max(1, int(np.sqrt(self.n_features_in_)))

        # Generate tree seeds
        rng = np.random.RandomState(self.random_state)
        tree_seeds = rng.randint(1, 2**31 - 1, size=self.n_trees).tolist()

        # Determine lincomb_type: use custom if lincomb_func is provided
        lincomb_type = self._get_lincomb_type()
        if self.lincomb_func is not None:
            lincomb_type = _pyaorsf.LinearCombo.CUSTOM.value

        # Determine oobag_eval_type: use custom if oobag_eval_func is provided
        oobag_eval_type = _pyaorsf.EvalType.NONE.value
        if self.oobag_eval_func is not None:
            oobag_eval_type = _pyaorsf.EvalType.CUSTOM.value

        # Call C++ fit function
        self._forest_data = _pyaorsf.fit_forest(
            x=X, y=y_2d, w=sample_weight,
            tree_type=_pyaorsf.TreeType.CLASSIFICATION.value,
            tree_seeds=tree_seeds,
            n_tree=self.n_trees,
            mtry=mtry,
            sample_with_replacement=self.sample_with_replacement,
            sample_fraction=self.sample_fraction,
            vi_type=self._get_vi_type(),
            vi_max_pvalue=0.01,
            leaf_min_events=1.0,
            leaf_min_obs=float(self.leaf_min_obs),
            split_rule=self._get_split_rule(),
            split_min_events=1.0,
            split_min_obs=float(self.split_min_obs),
            split_min_stat=self.split_min_stat,
            split_max_cuts=self.split_max_cuts,
            split_max_retry=self.split_max_retry,
            lincomb_type=lincomb_type,
            lincomb_eps=self.lincomb_eps,
            lincomb_iter_max=self.lincomb_iter_max,
            lincomb_scale=self.lincomb_scale,
            lincomb_alpha=self.lincomb_alpha,
            lincomb_df_target=0,
            lincomb_ties_method=0,
            pred_horizon=[],
            pred_type=_pyaorsf.PredType.PROBABILITY.value,
            oobag=True,
            oobag_eval_type=oobag_eval_type,
            oobag_eval_every=self.n_trees,
            n_thread=self.n_threads,
            verbose=self.verbose,
            lincomb_func=self.lincomb_func,
            oobag_func=self.oobag_eval_func
        )

        # Store OOB predictions for later use
        if 'oob_predictions' in self._forest_data:
            self._oob_predictions = self._forest_data['oob_predictions']
        else:
            self._oob_predictions = None

        # Extract feature importances if computed
        if 'importance' in self._forest_data:
            self.feature_importances_ = self._forest_data['importance']
        else:
            self.feature_importances_ = None

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
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64)

        # Use C++ predict function
        pred = _pyaorsf.predict_forest(
            x=X,
            cutpoint=self._forest_data['cutpoint'],
            child_left=self._forest_data['child_left'],
            coef_values=self._forest_data['coef_values'],
            coef_indices=self._forest_data['coef_indices'],
            leaf_summary=self._forest_data['leaf_summary'],
            tree_type=_pyaorsf.TreeType.CLASSIFICATION.value,
            n_class=self.n_classes_,
            aggregate=True
        )

        return pred

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
            'split_max_cuts': self.split_max_cuts,
            'split_max_retry': self.split_max_retry,
            'sample_fraction': self.sample_fraction,
            'sample_with_replacement': self.sample_with_replacement,
            'lincomb_type': self.lincomb_type,
            'lincomb_eps': self.lincomb_eps,
            'lincomb_iter_max': self.lincomb_iter_max,
            'lincomb_scale': self.lincomb_scale,
            'lincomb_alpha': self.lincomb_alpha,
            'n_threads': self.n_threads,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'importance': self.importance,
            'lincomb_func': self.lincomb_func,
            'oobag_eval_func': self.oobag_eval_func,
        }

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class ObliqueForestRegressor(BaseEstimator, RegressorMixin):
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
    split_min_stat : float, default=0.0
        Minimum improvement in split statistic to make a split.
    split_max_cuts : int, default=5
        Maximum number of cut-points to evaluate.
    split_max_retry : int, default=3
        Maximum retries if split fails.
    sample_fraction : float, default=0.632
        Fraction of samples to use for each tree (bootstrap).
    sample_with_replacement : bool, default=True
        Whether to sample with replacement.
    lincomb_type : str, default='random'
        Type of linear combination. One of 'random', 'glm', 'glmnet'.
    lincomb_eps : float, default=1e-9
        Epsilon for linear combination convergence.
    lincomb_iter_max : int, default=20
        Maximum iterations for linear combination.
    lincomb_scale : bool, default=True
        Whether to scale features in linear combination.
    lincomb_alpha : float, default=0.5
        Alpha for elastic net (if using glmnet).
    n_threads : int, default=1
        Number of threads to use for parallel processing.
    random_state : int, default=None
        Random seed for reproducibility.
    verbose : int, default=0
        Verbosity level.
    importance : str, default=None
        Variable importance method. One of 'negate', 'permute', 'anova', or None.
        If None, variable importance is not computed.
    lincomb_func : callable, default=None
        Custom function for computing linear combinations at each node.
        Signature: lincomb_func(x, y, w) -> coefficients
    oobag_eval_func : callable, default=None
        Custom function for out-of-bag evaluation.
        Signature: oobag_eval_func(y, w, p) -> float

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        Feature importances. Only available if importance parameter is set.
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
        split_min_stat: float = 0.0,
        split_max_cuts: int = 5,
        split_max_retry: int = 3,
        sample_fraction: float = 0.632,
        sample_with_replacement: bool = True,
        lincomb_type: str = 'random',
        lincomb_eps: float = 1e-9,
        lincomb_iter_max: int = 20,
        lincomb_scale: bool = True,
        lincomb_alpha: float = 0.5,
        n_threads: int = 1,
        random_state: Optional[int] = None,
        verbose: int = 0,
        importance: Optional[str] = None,
        lincomb_func=None,
        oobag_eval_func=None,
    ):
        self.n_trees = n_trees
        self.mtry = mtry
        self.leaf_min_obs = leaf_min_obs
        self.split_min_obs = split_min_obs
        self.split_min_stat = split_min_stat
        self.split_max_cuts = split_max_cuts
        self.split_max_retry = split_max_retry
        self.sample_fraction = sample_fraction
        self.sample_with_replacement = sample_with_replacement
        self.lincomb_type = lincomb_type
        self.lincomb_eps = lincomb_eps
        self.lincomb_iter_max = lincomb_iter_max
        self.lincomb_scale = lincomb_scale
        self.lincomb_alpha = lincomb_alpha
        self.n_threads = n_threads
        self.random_state = random_state
        self.verbose = verbose
        self.importance = importance
        self.lincomb_func = lincomb_func
        self.oobag_eval_func = oobag_eval_func

        # Attributes set during fit
        self._forest_data = None
        self.n_features_in_ = None
        self.feature_importances_ = None

    def _get_lincomb_type(self):
        """Convert string lincomb type to enum value."""
        types = {
            'random': _pyaorsf.LinearCombo.RANDOM.value,
            'glm': _pyaorsf.LinearCombo.GLM.value,
            'glmnet': _pyaorsf.LinearCombo.GLMNET.value,
        }
        return types.get(self.lincomb_type.lower(), _pyaorsf.LinearCombo.RANDOM.value)

    def _get_vi_type(self):
        """Convert string importance type to enum value."""
        if self.importance is None:
            return _pyaorsf.VariableImportance.NONE.value
        types = {
            'negate': _pyaorsf.VariableImportance.NEGATE.value,
            'permute': _pyaorsf.VariableImportance.PERMUTE.value,
            'anova': _pyaorsf.VariableImportance.ANOVA.value,
        }
        vi_type = types.get(self.importance.lower())
        if vi_type is None:
            raise ValueError(
                f"importance must be one of 'negate', 'permute', 'anova', or None, "
                f"got '{self.importance}'"
            )
        return vi_type

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
        X, y = check_X_y(X, y, dtype=np.float64)
        y = y.reshape(-1, 1)

        self.n_features_in_ = X.shape[1]

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)

        # Determine mtry
        mtry = self.mtry
        if mtry is None:
            mtry = max(1, self.n_features_in_ // 3)

        # Generate tree seeds
        rng = np.random.RandomState(self.random_state)
        tree_seeds = rng.randint(1, 2**31 - 1, size=self.n_trees).tolist()

        # Determine lincomb_type: use custom if lincomb_func is provided
        lincomb_type = self._get_lincomb_type()
        if self.lincomb_func is not None:
            lincomb_type = _pyaorsf.LinearCombo.CUSTOM.value

        # Determine oobag_eval_type: use custom if oobag_eval_func is provided
        oobag_eval_type = _pyaorsf.EvalType.NONE.value
        if self.oobag_eval_func is not None:
            oobag_eval_type = _pyaorsf.EvalType.CUSTOM.value

        # Call C++ fit function
        self._forest_data = _pyaorsf.fit_forest(
            x=X, y=y, w=sample_weight,
            tree_type=_pyaorsf.TreeType.REGRESSION.value,
            tree_seeds=tree_seeds,
            n_tree=self.n_trees,
            mtry=mtry,
            sample_with_replacement=self.sample_with_replacement,
            sample_fraction=self.sample_fraction,
            vi_type=self._get_vi_type(),
            vi_max_pvalue=0.01,
            leaf_min_events=1.0,
            leaf_min_obs=float(self.leaf_min_obs),
            split_rule=_pyaorsf.SplitRule.VARIANCE.value,
            split_min_events=1.0,
            split_min_obs=float(self.split_min_obs),
            split_min_stat=self.split_min_stat,
            split_max_cuts=self.split_max_cuts,
            split_max_retry=self.split_max_retry,
            lincomb_type=lincomb_type,
            lincomb_eps=self.lincomb_eps,
            lincomb_iter_max=self.lincomb_iter_max,
            lincomb_scale=self.lincomb_scale,
            lincomb_alpha=self.lincomb_alpha,
            lincomb_df_target=0,
            lincomb_ties_method=0,
            pred_horizon=[],
            pred_type=_pyaorsf.PredType.MEAN.value,
            oobag=True,
            oobag_eval_type=oobag_eval_type,
            oobag_eval_every=self.n_trees,
            n_thread=self.n_threads,
            verbose=self.verbose,
            lincomb_func=self.lincomb_func,
            oobag_func=self.oobag_eval_func
        )

        # Extract feature importances if computed
        if 'importance' in self._forest_data:
            self.feature_importances_ = self._forest_data['importance']
        else:
            self.feature_importances_ = None

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
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64)

        # Use C++ predict function
        pred = _pyaorsf.predict_forest(
            x=X,
            cutpoint=self._forest_data['cutpoint'],
            child_left=self._forest_data['child_left'],
            coef_values=self._forest_data['coef_values'],
            coef_indices=self._forest_data['coef_indices'],
            leaf_summary=self._forest_data['leaf_summary'],
            tree_type=_pyaorsf.TreeType.REGRESSION.value,
            n_class=1,
            aggregate=True
        )

        return pred.ravel()

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
            'split_max_cuts': self.split_max_cuts,
            'split_max_retry': self.split_max_retry,
            'sample_fraction': self.sample_fraction,
            'sample_with_replacement': self.sample_with_replacement,
            'lincomb_type': self.lincomb_type,
            'lincomb_eps': self.lincomb_eps,
            'lincomb_iter_max': self.lincomb_iter_max,
            'lincomb_scale': self.lincomb_scale,
            'lincomb_alpha': self.lincomb_alpha,
            'n_threads': self.n_threads,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'importance': self.importance,
            'lincomb_func': self.lincomb_func,
            'oobag_eval_func': self.oobag_eval_func,
        }

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
