"""
Oblique random survival forest for time-to-event data.
"""

import numpy as np
from typing import Optional, Union, List

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted

from . import _pyaorsf
from .utils import concordance_index, validate_survival_data


class ObliqueForestSurvival(BaseEstimator):
    """
    Oblique Random Survival Forest.

    An oblique random survival forest for time-to-event (survival) data.
    Uses accelerated Cox regression for computing linear combinations of
    features at each split.

    Parameters
    ----------
    n_trees : int, default=500
        Number of trees in the forest.
    mtry : int, default=None
        Number of features to consider at each split. If None, uses sqrt(n_features).
    leaf_min_obs : int, default=5
        Minimum number of observations in a leaf node.
    leaf_min_events : int, default=2
        Minimum number of events in a leaf node.
    split_min_obs : int, default=10
        Minimum number of observations required to attempt a split.
    split_min_events : int, default=5
        Minimum number of events required to attempt a split.
    split_min_stat : float, default=0.0
        Minimum improvement in split statistic to make a split.
    split_rule : str, default='logrank'
        Split rule to use. One of 'logrank' or 'cstat'.
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
    pred_horizon : array-like, default=None
        Time points at which to predict survival probability.
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
    unique_times_ : ndarray
        Unique event times from training data.

    Examples
    --------
    >>> from pyaorsf import ObliqueForestSurvival
    >>> import numpy as np
    >>> # Create survival data: X features, y = [time, status]
    >>> X = np.random.randn(100, 10)
    >>> time = np.random.exponential(10, 100)
    >>> status = np.random.binomial(1, 0.7, 100)
    >>> y = np.column_stack([time, status])
    >>> surv = ObliqueForestSurvival(n_trees=10)
    >>> surv.fit(X, y)
    >>> risk = surv.predict(X[:5])
    """

    def __init__(
        self,
        n_trees: int = 500,
        mtry: Optional[int] = None,
        leaf_min_obs: int = 5,
        leaf_min_events: int = 2,
        split_min_obs: int = 10,
        split_min_events: int = 5,
        split_min_stat: float = 0.0,
        split_rule: str = 'logrank',
        split_max_cuts: int = 5,
        split_max_retry: int = 3,
        sample_fraction: float = 0.632,
        sample_with_replacement: bool = True,
        lincomb_type: str = 'random',
        lincomb_eps: float = 1e-9,
        lincomb_iter_max: int = 20,
        lincomb_scale: bool = True,
        lincomb_alpha: float = 0.5,
        pred_horizon: Optional[Union[float, List[float]]] = None,
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
        self.leaf_min_events = leaf_min_events
        self.split_min_obs = split_min_obs
        self.split_min_events = split_min_events
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
        self.pred_horizon = pred_horizon
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
        self.unique_times_ = None
        self._pred_horizon = None

    def _get_split_rule(self):
        """Convert string split rule to enum value."""
        rules = {
            'logrank': _pyaorsf.SplitRule.LOGRANK.value,
            'cstat': _pyaorsf.SplitRule.CONCORD.value,
        }
        return rules.get(self.split_rule.lower(), _pyaorsf.SplitRule.LOGRANK.value)

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
        Fit the oblique random survival forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples, 2)
            Survival outcome. First column is time, second column is event status
            (1 = event occurred, 0 = censored).
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        self : ObliqueForestSurvival
            Fitted estimator.
        """
        X = check_array(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Validate survival data
        time, status = validate_survival_data(y)

        self.n_features_in_ = X.shape[1]

        # Extract unique event times
        event_mask = status == 1
        self.unique_times_ = np.unique(time[event_mask])

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)

        # Sort data by time (required by aorsf)
        sort_idx = np.argsort(time)
        X = X[sort_idx]
        y = y[sort_idx]
        sample_weight = sample_weight[sort_idx]

        # Set default prediction horizon if not specified
        if self.pred_horizon is None:
            self._pred_horizon = [float(np.median(self.unique_times_))]
        else:
            self._pred_horizon = list(np.atleast_1d(self.pred_horizon).astype(np.float64))

        # Determine mtry
        mtry = self.mtry
        if mtry is None:
            mtry = max(1, int(np.sqrt(self.n_features_in_)))

        # Generate tree seeds
        rng = np.random.RandomState(self.random_state)
        tree_seeds = rng.randint(1, 2**31 - 1, size=self.n_trees).tolist()

        # Ensure leaf_min_events and split_min_events >= 2
        # (required by aorsf to avoid division by zero)
        safe_leaf_min_events = max(2, self.leaf_min_events)
        safe_split_min_events = max(2, self.split_min_events)

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
            tree_type=_pyaorsf.TreeType.SURVIVAL.value,
            tree_seeds=tree_seeds,
            n_tree=self.n_trees,
            mtry=mtry,
            sample_with_replacement=self.sample_with_replacement,
            sample_fraction=self.sample_fraction,
            vi_type=self._get_vi_type(),
            vi_max_pvalue=0.01,
            leaf_min_events=float(safe_leaf_min_events),
            leaf_min_obs=float(self.leaf_min_obs),
            split_rule=self._get_split_rule(),
            split_min_events=float(safe_split_min_events),
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
            pred_horizon=self._pred_horizon,
            pred_type=_pyaorsf.PredType.RISK.value,
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

        # Store OOB predictions for later use
        if 'oob_predictions' in self._forest_data:
            self._oob_predictions = self._forest_data['oob_predictions']
        else:
            self._oob_predictions = None

        return self

    def predict(self, X):
        """
        Predict risk scores for samples in X.

        Higher risk scores indicate higher risk of the event occurring.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        risk : ndarray of shape (n_samples,) or (n_samples, n_horizons)
            Predicted risk scores. If multiple prediction horizons are set,
            returns a 2D array.
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
            tree_type=_pyaorsf.TreeType.SURVIVAL.value,
            n_class=1,
            aggregate=True
        )

        if pred.shape[1] == 1:
            return pred.ravel()
        return pred

    def predict_survival(self, X, times=None):
        """
        Predict survival probability at specified times.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        times : array-like, default=None
            Time points at which to predict survival probability.
            If None, uses pred_horizon set at initialization.

        Returns
        -------
        survival : ndarray of shape (n_samples, n_times)
            Predicted survival probabilities.
        """
        check_is_fitted(self)
        # predict already does validation
        risk = self.predict(X)
        if risk.ndim == 1:
            risk = risk.reshape(-1, 1)

        # Convert risk to survival probability (simple approximation)
        survival = np.exp(-risk)
        return survival

    def predict_cumulative_hazard(self, X, times=None):
        """
        Predict cumulative hazard at specified times.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        times : array-like, default=None
            Time points at which to predict cumulative hazard.
            If None, uses pred_horizon set at initialization.

        Returns
        -------
        cumhaz : ndarray of shape (n_samples, n_times)
            Predicted cumulative hazard.
        """
        check_is_fitted(self)
        # predict already does validation
        risk = self.predict(X)
        if risk.ndim == 1:
            risk = risk.reshape(-1, 1)
        return risk

    def score(self, X, y, sample_weight=None):
        """
        Return Harrell's C-index on the given test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples, 2)
            True survival outcomes [time, status].
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights (not currently used).

        Returns
        -------
        score : float
            Harrell's C-index.
        """
        risk = self.predict(X)
        if risk.ndim > 1:
            risk = risk[:, 0]  # Use first horizon
        y = np.asarray(y)
        time, status = validate_survival_data(y)
        return concordance_index(time, status, risk)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'n_trees': self.n_trees,
            'mtry': self.mtry,
            'leaf_min_obs': self.leaf_min_obs,
            'leaf_min_events': self.leaf_min_events,
            'split_min_obs': self.split_min_obs,
            'split_min_events': self.split_min_events,
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
            'pred_horizon': self.pred_horizon,
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
