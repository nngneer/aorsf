"""
Oblique random survival forest for time-to-event data.
"""

import numpy as np
from typing import Optional, Union, List

# Will import from C++ bindings once implemented
# from .._pyaorsf import ForestSurvival


class ObliqueForestSurvival:
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
    leaf_min_events : int, default=1
        Minimum number of events in a leaf node.
    split_min_obs : int, default=10
        Minimum number of observations required to attempt a split.
    split_min_events : int, default=5
        Minimum number of events required to attempt a split.
    split_min_stat : float, default=None
        Minimum improvement in split statistic to make a split.
    split_rule : str, default='logrank'
        Split rule to use. One of 'logrank' or 'cstat'.
    pred_horizon : array-like, default=None
        Time points at which to predict survival probability.
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
        leaf_min_events: int = 1,
        split_min_obs: int = 10,
        split_min_events: int = 5,
        split_min_stat: Optional[float] = None,
        split_rule: str = 'logrank',
        pred_horizon: Optional[Union[float, List[float]]] = None,
        n_threads: int = 1,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        self.n_trees = n_trees
        self.mtry = mtry
        self.leaf_min_obs = leaf_min_obs
        self.leaf_min_events = leaf_min_events
        self.split_min_obs = split_min_obs
        self.split_min_events = split_min_events
        self.split_min_stat = split_min_stat
        self.split_rule = split_rule
        self.pred_horizon = pred_horizon
        self.n_threads = n_threads
        self.random_state = random_state
        self.verbose = verbose

        # Attributes set during fit
        self._forest = None
        self.n_features_in_ = None
        self.feature_importances_ = None
        self.unique_times_ = None

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
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if y.ndim == 1:
            raise ValueError("y must be 2D array with columns [time, status]")
        if y.shape[1] != 2:
            raise ValueError("y must have exactly 2 columns: [time, status]")

        self.n_features_in_ = X.shape[1]

        # Extract unique event times
        event_mask = y[:, 1] == 1
        self.unique_times_ = np.unique(y[event_mask, 0])

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)

        # Set default prediction horizon if not specified
        if self.pred_horizon is None:
            # Use median event time
            self._pred_horizon = np.array([np.median(self.unique_times_)])
        else:
            self._pred_horizon = np.atleast_1d(self.pred_horizon).astype(np.float64)

        # TODO: Call C++ forest training once bindings are implemented
        # self._forest = ForestSurvival(...)
        # self._forest.train(X, y, sample_weight, ...)

        # Placeholder for feature importances
        self.feature_importances_ = np.zeros(self.n_features_in_)

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
        X = np.asarray(X, dtype=np.float64)

        # TODO: Call C++ forest prediction once bindings are implemented
        # return self._forest.predict(X)

        # Placeholder
        n_samples = X.shape[0]
        if len(self._pred_horizon) == 1:
            return np.random.rand(n_samples)
        return np.random.rand(n_samples, len(self._pred_horizon))

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
        X = np.asarray(X, dtype=np.float64)

        if times is None:
            times = self._pred_horizon
        else:
            times = np.atleast_1d(times).astype(np.float64)

        # TODO: Call C++ forest prediction once bindings are implemented
        # return self._forest.predict_survival(X, times)

        # Placeholder
        n_samples = X.shape[0]
        n_times = len(times)
        return np.random.rand(n_samples, n_times)

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
        X = np.asarray(X, dtype=np.float64)

        if times is None:
            times = self._pred_horizon
        else:
            times = np.atleast_1d(times).astype(np.float64)

        # TODO: Call C++ forest prediction once bindings are implemented
        # return self._forest.predict_cumulative_hazard(X, times)

        # Placeholder
        n_samples = X.shape[0]
        n_times = len(times)
        return np.random.rand(n_samples, n_times)

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
        from .utils import concordance_index
        risk = self.predict(X)
        if risk.ndim > 1:
            risk = risk[:, 0]  # Use first horizon
        y = np.asarray(y)
        return concordance_index(y[:, 0], y[:, 1], risk)

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
            'pred_horizon': self.pred_horizon,
            'n_threads': self.n_threads,
            'random_state': self.random_state,
            'verbose': self.verbose,
        }

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
