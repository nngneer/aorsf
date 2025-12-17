"""
Tests for ObliqueForestSurvival.
"""

import pytest
import numpy as np
from pyaorsf import ObliqueForestSurvival
from pyaorsf.utils import concordance_index


class TestObliqueForestSurvival:
    """Tests for the survival forest."""

    def test_init_default_params(self):
        """Test default parameter initialization."""
        surv = ObliqueForestSurvival()
        assert surv.n_trees == 500
        assert surv.mtry is None
        assert surv.leaf_min_obs == 5
        assert surv.leaf_min_events == 1
        assert surv.n_threads == 1

    def test_init_custom_params(self):
        """Test custom parameter initialization."""
        surv = ObliqueForestSurvival(
            n_trees=100,
            mtry=5,
            leaf_min_obs=10,
            leaf_min_events=5,
            n_threads=4,
            random_state=42
        )
        assert surv.n_trees == 100
        assert surv.mtry == 5
        assert surv.leaf_min_obs == 10
        assert surv.leaf_min_events == 5
        assert surv.n_threads == 4
        assert surv.random_state == 42

    def test_get_params(self):
        """Test get_params method."""
        surv = ObliqueForestSurvival(n_trees=100, mtry=5)
        params = surv.get_params()
        assert params['n_trees'] == 100
        assert params['mtry'] == 5

    def test_set_params(self):
        """Test set_params method."""
        surv = ObliqueForestSurvival()
        surv.set_params(n_trees=200, mtry=10)
        assert surv.n_trees == 200
        assert surv.mtry == 10

    def test_fit_basic(self, survival_data):
        """Test basic fitting."""
        X, y = survival_data
        surv = ObliqueForestSurvival(n_trees=10)
        surv.fit(X, y)

        assert surv.n_features_in_ == X.shape[1]
        assert surv.unique_times_ is not None

    def test_fit_with_weights(self, survival_data):
        """Test fitting with sample weights."""
        X, y = survival_data
        weights = np.ones(len(y))
        weights[:50] = 2.0

        surv = ObliqueForestSurvival(n_trees=10)
        surv.fit(X, y, sample_weight=weights)
        assert surv.n_features_in_ == X.shape[1]

    def test_fit_invalid_y_1d(self, survival_data):
        """Test that 1D y raises error."""
        X, y = survival_data
        surv = ObliqueForestSurvival(n_trees=10)

        with pytest.raises(ValueError, match="2D array"):
            surv.fit(X, y[:, 0])

    def test_fit_invalid_y_wrong_cols(self, survival_data):
        """Test that wrong number of columns raises error."""
        X, y = survival_data
        surv = ObliqueForestSurvival(n_trees=10)

        with pytest.raises(ValueError, match="2 columns"):
            surv.fit(X, np.column_stack([y, y[:, 0]]))

    def test_predict_shape(self, survival_data):
        """Test predict output shape."""
        X, y = survival_data
        surv = ObliqueForestSurvival(n_trees=10)
        surv.fit(X, y)

        risk = surv.predict(X)
        assert risk.shape == (len(y),)

    def test_predict_survival_shape(self, survival_data):
        """Test predict_survival output shape."""
        X, y = survival_data
        surv = ObliqueForestSurvival(n_trees=10, pred_horizon=[30, 60, 90])
        surv.fit(X, y)

        survival = surv.predict_survival(X)
        assert survival.shape == (len(y), 3)

    def test_predict_survival_custom_times(self, survival_data):
        """Test predict_survival with custom times."""
        X, y = survival_data
        surv = ObliqueForestSurvival(n_trees=10)
        surv.fit(X, y)

        times = [10, 20, 30, 40]
        survival = surv.predict_survival(X, times=times)
        assert survival.shape == (len(y), 4)

    def test_predict_cumulative_hazard_shape(self, survival_data):
        """Test predict_cumulative_hazard output shape."""
        X, y = survival_data
        surv = ObliqueForestSurvival(n_trees=10, pred_horizon=[30, 60])
        surv.fit(X, y)

        cumhaz = surv.predict_cumulative_hazard(X)
        assert cumhaz.shape == (len(y), 2)

    def test_feature_importances(self, survival_data):
        """Test feature importances attribute."""
        X, y = survival_data
        surv = ObliqueForestSurvival(n_trees=10)
        surv.fit(X, y)

        assert surv.feature_importances_ is not None
        assert len(surv.feature_importances_) == X.shape[1]


class TestConcordanceIndex:
    """Tests for concordance index calculation."""

    def test_perfect_concordance(self):
        """Test perfect concordance (C-index = 1)."""
        time = np.array([1, 2, 3, 4, 5])
        status = np.array([1, 1, 1, 1, 1])
        risk = np.array([5, 4, 3, 2, 1])  # Higher risk = shorter time

        c_index = concordance_index(time, status, risk)
        assert c_index == 1.0

    def test_perfect_discordance(self):
        """Test perfect discordance (C-index = 0)."""
        time = np.array([1, 2, 3, 4, 5])
        status = np.array([1, 1, 1, 1, 1])
        risk = np.array([1, 2, 3, 4, 5])  # Lower risk = shorter time (wrong)

        c_index = concordance_index(time, status, risk)
        assert c_index == 0.0

    def test_random_concordance(self):
        """Test that random predictions give ~0.5."""
        np.random.seed(42)
        n = 1000
        time = np.random.exponential(10, n)
        status = np.ones(n)
        risk = np.random.randn(n)

        c_index = concordance_index(time, status, risk)
        assert 0.4 <= c_index <= 0.6

    def test_with_censoring(self):
        """Test C-index with censored observations."""
        time = np.array([1, 2, 3, 4, 5])
        status = np.array([1, 0, 1, 0, 1])  # Some censored
        risk = np.array([5, 4, 3, 2, 1])

        c_index = concordance_index(time, status, risk)
        # Should still work, just with fewer comparable pairs
        assert 0.0 <= c_index <= 1.0

    def test_tied_times(self):
        """Test C-index with tied times."""
        time = np.array([1, 1, 2, 2, 3])
        status = np.array([1, 1, 1, 1, 1])
        risk = np.array([5, 4, 3, 2, 1])

        c_index = concordance_index(time, status, risk)
        assert 0.0 <= c_index <= 1.0
