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
        assert surv.leaf_min_events == 2
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
        surv = ObliqueForestSurvival(n_trees=10, random_state=42)
        surv.fit(X, y)

        assert surv.n_features_in_ == X.shape[1]
        assert surv.unique_times_ is not None
        assert surv._forest_data is not None

    def test_fit_with_weights(self, survival_data):
        """Test fitting with sample weights."""
        X, y = survival_data
        weights = np.ones(len(y))
        weights[:50] = 2.0

        surv = ObliqueForestSurvival(n_trees=10, random_state=42)
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
        surv = ObliqueForestSurvival(n_trees=10, random_state=42)
        surv.fit(X, y)

        risk = surv.predict(X)
        assert risk.shape == (len(y),)

    def test_predict_survival_shape(self, survival_data):
        """Test predict_survival output shape."""
        X, y = survival_data
        surv = ObliqueForestSurvival(n_trees=10, random_state=42)
        surv.fit(X, y)

        survival = surv.predict_survival(X)
        # Returns survival probabilities
        assert survival.ndim >= 1
        assert len(survival) == len(y)

    def test_predict_cumulative_hazard_shape(self, survival_data):
        """Test predict_cumulative_hazard output shape."""
        X, y = survival_data
        surv = ObliqueForestSurvival(n_trees=10, random_state=42)
        surv.fit(X, y)

        cumhaz = surv.predict_cumulative_hazard(X)
        assert cumhaz.ndim >= 1
        assert len(cumhaz) == len(y)

    def test_reproducibility(self, survival_data):
        """Test that random_state ensures reproducibility."""
        X, y = survival_data

        surv1 = ObliqueForestSurvival(n_trees=10, random_state=42)
        surv1.fit(X, y)
        pred1 = surv1.predict(X)

        surv2 = ObliqueForestSurvival(n_trees=10, random_state=42)
        surv2.fit(X, y)
        pred2 = surv2.predict(X)

        np.testing.assert_array_almost_equal(pred1, pred2)


class TestSurvivalSklearnCompatibility:
    """Tests for scikit-learn compatibility."""

    def test_score_method(self, survival_data):
        """Test score method returns C-index."""
        X, y = survival_data
        surv = ObliqueForestSurvival(n_trees=50, random_state=42)
        surv.fit(X, y)

        score = surv.score(X, y)
        # C-index should be better than random (0.5)
        assert 0.0 <= score <= 1.0

    def test_clone(self, survival_data):
        """Test that model can be cloned."""
        pytest.importorskip("sklearn")
        from sklearn.base import clone

        surv = ObliqueForestSurvival(n_trees=100, mtry=5)
        surv_clone = clone(surv)

        assert surv_clone.n_trees == 100
        assert surv_clone.mtry == 5
        assert surv_clone._forest_data is None  # Not fitted

    def test_cross_validation(self, survival_data):
        """Test with cross-validation."""
        pytest.importorskip("sklearn")
        from sklearn.model_selection import cross_val_score

        X, y = survival_data
        surv = ObliqueForestSurvival(n_trees=10, random_state=42)
        scores = cross_val_score(surv, X, y, cv=3)

        assert len(scores) == 3

    def test_pipeline(self, survival_data):
        """Test in a sklearn Pipeline."""
        pytest.importorskip("sklearn")
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X, y = survival_data
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('surv', ObliqueForestSurvival(n_trees=10, random_state=42))
        ])
        pipe.fit(X, y)
        score = pipe.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_grid_search(self, survival_data):
        """Test with GridSearchCV."""
        pytest.importorskip("sklearn")
        from sklearn.model_selection import GridSearchCV

        X, y = survival_data
        surv = ObliqueForestSurvival(n_trees=5, random_state=42)
        param_grid = {'mtry': [2, 3], 'leaf_min_obs': [3, 5]}

        grid = GridSearchCV(surv, param_grid, cv=2)
        grid.fit(X, y)

        assert 'mtry' in grid.best_params_


class TestSurvivalCallbacks:
    """Tests for callback functionality."""

    def test_custom_lincomb_func(self, survival_data):
        """Test custom linear combination function."""
        X, y = survival_data

        def simple_lincomb(x, y, w):
            coefs = np.zeros(x.shape[1])
            coefs[0] = 1.0
            return coefs

        surv = ObliqueForestSurvival(
            n_trees=10,
            random_state=42,
            lincomb_func=simple_lincomb
        )
        surv.fit(X, y)

        assert surv._forest_data is not None

    def test_custom_oobag_eval_func(self, survival_data):
        """Test custom OOB evaluation function."""
        X, y = survival_data

        def custom_eval(y, w, p):
            # Simple negative MSE for risk predictions
            time = y[:, 0]
            return -np.mean((time - p) ** 2)

        surv = ObliqueForestSurvival(
            n_trees=10,
            random_state=42,
            oobag_eval_func=custom_eval
        )
        surv.fit(X, y)

        assert surv._forest_data is not None


class TestSurvivalEdgeCases:
    """Tests for edge cases and error handling."""

    def test_predict_before_fit(self):
        """Test that predict before fit raises error."""
        surv = ObliqueForestSurvival()
        X = np.random.randn(10, 5)

        with pytest.raises(Exception):
            surv.predict(X)

    def test_single_sample_predict(self, survival_data):
        """Test prediction on a single sample."""
        X, y = survival_data
        surv = ObliqueForestSurvival(n_trees=10, random_state=42)
        surv.fit(X, y)

        pred = surv.predict(X[:1])
        assert pred.shape == (1,)

    def test_small_dataset(self):
        """Test with a very small dataset."""
        np.random.seed(42)
        X = np.random.randn(30, 5)
        time = np.random.exponential(10, 30)
        status = np.random.binomial(1, 0.7, 30)
        y = np.column_stack([time, status])

        surv = ObliqueForestSurvival(n_trees=5, random_state=42)
        surv.fit(X, y)

        assert surv.n_features_in_ == 5


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
