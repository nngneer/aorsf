"""
Tests for ObliqueForestRegressor.
"""

import pytest
import numpy as np
from pyaorsf import ObliqueForestRegressor


class TestObliqueForestRegressor:
    """Tests for the regression forest."""

    def test_init_default_params(self):
        """Test default parameter initialization."""
        reg = ObliqueForestRegressor()
        assert reg.n_trees == 500
        assert reg.mtry is None
        assert reg.leaf_min_obs == 5
        assert reg.n_threads == 1

    def test_init_custom_params(self):
        """Test custom parameter initialization."""
        reg = ObliqueForestRegressor(
            n_trees=100,
            mtry=5,
            leaf_min_obs=10,
            n_threads=4,
            random_state=42
        )
        assert reg.n_trees == 100
        assert reg.mtry == 5
        assert reg.leaf_min_obs == 10
        assert reg.n_threads == 4
        assert reg.random_state == 42

    def test_get_params(self):
        """Test get_params method."""
        reg = ObliqueForestRegressor(n_trees=100, mtry=5)
        params = reg.get_params()
        assert params['n_trees'] == 100
        assert params['mtry'] == 5

    def test_set_params(self):
        """Test set_params method."""
        reg = ObliqueForestRegressor()
        reg.set_params(n_trees=200, mtry=10)
        assert reg.n_trees == 200
        assert reg.mtry == 10

    def test_fit_basic(self, regression_data):
        """Test basic fitting."""
        X, y = regression_data
        reg = ObliqueForestRegressor(n_trees=10, random_state=42)
        reg.fit(X, y)

        assert reg.n_features_in_ == X.shape[1]
        assert reg._forest_data is not None

    def test_fit_with_weights(self, regression_data):
        """Test fitting with sample weights."""
        X, y = regression_data
        weights = np.ones(len(y))
        weights[:50] = 2.0

        reg = ObliqueForestRegressor(n_trees=10, random_state=42)
        reg.fit(X, y, sample_weight=weights)
        assert reg.n_features_in_ == X.shape[1]

    def test_predict_shape(self, regression_data):
        """Test predict output shape."""
        X, y = regression_data
        reg = ObliqueForestRegressor(n_trees=10, random_state=42)
        reg.fit(X, y)

        predictions = reg.predict(X)
        assert predictions.shape == (len(y),)

    def test_predict_reasonable_values(self, regression_data):
        """Test that predictions are in reasonable range."""
        X, y = regression_data
        reg = ObliqueForestRegressor(n_trees=50, random_state=42)
        reg.fit(X, y)

        predictions = reg.predict(X)
        # Predictions should be in a reasonable range
        assert predictions.min() >= y.min() - np.std(y) * 3
        assert predictions.max() <= y.max() + np.std(y) * 3

    def test_reproducibility(self, regression_data):
        """Test that random_state ensures reproducibility."""
        X, y = regression_data

        reg1 = ObliqueForestRegressor(n_trees=10, random_state=42)
        reg1.fit(X, y)
        pred1 = reg1.predict(X)

        reg2 = ObliqueForestRegressor(n_trees=10, random_state=42)
        reg2.fit(X, y)
        pred2 = reg2.predict(X)

        np.testing.assert_array_almost_equal(pred1, pred2)


class TestRegressorSklearnCompatibility:
    """Tests for scikit-learn compatibility."""

    def test_score_method(self, regression_data):
        """Test score method returns R^2."""
        X, y = regression_data
        reg = ObliqueForestRegressor(n_trees=50, random_state=42)
        reg.fit(X, y)

        score = reg.score(X, y)
        # R^2 should be positive for good fit on training data
        assert score > 0.0
        assert score <= 1.0

    def test_clone(self, regression_data):
        """Test that regressor can be cloned."""
        pytest.importorskip("sklearn")
        from sklearn.base import clone

        reg = ObliqueForestRegressor(n_trees=100, mtry=5)
        reg_clone = clone(reg)

        assert reg_clone.n_trees == 100
        assert reg_clone.mtry == 5
        assert reg_clone._forest_data is None  # Not fitted

    def test_cross_validation(self, regression_data):
        """Test with cross-validation."""
        pytest.importorskip("sklearn")
        from sklearn.model_selection import cross_val_score

        X, y = regression_data
        reg = ObliqueForestRegressor(n_trees=10, random_state=42)
        scores = cross_val_score(reg, X, y, cv=3)

        assert len(scores) == 3

    def test_pipeline(self, regression_data):
        """Test in a sklearn Pipeline."""
        pytest.importorskip("sklearn")
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X, y = regression_data
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('reg', ObliqueForestRegressor(n_trees=10, random_state=42))
        ])
        pipe.fit(X, y)
        score = pipe.score(X, y)
        assert score > 0.0

    def test_grid_search(self, regression_data):
        """Test with GridSearchCV."""
        pytest.importorskip("sklearn")
        from sklearn.model_selection import GridSearchCV

        X, y = regression_data
        reg = ObliqueForestRegressor(n_trees=5, random_state=42)
        param_grid = {'mtry': [2, 3], 'leaf_min_obs': [3, 5]}

        grid = GridSearchCV(reg, param_grid, cv=2, scoring='r2')
        grid.fit(X, y)

        assert 'mtry' in grid.best_params_


class TestRegressorCallbacks:
    """Tests for callback functionality."""

    def test_custom_lincomb_func(self, regression_data):
        """Test custom linear combination function."""
        X, y = regression_data

        def simple_lincomb(x, y, w):
            coefs = np.zeros(x.shape[1])
            coefs[0] = 1.0
            return coefs

        reg = ObliqueForestRegressor(
            n_trees=10,
            random_state=42,
            lincomb_func=simple_lincomb
        )
        reg.fit(X, y)

        assert reg._forest_data is not None

    def test_custom_oobag_eval_func(self, regression_data):
        """Test custom OOB evaluation function."""
        X, y = regression_data

        def custom_eval(y, w, p):
            # Mean absolute error (negative so higher is better)
            return -np.mean(np.abs(y[:, 0] - p))

        reg = ObliqueForestRegressor(
            n_trees=10,
            random_state=42,
            oobag_eval_func=custom_eval
        )
        reg.fit(X, y)

        assert reg._forest_data is not None


class TestRegressorEdgeCases:
    """Tests for edge cases and error handling."""

    def test_predict_before_fit(self):
        """Test that predict before fit raises error."""
        reg = ObliqueForestRegressor()
        X = np.random.randn(10, 5)

        with pytest.raises(Exception):
            reg.predict(X)

    def test_single_sample_predict(self, regression_data):
        """Test prediction on a single sample."""
        X, y = regression_data
        reg = ObliqueForestRegressor(n_trees=10, random_state=42)
        reg.fit(X, y)

        pred = reg.predict(X[:1])
        assert pred.shape == (1,)

    def test_small_dataset(self):
        """Test with a very small dataset."""
        X = np.random.randn(20, 5)
        y = X[:, 0] + np.random.randn(20) * 0.1

        reg = ObliqueForestRegressor(n_trees=5, random_state=42)
        reg.fit(X, y)

        assert reg.n_features_in_ == 5
