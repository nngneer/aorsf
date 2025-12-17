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
        reg = ObliqueForestRegressor(n_trees=10)
        reg.fit(X, y)

        assert reg.n_features_in_ == X.shape[1]

    def test_fit_with_weights(self, regression_data):
        """Test fitting with sample weights."""
        X, y = regression_data
        weights = np.ones(len(y))
        weights[:50] = 2.0

        reg = ObliqueForestRegressor(n_trees=10)
        reg.fit(X, y, sample_weight=weights)
        assert reg.n_features_in_ == X.shape[1]

    def test_predict_shape(self, regression_data):
        """Test predict output shape."""
        X, y = regression_data
        reg = ObliqueForestRegressor(n_trees=10)
        reg.fit(X, y)

        predictions = reg.predict(X)
        assert predictions.shape == (len(y),)

    def test_feature_importances(self, regression_data):
        """Test feature importances attribute."""
        X, y = regression_data
        reg = ObliqueForestRegressor(n_trees=10)
        reg.fit(X, y)

        assert reg.feature_importances_ is not None
        assert len(reg.feature_importances_) == X.shape[1]


class TestRegressorSklearnCompatibility:
    """Tests for scikit-learn compatibility."""

    def test_score_method(self, regression_data):
        """Test score method returns R^2."""
        X, y = regression_data
        reg = ObliqueForestRegressor(n_trees=10)
        reg.fit(X, y)

        score = reg.score(X, y)
        # R^2 can be negative for bad predictions
        assert isinstance(score, float)

    def test_clone(self, regression_data):
        """Test that regressor can be cloned."""
        pytest.importorskip("sklearn")
        from sklearn.base import clone

        reg = ObliqueForestRegressor(n_trees=100, mtry=5)
        reg_clone = clone(reg)

        assert reg_clone.n_trees == 100
        assert reg_clone.mtry == 5
        assert reg_clone._forest is None  # Not fitted

    @pytest.mark.skip(reason="C++ bindings not yet implemented")
    def test_cross_validation(self, regression_data):
        """Test with cross-validation."""
        pytest.importorskip("sklearn")
        from sklearn.model_selection import cross_val_score

        X, y = regression_data
        reg = ObliqueForestRegressor(n_trees=10)
        scores = cross_val_score(reg, X, y, cv=3)

        assert len(scores) == 3
