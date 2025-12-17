"""
Tests for ObliqueForestClassifier.
"""

import pytest
import numpy as np
from pyaorsf import ObliqueForestClassifier


class TestObliqueForestClassifier:
    """Tests for the classification forest."""

    def test_init_default_params(self):
        """Test default parameter initialization."""
        clf = ObliqueForestClassifier()
        assert clf.n_trees == 500
        assert clf.mtry is None
        assert clf.leaf_min_obs == 5
        assert clf.n_threads == 1

    def test_init_custom_params(self):
        """Test custom parameter initialization."""
        clf = ObliqueForestClassifier(
            n_trees=100,
            mtry=5,
            leaf_min_obs=10,
            n_threads=4,
            random_state=42
        )
        assert clf.n_trees == 100
        assert clf.mtry == 5
        assert clf.leaf_min_obs == 10
        assert clf.n_threads == 4
        assert clf.random_state == 42

    def test_get_params(self):
        """Test get_params method."""
        clf = ObliqueForestClassifier(n_trees=100, mtry=5)
        params = clf.get_params()
        assert params['n_trees'] == 100
        assert params['mtry'] == 5

    def test_set_params(self):
        """Test set_params method."""
        clf = ObliqueForestClassifier()
        clf.set_params(n_trees=200, mtry=10)
        assert clf.n_trees == 200
        assert clf.mtry == 10

    def test_fit_basic(self, classification_data):
        """Test basic fitting."""
        X, y = classification_data
        clf = ObliqueForestClassifier(n_trees=10)
        clf.fit(X, y)

        assert clf.n_features_in_ == X.shape[1]
        assert clf.n_classes_ == 2
        assert len(clf.classes_) == 2

    def test_fit_with_weights(self, classification_data):
        """Test fitting with sample weights."""
        X, y = classification_data
        weights = np.ones(len(y))
        weights[:50] = 2.0

        clf = ObliqueForestClassifier(n_trees=10)
        clf.fit(X, y, sample_weight=weights)
        assert clf.n_features_in_ == X.shape[1]

    def test_predict_shape(self, classification_data):
        """Test predict output shape."""
        X, y = classification_data
        clf = ObliqueForestClassifier(n_trees=10)
        clf.fit(X, y)

        predictions = clf.predict(X)
        assert predictions.shape == (len(y),)

    def test_predict_proba_shape(self, classification_data):
        """Test predict_proba output shape."""
        X, y = classification_data
        clf = ObliqueForestClassifier(n_trees=10)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 2)

    def test_predict_proba_sums_to_one(self, classification_data):
        """Test that probabilities sum to 1."""
        X, y = classification_data
        clf = ObliqueForestClassifier(n_trees=10)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)

    def test_multiclass(self, multiclass_data):
        """Test multiclass classification."""
        X, y = multiclass_data
        clf = ObliqueForestClassifier(n_trees=10)
        clf.fit(X, y)

        assert clf.n_classes_ == 3
        assert len(clf.classes_) == 3

        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 3)

    def test_feature_importances(self, classification_data):
        """Test feature importances attribute."""
        X, y = classification_data
        clf = ObliqueForestClassifier(n_trees=10)
        clf.fit(X, y)

        assert clf.feature_importances_ is not None
        assert len(clf.feature_importances_) == X.shape[1]


class TestClassifierSklearnCompatibility:
    """Tests for scikit-learn compatibility."""

    def test_score_method(self, classification_data):
        """Test score method returns accuracy."""
        X, y = classification_data
        clf = ObliqueForestClassifier(n_trees=10)
        clf.fit(X, y)

        score = clf.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_clone(self, classification_data):
        """Test that classifier can be cloned."""
        pytest.importorskip("sklearn")
        from sklearn.base import clone

        clf = ObliqueForestClassifier(n_trees=100, mtry=5)
        clf_clone = clone(clf)

        assert clf_clone.n_trees == 100
        assert clf_clone.mtry == 5
        assert clf_clone._forest is None  # Not fitted

    @pytest.mark.skip(reason="C++ bindings not yet implemented")
    def test_cross_validation(self, classification_data):
        """Test with cross-validation."""
        pytest.importorskip("sklearn")
        from sklearn.model_selection import cross_val_score

        X, y = classification_data
        clf = ObliqueForestClassifier(n_trees=10)
        scores = cross_val_score(clf, X, y, cv=3)

        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)
