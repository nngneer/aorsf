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
        clf = ObliqueForestClassifier(n_trees=10, random_state=42)
        clf.fit(X, y)

        assert clf.n_features_in_ == X.shape[1]
        assert clf.n_classes_ == 2
        assert len(clf.classes_) == 2
        assert clf._forest_data is not None

    def test_fit_with_weights(self, classification_data):
        """Test fitting with sample weights."""
        X, y = classification_data
        weights = np.ones(len(y))
        weights[:50] = 2.0

        clf = ObliqueForestClassifier(n_trees=10, random_state=42)
        clf.fit(X, y, sample_weight=weights)
        assert clf.n_features_in_ == X.shape[1]

    def test_predict_shape(self, classification_data):
        """Test predict output shape."""
        X, y = classification_data
        clf = ObliqueForestClassifier(n_trees=10, random_state=42)
        clf.fit(X, y)

        predictions = clf.predict(X)
        assert predictions.shape == (len(y),)

    def test_predict_proba_shape(self, classification_data):
        """Test predict_proba output shape."""
        X, y = classification_data
        clf = ObliqueForestClassifier(n_trees=10, random_state=42)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 2)

    def test_predict_proba_sums_to_one(self, classification_data):
        """Test that probabilities sum to 1."""
        X, y = classification_data
        clf = ObliqueForestClassifier(n_trees=10, random_state=42)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)

    def test_multiclass(self, multiclass_data):
        """Test multiclass classification."""
        X, y = multiclass_data
        clf = ObliqueForestClassifier(n_trees=10, random_state=42)
        clf.fit(X, y)

        assert clf.n_classes_ == 3
        assert len(clf.classes_) == 3

        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 3)

    def test_predict_values_in_classes(self, classification_data):
        """Test that predictions are valid class labels."""
        X, y = classification_data
        clf = ObliqueForestClassifier(n_trees=10, random_state=42)
        clf.fit(X, y)

        predictions = clf.predict(X)
        assert all(p in clf.classes_ for p in predictions)

    def test_reproducibility(self, classification_data):
        """Test that random_state ensures reproducibility."""
        X, y = classification_data

        clf1 = ObliqueForestClassifier(n_trees=10, random_state=42)
        clf1.fit(X, y)
        pred1 = clf1.predict(X)

        clf2 = ObliqueForestClassifier(n_trees=10, random_state=42)
        clf2.fit(X, y)
        pred2 = clf2.predict(X)

        np.testing.assert_array_equal(pred1, pred2)


class TestClassifierSklearnCompatibility:
    """Tests for scikit-learn compatibility."""

    def test_score_method(self, classification_data):
        """Test score method returns accuracy."""
        X, y = classification_data
        clf = ObliqueForestClassifier(n_trees=10, random_state=42)
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
        assert clf_clone._forest_data is None  # Not fitted

    def test_cross_validation(self, classification_data):
        """Test with cross-validation."""
        pytest.importorskip("sklearn")
        from sklearn.model_selection import cross_val_score

        X, y = classification_data
        clf = ObliqueForestClassifier(n_trees=10, random_state=42)
        scores = cross_val_score(clf, X, y, cv=3)

        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)

    def test_pipeline(self, classification_data):
        """Test in a sklearn Pipeline."""
        pytest.importorskip("sklearn")
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X, y = classification_data
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', ObliqueForestClassifier(n_trees=10, random_state=42))
        ])
        pipe.fit(X, y)
        score = pipe.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_grid_search(self, classification_data):
        """Test with GridSearchCV."""
        pytest.importorskip("sklearn")
        from sklearn.model_selection import GridSearchCV

        X, y = classification_data
        clf = ObliqueForestClassifier(n_trees=5, random_state=42)
        param_grid = {'mtry': [2, 3], 'leaf_min_obs': [3, 5]}

        grid = GridSearchCV(clf, param_grid, cv=2, scoring='accuracy')
        grid.fit(X, y)

        assert grid.best_score_ > 0.5
        assert 'mtry' in grid.best_params_


class TestClassifierCallbacks:
    """Tests for callback functionality."""

    def test_custom_lincomb_func(self, classification_data):
        """Test custom linear combination function."""
        X, y = classification_data

        def simple_lincomb(x, y, w):
            # Use only first feature
            coefs = np.zeros(x.shape[1])
            coefs[0] = 1.0
            return coefs

        clf = ObliqueForestClassifier(
            n_trees=10,
            random_state=42,
            lincomb_func=simple_lincomb
        )
        clf.fit(X, y)

        score = clf.score(X, y)
        assert score > 0.5  # Should be better than random

    def test_custom_oobag_eval_func(self, classification_data):
        """Test custom OOB evaluation function."""
        X, y = classification_data

        def custom_eval(y, w, p):
            # Simple accuracy
            y_true = y[:, 0]
            y_pred = (p > 0.5).astype(float)
            return np.mean(y_true == y_pred)

        clf = ObliqueForestClassifier(
            n_trees=10,
            random_state=42,
            oobag_eval_func=custom_eval
        )
        clf.fit(X, y)

        # Should still work
        assert clf._forest_data is not None


class TestClassifierEdgeCases:
    """Tests for edge cases and error handling."""

    def test_predict_before_fit(self):
        """Test that predict before fit raises error."""
        clf = ObliqueForestClassifier()
        X = np.random.randn(10, 5)

        with pytest.raises(Exception):
            clf.predict(X)

    def test_single_sample_predict(self, classification_data):
        """Test prediction on a single sample."""
        X, y = classification_data
        clf = ObliqueForestClassifier(n_trees=10, random_state=42)
        clf.fit(X, y)

        pred = clf.predict(X[:1])
        assert pred.shape == (1,)

    def test_small_dataset(self):
        """Test with a very small dataset."""
        X = np.random.randn(20, 5)
        y = np.array([0] * 10 + [1] * 10)

        clf = ObliqueForestClassifier(n_trees=5, random_state=42)
        clf.fit(X, y)

        assert clf.n_features_in_ == 5
