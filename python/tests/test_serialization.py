"""
Tests for model serialization (pickle).
"""

import pickle
import tempfile
import os

import pytest
import numpy as np
from pyaorsf import ObliqueForestClassifier, ObliqueForestRegressor, ObliqueForestSurvival


class TestClassifierSerialization:
    """Tests for classifier pickle serialization."""

    def test_pickle_unfitted(self):
        """Test pickling unfitted classifier."""
        clf = ObliqueForestClassifier(n_trees=100, mtry=5)

        pickled = pickle.dumps(clf)
        clf_loaded = pickle.loads(pickled)

        assert clf_loaded.n_trees == 100
        assert clf_loaded.mtry == 5
        assert clf_loaded._forest_data is None

    def test_pickle_fitted(self, classification_data):
        """Test pickling fitted classifier."""
        X, y = classification_data
        clf = ObliqueForestClassifier(n_trees=10, random_state=42)
        clf.fit(X, y)

        # Pickle and unpickle
        pickled = pickle.dumps(clf)
        clf_loaded = pickle.loads(pickled)

        # Predictions should match
        pred_original = clf.predict(X)
        pred_loaded = clf_loaded.predict(X)
        np.testing.assert_array_equal(pred_original, pred_loaded)

    def test_pickle_proba_matches(self, classification_data):
        """Test that predict_proba matches after pickle."""
        X, y = classification_data
        clf = ObliqueForestClassifier(n_trees=10, random_state=42)
        clf.fit(X, y)

        pickled = pickle.dumps(clf)
        clf_loaded = pickle.loads(pickled)

        proba_original = clf.predict_proba(X)
        proba_loaded = clf_loaded.predict_proba(X)
        np.testing.assert_array_almost_equal(proba_original, proba_loaded)

    def test_pickle_to_file(self, classification_data):
        """Test pickling to file."""
        X, y = classification_data
        clf = ObliqueForestClassifier(n_trees=10, random_state=42)
        clf.fit(X, y)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            pickle.dump(clf, f)
            temp_path = f.name

        try:
            with open(temp_path, 'rb') as f:
                clf_loaded = pickle.load(f)

            pred_original = clf.predict(X)
            pred_loaded = clf_loaded.predict(X)
            np.testing.assert_array_equal(pred_original, pred_loaded)
        finally:
            os.unlink(temp_path)

    def test_pickle_with_importance(self, classification_data):
        """Test pickling with importance computed."""
        X, y = classification_data
        clf = ObliqueForestClassifier(
            n_trees=10,
            importance='negate',
            random_state=42
        )
        clf.fit(X, y)

        pickled = pickle.dumps(clf)
        clf_loaded = pickle.loads(pickled)

        np.testing.assert_array_almost_equal(
            clf.feature_importances_,
            clf_loaded.feature_importances_
        )

    def test_pickle_with_callbacks(self, classification_data):
        """Test that models with callbacks cannot be pickled by default.

        This is a known limitation: local functions cannot be pickled.
        Users should either use module-level functions or set lincomb_func=None
        before pickling.
        """
        X, y = classification_data

        def custom_lincomb(x, y, w):
            coefs = np.zeros(x.shape[1])
            coefs[0] = 1.0
            return coefs

        clf = ObliqueForestClassifier(
            n_trees=10,
            lincomb_func=custom_lincomb,
            random_state=42
        )
        clf.fit(X, y)

        # Local functions cannot be pickled - this should raise an error
        with pytest.raises(pickle.PicklingError):
            pickle.dumps(clf)

        # But we can pickle after removing the callback
        clf.lincomb_func = None
        pickled = pickle.dumps(clf)
        clf_loaded = pickle.loads(pickled)

        # Prediction should still work (uses stored forest structure)
        pred = clf_loaded.predict(X)
        assert pred.shape == (len(y),)

    def test_pickle_preserves_attributes(self, classification_data):
        """Test that all relevant attributes are preserved."""
        X, y = classification_data
        clf = ObliqueForestClassifier(n_trees=10, random_state=42)
        clf.fit(X, y)

        pickled = pickle.dumps(clf)
        clf_loaded = pickle.loads(pickled)

        assert clf_loaded.n_features_in_ == clf.n_features_in_
        assert clf_loaded.n_classes_ == clf.n_classes_
        np.testing.assert_array_equal(clf_loaded.classes_, clf.classes_)


class TestRegressorSerialization:
    """Tests for regressor pickle serialization."""

    def test_pickle_unfitted(self):
        """Test pickling unfitted regressor."""
        reg = ObliqueForestRegressor(n_trees=100, mtry=5)

        pickled = pickle.dumps(reg)
        reg_loaded = pickle.loads(pickled)

        assert reg_loaded.n_trees == 100
        assert reg_loaded.mtry == 5
        assert reg_loaded._forest_data is None

    def test_pickle_fitted(self, regression_data):
        """Test pickling fitted regressor."""
        X, y = regression_data
        reg = ObliqueForestRegressor(n_trees=10, random_state=42)
        reg.fit(X, y)

        pickled = pickle.dumps(reg)
        reg_loaded = pickle.loads(pickled)

        pred_original = reg.predict(X)
        pred_loaded = reg_loaded.predict(X)
        np.testing.assert_array_almost_equal(pred_original, pred_loaded)

    def test_pickle_to_file(self, regression_data):
        """Test pickling to file."""
        X, y = regression_data
        reg = ObliqueForestRegressor(n_trees=10, random_state=42)
        reg.fit(X, y)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            pickle.dump(reg, f)
            temp_path = f.name

        try:
            with open(temp_path, 'rb') as f:
                reg_loaded = pickle.load(f)

            pred_original = reg.predict(X)
            pred_loaded = reg_loaded.predict(X)
            np.testing.assert_array_almost_equal(pred_original, pred_loaded)
        finally:
            os.unlink(temp_path)


class TestSurvivalSerialization:
    """Tests for survival forest pickle serialization."""

    def test_pickle_unfitted(self):
        """Test pickling unfitted survival forest."""
        surv = ObliqueForestSurvival(n_trees=100, mtry=5)

        pickled = pickle.dumps(surv)
        surv_loaded = pickle.loads(pickled)

        assert surv_loaded.n_trees == 100
        assert surv_loaded.mtry == 5
        assert surv_loaded._forest_data is None

    def test_pickle_fitted(self, survival_data):
        """Test pickling fitted survival forest."""
        X, y = survival_data
        surv = ObliqueForestSurvival(n_trees=10, random_state=42)
        surv.fit(X, y)

        pickled = pickle.dumps(surv)
        surv_loaded = pickle.loads(pickled)

        risk_original = surv.predict(X)
        risk_loaded = surv_loaded.predict(X)
        np.testing.assert_array_almost_equal(risk_original, risk_loaded)

    def test_pickle_preserves_unique_times(self, survival_data):
        """Test that unique_times_ is preserved."""
        X, y = survival_data
        surv = ObliqueForestSurvival(n_trees=10, random_state=42)
        surv.fit(X, y)

        pickled = pickle.dumps(surv)
        surv_loaded = pickle.loads(pickled)

        np.testing.assert_array_equal(
            surv.unique_times_,
            surv_loaded.unique_times_
        )

    def test_pickle_to_file(self, survival_data):
        """Test pickling to file."""
        X, y = survival_data
        surv = ObliqueForestSurvival(n_trees=10, random_state=42)
        surv.fit(X, y)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            pickle.dump(surv, f)
            temp_path = f.name

        try:
            with open(temp_path, 'rb') as f:
                surv_loaded = pickle.load(f)

            risk_original = surv.predict(X)
            risk_loaded = surv_loaded.predict(X)
            np.testing.assert_array_almost_equal(risk_original, risk_loaded)
        finally:
            os.unlink(temp_path)


class TestPickleProtocols:
    """Test different pickle protocols."""

    @pytest.mark.parametrize("protocol", [0, 1, 2, 3, 4, 5])
    def test_all_protocols(self, classification_data, protocol):
        """Test all pickle protocols."""
        if protocol > pickle.HIGHEST_PROTOCOL:
            pytest.skip(f"Protocol {protocol} not available")

        X, y = classification_data
        clf = ObliqueForestClassifier(n_trees=5, random_state=42)
        clf.fit(X, y)

        pickled = pickle.dumps(clf, protocol=protocol)
        clf_loaded = pickle.loads(pickled)

        pred_original = clf.predict(X)
        pred_loaded = clf_loaded.predict(X)
        np.testing.assert_array_equal(pred_original, pred_loaded)


class TestJoblib:
    """Test joblib serialization (common in sklearn)."""

    def test_joblib_dump_load(self, classification_data):
        """Test joblib dump and load."""
        joblib = pytest.importorskip("joblib")

        X, y = classification_data
        clf = ObliqueForestClassifier(n_trees=10, random_state=42)
        clf.fit(X, y)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as f:
            joblib.dump(clf, f.name)
            temp_path = f.name

        try:
            clf_loaded = joblib.load(temp_path)
            pred_original = clf.predict(X)
            pred_loaded = clf_loaded.predict(X)
            np.testing.assert_array_equal(pred_original, pred_loaded)
        finally:
            os.unlink(temp_path)

    def test_joblib_regressor(self, regression_data):
        """Test joblib with regressor."""
        joblib = pytest.importorskip("joblib")

        X, y = regression_data
        reg = ObliqueForestRegressor(n_trees=10, random_state=42)
        reg.fit(X, y)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as f:
            joblib.dump(reg, f.name)
            temp_path = f.name

        try:
            reg_loaded = joblib.load(temp_path)
            pred_original = reg.predict(X)
            pred_loaded = reg_loaded.predict(X)
            np.testing.assert_array_almost_equal(pred_original, pred_loaded)
        finally:
            os.unlink(temp_path)

    def test_joblib_survival(self, survival_data):
        """Test joblib with survival forest."""
        joblib = pytest.importorskip("joblib")

        X, y = survival_data
        surv = ObliqueForestSurvival(n_trees=10, random_state=42)
        surv.fit(X, y)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as f:
            joblib.dump(surv, f.name)
            temp_path = f.name

        try:
            surv_loaded = joblib.load(temp_path)
            risk_original = surv.predict(X)
            risk_loaded = surv_loaded.predict(X)
            np.testing.assert_array_almost_equal(risk_original, risk_loaded)
        finally:
            os.unlink(temp_path)
