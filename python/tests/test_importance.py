"""
Tests for variable importance computation.
"""

import pytest
import numpy as np
from pyaorsf import ObliqueForestClassifier, ObliqueForestRegressor, ObliqueForestSurvival


class TestClassifierImportance:
    """Tests for classifier variable importance."""

    def test_negate_importance(self, classification_data):
        """Test negation importance."""
        X, y = classification_data
        clf = ObliqueForestClassifier(
            n_trees=50,
            importance='negate',
            random_state=42
        )
        clf.fit(X, y)

        assert clf.feature_importances_ is not None
        assert clf.feature_importances_.shape == (X.shape[1],)
        assert not np.all(clf.feature_importances_ == 0)

    def test_permute_importance(self, classification_data):
        """Test permutation importance."""
        X, y = classification_data
        clf = ObliqueForestClassifier(
            n_trees=50,
            importance='permute',
            random_state=42
        )
        clf.fit(X, y)

        assert clf.feature_importances_ is not None
        assert clf.feature_importances_.shape == (X.shape[1],)

    def test_anova_importance(self, classification_data):
        """Test ANOVA importance."""
        X, y = classification_data
        clf = ObliqueForestClassifier(
            n_trees=50,
            importance='anova',
            random_state=42
        )
        clf.fit(X, y)

        assert clf.feature_importances_ is not None
        assert clf.feature_importances_.shape == (X.shape[1],)

    def test_no_importance(self, classification_data):
        """Test that importance is None when not requested."""
        X, y = classification_data
        clf = ObliqueForestClassifier(n_trees=10, random_state=42)
        clf.fit(X, y)

        assert clf.feature_importances_ is None

    def test_invalid_importance_type(self, classification_data):
        """Test that invalid importance type raises error."""
        X, y = classification_data
        clf = ObliqueForestClassifier(n_trees=10, importance='invalid')

        with pytest.raises(ValueError, match="importance must be one of"):
            clf.fit(X, y)

    def test_importance_case_insensitive(self, classification_data):
        """Test that importance type is case insensitive."""
        X, y = classification_data
        clf = ObliqueForestClassifier(
            n_trees=10,
            importance='NEGATE',
            random_state=42
        )
        clf.fit(X, y)
        assert clf.feature_importances_ is not None

    def test_importance_in_get_params(self, classification_data):
        """Test that importance is included in get_params."""
        clf = ObliqueForestClassifier(n_trees=10, importance='negate')
        params = clf.get_params()
        assert 'importance' in params
        assert params['importance'] == 'negate'


class TestRegressorImportance:
    """Tests for regressor variable importance."""

    def test_negate_importance(self, regression_data):
        """Test negation importance for regressor."""
        X, y = regression_data
        reg = ObliqueForestRegressor(
            n_trees=50,
            importance='negate',
            random_state=42
        )
        reg.fit(X, y)

        assert reg.feature_importances_ is not None
        assert reg.feature_importances_.shape == (X.shape[1],)

    def test_permute_importance(self, regression_data):
        """Test permutation importance for regressor."""
        X, y = regression_data
        reg = ObliqueForestRegressor(
            n_trees=50,
            importance='permute',
            random_state=42
        )
        reg.fit(X, y)

        assert reg.feature_importances_ is not None
        assert reg.feature_importances_.shape == (X.shape[1],)

    def test_anova_importance(self, regression_data):
        """Test ANOVA importance for regressor."""
        X, y = regression_data
        reg = ObliqueForestRegressor(
            n_trees=50,
            importance='anova',
            random_state=42
        )
        reg.fit(X, y)

        assert reg.feature_importances_ is not None
        assert reg.feature_importances_.shape == (X.shape[1],)

    def test_no_importance(self, regression_data):
        """Test that importance is None when not requested."""
        X, y = regression_data
        reg = ObliqueForestRegressor(n_trees=10, random_state=42)
        reg.fit(X, y)

        assert reg.feature_importances_ is None


class TestSurvivalImportance:
    """Tests for survival forest variable importance."""

    def test_negate_importance(self, survival_data):
        """Test negation importance for survival."""
        X, y = survival_data
        surv = ObliqueForestSurvival(
            n_trees=50,
            importance='negate',
            random_state=42
        )
        surv.fit(X, y)

        assert surv.feature_importances_ is not None
        assert surv.feature_importances_.shape == (X.shape[1],)

    def test_permute_importance(self, survival_data):
        """Test permutation importance for survival."""
        X, y = survival_data
        surv = ObliqueForestSurvival(
            n_trees=50,
            importance='permute',
            random_state=42
        )
        surv.fit(X, y)

        assert surv.feature_importances_ is not None
        assert surv.feature_importances_.shape == (X.shape[1],)

    def test_anova_importance(self, survival_data):
        """Test ANOVA importance for survival."""
        X, y = survival_data
        surv = ObliqueForestSurvival(
            n_trees=50,
            importance='anova',
            random_state=42
        )
        surv.fit(X, y)

        assert surv.feature_importances_ is not None
        assert surv.feature_importances_.shape == (X.shape[1],)

    def test_no_importance(self, survival_data):
        """Test that importance is None when not requested."""
        X, y = survival_data
        surv = ObliqueForestSurvival(n_trees=10, random_state=42)
        surv.fit(X, y)

        assert surv.feature_importances_ is None


class TestImportanceInterpretation:
    """Tests for importance value interpretation."""

    def test_informative_features_ranked_higher(self):
        """Test that truly informative features have higher importance."""
        np.random.seed(42)
        n_samples = 500

        # Create data where only first 2 features matter
        X = np.random.randn(n_samples, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        clf = ObliqueForestClassifier(
            n_trees=100,
            importance='negate',
            random_state=42
        )
        clf.fit(X, y)

        # First two features should have highest importance
        top_2 = np.argsort(clf.feature_importances_)[-2:]
        assert 0 in top_2 or 1 in top_2  # At least one of the informative features

    def test_importance_values_reasonable(self, classification_data):
        """Test that importance values are reasonable."""
        X, y = classification_data
        clf = ObliqueForestClassifier(
            n_trees=50,
            importance='negate',
            random_state=42
        )
        clf.fit(X, y)

        # Importances should not all be zero
        assert not np.all(clf.feature_importances_ == 0)
        # Importances should not all be identical
        assert np.std(clf.feature_importances_) > 0
        # Importances should be finite
        assert np.all(np.isfinite(clf.feature_importances_))
