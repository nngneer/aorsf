"""
pytest configuration and fixtures for pyaorsf tests.
"""

import pytest
import numpy as np


@pytest.fixture
def random_state():
    """Fixed random state for reproducibility."""
    return np.random.RandomState(42)


@pytest.fixture
def classification_data(random_state):
    """Generate sample classification data."""
    n_samples = 200
    n_features = 10

    X = random_state.randn(n_samples, n_features)
    # Create two clusters
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    return X, y


@pytest.fixture
def regression_data(random_state):
    """Generate sample regression data."""
    n_samples = 200
    n_features = 10

    X = random_state.randn(n_samples, n_features)
    # Linear relationship with noise
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + random_state.randn(n_samples) * 0.1

    return X, y


@pytest.fixture
def survival_data(random_state):
    """Generate sample survival data."""
    n_samples = 200
    n_features = 10

    X = random_state.randn(n_samples, n_features)

    # Generate survival times with exponential distribution
    # Higher X[:, 0] leads to higher risk (shorter time)
    hazard = np.exp(X[:, 0] * 0.5)
    time = random_state.exponential(10 / hazard)

    # Random censoring
    censor_time = random_state.exponential(15, n_samples)
    status = (time <= censor_time).astype(float)
    time = np.minimum(time, censor_time)

    y = np.column_stack([time, status])

    return X, y


@pytest.fixture
def multiclass_data(random_state):
    """Generate sample multiclass classification data."""
    n_samples = 300
    n_features = 10

    X = random_state.randn(n_samples, n_features)
    # Create three clusters based on X[:, 0]
    y = np.zeros(n_samples, dtype=int)
    y[X[:, 0] > 0.5] = 1
    y[X[:, 0] > 1.0] = 2

    return X, y
