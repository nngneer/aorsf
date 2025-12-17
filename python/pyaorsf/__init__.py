"""
pyaorsf: Accelerated Oblique Random Forests for Python
======================================================

A Python interface to the aorsf C++ library for fitting oblique random forests.
Provides scikit-learn compatible estimators for classification, regression,
and survival analysis.

Main Classes
------------
ObliqueForestClassifier
    Oblique random forest for classification tasks.
ObliqueForestRegressor
    Oblique random forest for regression tasks.
ObliqueForestSurvival
    Oblique random survival forest for time-to-event data.

Examples
--------
>>> from pyaorsf import ObliqueForestClassifier
>>> clf = ObliqueForestClassifier(n_trees=100)
>>> clf.fit(X_train, y_train)
>>> predictions = clf.predict(X_test)
"""

from ._version import __version__
from .forest import ObliqueForestClassifier, ObliqueForestRegressor
from .survival import ObliqueForestSurvival

__all__ = [
    "__version__",
    "ObliqueForestClassifier",
    "ObliqueForestRegressor",
    "ObliqueForestSurvival",
]
