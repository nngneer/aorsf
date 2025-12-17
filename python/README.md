# pyaorsf

Python interface to Accelerated Oblique Random Forests.

## Installation

```bash
pip install pyaorsf
```

### From Source

```bash
# Clone the repository
git clone https://github.com/ropensci/aorsf.git
cd aorsf/python

# Install in development mode
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.9
- NumPy >= 1.20
- Armadillo (system library)
- OpenMP (optional, for parallelization)

## Quick Start

```python
from pyaorsf import ObliqueForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create sample data
X, y = make_classification(n_samples=1000, n_features=20)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train model
clf = ObliqueForestClassifier(n_trees=100, n_threads=4)
clf.fit(X_train, y_train)

# Predict
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

# Evaluate
print(f"Accuracy: {clf.score(X_test, y_test):.3f}")
```

## Features

- **Oblique splits**: Linear combination splits for better decision boundaries
- **Multiple forest types**: Classification, regression, and survival analysis
- **Survival analysis**: Accelerated Cox regression for time-to-event data
- **Variable importance**: Negation, permutation, and ANOVA methods
- **Partial dependence**: Understand marginal effects of predictors
- **scikit-learn compatible**: Works with pipelines, cross-validation, etc.
- **Multi-threaded**: Parallel training and prediction

## Forest Types

### Classification

```python
from pyaorsf import ObliqueForestClassifier

clf = ObliqueForestClassifier(
    n_trees=500,
    mtry=3,
    leaf_min_obs=5,
    n_threads=4
)
clf.fit(X_train, y_train)
```

### Regression

```python
from pyaorsf import ObliqueForestRegressor

reg = ObliqueForestRegressor(
    n_trees=500,
    mtry=3,
    leaf_min_obs=5,
    n_threads=4
)
reg.fit(X_train, y_train)
```

### Survival Analysis

```python
from pyaorsf import ObliqueForestSurvival

# y should be a structured array with 'time' and 'status' fields
# or a 2D array with columns [time, status]
surv = ObliqueForestSurvival(
    n_trees=500,
    leaf_min_events=1,
    n_threads=4
)
surv.fit(X_train, y_train)

# Predict risk scores
risk = surv.predict(X_test)

# Predict survival probability at specific times
surv_prob = surv.predict_survival(X_test, times=[30, 60, 90])
```

## Variable Importance

```python
# Negation importance (fast, emphasizes coefficients)
importance = clf.feature_importances_

# Permutation importance
perm_importance = clf.permutation_importance(X_test, y_test)

# ANOVA importance
anova_importance = clf.anova_importance()
```

## Documentation

Full documentation is available at [docs.ropensci.org/aorsf](https://docs.ropensci.org/aorsf/).

## Citation

If you use pyaorsf in your research, please cite:

```bibtex
@article{jaeger2023accelerated,
  title={Accelerated and interpretable oblique random survival forests},
  author={Jaeger, Byron C and Welden, Sawyer and Lenoir, Kristin and others},
  journal={Journal of Computational and Graphical Statistics},
  year={2023},
  publisher={Taylor \& Francis}
}
```

## License

MIT License - see LICENSE file for details.
