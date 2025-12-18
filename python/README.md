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
- **Custom callbacks**: User-defined linear combinations and evaluation functions
- **scikit-learn compatible**: Works with pipelines, cross-validation, GridSearchCV
- **Multi-threaded**: Parallel training and prediction

## Forest Types

### Classification

```python
from pyaorsf import ObliqueForestClassifier

clf = ObliqueForestClassifier(
    n_trees=500,
    mtry=3,              # Features per split
    leaf_min_obs=5,      # Min samples per leaf
    n_threads=4,
    random_state=42
)
clf.fit(X_train, y_train)

# Predict class labels
predictions = clf.predict(X_test)

# Predict probabilities
probabilities = clf.predict_proba(X_test)

# Evaluate accuracy
accuracy = clf.score(X_test, y_test)
```

### Regression

```python
from pyaorsf import ObliqueForestRegressor

reg = ObliqueForestRegressor(
    n_trees=500,
    mtry=3,
    leaf_min_obs=5,
    n_threads=4,
    random_state=42
)
reg.fit(X_train, y_train)

# Predict values
predictions = reg.predict(X_test)

# Evaluate R^2 score
r2_score = reg.score(X_test, y_test)
```

### Survival Analysis

```python
import numpy as np
from pyaorsf import ObliqueForestSurvival

# Create survival data: y = [time, status]
# status: 1 = event occurred, 0 = censored
time = np.random.exponential(10, 100)
status = np.random.binomial(1, 0.7, 100)
y = np.column_stack([time, status])

surv = ObliqueForestSurvival(
    n_trees=500,
    leaf_min_events=2,   # Min events per leaf
    n_threads=4,
    random_state=42
)
surv.fit(X_train, y)

# Predict risk scores (higher = higher risk)
risk = surv.predict(X_test)

# Predict survival probability
survival = surv.predict_survival(X_test)

# Evaluate using C-index
c_index = surv.score(X_test, y_test)
```

## Custom Callbacks

pyaorsf supports custom Python callbacks for advanced use cases.

### Custom Linear Combination Function

Define a custom function for computing linear combinations at each split:

```python
import numpy as np
from pyaorsf import ObliqueForestClassifier

def custom_lincomb(x, y, w):
    """
    Custom linear combination function.

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_features)
        Feature matrix for samples at the current node.
    y : ndarray of shape (n_samples, n_targets)
        Target values (class labels for classification, values for regression).
    w : ndarray of shape (n_samples,)
        Sample weights.

    Returns
    -------
    coefs : ndarray of shape (n_features,)
        Coefficients for the linear combination.
    """
    # Simple example: use only the first feature
    coefs = np.zeros(x.shape[1])
    coefs[0] = 1.0
    return coefs

clf = ObliqueForestClassifier(
    n_trees=100,
    lincomb_func=custom_lincomb,
    random_state=42
)
clf.fit(X_train, y_train)
```

### Custom Out-of-Bag Evaluation Function

Define a custom function for out-of-bag evaluation:

```python
def custom_oob_eval(y, w, p):
    """
    Custom OOB evaluation function.

    Parameters
    ----------
    y : ndarray of shape (n_samples, n_targets)
        True target values.
    w : ndarray of shape (n_samples,)
        Sample weights.
    p : ndarray of shape (n_samples,) or (n_samples, n_classes)
        Predictions (risk scores, probabilities, or values).

    Returns
    -------
    score : float
        Evaluation score (higher is better).
    """
    # Example: negative mean squared error
    return -np.mean((y[:, 0] - p) ** 2)

clf = ObliqueForestClassifier(
    n_trees=100,
    oobag_eval_func=custom_oob_eval,
    random_state=42
)
clf.fit(X_train, y_train)
```

## Variable Importance

Compute feature importance during training:

```python
clf = ObliqueForestClassifier(
    n_trees=100,
    importance='negate',  # 'negate', 'permute', or 'anova'
    random_state=42
)
clf.fit(X_train, y_train)

# Access importance scores
print(clf.feature_importances_)
```

### Importance Methods

| Method | Description | Speed |
|--------|-------------|-------|
| `negate` | Negation importance (recommended) | Fast |
| `permute` | Permutation importance | Medium |
| `anova` | ANOVA-based importance | Fast |

## Model Serialization

Save and load trained models with pickle:

```python
import pickle

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Load model
with open('model.pkl', 'rb') as f:
    clf = pickle.load(f)

# Or with joblib (recommended for large models)
import joblib
joblib.dump(clf, 'model.joblib')
clf = joblib.load('model.joblib')
```

**Note**: Models with custom callback functions (`lincomb_func`, `oobag_eval_func`) cannot be pickled directly. Set these to `None` before pickling, or use module-level functions.

## scikit-learn Compatibility

pyaorsf estimators are fully compatible with scikit-learn.

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

clf = ObliqueForestClassifier(n_trees=100, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', ObliqueForestClassifier(n_trees=100, random_state=42))
])
pipe.fit(X_train, y_train)
print(f"Accuracy: {pipe.score(X_test, y_test):.3f}")
```

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

clf = ObliqueForestClassifier(n_trees=50, random_state=42)
param_grid = {
    'mtry': [2, 3, 5],
    'leaf_min_obs': [3, 5, 10]
}

grid = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy')
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.3f}")
```

## Parameters

### Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_trees` | 500 | Number of trees in the forest |
| `mtry` | None | Features to consider per split (default: sqrt(n_features)) |
| `leaf_min_obs` | 5 | Minimum samples per leaf |
| `sample_fraction` | 0.632 | Fraction of samples per tree |
| `sample_with_replacement` | True | Bootstrap sampling |
| `n_threads` | 1 | Number of threads for parallelization |
| `random_state` | None | Random seed for reproducibility |
| `importance` | None | Variable importance method: 'negate', 'permute', 'anova' |
| `lincomb_func` | None | Custom linear combination function |
| `oobag_eval_func` | None | Custom OOB evaluation function |

### Survival-Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `leaf_min_events` | 2 | Minimum events per leaf |
| `split_min_events` | 5 | Minimum events to attempt split |
| `split_rule` | 'logrank' | Split rule ('logrank' or 'cstat') |
| `pred_horizon` | None | Time points for survival prediction |

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### Building Documentation

```bash
pip install -e ".[docs]"
cd docs && make html
```

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

## Documentation

Full documentation for the R package is available at [docs.ropensci.org/aorsf](https://docs.ropensci.org/aorsf/).

## License

MIT License - see LICENSE file for details.
